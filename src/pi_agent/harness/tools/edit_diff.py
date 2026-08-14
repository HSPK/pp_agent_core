"""Shared diff computation utilities for the harness edit tool.

Python port of `packages/agent/src/harness/tools/edit-diff.ts`. That file is
byte-identical to `packages/coding-agent/src/core/tools/edit-diff.ts` up to
the latter's extra `computeEditsDiff`/`computeEditDiff` preview helpers, which
belong to the coding agent and are not part of the harness surface. The
TypeScript version uses the `diff` npm package for line-level diffing; this
port uses the stdlib `difflib.SequenceMatcher` over line arrays (equivalent
to `Diff.diffLines`) while reproducing the exact display format (line-numbered
`+`/`-`/` ` markers with collapsed context) and the unified-patch output.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher, unified_diff

_SMART_SINGLE_QUOTES = re.compile("[\u2018\u2019\u201a\u201b]")
_SMART_DOUBLE_QUOTES = re.compile("[\u201c\u201d\u201e\u201f]")
_UNICODE_DASHES = re.compile("[\u2010\u2011\u2012\u2013\u2014\u2015\u2212]")
_SPECIAL_SPACES = re.compile("[\u00a0\u2002-\u200a\u202f\u205f\u3000]")
_LINE_SPLIT_RE = re.compile(r"[^\n]*\n|[^\n]+")


def detect_line_ending(content: str) -> str:
    crlf_idx = content.find("\r\n")
    lf_idx = content.find("\n")
    if lf_idx == -1:
        return "\n"
    if crlf_idx == -1:
        return "\n"
    return "\r\n" if crlf_idx < lf_idx else "\n"


def normalize_to_lf(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def restore_line_endings(text: str, ending: str) -> str:
    return text.replace("\n", "\r\n") if ending == "\r\n" else text


def normalize_for_fuzzy_match(text: str) -> str:
    """Normalize text for fuzzy matching.

    Applies progressive transformations:
    - Strip trailing whitespace from each line
    - Normalize smart quotes to ASCII equivalents
    - Normalize Unicode dashes/hyphens to ASCII hyphen
    - Normalize special Unicode spaces to regular space
    """
    normalized = unicodedata.normalize("NFKC", text)
    normalized = "\n".join(line.rstrip() for line in normalized.split("\n"))
    normalized = _SMART_SINGLE_QUOTES.sub("'", normalized)
    normalized = _SMART_DOUBLE_QUOTES.sub('"', normalized)
    normalized = _UNICODE_DASHES.sub("-", normalized)
    normalized = _SPECIAL_SPACES.sub(" ", normalized)
    return normalized


def _split_lines_with_endings(content: str) -> list[str]:
    return _LINE_SPLIT_RE.findall(content)


@dataclass
class _LineSpan:
    start: int
    end: int


def _get_line_spans(content: str) -> list[_LineSpan]:
    offset = 0
    spans = []
    for line in _split_lines_with_endings(content):
        span = _LineSpan(start=offset, end=offset + len(line))
        offset = span.end
        spans.append(span)
    return spans


@dataclass
class _TextReplacement:
    match_index: int
    match_length: int
    new_text: str


def _get_replacement_line_range(lines: list[_LineSpan], replacement: _TextReplacement) -> tuple[int, int]:
    replacement_start = replacement.match_index
    replacement_end = replacement.match_index + replacement.match_length

    start_line = -1
    for i, line in enumerate(lines):
        if replacement_start >= line.start and replacement_start < line.end:
            start_line = i
            break
    if start_line == -1:
        raise ValueError("Replacement range is outside the base content.")

    end_line = start_line
    while end_line < len(lines) and lines[end_line].end < replacement_end:
        end_line += 1
    if end_line >= len(lines):
        raise ValueError("Replacement range is outside the base content.")

    return start_line, end_line + 1


def _apply_replacements(content: str, replacements: list[_TextReplacement], offset: int = 0) -> str:
    result = content
    for replacement in reversed(replacements):
        match_index = replacement.match_index - offset
        result = result[:match_index] + replacement.new_text + result[match_index + replacement.match_length :]
    return result


def apply_replacements_preserving_unchanged_lines(
    original_content: str,
    base_content: str,
    replacements: list[_TextReplacement],
) -> str:
    """Apply replacements matched against `base_content` to `original_content`, preserving unchanged lines.

    This is useful when `base_content` is a normalized view of the original. Each
    replacement is widened to the lines it actually touches, those touched lines
    are rewritten from the normalized base, and all other lines are copied back
    from `original_content`. The actual replacement ranges drive preservation so
    duplicate normalized lines cannot be aligned to the wrong occurrence.
    """
    original_lines = _split_lines_with_endings(original_content)
    base_lines = _get_line_spans(base_content)
    if len(original_lines) != len(base_lines):
        raise ValueError("Cannot preserve unchanged lines because the base content has a different line count.")

    groups: list[dict] = []
    sorted_replacements = sorted(replacements, key=lambda r: r.match_index)
    for replacement in sorted_replacements:
        start_line, end_line = _get_replacement_line_range(base_lines, replacement)
        if groups and start_line < groups[-1]["end_line"]:
            groups[-1]["end_line"] = max(groups[-1]["end_line"], end_line)
            groups[-1]["replacements"].append(replacement)
            continue
        groups.append({"start_line": start_line, "end_line": end_line, "replacements": [replacement]})

    original_line_index = 0
    result = ""
    for group in groups:
        result += "".join(original_lines[original_line_index : group["start_line"]])

        group_start_offset = base_lines[group["start_line"]].start
        group_end_offset = base_lines[group["end_line"] - 1].end
        result += _apply_replacements(
            base_content[group_start_offset:group_end_offset],
            group["replacements"],
            group_start_offset,
        )
        original_line_index = group["end_line"]
    result += "".join(original_lines[original_line_index:])

    return result


@dataclass
class FuzzyMatchResult:
    found: bool
    index: int
    match_length: int
    used_fuzzy_match: bool
    content_for_replacement: str


def fuzzy_find_text(content: str, old_text: str) -> FuzzyMatchResult:
    """Find `old_text` in `content`, trying exact match first, then fuzzy match.

    When fuzzy matching is used, the returned `content_for_replacement` is the
    fuzzy-normalized version of the content (trailing whitespace stripped,
    Unicode quotes/dashes normalized to ASCII).
    """
    exact_index = content.find(old_text)
    if exact_index != -1:
        return FuzzyMatchResult(
            found=True,
            index=exact_index,
            match_length=len(old_text),
            used_fuzzy_match=False,
            content_for_replacement=content,
        )

    fuzzy_content = normalize_for_fuzzy_match(content)
    fuzzy_old_text = normalize_for_fuzzy_match(old_text)
    fuzzy_index = fuzzy_content.find(fuzzy_old_text)

    if fuzzy_index == -1:
        return FuzzyMatchResult(
            found=False,
            index=-1,
            match_length=0,
            used_fuzzy_match=False,
            content_for_replacement=content,
        )

    return FuzzyMatchResult(
        found=True,
        index=fuzzy_index,
        match_length=len(fuzzy_old_text),
        used_fuzzy_match=True,
        content_for_replacement=fuzzy_content,
    )


def strip_bom(content: str) -> tuple[str, str]:
    """Strip UTF-8 BOM if present. Returns `(bom, text_without_bom)`."""
    if content.startswith("\ufeff"):
        return "\ufeff", content[1:]
    return "", content


def _count_occurrences(content: str, old_text: str) -> int:
    fuzzy_content = normalize_for_fuzzy_match(content)
    fuzzy_old_text = normalize_for_fuzzy_match(old_text)
    return fuzzy_content.count(fuzzy_old_text)


def _not_found_error(path: str, edit_index: int, total_edits: int) -> ValueError:
    if total_edits == 1:
        return ValueError(
            f"Could not find the exact text in {path}. "
            "The old text must match exactly including all whitespace and newlines."
        )
    return ValueError(
        f"Could not find edits[{edit_index}] in {path}. "
        "The oldText must match exactly including all whitespace and newlines."
    )


def _duplicate_error(path: str, edit_index: int, total_edits: int, occurrences: int) -> ValueError:
    if total_edits == 1:
        return ValueError(
            f"Found {occurrences} occurrences of the text in {path}. "
            "The text must be unique. Please provide more context to make it unique."
        )
    return ValueError(
        f"Found {occurrences} occurrences of edits[{edit_index}] in {path}. "
        "Each oldText must be unique. Please provide more context to make it unique."
    )


def _empty_old_text_error(path: str, edit_index: int, total_edits: int) -> ValueError:
    if total_edits == 1:
        return ValueError(f"oldText must not be empty in {path}.")
    return ValueError(f"edits[{edit_index}].oldText must not be empty in {path}.")


def _no_change_error(path: str, total_edits: int) -> ValueError:
    if total_edits == 1:
        return ValueError(
            f"No changes made to {path}. The replacement produced identical content. "
            "This might indicate an issue with special characters or the text not existing as expected."
        )
    return ValueError(f"No changes made to {path}. The replacements produced identical content.")


@dataclass
class Edit:
    old_text: str
    new_text: str


@dataclass
class AppliedEditsResult:
    base_content: str
    new_content: str


def apply_edits_to_normalized_content(normalized_content: str, edits: list[Edit], path: str) -> AppliedEditsResult:
    """Apply one or more exact-text replacements to LF-normalized content.

    All edits are matched against the same original content. Replacements are
    then applied in reverse order so offsets remain stable. If any edit needs
    fuzzy matching, the operation runs in fuzzy-normalized content space and then
    overlays those line-level changes onto the original content so unchanged line
    blocks keep their original bytes.
    """
    normalized_edits = [
        Edit(old_text=normalize_to_lf(edit.old_text), new_text=normalize_to_lf(edit.new_text)) for edit in edits
    ]

    for i, edit in enumerate(normalized_edits):
        if len(edit.old_text) == 0:
            raise _empty_old_text_error(path, i, len(normalized_edits))

    initial_matches = [fuzzy_find_text(normalized_content, edit.old_text) for edit in normalized_edits]
    used_fuzzy_match = any(match.used_fuzzy_match for match in initial_matches)
    replacement_base_content = normalize_for_fuzzy_match(normalized_content) if used_fuzzy_match else normalized_content

    matched_edits: list[_TextReplacement] = []
    matched_edit_indices: list[int] = []
    for i, edit in enumerate(normalized_edits):
        match_result = fuzzy_find_text(replacement_base_content, edit.old_text)
        if not match_result.found:
            raise _not_found_error(path, i, len(normalized_edits))

        occurrences = _count_occurrences(replacement_base_content, edit.old_text)
        if occurrences > 1:
            raise _duplicate_error(path, i, len(normalized_edits), occurrences)

        matched_edits.append(
            _TextReplacement(
                match_index=match_result.index, match_length=match_result.match_length, new_text=edit.new_text
            )
        )
        matched_edit_indices.append(i)

    order = sorted(range(len(matched_edits)), key=lambda idx: matched_edits[idx].match_index)
    sorted_edits = [matched_edits[idx] for idx in order]
    sorted_indices = [matched_edit_indices[idx] for idx in order]
    for i in range(1, len(sorted_edits)):
        previous = sorted_edits[i - 1]
        current = sorted_edits[i]
        if previous.match_index + previous.match_length > current.match_index:
            raise ValueError(
                f"edits[{sorted_indices[i - 1]}] and edits[{sorted_indices[i]}] overlap in {path}. "
                "Merge them into one edit or target disjoint regions."
            )

    base_content = normalized_content
    if used_fuzzy_match:
        new_content = apply_replacements_preserving_unchanged_lines(
            normalized_content, replacement_base_content, sorted_edits
        )
    else:
        new_content = _apply_replacements(replacement_base_content, sorted_edits)

    if base_content == new_content:
        raise _no_change_error(path, len(normalized_edits))

    return AppliedEditsResult(base_content=base_content, new_content=new_content)


def generate_unified_patch(path: str, old_content: str, new_content: str, context_lines: int = 4) -> str:
    """Generate a standard unified patch."""
    diff_lines = unified_diff(
        old_content.splitlines(keepends=True),
        new_content.splitlines(keepends=True),
        fromfile=path,
        tofile=path,
        n=context_lines,
        lineterm="\n",
    )
    patch = "".join(diff_lines)
    if patch and not patch.endswith("\n"):
        patch += "\n"
    return patch


def generate_diff_string(old_content: str, new_content: str, context_lines: int = 4) -> tuple[str, int | None]:
    """Generate a display-oriented diff string with line numbers and context.

    Returns both the diff string and the first changed line number (in the new file).
    """
    old_lines = old_content.split("\n")
    new_lines = new_content.split("\n")
    matcher = SequenceMatcher(None, old_lines, new_lines, autojunk=False)

    # Each part mirrors one chunk of `Diff.diffLines`: a run of lines that are
    # either unchanged ("equal"), removed, or added.
    parts: list[tuple[list[str], bool, bool]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            parts.append((old_lines[i1:i2], False, False))
        elif tag == "delete":
            parts.append((old_lines[i1:i2], False, True))
        elif tag == "insert":
            parts.append((new_lines[j1:j2], True, False))
        elif tag == "replace":
            parts.append((old_lines[i1:i2], False, True))
            parts.append((new_lines[j1:j2], True, False))

    max_line_num = max(len(old_lines), len(new_lines))
    line_num_width = len(str(max_line_num))

    output: list[str] = []
    old_line_num = 1
    new_line_num = 1
    last_was_change = False
    first_changed_line: int | None = None

    for idx, (raw, added, removed) in enumerate(parts):
        if added or removed:
            if first_changed_line is None:
                first_changed_line = new_line_num

            for line in raw:
                if added:
                    output.append(f"+{str(new_line_num).rjust(line_num_width)} {line}")
                    new_line_num += 1
                else:
                    output.append(f"-{str(old_line_num).rjust(line_num_width)} {line}")
                    old_line_num += 1
            last_was_change = True
        else:
            next_part_is_change = idx < len(parts) - 1 and (parts[idx + 1][1] or parts[idx + 1][2])
            has_leading_change = last_was_change
            has_trailing_change = next_part_is_change

            if has_leading_change and has_trailing_change:
                if len(raw) <= context_lines * 2:
                    for line in raw:
                        output.append(f" {str(old_line_num).rjust(line_num_width)} {line}")
                        old_line_num += 1
                        new_line_num += 1
                else:
                    leading_lines = raw[:context_lines]
                    trailing_lines = raw[len(raw) - context_lines :]
                    skipped_lines = len(raw) - len(leading_lines) - len(trailing_lines)

                    for line in leading_lines:
                        output.append(f" {str(old_line_num).rjust(line_num_width)} {line}")
                        old_line_num += 1
                        new_line_num += 1

                    output.append(f" {''.rjust(line_num_width)} ...")
                    old_line_num += skipped_lines
                    new_line_num += skipped_lines

                    for line in trailing_lines:
                        output.append(f" {str(old_line_num).rjust(line_num_width)} {line}")
                        old_line_num += 1
                        new_line_num += 1
            elif has_leading_change:
                shown_lines = raw[:context_lines]
                skipped_lines = len(raw) - len(shown_lines)

                for line in shown_lines:
                    output.append(f" {str(old_line_num).rjust(line_num_width)} {line}")
                    old_line_num += 1
                    new_line_num += 1

                if skipped_lines > 0:
                    output.append(f" {''.rjust(line_num_width)} ...")
                    old_line_num += skipped_lines
                    new_line_num += skipped_lines
            elif has_trailing_change:
                skipped_lines = max(0, len(raw) - context_lines)
                if skipped_lines > 0:
                    output.append(f" {''.rjust(line_num_width)} ...")
                    old_line_num += skipped_lines
                    new_line_num += skipped_lines

                for line in raw[skipped_lines:]:
                    output.append(f" {str(old_line_num).rjust(line_num_width)} {line}")
                    old_line_num += 1
                    new_line_num += 1
            else:
                old_line_num += len(raw)
                new_line_num += len(raw)

            last_was_change = False

    return "\n".join(output), first_changed_line


__all__ = [
    "AppliedEditsResult",
    "Edit",
    "FuzzyMatchResult",
    "apply_edits_to_normalized_content",
    "apply_replacements_preserving_unchanged_lines",
    "detect_line_ending",
    "fuzzy_find_text",
    "generate_diff_string",
    "generate_unified_patch",
    "normalize_for_fuzzy_match",
    "normalize_to_lf",
    "restore_line_endings",
    "strip_bom",
]
