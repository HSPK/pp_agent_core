"""Skill discovery from `SKILL.md` files.

Python port of `packages/agent/src/harness/skills.ts`.

Like `pi_agent.harness.prompt_templates`, this port reads directly from
`pathlib.Path` instead of threading an injectable `ExecutionEnv` through
every filesystem call: the directory to scan is a plain path parameter,
which keeps this module testable against `tmp_path` without needing a
ported `ExecutionEnv` implementation (the concrete Node backend,
`harness/env/nodejs.ts`, is out of scope for this port). `Path.is_dir()`/
`Path.is_file()` already resolve symlinks to their target kind, so the
`resolveKind`/`canonicalPath` fallback dance in the TypeScript source has no
counterpart here.

TypeScript uses the `ignore` npm package for `.gitignore`/`.ignore`/
`.fdignore` pattern matching. This port implements `_IgnoreMatcher`, a
small gitignore-pattern-to-regex translator covering the subset of the spec
relevant here (`*`, `**`, `?`, leading `/` anchors, trailing `/` restricts to
directories, leading `!` negates, backslash-escaped leading `!`/`#`), rather
than adding a new third-party dependency for it.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, Literal, TypeVar

import yaml

from .types import Skill

MAX_NAME_LENGTH = 64
MAX_DESCRIPTION_LENGTH = 1024
IGNORE_FILE_NAMES = (".gitignore", ".ignore", ".fdignore")

SkillDiagnosticCode = Literal["file_info_failed", "list_failed", "read_failed", "parse_failed", "invalid_metadata"]


@dataclass(kw_only=True)
class SkillDiagnostic:
    """Warning produced while loading skills."""

    code: SkillDiagnosticCode
    """Stable diagnostic code."""
    message: str
    """Human-readable diagnostic message."""
    path: str
    """Path associated with the diagnostic."""
    type: Literal["warning"] = "warning"
    """Diagnostic severity. Currently only warnings are emitted."""


TSource = TypeVar("TSource")


@dataclass(kw_only=True)
class SourcedSkill(Generic[TSource]):
    skill: Skill
    source: TSource


@dataclass(kw_only=True)
class SourcedSkillDiagnostic(Generic[TSource]):
    diagnostic: SkillDiagnostic
    source: TSource


def _pattern_to_regex(pattern: str) -> tuple[re.Pattern[str], bool]:
    """Translate one gitignore-style pattern into `(regex, directory_only)`.

    The traversal probes directories as ``"name/"`` and files as ``"name"``, so
    the regexes must honour three gitignore rules that are easy to get wrong:

    - A pattern without a trailing slash matches BOTH a file and a directory,
      so ``build`` must match the probe ``build/``.
    - A pattern WITH a trailing slash matches only a directory, so ``logs/``
      must not match the file probe ``logs``.
    - ``a/**/b`` matches ``a/b``: ``**`` collapses to zero path segments.
    """
    directory_only = pattern.endswith("/")
    body = pattern[:-1] if directory_only else pattern
    anchored = body.startswith("/")
    if anchored:
        body = body[1:]

    segments = body.split("/")
    regex_parts: list[str] = []
    for index, segment in enumerate(segments):
        if segment == "**":
            # Consume the separator that follows so "a/**/b" can match "a/b".
            regex_parts.append("(?:.*/)?" if index < len(segments) - 1 else ".*")
            continue
        piece = ""
        i = 0
        while i < len(segment):
            char = segment[i]
            if char == "*":
                if segment[i : i + 2] == "**":
                    piece += ".*"
                    i += 2
                    continue
                piece += "[^/]*"
            elif char == "?":
                piece += "[^/]"
            else:
                piece += re.escape(char)
            i += 1
        regex_parts.append(piece)

    joined = ""
    for index, part in enumerate(regex_parts):
        if index == 0:
            joined = part
        elif part.endswith("/)?") or regex_parts[index - 1].endswith("/)?"):
            joined += part
        else:
            joined += "/" + part

    prefix = "^" if anchored else "(?:^|.*/)"
    if directory_only:
        # Only ever matches a directory probe, which always ends in "/".
        pattern_re = f"{prefix}{joined}/.*$|{prefix}{joined}/$"
    else:
        # Matches the file itself, the directory probe, and anything beneath it.
        pattern_re = f"{prefix}{joined}(?:/.*)?$|{prefix}{joined}/$"
    return re.compile(pattern_re), directory_only


@dataclass
class _CompiledPattern:
    regex: re.Pattern[str]
    negated: bool


class _IgnoreMatcher:
    """Minimal gitignore-pattern matcher. See module docstring."""

    def __init__(self) -> None:
        self._patterns: list[_CompiledPattern] = []

    def add(self, patterns: Sequence[str]) -> None:
        for raw in patterns:
            negated = raw.startswith("!")
            body = raw[1:] if negated else raw
            regex, _ = _pattern_to_regex(body)
            self._patterns.append(_CompiledPattern(regex=regex, negated=negated))

    def ignores(self, path: str) -> bool:
        ignored = False
        for compiled in self._patterns:
            if compiled.regex.match(path):
                ignored = not compiled.negated
        return ignored


def format_skill_invocation(skill: Skill, additional_instructions: str | None = None) -> str:
    """Format a skill invocation prompt, optionally appending additional user instructions."""
    skill_block = (
        f'<skill name="{skill.name}" location="{skill.file_path}">\n'
        f"References are relative to {_dirname_env_path(skill.file_path)}.\n\n"
        f"{skill.content}\n</skill>"
    )
    return f"{skill_block}\n\n{additional_instructions}" if additional_instructions else skill_block


async def load_skills(dirs: str | Path | Sequence[str | Path]) -> tuple[list[Skill], list[SkillDiagnostic]]:
    """Load skills from one or more directories.

    Traverses directories recursively, loads `SKILL.md` files, loads direct
    root `.md` files as skills, honors ignore files, and returns diagnostics
    for invalid skill files. Missing input directories are skipped.
    """
    skills: list[Skill] = []
    diagnostics: list[SkillDiagnostic] = []
    dir_list = [dirs] if isinstance(dirs, (str, Path)) else list(dirs)
    for raw_dir in dir_list:
        root = Path(raw_dir)
        try:
            exists = root.exists()
        except OSError as error:
            diagnostics.append(SkillDiagnostic(code="file_info_failed", message=str(error), path=str(root)))
            continue
        if not exists or not root.is_dir():
            continue
        dir_skills, dir_diagnostics = _load_skills_from_dir(root, True, _IgnoreMatcher(), root)
        skills.extend(dir_skills)
        diagnostics.extend(dir_diagnostics)
    return skills, diagnostics


async def load_sourced_skills(
    inputs: Sequence[tuple[str | Path, TSource]],
    map_skill: Callable[[Skill, TSource], Skill] | None = None,
) -> tuple[list[SourcedSkill[TSource]], list[SourcedSkillDiagnostic[TSource]]]:
    """Load skills from source-tagged directories.

    Source values are preserved exactly and attached to every loaded skill
    and diagnostic. The agent package does not interpret source values;
    applications define their own provenance shape.
    """
    skills: list[SourcedSkill[TSource]] = []
    diagnostics: list[SourcedSkillDiagnostic[TSource]] = []
    for path, source in inputs:
        result_skills, result_diagnostics = await load_skills(path)
        for skill in result_skills:
            mapped = map_skill(skill, source) if map_skill else skill
            skills.append(SourcedSkill(skill=mapped, source=source))
        for diagnostic in result_diagnostics:
            diagnostics.append(SourcedSkillDiagnostic(diagnostic=diagnostic, source=source))
    return skills, diagnostics


def _load_skills_from_dir(
    directory: Path, include_root_files: bool, ignore_matcher: _IgnoreMatcher, root_dir: Path
) -> tuple[list[Skill], list[SkillDiagnostic]]:
    skills: list[Skill] = []
    diagnostics: list[SkillDiagnostic] = []

    if not directory.is_dir():
        return skills, diagnostics

    _add_ignore_rules(ignore_matcher, directory, root_dir, diagnostics)

    try:
        entries = list(directory.iterdir())
    except OSError as error:
        diagnostics.append(SkillDiagnostic(code="list_failed", message=str(error), path=str(directory)))
        return skills, diagnostics

    for entry in entries:
        if entry.name != "SKILL.md":
            continue
        if not entry.is_file():
            continue
        rel_path = _relative_env_path(root_dir, entry)
        if ignore_matcher.ignores(rel_path):
            continue
        skill, file_diagnostics = _load_skill_from_file(entry, directory.name)
        if skill is not None:
            skills.append(skill)
        diagnostics.extend(file_diagnostics)
        return skills, diagnostics

    for entry in sorted(entries, key=lambda e: e.name):
        if entry.name.startswith(".") or entry.name == "node_modules":
            continue
        is_dir = entry.is_dir()
        is_file = entry.is_file()
        if not is_dir and not is_file:
            continue

        rel_path = _relative_env_path(root_dir, entry)
        ignore_path = f"{rel_path}/" if is_dir else rel_path
        if ignore_matcher.ignores(ignore_path):
            continue

        if is_dir:
            nested_skills, nested_diagnostics = _load_skills_from_dir(entry, False, ignore_matcher, root_dir)
            skills.extend(nested_skills)
            diagnostics.extend(nested_diagnostics)
            continue

        if not include_root_files or not entry.name.endswith(".md"):
            continue
        skill, file_diagnostics = _load_skill_from_file(entry, directory.name)
        if skill is not None:
            skills.append(skill)
        diagnostics.extend(file_diagnostics)

    return skills, diagnostics


def _add_ignore_rules(ig: _IgnoreMatcher, directory: Path, root_dir: Path, diagnostics: list[SkillDiagnostic]) -> None:
    relative_dir = _relative_env_path(root_dir, directory)
    prefix = f"{relative_dir}/" if relative_dir else ""

    for filename in IGNORE_FILE_NAMES:
        ignore_path = directory / filename
        if not ignore_path.is_file():
            continue
        try:
            content = ignore_path.read_text(encoding="utf-8")
        except OSError as error:
            diagnostics.append(SkillDiagnostic(code="read_failed", message=str(error), path=str(ignore_path)))
            continue
        patterns = [
            pattern
            for pattern in (_prefix_ignore_pattern(line, prefix) for line in re.split(r"\r?\n", content))
            if pattern
        ]
        if patterns:
            ig.add(patterns)


def _prefix_ignore_pattern(line: str, prefix: str) -> str | None:
    trimmed = line.strip()
    if not trimmed:
        return None
    if trimmed.startswith("#") and not trimmed.startswith("\\#"):
        return None

    pattern = line
    negated = False
    if pattern.startswith("!"):
        negated = True
        pattern = pattern[1:]
    elif pattern.startswith("\\!"):
        pattern = pattern[1:]
    if pattern.startswith("/"):
        pattern = pattern[1:]
    prefixed = f"{prefix}{pattern}" if prefix else pattern
    return f"!{prefixed}" if negated else prefixed


def _load_skill_from_file(path: Path, parent_dir_name: str) -> tuple[Skill | None, list[SkillDiagnostic]]:
    diagnostics: list[SkillDiagnostic] = []
    try:
        raw_content = path.read_text(encoding="utf-8")
    except OSError as error:
        diagnostics.append(SkillDiagnostic(code="read_failed", message=str(error), path=str(path)))
        return None, diagnostics

    try:
        frontmatter, body = _parse_frontmatter(raw_content)
    except Exception as error:
        diagnostics.append(SkillDiagnostic(code="parse_failed", message=str(error), path=str(path)))
        return None, diagnostics

    description = frontmatter.get("description") if isinstance(frontmatter.get("description"), str) else None

    for error_message in _validate_description(description):
        diagnostics.append(SkillDiagnostic(code="invalid_metadata", message=error_message, path=str(path)))

    frontmatter_name = frontmatter.get("name") if isinstance(frontmatter.get("name"), str) else None
    name = frontmatter_name or parent_dir_name
    for error_message in _validate_name(name, parent_dir_name):
        diagnostics.append(SkillDiagnostic(code="invalid_metadata", message=error_message, path=str(path)))

    if not description or description.strip() == "":
        return None, diagnostics

    return (
        Skill(
            name=name,
            description=description,
            content=body,
            file_path=str(path),
            disable_model_invocation=frontmatter.get("disable-model-invocation") is True,
        ),
        diagnostics,
    )


def _validate_name(name: str, parent_dir_name: str) -> list[str]:
    errors: list[str] = []
    if name != parent_dir_name:
        errors.append(f'name "{name}" does not match parent directory "{parent_dir_name}"')
    if len(name) > MAX_NAME_LENGTH:
        errors.append(f"name exceeds {MAX_NAME_LENGTH} characters ({len(name)})")
    if not re.match(r"^[a-z0-9-]+$", name):
        errors.append("name contains invalid characters (must be lowercase a-z, 0-9, hyphens only)")
    if name.startswith("-") or name.endswith("-"):
        errors.append("name must not start or end with a hyphen")
    if "--" in name:
        errors.append("name must not contain consecutive hyphens")
    return errors


def _validate_description(description: str | None) -> list[str]:
    errors: list[str] = []
    if not description or description.strip() == "":
        errors.append("description is required")
    elif len(description) > MAX_DESCRIPTION_LENGTH:
        errors.append(f"description exceeds {MAX_DESCRIPTION_LENGTH} characters ({len(description)})")
    return errors


def _parse_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    if not normalized.startswith("---"):
        return {}, normalized
    end_index = normalized.find("\n---", 3)
    if end_index == -1:
        return {}, normalized
    yaml_string = normalized[4:end_index]
    body = normalized[end_index + 4 :].strip()
    return (yaml.safe_load(yaml_string) or {}), body


def _dirname_env_path(path: str) -> str:
    normalized = path.rstrip("/\\")
    separator_index = max(normalized.rfind("/"), normalized.rfind("\\"))
    if separator_index == 2 and normalized[1] == ":":
        return normalized[:3]
    return "/" if separator_index <= 0 else normalized[:separator_index]


def _relative_env_path(root: Path, path: Path) -> str:
    normalized_root = str(root).replace("\\", "/").rstrip("/")
    normalized_path = str(path).replace("\\", "/").rstrip("/")
    if normalized_path == normalized_root:
        return ""
    if normalized_path.startswith(f"{normalized_root}/"):
        return normalized_path[len(normalized_root) + 1 :]
    return normalized_path.lstrip("/")


__all__ = [
    "SkillDiagnostic",
    "SkillDiagnosticCode",
    "SourcedSkill",
    "SourcedSkillDiagnostic",
    "format_skill_invocation",
    "load_skills",
    "load_sourced_skills",
]
