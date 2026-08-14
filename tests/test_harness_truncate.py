"""Tests for `pi_agent.harness.utils.truncate`.

Ported from `packages/agent/test/harness/truncate.test.ts`. Two TypeScript
cases exercise unpaired UTF-16 surrogates ("matches Buffer tail truncation
semantics for surrogate edge cases" and the deterministic fuzz test, whose
alphabet includes lone surrogate code units). Python `str` stores code points,
not UTF-16 code units (see the module docstring in `truncate.py`), so those
inputs cannot be constructed: `"\ud83d".encode("utf-8")` raises
`UnicodeEncodeError`. Both cases are ported with the surrogate inputs dropped
and the remaining valid-Unicode inputs kept, so the same "truncated tail bytes
match a manual byte-boundary trim" property is still asserted.
"""

from __future__ import annotations

from pi_agent.harness.utils.truncate import GREP_MAX_LINE_LENGTH, truncate_head, truncate_line, truncate_tail


def _byte_length(content: str) -> int:
    return len(content.encode("utf-8"))


def _buffer_tail(content: str, max_bytes: int) -> str:
    encoded = content.encode("utf-8")
    if len(encoded) <= max_bytes:
        return content
    start = len(encoded) - max_bytes
    while start < len(encoded) and (encoded[start] & 0xC0) == 0x80:
        start += 1
    return encoded[start:].decode("utf-8")


def _assert_matches_buffer_tail(input_str: str, max_byte_values: list[int] | None = None) -> None:
    total_bytes = _byte_length(input_str)
    values = max_byte_values if max_byte_values is not None else list(range(total_bytes + 5))
    for max_bytes in values:
        result = truncate_tail(input_str, max_bytes=max_bytes, max_lines=10)
        expected = _buffer_tail(input_str, max_bytes)
        assert result.content == expected, (
            f"tail mismatch input={input_str!r} maxBytes={max_bytes} expected={expected!r} actual={result.content!r}"
        )
        output_bytes = _byte_length(result.content)
        assert output_bytes <= max_bytes, (
            f"tail output exceeded byte limit input={input_str!r} maxBytes={max_bytes} outputBytes={output_bytes}"
        )


def _sampled_byte_limits(input_str: str) -> list[int]:
    total_bytes = _byte_length(input_str)
    candidates = [
        0,
        1,
        2,
        3,
        4,
        5,
        8,
        total_bytes // 2 - 1,
        total_bytes // 2,
        total_bytes // 2 + 1,
        total_bytes - 8,
        total_bytes - 5,
        total_bytes - 4,
        total_bytes - 3,
        total_bytes - 2,
        total_bytes - 1,
        total_bytes,
        total_bytes + 1,
        total_bytes + 4,
    ]
    return sorted({value for value in candidates if value >= 0})


def test_counts_utf8_bytes_without_node_buffer():
    content = "aé🙂\nb"
    result = truncate_head(content, max_bytes=100, max_lines=10)

    assert result.truncated is False
    assert result.total_bytes == _byte_length(content)
    assert result.output_bytes == _byte_length(content)
    assert result.total_bytes == 9


def test_does_not_count_trailing_newline_as_extra_line():
    content = "\n".join(["line"] * 3) + "\n"
    head = truncate_head(content, max_bytes=100, max_lines=3)
    tail = truncate_tail(content, max_bytes=100, max_lines=3)

    assert head.truncated is False
    assert head.total_lines == 3
    assert head.output_lines == 3
    assert tail.truncated is False
    assert tail.total_lines == 3
    assert tail.output_lines == 3


def test_truncates_head_on_utf8_byte_limits_without_partial_lines():
    content = "éé\nabc"
    result = truncate_head(content, max_bytes=4, max_lines=10)

    assert result.content == "éé"
    assert result.truncated is True
    assert result.truncated_by == "bytes"
    assert result.output_bytes == 4
    assert result.first_line_exceeds_limit is False


def test_reports_head_truncation_when_first_line_exceeds_byte_limit():
    result = truncate_head("éé\nabc", max_bytes=3, max_lines=10)

    assert result.content == ""
    assert result.truncated is True
    assert result.truncated_by == "bytes"
    assert result.first_line_exceeds_limit is True


def test_truncates_tail_on_utf8_boundaries_when_only_partial_last_line_fits():
    result = truncate_tail("aé🙂b", max_bytes=5, max_lines=10)

    assert result.content == "🙂b"
    assert result.truncated is True
    assert result.truncated_by == "bytes"
    assert result.last_line_partial is True
    assert result.output_bytes == 5


def test_truncates_oversized_single_line_with_trailing_newline():
    input_str = "X" * 300_000 + "\n"
    result = truncate_tail(input_str, max_bytes=1024, max_lines=100)

    assert result.content == "X" * 1024
    assert result.output_bytes == 1024
    assert result.output_lines == 1
    assert result.last_line_partial is True
    assert result.truncated_by == "bytes"


def test_drops_oversized_trailing_character_when_it_cannot_fit_in_tail_byte_limit():
    result = truncate_tail("abc🙂", max_bytes=3, max_lines=10)

    assert result.content == ""
    assert result.truncated is True
    assert result.truncated_by == "bytes"
    assert result.last_line_partial is True
    assert result.output_bytes == 0


def test_matches_buffer_tail_truncation_semantics_for_surrogate_edge_cases():
    """Port of the surrogate edge-case test, restricted to its one portable input.

    The TypeScript inputs "a\ud83d", "\ude42b", "a\ude42b", "\ud83d\ud83d\ude42" and
    "\ud83d\ude42\ude42" are lone or mis-paired UTF-16 surrogates. Python `str` stores
    code points, not UTF-16 code units, so those strings cannot be constructed at all
    (`"\ud83d".encode("utf-8")` raises `UnicodeEncodeError`). Only the ZWJ sequence
    input applies.
    """
    _assert_matches_buffer_tail("\U0001f469\u200d\U0001f4bb")


def test_matches_buffer_tail_truncation_semantics_across_deterministic_fuzz_cases():
    alphabet = [
        "a",
        "\u007f",
        "\u0080",
        "é",
        "\u07ff",
        "\u0800",
        "中",
        "\ud7ff",
        "\ue000",
        "🙂",
        "\uffff",
    ]

    def check_exhaustive(prefix: str, depth: int) -> None:
        _assert_matches_buffer_tail(prefix, _sampled_byte_limits(prefix))
        if depth == 0:
            return
        for character in alphabet:
            check_exhaustive(prefix + character, depth - 1)

    check_exhaustive("", 3)

    seed = 0x12345678

    def random() -> float:
        nonlocal seed
        seed = (seed * 1664525 + 1013904223) & 0xFFFFFFFF
        return seed / 0x100000000

    for _ in range(1_000):
        length = int(random() * 80)
        input_str = "".join(alphabet[int(random() * len(alphabet))] for _ in range(length))
        _assert_matches_buffer_tail(input_str, _sampled_byte_limits(input_str))


def test_truncate_line_returns_unchanged_when_within_limit():
    result = truncate_line("short line")

    assert result.text == "short line"
    assert result.was_truncated is False


def test_truncate_line_truncates_and_appends_suffix_when_exceeding_max_chars():
    line = "x" * (GREP_MAX_LINE_LENGTH + 50)
    result = truncate_line(line)

    assert result.text == "x" * GREP_MAX_LINE_LENGTH + "... [truncated]"
    assert result.was_truncated is True
