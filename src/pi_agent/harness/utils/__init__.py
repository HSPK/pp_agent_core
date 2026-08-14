"""Harness utility helpers: output truncation and shell-capture formatting."""

from __future__ import annotations

from .shell_output import (
    ShellCaptureOptions,
    ShellCaptureProgress,
    ShellCaptureResult,
    execute_shell_with_capture,
    sanitize_binary_output,
)
from .truncate import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_LINES,
    GREP_MAX_LINE_LENGTH,
    TruncatedLine,
    TruncationResult,
    format_size,
    truncate_head,
    truncate_line,
    truncate_tail,
)

__all__ = [
    "DEFAULT_MAX_BYTES",
    "DEFAULT_MAX_LINES",
    "GREP_MAX_LINE_LENGTH",
    "ShellCaptureOptions",
    "ShellCaptureProgress",
    "ShellCaptureResult",
    "TruncatedLine",
    "TruncationResult",
    "execute_shell_with_capture",
    "format_size",
    "sanitize_binary_output",
    "truncate_head",
    "truncate_line",
    "truncate_tail",
]
