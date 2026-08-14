"""Shell execution with tail-truncated, mirrored full-output capture.

Python port of `packages/agent/src/harness/utils/shell-output.ts`.
`execute_shell_with_capture` streams stdout/stderr chunks through an
`ExecutionEnv`, keeping a byte-bounded tail in memory while mirroring the
full, untruncated output to a temp file once either limit is exceeded.

TypeScript's `onStdout`/`onStderr` callbacks run synchronously per chunk but
schedule their file writes on a promise chain (`writeChain =
writeChain.then(...)`) so writes stay ordered without blocking the callback;
the chain is awaited once, after `exec()` resolves. This port reproduces that
with `_chain`, which schedules each write as an `asyncio.Task` appended after
the previously scheduled one, and awaits the final task after `env.exec`
returns.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, replace
from typing import Any

from ..types import ExecutionEnv, ExecutionError, Result, ShellExecOptions, err, ok, to_error
from .truncate import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, TruncationResult, truncate_tail


@dataclass(kw_only=True)
class ShellCaptureProgress:
    output: str
    truncation: TruncationResult
    last_line_bytes: int
    full_output_path: str | None = None


@dataclass(kw_only=True)
class ShellCaptureOptions:
    """`ShellExecOptions` minus the `on_stdout`/`on_stderr` callbacks, plus capture-specific options."""

    cwd: str | None = None
    env: dict[str, str] | None = None
    inherit_env: bool = True
    timeout: float | None = None
    abort_signal: object | None = None
    on_chunk: Callable[[str, Callable[[], ShellCaptureProgress]], None] | None = None
    return_execution_errors: bool = False
    """Return shell execution failures with captured output instead of as a failed `Result`."""


@dataclass(kw_only=True)
class ShellCaptureResult(ShellCaptureProgress):
    exit_code: int | None
    cancelled: bool
    truncated: bool
    execution_error: ExecutionError | None = None


def _to_execution_error(error: BaseException) -> ExecutionError:
    if isinstance(error, ExecutionError):
        return error
    cause = to_error(error)
    return ExecutionError("unknown", str(cause), cause)


def sanitize_binary_output(text: str) -> str:
    kept: list[str] = []
    for char in text:
        code = ord(char)
        if code in (0x09, 0x0A, 0x0D):
            kept.append(char)
            continue
        if code <= 0x1F:
            continue
        if 0xFFF9 <= code <= 0xFFFB:
            continue
        kept.append(char)
    return "".join(kept)


def _trim_to_last_utf8_bytes(text: str, max_bytes: int) -> str:
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    start = len(encoded) - max_bytes
    while start < len(encoded) and (encoded[start] & 0xC0) == 0x80:
        start += 1
    return encoded[start:].decode("utf-8")


async def execute_shell_with_capture(
    env: ExecutionEnv, command: str, options: ShellCaptureOptions | None = None
) -> Result[ShellCaptureResult, ExecutionError]:
    options = options or ShellCaptureOptions()
    tail_output = ""
    max_output_bytes = DEFAULT_MAX_BYTES * 2

    total_bytes = 0
    completed_lines = 0
    has_open_line = False
    current_line_bytes = 0
    full_output_path: str | None = None
    full_output_requested = False
    accepting_output = True
    capture_error: ExecutionError | None = None
    write_chain: asyncio.Future[Result[None, ExecutionError]] = asyncio.get_event_loop().create_future()
    write_chain.set_result(ok(None))

    def chain(factory: Callable[[], Coroutine[Any, Any, Result[None, ExecutionError]]]) -> None:
        nonlocal write_chain
        previous = write_chain

        async def step() -> Result[None, ExecutionError]:
            result = await previous
            if not result.ok:
                return result
            return await factory()

        write_chain = asyncio.ensure_future(step())

    async def append_full_output_step(text: str) -> Result[None, ExecutionError]:
        if not full_output_requested or capture_error is not None:
            return ok(None)
        if full_output_path is None:
            return err(ExecutionError("unknown", "Full output path was not created"))
        append_result = await env.append_file(full_output_path, text)
        return append_result if append_result.ok else err(_to_execution_error(append_result.error))

    def append_full_output(text: str) -> None:
        if not full_output_requested or capture_error is not None:
            return
        chain(lambda: append_full_output_step(text))

    async def ensure_full_output_file_step(initial_content: str) -> Result[None, ExecutionError]:
        nonlocal full_output_path
        temp_file = await env.create_temp_file(prefix="bash-", suffix=".log")
        if not temp_file.ok:
            return err(_to_execution_error(temp_file.error))
        full_output_path = temp_file.value
        append_result = await env.append_file(temp_file.value, initial_content)
        return ok(None) if append_result.ok else err(_to_execution_error(append_result.error))

    def ensure_full_output_file(initial_content: str) -> None:
        nonlocal full_output_requested
        if full_output_requested or capture_error is not None:
            return
        full_output_requested = True
        chain(lambda: ensure_full_output_file_step(initial_content))

    def create_progress() -> ShellCaptureProgress:
        tail_truncation = truncate_tail(tail_output)
        total_lines = completed_lines + (1 if has_open_line else 0)
        truncated = total_lines > DEFAULT_MAX_LINES or total_bytes > DEFAULT_MAX_BYTES
        truncation = replace(
            tail_truncation,
            truncated=truncated,
            truncated_by=(
                (tail_truncation.truncated_by or ("bytes" if total_bytes > DEFAULT_MAX_BYTES else "lines"))
                if truncated
                else None
            ),
            total_lines=total_lines,
            total_bytes=total_bytes,
        )
        return ShellCaptureProgress(
            output=truncation.content if truncated else tail_output,
            truncation=truncation,
            full_output_path=full_output_path,
            last_line_bytes=current_line_bytes,
        )

    def on_chunk(chunk: str) -> None:
        nonlocal tail_output, total_bytes, completed_lines, has_open_line, current_line_bytes, capture_error
        if not accepting_output:
            return
        try:
            text = sanitize_binary_output(chunk).replace("\r", "")
            text_bytes = len(text.encode("utf-8"))
            total_bytes += text_bytes
            newline_count = text.count("\n")
            completed_lines += newline_count
            last_newline = text.rfind("\n")
            if last_newline >= 0:
                trailing_text = text[last_newline + 1 :]
                current_line_bytes = len(trailing_text.encode("utf-8"))
                has_open_line = len(trailing_text) > 0
            elif len(text) > 0:
                current_line_bytes += text_bytes
                has_open_line = True

            tail_output += text
            total_lines = completed_lines + (1 if has_open_line else 0)
            if (total_bytes > DEFAULT_MAX_BYTES or total_lines > DEFAULT_MAX_LINES) and not full_output_requested:
                ensure_full_output_file(tail_output)
            elif full_output_requested:
                append_full_output(text)
            tail_output = _trim_to_last_utf8_bytes(tail_output, max_output_bytes)
            if options.on_chunk is not None:
                options.on_chunk(text, create_progress)
        except Exception as error:
            capture_error = _to_execution_error(error)

    try:
        exec_options = ShellExecOptions(
            cwd=options.cwd,
            env=options.env,
            inherit_env=options.inherit_env,
            timeout=options.timeout,
            abort_signal=options.abort_signal,  # type: ignore[arg-type]
            on_stdout=on_chunk,
            on_stderr=on_chunk,
        )
        result = await env.exec(command, exec_options)
        accepting_output = False
        progress = create_progress()
        if progress.truncation.truncated and not full_output_requested:
            ensure_full_output_file(tail_output)
        write_result = await write_chain
        if not write_result.ok:
            return err(write_result.error)
        if capture_error is not None:
            return err(capture_error)
        progress = create_progress()

        aborted = bool(options.abort_signal and getattr(options.abort_signal, "aborted", False))

        if not result.ok:
            if result.error.code == "aborted" or aborted:
                return ok(
                    ShellCaptureResult(
                        output=progress.output,
                        truncation=progress.truncation,
                        full_output_path=progress.full_output_path,
                        last_line_bytes=progress.last_line_bytes,
                        exit_code=None,
                        cancelled=True,
                        truncated=progress.truncation.truncated,
                    )
                )
            if options.return_execution_errors:
                return ok(
                    ShellCaptureResult(
                        output=progress.output,
                        truncation=progress.truncation,
                        full_output_path=progress.full_output_path,
                        last_line_bytes=progress.last_line_bytes,
                        exit_code=None,
                        cancelled=False,
                        truncated=progress.truncation.truncated,
                        execution_error=result.error,
                    )
                )
            return err(result.error)

        cancelled = aborted
        return ok(
            ShellCaptureResult(
                output=progress.output,
                truncation=progress.truncation,
                full_output_path=progress.full_output_path,
                last_line_bytes=progress.last_line_bytes,
                exit_code=None if cancelled else result.value.exit_code,
                cancelled=cancelled,
                truncated=progress.truncation.truncated,
            )
        )
    except Exception as error:
        accepting_output = False
        return err(_to_execution_error(error))


__all__ = [
    "ShellCaptureOptions",
    "ShellCaptureProgress",
    "ShellCaptureResult",
    "execute_shell_with_capture",
    "sanitize_binary_output",
]
