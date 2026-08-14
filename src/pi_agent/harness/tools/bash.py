"""The harness `bash` tool.

Python port of `packages/agent/src/harness/tools/bash.ts`.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from pi_ai.types import TextContent
from pi_ai.utils.abort import AbortSignal

from ..types import AgentHarnessTool, AgentToolResult, AgentToolUpdateCallback, ExecutionError, get_or_throw
from ..utils.shell_output import ShellCaptureOptions, ShellCaptureProgress, execute_shell_with_capture
from ..utils.truncate import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, TruncationResult, format_size
from .tool_context import ExecutionToolContext

MAX_TIMEOUT_SECONDS = 2_147_483_647 / 1000
BASH_UPDATE_THROTTLE_SECONDS = 0.1

BASH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "command": {"type": "string", "description": "Bash command to execute"},
        "timeout": {"type": "number", "description": "Timeout in seconds (optional, no default timeout)"},
    },
    "required": ["command"],
}


@dataclass
class BashToolDetails:
    truncation: TruncationResult | None = None
    full_output_path: str | None = None


@dataclass
class BashExecution:
    """The command, cwd and environment a `BashPrepare` hook may mutate before spawning."""

    command: str
    cwd: str
    env: dict[str, str] = field(default_factory=dict)
    inherit_env: bool = True


BashPrepare = Callable[[BashExecution, ExecutionToolContext, "AbortSignal | None"], Awaitable[None] | None]


def _validate_timeout(timeout: float | None) -> None:
    if timeout is None:
        return
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Invalid timeout: must be a finite number of seconds")
    if timeout > MAX_TIMEOUT_SECONDS:
        raise ValueError(f"Invalid timeout: maximum is {MAX_TIMEOUT_SECONDS} seconds")


def create_bash_tool(
    command_prefix: str | None = None, prepare: BashPrepare | None = None
) -> AgentHarnessTool[ExecutionToolContext]:
    """Create the `bash` tool: run a command, streaming throttled output updates."""

    async def execute(
        _tool_call_id: str,
        args: dict[str, Any],
        signal: AbortSignal | None,
        on_update: AgentToolUpdateCallback | None,
        context: ExecutionToolContext,
    ) -> AgentToolResult:
        command = args["command"]
        timeout = args.get("timeout")
        _validate_timeout(timeout)
        env = context.env
        execution = BashExecution(
            command=f"{command_prefix}\n{command}" if command_prefix else command,
            cwd=env.cwd,
        )
        if prepare is not None:
            result = prepare(execution, context, signal)
            if asyncio.iscoroutine(result):
                await result

        state = _UpdateThrottle(on_update)
        if on_update is not None:
            on_update(AgentToolResult(content=[], details=None))
        try:
            capture = get_or_throw(
                await execute_shell_with_capture(
                    env,
                    execution.command,
                    ShellCaptureOptions(
                        cwd=execution.cwd,
                        env=execution.env,
                        inherit_env=execution.inherit_env,
                        timeout=timeout,
                        abort_signal=signal,
                        return_execution_errors=True,
                        on_chunk=state.on_chunk,
                    ),
                )
            )
            state.cancel_timer()
            state.get_latest_progress = lambda: capture
            state.dirty = True
            state.emit()

            output_text = capture.output
            details: BashToolDetails | None = None
            if capture.truncation.truncated:
                details = BashToolDetails(truncation=capture.truncation, full_output_path=capture.full_output_path)
                start_line = capture.truncation.total_lines - capture.truncation.output_lines + 1
                end_line = capture.truncation.total_lines
                if capture.truncation.last_line_partial:
                    last_line_size = format_size(capture.last_line_bytes)
                    output_text += (
                        f"\n\n[Showing last {format_size(capture.truncation.output_bytes)} of line {end_line} "
                        f"(line is {last_line_size}). Full output: {capture.full_output_path}]"
                    )
                elif capture.truncation.truncated_by == "lines":
                    output_text += (
                        f"\n\n[Showing lines {start_line}-{end_line} of {capture.truncation.total_lines}. "
                        f"Full output: {capture.full_output_path}]"
                    )
                else:
                    output_text += (
                        f"\n\n[Showing lines {start_line}-{end_line} of {capture.truncation.total_lines} "
                        f"({format_size(DEFAULT_MAX_BYTES)} limit). Full output: {capture.full_output_path}]"
                    )

            def append_status(status: str) -> str:
                return f"{output_text}\n\n{status}" if output_text else status

            if capture.cancelled:
                raise RuntimeError(append_status("Command aborted"))
            if capture.execution_error is not None and capture.execution_error.code == "timeout":
                failure = RuntimeError(append_status(f"Command timed out after {timeout} seconds"))
                failure.__cause__ = capture.execution_error
                raise failure
            if capture.execution_error is not None:
                raise capture.execution_error
            if capture.exit_code != 0 and capture.exit_code is not None:
                raise RuntimeError(append_status(f"Command exited with code {capture.exit_code}"))
            return AgentToolResult(content=[TextContent(text=output_text or "(no output)")], details=details)
        finally:
            state.cancel_timer()

    return AgentHarnessTool(
        name="bash",
        label="bash",
        description=(
            "Execute a bash command in the current working directory. Returns stdout and stderr. Output is "
            f"truncated to last {DEFAULT_MAX_LINES} lines or {DEFAULT_MAX_BYTES // 1024}KB (whichever is hit "
            "first). If truncated, full output is saved to a temp file. Optionally provide a timeout in seconds."
        ),
        parameters=BASH_SCHEMA,
        execute_with_context=execute,
    )


class _UpdateThrottle:
    """Coalesce streamed output into at most one `on_update` per throttle window."""

    def __init__(self, on_update: AgentToolUpdateCallback | None) -> None:
        self._on_update = on_update
        self.get_latest_progress: Callable[[], ShellCaptureProgress] | None = None
        self.dirty = False
        self._last_update_at = 0.0
        self._timer: asyncio.TimerHandle | None = None

    def emit(self) -> None:
        if self._on_update is None or not self.dirty or self.get_latest_progress is None:
            return
        self.dirty = False
        self._last_update_at = time.monotonic()
        progress = self.get_latest_progress()
        self._on_update(
            AgentToolResult(
                content=[TextContent(text=progress.output)],
                details=BashToolDetails(
                    truncation=progress.truncation if progress.truncation.truncated else None,
                    full_output_path=progress.full_output_path,
                ),
            )
        )

    def cancel_timer(self) -> None:
        if self._timer is None:
            return
        self._timer.cancel()
        self._timer = None

    def on_chunk(self, _chunk: str, get_progress: Callable[[], ShellCaptureProgress]) -> None:
        self.get_latest_progress = get_progress
        if self._on_update is None:
            return
        self.dirty = True
        delay = BASH_UPDATE_THROTTLE_SECONDS - (time.monotonic() - self._last_update_at)
        if delay <= 0:
            self.cancel_timer()
            self.emit()
            return
        if self._timer is None:
            self._timer = asyncio.get_running_loop().call_later(delay, self._on_timer)

    def _on_timer(self) -> None:
        self._timer = None
        self.emit()


__all__ = [
    "BASH_SCHEMA",
    "BASH_UPDATE_THROTTLE_SECONDS",
    "MAX_TIMEOUT_SECONDS",
    "BashExecution",
    "BashPrepare",
    "BashToolDetails",
    "ExecutionError",
    "create_bash_tool",
]
