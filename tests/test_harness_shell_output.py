"""Tests for `pi_agent.harness.utils.shell_output`.

No TS test file exists for `harness/utils/shell-output.ts`; this suite
exercises `execute_shell_with_capture` and `sanitize_binary_output` directly
against a minimal fake `ExecutionEnv`.
"""

from __future__ import annotations

from collections.abc import Callable

from pi_agent.harness.types import ExecutionError, FileError, ShellExecOptions, ShellExecResult, err, ok
from pi_agent.harness.utils.shell_output import (
    ShellCaptureOptions,
    execute_shell_with_capture,
    sanitize_binary_output,
)
from pi_agent.harness.utils.truncate import DEFAULT_MAX_BYTES


class FakeExecutionEnv:
    """Minimal `ExecutionEnv` stub implementing only what `execute_shell_with_capture` uses."""

    def __init__(
        self,
        exec_fn: Callable[[str, ShellExecOptions], object],
    ) -> None:
        self._exec_fn = exec_fn
        self.files: dict[str, str] = {}
        self._temp_counter = 0

    async def exec(self, command: str, options: ShellExecOptions | None = None):
        return await self._exec_fn(command, options)

    async def append_file(self, path: str, content: str, abort_signal=None):
        self.files[path] = self.files.get(path, "") + content
        return ok(None)

    async def create_temp_file(self, prefix: str = "", suffix: str = "", abort_signal=None):
        self._temp_counter += 1
        path = f"/tmp/{prefix}{self._temp_counter}{suffix}"
        self.files[path] = ""
        return ok(path)


async def test_captures_small_output_without_full_output_file():
    async def exec_fn(command, options):
        options.on_stdout("hello\n")
        options.on_stdout("world")
        return ok(ShellExecResult(stdout="hello\nworld", stderr="", exit_code=0))

    env = FakeExecutionEnv(exec_fn)
    result = await execute_shell_with_capture(env, "echo hello")

    assert result.ok
    assert result.value.output == "hello\nworld"
    assert result.value.exit_code == 0
    assert result.value.cancelled is False
    assert result.value.truncated is False
    assert result.value.full_output_path is None


async def test_mirrors_full_output_to_temp_file_when_output_exceeds_limit():
    big_chunk = "x" * (DEFAULT_MAX_BYTES + 100) + "\n"

    async def exec_fn(command, options):
        options.on_stdout(big_chunk)
        return ok(ShellExecResult(stdout=big_chunk, stderr="", exit_code=0))

    env = FakeExecutionEnv(exec_fn)
    result = await execute_shell_with_capture(env, "big-output")

    assert result.ok
    assert result.value.truncated is True
    assert result.value.full_output_path is not None
    assert env.files[result.value.full_output_path] == big_chunk


async def test_reports_cancellation_when_exec_returns_aborted_error():
    async def exec_fn(command, options):
        return err(ExecutionError("aborted", "command aborted"))

    env = FakeExecutionEnv(exec_fn)
    result = await execute_shell_with_capture(env, "sleep 10")

    assert result.ok
    assert result.value.cancelled is True
    assert result.value.exit_code is None


async def test_returns_error_result_when_exec_fails_and_not_requested_as_execution_error():
    async def exec_fn(command, options):
        return err(ExecutionError("spawn_error", "could not spawn"))

    env = FakeExecutionEnv(exec_fn)
    result = await execute_shell_with_capture(env, "bad-command")

    assert not result.ok
    assert result.error.code == "spawn_error"


async def test_returns_execution_error_inline_when_requested():
    async def exec_fn(command, options):
        return err(ExecutionError("spawn_error", "could not spawn"))

    env = FakeExecutionEnv(exec_fn)
    result = await execute_shell_with_capture(env, "bad-command", ShellCaptureOptions(return_execution_errors=True))

    assert result.ok
    assert result.value.execution_error is not None
    assert result.value.execution_error.code == "spawn_error"
    assert result.value.exit_code is None


async def test_returns_error_when_full_output_file_creation_fails():
    big_chunk = "x" * (DEFAULT_MAX_BYTES + 100) + "\n"

    async def exec_fn(command, options):
        options.on_stdout(big_chunk)
        return ok(ShellExecResult(stdout=big_chunk, stderr="", exit_code=0))

    env = FakeExecutionEnv(exec_fn)

    async def failing_create_temp_file(prefix="", suffix="", abort_signal=None):
        return err(FileError("unknown", "disk full"))

    env.create_temp_file = failing_create_temp_file  # type: ignore[method-assign]

    result = await execute_shell_with_capture(env, "big-output")

    assert not result.ok
    assert result.error.code == "unknown"


def test_sanitize_binary_output_strips_control_characters_but_keeps_whitespace():
    text = "line1\tvalue\x00\x01\x1fline2\nline3\r\n"
    sanitized = sanitize_binary_output(text)

    assert sanitized == "line1\tvalueline2\nline3\r\n"


def test_sanitize_binary_output_removes_object_replacement_characters():
    text = "a\ufffab"
    assert sanitize_binary_output(text) == "ab"
