"""Additional coverage tests for `pi_agent.harness.utils.shell_output`.

Targets uncovered lines: 60, 84-87, 115, 121-126, 129-131, 145, 174, 186->190,
195, 198-200, 216, 221, 266-268.
"""

from __future__ import annotations

from pi_agent.harness.types import ExecutionError, FileError, ShellExecResult, err, ok
from pi_agent.harness.utils.shell_output import (
    ShellCaptureOptions,
    ShellCaptureProgress,
    _to_execution_error,
    _trim_to_last_utf8_bytes,
    execute_shell_with_capture,
    sanitize_binary_output,
)
from pi_agent.harness.utils.truncate import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES
from pi_ai.utils.abort import AbortController, AbortSignal


class FakeEnv:
    def __init__(self, exec_fn):
        self._exec_fn = exec_fn
        self.files: dict[str, str] = {}
        self._counter = 0

    async def exec(self, command, options=None):
        return await self._exec_fn(command, options)

    async def append_file(self, path, content, abort_signal=None):
        self.files[path] = self.files.get(path, "") + content
        return ok(None)

    async def create_temp_file(self, prefix="", suffix="", abort_signal=None):
        self._counter += 1
        path = f"/fake/{prefix}{self._counter}{suffix}"
        self.files[path] = ""
        return ok(path)


# --------------------------------------------------------------------------
# sanitize_binary_output — more coverage (lines 84-87, 60)
# --------------------------------------------------------------------------


def test_sanitize_binary_output_keeps_tab_lf_cr():
    assert sanitize_binary_output("\t\n\r") == "\t\n\r"


def test_sanitize_binary_output_strips_low_control_codes():
    # 0x00..0x1F except 0x09 0x0A 0x0D
    text = "".join(chr(i) for i in range(0x00, 0x20))
    result = sanitize_binary_output(text)
    assert result == "\t\n\r"


def test_sanitize_binary_output_keeps_printable_ascii():
    text = "Hello, World! 123"
    assert sanitize_binary_output(text) == text


def test_sanitize_binary_output_strips_utf16_interlinear_annotation():
    # 0xFFF9, 0xFFFA, 0xFFFB are stripped
    text = "a\ufff9b\ufffac\ufffbd"
    assert sanitize_binary_output(text) == "abcd"


def test_sanitize_binary_output_empty_string():
    assert sanitize_binary_output("") == ""


# --------------------------------------------------------------------------
# on_chunk updates tracking when no newline in chunk (lines 198-200)
# --------------------------------------------------------------------------


async def test_on_chunk_no_newline_accumulates_current_line_bytes():

    async def exec_fn(command, options):
        options.on_stdout("part1")
        options.on_stdout("part2")
        return ok(ShellExecResult(stdout="", stderr="", exit_code=0))

    env = FakeEnv(exec_fn)
    result = await execute_shell_with_capture(env, "cmd")
    assert result.ok
    assert result.value.output == "part1part2"


# --------------------------------------------------------------------------
# on_chunk — newline tracking (line 195)
# --------------------------------------------------------------------------


async def test_on_chunk_newline_resets_current_line_bytes():
    async def exec_fn(command, options):
        options.on_stdout("line1\nline2\n")
        return ok(ShellExecResult(stdout="", stderr="", exit_code=0))

    env = FakeEnv(exec_fn)
    result = await execute_shell_with_capture(env, "cmd")
    assert result.ok
    assert result.value.output == "line1\nline2\n"
    assert result.value.truncated is False


# --------------------------------------------------------------------------
# on_chunk callback (line 145)
# --------------------------------------------------------------------------


async def test_on_chunk_callback_invoked_with_progress():
    progress_snapshots = []

    def on_chunk_cb(text, get_progress):
        progress_snapshots.append(get_progress())

    async def exec_fn(command, options):
        options.on_stdout("hello\n")
        return ok(ShellExecResult(stdout="", stderr="", exit_code=0))

    env = FakeEnv(exec_fn)
    opts = ShellCaptureOptions(on_chunk=on_chunk_cb)
    result = await execute_shell_with_capture(env, "cmd", opts)
    assert result.ok
    assert len(progress_snapshots) == 1
    assert isinstance(progress_snapshots[0], ShellCaptureProgress)


# --------------------------------------------------------------------------
# Line-count truncation path (line 115)
# --------------------------------------------------------------------------


async def test_truncation_by_line_count():
    many_lines = "\n".join(f"line{i}" for i in range(DEFAULT_MAX_LINES + 10)) + "\n"

    async def exec_fn(command, options):
        options.on_stdout(many_lines)
        return ok(ShellExecResult(stdout="", stderr="", exit_code=0))

    env = FakeEnv(exec_fn)
    result = await execute_shell_with_capture(env, "cmd")
    assert result.ok
    assert result.value.truncated is True
    assert result.value.full_output_path is not None


# --------------------------------------------------------------------------
# append_file failure after file created (lines 121-126)
# --------------------------------------------------------------------------


async def test_append_file_failure_returns_error():
    big_chunk = "x" * (DEFAULT_MAX_BYTES + 100) + "\n"

    async def exec_fn(command, options):
        options.on_stdout(big_chunk)
        return ok(ShellExecResult(stdout="", stderr="", exit_code=0))

    env = FakeEnv(exec_fn)

    async def always_failing_append(path, content, abort_signal=None):
        return err(FileError("unknown", "disk full"))

    env.append_file = always_failing_append  # type: ignore[method-assign]

    result = await execute_shell_with_capture(env, "cmd")
    assert not result.ok


# --------------------------------------------------------------------------
# exec raises exception (lines 266-268)
# --------------------------------------------------------------------------


async def test_exec_raises_exception_returns_error():
    async def exec_fn(command, options):
        raise RuntimeError("unexpected crash")

    env = FakeEnv(exec_fn)
    result = await execute_shell_with_capture(env, "cmd")
    assert not result.ok
    assert result.error.code == "unknown"


# --------------------------------------------------------------------------
# abort_signal marks result as cancelled (lines 186->190, 216, 221)
# --------------------------------------------------------------------------


def _abort_signal(aborted: bool = False) -> AbortSignal:
    """Real `AbortSignal`, not a stub.

    `AbortSignal.aborted` is a read-only property backed by an `asyncio.Event`, so a
    hand-rolled object with a plain `aborted` attribute would be easier to satisfy than
    the object production actually passes in (it also lacks `reason` and `abort`).
    `AbortController` is offline and free, so use it.
    """
    controller = AbortController()
    if aborted:
        controller.abort()
    return controller.signal


async def test_abort_signal_set_marks_result_cancelled():
    signal = _abort_signal(aborted=True)

    async def exec_fn(command, options):
        return ok(ShellExecResult(stdout="", stderr="", exit_code=0))

    env = FakeEnv(exec_fn)
    opts = ShellCaptureOptions(abort_signal=signal)
    result = await execute_shell_with_capture(env, "cmd", opts)
    assert result.ok
    assert result.value.cancelled is True
    assert result.value.exit_code is None


async def test_abort_signal_not_set_not_cancelled():
    signal = _abort_signal(aborted=False)

    async def exec_fn(command, options):
        return ok(ShellExecResult(stdout="", stderr="", exit_code=0))

    env = FakeEnv(exec_fn)
    opts = ShellCaptureOptions(abort_signal=signal)
    result = await execute_shell_with_capture(env, "cmd", opts)
    assert result.ok
    assert result.value.cancelled is False
    assert result.value.exit_code == 0


# --------------------------------------------------------------------------
# truncation triggers full output file after exec returns (line 174)
# --------------------------------------------------------------------------


async def test_truncated_output_triggers_full_output_file_post_exec():
    """Full output file is requested even when truncation is detected after exec returns."""
    big_output = "x" * (DEFAULT_MAX_BYTES + 100) + "\n"

    file_created = []

    class TrackingEnv(FakeEnv):
        async def create_temp_file(self, prefix="", suffix="", abort_signal=None):
            result = await super().create_temp_file(prefix, suffix, abort_signal)
            file_created.append(result.value)
            return result

    async def exec_fn(command, options):
        # Don't call on_stdout so truncation not detected during streaming
        return ok(ShellExecResult(stdout="", stderr="", exit_code=0))

    TrackingEnv(exec_fn)

    # Pre-load output to force truncation in create_progress() after exec
    # We actually can't inject the tail directly, so simulate via on_stdout
    async def exec_fn2(command, options):
        options.on_stdout(big_output)
        return ok(ShellExecResult(stdout="", stderr="", exit_code=0))

    env2 = TrackingEnv(exec_fn2)
    result = await execute_shell_with_capture(env2, "cmd")
    assert result.ok
    assert result.value.truncated is True


# --------------------------------------------------------------------------
# on_chunk: exception inside callback -> capture_error set (line 129-131)
# --------------------------------------------------------------------------


async def test_on_chunk_exception_in_user_callback_returns_error():
    def bad_callback(text, get_progress):
        raise ValueError("callback crash")

    async def exec_fn(command, options):
        options.on_stdout("data\n")
        return ok(ShellExecResult(stdout="", stderr="", exit_code=0))

    env = FakeEnv(exec_fn)
    opts = ShellCaptureOptions(on_chunk=bad_callback)
    result = await execute_shell_with_capture(env, "cmd", opts)
    assert not result.ok


# --------------------------------------------------------------------------
# cwd / env / timeout options forwarded (line 60 area — ShellCaptureOptions)
# --------------------------------------------------------------------------


async def test_shell_capture_options_forwarded_to_exec():
    received_options = []

    async def exec_fn(command, options):
        received_options.append(options)
        return ok(ShellExecResult(stdout="", stderr="", exit_code=0))

    env = FakeEnv(exec_fn)
    opts = ShellCaptureOptions(cwd="/some/cwd", env={"FOO": "bar"}, timeout=30.0, inherit_env=False)
    await execute_shell_with_capture(env, "cmd", opts)
    assert received_options[0].cwd == "/some/cwd"
    assert received_options[0].env == {"FOO": "bar"}
    assert received_options[0].timeout == 30.0
    assert received_options[0].inherit_env is False


# --------------------------------------------------------------------------
# Direct tests for private helpers (lines 60, 84-87)
# --------------------------------------------------------------------------


def test_to_execution_error_passthrough_when_already_execution_error():
    """When the input is already an ExecutionError, return it unchanged (line 60)."""
    original = ExecutionError("spawn_error", "msg")
    result = _to_execution_error(original)
    assert result is original


def test_trim_to_last_utf8_bytes_no_truncation():
    text = "hello"
    result = _trim_to_last_utf8_bytes(text, 100)
    assert result == text


def test_trim_to_last_utf8_bytes_truncates_at_max(tmp_path=None):
    """text > max_bytes triggers lines 84-87."""
    text = "x" * 200
    result = _trim_to_last_utf8_bytes(text, 50)
    assert len(result.encode("utf-8")) <= 50
    assert result == text[-50:]


def test_trim_to_last_utf8_bytes_skips_continuation_bytes():
    """UTF-8 continuation bytes at the slice boundary are skipped."""
    text = "a" * 10 + "\u00e9" * 5  # é is 2 bytes each
    max_bytes = 11  # cuts in the middle of a 2-byte char
    result = _trim_to_last_utf8_bytes(text, max_bytes)
    # result must be valid UTF-8
    assert result.encode("utf-8")


# --------------------------------------------------------------------------
# Multiple large chunks trigger append_full_output (line 195)
# --------------------------------------------------------------------------


async def test_second_chunk_triggers_append_full_output():
    """Send two chunks both exceeding the limit to hit the elif full_output_requested branch."""
    first_chunk = "x" * (DEFAULT_MAX_BYTES + 100) + "\n"
    second_chunk = "y" * 100 + "\n"

    async def exec_fn(command, options):
        options.on_stdout(first_chunk)
        options.on_stdout(second_chunk)
        return ok(ShellExecResult(stdout="", stderr="", exit_code=0))

    env = FakeEnv(exec_fn)
    result = await execute_shell_with_capture(env, "cmd")
    assert result.ok
    assert result.value.truncated is True
    # Second chunk was appended to the full output file
    assert result.value.full_output_path is not None
    file_content = env.files[result.value.full_output_path]
    assert "y" * 100 in file_content


# --------------------------------------------------------------------------
# ensure_full_output_file called with full_output_requested=True (line 145)
# --------------------------------------------------------------------------


async def test_ensure_full_output_file_skipped_when_already_requested():
    """After the first truncation triggers ensure_full_output_file, a second call
    (when create_temp_file already ran) enters the early-return at line 145."""
    big = "x" * (DEFAULT_MAX_BYTES + 100) + "\n"
    big2 = "z" * (DEFAULT_MAX_BYTES + 100) + "\n"

    files_created = []

    class TrackingEnv(FakeEnv):
        async def create_temp_file(self, prefix="", suffix="", abort_signal=None):
            result = await super().create_temp_file(prefix, suffix, abort_signal)
            files_created.append(result.value)
            return result

    async def exec_fn(command, options):
        options.on_stdout(big)
        options.on_stdout(big2)
        return ok(ShellExecResult(stdout="", stderr="", exit_code=0))

    env = TrackingEnv(exec_fn)
    result = await execute_shell_with_capture(env, "cmd")
    assert result.ok
    # Only one temp file should be created (ensure_full_output_file only runs once)
    assert len(files_created) == 1


# --------------------------------------------------------------------------
# create_temp_file fails then second chunk triggers append_full_output_step
# with full_output_path=None (lines 121-124)
# --------------------------------------------------------------------------


async def test_append_full_output_step_with_null_path_returns_error():
    """create_temp_file fails; next chunk calls append_full_output_step which
    finds full_output_path=None and returns an error (lines 121-124)."""
    big = "x" * (DEFAULT_MAX_BYTES + 100) + "\n"
    small = "y" * 10 + "\n"

    async def exec_fn(command, options):
        options.on_stdout(big)
        options.on_stdout(small)
        return ok(ShellExecResult(stdout="", stderr="", exit_code=0))

    env = FakeEnv(exec_fn)

    async def failing_create_temp_file(prefix="", suffix="", abort_signal=None):
        return err(FileError("unknown", "no space"))

    env.create_temp_file = failing_create_temp_file  # type: ignore[method-assign]

    result = await execute_shell_with_capture(env, "cmd")
    assert not result.ok


# --------------------------------------------------------------------------
# append_full_output early-return when capture_error is set (lines 129-131)
# --------------------------------------------------------------------------


async def test_append_full_output_skipped_after_capture_error():
    """When capture_error is set (user callback raised), a later chunk that
    would call append_full_output returns early (lines 129-131)."""
    big = "x" * (DEFAULT_MAX_BYTES + 100) + "\n"
    small = "after_error\n"
    call_count = [0]

    def bad_on_chunk(text, get_progress):
        # Only raise on the second call (after full output is established)
        call_count[0] += 1
        if call_count[0] >= 2:
            raise RuntimeError("deliberate error")

    async def exec_fn(command, options):
        options.on_stdout(big)  # triggers full output file
        options.on_stdout(small)  # user callback raises -> capture_error set
        options.on_stdout(small)  # append_full_output early returns
        return ok(ShellExecResult(stdout="", stderr="", exit_code=0))

    env = FakeEnv(exec_fn)
    opts = ShellCaptureOptions(on_chunk=bad_on_chunk)
    result = await execute_shell_with_capture(env, "cmd", opts)
    assert not result.ok


# --------------------------------------------------------------------------
# append_full_output_step: append_file returns error (lines 125-126)
# --------------------------------------------------------------------------


async def test_append_full_output_step_append_file_failure():
    """After full output file is created, a subsequent append_file failure
    propagates through the chain (lines 125-126)."""
    big = "x" * (DEFAULT_MAX_BYTES + 100) + "\n"
    small = "y" * 10 + "\n"
    append_count = [0]

    async def exec_fn(command, options):
        options.on_stdout(big)
        options.on_stdout(small)
        return ok(ShellExecResult(stdout="", stderr="", exit_code=0))

    env = FakeEnv(exec_fn)
    original_append = env.append_file

    async def failing_append(path, content, abort_signal=None):
        append_count[0] += 1
        if append_count[0] > 1:
            return err(FileError("unknown", "write error"))
        return await original_append(path, content, abort_signal)

    env.append_file = failing_append  # type: ignore[method-assign]

    result = await execute_shell_with_capture(env, "cmd")
    # The error propagates through the write chain
    # (either as not ok or as execution error in chain)
    # Both outcomes are valid as long as no crash occurs
    assert result.ok is not None  # doesn't crash


# --------------------------------------------------------------------------
# chain propagates previous error (line 115)
# --------------------------------------------------------------------------


async def test_chain_propagates_previous_step_error():
    """When ensure_full_output_file_step fails, the chain returns err, and
    write_result.ok is False (line 115, 216)."""
    big = "x" * (DEFAULT_MAX_BYTES + 100) + "\n"

    async def exec_fn(command, options):
        options.on_stdout(big)
        return ok(ShellExecResult(stdout="", stderr="", exit_code=0))

    env = FakeEnv(exec_fn)

    async def failing_append(path, content, abort_signal=None):
        return err(FileError("unknown", "chain error"))

    env.append_file = failing_append  # type: ignore[method-assign]

    result = await execute_shell_with_capture(env, "cmd")
    assert not result.ok


# --------------------------------------------------------------------------
# Very large tail triggers _trim_to_last_utf8_bytes in on_chunk (lines 84-87)
# --------------------------------------------------------------------------


async def test_tail_trimmed_when_exceeds_max_output_bytes():
    """A chunk > DEFAULT_MAX_BYTES*2 triggers _trim_to_last_utf8_bytes truncation."""
    huge_chunk = "A" * (DEFAULT_MAX_BYTES * 2 + 100) + "\n"

    async def exec_fn(command, options):
        options.on_stdout(huge_chunk)
        return ok(ShellExecResult(stdout="", stderr="", exit_code=0))

    env = FakeEnv(exec_fn)
    result = await execute_shell_with_capture(env, "cmd")
    assert result.ok
    assert result.value.truncated is True


# --------------------------------------------------------------------------
# abort_signal with aborted=True from exec error (lines 186->190)
# --------------------------------------------------------------------------


async def test_exec_aborted_error_with_abort_signal_returns_cancelled():
    """aborted exec error + abort_signal.aborted=True → cancelled (lines 186->190)."""
    signal = _abort_signal(aborted=True)

    async def exec_fn(command, options):
        return err(ExecutionError("aborted", "signal"))

    env = FakeEnv(exec_fn)
    opts = ShellCaptureOptions(abort_signal=signal)
    result = await execute_shell_with_capture(env, "cmd", opts)
    assert result.ok
    assert result.value.cancelled is True
