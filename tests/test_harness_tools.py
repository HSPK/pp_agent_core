"""Python port of `packages/agent/test/harness/tools.test.ts`.

Covers the built-in harness tools (`read`, `write`, `edit`, `bash`) running
against `LocalExecutionEnv`, the Python equivalent of `NodeExecutionEnv`.

Deviations from the TypeScript original:

- The TS suite verifies the emitted unified patch with `applyPatch` from the
  `diff` npm package. Python has no stdlib patch applier, so `_apply_patch`
  below is a minimal unified-diff applier used for the same assertion.
- `createBashTool` is generic over the turn context in TypeScript; Python uses
  a plain dataclass subclass of `ExecutionToolContext` instead.
"""

from __future__ import annotations

import asyncio
import base64
import os
import re
import struct
import tempfile
from dataclasses import dataclass
from typing import Any

import pytest
from pi_ai.types import ImageContent, TextContent
from pi_ai.utils.abort import AbortSignal

from pi_agent.harness.env.local import LocalExecutionEnv
from pi_agent.harness.tools.bash import BashExecution, BashToolDetails, create_bash_tool
from pi_agent.harness.tools.edit import create_edit_tool
from pi_agent.harness.tools.read import ReadImageProcessorResult, create_read_tool
from pi_agent.harness.tools.tool_context import ExecutionToolContext
from pi_agent.harness.tools.write import create_write_tool
from pi_agent.harness.types import (
    ExecutionError,
    Result,
    ShellExecOptions,
    ShellExecResult,
    err,
    get_or_throw,
    ok,
)
from pi_agent.harness.utils.truncate import DEFAULT_MAX_LINES

TRUNCATED_OUTPUT_LINES = DEFAULT_MAX_LINES + 1

PNG_1X1 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGD4DwABBAEAX+XDSwAAAABJRU5ErkJggg=="


@pytest.fixture(autouse=True)
def _sandbox_tempdir(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    sandbox = tmp_path / "tmp"
    sandbox.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(sandbox))


def text_output(result: Any) -> str:
    return "\n".join(part.text for part in result.content if isinstance(part, TextContent))


def create_context(tmp_path: Any, name: str = "cwd") -> ExecutionToolContext:
    cwd = tmp_path / name
    cwd.mkdir(exist_ok=True)
    return ExecutionToolContext(env=LocalExecutionEnv(cwd=str(cwd)))


def create_env(tmp_path: Any, cls: type = LocalExecutionEnv, name: str = "cwd", **kwargs: Any) -> Any:
    cwd = tmp_path / name
    cwd.mkdir(exist_ok=True)
    return cls(cwd=str(cwd), **kwargs)


def _apply_patch(original: str, patch: str) -> str:
    """Apply a unified diff produced by `generate_unified_patch`."""
    lines = original.split("\n")
    output: list[str] = []
    cursor = 0
    patch_lines = patch.split("\n")
    index = 0
    while index < len(patch_lines):
        line = patch_lines[index]
        if not line.startswith("@@"):
            index += 1
            continue
        header = line.split("@@")[1].strip()
        old_range = header.split(" ")[0]
        old_start = int(old_range[1:].split(",")[0]) - 1
        output.extend(lines[cursor:old_start])
        cursor = old_start
        index += 1
        while index < len(patch_lines) and not patch_lines[index].startswith("@@"):
            hunk_line = patch_lines[index]
            index += 1
            if hunk_line.startswith("\\"):
                continue
            if hunk_line.startswith("+"):
                output.append(hunk_line[1:])
            elif hunk_line.startswith("-"):
                cursor += 1
            elif hunk_line.startswith(" ") or hunk_line == "":
                if cursor >= len(lines):
                    break
                output.append(lines[cursor])
                cursor += 1
    output.extend(lines[cursor:])
    return "\n".join(output)


def create_tiny_bmp() -> bytes:
    data = bytearray(58)
    data[0] = 0x42
    data[1] = 0x4D
    struct.pack_into("<I", data, 2, len(data))
    struct.pack_into("<I", data, 10, 54)
    struct.pack_into("<I", data, 14, 40)
    struct.pack_into("<i", data, 18, 1)
    struct.pack_into("<i", data, 22, 1)
    struct.pack_into("<H", data, 26, 1)
    struct.pack_into("<H", data, 28, 24)
    struct.pack_into("<I", data, 34, 4)
    return bytes(data)


class SlowReadExecutionEnv(LocalExecutionEnv):
    async def read_text_file(self, path: str, abort_signal: AbortSignal | None = None) -> Result[str, Any]:
        await asyncio.sleep(0.02)
        return await super().read_text_file(path, abort_signal)


class BlockingWriteExecutionEnv(LocalExecutionEnv):
    def __init__(self, cwd: str) -> None:
        super().__init__(cwd=cwd)
        self.first_write_started = asyncio.Event()
        self.finish_first_write = asyncio.Event()
        self.second_write_started = False

    async def write_file(
        self, path: str, content: str | bytes, abort_signal: AbortSignal | None = None
    ) -> Result[None, Any]:
        if content == "first\n":
            self.first_write_started.set()
            await self.finish_first_write.wait()
        elif content == "second\n":
            self.second_write_started = True
        return await super().write_file(path, content, abort_signal)


class BlockingEditExecutionEnv(LocalExecutionEnv):
    def __init__(self, cwd: str) -> None:
        super().__init__(cwd=cwd)
        self.first_edit_write_started = asyncio.Event()
        self.finish_first_edit_write = asyncio.Event()
        self.first_edit_write_settled = False
        self.second_edit_write_started = False

    async def write_file(
        self, path: str, content: str | bytes, abort_signal: AbortSignal | None = None
    ) -> Result[None, Any]:
        if content == "ALPHA\nbeta\n":
            self.first_edit_write_started.set()
            await self.finish_first_edit_write.wait()
            result = await super().write_file(path, content)
            self.first_edit_write_settled = True
            return result
        if content in ("ALPHA\nBETA\n", "alpha\nBETA\n"):
            self.second_edit_write_started = True
        return await super().write_file(path, content, abort_signal)


class LateOutputExecutionEnv(LocalExecutionEnv):
    async def exec(
        self, command: str, options: ShellExecOptions | None = None
    ) -> Result[ShellExecResult, ExecutionError]:
        if options is not None and options.on_stdout is not None:
            on_stdout = options.on_stdout
            on_stdout("before\n")
            asyncio.get_running_loop().call_later(0, on_stdout, "late\n")
        return ok(ShellExecResult(stdout="before\n", stderr="", exit_code=0))


class TimeoutOutputExecutionEnv(LocalExecutionEnv):
    async def exec(
        self, command: str, options: ShellExecOptions | None = None
    ) -> Result[ShellExecResult, ExecutionError]:
        output = "\n".join(f"line-{index + 1}" for index in range(TRUNCATED_OUTPUT_LINES)) + "\n"
        if options is not None and options.on_stdout is not None:
            options.on_stdout(output)
        return err(ExecutionError("timeout", f"timeout:{options.timeout if options else None}"))


# --- read ---------------------------------------------------------------------


async def test_read_reads_text_with_offsets_limits_and_continuation_notices(tmp_path: Any) -> None:
    context = create_context(tmp_path)
    get_or_throw(await context.env.write_file("test.txt", "\n".join(f"Line {index + 1}" for index in range(100))))

    result = await create_read_tool().execute_with_context(
        "read-1", {"path": "test.txt", "offset": 41, "limit": 20}, None, None, context
    )
    output = text_output(result)

    assert "Line 40" not in output
    assert "Line 41" in output
    assert "Line 60" in output
    assert "Line 61" not in output
    assert "[40 more lines in file. Use offset=61 to continue.]" in output


async def test_read_truncates_large_text_by_line_count(tmp_path: Any) -> None:
    context = create_context(tmp_path)
    get_or_throw(await context.env.write_file("large.txt", "\n".join(f"Line {index + 1}" for index in range(2500))))

    result = await create_read_tool().execute_with_context("read-2", {"path": "large.txt"}, None, None, context)

    assert "[Showing lines 1-2000 of 2500. Use offset=2001 to continue.]" in text_output(result)
    truncation = result.details.truncation
    assert truncation.truncated is True
    assert truncation.truncated_by == "lines"
    assert truncation.total_lines == 2500
    assert truncation.output_lines == 2000


async def test_read_does_not_count_trailing_newline_as_extra_line_at_limit(tmp_path: Any) -> None:
    context = create_context(tmp_path)
    get_or_throw(await context.env.write_file("exact.txt", "\n".join("x" for _ in range(2000)) + "\n"))

    result = await create_read_tool().execute_with_context("read-exact", {"path": "exact.txt"}, None, None, context)

    assert result.details is None
    assert "Use offset=" not in text_output(result)


async def test_read_rejects_offsets_beyond_the_file(tmp_path: Any) -> None:
    context = create_context(tmp_path)
    get_or_throw(await context.env.write_file("short.txt", "one\ntwo\nthree"))

    with pytest.raises(ValueError, match=r"Offset 100 is beyond end of file \(3 lines total\)"):
        await create_read_tool().execute_with_context(
            "read-3", {"path": "short.txt", "offset": 100}, None, None, context
        )


async def test_read_detects_supported_images_by_content(tmp_path: Any) -> None:
    context = create_context(tmp_path)
    png = base64.b64decode(PNG_1X1)
    get_or_throw(await context.env.write_file("image.txt", png))

    result = await create_read_tool().execute_with_context("read-4", {"path": "image.txt"}, None, None, context)

    assert "Read image file [image/png]" in text_output(result)
    assert ImageContent(data=PNG_1X1, mime_type="image/png") in result.content


async def test_read_delegates_image_conversion_to_injected_processor(tmp_path: Any) -> None:
    context = create_context(tmp_path)
    bmp = create_tiny_bmp()
    get_or_throw(await context.env.write_file("image.bmp", bmp))
    received: dict[str, Any] = {}

    async def image_processor(data: bytes, mime_type: str, auto_resize_images: bool) -> ReadImageProcessorResult:
        received.update({"bytes": data, "mime_type": mime_type, "auto_resize_images": auto_resize_images})
        return ReadImageProcessorResult(
            ok=True,
            data="converted",
            mime_type="image/png",
            hints=["[Image converted from image/bmp to image/png.]"],
        )

    tool = create_read_tool(auto_resize_images=False, image_processor=image_processor)
    result = await tool.execute_with_context("read-bmp", {"path": "image.bmp"}, None, None, context)

    assert received["mime_type"] == "image/bmp"
    assert received["auto_resize_images"] is False
    assert received["bytes"] == bmp
    assert "[Image converted from image/bmp to image/png.]" in text_output(result)
    assert ImageContent(data="converted", mime_type="image/png") in result.content


# --- write --------------------------------------------------------------------


async def test_write_writes_files_and_creates_parent_directories(tmp_path: Any) -> None:
    context = create_context(tmp_path)

    result = await create_write_tool().execute_with_context(
        "write-1", {"path": "nested/dir/file.txt", "content": "hello"}, None, None, context
    )

    assert text_output(result) == "Successfully wrote 5 bytes to nested/dir/file.txt"
    assert get_or_throw(await context.env.read_text_file("nested/dir/file.txt")) == "hello"


async def test_write_keeps_mutation_queue_locked_until_aborted_write_settles(tmp_path: Any) -> None:
    env = create_env(tmp_path, BlockingWriteExecutionEnv)
    tool = create_write_tool()
    signal = AbortSignal()
    first_write = asyncio.ensure_future(
        tool.execute_with_context(
            "write-first", {"path": "file.txt", "content": "first\n"}, signal, None, ExecutionToolContext(env=env)
        )
    )
    await env.first_write_started.wait()
    signal.abort()
    second_write = asyncio.ensure_future(
        tool.execute_with_context(
            "write-second", {"path": "file.txt", "content": "second\n"}, None, None, ExecutionToolContext(env=env)
        )
    )

    await asyncio.sleep(0.02)
    assert env.second_write_started is False
    env.finish_first_write.set()
    with pytest.raises(Exception):  # noqa: B017
        await first_write
    await second_write
    assert get_or_throw(await env.read_text_file("file.txt")) == "second\n"


# --- edit ---------------------------------------------------------------------


async def test_edit_applies_disjoint_edits_and_returns_both_diff_formats(tmp_path: Any) -> None:
    context = create_context(tmp_path)
    original = "alpha\nbeta\ngamma\ndelta\n"
    get_or_throw(await context.env.write_file("edit.txt", original))

    result = await create_edit_tool().execute_with_context(
        "edit-1",
        {
            "path": "edit.txt",
            "edits": [
                {"oldText": "alpha\n", "newText": "ALPHA\n"},
                {"oldText": "gamma\n", "newText": "GAMMA\n"},
            ],
        },
        None,
        None,
        context,
    )

    assert text_output(result) == "Successfully replaced 2 block(s) in edit.txt."
    assert "ALPHA" in result.details.diff
    assert "GAMMA" in result.details.diff
    assert _apply_patch(original, result.details.patch) == "ALPHA\nbeta\nGAMMA\ndelta\n"
    assert get_or_throw(await context.env.read_text_file("edit.txt")) == "ALPHA\nbeta\nGAMMA\ndelta\n"


async def test_edit_matches_all_edits_against_the_original_and_rejects_overlaps(tmp_path: Any) -> None:
    context = create_context(tmp_path)
    get_or_throw(await context.env.write_file("edit.txt", "one\ntwo\nthree\n"))

    with pytest.raises(Exception, match="overlap"):
        await create_edit_tool().execute_with_context(
            "edit-2",
            {
                "path": "edit.txt",
                "edits": [
                    {"oldText": "one\ntwo\n", "newText": "ONE\nTWO\n"},
                    {"oldText": "two\nthree\n", "newText": "TWO\nTHREE\n"},
                ],
            },
            None,
            None,
            context,
        )
    assert get_or_throw(await context.env.read_text_file("edit.txt")) == "one\ntwo\nthree\n"


async def test_edit_rejects_missing_and_duplicate_target_text(tmp_path: Any) -> None:
    context = create_context(tmp_path)
    get_or_throw(await context.env.write_file("edit.txt", "foo foo foo"))
    tool = create_edit_tool()

    with pytest.raises(Exception, match="Could not find the exact text"):
        await tool.execute_with_context(
            "edit-3", {"path": "edit.txt", "edits": [{"oldText": "bar", "newText": "baz"}]}, None, None, context
        )
    with pytest.raises(Exception, match="Found 3 occurrences"):
        await tool.execute_with_context(
            "edit-4", {"path": "edit.txt", "edits": [{"oldText": "foo", "newText": "bar"}]}, None, None, context
        )


async def test_edit_keeps_mutation_queue_locked_until_aborted_edit_write_settles(tmp_path: Any) -> None:
    env = create_env(tmp_path, BlockingEditExecutionEnv)
    get_or_throw(await env.write_file("file.txt", "alpha\nbeta\n"))
    tool = create_edit_tool()
    signal = AbortSignal()
    first_edit = asyncio.ensure_future(
        tool.execute_with_context(
            "edit-first",
            {"path": "file.txt", "edits": [{"oldText": "alpha", "newText": "ALPHA"}]},
            signal,
            None,
            ExecutionToolContext(env=env),
        )
    )
    await env.first_edit_write_started.wait()
    signal.abort()
    second_edit = asyncio.ensure_future(
        tool.execute_with_context(
            "edit-second",
            {"path": "file.txt", "edits": [{"oldText": "beta", "newText": "BETA"}]},
            None,
            None,
            ExecutionToolContext(env=env),
        )
    )

    await asyncio.sleep(0.02)
    assert env.second_edit_write_started is False
    env.finish_first_edit_write.set()
    with pytest.raises(Exception, match="Operation aborted"):
        await first_edit
    await second_edit
    assert env.first_edit_write_settled is True
    assert get_or_throw(await env.read_text_file("file.txt")) == "ALPHA\nBETA\n"


async def test_edit_serializes_concurrent_edits_through_canonical_and_symlink_paths(tmp_path: Any) -> None:
    env = create_env(tmp_path, SlowReadExecutionEnv)
    get_or_throw(await env.write_file("target.txt", "alpha\nbeta\ngamma\n"))
    os.symlink("target.txt", f"{env.cwd}/link.txt")
    tool = create_edit_tool()

    await asyncio.gather(
        tool.execute_with_context(
            "edit-target",
            {"path": "target.txt", "edits": [{"oldText": "alpha", "newText": "ALPHA"}]},
            None,
            None,
            ExecutionToolContext(env=env),
        ),
        tool.execute_with_context(
            "edit-link",
            {"path": "link.txt", "edits": [{"oldText": "beta", "newText": "BETA"}]},
            None,
            None,
            ExecutionToolContext(env=env),
        ),
    )

    assert get_or_throw(await env.read_text_file("target.txt")) == "ALPHA\nBETA\ngamma\n"


async def test_edit_edits_regular_files_through_symlinks(tmp_path: Any) -> None:
    context = create_context(tmp_path)
    get_or_throw(await context.env.write_file("target.txt", "before\n"))
    os.symlink("target.txt", f"{context.env.cwd}/link.txt")

    await create_edit_tool().execute_with_context(
        "edit-symlink",
        {"path": "link.txt", "edits": [{"oldText": "before", "newText": "after"}]},
        None,
        None,
        context,
    )

    assert get_or_throw(await context.env.read_text_file("target.txt")) == "after\n"


async def test_edit_preserves_bom_and_crlf_line_endings(tmp_path: Any) -> None:
    context = create_context(tmp_path)
    get_or_throw(await context.env.write_file("edit.txt", "\ufeffone\r\ntwo\r\n"))

    await create_edit_tool().execute_with_context(
        "edit-5", {"path": "edit.txt", "edits": [{"oldText": "two", "newText": "TWO"}]}, None, None, context
    )

    assert get_or_throw(await context.env.read_text_file("edit.txt")) == "\ufeffone\r\nTWO\r\n"


# --- bash ---------------------------------------------------------------------


async def test_bash_executes_commands_and_combines_stdout_and_stderr(tmp_path: Any) -> None:
    context = create_context(tmp_path)

    result = await create_bash_tool().execute_with_context(
        "bash-1", {"command": "printf out; printf err >&2"}, None, None, context
    )

    assert "out" in text_output(result)
    assert "err" in text_output(result)


async def test_bash_reports_nonzero_exits_and_timeouts(tmp_path: Any) -> None:
    context = create_context(tmp_path)
    tool = create_bash_tool()

    with pytest.raises(Exception, match=r"failed[\s\S]*Command exited with code 7"):
        await tool.execute_with_context("bash-2", {"command": "printf failed; exit 7"}, None, None, context)
    with pytest.raises(Exception, match=r"Command timed out after 0.01 seconds"):
        await tool.execute_with_context("bash-3", {"command": "sleep 2", "timeout": 0.01}, None, None, context)


async def test_bash_preserves_truncated_output_when_a_command_times_out(tmp_path: Any) -> None:
    context = ExecutionToolContext(env=create_env(tmp_path, TimeoutOutputExecutionEnv))
    error: BaseException | None = None
    try:
        await create_bash_tool().execute_with_context(
            "bash-timeout-output", {"command": "emit-output-then-time-out", "timeout": 0.05}, None, None, context
        )
    except BaseException as cause:
        error = cause

    assert isinstance(error, Exception)
    message = str(error)
    assert "Command timed out after 0.05 seconds" in message
    match = re.search(r"Full output: ([^\]\n]+)", message)
    assert match is not None
    full_output = get_or_throw(await context.env.read_text_file(match.group(1)))
    assert "line-1\nline-2" in full_output
    assert f"line-{DEFAULT_MAX_LINES}\nline-{TRUNCATED_OUTPUT_LINES}" in full_output


async def test_bash_ignores_output_callbacks_after_execution_settles(tmp_path: Any) -> None:
    env = create_env(tmp_path, LateOutputExecutionEnv)
    updates: list[str] = []

    result = await create_bash_tool().execute_with_context(
        "bash-late",
        {"command": "late"},
        None,
        lambda update: updates.append(text_output(update)),
        ExecutionToolContext(env=env),
    )
    await asyncio.sleep(0.02)

    assert text_output(result) == "before\n"
    assert not any("late" in update for update in updates)


async def test_bash_reports_the_total_size_of_an_oversized_final_line(tmp_path: Any) -> None:
    context = create_context(tmp_path)

    result = await create_bash_tool().execute_with_context(
        "bash-long-line", {"command": "printf '%060000d' 0"}, None, None, context
    )

    assert re.search(r"Showing last 50\.0KB of line 1 \(line is 58\.6KB\)\. Full output:", text_output(result))


@dataclass
class _PrepareContext(ExecutionToolContext):
    workspace: str = ""


async def test_bash_prepares_command_cwd_and_explicit_env_with_the_turn_context(tmp_path: Any) -> None:
    env = create_env(tmp_path, shell_env={"PI_BASH_PREPARE_INHERITED": "inherited"})
    get_or_throw(await env.create_dir("workspace"))
    context = _PrepareContext(env=env, workspace=f"{env.cwd}/workspace")
    signal = AbortSignal()
    received: dict[str, Any] = {}

    async def prepare(
        execution: BashExecution, turn_context: ExecutionToolContext, prepare_signal: AbortSignal | None
    ) -> None:
        received.update({"context": turn_context, "signal": prepare_signal})
        assert isinstance(turn_context, _PrepareContext)
        execution.cwd = turn_context.workspace
        execution.env = {"PI_BASH_PREPARE_EXPLICIT": "explicit"}
        execution.inherit_env = False
        execution.command += (
            '\nprintf \'%s:%s:%s:%s\' "$prefix" "${PI_BASH_PREPARE_INHERITED-}" "$PI_BASH_PREPARE_EXPLICIT" "$PWD"'
        )

    tool = create_bash_tool(command_prefix="prefix=ready", prepare=prepare)
    result = await tool.execute_with_context("bash-prepare", {"command": ":"}, signal, None, context)

    assert received["context"] is context
    assert received["signal"] is signal
    canonical = get_or_throw(await env.canonical_path(context.workspace))
    assert text_output(result) == f"ready::explicit:{canonical}"


async def test_bash_supports_command_prefixes(tmp_path: Any) -> None:
    context = create_context(tmp_path)

    result = await create_bash_tool(command_prefix="value=hello").execute_with_context(
        "bash-4", {"command": "printf $value"}, None, None, context
    )

    assert text_output(result) == "hello"


async def test_bash_coalesces_updates_and_persists_truncated_full_output(tmp_path: Any) -> None:
    context = create_context(tmp_path)
    updates: list[Any] = []

    result = await create_bash_tool().execute_with_context(
        "bash-5",
        {"command": "i=1; while [ $i -le 3000 ]; do echo line-$i; i=$((i + 1)); done"},
        None,
        updates.append,
        context,
    )

    assert len(updates) < 25
    truncation = result.details.truncation
    assert truncation.truncated is True
    assert truncation.truncated_by == "lines"
    assert truncation.total_lines == 3000
    assert truncation.output_lines == 2000
    assert "line-3000" in text_output(result)
    assert result.details.full_output_path is not None
    final_update = updates[-1]
    assert "line-3000" in text_output(final_update)
    assert isinstance(final_update.details, BashToolDetails)
    assert final_update.details.truncation.total_lines == 3000
    assert isinstance(final_update.details.truncation.total_bytes, int)
    assert final_update.details.full_output_path == result.details.full_output_path
    full_output = get_or_throw(await context.env.read_text_file(result.details.full_output_path))
    assert "line-1\nline-2" in full_output
    assert "line-2999\nline-3000" in full_output
