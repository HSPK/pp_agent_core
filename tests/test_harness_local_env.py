"""Python port of `packages/agent/test/harness/nodejs-env.test.ts`.

The TypeScript suite drives `NodeExecutionEnv` (`src/harness/env/nodejs.ts`);
this drives its Python equivalent `pi_agent.harness.env.LocalExecutionEnv`.

Assertions that do not apply to the Python port and why:

- "uses stdin command transport for legacy WSL bash paths": upstream detects
  `C:\\Windows\\System32\\bash.exe` and feeds the command over stdin with
  `bash -s`. That branch is Windows/WSL-only and the TypeScript test only
  reaches it by monkey-patching `process.platform` to `"win32"`, which has no
  Python analogue (`sys.platform` is not consulted by `LocalExecutionEnv`,
  which is POSIX-only by design). Not ported.
- "settles after the shell exits when a detached descendant retains inherited
  stdio": upstream marks it `it.skipIf(process.platform !== "win32")`, so it
  never runs on this platform either. The idle-drain behavior it guards is
  still ported (`LocalExecutionEnv._await_readers_after_exit`).

`createTempDir()` from the TypeScript helpers is replaced by pytest's
`tmp_path`. Tests that exercise `create_temp_dir`/`create_temp_file` redirect
`tempfile.tempdir` at `tmp_path` so nothing escapes the test sandbox.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

import pytest
from pi_agent.harness.env import LocalExecutionEnv
from pi_agent.harness.types import FileError, ShellExecOptions, get_or_throw
from pi_agent.harness.utils.shell_output import execute_shell_with_capture
from pi_ai.utils.abort import AbortSignal


async def test_reads_writes_lists_and_removes_files_and_directories(tmp_path: Path) -> None:
    root = str(tmp_path)
    env = LocalExecutionEnv(cwd=root)
    assert get_or_throw(await env.absolute_path("nested/child")) == os.path.join(root, "nested/child")
    assert get_or_throw(await env.join_path([root, "nested", "child"])) == os.path.join(root, "nested", "child")
    get_or_throw(await env.create_dir("nested/child"))
    get_or_throw(await env.write_file("nested/child/file.txt", "hel"))
    get_or_throw(await env.append_file("nested/child/file.txt", "lo"))
    assert get_or_throw(await env.read_text_file("nested/child/file.txt")) == "hello"
    assert get_or_throw(await env.read_text_lines("nested/child/file.txt", max_lines=1)) == ["hello"]
    assert get_or_throw(await env.read_binary_file("nested/child/file.txt")).decode("utf-8") == "hello"

    entries = get_or_throw(await env.list_dir("nested/child"))
    assert len(entries) == 1
    assert entries[0].name == "file.txt"
    assert entries[0].path == os.path.join(root, "nested/child/file.txt")
    assert entries[0].kind == "file"
    assert entries[0].size == 5
    assert isinstance(entries[0].mtime_ms, int)

    assert get_or_throw(await env.exists("nested/child/file.txt")) is True
    get_or_throw(await env.remove("nested/child/file.txt"))
    assert get_or_throw(await env.exists("nested/child/file.txt")) is False


async def test_expands_home_relative_paths_and_file_urls(tmp_path: Path) -> None:
    root = str(tmp_path)
    env = LocalExecutionEnv(cwd=root)
    assert get_or_throw(await env.absolute_path("~/pi-node-env-test")) == os.path.join(
        os.path.expanduser("~"), "pi-node-env-test"
    )
    file_path = os.path.join(root, "file with spaces.txt")
    assert get_or_throw(await env.absolute_path(Path(file_path).as_uri())) == file_path


async def test_returns_file_info_without_following_symlinks(tmp_path: Path) -> None:
    root = str(tmp_path)
    env = LocalExecutionEnv(cwd=root)
    get_or_throw(await env.create_dir("dir", recursive=True))
    get_or_throw(await env.write_file("dir/file.txt", "hello"))
    os.symlink(os.path.join(root, "dir/file.txt"), os.path.join(root, "file-link"))
    os.symlink(os.path.join(root, "dir"), os.path.join(root, "dir-link"))

    dir_info = get_or_throw(await env.file_info("dir"))
    assert (dir_info.name, dir_info.path, dir_info.kind) == ("dir", os.path.join(root, "dir"), "directory")

    file_info = get_or_throw(await env.file_info("dir/file.txt"))
    assert (file_info.name, file_info.path, file_info.kind, file_info.size) == (
        "file.txt",
        os.path.join(root, "dir/file.txt"),
        "file",
        5,
    )

    file_link_info = get_or_throw(await env.file_info("file-link"))
    assert (file_link_info.name, file_link_info.path, file_link_info.kind) == (
        "file-link",
        os.path.join(root, "file-link"),
        "symlink",
    )

    dir_link_info = get_or_throw(await env.file_info("dir-link"))
    assert (dir_link_info.name, dir_link_info.path, dir_link_info.kind) == (
        "dir-link",
        os.path.join(root, "dir-link"),
        "symlink",
    )

    assert get_or_throw(await env.canonical_path("file-link")) == os.path.realpath(os.path.join(root, "dir/file.txt"))


async def test_lists_symlinks_as_symlinks(tmp_path: Path) -> None:
    root = str(tmp_path)
    env = LocalExecutionEnv(cwd=root)
    get_or_throw(await env.write_file("target.txt", "hello"))
    os.symlink(os.path.join(root, "target.txt"), os.path.join(root, "link.txt"))

    entries = get_or_throw(await env.list_dir("."))
    assert sorted((entry.name, entry.kind) for entry in entries) == [
        ("link.txt", "symlink"),
        ("target.txt", "file"),
    ]


async def test_stops_reading_text_lines_at_the_requested_limit(tmp_path: Path) -> None:
    env = LocalExecutionEnv(cwd=str(tmp_path))
    get_or_throw(await env.write_file("file.txt", "one\ntwo\nthree"))
    assert get_or_throw(await env.read_text_lines("file.txt", max_lines=1)) == ["one"]


async def test_returns_file_error_for_missing_paths(tmp_path: Path) -> None:
    root = str(tmp_path)
    env = LocalExecutionEnv(cwd=root)
    info = await env.file_info("missing.txt")
    assert info.ok is False
    assert isinstance(info.error, FileError)
    assert info.error.code == "not_found"
    assert info.error.path == os.path.join(root, "missing.txt")
    assert get_or_throw(await env.exists("missing.txt")) is False


async def test_returns_file_error_for_listing_non_directories(tmp_path: Path) -> None:
    env = LocalExecutionEnv(cwd=str(tmp_path))
    get_or_throw(await env.write_file("file.txt", "hello"))
    result = await env.list_dir("file.txt")
    assert result.ok is False
    assert isinstance(result.error, FileError)
    assert result.error.code == "not_directory"


async def test_appends_to_new_files_and_creates_parent_directories(tmp_path: Path) -> None:
    env = LocalExecutionEnv(cwd=str(tmp_path))
    get_or_throw(await env.append_file("new/nested/file.txt", "a"))
    get_or_throw(await env.append_file("new/nested/file.txt", "b"))
    assert get_or_throw(await env.read_text_file("new/nested/file.txt")) == "ab"


async def test_atomically_renames_a_file_and_replaces_the_destination(tmp_path: Path) -> None:
    env = LocalExecutionEnv(cwd=str(tmp_path))
    get_or_throw(await env.write_file("source.txt", "new"))
    get_or_throw(await env.write_file("destination.txt", "old"))

    get_or_throw(await env.rename_file("source.txt", "destination.txt"))

    assert get_or_throw(await env.exists("source.txt")) is False
    assert get_or_throw(await env.read_text_file("destination.txt")) == "new"


async def test_reports_the_source_path_when_rename_fails(tmp_path: Path) -> None:
    root = str(tmp_path)
    env = LocalExecutionEnv(cwd=root)
    get_or_throw(await env.write_file("destination.txt", "unchanged"))

    result = await env.rename_file("missing-source.txt", "destination.txt")

    assert result.ok is False
    assert result.error.code == "not_found"
    assert result.error.path == os.path.join(root, "missing-source.txt")
    assert get_or_throw(await env.read_text_file("destination.txt")) == "unchanged"


async def test_creates_temporary_directories_and_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir()
    env = LocalExecutionEnv(cwd=str(tmp_path))
    temp_dir = get_or_throw(await env.create_temp_dir("node-env-test-"))
    assert os.path.exists(temp_dir)
    temp_file = get_or_throw(await env.create_temp_file(prefix="prefix-", suffix=".txt"))
    assert os.path.exists(temp_file)
    assert temp_file.endswith(".txt")


async def test_honors_create_dir_recursive_false_and_remove_options(tmp_path: Path) -> None:
    env = LocalExecutionEnv(cwd=str(tmp_path))
    create_result = await env.create_dir("missing/child", recursive=False)
    assert create_result.ok is False
    assert create_result.error.code == "not_found"

    get_or_throw(await env.write_file("dir/child/file.txt", "hello"))
    remove_directory = await env.remove("dir", recursive=False)
    assert remove_directory.ok is False
    get_or_throw(await env.remove("dir", recursive=True))
    assert get_or_throw(await env.exists("dir")) is False

    remove_missing = await env.remove("missing", force=False)
    assert remove_missing.ok is False
    get_or_throw(await env.remove("missing", force=True))


async def test_returns_aborted_results_for_pre_aborted_file_operations(tmp_path: Path) -> None:
    env = LocalExecutionEnv(cwd=str(tmp_path))
    get_or_throw(await env.write_file("file.txt", "hello"))
    signal = AbortSignal()
    signal.abort()

    results = await asyncio.gather(
        env.read_text_file("file.txt", signal),
        env.read_text_lines("file.txt", abort_signal=signal),
        env.read_binary_file("file.txt", signal),
        env.write_file("other.txt", "hello", signal),
        env.rename_file("file.txt", "renamed.txt", signal),
        env.list_dir(".", signal),
    )
    for result in results:
        assert result.ok is False
        assert result.error.code == "aborted"


async def test_cleanup_is_best_effort(tmp_path: Path) -> None:
    env = LocalExecutionEnv(cwd=str(tmp_path))
    assert await env.cleanup() is None


async def test_executes_commands_in_cwd_with_env_overrides(tmp_path: Path) -> None:
    root = str(tmp_path)
    env = LocalExecutionEnv(cwd=root)
    result = get_or_throw(
        await env.exec(
            'printf \'%s:%s\' "$PWD" "$NODE_ENV_TEST"',
            ShellExecOptions(env={"NODE_ENV_TEST": "ok"}),
        )
    )
    assert (result.stdout, result.stderr, result.exit_code) == (f"{os.path.realpath(root)}:ok", "", 0)


@pytest.mark.parametrize(
    ("description", "overrides", "expected_session_file"),
    [
        ("a missing override preserves the base value", None, "x:/stale/parent.jsonl"),
        ("an empty override shadows the base value", {"PI_SESSION_FILE": ""}, "x:"),
        (
            "a string override replaces the base value",
            {"PI_SESSION_FILE": "/sessions/current.jsonl"},
            "x:/sessions/current.jsonl",
        ),
    ],
)
async def test_applies_string_shell_environment_overrides(
    tmp_path: Path,
    description: str,
    overrides: dict[str, str] | None,
    expected_session_file: str,
) -> None:
    env = LocalExecutionEnv(
        cwd=str(tmp_path),
        shell_env={
            "PI_SESSION_FILE": "/stale/parent.jsonl",
            "PI_CODING_AGENT": "true",
            "PI_NODE_ENV_PRESERVED_TEST": "preserved",
        },
    )
    result = get_or_throw(
        await env.exec(
            'printf \'%s:%s|%s|%s\' "${PI_SESSION_FILE+x}" "${PI_SESSION_FILE-}" '
            '"$PI_CODING_AGENT" "$PI_NODE_ENV_PRESERVED_TEST"',
            ShellExecOptions(env=overrides),
        )
    )
    assert result.stdout == f"{expected_session_file}|true|preserved"


async def test_can_replace_rather_than_inherit_the_default_shell_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inherited_key = "PI_NODE_ENV_INHERITED_TEST"
    configured_key = "PI_NODE_ENV_CONFIGURED_TEST"
    explicit_key = "PI_NODE_ENV_EXPLICIT_TEST"
    monkeypatch.setenv(inherited_key, "host")
    env = LocalExecutionEnv(cwd=str(tmp_path), shell_env={configured_key: "configured"})
    result = get_or_throw(
        await env.exec(
            f'printf \'%s:%s:%s\' "${{{inherited_key}-}}" "${{{configured_key}-}}" "${{{explicit_key}-}}"',
            ShellExecOptions(inherit_env=False, env={explicit_key: "explicit"}),
        )
    )
    assert result.stdout == "::explicit"


async def test_cleanup_terminates_active_shell_processes(tmp_path: Path) -> None:
    env = LocalExecutionEnv(cwd=str(tmp_path))
    execution = asyncio.ensure_future(env.exec("touch started; sleep 60"))
    for _attempt in range(100):
        if get_or_throw(await env.exists("started")):
            break
        await asyncio.sleep(0.01)
    assert get_or_throw(await env.exists("started")) is True
    await env.cleanup()
    result = await asyncio.wait_for(execution, timeout=3)
    assert result.ok is True


async def test_streams_stdout_and_stderr_chunks(tmp_path: Path) -> None:
    env = LocalExecutionEnv(cwd=str(tmp_path))
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    result = get_or_throw(
        await env.exec(
            "printf out; printf err >&2",
            ShellExecOptions(on_stdout=stdout_chunks.append, on_stderr=stderr_chunks.append),
        )
    )
    assert (result.stdout, result.stderr, result.exit_code) == ("out", "err", 0)
    assert "".join(stdout_chunks) == "out"
    assert "".join(stderr_chunks) == "err"


async def test_reports_a_missing_working_directory_before_spawning(tmp_path: Path) -> None:
    env = LocalExecutionEnv(cwd=os.path.join(str(tmp_path), "missing"))
    result = await env.exec("printf ok")

    assert result.ok is False
    assert result.error.code == "spawn_error"
    assert "Working directory does not exist" in str(result.error)


async def test_returns_non_zero_exit_codes_as_successful_execution_results(tmp_path: Path) -> None:
    env = LocalExecutionEnv(cwd=str(tmp_path))
    result = get_or_throw(await env.exec("exit 7"))
    assert (result.stdout, result.stderr, result.exit_code) == ("", "", 7)


async def test_returns_timeout_errors_for_commands_exceeding_the_timeout(tmp_path: Path) -> None:
    env = LocalExecutionEnv(cwd=str(tmp_path))
    result = await env.exec("sleep 5", ShellExecOptions(timeout=0.01))
    assert result.ok is False
    assert result.error.code == "timeout"


async def test_returns_callback_errors_from_exec_stream_handlers(tmp_path: Path) -> None:
    env = LocalExecutionEnv(cwd=str(tmp_path))

    def fail(_chunk: str) -> None:
        raise RuntimeError("callback failed")

    result = await env.exec("printf out", ShellExecOptions(on_stdout=fail))
    assert result.ok is False
    assert result.error.code == "callback_error"
    assert str(result.error) == "callback failed"


async def test_returns_shell_unavailable_and_spawn_errors(tmp_path: Path) -> None:
    root = str(tmp_path)
    missing_shell_env = LocalExecutionEnv(cwd=root, shell_path=os.path.join(root, "missing-shell"))
    missing_shell = await missing_shell_env.exec("printf ok")
    assert missing_shell.ok is False
    assert missing_shell.error.code == "shell_unavailable"

    shell_path = os.path.join(root, "not-executable-shell")
    env = LocalExecutionEnv(cwd=root)
    get_or_throw(await env.write_file(shell_path, "not executable"))
    spawn_error_env = LocalExecutionEnv(cwd=root, shell_path=shell_path)
    spawn_error = await spawn_error_env.exec("printf ok")
    assert spawn_error.ok is False
    assert spawn_error.error.code == "spawn_error"


async def test_returns_an_aborted_result_for_aborted_commands(tmp_path: Path) -> None:
    env = LocalExecutionEnv(cwd=str(tmp_path))
    signal = AbortSignal()
    execution = asyncio.ensure_future(env.exec("sleep 5", ShellExecOptions(abort_signal=signal)))
    await asyncio.sleep(0)
    signal.abort()
    result = await execution
    assert result.ok is False
    assert result.error.code == "aborted"


async def test_captures_large_shell_output_to_a_full_output_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir()
    env = LocalExecutionEnv(cwd=str(tmp_path))
    result = get_or_throw(await execute_shell_with_capture(env, "yes line | head -n 15000"))
    assert result.truncated is True
    assert result.full_output_path is not None
    full_output = get_or_throw(await env.read_text_file(result.full_output_path))
    assert len(full_output.split("\n")) > 10000
    assert len(result.output) < len(full_output)
