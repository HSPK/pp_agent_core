"""Local filesystem and process `ExecutionEnv` implementation.

Python port of `packages/agent/src/harness/env/nodejs.ts`. The TypeScript
class is named `NodeExecutionEnv` because it is the Node-runtime backend of
the `ExecutionEnv` interface; the Python equivalent is `LocalExecutionEnv`,
built on `pathlib`/`os`/`asyncio.subprocess`.

Windows-only behavior is not ported: this port always uses POSIX process
groups (`os.setsid` + `os.killpg`) instead of `taskkill`, never searches for
Git Bash under `%ProgramFiles%`, and drops the legacy WSL
`C:\\Windows\\System32\\bash.exe` stdin command transport (upstream feeds the
command over stdin with `bash -s` for that one path because argv-passed
commands are mangled by the WSL launcher). Shell resolution therefore
reduces to: an explicit `shell_path` if it exists, else `/bin/bash`, else
`bash` from `PATH`, else `sh`.
"""

from __future__ import annotations

import asyncio
import contextlib
import errno
import os
import shutil
import signal
import stat
import tempfile
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar
from urllib.parse import unquote, urlparse

from pi_ai.utils.abort import AbortSignal

from ..types import (
    ExecutionError,
    FileError,
    FileInfo,
    FileKind,
    Result,
    ShellExecOptions,
    ShellExecResult,
    err,
    ok,
    to_error,
)

MAX_TIMEOUT_MS = 2_147_483_647
MAX_TIMEOUT_SECONDS = MAX_TIMEOUT_MS / 1000
EXIT_STDIO_GRACE_SECONDS = 0.1

TValue = TypeVar("TValue")


def _resolve_timeout_seconds(timeout: float | None) -> Result[float | None, ExecutionError]:
    if timeout is None:
        return ok(None)
    if timeout != timeout or timeout in (float("inf"), float("-inf")) or timeout <= 0:
        return err(ExecutionError("timeout", "Invalid timeout: must be a finite number of seconds"))
    if timeout * 1000 > MAX_TIMEOUT_MS:
        return err(ExecutionError("timeout", f"Invalid timeout: maximum is {MAX_TIMEOUT_SECONDS} seconds"))
    return ok(timeout)


def _file_url_to_path(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme != "file":
        raise ValueError(f"Not a file URL: {url}")
    if parsed.netloc not in ("", "localhost"):
        raise ValueError(f"File URL host must be 'localhost' or empty: {url}")
    return unquote(parsed.path)


def resolve_path(cwd: str, path: str) -> str:
    """Expand `~` and `file://` prefixes, then normalize against `cwd`."""
    normalized = path
    if normalized == "~":
        normalized = os.path.expanduser("~")
    elif normalized.startswith("~/"):
        normalized = os.path.join(os.path.expanduser("~"), normalized[2:])
    elif normalized.startswith("file://"):
        # Keep malformed URLs as ordinary paths so filesystem methods preserve their non-throwing contract.
        with contextlib.suppress(Exception):
            normalized = _file_url_to_path(normalized)
    if os.path.isabs(normalized):
        return os.path.normpath(normalized)
    return os.path.normpath(os.path.join(cwd, normalized))


def _file_kind_from_stats(stats: os.stat_result) -> FileKind | None:
    if stat.S_ISLNK(stats.st_mode):
        return "symlink"
    if stat.S_ISREG(stats.st_mode):
        return "file"
    if stat.S_ISDIR(stats.st_mode):
        return "directory"
    return None


def _file_info_from_stats(path: str, stats: os.stat_result) -> Result[FileInfo, FileError]:
    kind = _file_kind_from_stats(stats)
    if kind is None:
        return err(FileError("invalid", "Unsupported file type", path))
    return ok(
        FileInfo(
            name=os.path.basename(path),
            path=path,
            kind=kind,
            size=stats.st_size,
            mtime_ms=int(stats.st_mtime * 1000),
        )
    )


_ERRNO_FILE_ERROR_CODES: dict[int, str] = {
    errno.ENOENT: "not_found",
    errno.EACCES: "permission_denied",
    errno.EPERM: "permission_denied",
    errno.ENOTDIR: "not_directory",
    errno.EISDIR: "is_directory",
    errno.ENOTEMPTY: "invalid",
    errno.EINVAL: "invalid",
}


def to_file_error(error: BaseException, fallback_path: str | None = None) -> FileError:
    """Map an OS error onto the backend-independent `FileError` codes."""
    if isinstance(error, FileError):
        return error
    cause = to_error(error)
    path = fallback_path
    if isinstance(error, OSError):
        if isinstance(error.filename, str):
            path = error.filename
        code = _ERRNO_FILE_ERROR_CODES.get(error.errno or 0)
        if code is not None:
            return FileError(code, str(error), path, cause)  # type: ignore[arg-type]
    return FileError("unknown", str(cause), path, cause)


def _abort_result(signal_: AbortSignal | None, path: str | None = None) -> Result[Any, FileError] | None:
    if signal_ is not None and signal_.aborted:
        return err(FileError("aborted", "aborted", path))
    return None


def _path_exists(path: str) -> bool:
    return os.path.exists(path)


@dataclass(frozen=True)
class ShellConfig:
    shell: str
    args: tuple[str, ...]


def get_shell_config(custom_shell_path: str | None = None) -> Result[ShellConfig, ExecutionError]:
    """Resolve the bash-compatible shell used to run commands."""
    if custom_shell_path:
        if _path_exists(custom_shell_path):
            return ok(ShellConfig(custom_shell_path, ("-c",)))
        return err(ExecutionError("shell_unavailable", f"Custom shell path not found: {custom_shell_path}"))
    if _path_exists("/bin/bash"):
        return ok(ShellConfig("/bin/bash", ("-c",)))
    bash_on_path = shutil.which("bash")
    if bash_on_path:
        return ok(ShellConfig(bash_on_path, ("-c",)))
    return ok(ShellConfig("sh", ("-c",)))


def get_shell_env(
    base_env: dict[str, str] | None = None,
    extra_env: dict[str, str] | None = None,
    inherit_env: bool = True,
) -> dict[str, str]:
    if not inherit_env:
        return dict(extra_env or {})
    merged = dict(os.environ)
    merged.update(base_env or {})
    merged.update(extra_env or {})
    return merged


def kill_process_tree(pid: int) -> None:
    """Kill the whole process group, falling back to the single process."""
    try:
        os.killpg(pid, signal.SIGKILL)
    except OSError:
        # Process already dead.
        with contextlib.suppress(OSError):
            os.kill(pid, signal.SIGKILL)


class LocalExecutionEnv:
    """`ExecutionEnv` backed by the local filesystem and local processes."""

    def __init__(
        self,
        cwd: str,
        shell_path: str | None = None,
        shell_env: dict[str, str] | None = None,
    ) -> None:
        self.cwd = cwd
        self._shell_path = shell_path
        self._shell_env = shell_env
        self._active_child_pids: set[int] = set()

    async def absolute_path(self, path: str, abort_signal: AbortSignal | None = None) -> Result[str, FileError]:
        return ok(resolve_path(self.cwd, path))

    async def join_path(self, parts: list[str], abort_signal: AbortSignal | None = None) -> Result[str, FileError]:
        return ok(os.path.join(*parts))

    async def exec(
        self, command: str, options: ShellExecOptions | None = None
    ) -> Result[ShellExecResult, ExecutionError]:
        if options is not None and options.abort_signal is not None and options.abort_signal.aborted:
            return err(ExecutionError("aborted", "aborted"))
        timeout_result = _resolve_timeout_seconds(options.timeout if options else None)
        if not timeout_result.ok:
            return err(timeout_result.error)
        timeout_seconds = timeout_result.value

        cwd = resolve_path(self.cwd, options.cwd) if options is not None and options.cwd else self.cwd
        shell_config = get_shell_config(self._shell_path)
        if not shell_config.ok:
            return err(shell_config.error)
        if not os.path.exists(cwd):
            return err(
                ExecutionError(
                    "spawn_error",
                    f"Working directory does not exist: {cwd}\nCannot execute bash commands.",
                )
            )

        return await self._run_command(command, options, shell_config.value, cwd, timeout_seconds)

    async def _run_command(
        self,
        command: str,
        options: ShellExecOptions | None,
        shell_config: ShellConfig,
        cwd: str,
        timeout_seconds: float | None,
    ) -> Result[ShellExecResult, ExecutionError]:
        try:
            process = await asyncio.create_subprocess_exec(
                shell_config.shell,
                *shell_config.args,
                command,
                cwd=cwd,
                env=get_shell_env(
                    self._shell_env,
                    options.env if options else None,
                    options.inherit_env if options else True,
                ),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        except Exception as error:
            cause = to_error(error)
            return err(ExecutionError("spawn_error", str(cause), cause))

        pid = process.pid
        self._active_child_pids.add(pid)

        stdout = ""
        stderr = ""
        callback_error: ExecutionError | None = None
        timed_out = False
        last_data_at = time.monotonic()

        def kill() -> None:
            kill_process_tree(pid)

        async def pump(stream: asyncio.StreamReader | None, on_chunk: Callable[[str], None] | None, sink: str) -> None:
            nonlocal stdout, stderr, callback_error, last_data_at
            if stream is None:
                return
            while True:
                chunk = await stream.read(8192)
                if not chunk:
                    return
                text = chunk.decode("utf-8", errors="replace")
                last_data_at = time.monotonic()
                if sink == "stdout":
                    stdout += text
                else:
                    stderr += text
                if on_chunk is None:
                    continue
                try:
                    on_chunk(text)
                except Exception as error:
                    cause = to_error(error)
                    callback_error = ExecutionError("callback_error", str(cause), cause)
                    kill()

        readers = [
            asyncio.ensure_future(pump(process.stdout, options.on_stdout if options else None, "stdout")),
            asyncio.ensure_future(pump(process.stderr, options.on_stderr if options else None, "stderr")),
        ]

        watchers: list[asyncio.Task[None]] = []

        async def watch_timeout() -> None:
            nonlocal timed_out
            assert timeout_seconds is not None
            await asyncio.sleep(timeout_seconds)
            timed_out = True
            kill()

        async def watch_abort() -> None:
            assert options is not None and options.abort_signal is not None
            await options.abort_signal.wait()
            kill()

        if timeout_seconds is not None:
            watchers.append(asyncio.ensure_future(watch_timeout()))
        if options is not None and options.abort_signal is not None:
            watchers.append(asyncio.ensure_future(watch_abort()))

        try:
            exit_code = await process.wait()
            await self._await_readers_after_exit(readers, lambda: last_data_at)
        finally:
            for reader in readers:
                reader.cancel()
            for watcher in watchers:
                watcher.cancel()
            self._active_child_pids.discard(pid)

        if callback_error is not None:
            return err(callback_error)
        if timed_out:
            return err(ExecutionError("timeout", f"timeout:{options.timeout if options else None}"))
        if options is not None and options.abort_signal is not None and options.abort_signal.aborted:
            return err(ExecutionError("aborted", "aborted"))
        return ok(ShellExecResult(stdout=stdout, stderr=stderr, exit_code=exit_code if exit_code is not None else 0))

    @staticmethod
    async def _await_readers_after_exit(
        readers: Sequence[asyncio.Future[None]], last_data_at: Callable[[], float]
    ) -> None:
        """Drain pipes after exit, giving up once they go idle.

        A detached grandchild can inherit stdout/stderr and hold the pipes open
        long after the shell exits. Upstream arms a 100ms idle timer that is
        rearmed on every late chunk; this reproduces it.
        """
        pending = {reader for reader in readers if not reader.done()}
        while pending:
            remaining = EXIT_STDIO_GRACE_SECONDS - (time.monotonic() - last_data_at())
            if remaining <= 0:
                return
            _done, pending = await asyncio.wait(pending, timeout=remaining)

    async def read_text_file(self, path: str, abort_signal: AbortSignal | None = None) -> Result[str, FileError]:
        resolved = resolve_path(self.cwd, path)
        aborted = _abort_result(abort_signal, resolved)
        if aborted is not None:
            return aborted
        try:
            # `newline=""` disables universal-newline translation so CRLF files round-trip
            # byte-for-byte, matching Node's `readFile(path, "utf-8")`.
            return ok(await asyncio.to_thread(_read_text, resolved))
        except Exception as error:
            return err(to_file_error(error, resolved))

    async def read_text_lines(
        self, path: str, max_lines: int | None = None, abort_signal: AbortSignal | None = None
    ) -> Result[list[str], FileError]:
        resolved = resolve_path(self.cwd, path)
        aborted = _abort_result(abort_signal, resolved)
        if aborted is not None:
            return aborted
        if max_lines is not None and max_lines <= 0:
            return ok([])

        def read() -> list[str]:
            lines: list[str] = []
            with open(resolved, encoding="utf-8", newline="") as handle:
                for raw_line in handle:
                    lines.append(raw_line.rstrip("\n").rstrip("\r"))
                    if max_lines is not None and len(lines) >= max_lines:
                        break
            return lines

        try:
            lines = await asyncio.to_thread(read)
        except Exception as error:
            return err(to_file_error(error, resolved))
        after_read_abort = _abort_result(abort_signal, resolved)
        if after_read_abort is not None:
            return after_read_abort
        return ok(lines)

    async def read_binary_file(self, path: str, abort_signal: AbortSignal | None = None) -> Result[bytes, FileError]:
        resolved = resolve_path(self.cwd, path)
        aborted = _abort_result(abort_signal, resolved)
        if aborted is not None:
            return aborted
        try:
            return ok(await asyncio.to_thread(Path(resolved).read_bytes))
        except Exception as error:
            return err(to_file_error(error, resolved))

    async def write_file(
        self, path: str, content: str | bytes, abort_signal: AbortSignal | None = None
    ) -> Result[None, FileError]:
        resolved = resolve_path(self.cwd, path)
        aborted = _abort_result(abort_signal, resolved)
        if aborted is not None:
            return aborted
        try:
            await asyncio.to_thread(os.makedirs, os.path.dirname(resolved), exist_ok=True)
            after_mkdir_abort = _abort_result(abort_signal, resolved)
            if after_mkdir_abort is not None:
                return after_mkdir_abort
            await asyncio.to_thread(_write, resolved, content, "wb")
            return ok(None)
        except Exception as error:
            return err(to_file_error(error, resolved))

    async def append_file(
        self, path: str, content: str | bytes, abort_signal: AbortSignal | None = None
    ) -> Result[None, FileError]:
        resolved = resolve_path(self.cwd, path)
        try:
            await asyncio.to_thread(os.makedirs, os.path.dirname(resolved), exist_ok=True)
            await asyncio.to_thread(_write, resolved, content, "ab")
            return ok(None)
        except Exception as error:
            return err(to_file_error(error, resolved))

    async def rename_file(
        self, source_path: str, destination_path: str, abort_signal: AbortSignal | None = None
    ) -> Result[None, FileError]:
        source = resolve_path(self.cwd, source_path)
        destination = resolve_path(self.cwd, destination_path)
        aborted = _abort_result(abort_signal, destination)
        if aborted is not None:
            return aborted
        try:
            await asyncio.to_thread(os.replace, source, destination)
            return ok(None)
        except Exception as error:
            return err(to_file_error(error, source))

    async def file_info(self, path: str, abort_signal: AbortSignal | None = None) -> Result[FileInfo, FileError]:
        resolved = resolve_path(self.cwd, path)
        try:
            return _file_info_from_stats(resolved, await asyncio.to_thread(os.lstat, resolved))
        except Exception as error:
            return err(to_file_error(error, resolved))

    async def list_dir(self, path: str, abort_signal: AbortSignal | None = None) -> Result[list[FileInfo], FileError]:
        resolved = resolve_path(self.cwd, path)
        aborted = _abort_result(abort_signal, resolved)
        if aborted is not None:
            return aborted
        try:
            names = await asyncio.to_thread(os.listdir, resolved)
        except Exception as error:
            return err(to_file_error(error, resolved))
        infos: list[FileInfo] = []
        for name in names:
            loop_abort = _abort_result(abort_signal, resolved)
            if loop_abort is not None:
                return loop_abort
            entry_path = os.path.join(resolved, name)
            try:
                info = _file_info_from_stats(entry_path, await asyncio.to_thread(os.lstat, entry_path))
            except Exception as error:
                return err(to_file_error(error, entry_path))
            if info.ok:
                infos.append(info.value)
        return ok(infos)

    async def canonical_path(self, path: str, abort_signal: AbortSignal | None = None) -> Result[str, FileError]:
        resolved = resolve_path(self.cwd, path)
        try:
            return ok(await asyncio.to_thread(os.path.realpath, resolved, strict=True))
        except Exception as error:
            return err(to_file_error(error, resolved))

    async def exists(self, path: str, abort_signal: AbortSignal | None = None) -> Result[bool, FileError]:
        result = await self.file_info(path)
        if result.ok:
            return ok(True)
        if result.error.code == "not_found":
            return ok(False)
        return err(result.error)

    async def create_dir(
        self, path: str, recursive: bool = True, abort_signal: AbortSignal | None = None
    ) -> Result[None, FileError]:
        resolved = resolve_path(self.cwd, path)
        try:
            if recursive:
                await asyncio.to_thread(os.makedirs, resolved, exist_ok=True)
            else:
                await asyncio.to_thread(os.mkdir, resolved)
            return ok(None)
        except Exception as error:
            return err(to_file_error(error, resolved))

    async def remove(
        self,
        path: str,
        recursive: bool = False,
        force: bool = False,
        abort_signal: AbortSignal | None = None,
    ) -> Result[None, FileError]:
        resolved = resolve_path(self.cwd, path)
        try:
            await asyncio.to_thread(_remove, resolved, recursive, force)
            return ok(None)
        except Exception as error:
            return err(to_file_error(error, resolved))

    async def create_temp_dir(
        self, prefix: str = "tmp-", abort_signal: AbortSignal | None = None
    ) -> Result[str, FileError]:
        try:
            return ok(await asyncio.to_thread(tempfile.mkdtemp, "", prefix))
        except Exception as error:
            return err(to_file_error(error))

    async def create_temp_file(
        self, prefix: str = "", suffix: str = "", abort_signal: AbortSignal | None = None
    ) -> Result[str, FileError]:
        directory = await self.create_temp_dir("tmp-")
        if not directory.ok:
            return directory
        file_path = os.path.join(directory.value, f"{prefix}{uuid.uuid4()}{suffix}")
        try:
            await asyncio.to_thread(_write, file_path, "", "wb")
            return ok(file_path)
        except Exception as error:
            return err(to_file_error(error, file_path))

    async def cleanup(self) -> None:
        for pid in self._active_child_pids:
            kill_process_tree(pid)
        self._active_child_pids.clear()


def _read_text(path: str) -> str:
    with open(path, encoding="utf-8", newline="") as handle:
        return handle.read()


def _write(path: str, content: str | bytes, mode: str) -> None:
    data = content.encode("utf-8") if isinstance(content, str) else content
    with open(path, mode) as handle:
        handle.write(data)


def _remove(path: str, recursive: bool, force: bool) -> None:
    try:
        entry = os.lstat(path)
    except FileNotFoundError:
        if force:
            return
        raise
    if stat.S_ISDIR(entry.st_mode):
        if recursive:
            shutil.rmtree(path)
        else:
            os.rmdir(path)
        return
    os.unlink(path)


__all__ = [
    "EXIT_STDIO_GRACE_SECONDS",
    "MAX_TIMEOUT_MS",
    "MAX_TIMEOUT_SECONDS",
    "LocalExecutionEnv",
    "ShellConfig",
    "get_shell_config",
    "get_shell_env",
    "kill_process_tree",
    "resolve_path",
    "to_file_error",
]
