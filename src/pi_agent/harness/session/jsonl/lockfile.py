"""Minimal cross-process file lock for JSONL session files.

Python-specific addition: the TypeScript `pi` uses the `proper-lockfile` npm
package for cross-process advisory locking elsewhere in the monorepo (for
example `packages/coding-agent/src/core/settings-manager.ts`), but the
session module ported here does not use file locking at all -- Node
processes serialize writes to the same file in-process via a promise chain
(`JsonlSessionStorage`'s `tail`/`enqueue`), and same-process create/fork races
are guarded by `JsonlSessionRepo`'s in-memory `activeCreateDestinations` set.
Neither mechanism protects a JSONL session file against a second OS process
appending to it concurrently.

This module adds that missing cross-process guard using only the standard
library: a sibling `<path>.lock` file created with `O_CREAT | O_EXCL` (atomic
across processes on POSIX and Windows), with stale-lock takeover based on the
lock file's mtime. No third-party dependency is introduced.
"""

from __future__ import annotations

import contextlib
import os
import time
from pathlib import Path
from types import TracebackType


class LockTimeoutError(TimeoutError):
    """Raised when a lock could not be acquired before `timeout` elapsed."""


class FileLock:
    """Advisory lock on `<path>.lock`, held via a synchronous context manager.

    A lock file older than `stale_after` seconds is treated as abandoned (for
    example, left behind by a process that crashed before releasing it) and is
    taken over by the next acquirer.
    """

    def __init__(self, path: str | Path, *, stale_after: float = 30.0, poll_interval: float = 0.05) -> None:
        self._lock_path = Path(f"{path}.lock")
        self._stale_after = stale_after
        self._poll_interval = poll_interval
        self._held = False

    @property
    def path(self) -> Path:
        return self._lock_path

    def acquire(self, *, timeout: float | None = 5.0) -> None:
        if self._held:
            raise RuntimeError(f"Lock already held: {self._lock_path}")
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            try:
                fd = os.open(self._lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                try:
                    os.write(fd, str(os.getpid()).encode("ascii"))
                finally:
                    os.close(fd)
                self._held = True
                return
            except FileExistsError:
                if self._reclaim_if_stale():
                    continue
                if deadline is not None and time.monotonic() >= deadline:
                    raise LockTimeoutError(f"Timed out waiting for lock: {self._lock_path}") from None
                time.sleep(self._poll_interval)

    def release(self) -> None:
        if not self._held:
            return
        self._lock_path.unlink(missing_ok=True)
        self._held = False

    def _reclaim_if_stale(self) -> bool:
        try:
            age = time.time() - self._lock_path.stat().st_mtime
        except FileNotFoundError:
            return True
        if age <= self._stale_after:
            return False
        with contextlib.suppress(FileNotFoundError):
            self._lock_path.unlink()
        return True

    def __enter__(self) -> FileLock:
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()


__all__ = ["FileLock", "LockTimeoutError"]
