"""Tests for `pi_agent.harness.session.jsonl.lockfile`.

This module is a Python-specific addition (see its docstring): the ported
TypeScript session code does not use file locking at all, so there is no
TypeScript conformance test to port here. These tests instead directly
exercise the four documented lock lifecycle properties: acquire, contend,
release, and stale-lock takeover.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import pytest

from pi_agent.harness.session.jsonl.lockfile import FileLock, LockTimeoutError


def test_acquire_creates_lock_file_and_release_removes_it(tmp_path: Path) -> None:
    target = tmp_path / "session.jsonl"
    lock = FileLock(target)

    lock.acquire()
    try:
        assert lock.path.exists()
        assert lock.path == tmp_path / "session.jsonl.lock"
    finally:
        lock.release()

    assert not lock.path.exists()


def test_release_is_idempotent_when_not_held(tmp_path: Path) -> None:
    lock = FileLock(tmp_path / "session.jsonl")
    # Releasing a lock that was never acquired must not raise.
    lock.release()
    lock.release()


def test_acquire_raises_if_already_held_by_self(tmp_path: Path) -> None:
    lock = FileLock(tmp_path / "session.jsonl")
    lock.acquire()
    try:
        with pytest.raises(RuntimeError):
            lock.acquire()
    finally:
        lock.release()


def test_context_manager_acquires_and_releases(tmp_path: Path) -> None:
    target = tmp_path / "session.jsonl"
    lock = FileLock(target)
    with lock:
        assert lock.path.exists()
    assert not lock.path.exists()


def test_context_manager_releases_on_exception(tmp_path: Path) -> None:
    target = tmp_path / "session.jsonl"
    lock = FileLock(target)
    with pytest.raises(ValueError, match="boom"), lock:
        assert lock.path.exists()
        raise ValueError("boom")
    assert not lock.path.exists()


def test_second_acquirer_blocks_until_first_releases(tmp_path: Path) -> None:
    target = tmp_path / "session.jsonl"
    first = FileLock(target, poll_interval=0.01)
    second = FileLock(target, poll_interval=0.01)

    first.acquire()
    acquired_at: list[float] = []
    released_at: list[float] = []

    def contend() -> None:
        second.acquire(timeout=5.0)
        acquired_at.append(time.monotonic())
        second.release()

    thread = threading.Thread(target=contend)
    thread.start()
    time.sleep(0.2)  # give the contender time to start polling
    released_at.append(time.monotonic())
    first.release()
    thread.join(timeout=5.0)

    assert not thread.is_alive()
    assert len(acquired_at) == 1
    # The second lock must not have acquired before the first was released.
    assert acquired_at[0] >= released_at[0]


def test_acquire_times_out_when_lock_is_held_and_not_stale(tmp_path: Path) -> None:
    target = tmp_path / "session.jsonl"
    holder = FileLock(target)
    contender = FileLock(target, stale_after=60.0, poll_interval=0.01)

    holder.acquire()
    try:
        with pytest.raises(LockTimeoutError):
            contender.acquire(timeout=0.1)
    finally:
        holder.release()


def test_stale_lock_is_taken_over_by_next_acquirer(tmp_path: Path) -> None:
    target = tmp_path / "session.jsonl"
    abandoned = FileLock(target)
    abandoned.acquire()
    # Simulate a crashed holder: back-date the lock file's mtime past
    # `stale_after` without releasing it (a real crash leaves the file behind
    # with no owning process to call `release()`).
    old_time = time.time() - 120
    os.utime(abandoned.path, (old_time, old_time))

    successor = FileLock(target, stale_after=30.0, poll_interval=0.01)
    successor.acquire(timeout=1.0)
    try:
        assert successor.path.exists()
        # The stale takeover must have replaced the lock file's contents/mtime.
        age = time.time() - successor.path.stat().st_mtime
        assert age < 30.0
    finally:
        successor.release()


def test_fresh_lock_is_not_treated_as_stale(tmp_path: Path) -> None:
    target = tmp_path / "session.jsonl"
    holder = FileLock(target)
    contender = FileLock(target, stale_after=30.0, poll_interval=0.01)

    holder.acquire()
    try:
        with pytest.raises(LockTimeoutError):
            contender.acquire(timeout=0.2)
    finally:
        holder.release()
