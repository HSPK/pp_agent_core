"""Per-path serialization of harness file mutations.

Python port of `packages/agent/src/harness/tools/file-mutation-queue.ts`.
TypeScript chains promises per canonical path in a `WeakMap` keyed by the
execution environment; this uses an `asyncio.Lock` per (env, canonical path)
pair held in a `WeakKeyDictionary`, which gives the same "one mutation at a
time per canonical path per environment" guarantee. Entries are reference
counted so an entry is only dropped once no caller is queued behind it
(TypeScript gets this for free from its promise chain identity check).
"""

from __future__ import annotations

import asyncio
import weakref
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TypeVar

from ..types import ExecutionEnv, get_or_throw

T = TypeVar("T")


@dataclass
class _QueueEntry:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    waiters: int = 0


_states: weakref.WeakKeyDictionary[object, dict[str, _QueueEntry]] = weakref.WeakKeyDictionary()


def _get_state(env: ExecutionEnv) -> dict[str, _QueueEntry]:
    state = _states.get(env)
    if state is None:
        state = {}
        _states[env] = state
    return state


async def _get_mutation_queue_key(env: ExecutionEnv, path: str) -> str:
    absolute_path = get_or_throw(await env.absolute_path(path))
    canonical_path = await env.canonical_path(absolute_path)
    if canonical_path.ok:
        return canonical_path.value
    if canonical_path.error.code in ("not_found", "not_supported"):
        return absolute_path
    raise canonical_path.error


async def with_file_mutation_queue(env: ExecutionEnv, path: str, fn: Callable[[], Awaitable[T]]) -> T:
    """Serialize file mutations targeting the same environment and canonical path."""
    state = _get_state(env)
    key = await _get_mutation_queue_key(env, path)
    entry = state.get(key)
    if entry is None:
        entry = _QueueEntry()
        state[key] = entry
    entry.waiters += 1

    async with entry.lock:
        try:
            return await fn()
        finally:
            entry.waiters -= 1
            if entry.waiters == 0 and state.get(key) is entry:
                del state[key]


__all__ = ["with_file_mutation_queue"]
