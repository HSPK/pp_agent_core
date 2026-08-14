"""Harness-level run lifecycle event bus.

Python port of `packages/agent/src/harness/events.ts`. `HarnessEventBus`
supports two subscription styles: `on(type, listener)` for passive
type-scoped listeners (no replay, no snapshot), and `watch(capture_snapshot)`
for a snapshot-plus-buffered-then-live subscription (used by lane/session
watchers that need "current state plus everything from now on").
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar


@dataclass(kw_only=True)
class RunStartEvent:
    lane: str
    run_id: str
    type: Literal["run_start"] = "run_start"


@dataclass(kw_only=True)
class RunEndEvent:
    lane: str
    run_id: str
    outcome: Literal["completed", "aborted", "failed"]
    leaf_id: str
    type: Literal["run_end"] = "run_end"


HarnessEvent = RunStartEvent | RunEndEvent
HarnessEventType = Literal["run_start", "run_end"]
HarnessEventListener = Callable[[HarnessEvent], Awaitable[None] | None]

TSnapshot = TypeVar("TSnapshot")


@dataclass
class WatchHandle(Generic[TSnapshot]):
    snapshot: TSnapshot
    start: Callable[[HarnessEventListener], None]
    unsubscribe: Callable[[], None]


class HarnessEventBus:
    def __init__(self) -> None:
        self._listeners: dict[HarnessEventType, set[HarnessEventListener]] = {}
        self._watch_listeners: set[Callable[[HarnessEvent], None]] = set()
        self._background_tasks: set[asyncio.Task[None]] = set()

    def _fire_and_forget(self, awaitable: Awaitable[None]) -> None:
        """Schedules a listener's async result without awaiting it (`emit`/delivery are
        synchronous). Keeps a strong reference to the task until it completes so it is
        not garbage-collected mid-flight."""
        task = asyncio.ensure_future(awaitable)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    def on(self, type: HarnessEventType, listener: HarnessEventListener) -> Callable[[], None]:
        """Register a listener for future events of one type and return its unsubscribe function.

        Earlier events are not replayed, and no snapshot or event buffer is provided.
        """
        # Reuse this event type's listener set, or create its first set.
        listeners = self._listeners.setdefault(type, set())

        # Wrap this event-specific callback so it can be stored as a general HarnessEvent listener.
        # Keep the wrapper reference so unsubscribe can remove that exact function from the set.
        def receive(event: HarnessEvent) -> Awaitable[None] | None:
            if event.type == type:
                return listener(event)
            return None

        listeners.add(receive)

        def unsubscribe() -> None:
            listeners.discard(receive)
            if not listeners:
                self._listeners.pop(type, None)

        return unsubscribe

    def emit(self, event: HarnessEvent) -> None:
        """Publish an event to current event subscriptions and watch subscriptions."""
        # Deliver only to direct listeners registered for this event type.
        # Async results are not awaited because emit() is synchronous.
        for listener in list(self._listeners.get(event.type, ())):
            result = listener(event)
            if result is not None:
                self._fire_and_forget(result)

        # Deliver every event to each watcher; watch() handles buffering until start().
        for watch_listener in list(self._watch_listeners):
            watch_listener(event)

    def watch(self, capture_snapshot: Callable[[], TSnapshot]) -> WatchHandle[TSnapshot]:
        state: dict[str, object] = {"listener": None, "buffered": []}

        def receive(event: HarnessEvent) -> None:
            listener = state["listener"]
            if listener is not None:
                result = listener(event)  # type: ignore[operator]
                if result is not None:
                    self._fire_and_forget(result)
            else:
                state["buffered"].append(event)  # type: ignore[attr-defined]

        self._watch_listeners.add(receive)
        snapshot = capture_snapshot()

        def start(next_listener: HarnessEventListener) -> None:
            # Stay in buffering mode while flushing so reentrant emissions preserve order.
            while state["buffered"]:
                pending = state["buffered"]
                state["buffered"] = []
                for event in pending:  # type: ignore[union-attr]
                    result = next_listener(event)
                    if result is not None:
                        self._fire_and_forget(result)
            state["listener"] = next_listener

        def unsubscribe() -> None:
            self._watch_listeners.discard(receive)
            state["buffered"] = []

        return WatchHandle(snapshot=snapshot, start=start, unsubscribe=unsubscribe)


__all__ = [
    "HarnessEvent",
    "HarnessEventBus",
    "HarnessEventListener",
    "HarnessEventType",
    "RunEndEvent",
    "RunStartEvent",
    "WatchHandle",
]
