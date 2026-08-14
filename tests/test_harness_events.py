"""Tests for `pi_agent.harness.events`.

Ported from `packages/agent/test/harness/events.test.ts`.
"""

from __future__ import annotations

from pi_agent.harness.events import HarnessEvent, HarnessEventBus, RunEndEvent, RunStartEvent

run_start_event = RunStartEvent(lane="main", run_id="run-1")
run_end_event = RunEndEvent(lane="main", run_id="run-1", outcome="completed", leaf_id="entry-1")


def test_delivers_matching_events_to_direct_listeners_and_watchers():
    events = HarnessEventBus()
    direct: list[RunStartEvent] = []
    watch_events: list[HarnessEvent] = []

    off = events.on("run_start", lambda event: direct.append(event))
    watch = events.watch(lambda: None)
    watch.start(lambda event: watch_events.append(event))

    events.emit(run_start_event)
    events.emit(run_end_event)
    off()
    events.emit(run_start_event)

    assert direct == [run_start_event]
    assert watch_events == [run_start_event, run_end_event, run_start_event]


def test_captures_snapshot_without_event_gap_then_flushes_and_delivers_live_events():
    events = HarnessEventBus()
    expected_snapshot = {"leaf_id": None}

    def capture_snapshot():
        events.emit(run_start_event)
        return expected_snapshot

    watch = events.watch(capture_snapshot)
    received: list[HarnessEvent] = []

    assert watch.snapshot is expected_snapshot
    assert received == []

    watch.start(lambda event: received.append(event))
    assert received == [run_start_event]

    events.emit(run_end_event)
    assert received == [run_start_event, run_end_event]

    watch.unsubscribe()
    events.emit(run_start_event)
    assert received == [run_start_event, run_end_event]
