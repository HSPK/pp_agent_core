"""Conformance suite for every `SessionRepo` backend.

Python port of `packages/agent/src/harness/session/testing/conformance.ts`
(1016 lines). Each TypeScript `createCase(factory, group, name, test)` becomes
one `async def test_...` function here; the `backend` fixture below is
parametrized over `"memory"` (`InMemorySessionRepo`) and `"jsonl"`
(`JsonlSessionRepo` rooted in `tmp_path`), so every test runs against both
backends -- mirroring how `memory.test.ts` and `jsonl.test.ts` each call
`createSessionBackendConformance` with their own fixture factory.

Grouping comments below match the TypeScript `group` labels
("entries and lanes", "records and log", "queries and facts", "validation and
immutability", "repository and forks") for traceability back to the source.

Python-specific adaptations to the "rejects non-JSON ... records" cases: the
TypeScript source builds invalid payloads with `undefined` fields, a
`BigInt`, and a `Map` -- none of which have a direct Python/JSON equivalent.
This port substitutes Python-native values that exercise the same
`assert_json_serializable` checks (non-finite floats, unsupported object
types, and reference cycles); see the comment at each adapted case.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest
from pi_ai import Cost, TextContent, ToolResultMessage, Usage
from session_conformance_helpers import (
    assert_rejects_with_code,
    create_assistant_message,
    create_user_message,
    entry_ids,
    operation_started,
)

from pi_agent.harness.session import (
    BranchBounds,
    CompactionEntry,
    CustomEntry,
    EntryCursor,
    EntryQuery,
    ForkOptions,
    InMemorySessionRepo,
    InMemorySessionStorage,
    JsonlForkOptions,
    JsonlSessionCreateOptions,
    JsonlSessionRepo,
    JsonlSessionRepoOptions,
    LogOptions,
    MessageEntry,
    OperationFinishedRecord,
    QueueCancelledRecord,
    QueueEnqueuedRecord,
    RecordQuery,
    Session,
    SessionCreateOptions,
    SessionMetadata,
    StepAttemptRecord,
    ToolStartedRecord,
    UsageRecord,
)


@dataclass(kw_only=True)
class ConformanceBackend:
    name: str
    repository: Any

    def create_options(self, **kwargs: Any) -> Any:
        if self.name == "memory":
            return SessionCreateOptions(**kwargs)
        return JsonlSessionCreateOptions(**kwargs)

    def fork_options(self, **kwargs: Any) -> Any:
        if self.name == "memory":
            return ForkOptions(**kwargs)
        return JsonlForkOptions(**kwargs)


@pytest.fixture(params=["memory", "jsonl"])
def backend(request: pytest.FixtureRequest, tmp_path: Any) -> ConformanceBackend:
    if request.param == "memory":
        return ConformanceBackend(name="memory", repository=InMemorySessionRepo())
    repository = JsonlSessionRepo(JsonlSessionRepoOptions(sessions_root=tmp_path / "sessions"))
    return ConformanceBackend(name="jsonl", repository=repository)


# --------------------------------------------------------------------------
# entries and lanes
# --------------------------------------------------------------------------


async def test_assigns_parents_and_one_sequence_across_every_mutation(backend: ConformanceBackend) -> None:
    repository = backend.repository
    session = await repository.create(backend.create_options(id="session"))
    root = await session.append_entry(MessageEntry(id="root", message=create_user_message("root")), "main")
    await session.create_lane("thread", root.id)
    child = await session.append_entry(CustomEntry(id="child", custom_type="note", data={"value": 1}), "thread")
    record = await session.append_record(operation_started("run", lane="thread", kind="run"))
    await session.set_name("Example")
    await session.set_label(root.id, "checkpoint")
    await session.move_lane("main", child.id)

    assert (root.parent_id, root.seq) == (None, 1)
    assert (child.parent_id, child.seq) == ("root", 3)
    assert record.seq == 4
    for timestamp in (root.timestamp, child.timestamp, record.timestamp):
        assert isinstance(timestamp, int) and timestamp >= 0, "storage-assigned timestamps must be Unix milliseconds"

    log = await session.get_log()
    assert [(item.kind, item.seq) for item in log] == [
        ("entry", 1),
        ("lane", 2),
        ("entry", 3),
        ("record", 4),
        ("fact", 5),
        ("fact", 6),
        ("lane", 7),
    ]
    lanes = await session.get_lanes()
    assert [(pointer.lane, pointer.leaf_id) for pointer in lanes] == [("main", "child"), ("thread", "child")]


async def test_rejects_duplicate_ids_without_changing_state(backend: ConformanceBackend) -> None:
    repository = backend.repository
    session = await repository.create(backend.create_options(id="session"))
    await session.append_entry(MessageEntry(id="shared", message=create_user_message("root")), "main")
    await assert_rejects_with_code(
        session.append_record(operation_started("shared", lane="main", kind="run")), "already_exists"
    )
    await session.append_record(operation_started("run", lane="main", kind="run"))
    await assert_rejects_with_code(
        session.append_entry(CustomEntry(id="run", custom_type="note"), "main"), "already_exists"
    )
    log = await session.get_log()
    assert [item.seq for item in log] == [1, 2]


async def test_isolates_lanes_while_sharing_the_tree(backend: ConformanceBackend) -> None:
    repository = backend.repository
    session = await repository.create(backend.create_options(id="session"))
    await session.append_entry(MessageEntry(id="root", message=create_user_message("root")), "main")
    await session.create_lane("thread", "root")
    await session.append_entry(MessageEntry(id="main-child", message=create_user_message("main")), "main")
    await session.append_entry(MessageEntry(id="thread-child", message=create_user_message("thread")), "thread")

    lanes = await session.get_lanes()
    assert [(pointer.lane, pointer.leaf_id) for pointer in lanes] == [
        ("main", "main-child"),
        ("thread", "thread-child"),
    ]
    assert await entry_ids(
        session.find_entries_on_branch(EntryQuery(order="oldestFirst"), BranchBounds(start="main-child"))
    ) == ["root", "main-child"]
    assert await entry_ids(
        session.find_entries_on_branch(EntryQuery(order="oldestFirst"), BranchBounds(start="thread-child"))
    ) == ["root", "thread-child"]


async def test_validates_lane_lifecycle_and_targets(backend: ConformanceBackend) -> None:
    repository = backend.repository
    session = await repository.create(backend.create_options(id="session"))
    await assert_rejects_with_code(session.create_lane("main", None), "already_exists")
    await assert_rejects_with_code(session.create_lane("thread", "missing"), "not_found")
    await assert_rejects_with_code(session.move_lane("missing", None), "invalid_lane")


async def test_binds_lane_views_without_caching_leaves(backend: ConformanceBackend) -> None:
    repository = backend.repository
    session = await repository.create(backend.create_options(id="session"))
    root = await session.append_message(create_user_message("root"))
    await session.create_lane("thread", root)
    thread = session.view("thread")
    main_child, thread_child = await asyncio.gather(
        session.append_message(create_user_message("main")),
        thread.append_message(create_user_message("thread")),
    )

    assert await session.get_leaf_id() == main_child
    assert await thread.get_leaf_id() == thread_child
    assert await entry_ids(session.find_entries_on_branch(EntryQuery(order="oldestFirst"))) == [root, main_child]
    assert await entry_ids(thread.find_entries_on_branch(EntryQuery(order="oldestFirst"))) == [root, thread_child]
    empty = await repository.create(backend.create_options(id="empty"))
    assert await empty.find_entries_on_branch() == []


async def test_appends_provisioned_entries_with_their_existing_ids(backend: ConformanceBackend) -> None:
    repository = backend.repository
    session = await repository.create(backend.create_options(id="session"))
    entry = await session.append_entry(CustomEntry(id="provisioned", custom_type="note", data={"value": 1}), "main")

    assert entry.custom_type == "note"
    assert (entry.id, entry.parent_id, entry.seq) == ("provisioned", None, 1)
    assert await session.get_leaf_id() == "provisioned"


async def test_persists_tool_result_termination_decisions(backend: ConformanceBackend) -> None:
    repository = backend.repository
    session = await repository.create(backend.create_options(id="session"))
    entry = await session.append_entry(
        MessageEntry(
            id="tool-result",
            message=ToolResultMessage(
                tool_call_id="call-1",
                tool_name="example",
                content=[TextContent(text="done")],
                is_error=False,
                timestamp=1,
            ),
            terminate=True,
        ),
        "main",
    )

    assert entry.terminate is True
    stored = await session.get_entry(entry.id)
    assert stored.terminate is True
    assert await session.find_entries() == [entry]
    log = await session.get_log()
    assert len(log) == 1 and log[0].kind == "entry" and log[0].seq == entry.seq and log[0].entry == entry


async def test_linearizes_concurrent_writes_across_two_lanes(backend: ConformanceBackend) -> None:
    repository = backend.repository
    session = await repository.create(backend.create_options(id="session"))
    await session.append_entry(MessageEntry(id="root", message=create_user_message("root")), "main")
    await session.create_lane("thread", "root")
    completion_order: list[str] = []

    async def track(coro: Any) -> Any:
        entry = await coro
        completion_order.append(entry.id)
        return entry

    writes = [
        track(session.append_entry(CustomEntry(id="main-1", custom_type="note"), "main")),
        track(session.append_entry(CustomEntry(id="thread-1", custom_type="note"), "thread")),
        track(session.append_entry(CustomEntry(id="main-2", custom_type="note"), "main")),
        track(session.append_entry(CustomEntry(id="thread-2", custom_type="note"), "thread")),
    ]
    entries = await asyncio.gather(*writes)
    commit_order = [entry.id for entry in sorted(entries, key=lambda entry: entry.seq)]

    assert len({entry.seq for entry in entries}) == len(entries)
    assert completion_order == commit_order
    concurrent_ids = {entry.id for entry in entries}
    log = await session.get_log()
    assert [item.entry.id for item in log if item.kind == "entry" and item.entry.id in concurrent_ids] == commit_order
    sequences = [item.seq for item in log]
    assert sequences == sorted(sequences)


# --------------------------------------------------------------------------
# records and log
# --------------------------------------------------------------------------


async def test_commits_records_and_lane_moves_as_separate_mutations(backend: ConformanceBackend) -> None:
    repository = backend.repository
    session = await repository.create(backend.create_options(id="session"))
    root = await session.append_entry(MessageEntry(id="root", message=create_user_message("root")), "main")
    finished = await session.append_record(
        _operation_finished(id="finish", lane="main", run_id="run", outcome="completed")
    )

    assert finished.seq == 2
    lanes = await session.get_lanes()
    assert [(pointer.lane, pointer.leaf_id) for pointer in lanes] == [("main", "root")]
    await session.move_lane("main", None)
    lanes = await session.get_lanes()
    assert [(pointer.lane, pointer.leaf_id) for pointer in lanes] == [("main", None)]

    log = await session.get_log()
    assert [(item.kind, item.seq) for item in log] == [("entry", 1), ("record", 2), ("lane", 3)]
    assert log[0].entry == root
    assert log[1].record == finished

    await assert_rejects_with_code(session.move_lane("main", "missing"), "not_found")
    assert len(await session.find_records()) == 1
    log = await session.get_log()
    assert [item.seq for item in log] == [1, 2, 3]


async def test_keeps_lane_names_permanent_with_their_recovery_records(backend: ConformanceBackend) -> None:
    repository = backend.repository
    session = await repository.create(backend.create_options(id="session"))
    await session.create_lane("thread", None)
    await session.append_record(operation_started("old-run", lane="thread", kind="run"))
    await session.append_record(
        QueueEnqueuedRecord(
            id="old-next-run",
            lane="thread",
            queue="nextRun",
            target=MessageEntry(id="queued-message", message=create_user_message("queued")),
        )
    )

    records = await session.find_records(RecordQuery(lane="thread"))
    assert [record.id for record in records] == ["old-next-run", "old-run"]
    log = await session.get_log()
    assert [item.record.id for item in log if item.kind == "record"] == ["old-run", "old-next-run"]
    await assert_rejects_with_code(session.create_lane("thread", None), "already_exists")


async def test_persists_queue_cancellation_without_consuming_its_target(backend: ConformanceBackend) -> None:
    repository = backend.repository
    session = await repository.create(backend.create_options(id="session"))
    enqueued = await session.append_record(
        QueueEnqueuedRecord(
            id="enqueue",
            lane="main",
            queue="nextRun",
            target=MessageEntry(id="queued-message", message=create_user_message("queued")),
        )
    )
    cancelled = await session.append_record(QueueCancelledRecord(id="cancel", lane="main", entry_id="queued-message"))
    assert (cancelled.seq, cancelled.entry_id) == (2, "queued-message")
    # TypeScript asserts `"runId" in cancelled === false`. A Python dataclass always
    # has the attribute, so the equivalent claim is that it stays unset.
    assert cancelled.run_id is None
    assert await session.get_entry("queued-message") is None
    cancellations = await session.find_records(RecordQuery(type="queue_cancelled"))
    assert cancellations[0].entry_id == "queued-message"
    assert cancellations == [cancelled]
    log = await session.get_log()
    assert [(item.kind, item.seq) for item in log] == [("record", enqueued.seq), ("record", cancelled.seq)]
    assert log[0].record == enqueued
    assert log[1].record == cancelled


async def test_filters_records_by_lane_type_run_sequence_and_order(backend: ConformanceBackend) -> None:
    repository = backend.repository
    session = await repository.create(backend.create_options(id="session"))
    await session.append_record(operation_started("run-1", lane="main", kind="run"))
    await session.append_record(
        _step_attempt(
            id="attempt-1", lane="main", run_id="run-1", step="assistant", attempt=1, result_entry_id="assistant-1"
        )
    )
    await session.create_lane("thread", None)
    await session.append_record(operation_started("run-2", lane="thread", kind="run"))
    await session.append_record(
        _step_attempt(
            id="attempt-2", lane="thread", run_id="run-2", step="assistant", attempt=1, result_entry_id="assistant-2"
        )
    )

    records = await session.find_records(RecordQuery(lane="thread"))
    assert [record.id for record in records] == ["attempt-2", "run-2"]
    records = await session.find_records(RecordQuery(type="step_attempt", order="oldestFirst"))
    assert [record.id for record in records] == ["attempt-1", "attempt-2"]
    records = await session.find_records(RecordQuery(run_id="run-1", after_seq=1))
    assert [record.id for record in records] == ["attempt-1"]
    records = await session.find_records(RecordQuery(limit=1))
    assert [record.id for record in records] == ["attempt-2"]


async def test_filters_operation_starts_by_operation_kind(backend: ConformanceBackend) -> None:
    repository = backend.repository
    session = await repository.create(backend.create_options(id="session"))
    await session.append_record(operation_started("run-old", lane="main", kind="run"))
    await session.append_record(
        _operation_finished(id="run-old-finished", lane="main", run_id="run-old", outcome="completed")
    )
    await session.append_record(operation_started("compaction", lane="main", kind="compaction"))
    await session.append_record(
        _operation_finished(id="compaction-finished", lane="main", run_id="compaction", outcome="completed")
    )
    await session.append_record(operation_started("navigation", lane="main", kind="navigation"))
    await session.append_record(
        _operation_finished(id="navigation-finished", lane="main", run_id="navigation", outcome="completed")
    )
    await session.append_record(operation_started("run-new", lane="main", kind="run"))

    records = await session.find_records(
        RecordQuery(type="operation_started", operation_kind="run", order="oldestFirst")
    )
    assert [record.id for record in records] == ["run-old", "run-new"]
    records = await session.find_records(RecordQuery(type="operation_started", operation_kind="compaction"))
    assert [record.id for record in records] == ["compaction"]
    records = await session.find_records(RecordQuery(type="operation_started", operation_kind="navigation"))
    assert [record.id for record in records] == ["navigation"]
    records = await session.find_records(RecordQuery(type="operation_started", operation_kind="run", limit=1))
    assert [record.id for record in records] == ["run-new"]


async def test_tracks_and_enforces_one_open_operation_per_lane(backend: ConformanceBackend) -> None:
    repository = backend.repository
    session = await repository.create(backend.create_options(id="session"))
    assert await session.find_open_operations("main", limit=2) == []

    first = await session.append_record(operation_started("first", lane="main", kind="run"))
    assert await session.find_open_operations("main", limit=2) == [first]
    await assert_rejects_with_code(
        session.append_record(operation_started("second", lane="main", kind="run")), "storage"
    )
    assert await session.find_open_operations("main", limit=2) == [first]

    await session.append_record(
        _operation_finished(id="finish-first", lane="main", run_id=first.id, outcome="completed")
    )
    assert await session.find_open_operations("main", limit=2) == []


async def test_does_not_let_an_earlier_finish_close_a_later_start(backend: ConformanceBackend) -> None:
    repository = backend.repository
    session = await repository.create(backend.create_options(id="session"))
    await session.append_record(
        _operation_finished(id="finish-before-start", lane="main", run_id="run", outcome="completed")
    )
    started = await session.append_record(operation_started("run", lane="main", kind="run"))
    assert await session.find_open_operations("main", limit=2) == [started]


async def test_scopes_open_operations_by_lane_and_limit(backend: ConformanceBackend) -> None:
    repository = backend.repository
    session = await repository.create(backend.create_options(id="session"))
    await session.create_lane("thread", None)
    main_run = await session.append_record(operation_started("main-run", lane="main", kind="run"))
    thread_navigation = await session.append_record(
        operation_started("thread-navigation", lane="thread", kind="navigation")
    )

    assert await session.find_open_operations("main") == [main_run]
    assert await session.find_open_operations("main", limit=1) == [main_run]
    assert await session.find_open_operations("thread", limit=2) == [thread_navigation]


async def test_keeps_latest_value_facts_and_computes_ledger_statistics_across_lanes(
    backend: ConformanceBackend,
) -> None:
    repository = backend.repository
    session = await repository.create(backend.create_options(id="session"))
    assistant = create_assistant_message("answer")
    assistant.usage = Usage(
        input=10,
        output=5,
        cache_read=3,
        cache_write=2,
        total_tokens=20,
        cost=Cost(input=1, output=2, cache_read=3, cache_write=4, total=10),
    )
    await session.append_entry(MessageEntry(id="user", message=create_user_message("question")), "main")
    await session.append_entry(MessageEntry(id="assistant", message=assistant), "main")
    await session.append_record(
        UsageRecord(
            id="assistant-usage",
            lane="main",
            cause="assistant",
            run_id="run",
            entry_id="assistant",
            attempt=1,
            stop_reason="stop",
            usage=assistant.usage,
        )
    )
    await session.append_record(
        UsageRecord(
            id="deferred-usage",
            lane="main",
            cause="deferred_fetch",
            run_id="run",
            entry_id="deferred-result",
            attempt=1,
            stop_reason="deferred",
            usage=Usage(input=0, output=0, cache_read=0, cache_write=0, total_tokens=0, cost=Cost()),
        )
    )
    await session.create_lane("thread", "assistant")
    await session.append_record(
        UsageRecord(
            id="correction",
            lane="thread",
            cause="adjustment",
            details={"reason": "provider correction"},
            usage=Usage(
                input=-2,
                output=0,
                cache_read=0,
                cache_write=0,
                total_tokens=-2,
                cost=Cost(input=-0.5, output=0, cache_read=0, cache_write=0, total=-0.5),
            ),
        )
    )
    await session.set_name("First")
    await session.set_name("Second")
    await session.set_label("user", "keep")
    await session.set_label("user", None)
    await assert_rejects_with_code(session.set_label("missing", "checkpoint"), "not_found")

    assert await session.get_name() == "Second"
    assert await session.get_label("user") is None
    usage_records = await session.find_records(RecordQuery(type="usage", order="oldestFirst"))
    assert [record.cause for record in usage_records] == ["assistant", "deferred_fetch", "adjustment"]
    deferred_usage = next(record for record in usage_records if record.cause == "deferred_fetch")
    assert deferred_usage.stop_reason == "deferred"
    stats = await session.get_stats()
    assert (stats.message_count, stats.cached_tokens, stats.uncached_tokens, stats.total_tokens, stats.cost_total) == (
        2,
        3,
        10,
        18,
        9.5,
    )


async def test_clears_session_names_durably(backend: ConformanceBackend) -> None:
    repository = backend.repository
    session = await repository.create(backend.create_options(id="session"))
    await session.set_name("Temporary")
    await session.set_name(None)

    assert await session.get_name() is None
    log = await session.get_log()
    assert [(item.kind, item.seq, item.fact, item.name) for item in log] == [
        ("fact", 1, "name", "Temporary"),
        ("fact", 2, "name", None),
    ]

    metadata = await session.get_metadata()
    reopened = await repository.open(metadata)
    assert await reopened.get_name() is None
    log = await reopened.get_log()
    assert [(item.kind, item.seq, item.fact, item.name) for item in log] == [
        ("fact", 1, "name", "Temporary"),
        ("fact", 2, "name", None),
    ]

    fork = await repository.fork(metadata, backend.fork_options(id="fork"))
    assert await fork.get_name() is None


# --------------------------------------------------------------------------
# queries and facts
# --------------------------------------------------------------------------


async def test_rejects_invalid_queries_before_empty_reads(backend: ConformanceBackend) -> None:
    repository = backend.repository
    session = await repository.create(backend.create_options(id="invalid-queries"))
    await session.create_lane("thread", None)
    thread = session.view("thread")

    await assert_rejects_with_code(session.find_entries(EntryQuery(limit=0)), "invalid_query")
    await assert_rejects_with_code(session.find_entry(EntryQuery(limit=0)), "invalid_query")
    await assert_rejects_with_code(session.find_entries_on_branch(EntryQuery(limit=0)), "invalid_query")
    await assert_rejects_with_code(
        thread.find_entries_on_branch(EntryQuery(cursor=EntryCursor(after_seq=-1))), "invalid_query"
    )
    await assert_rejects_with_code(thread.find_entry_on_branch(EntryQuery(limit=0)), "invalid_query")
    await assert_rejects_with_code(session.find_records(RecordQuery(limit=0)), "invalid_query")
    await assert_rejects_with_code(session.find_records(RecordQuery(operation_kind="run")), "invalid_query")
    await assert_rejects_with_code(
        session.find_records(RecordQuery(type="step_attempt", operation_kind="run")), "invalid_query"
    )
    await assert_rejects_with_code(session.find_open_operations("main", limit=0), "invalid_query")
    await assert_rejects_with_code(session.find_open_operations("main", limit=-1), "invalid_query")
    await assert_rejects_with_code(session.get_log(LogOptions(after_seq=-1)), "invalid_query")


async def test_supports_bounded_filtered_and_cursor_based_queries(backend: ConformanceBackend) -> None:
    repository = backend.repository
    session = await repository.create(backend.create_options(id="session"))
    await session.append_entry(MessageEntry(id="root", message=create_user_message("root")), "main")
    await session.append_entry(CustomEntry(id="old-note", custom_type="note", data=1), "main")
    await session.append_entry(_compaction_entry(id="compact", summary="summary", tokens_before=10), "main")
    await session.append_entry(CustomEntry(id="new-note", custom_type="note", data=2), "main")
    await session.append_entry(MessageEntry(id="tail", message=create_assistant_message("tail")), "main")

    assert await entry_ids(session.find_entries()) == ["tail", "new-note", "compact", "old-note", "root"]
    assert await entry_ids(
        session.find_entries(EntryQuery(order="oldestFirst", cursor=EntryCursor(after_seq=2), limit=2))
    ) == ["compact", "new-note"]
    assert await entry_ids(session.find_entries(EntryQuery(custom_type="note"))) == ["new-note", "old-note"]
    assert await entry_ids(
        session.find_entries_on_branch(EntryQuery(custom_type="note", limit=1), BranchBounds(start="tail"))
    ) == ["new-note"]
    assert await entry_ids(
        session.find_entries_on_branch(
            EntryQuery(type="message"), BranchBounds(start="tail", stop_at_type="compaction")
        )
    ) == ["tail"]
    assert (
        await entry_ids(
            session.find_entries_on_branch(EntryQuery(type="custom"), BranchBounds(start="tail", stop_at_id="tail"))
        )
        == []
    )
    assert await entry_ids(
        session.find_entries_on_branch(
            EntryQuery(order="oldestFirst"), BranchBounds(start="tail", stop_at_type="custom")
        )
    ) == ["root", "old-note"]
    await assert_rejects_with_code(session.find_entries(EntryQuery(limit=0)), "invalid_query")
    await assert_rejects_with_code(
        session.find_entries_on_branch(EntryQuery(), BranchBounds(start="missing")), "not_found"
    )


async def test_rejects_non_json_entries_before_storage_mutation(backend: ConformanceBackend) -> None:
    """Python-adapted invalid payloads: no `undefined`/`BigInt`/`Map` exist in Python,
    so this substitutes non-finite floats, an unsupported object type, and a
    reference cycle -- all rejected by the same `assert_json_serializable` walk."""
    repository = backend.repository
    session = await repository.create(backend.create_options(id="session"))
    cyclic: dict[str, Any] = {}
    cyclic["self"] = cyclic

    invalid_payloads: list[Any] = [
        {"value": float("nan")},
        [float("inf")],
        {"value": object()},
        {"value": {1, 2, 3}},
        cyclic,
    ]
    for data in invalid_payloads:
        await assert_rejects_with_code(session.append_custom_entry("invalid", data), "invalid_payload")

    assert await session.get_leaf_id() is None
    assert await session.find_entries() == []
    assert await session.get_log() == []
    valid_id = await session.append_custom_entry("valid", {"value": 1})
    entry = await session.get_entry(valid_id)
    assert entry.seq == 1


async def test_rejects_non_json_records_before_storage_mutation(backend: ConformanceBackend) -> None:
    """Python-adapted invalid payloads (see previous case's docstring)."""
    repository = backend.repository
    session = await repository.create(backend.create_options(id="session"))
    for record_id, value in [("nan-record", float("nan")), ("object-record", object())]:
        await assert_rejects_with_code(
            session.append_record(
                ToolStartedRecord(
                    id=record_id,
                    lane="main",
                    run_id="run",
                    assistant_entry_id="assistant",
                    tool_index=0,
                    tool_call_id="call",
                    tool_name="example",
                    effective_args={"value": value},
                    result_entry_id="result",
                    replay="never",
                )
            ),
            "invalid_payload",
        )

    assert await session.find_records() == []
    assert await session.get_log() == []
    valid = await session.append_record(operation_started("valid-record", lane="main", kind="run"))
    assert valid.seq == 1


async def test_returns_immutable_open_operation_records(backend: ConformanceBackend) -> None:
    repository = backend.repository
    session = await repository.create(backend.create_options(id="session"))
    committed = await session.append_record(operation_started("run", lane="main", kind="run"))
    open_operations = await session.find_open_operations("main")
    read = open_operations[0]
    assert read.intent.kind == "run"
    read.intent.original_prompt.append(create_user_message("mutated"))

    assert await session.find_open_operations("main") == [committed]


async def test_returns_immutable_copies_from_reads(backend: ConformanceBackend) -> None:
    repository = backend.repository
    session = await repository.create(backend.create_options(id="immutable"))
    metadata = await session.get_metadata()
    data = {"nested": {"value": 1}}
    await session.append_entry(CustomEntry(id="custom", custom_type="note", data=data), "main")
    data["nested"]["value"] = 50
    read = await session.get_entry("custom")
    read.data["nested"]["value"] = 99
    read_metadata = await session.get_metadata()
    read_metadata.id = "changed"
    log = await session.get_log()
    log[0].entry.data["nested"]["value"] = 100

    assert await session.get_metadata() == metadata
    final = await session.get_entry("custom")
    assert final == CustomEntry(
        id="custom",
        custom_type="note",
        data={"nested": {"value": 1}},
        parent_id=None,
        seq=1,
        timestamp=read.timestamp,
    )


# --------------------------------------------------------------------------
# repository and forks
# --------------------------------------------------------------------------


async def test_creates_lists_and_opens_sessions(backend: ConformanceBackend) -> None:
    repository = backend.repository
    session = await repository.create(backend.create_options(id="one"))
    entry_id = await session.append_message(create_user_message("persisted"))
    metadata = await session.get_metadata()

    listed = await repository.list()
    assert len(listed) == 1
    assert listed[0].id == metadata.id
    assert listed[0].created_at == metadata.created_at
    assert listed[0].parent_session_id == metadata.parent_session_id
    reopened = await repository.open(metadata)
    assert await entry_ids(reopened.find_entries()) == [entry_id]
    await assert_rejects_with_code(repository.create(backend.create_options(id="one")), "already_exists")


async def test_deletes_sessions_idempotently(backend: ConformanceBackend) -> None:
    repository = backend.repository
    session = await repository.create(backend.create_options(id="one"))
    metadata = await session.get_metadata()

    await repository.delete(metadata)
    await assert_rejects_with_code(repository.open(metadata), "not_found")
    await repository.delete(metadata)


async def test_forks_one_branch_with_selected_facts_and_no_records(backend: ConformanceBackend) -> None:
    repository = backend.repository
    source = await repository.create(backend.create_options(id="source"))
    root = await source.append_message(create_user_message("root"))
    shared = await source.append_message(create_assistant_message("shared"))
    await source.create_lane("thread", shared)
    thread_child = await source.view("thread").append_message(create_user_message("thread"))
    main_child = await source.append_message(create_user_message("main"))
    await source.set_name("Source")
    await source.set_label(shared, "copied")
    await source.set_label(thread_child, "excluded")
    await source.append_record(operation_started("run", lane="main", kind="run"))
    await source.append_record(
        UsageRecord(
            id="source-usage",
            lane="main",
            cause="adjustment",
            usage=Usage(
                input=10,
                output=5,
                cache_read=3,
                cache_write=2,
                total_tokens=20,
                cost=Cost(input=1, output=2, cache_read=3, cache_write=4, total=10),
            ),
        )
    )

    fork = await repository.fork(
        await source.get_metadata(),
        backend.fork_options(scope="branch", entry_id=main_child, position="at", id="branch-fork"),
    )

    assert await entry_ids(fork.find_entries(EntryQuery(order="oldestFirst"))) == [root, shared, main_child]
    lanes = await fork.get_lanes()
    assert [(pointer.lane, pointer.leaf_id) for pointer in lanes] == [("main", main_child)]
    assert await fork.get_name() == "Source"
    assert await fork.get_label(shared) == "copied"
    assert await fork.get_label(thread_child) is None
    assert await fork.find_records() == []
    stats = await fork.get_stats()
    assert (stats.message_count, stats.cached_tokens, stats.uncached_tokens, stats.total_tokens, stats.cost_total) == (
        3,
        0,
        0,
        0,
        0,
    )
    await fork.append_message(create_user_message("after fork"))
    assert (await fork.get_stats()).message_count == 4
    metadata = await fork.get_metadata()
    assert (metadata.id, metadata.parent_session_id) == ("branch-fork", "source")


async def test_forks_a_complete_tree_with_lanes_and_facts(backend: ConformanceBackend) -> None:
    repository = backend.repository
    source = await repository.create(backend.create_options(id="source"))
    root = await source.append_message(create_user_message("root"))
    await source.create_lane("thread", root)
    main_child = await source.append_message(create_user_message("main"))
    thread_child = await source.view("thread").append_message(create_user_message("thread"))
    await source.set_label(thread_child, "thread-tip")

    fork = await repository.fork(await source.get_metadata(), backend.fork_options(scope="tree", id="tree-fork"))
    assert await entry_ids(fork.find_entries(EntryQuery(order="oldestFirst"))) == [root, main_child, thread_child]
    lanes = await fork.get_lanes()
    assert [(pointer.lane, pointer.leaf_id) for pointer in lanes] == [
        ("main", main_child),
        ("thread", thread_child),
    ]
    assert await fork.get_label(thread_child) == "thread-tip"
    assert (await fork.get_stats()).message_count == 3
    log = await fork.get_log()
    lane_items = [(item.seq, item.lane, item.leaf_id) for item in log if item.kind == "lane"]
    assert lane_items == [(4, "main", main_child), (5, "thread", thread_child)]


async def test_forks_before_an_entry_without_modifying_the_source(backend: ConformanceBackend) -> None:
    repository = backend.repository
    source = await repository.create(backend.create_options(id="source"))
    root = await source.append_message(create_user_message("root"))
    tail = await source.append_message(create_user_message("tail"))
    fork = await repository.fork(await source.get_metadata(), backend.fork_options(entry_id=tail, id="fork"))

    assert await entry_ids(fork.find_entries(EntryQuery(order="oldestFirst"))) == [root]
    assert await fork.get_leaf_id() == root
    assert await source.get_leaf_id() == tail

    before_default_target = await repository.fork(
        await source.get_metadata(), backend.fork_options(position="before", id="before-default-target")
    )
    assert await entry_ids(before_default_target.find_entries(EntryQuery(order="oldestFirst"))) == [root]
    assert await before_default_target.get_leaf_id() == root

    at_default_target = await repository.fork(
        await source.get_metadata(), backend.fork_options(position="at", id="at-default-target")
    )
    assert await entry_ids(at_default_target.find_entries(EntryQuery(order="oldestFirst"))) == [root, tail]
    assert await at_default_target.get_leaf_id() == tail

    await assert_rejects_with_code(
        repository.fork(await source.get_metadata(), backend.fork_options(entry_id="missing")),
        "invalid_fork_target",
    )


async def test_validates_the_default_fork_target(backend: ConformanceBackend) -> None:
    repository = backend.repository
    source = await repository.create(backend.create_options(id="source-with-custom-leaf"))
    await source.append_custom_entry("not-a-message")

    await assert_rejects_with_code(
        repository.fork(await source.get_metadata(), backend.fork_options(id="fork")), "invalid_fork_target"
    )


# --------------------------------------------------------------------------
# local constructors mirroring conformance.ts inline object literals
# --------------------------------------------------------------------------


def _operation_finished(*, id: str, lane: str, run_id: str, outcome: str) -> Any:
    return OperationFinishedRecord(id=id, lane=lane, run_id=run_id, outcome=outcome)


def _step_attempt(*, id: str, lane: str, run_id: str, step: str, attempt: int, result_entry_id: str) -> Any:
    return StepAttemptRecord(
        id=id, lane=lane, run_id=run_id, step=step, attempt=attempt, result_entry_id=result_entry_id
    )


def _compaction_entry(*, id: str, summary: str, tokens_before: int) -> Any:
    return CompactionEntry(id=id, summary=summary, retained_tail=[], tokens_before=tokens_before)


# --------------------------------------------------------------------------
# `describe("Session with in-memory storage")` from
# `packages/agent/test/harness/session/memory.test.ts`
# --------------------------------------------------------------------------


async def test_uses_one_injectable_id_generator_across_lane_views() -> None:
    counter = {"next": 0}

    class SequentialIdGenerator:
        def next(self) -> str:
            counter["next"] += 1
            return f"generated-{counter['next']}"

    session = Session(
        InMemorySessionStorage(SessionMetadata(id="session", created_at=1)),
        id_generator=SequentialIdGenerator(),
    )
    main_id = await session.append_custom_entry("note")
    await session.create_lane("thread", main_id)
    thread_id = await session.view("thread").append_custom_entry("note")

    assert main_id == "generated-1"
    assert thread_id == "generated-2"
