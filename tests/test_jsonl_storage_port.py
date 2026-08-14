"""Python port of `packages/agent/test/harness/session/jsonl-storage.test.ts`.

TypeScript injects a `NodeExecutionEnv` filesystem into `JsonlSessionRepo`;
this port has no such abstraction (see `jsonl/types.py`) and operates on
`pathlib.Path` directly, so the repository is built from a temp dir only. The
injected-append-failure case therefore makes the session file genuinely
unwritable instead of stubbing `appendFile`.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import pytest
from pi_agent.harness.session import (
    ActiveToolsEntry,
    BranchBounds,
    BranchSummaryEntry,
    CompactionEntry,
    CompactionIntent,
    CustomEntry,
    EntryCursor,
    EntryQuery,
    JsonlSessionCreateOptions,
    JsonlSessionRepo,
    JsonlSessionRepoOptions,
    LogEntryItem,
    MessageEntry,
    ModelChangeEntry,
    NavigationIntent,
    OperationFinishedRecord,
    QueueCancelledRecord,
    QueueEnqueuedRecord,
    RecordQuery,
    RunIntent,
    SessionError,
    SessionStats,
    StepAttemptRecord,
    ThinkingLevelEntry,
    ToolStartedRecord,
    UsageRecord,
    WriteDeferredRecord,
)
from pi_agent.harness.session.types import AbortRequestedRecord, OperationStartedRecord
from pi_ai import AssistantMessage, Cost, TextContent, ToolCall, ToolResultMessage, Usage, UserMessage

pytestmark = pytest.mark.asyncio


def create_repository(root: Path) -> JsonlSessionRepo:
    return JsonlSessionRepo(JsonlSessionRepoOptions(sessions_root=root))


def user_message(text: str) -> UserMessage:
    return UserMessage(content=[TextContent(text=text)], timestamp=1)


def create_usage(multiplier: int) -> Usage:
    return Usage(
        input=multiplier,
        output=multiplier * 2,
        cache_read=multiplier * 3,
        cache_write=multiplier * 4,
        total_tokens=multiplier * 10,
        cost=Cost(
            input=multiplier * 0.1,
            output=multiplier * 0.2,
            cache_read=multiplier * 0.3,
            cache_write=multiplier * 0.4,
            total=multiplier,
        ),
    )


async def reopen(root: Path, session: Any) -> Any:
    return await create_repository(root).open(await session.get_metadata())


async def test_round_trips_every_entry_type_and_bounded_branch_queries(tmp_path: Path) -> None:
    root = tmp_path
    session = await create_repository(root).create(JsonlSessionCreateOptions(id="entries", cwd=str(root)))
    committed: list[Any] = []
    committed.append(await session.append_entry(MessageEntry(id="message", message=user_message("question")), "main"))
    committed.append(
        await session.append_entry(
            MessageEntry(
                id="assistant-tool-call",
                message=AssistantMessage(
                    content=[
                        TextContent(text="I'll inspect it."),
                        ToolCall(id="call-1", name="read", arguments={"path": "README.md"}),
                    ],
                    api="anthropic-messages",
                    provider="anthropic",
                    model="claude-sonnet-4-5",
                    usage=create_usage(1),
                    stop_reason="toolUse",
                    timestamp=2,
                ),
            ),
            "main",
        )
    )
    committed.append(
        await session.append_entry(
            MessageEntry(
                id="tool-result",
                message=ToolResultMessage(
                    tool_call_id="call-1",
                    tool_name="read",
                    content=[TextContent(text="contents")],
                    details={"path": "README.md"},
                    usage=create_usage(2),
                    is_error=False,
                    timestamp=3,
                ),
                terminate=True,
            ),
            "main",
        )
    )
    committed.append(
        await session.append_entry(
            ModelChangeEntry(id="model", provider="anthropic", model_id="claude-sonnet-4-5"), "main"
        )
    )
    committed.append(await session.append_entry(ThinkingLevelEntry(id="thinking", thinking_level="high"), "main"))
    committed.append(
        await session.append_entry(ActiveToolsEntry(id="tools", active_tool_names=["read", "bash"]), "main")
    )
    committed.append(
        await session.append_entry(
            CompactionEntry(
                id="compaction",
                summary="summary",
                retained_tail=[user_message("retained")],
                tokens_before=123,
                details={"source": "test"},
                usage=create_usage(1),
            ),
            "main",
        )
    )
    committed.append(
        await session.append_entry(
            BranchSummaryEntry(
                id="branch-summary",
                from_id="message",
                summary="branch",
                details={"reason": "navigation"},
                usage=create_usage(2),
            ),
            "main",
        )
    )
    committed.append(
        await session.append_entry(CustomEntry(id="custom", custom_type="note", data={"nested": {"value": 1}}), "main")
    )

    restored = await reopen(root, session)
    assert await restored.find_entries(EntryQuery(order="oldestFirst")) == committed
    branch = await restored.find_entries_on_branch(bounds=BranchBounds(stop_at_type="compaction"))
    assert [entry.id for entry in branch] == ["custom", "branch-summary", "compaction"]
    page = await restored.find_entries(
        EntryQuery(order="oldestFirst", cursor=EntryCursor(after_seq=committed[5].seq), limit=2)
    )
    assert [entry.id for entry in page] == ["compaction", "branch-summary"]
    assert [entry.id for entry in await restored.find_entries(EntryQuery(custom_type="note"))] == ["custom"]
    assert await restored.get_stats() == SessionStats(
        message_count=3, cached_tokens=0, uncached_tokens=0, total_tokens=0, cost_total=0
    )

    custom = await restored.get_entry("custom")
    assert custom is not None and custom.type == "custom"
    custom.data["nested"]["value"] = 99
    log_custom = next(item for item in await restored.get_log() if item.kind == "entry" and item.entry.type == "custom")
    log_custom.entry.data["nested"]["value"] = 100

    assert await restored.get_entry("custom") == committed[-1]
    assert await restored.find_entries(EntryQuery(order="oldestFirst")) == committed


async def test_round_trips_every_record_type_recovery_projection_and_ledger_statistics(
    tmp_path: Path,
) -> None:
    root = tmp_path
    session = await create_repository(root).create(JsonlSessionCreateOptions(id="records", cwd=str(root)))
    await session.append_custom_entry("anchor")
    records: list[Any] = []

    async def append(record: Any) -> None:
        records.append(await session.append_record(record))

    await append(
        OperationStartedRecord(
            id="run",
            lane="main",
            source_leaf_id="anchor",
            intent=RunIntent(
                original_prompt=[user_message("prompt")],
                initial_messages=[MessageEntry(id="initial", message=user_message("initial"))],
                system_prompt_override="system",
                resume_data={"extension": {"version": 1}},
            ),
        )
    )
    await append(
        QueueEnqueuedRecord(
            id="steer",
            lane="main",
            queue="steer",
            run_id="run",
            target=MessageEntry(id="steer-message", message=user_message("steer")),
        )
    )
    await append(
        QueueEnqueuedRecord(
            id="follow-up",
            lane="main",
            queue="followUp",
            run_id="run",
            target=MessageEntry(id="follow-up-message", message=user_message("follow up")),
        )
    )
    await append(
        StepAttemptRecord(
            id="assistant-attempt",
            lane="main",
            run_id="run",
            step="assistant",
            attempt=1,
            result_entry_id="assistant-result",
        )
    )
    await append(
        ToolStartedRecord(
            id="tool",
            lane="main",
            run_id="run",
            assistant_entry_id="assistant-result",
            tool_index=0,
            tool_call_id="call-1",
            tool_name="read",
            effective_args={"path": "README.md"},
            result_entry_id="tool-result",
            replay="safe",
        )
    )
    await append(
        WriteDeferredRecord(
            id="deferred-write",
            lane="main",
            run_id="run",
            target=CustomEntry(id="deferred-entry", custom_type="fact", data={"value": True}),
        )
    )
    await append(
        UsageRecord(
            id="assistant-usage",
            lane="main",
            cause="assistant",
            run_id="run",
            entry_id="assistant-result",
            attempt=1,
            stop_reason="stop",
            usage=create_usage(1),
        )
    )
    await append(
        UsageRecord(
            id="deferred-usage",
            lane="main",
            cause="deferred_fetch",
            run_id="run",
            entry_id="deferred-result",
            attempt=1,
            stop_reason="deferred",
            usage=create_usage(2),
        )
    )
    await append(
        UsageRecord(
            id="tool-usage",
            lane="main",
            cause="tool",
            run_id="run",
            entry_id="tool-result",
            tool_call_id="call-1",
            usage=create_usage(3),
        )
    )
    await append(
        UsageRecord(
            id="hook-usage",
            lane="main",
            cause="hook",
            run_id="run",
            entry_id="hook-result",
            usage=create_usage(4),
        )
    )
    await append(
        UsageRecord(
            id="adjustment",
            lane="main",
            cause="adjustment",
            details={"reason": "correction"},
            usage=create_usage(5),
        )
    )
    await append(AbortRequestedRecord(id="abort", lane="main", run_id="run"))
    await append(OperationFinishedRecord(id="run-finished", lane="main", run_id="run", outcome="aborted"))
    await append(
        QueueEnqueuedRecord(
            id="next-run",
            lane="main",
            queue="nextRun",
            target=MessageEntry(id="next-message", message=user_message("next")),
        )
    )
    await append(QueueCancelledRecord(id="queue-cancelled", lane="main", entry_id="next-message"))
    await append(
        OperationStartedRecord(
            id="compaction",
            lane="main",
            source_leaf_id="anchor",
            intent=CompactionIntent(custom_instructions="short", result_entry_id="compaction-result"),
        )
    )
    await append(
        StepAttemptRecord(
            id="compaction-attempt",
            lane="main",
            run_id="compaction",
            step="compaction",
            attempt=1,
            result_entry_id="compaction-result",
            compaction_reason="manual",
        )
    )
    await append(
        OperationFinishedRecord(id="compaction-finished", lane="main", run_id="compaction", outcome="completed")
    )
    await append(
        OperationStartedRecord(
            id="navigation",
            lane="main",
            source_leaf_id="anchor",
            intent=NavigationIntent(
                target_id=None,
                summarize=True,
                custom_instructions="summarize",
                label="checkpoint",
                summary_entry_id="navigation-summary",
            ),
        )
    )
    await append(
        StepAttemptRecord(
            id="branch-attempt",
            lane="main",
            run_id="navigation",
            step="branch_summary",
            attempt=1,
            result_entry_id="navigation-summary",
        )
    )

    restored = await reopen(root, session)
    assert await restored.find_records(RecordQuery(order="oldestFirst")) == records
    started = await restored.find_records(RecordQuery(type="operation_started", operation_kind="run", limit=1))
    assert [record.id for record in started] == ["run"]
    assert [
        record.id for record in await restored.find_records(RecordQuery(run_id="compaction", order="oldestFirst"))
    ] == ["compaction", "compaction-attempt", "compaction-finished"]
    assert [
        record.id
        for record in await restored.find_records(RecordQuery(type="usage", after_seq=records[6].seq, limit=2))
    ] == ["adjustment", "hook-usage"]
    assert [record.id for record in await restored.find_open_operations("main", limit=2)] == ["navigation"]
    assert await restored.get_stats() == SessionStats(
        message_count=0, cached_tokens=45, uncached_tokens=75, total_tokens=150, cost_total=15
    )

    run_records = await restored.find_records(RecordQuery(type="operation_started", operation_kind="run"))
    assert run_records[0].intent.kind == "run"
    run_records[0].intent.original_prompt.append(user_message("mutated"))
    assert await restored.find_records(RecordQuery(order="oldestFirst")) == records


async def test_persists_concurrent_cross_lane_writes_in_shared_sequence_order(tmp_path: Path) -> None:
    root = tmp_path
    session = await create_repository(root).create(JsonlSessionCreateOptions(id="concurrent", cwd=str(root)))
    root_entry = await session.append_entry(CustomEntry(id="root", custom_type="root"), "main")
    await session.create_lane("thread", root_entry.id)

    entries = await asyncio.gather(
        session.append_entry(CustomEntry(id="main-1", custom_type="note"), "main"),
        session.append_entry(CustomEntry(id="thread-1", custom_type="note"), "thread"),
        session.append_entry(CustomEntry(id="main-2", custom_type="note"), "main"),
        session.append_entry(CustomEntry(id="thread-2", custom_type="note"), "thread"),
    )
    commit_order = [entry.id for entry in sorted(entries, key=lambda entry: entry.seq)]

    restored = await reopen(root, session)
    log = await restored.get_log()
    restored_concurrent_entries = [item.entry for item in log if item.kind == "entry" and item.entry.id != "root"]
    assert [entry.id for entry in restored_concurrent_entries] == commit_order
    assert len({entry.seq for entry in restored_concurrent_entries}) == len(entries)
    assert [item.seq for item in await restored.get_log()] == [1, 2, 3, 4, 5, 6]


async def test_rejects_non_json_payloads_without_changing_the_durable_prefix(tmp_path: Path) -> None:
    root = tmp_path
    session = await create_repository(root).create(JsonlSessionCreateOptions(id="validation", cwd=str(root)))
    metadata = await session.get_metadata()
    prefix = Path(metadata.path).read_text()
    cyclic: dict[str, Any] = {}
    cyclic["self"] = cyclic

    with pytest.raises(SessionError) as caught:
        await session.append_custom_entry("invalid", cyclic)
    assert caught.value.code == "invalid_payload"

    # TypeScript rejects an `undefined` field because `JSON.stringify` would
    # silently omit it, changing the effective arguments used during recovery.
    # Python has no `undefined`; the equivalent lossy value is a non-finite
    # float, which `json.dumps` would emit as the invalid token `NaN`.
    with pytest.raises(SessionError) as caught:
        await session.append_record(
            ToolStartedRecord(
                id="invalid-record",
                lane="main",
                run_id="run",
                assistant_entry_id="assistant",
                tool_index=0,
                tool_call_id="call",
                tool_name="read",
                effective_args={"value": float("nan")},
                result_entry_id="result",
                replay="never",
            )
        )
    assert caught.value.code == "invalid_payload"
    assert Path(metadata.path).read_text() == prefix

    restored = await reopen(root, session)
    assert await restored.get_log() == []
    valid = await restored.append_entry(CustomEntry(id="valid", custom_type="note", data={"value": 1}), "main")
    assert valid.seq == 1
    assert await (await reopen(root, restored)).get_entry("valid") == valid


async def test_does_not_advance_state_or_poison_the_write_queue_after_an_append_failure(
    tmp_path: Path,
) -> None:
    root = tmp_path
    repository = create_repository(root)
    session = await repository.create(JsonlSessionCreateOptions(id="append-failure", cwd=str(root)))
    metadata = await session.get_metadata()
    path = Path(metadata.path)

    original_mode = path.stat().st_mode
    os.chmod(path, 0o444)
    try:
        with pytest.raises(SessionError) as caught:
            await session.append_custom_entry("rejected")
        assert caught.value.code == "storage"
    finally:
        os.chmod(path, original_mode)

    assert await session.get_log() == []
    committed = await session.append_entry(CustomEntry(id="committed", custom_type="note"), "main")
    assert committed.seq == 1

    reopened = await create_repository(root).open(await session.get_metadata())
    assert await reopened.get_log() == [LogEntryItem(seq=1, entry=committed)]


async def test_module_level_listing_and_loading_match_the_repo(tmp_path):
    """Port of `listJsonlSessionMetadata` / `loadJsonlSessionStorage`.

    Upstream lifted both out of `JsonlSessionRepo` so a caller can enumerate and
    open sessions without constructing a repo (`jsonl/repo.ts:65,89`). They must
    stay the same code path as the repo methods, not a parallel implementation
    that can drift.
    """
    from pi_agent.harness.session.jsonl.repo import (
        list_jsonl_session_metadata,
        load_jsonl_session_storage,
    )

    options = JsonlSessionRepoOptions(sessions_root=str(tmp_path))
    repo = JsonlSessionRepo(options)
    await repo.create(JsonlSessionCreateOptions(cwd=str(tmp_path)))

    listed = await list_jsonl_session_metadata(options)
    via_repo = await repo.list()

    assert [m.id for m in listed] == [m.id for m in via_repo]
    assert len(listed) == 1

    storage = await load_jsonl_session_storage(options, listed[0])
    assert (await storage.get_metadata()).id == listed[0].id


async def test_loading_a_missing_session_raises_not_found(tmp_path):
    from pi_agent.harness.session.jsonl.repo import load_jsonl_session_storage

    options = JsonlSessionRepoOptions(sessions_root=str(tmp_path))
    repo = JsonlSessionRepo(options)
    await repo.create(JsonlSessionCreateOptions(cwd=str(tmp_path)))
    metadata = (await repo.list())[0]
    Path(metadata.path).unlink()

    with pytest.raises(SessionError) as excinfo:
        await load_jsonl_session_storage(options, metadata)
    assert excinfo.value.code == "not_found"
