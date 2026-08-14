"""Tests for `pi_agent.harness.reducer`.

Ported from `packages/agent/test/harness/reducer.test.ts`. Fixture builders
mirror the TS helpers of the same name (snake_case, dataclass constructors
instead of object literals). `structuredClone`/`Object.freeze` have no direct
Python equivalent; the "does not mutate" style assertions instead compare via
`copy.deepcopy` snapshots taken before the call under test.
"""

from __future__ import annotations

import copy
from dataclasses import replace

import pytest
from pi_ai.types import (
    AssistantMessage,
    Cost,
    DeferredHandle,
    TextContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)

from pi_agent.harness.reducer import (
    EffectiveLaneConfiguration,
    LaneModelConfig,
    LaneReductionInput,
    OperationTargets,
    RecordLogCorruption,
    RecordLogCorruptionReason,
    RecordLogSlice,
    reduce_lane_state,
    validate_record_log,
)
from pi_agent.harness.session.types import (
    AbortRequestedRecord,
    ActiveToolsEntry,
    CompactionIntent,
    Entry,
    LaneRecord,
    ModelChangeEntry,
    NavigationIntent,
    OperationFinishedRecord,
    OperationStartedRecord,
    QueueCancelledRecord,
    QueueEnqueuedRecord,
    RunIntent,
    SessionStopReason,
    StepAttemptRecord,
    ThinkingLevelEntry,
    ToolStartedRecord,
    UsageRecord,
    WriteDeferredRecord,
)
from pi_agent.harness.session.types import (
    BranchSummaryEntry as BranchSummaryEntryType,
)
from pi_agent.harness.session.types import (
    CompactionEntry as CompactionEntryType,
)
from pi_agent.harness.session.types import (
    MessageEntry as MessageEntryType,
)

usage = Usage(input=1, output=1, cache_read=0, cache_write=0, total_tokens=2, cost=Cost())


def user_message(text: str) -> UserMessage:
    return UserMessage(role="user", content=text, timestamp=1)


def assistant_message(content: list, stop_reason: str = "stop") -> AssistantMessage:
    return AssistantMessage(
        role="assistant",
        content=content,
        api="openai-responses",
        provider="openai",
        model="test-model",
        usage=usage,
        stop_reason=stop_reason,
        timestamp=1,
        deferred=(
            DeferredHandle(provider="openai", model_id="test-model", api="openai-responses", id="deferred-1")
            if stop_reason == "deferred"
            else None
        ),
    )


def tool_result_message(tool_call_id: str = "call-1", tool_name: str = "tool-1") -> ToolResultMessage:
    return ToolResultMessage(
        role="toolResult",
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        content=[TextContent(text="result")],
        is_error=False,
        timestamp=1,
    )


def message_target(id: str, message: UserMessage | AssistantMessage | ToolResultMessage) -> MessageEntryType:
    return MessageEntryType(id=id, message=message)


def persisted_entry(target: Entry, seq: int, parent_id: str | None = None) -> Entry:
    return replace(target, parent_id=parent_id, seq=seq, timestamp=seq)


def run_started(seq: int = 1, *, id: str = "run-1", initial_messages: list | None = None) -> OperationStartedRecord:
    return OperationStartedRecord(
        id=id,
        lane="main",
        seq=seq,
        timestamp=seq,
        source_leaf_id=None,
        intent=RunIntent(original_prompt=[], initial_messages=initial_messages or []),
    )


def compaction_started(seq: int, result_entry_id: str = "compaction-1") -> OperationStartedRecord:
    return OperationStartedRecord(
        id="compact-1",
        lane="main",
        seq=seq,
        timestamp=seq,
        source_leaf_id="source",
        intent=CompactionIntent(result_entry_id=result_entry_id),
    )


def navigation_started(seq: int, summary_entry_id: str = "summary-1") -> OperationStartedRecord:
    return OperationStartedRecord(
        id="navigate-1",
        lane="main",
        seq=seq,
        timestamp=seq,
        source_leaf_id="source",
        intent=NavigationIntent(target_id="target", summarize=True, summary_entry_id=summary_entry_id),
    )


def attempt(
    seq: int,
    run_id: str,
    step: str,
    attempt_number: int,
    result_entry_id: str,
    compaction_reason: str | None = None,
) -> StepAttemptRecord:
    return StepAttemptRecord(
        id=f"attempt-{seq}",
        lane="main",
        seq=seq,
        timestamp=seq,
        run_id=run_id,
        step=step,
        attempt=attempt_number,
        result_entry_id=result_entry_id,
        compaction_reason=(compaction_reason or "manual") if step == "compaction" else None,
    )


def abort_requested(seq: int, run_id: str = "run-1") -> AbortRequestedRecord:
    return AbortRequestedRecord(id=f"abort-{seq}", lane="main", seq=seq, timestamp=seq, run_id=run_id)


def operation_finished(seq: int, run_id: str = "run-1", outcome: str = "completed") -> OperationFinishedRecord:
    return OperationFinishedRecord(
        id=f"finish-{seq}", lane="main", seq=seq, timestamp=seq, run_id=run_id, outcome=outcome
    )


def tool_started(seq: int, **overrides) -> ToolStartedRecord:
    return ToolStartedRecord(
        id=f"tool-start-{seq}",
        lane="main",
        seq=seq,
        timestamp=seq,
        run_id="run-1",
        assistant_entry_id=overrides.get("assistant_entry_id", "assistant-tools"),
        tool_index=overrides.get("tool_index", 0),
        tool_call_id=overrides.get("tool_call_id", "call-1"),
        tool_name=overrides.get("tool_name", "tool-1"),
        effective_args={},
        result_entry_id=overrides.get("result_entry_id", "tool-result-1"),
        replay="never",
    )


def queue_enqueued(seq: int, target: Entry | None = None, queue: str = "steer") -> QueueEnqueuedRecord:
    target = target if target is not None else message_target("queue-1", user_message("queued"))
    return QueueEnqueuedRecord(
        id=f"queue-{seq}",
        lane="main",
        seq=seq,
        timestamp=seq,
        target=target,
        queue=queue,
        run_id=None if queue == "nextRun" else "run-1",
    )


def queue_cancelled(seq: int, entry_id: str = "queue-1", run_id: str | None = "run-1") -> QueueCancelledRecord:
    return QueueCancelledRecord(
        id=f"cancel-{seq}", lane="main", seq=seq, timestamp=seq, entry_id=entry_id, run_id=run_id
    )


def write_deferred(seq: int, target: Entry | None = None) -> WriteDeferredRecord:
    target = target if target is not None else message_target("write-1", user_message("deferred write"))
    return WriteDeferredRecord(id=f"write-{seq}", lane="main", seq=seq, timestamp=seq, run_id="run-1", target=target)


def usage_record(
    seq: int, result_entry_id: str, stop_reason: SessionStopReason = "error", attempt_number: int = 1
) -> UsageRecord:
    return UsageRecord(
        id=f"usage-{seq}",
        lane="main",
        seq=seq,
        timestamp=seq,
        cause="assistant",
        run_id="run-1",
        entry_id=result_entry_id,
        attempt=attempt_number,
        stop_reason=stop_reason,
        usage=usage,
    )


def compaction_entry(id: str, seq: int) -> CompactionEntryType:
    return CompactionEntryType(id=id, parent_id=None, seq=seq, timestamp=seq, summary="summary", tokens_before=10)


def branch_summary_entry(id: str, seq: int) -> BranchSummaryEntryType:
    return BranchSummaryEntryType(
        id=id, parent_id="target", seq=seq, timestamp=seq, from_id="source", summary="summary"
    )


def recovery_slice(records: list[LaneRecord], entries: list[Entry] | None = None) -> RecordLogSlice:
    entries = entries or []
    finished = {record.run_id for record in records if record.type == "operation_finished"}
    open_operations = sorted(
        (record for record in records if record.type == "operation_started" and record.id not in finished),
        key=lambda record: -record.seq,
    )
    return RecordLogSlice(lane="main", open_operations=open_operations, records=list(records), entries=entries)


defaults = EffectiveLaneConfiguration(
    model=LaneModelConfig(provider="default-provider", model_id="default-model"),
    thinking_level="off",
    active_tool_names=["default-tool"],
)


def reduction_input(
    records: list[LaneRecord],
    own_entries: list[Entry] | None = None,
    *,
    entries: list[Entry] | None = None,
    configuration_entries: list[Entry] | None = None,
    leaf_id: str | None = "__unset__",
    defaults: EffectiveLaneConfiguration = defaults,
) -> LaneReductionInput:
    own_entries = own_entries or []
    slice_ = recovery_slice(records, [*own_entries, *(entries or [])])
    resolved_leaf_id = (own_entries[-1].id if own_entries else None) if leaf_id == "__unset__" else leaf_id
    return LaneReductionInput(
        lane=slice_.lane,
        open_operations=slice_.open_operations,
        records=slice_.records,
        entries=slice_.entries,
        leaf_id=resolved_leaf_id,
        own_entries=own_entries,
        configuration_entries=configuration_entries or [],
        defaults=defaults,
    )


def expect_corruption(input_: RecordLogSlice, reason: RecordLogCorruptionReason) -> None:
    with pytest.raises(RecordLogCorruption) as exc_info:
        validate_record_log(input_)
    assert exc_info.value.reason == reason


assistant_tools_entry = persisted_entry(
    message_target(
        "assistant-tools",
        assistant_message([ToolCall(id="call-1", name="tool-1", arguments={})], "toolUse"),
    ),
    3,
)


# --------------------------------------------------------------------------
# record-log validity: corruption cases
# --------------------------------------------------------------------------

corruption_cases = [
    (
        "multiple operations are open",
        "multiple_open_operations",
        recovery_slice([run_started(1), run_started(2, id="run-2")]),
    ),
    (
        "a record references an operation that does not exist",
        "unknown_operation",
        recovery_slice([abort_requested(1, "missing")]),
    ),
    (
        "a record follows its operation finish",
        "record_after_finish",
        recovery_slice([run_started(1), operation_finished(2), abort_requested(3)]),
    ),
    (
        "attempt numbers skip within one assistant step",
        "non_consecutive_attempt",
        recovery_slice(
            [
                run_started(1),
                attempt(2, "run-1", "assistant", 1, "assistant-1"),
                attempt(3, "run-1", "assistant", 3, "assistant-2"),
            ]
        ),
    ),
    (
        "a non-compaction attempt carries compactionReason",
        "invalid_compaction_reason",
        recovery_slice(
            [run_started(1), replace(attempt(2, "run-1", "assistant", 1, "assistant-1"), compaction_reason="manual")]
        ),
    ),
    (
        "a compaction attempt omits compactionReason",
        "invalid_compaction_reason",
        recovery_slice(
            [
                run_started(1),
                replace(attempt(2, "run-1", "compaction", 1, "compaction-1"), compaction_reason=None),
            ]
        ),
    ),
    (
        "steering is enqueued after abort",
        "queue_after_abort",
        recovery_slice([run_started(1), abort_requested(2), queue_enqueued(3)]),
    ),
    (
        "a queue cancellation has no enqueue",
        "invalid_queue_cancellation",
        recovery_slice([run_started(1), queue_cancelled(2)]),
    ),
    (
        "a queue cancellation targets an entry that exists",
        "invalid_queue_cancellation",
        recovery_slice(
            [run_started(1), queue_enqueued(2), queue_cancelled(4)],
            [persisted_entry(message_target("queue-1", user_message("queued")), 3)],
        ),
    ),
    (
        "structural attempts disagree on resultEntryId",
        "inconsistent_step",
        recovery_slice(
            [
                run_started(1),
                attempt(2, "run-1", "compaction", 1, "compaction-1", "threshold"),
                attempt(3, "run-1", "compaction", 2, "compaction-2", "threshold"),
            ]
        ),
    ),
    (
        "structural attempts disagree on compactionReason",
        "inconsistent_step",
        recovery_slice(
            [
                run_started(1),
                attempt(2, "run-1", "compaction", 1, "compaction-1", "threshold"),
                attempt(3, "run-1", "compaction", 2, "compaction-1", "overflow"),
            ]
        ),
    ),
    (
        "tool_started does not match the assistant tool call",
        "tool_call_mismatch",
        recovery_slice(
            [run_started(1), tool_started(4, tool_call_id="different-call")],
            [assistant_tools_entry],
        ),
    ),
    (
        "two tool_started records share an invocation identity",
        "duplicate_tool_invocation",
        recovery_slice(
            [
                run_started(1),
                tool_started(4),
                replace(tool_started(5, result_entry_id="tool-result-2"), id="tool-start-duplicate"),
            ],
            [assistant_tools_entry],
        ),
    ),
    (
        "a provisioned id exists with different content",
        "provisioned_entry_mismatch",
        recovery_slice(
            [run_started(1, initial_messages=[message_target("prompt-1", user_message("expected"))])],
            [persisted_entry(message_target("prompt-1", user_message("different")), 2)],
        ),
    ),
    (
        "a deferred assistant message has no handle",
        "invalid_deferred_handle",
        recovery_slice(
            [run_started(1)],
            [
                persisted_entry(
                    message_target("assistant-deferred", replace(assistant_message([], "deferred"), deferred=None)),
                    2,
                )
            ],
        ),
    ),
]


@pytest.mark.parametrize("name,reason,input_", corruption_cases, ids=[case[0] for case in corruption_cases])
def test_rejects_corruption_case(name: str, reason: RecordLogCorruptionReason, input_: RecordLogSlice) -> None:
    expect_corruption(input_, reason)


def test_does_not_mutate_its_bounded_recovery_inputs() -> None:
    target = message_target("prompt-1", user_message("hello"))
    start = run_started(1, initial_messages=[target])
    entry = persisted_entry(target, 2)
    input_ = RecordLogSlice(lane="main", open_operations=[start], records=[start], entries=[entry])

    before_records = copy.deepcopy(input_.records)
    before_entries = copy.deepcopy(input_.entries)

    assert validate_record_log(input_) is None
    assert input_.records == before_records
    assert input_.entries == before_entries


# --------------------------------------------------------------------------
# valid section 6 durable prefixes
# --------------------------------------------------------------------------


def _valid_prefixes(trace: str, actions: list[tuple[str, LaneRecord | Entry]]) -> list[tuple[str, RecordLogSlice]]:
    """`actions` is a list of ("record"|"entry", value) pairs, mirroring the TS
    `DurableAction` union."""
    cases = []
    for index in range(len(actions)):
        prefix = actions[: index + 1]
        records = [value for kind, value in prefix if kind == "record"]
        entries = [value for kind, value in prefix if kind == "entry"]
        cases.append((f"{trace} after action {index + 1}", recovery_slice(records, entries)))
    return cases


prompt_target = message_target("prompt-1", user_message("fix the bug"))
assistant_tool_target = message_target(
    "assistant-tools",
    assistant_message([ToolCall(id="call-1", name="tool-1", arguments={})], "toolUse"),
)
tool_result_target = message_target("tool-result-1", tool_result_message())
assistant_final_target = message_target("assistant-final", assistant_message([TextContent(text="done")]))

valid_prefix_cases: list[tuple[str, RecordLogSlice]] = [
    *_valid_prefixes(
        "one-tool run X1-X5",
        [
            ("record", run_started(1, initial_messages=[prompt_target])),
            ("entry", persisted_entry(prompt_target, 2)),
            ("record", attempt(3, "run-1", "assistant", 1, "assistant-tools")),
            ("entry", persisted_entry(assistant_tool_target, 4, "prompt-1")),
            ("record", tool_started(5)),
            ("entry", persisted_entry(tool_result_target, 6, "assistant-tools")),
            ("record", attempt(7, "run-1", "assistant", 1, "assistant-final")),
            ("entry", persisted_entry(assistant_final_target, 8, "tool-result-1")),
            ("record", operation_finished(9)),
        ],
    ),
    *_valid_prefixes(
        "assistant retry",
        [
            ("record", run_started(1)),
            ("record", attempt(2, "run-1", "assistant", 1, "assistant-attempt-1")),
            ("record", usage_record(3, "assistant-attempt-1")),
            ("record", attempt(4, "run-1", "assistant", 2, "assistant-attempt-2")),
            ("record", usage_record(5, "assistant-attempt-2", "stop", 2)),
            (
                "entry",
                persisted_entry(message_target("assistant-attempt-2", assistant_message([TextContent(text="ok")])), 6),
            ),
        ],
    ),
    *_valid_prefixes(
        "terminal assistant failure",
        [
            ("record", run_started(1)),
            ("record", attempt(2, "run-1", "assistant", 1, "assistant-error")),
            (
                "entry",
                persisted_entry(
                    message_target("assistant-error", replace(assistant_message([], "error"), error_message="failed")),
                    3,
                ),
            ),
            ("record", operation_finished(4, "run-1", "failed")),
        ],
    ),
    *_valid_prefixes(
        "overflow compaction and retry",
        [
            ("record", run_started(1)),
            ("record", attempt(2, "run-1", "assistant", 1, "discarded-overflow")),
            ("record", usage_record(3, "discarded-overflow", "length")),
            ("record", attempt(4, "run-1", "compaction", 1, "overflow-compaction", "overflow")),
            ("entry", compaction_entry("overflow-compaction", 5)),
            ("record", attempt(6, "run-1", "assistant", 1, "assistant-after-compaction")),
            (
                "entry",
                persisted_entry(
                    message_target("assistant-after-compaction", assistant_message([TextContent(text="fits")])),
                    7,
                ),
            ),
        ],
    ),
    *_valid_prefixes(
        "steering acceptance and consumption",
        [
            ("record", run_started(1)),
            ("record", queue_enqueued(2)),
            ("entry", persisted_entry(message_target("queue-1", user_message("queued")), 3)),
        ],
    ),
    *_valid_prefixes(
        "queue cancellation",
        [
            ("record", run_started(1)),
            ("record", queue_enqueued(2)),
            ("record", queue_cancelled(3)),
        ],
    ),
    *_valid_prefixes(
        "deferred write acceptance and application",
        [
            ("record", run_started(1)),
            ("record", write_deferred(2)),
            ("entry", persisted_entry(message_target("write-1", user_message("deferred write")), 3)),
        ],
    ),
    *_valid_prefixes(
        "abort during a tool",
        [
            ("record", run_started(1)),
            ("record", attempt(2, "run-1", "assistant", 1, "assistant-tools")),
            ("entry", persisted_entry(assistant_tool_target, 3)),
            ("record", tool_started(4)),
            ("record", abort_requested(5)),
            (
                "entry",
                persisted_entry(
                    message_target(
                        "tool-result-1",
                        replace(
                            tool_result_message(),
                            content=[TextContent(text="interrupted")],
                            is_error=True,
                        ),
                    ),
                    6,
                ),
            ),
        ],
    ),
    *_valid_prefixes(
        "threshold auto-compaction",
        [
            ("record", run_started(1)),
            ("record", attempt(2, "run-1", "compaction", 1, "threshold-compaction", "threshold")),
            ("entry", compaction_entry("threshold-compaction", 3)),
            ("record", attempt(4, "run-1", "assistant", 1, "assistant-after-threshold")),
        ],
    ),
    *_valid_prefixes(
        "manual compaction",
        [
            ("record", compaction_started(1)),
            ("record", attempt(2, "compact-1", "compaction", 1, "compaction-1", "manual")),
            ("entry", compaction_entry("compaction-1", 3)),
            ("record", operation_finished(4, "compact-1")),
        ],
    ),
    *_valid_prefixes(
        "move-first navigation summary",
        [
            ("record", navigation_started(1)),
            ("record", attempt(2, "navigate-1", "branch_summary", 1, "summary-1")),
            ("entry", branch_summary_entry("summary-1", 3)),
            ("record", operation_finished(4, "navigate-1")),
        ],
    ),
    *_valid_prefixes(
        "blocked tool without an intent record",
        [
            ("record", run_started(1)),
            ("record", attempt(2, "run-1", "assistant", 1, "assistant-tools")),
            ("entry", persisted_entry(assistant_tool_target, 3)),
            (
                "entry",
                persisted_entry(
                    message_target(
                        "blocked-result",
                        replace(tool_result_message(), content=[TextContent(text="blocked")], is_error=True),
                    ),
                    4,
                ),
            ),
        ],
    ),
    *_valid_prefixes(
        "idle next-run cancellation",
        [
            ("record", queue_enqueued(1, message_target("next-1", user_message("later")), "nextRun")),
            ("record", queue_cancelled(2, "next-1", None)),
        ],
    ),
    *_valid_prefixes(
        "next-run enqueue after abort",
        [
            ("record", run_started(1)),
            ("record", abort_requested(2)),
            ("record", queue_enqueued(3, message_target("next-1", user_message("later")), "nextRun")),
        ],
    ),
    *_valid_prefixes(
        "deferred write applied during abort reconciliation",
        [
            ("record", run_started(1)),
            ("record", write_deferred(2)),
            ("record", abort_requested(3)),
            ("entry", persisted_entry(message_target("write-1", user_message("deferred write")), 4)),
        ],
    ),
    *_valid_prefixes(
        "accepted steering killed by abort",
        [
            ("record", run_started(1)),
            ("record", queue_enqueued(2)),
            ("record", abort_requested(3)),
        ],
    ),
    *_valid_prefixes(
        "compaction retry",
        [
            ("record", run_started(1)),
            ("record", attempt(2, "run-1", "compaction", 1, "threshold-compaction", "threshold")),
            ("record", attempt(3, "run-1", "compaction", 2, "threshold-compaction", "threshold")),
            ("entry", compaction_entry("threshold-compaction", 4)),
        ],
    ),
    *_valid_prefixes(
        "hook-supplied manual compaction",
        [
            ("record", compaction_started(1)),
            ("entry", compaction_entry("compaction-1", 2)),
            ("record", operation_finished(3, "compact-1")),
        ],
    ),
    *_valid_prefixes(
        "hook-supplied navigation summary",
        [
            ("record", navigation_started(1)),
            ("entry", branch_summary_entry("summary-1", 2)),
            ("record", operation_finished(3, "navigate-1")),
        ],
    ),
    *_valid_prefixes(
        "deferred provider suspension and redemption",
        [
            ("record", run_started(1)),
            ("record", attempt(2, "run-1", "assistant", 1, "assistant-deferred")),
            ("entry", persisted_entry(message_target("assistant-deferred", assistant_message([], "deferred")), 3)),
            (
                "entry",
                persisted_entry(
                    message_target("assistant-redeemed", assistant_message([TextContent(text="ready")])), 4
                ),
            ),
        ],
    ),
    *_valid_prefixes(
        "abort of a deferred provider request",
        [
            ("record", run_started(1)),
            ("record", attempt(2, "run-1", "assistant", 1, "assistant-deferred")),
            ("entry", persisted_entry(message_target("assistant-deferred", assistant_message([], "deferred")), 3)),
            ("record", abort_requested(4)),
        ],
    ),
]


@pytest.mark.parametrize("name,input_", valid_prefix_cases, ids=[case[0] for case in valid_prefix_cases])
def test_accepts_valid_prefix(name: str, input_: RecordLogSlice) -> None:
    assert validate_record_log(input_) is None


# --------------------------------------------------------------------------
# lane-state reduction
# --------------------------------------------------------------------------


def test_reduces_an_idle_lane_to_pending_next_run_input_and_default_configuration() -> None:
    pending = message_target("next-pending", user_message("pending"))
    cancelled = message_target("next-cancelled", user_message("cancelled"))
    consumed = message_target("next-consumed", user_message("consumed"))
    input_ = reduction_input(
        [
            queue_enqueued(1, pending, "nextRun"),
            queue_enqueued(2, cancelled, "nextRun"),
            queue_cancelled(3, cancelled.id, None),
            queue_enqueued(4, consumed, "nextRun"),
        ],
        [],
        entries=[persisted_entry(consumed, 5)],
        leaf_id="idle-leaf",
    )

    result = reduce_lane_state(input_)
    assert result.lane_state.lane == "main"
    assert result.lane_state.leaf_id == "idle-leaf"
    assert result.lane_state.operation is None
    assert result.lane_state.pending_next_run == [pending]
    assert result.effective_configuration == defaults
    assert result.terminal_failure is None


def test_folds_persisted_configuration_over_copied_defaults_in_sequence() -> None:
    configuration_entries = [
        ModelChangeEntry(
            id="model-change",
            parent_id=None,
            seq=1,
            timestamp=1,
            provider="persisted-provider",
            model_id="persisted-model",
        ),
        ThinkingLevelEntry(id="thinking-change", parent_id="model-change", seq=2, timestamp=2, thinking_level="high"),
        ActiveToolsEntry(
            id="tools-change", parent_id="thinking-change", seq=3, timestamp=3, active_tool_names=["persisted-tool"]
        ),
    ]
    input_ = reduction_input([], [], configuration_entries=configuration_entries)

    result = reduce_lane_state(input_)
    assert result.effective_configuration == EffectiveLaneConfiguration(
        model=LaneModelConfig(provider="persisted-provider", model_id="persisted-model"),
        thinking_level="high",
        active_tool_names=["persisted-tool"],
    )
    assert input_.defaults == defaults


def test_applies_committed_operation_owned_configuration_after_the_anchor() -> None:
    assistant = persisted_entry(
        message_target(
            "assistant-config",
            replace(
                assistant_message([TextContent(text="response")]),
                provider="response-provider",
                model="response-model",
            ),
        ),
        2,
    )
    tools = ActiveToolsEntry(
        id="operation-tools", parent_id=assistant.id, seq=3, timestamp=3, active_tool_names=["operation-tool"]
    )
    result = reduce_lane_state(reduction_input([run_started(1)], [assistant, tools]))

    assert result.effective_configuration == EffectiveLaneConfiguration(
        model=LaneModelConfig(provider="response-provider", model_id="response-model"),
        thinking_level="off",
        active_tool_names=["operation-tool"],
    )


def test_keeps_captured_next_run_input_with_the_open_run_instead_of_pending_next_run() -> None:
    captured = message_target("next-captured", user_message("captured"))
    later = message_target("next-later", user_message("later"))
    start = run_started(2, initial_messages=[captured])

    result = reduce_lane_state(
        reduction_input([queue_enqueued(1, captured, "nextRun"), start, queue_enqueued(3, later, "nextRun")])
    )

    assert result.lane_state.pending_next_run == [later]
    assert result.lane_state.operation is not None
    assert result.lane_state.operation.missing_initial_messages == [captured]


def test_derives_missing_input_queues_deferred_writes_and_the_unfinished_attempt() -> None:
    missing_prompt = message_target("prompt-missing", user_message("missing"))
    committed_prompt = message_target("prompt-committed", user_message("committed"))
    steer = message_target("steer-pending", user_message("steer"))
    consumed_follow_up = message_target("follow-consumed", user_message("follow"))
    next_run = message_target("next-run", user_message("next"))
    pending_write = message_target("write-pending", user_message("write"))
    applied_write = message_target("write-applied", user_message("applied"))
    start = run_started(1, initial_messages=[missing_prompt, committed_prompt])
    committed_prompt_entry = persisted_entry(committed_prompt, 2)
    consumed_follow_up_entry = persisted_entry(consumed_follow_up, 6, committed_prompt.id)
    applied_write_entry = persisted_entry(applied_write, 9, consumed_follow_up.id)
    input_ = reduction_input(
        [
            start,
            queue_enqueued(3, steer),
            queue_enqueued(4, consumed_follow_up, "followUp"),
            queue_enqueued(5, next_run, "nextRun"),
            write_deferred(7, pending_write),
            write_deferred(8, applied_write),
            attempt(10, start.id, "assistant", 1, "assistant-pending"),
        ],
        [committed_prompt_entry, consumed_follow_up_entry, applied_write_entry],
    )

    result = reduce_lane_state(input_)
    assert result.lane_state.pending_next_run == [next_run]
    operation = result.lane_state.operation
    assert operation is not None
    assert operation.id == start.id
    assert operation.kind == "run"
    assert operation.aborting is False
    assert operation.missing_initial_messages == [missing_prompt]
    assert operation.pending_steer == [steer]
    assert operation.pending_follow_up == []
    assert operation.pending_writes == [pending_write]
    assert operation.step is not None
    assert operation.step.kind == "assistant"
    assert operation.step.attempts == 1
    assert operation.step.result_entry_id == "assistant-pending"
    assert operation.newest_own is not None
    assert operation.newest_own.entry_id == applied_write.id
    assert operation.newest_own.type == "message"
    assert operation.newest_own.role == "user"


def test_kills_steer_and_follow_up_queues_on_abort_while_preserving_writes_and_next_run_input() -> None:
    steer = message_target("steer-aborted", user_message("steer"))
    follow_up = message_target("follow-aborted", user_message("follow"))
    next_run = message_target("next-after-abort", user_message("next"))
    pending_write = message_target("write-after-abort", user_message("write"))
    input_ = reduction_input(
        [
            run_started(1),
            queue_enqueued(2, steer),
            queue_enqueued(3, follow_up, "followUp"),
            queue_enqueued(4, next_run, "nextRun"),
            write_deferred(5, pending_write),
            abort_requested(6),
        ]
    )

    result = reduce_lane_state(input_)
    assert result.lane_state.pending_next_run == [next_run]
    operation = result.lane_state.operation
    assert operation is not None
    assert operation.aborting is True
    assert operation.pending_steer == []
    assert operation.pending_follow_up == []
    assert operation.pending_writes == [pending_write]


unfinished_step_cases = [
    (
        "assistant",
        attempt(2, "run-1", "assistant", 1, "result"),
        {"kind": "assistant", "attempts": 1, "result_entry_id": "result", "compaction_reason": None},
    ),
    (
        "compaction",
        attempt(2, "run-1", "compaction", 1, "result", "overflow"),
        {"kind": "compaction", "attempts": 1, "result_entry_id": "result", "compaction_reason": "overflow"},
    ),
    (
        "branch summary",
        attempt(2, "run-1", "branch_summary", 1, "result"),
        {"kind": "branch_summary", "attempts": 1, "result_entry_id": "result", "compaction_reason": None},
    ),
]


@pytest.mark.parametrize("name,record,expected", unfinished_step_cases, ids=[case[0] for case in unfinished_step_cases])
def test_reduces_an_unfinished_step(name: str, record: StepAttemptRecord, expected: dict) -> None:
    result = reduce_lane_state(reduction_input([run_started(1), record]))
    step = result.lane_state.operation.step
    assert step is not None
    assert step.kind == expected["kind"]
    assert step.attempts == expected["attempts"]
    assert step.result_entry_id == expected["result_entry_id"]
    assert step.compaction_reason == expected["compaction_reason"]


def test_closes_the_newest_attempt_only_when_its_provisioned_result_exists() -> None:
    target = message_target("result", assistant_message([TextContent(text="done")]))
    result = reduce_lane_state(
        reduction_input([run_started(1), attempt(2, "run-1", "assistant", 1, target.id)], [persisted_entry(target, 3)])
    )
    assert result.lane_state.operation.step is None


def test_ignores_unfulfilled_result_ids_from_earlier_attempts() -> None:
    target = message_target("attempt-2-result", assistant_message([TextContent(text="done")]))
    result = reduce_lane_state(
        reduction_input(
            [
                run_started(1),
                attempt(2, "run-1", "assistant", 1, "attempt-1-result"),
                attempt(3, "run-1", "assistant", 2, target.id),
            ],
            [persisted_entry(target, 4)],
        )
    )
    assert result.lane_state.operation.step is None


def test_reduces_tool_batch_state_at_x1() -> None:
    records = [run_started(1), attempt(2, "run-1", "assistant", 1, "assistant-tools")]
    reduction = reduce_lane_state(reduction_input(records, [assistant_tools_entry]))
    tool_batch = reduction.lane_state.operation.tool_batch
    assert tool_batch is not None
    assert tool_batch.assistant_entry_id == assistant_tools_entry.id
    assert tool_batch.truncated is False
    assert tool_batch.unresolved is True
    call = tool_batch.calls[0]
    assert call.tool_index == 0
    assert call.tool_call.id == "call-1"
    assert call.tool_call.name == "tool-1"
    assert call.result_exists is False
    assert (call.started is not None) == any(record.type == "tool_started" for record in records)


def test_reduces_tool_batch_state_at_x3() -> None:
    records = [run_started(1), attempt(2, "run-1", "assistant", 1, "assistant-tools"), tool_started(4)]
    reduction = reduce_lane_state(reduction_input(records, [assistant_tools_entry]))
    tool_batch = reduction.lane_state.operation.tool_batch
    assert tool_batch is not None
    assert tool_batch.unresolved is True
    call = tool_batch.calls[0]
    assert call.result_exists is False
    assert (call.started is not None) == any(record.type == "tool_started" for record in records)


def test_reduces_tool_batch_state_at_x5() -> None:
    records = [run_started(1), attempt(2, "run-1", "assistant", 1, "assistant-tools"), tool_started(4)]
    result_entry = replace(persisted_entry(tool_result_target, 5, assistant_tools_entry.id), terminate=True)
    reduction = reduce_lane_state(reduction_input(records, [assistant_tools_entry, result_entry]))
    tool_batch = reduction.lane_state.operation.tool_batch
    assert tool_batch is not None
    assert tool_batch.unresolved is False
    call = tool_batch.calls[0]
    assert call.result_exists is True
    assert call.terminate is True
    assert (call.started is not None) == any(record.type == "tool_started" for record in records)


def test_does_not_resolve_a_tool_batch_from_a_deferred_write_tool_result() -> None:
    assistant = persisted_entry(assistant_tool_target, 3)
    written_result = message_target("written-tool-result", tool_result_message())
    result = reduce_lane_state(
        reduction_input(
            [run_started(1), attempt(2, "run-1", "assistant", 1, assistant.id), write_deferred(4, written_result)],
            [assistant, persisted_entry(written_result, 5, assistant.id)],
        )
    )
    tool_batch = result.lane_state.operation.tool_batch
    assert tool_batch.calls[0].result_exists is False
    assert tool_batch.unresolved is True


def test_matches_blocked_results_without_tool_start_records_and_preserves_source_order() -> None:
    assistant = persisted_entry(
        message_target(
            "assistant-two-tools",
            assistant_message(
                [
                    ToolCall(id="call-1", name="tool-1", arguments={}),
                    ToolCall(id="call-2", name="tool-2", arguments={}),
                ],
                "toolUse",
            ),
        ),
        3,
    )
    blocked = persisted_entry(
        message_target(
            "blocked-result",
            replace(tool_result_message("call-1", "tool-1"), content=[TextContent(text="blocked")], is_error=True),
        ),
        4,
        assistant.id,
    )
    second_start = tool_started(
        5,
        assistant_entry_id=assistant.id,
        tool_index=1,
        tool_call_id="call-2",
        tool_name="tool-2",
        result_entry_id="call-2-result",
    )
    result = reduce_lane_state(
        reduction_input(
            [run_started(1), attempt(2, "run-1", "assistant", 1, assistant.id), second_start],
            [assistant, blocked],
        )
    )

    calls = result.lane_state.operation.tool_batch.calls
    assert calls[0].tool_index == 0
    assert calls[0].tool_call.id == "call-1"
    assert calls[0].result_exists is True
    assert calls[1].tool_index == 1
    assert calls[1].tool_call.id == "call-2"
    assert calls[1].started == second_start
    assert calls[1].result_exists is False


def test_marks_a_length_stopped_tool_batch_as_truncated_without_resolving_it() -> None:
    truncated = persisted_entry(
        message_target(
            "assistant-truncated",
            assistant_message([ToolCall(id="call-1", name="tool-1", arguments={})], "length"),
        ),
        3,
    )
    result = reduce_lane_state(
        reduction_input([run_started(1), attempt(2, "run-1", "assistant", 1, truncated.id)], [truncated])
    )
    tool_batch = result.lane_state.operation.tool_batch
    assert tool_batch.truncated is True
    assert tool_batch.unresolved is True


def test_detects_an_unredeemed_deferred_handle_only_at_the_operation_tail() -> None:
    deferred_message = assistant_message([], "deferred")
    deferred_entry = persisted_entry(message_target("assistant-deferred", deferred_message), 3)
    pending = reduce_lane_state(
        reduction_input([run_started(1), attempt(2, "run-1", "assistant", 1, deferred_entry.id)], [deferred_entry])
    )
    assert pending.lane_state.operation.deferred == deferred_message.deferred

    successor = persisted_entry(
        message_target("assistant-ready", assistant_message([TextContent(text="ready")])),
        4,
        deferred_entry.id,
    )
    redeemed = reduce_lane_state(
        reduction_input(
            [run_started(1), attempt(2, "run-1", "assistant", 1, deferred_entry.id)], [deferred_entry, successor]
        )
    )
    assert redeemed.lane_state.operation.deferred is None


def test_derives_step_terminal_failure_provenance() -> None:
    records = [run_started(1), attempt(2, "run-1", "assistant", 1, "assistant-error")]
    own_entries = [
        persisted_entry(
            message_target("assistant-error", replace(assistant_message([], "error"), error_message="failed")), 3
        )
    ]
    result = reduce_lane_state(reduction_input(records, own_entries))
    assert result.terminal_failure is not None
    assert result.terminal_failure.source == "step"


def test_derives_deferred_fetch_terminal_failure_provenance() -> None:
    records = [run_started(1), attempt(2, "run-1", "assistant", 1, "assistant-deferred")]
    own_entries = [
        persisted_entry(message_target("assistant-deferred", assistant_message([], "deferred")), 3),
        persisted_entry(
            message_target("deferred-error", replace(assistant_message([], "error"), error_message="expired")),
            4,
            "assistant-deferred",
        ),
    ]
    result = reduce_lane_state(reduction_input(records, own_entries))
    assert result.terminal_failure is not None
    assert result.terminal_failure.source == "deferred_fetch"


def test_derives_deferred_fetch_usage_record_terminal_failure_provenance() -> None:
    records = [
        run_started(1),
        UsageRecord(
            id="deferred-usage",
            lane="main",
            seq=3,
            timestamp=3,
            cause="deferred_fetch",
            run_id="run-1",
            entry_id="deferred-error",
            attempt=1,
            stop_reason="error",
            usage=usage,
        ),
    ]
    own_entries = [
        persisted_entry(
            message_target("deferred-error", replace(assistant_message([], "error"), error_message="expired")), 2
        )
    ]
    result = reduce_lane_state(reduction_input(records, own_entries))
    assert result.terminal_failure is not None
    assert result.terminal_failure.source == "deferred_fetch"


def test_does_not_classify_an_error_shaped_deferred_write_as_terminal_failure() -> None:
    target = message_target("written-error", replace(assistant_message([], "error"), error_message="note"))
    entry = persisted_entry(target, 3)
    result = reduce_lane_state(reduction_input([run_started(1), write_deferred(2, target)], [entry]))
    assert result.terminal_failure is None


structural_target_cases = [
    ("manual compaction result", [compaction_started(1)], [], OperationTargets(result=False)),
    (
        "completed manual compaction result",
        [compaction_started(1)],
        [compaction_entry("compaction-1", 2)],
        OperationTargets(result=True),
    ),
    ("missing navigation summary", [navigation_started(1)], [], OperationTargets(summary=False)),
    (
        "navigation summary",
        [navigation_started(1)],
        [branch_summary_entry("summary-1", 2)],
        OperationTargets(summary=True),
    ),
]


@pytest.mark.parametrize(
    "name,records,entries,expected",
    structural_target_cases,
    ids=[case[0] for case in structural_target_cases],
)
def test_derives_structural_target_state(
    name: str, records: list[LaneRecord], entries: list[Entry], expected: OperationTargets
) -> None:
    result = reduce_lane_state(reduction_input(records, entries))
    # TypeScript uses `toEqual`, so the unrelated target key must stay unset.
    assert result.lane_state.operation.targets == expected


def test_resets_the_overflow_guard_only_after_newer_conversational_input_is_consumed() -> None:
    initial = message_target("initial", user_message("initial"))
    steer = message_target("steer", user_message("steer"))
    start = run_started(1, initial_messages=[initial])
    initial_entry = persisted_entry(initial, 2)
    records = [
        start,
        attempt(3, start.id, "compaction", 1, "overflow-summary", "overflow"),
        queue_enqueued(5, steer),
    ]

    used = reduce_lane_state(reduction_input(records, [initial_entry]))
    assert used.lane_state.operation.overflow_recovery_used is True

    reset = reduce_lane_state(reduction_input(records, [initial_entry, persisted_entry(steer, 6, initial.id)]))
    assert reset.lane_state.operation.overflow_recovery_used is False


def test_is_deterministic_and_does_not_mutate_or_alias_its_inputs() -> None:
    pending = message_target("next", user_message("next"))
    input_ = reduction_input([queue_enqueued(1, pending, "nextRun")])
    before = copy.deepcopy(input_)
    first = reduce_lane_state(input_)
    second = reduce_lane_state(input_)

    assert first == second
    assert input_ == before

    first.lane_state.pending_next_run[0].id = "mutated-output"
    assert input_.records[0].type == "queue_enqueued"
    assert input_.records[0].target.id == "next"
