"""Lane state reduction: reconstruct a lane's orchestration state from records/entries.

Python port of `packages/agent/src/harness/reducer.ts`.

`Guard.IsDeepEqual` (typebox) has no Python port. `_matches_provisioned_entry`
uses `dataclasses.asdict` to structurally compare two entries (after
stripping the storage-assigned `parent_id`/`seq`/`timestamp` fields), which
gives the same field-by-field structural equality typebox's deep-equality
guard provides for these plain-data dataclasses.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, replace
from typing import Literal, NoReturn

from pi_ai.types import DeferredHandle

from ..types import AgentToolCall, ThinkingLevel
from .session.types import (
    CompactionReason,
    Entry,
    LaneRecord,
    OperationIntent,
    OperationStartedRecord,
    ProvisionedEntry,
    QueueEnqueuedRecord,
    StepAttemptRecord,
    ToolStartedRecord,
)

RecordLogCorruptionReason = Literal[
    "multiple_open_operations",
    "unknown_operation",
    "record_after_finish",
    "non_consecutive_attempt",
    "invalid_compaction_reason",
    "queue_after_abort",
    "invalid_queue_cancellation",
    "inconsistent_step",
    "tool_call_mismatch",
    "duplicate_tool_invocation",
    "provisioned_entry_mismatch",
    "invalid_deferred_handle",
]
"""Machine-readable category for a contradiction in a lane's durable recovery
slice. These indicate states the single-writer record protocol cannot
produce, not ordinary operation failures or incomplete-but-recoverable
intent/result prefixes. Restore must reject such states rather than repair or
continue it; the accompanying error message supplies human-readable detail."""


class RecordLogCorruption(Exception):
    def __init__(self, reason: RecordLogCorruptionReason, message: str) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass(kw_only=True)
class RecordLogSlice:
    lane: str
    open_operations: list[OperationStartedRecord]
    records: list[LaneRecord]
    entries: list[Entry]
    """Operation-owned entries plus entries fetched directly by provisioned or referenced ids."""


@dataclass(kw_only=True)
class LaneModelConfig:
    provider: str
    model_id: str


@dataclass(kw_only=True)
class EffectiveLaneConfiguration:
    model: LaneModelConfig
    thinking_level: ThinkingLevel
    active_tool_names: list[str]


@dataclass(kw_only=True)
class TerminalFailureState:
    entry_id: str
    source: Literal["step", "deferred_fetch"]
    message: object


@dataclass(kw_only=True)
class ToolBatchCall:
    tool_index: int
    tool_call: AgentToolCall
    started: ToolStartedRecord | None = None
    result_exists: bool = False
    terminate: bool | None = None


@dataclass(kw_only=True)
class ToolBatchState:
    assistant_entry_id: str
    calls: list[ToolBatchCall]
    truncated: bool
    unresolved: bool


@dataclass(kw_only=True)
class LaneStepState:
    kind: Literal["assistant", "compaction", "branch_summary"]
    attempts: int
    result_entry_id: str
    compaction_reason: CompactionReason | None = None


@dataclass(kw_only=True)
class NewestOwnEntry:
    entry_id: str
    type: str
    role: str | None = None
    stop_reason: str | None = None


@dataclass(kw_only=True)
class OperationTargets:
    result: bool | None = None
    summary: bool | None = None


@dataclass(kw_only=True)
class LaneOperationState:
    id: str
    kind: Literal["run", "compaction", "navigation"]
    intent: OperationIntent
    aborting: bool
    step: LaneStepState | None
    tool_batch: ToolBatchState | None
    missing_initial_messages: list[Entry]
    pending_steer: list[Entry]
    pending_follow_up: list[Entry]
    pending_writes: list[Entry]
    deferred: DeferredHandle | None
    overflow_recovery_used: bool
    newest_own: NewestOwnEntry | None
    targets: OperationTargets


@dataclass(kw_only=True)
class LaneState:
    lane: str
    leaf_id: str | None
    operation: LaneOperationState | None
    pending_next_run: list[Entry]


@dataclass(kw_only=True)
class LaneReductionInput(RecordLogSlice):
    leaf_id: str | None
    own_entries: list[Entry]
    """Entries appended by the open operation, oldest first. Empty when idle."""
    configuration_entries: list[Entry]
    """Bounded effective-state lookups at the operation anchor or idle leaf, oldest first."""
    defaults: EffectiveLaneConfiguration
    """Harness option fallbacks used when no persisted value exists."""


@dataclass(kw_only=True)
class LaneReductionResult:
    lane_state: LaneState
    effective_configuration: EffectiveLaneConfiguration
    terminal_failure: TerminalFailureState | None


@dataclass
class _AttemptSeries:
    record: StepAttemptRecord


def _corrupt(reason: RecordLogCorruptionReason, message: str) -> NoReturn:
    raise RecordLogCorruption(reason, message)


def _has_run_id(record: LaneRecord) -> bool:
    return isinstance(getattr(record, "run_id", None), str)


def _entry_payload(entry: Entry) -> dict:
    payload = asdict(entry)
    payload.pop("parent_id", None)
    payload.pop("seq", None)
    payload.pop("timestamp", None)
    return payload


def _matches_provisioned_entry(entry: Entry, target: ProvisionedEntry) -> bool:
    return _entry_payload(entry) == _entry_payload(target)


def _validate_exact_provisioned_entry(entries_by_id: dict[str, Entry], target: ProvisionedEntry) -> None:
    entry = entries_by_id.get(target.id)
    if entry is not None and not _matches_provisioned_entry(entry, target):
        _corrupt(
            "provisioned_entry_mismatch", f"Provisioned entry {target.id} exists with content different from its intent"
        )


def _validate_result_entry(entries_by_id: dict[str, Entry], result_entry_id: str, matches, description: str) -> None:
    entry = entries_by_id.get(result_entry_id)
    if entry is not None and not matches(entry):
        _corrupt(
            "provisioned_entry_mismatch",
            f"Provisioned {description} entry {result_entry_id} exists with different content",
        )


def _validate_attempt_reason(record: StepAttemptRecord) -> None:
    reason = record.compaction_reason
    if record.step == "compaction":
        if reason not in ("manual", "threshold", "overflow"):
            _corrupt("invalid_compaction_reason", f"Compaction attempt {record.id} has no valid compaction reason")
    elif reason is not None:
        _corrupt("invalid_compaction_reason", f"{record.step} attempt {record.id} has a compaction reason")


def _validate_attempt_sequence(
    record: StepAttemptRecord, previous: _AttemptSeries | None, entries_by_id: dict[str, Entry]
) -> None:
    previous_record = previous.record if previous else None
    previous_result = entries_by_id.get(previous_record.result_entry_id) if previous_record else None
    continues_series = (
        previous_record is not None
        and previous_record.step == record.step
        and (previous_result is None or previous_result.seq >= record.seq)
    )
    expected_attempt = previous_record.attempt + 1 if continues_series and previous_record else 1
    if record.attempt != expected_attempt:
        _corrupt(
            "non_consecutive_attempt",
            f"{record.step} attempt {record.id} is {record.attempt}; expected {expected_attempt}",
        )
    if not continues_series or record.step == "assistant" or previous_record is None:
        return
    if record.result_entry_id != previous_record.result_entry_id:
        _corrupt("inconsistent_step", f"{record.step} attempts disagree on their result entry id")
    if record.compaction_reason != previous_record.compaction_reason:
        _corrupt("inconsistent_step", f"{record.step} attempts disagree on their compaction reason")


def _validate_attempt_result(entries_by_id: dict[str, Entry], record: StepAttemptRecord) -> None:
    if record.step == "assistant":
        _validate_result_entry(
            entries_by_id,
            record.result_entry_id,
            lambda entry: entry.type == "message" and entry.message.role == "assistant",
            "assistant result",
        )
    elif record.step == "compaction":
        _validate_result_entry(
            entries_by_id, record.result_entry_id, lambda entry: entry.type == "compaction", "compaction result"
        )
    elif record.step == "branch_summary":
        _validate_result_entry(
            entries_by_id,
            record.result_entry_id,
            lambda entry: entry.type == "branch_summary",
            "branch-summary result",
        )


def _validate_tool_start(record: ToolStartedRecord, entries_by_id: dict[str, Entry], invocations: set[str]) -> None:
    invocation = f"{record.assistant_entry_id}\x00{record.tool_index}"
    if invocation in invocations:
        _corrupt(
            "duplicate_tool_invocation",
            f"Tool invocation {record.assistant_entry_id}:{record.tool_index} is duplicated",
        )
    invocations.add(invocation)

    assistant_entry = entries_by_id.get(record.assistant_entry_id)
    if assistant_entry is None or assistant_entry.type != "message" or assistant_entry.message.role != "assistant":
        _corrupt("tool_call_mismatch", f"Tool start {record.id} does not reference an assistant entry")
    tool_calls = [content for content in assistant_entry.message.content if content.type == "toolCall"]
    tool_call = tool_calls[record.tool_index] if 0 <= record.tool_index < len(tool_calls) else None
    if tool_call is None or tool_call.id != record.tool_call_id or tool_call.name != record.tool_name:
        _corrupt("tool_call_mismatch", f"Tool start {record.id} does not match its assistant tool-call ordinal")

    _validate_result_entry(
        entries_by_id,
        record.result_entry_id,
        lambda entry: (
            entry.type == "message"
            and entry.message.role == "toolResult"
            and entry.message.tool_call_id == record.tool_call_id
            and entry.message.tool_name == record.tool_name
        ),
        "tool result",
    )


def _validate_deferred_handles(entries) -> None:
    for entry in entries:
        if (
            entry.type == "message"
            and entry.message.role == "assistant"
            and entry.message.stop_reason == "deferred"
            and not entry.message.deferred
        ):
            _corrupt("invalid_deferred_handle", f"Deferred assistant entry {entry.id} does not carry a handle")


def _validate_operation_result(entries_by_id: dict[str, Entry], record: OperationStartedRecord) -> None:
    intent = record.intent
    if intent.kind == "run":
        for target in intent.initial_messages:
            _validate_exact_provisioned_entry(entries_by_id, target)
    elif intent.kind == "compaction":
        _validate_result_entry(
            entries_by_id, intent.result_entry_id, lambda entry: entry.type == "compaction", "manual compaction"
        )
    elif intent.kind == "navigation" and intent.summary_entry_id:
        _validate_result_entry(
            entries_by_id,
            intent.summary_entry_id,
            lambda entry: entry.type == "branch_summary",
            "navigation summary",
        )


def validate_record_log(input: RecordLogSlice) -> None:
    """Validates a bounded lane recovery slice without reading or mutating session state."""
    if len(input.open_operations) > 1:
        _corrupt("multiple_open_operations", f"Lane {input.lane} has at least two open operations")

    entries_by_id: dict[str, Entry] = {entry.id: entry for entry in input.entries}
    _validate_deferred_handles(entries_by_id.values())
    starts: dict[str, OperationStartedRecord] = {}
    finished_at: dict[str, int] = {}
    aborted_at: dict[str, int] = {}
    queue_enqueues: dict[str, QueueEnqueuedRecord] = {}
    latest_attempt: dict[str, _AttemptSeries] = {}
    tool_invocations: set[str] = set()
    records = sorted(input.records, key=lambda record: record.seq)

    for record in records:
        if record.type == "operation_started":
            starts[record.id] = record
            _validate_operation_result(entries_by_id, record)
            continue

        if _has_run_id(record):
            if record.run_id not in starts:
                _corrupt("unknown_operation", f"Record {record.id} references unknown operation {record.run_id}")
            finish_seq = finished_at.get(record.run_id)
            if finish_seq is not None and record.seq > finish_seq:
                _corrupt("record_after_finish", f"Record {record.id} follows the finish of operation {record.run_id}")

        if record.type == "operation_finished":
            finished_at[record.run_id] = record.seq
        elif record.type == "abort_requested":
            aborted_at[record.run_id] = record.seq
        elif record.type == "step_attempt":
            _validate_attempt_reason(record)
            _validate_attempt_sequence(record, latest_attempt.get(record.run_id), entries_by_id)
            _validate_attempt_result(entries_by_id, record)
            latest_attempt[record.run_id] = _AttemptSeries(record=record)
        elif record.type == "tool_started":
            _validate_tool_start(record, entries_by_id, tool_invocations)
        elif record.type == "queue_enqueued":
            aborted_seq = aborted_at.get(record.run_id) if record.run_id is not None else None
            if record.queue != "nextRun" and aborted_seq is not None and record.seq > aborted_seq:
                _corrupt("queue_after_abort", f"{record.queue} item {record.target.id} was enqueued after abort")
            queue_enqueues[record.target.id] = record
            _validate_exact_provisioned_entry(entries_by_id, record.target)
        elif record.type == "queue_cancelled":
            enqueue = queue_enqueues.get(record.entry_id)
            if (
                enqueue is None
                or enqueue.seq >= record.seq
                or enqueue.run_id != record.run_id
                or record.entry_id in entries_by_id
            ):
                _corrupt(
                    "invalid_queue_cancellation", f"Queue cancellation {record.id} has no pending matching enqueue"
                )
        elif record.type == "write_deferred":
            _validate_exact_provisioned_entry(entries_by_id, record.target)
        elif record.type == "usage":
            pass


def _clone(value):
    return copy.deepcopy(value)


def _by_sequence(values):
    return sorted(values, key=lambda value: value.seq)


def _derive_effective_configuration(input: LaneReductionInput) -> EffectiveLaneConfiguration:
    configuration = _clone(input.defaults)
    entries_by_id: dict[str, Entry] = {}
    for entry in [*input.configuration_entries, *input.own_entries]:
        entries_by_id[entry.id] = entry

    for entry in _by_sequence(list(entries_by_id.values())):
        if entry.type == "model_change":
            configuration = replace(
                configuration, model=LaneModelConfig(provider=entry.provider, model_id=entry.model_id)
            )
        elif entry.type == "thinking_level_change":
            configuration = replace(configuration, thinking_level=entry.thinking_level)
        elif entry.type == "active_tools_change":
            configuration = replace(configuration, active_tool_names=list(entry.active_tool_names))
        elif entry.type == "message" and entry.message.role == "assistant":
            configuration = replace(
                configuration, model=LaneModelConfig(provider=entry.message.provider, model_id=entry.message.model)
            )
    return configuration


def _derive_newest_own(entry: Entry | None) -> NewestOwnEntry | None:
    if entry is None:
        return None
    if entry.type != "message":
        return NewestOwnEntry(entry_id=entry.id, type=entry.type)
    if entry.message.role != "assistant":
        return NewestOwnEntry(entry_id=entry.id, type=entry.type, role=entry.message.role)
    return NewestOwnEntry(
        entry_id=entry.id, type=entry.type, role=entry.message.role, stop_reason=entry.message.stop_reason
    )


def _derive_tool_batch(
    operation_id: str,
    records: list[LaneRecord],
    own_entries: list[Entry],
    entries_by_id: dict[str, Entry],
    deferred_write_ids: set[str],
) -> ToolBatchState | None:
    assistant_entry: Entry | None = None
    for entry in reversed(own_entries):
        if (
            entry.type == "message"
            and entry.message.role == "assistant"
            and any(content.type == "toolCall" for content in entry.message.content)
        ):
            assistant_entry = entry
            break
    if assistant_entry is None or assistant_entry.type != "message" or assistant_entry.message.role != "assistant":
        return None

    tool_calls = [content for content in assistant_entry.message.content if content.type == "toolCall"]
    starts: dict[int, ToolStartedRecord] = {}
    for record in records:
        if (
            record.type == "tool_started"
            and record.run_id == operation_id
            and record.assistant_entry_id == assistant_entry.id
        ):
            starts[record.tool_index] = record

    calls: list[ToolBatchCall] = []
    for tool_index, tool_call in enumerate(tool_calls):
        started = starts.get(tool_index)
        started_result = entries_by_id.get(started.result_entry_id) if started is not None else None
        blocked_result: Entry | None = None
        for entry in own_entries:
            if (
                entry.seq > assistant_entry.seq
                and entry.id not in deferred_write_ids
                and entry.type == "message"
                and entry.message.role == "toolResult"
                and entry.message.tool_call_id == tool_call.id
            ):
                blocked_result = entry
                break
        result = started_result if started_result is not None else blocked_result
        calls.append(
            ToolBatchCall(
                tool_index=tool_index,
                tool_call=_clone(tool_call),
                started=_clone(started) if started is not None else None,
                result_exists=result is not None,
                terminate=(
                    True if (result is not None and result.type == "message" and result.terminate is True) else None
                ),
            )
        )

    return ToolBatchState(
        assistant_entry_id=assistant_entry.id,
        calls=calls,
        truncated=assistant_entry.message.stop_reason == "length",
        unresolved=any(not call.result_exists for call in calls),
    )


def reduce_lane_state(input: LaneReductionInput) -> LaneReductionResult:
    """Purely reconstructs one lane's orchestration state from its bounded recovery inputs."""
    validate_record_log(input)

    records = _by_sequence(input.records)
    own_entries = _by_sequence(input.own_entries)
    entries_by_id: dict[str, Entry] = {}
    for entry in [*input.entries, *own_entries]:
        entries_by_id[entry.id] = entry
    cancelled_queue_ids = {record.entry_id for record in records if record.type == "queue_cancelled"}
    pending_queue_records = [
        record
        for record in records
        if record.type == "queue_enqueued"
        and record.target.id not in entries_by_id
        and record.target.id not in cancelled_queue_ids
    ]
    started = input.open_operations[0] if input.open_operations else None
    captured_initial_message_ids: set[str] = set()
    if started is not None and started.intent.kind == "run":
        captured_initial_message_ids = {target.id for target in started.intent.initial_messages}
    pending_next_run = [
        _clone(record.target)
        for record in pending_queue_records
        if record.queue == "nextRun" and record.target.id not in captured_initial_message_ids
    ]
    effective_configuration = _derive_effective_configuration(input)

    if started is None:
        return LaneReductionResult(
            lane_state=LaneState(
                lane=input.lane, leaf_id=input.leaf_id, operation=None, pending_next_run=pending_next_run
            ),
            effective_configuration=effective_configuration,
            terminal_failure=None,
        )

    operation_records = [
        record
        for record in records
        if (
            record.id == started.id
            if record.type == "operation_started"
            else getattr(record, "run_id", None) == started.id
        )
    ]
    aborting = any(record.type == "abort_requested" for record in operation_records)
    pending_steer = (
        []
        if aborting
        else [
            _clone(record.target)
            for record in pending_queue_records
            if record.queue == "steer" and record.run_id == started.id
        ]
    )
    pending_follow_up = (
        []
        if aborting
        else [
            _clone(record.target)
            for record in pending_queue_records
            if record.queue == "followUp" and record.run_id == started.id
        ]
    )
    pending_writes = [
        _clone(record.target)
        for record in operation_records
        if record.type == "write_deferred" and record.target.id not in entries_by_id
    ]
    missing_initial_messages = (
        [_clone(target) for target in started.intent.initial_messages if target.id not in entries_by_id]
        if started.intent.kind == "run"
        else []
    )

    step_attempts = [record for record in operation_records if record.type == "step_attempt"]
    newest_attempt = step_attempts[-1] if step_attempts else None
    step: LaneStepState | None = None
    if newest_attempt is not None and newest_attempt.result_entry_id not in entries_by_id:
        step = LaneStepState(
            kind=newest_attempt.step,
            attempts=newest_attempt.attempt,
            result_entry_id=newest_attempt.result_entry_id,
            compaction_reason=newest_attempt.compaction_reason if newest_attempt.step == "compaction" else None,
        )

    consumed_input_ids: set[str] = set()
    if started.intent.kind == "run":
        consumed_input_ids.update(target.id for target in started.intent.initial_messages)
    for record in operation_records:
        if record.type == "queue_enqueued" and record.queue != "nextRun":
            consumed_input_ids.add(record.target.id)
    newest_consumed_input_sequence = float("-inf")
    for entry_id in consumed_input_ids:
        entry = entries_by_id.get(entry_id)
        if entry is not None and entry.type == "message":
            newest_consumed_input_sequence = max(newest_consumed_input_sequence, entry.seq)
    overflow_recovery_used = any(
        record.type == "step_attempt"
        and record.step == "compaction"
        and record.compaction_reason == "overflow"
        and record.seq > newest_consumed_input_sequence
        for record in operation_records
    )

    newest_own_entry = own_entries[-1] if own_entries else None
    newest_own = _derive_newest_own(newest_own_entry)
    deferred: DeferredHandle | None = None
    if (
        newest_own_entry is not None
        and newest_own_entry.type == "message"
        and newest_own_entry.message.role == "assistant"
        and newest_own_entry.message.stop_reason == "deferred"
        and newest_own_entry.message.deferred
    ):
        deferred = _clone(newest_own_entry.message.deferred)

    targets = OperationTargets()
    if started.intent.kind == "compaction":
        targets.result = started.intent.result_entry_id in entries_by_id
    elif started.intent.kind == "navigation" and started.intent.summary_entry_id:
        targets.summary = started.intent.summary_entry_id in entries_by_id

    deferred_write_ids = {record.target.id for record in operation_records if record.type == "write_deferred"}
    terminal_failure: TerminalFailureState | None = None
    if (
        newest_own_entry is not None
        and newest_own_entry.type == "message"
        and newest_own_entry.message.role == "assistant"
        and newest_own_entry.message.stop_reason == "error"
        and newest_own_entry.id not in deferred_write_ids
    ):
        produced_by_step = any(
            record.type == "step_attempt" and record.result_entry_id == newest_own_entry.id
            for record in operation_records
        )
        previous_own_entry = own_entries[-2] if len(own_entries) >= 2 else None
        produced_by_deferred_fetch = any(
            record.type == "usage" and record.cause == "deferred_fetch" and record.entry_id == newest_own_entry.id
            for record in operation_records
        ) or (
            previous_own_entry is not None
            and previous_own_entry.type == "message"
            and previous_own_entry.message.role == "assistant"
            and previous_own_entry.message.stop_reason == "deferred"
        )
        if produced_by_step or produced_by_deferred_fetch:
            terminal_failure = TerminalFailureState(
                entry_id=newest_own_entry.id,
                source="step" if produced_by_step else "deferred_fetch",
                message=_clone(newest_own_entry.message),
            )

    return LaneReductionResult(
        lane_state=LaneState(
            lane=input.lane,
            leaf_id=input.leaf_id,
            operation=LaneOperationState(
                id=started.id,
                kind=started.intent.kind,
                intent=_clone(started.intent),
                aborting=aborting,
                step=step,
                tool_batch=_derive_tool_batch(
                    started.id, operation_records, own_entries, entries_by_id, deferred_write_ids
                ),
                missing_initial_messages=missing_initial_messages,
                pending_steer=pending_steer,
                pending_follow_up=pending_follow_up,
                pending_writes=pending_writes,
                deferred=deferred,
                overflow_recovery_used=overflow_recovery_used,
                newest_own=newest_own,
                targets=targets,
            ),
            pending_next_run=pending_next_run,
        ),
        effective_configuration=effective_configuration,
        terminal_failure=terminal_failure,
    )


__all__ = [
    "EffectiveLaneConfiguration",
    "LaneModelConfig",
    "LaneOperationState",
    "LaneReductionInput",
    "LaneReductionResult",
    "LaneState",
    "LaneStepState",
    "NewestOwnEntry",
    "OperationTargets",
    "RecordLogCorruption",
    "RecordLogCorruptionReason",
    "RecordLogSlice",
    "TerminalFailureState",
    "ToolBatchCall",
    "ToolBatchState",
    "reduce_lane_state",
    "validate_record_log",
]
