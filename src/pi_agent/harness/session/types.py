"""Session/transcript entry and record model.

Python port of `packages/agent/src/harness/session/types.ts`. The TypeScript
source models "new" (caller-supplied) and "full" (storage-assigned) shapes of
the same object with a compile-time-only `Omit<TEntry, "parentId" | "seq" |
"timestamp">` (`ProvisionedEntry`) / `Omit<TRecord, "seq" | "timestamp">`
(`NewRecord`); at runtime both are the exact same plain object, just observed
before or after storage fills in the omitted fields. Python has no
compile-time-only view, so this port uses one dataclass per entry/record
variant with the storage-assigned fields defaulted (`parent_id=None`,
`seq=0`, `timestamp=0`); `ProvisionedEntry`/`NewRecord` are aliases of
`Entry`/`LaneRecord` documenting that call sites simply omit those fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from pi_ai.types import Usage

from ...types import AgentMessage

JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]

SessionStopReason = Literal["stop", "length", "toolUse", "error", "aborted", "deferred"]


class IdGenerator(Protocol):
    def next(self) -> str: ...


# --------------------------------------------------------------------------
# Entries
# --------------------------------------------------------------------------


@dataclass(kw_only=True)
class MessageEntry:
    id: str
    message: AgentMessage
    terminate: bool | None = None
    parent_id: str | None = None
    seq: int = 0
    timestamp: int = 0
    type: Literal["message"] = "message"


@dataclass(kw_only=True)
class ModelChangeEntry:
    id: str
    provider: str
    model_id: str
    parent_id: str | None = None
    seq: int = 0
    timestamp: int = 0
    type: Literal["model_change"] = "model_change"


@dataclass(kw_only=True)
class ThinkingLevelEntry:
    id: str
    thinking_level: str
    parent_id: str | None = None
    seq: int = 0
    timestamp: int = 0
    type: Literal["thinking_level_change"] = "thinking_level_change"


@dataclass(kw_only=True)
class ActiveToolsEntry:
    id: str
    active_tool_names: list[str] = field(default_factory=list)
    parent_id: str | None = None
    seq: int = 0
    timestamp: int = 0
    type: Literal["active_tools_change"] = "active_tools_change"


@dataclass(kw_only=True)
class CompactionEntry:
    id: str
    summary: str
    retained_tail: list[AgentMessage] = field(default_factory=list)
    tokens_before: int = 0
    details: Any = None
    usage: Usage | None = None
    parent_id: str | None = None
    seq: int = 0
    timestamp: int = 0
    type: Literal["compaction"] = "compaction"


@dataclass(kw_only=True)
class BranchSummaryEntry:
    id: str
    from_id: str
    summary: str
    details: Any = None
    usage: Usage | None = None
    parent_id: str | None = None
    seq: int = 0
    timestamp: int = 0
    type: Literal["branch_summary"] = "branch_summary"


@dataclass(kw_only=True)
class CustomEntry:
    id: str
    custom_type: str
    data: Any = None
    parent_id: str | None = None
    seq: int = 0
    timestamp: int = 0
    type: Literal["custom"] = "custom"


Entry = (
    MessageEntry
    | ModelChangeEntry
    | ThinkingLevelEntry
    | ActiveToolsEntry
    | CompactionEntry
    | BranchSummaryEntry
    | CustomEntry
)
"""Discriminated union of every entry kind. Dispatch on `.type`."""

ProvisionedEntry = Entry
"""Alias documenting a caller-supplied entry before `parent_id`/`seq`/`timestamp`
are storage-assigned. See module docstring."""

ENTRY_TYPES: frozenset[str] = frozenset(
    {
        "message",
        "model_change",
        "thinking_level_change",
        "active_tools_change",
        "compaction",
        "branch_summary",
        "custom",
    }
)


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------


@dataclass(kw_only=True)
class RunIntent:
    """Normalized caller input before `before_run`; kept for suspended operations and `before_resume`."""

    original_prompt: list[AgentMessage] = field(default_factory=list)
    """Captured nextRun items, then the prompt, then before_run injections."""
    initial_messages: list[ProvisionedEntry] = field(default_factory=list)
    system_prompt_override: str | None = None
    resume_data: dict[str, JsonValue] | None = None
    kind: Literal["run"] = "run"


@dataclass(kw_only=True)
class CompactionIntent:
    result_entry_id: str
    custom_instructions: str | None = None
    kind: Literal["compaction"] = "compaction"


@dataclass(kw_only=True)
class NavigationIntent:
    target_id: str | None
    summarize: bool
    custom_instructions: str | None = None
    label: str | None = None
    summary_entry_id: str | None = None
    kind: Literal["navigation"] = "navigation"


OperationIntent = RunIntent | CompactionIntent | NavigationIntent
OPERATION_KINDS: frozenset[str] = frozenset({"run", "compaction", "navigation"})


@dataclass(kw_only=True)
class OperationStartedRecord:
    id: str
    lane: str
    source_leaf_id: str | None
    intent: OperationIntent
    seq: int = 0
    timestamp: int = 0
    type: Literal["operation_started"] = "operation_started"


@dataclass(kw_only=True)
class AbortRequestedRecord:
    id: str
    lane: str
    run_id: str
    seq: int = 0
    timestamp: int = 0
    type: Literal["abort_requested"] = "abort_requested"


@dataclass(kw_only=True)
class OperationFinishedError:
    code: str
    message: str


@dataclass(kw_only=True)
class OperationFinishedRecord:
    id: str
    lane: str
    run_id: str
    outcome: Literal["completed", "aborted", "failed", "declined"]
    error: OperationFinishedError | None = None
    seq: int = 0
    timestamp: int = 0
    type: Literal["operation_finished"] = "operation_finished"


CompactionReason = Literal["manual", "threshold", "overflow"]


@dataclass(kw_only=True)
class StepAttemptRecord:
    id: str
    lane: str
    run_id: str
    step: Literal["assistant", "branch_summary", "compaction"]
    attempt: int
    result_entry_id: str
    compaction_reason: CompactionReason | None = None
    """Persists why compaction summary generation started so recovery resumes the same work.
    Present only when `step == "compaction"`."""
    seq: int = 0
    timestamp: int = 0
    type: Literal["step_attempt"] = "step_attempt"


@dataclass(kw_only=True)
class ToolStartedRecord:
    id: str
    lane: str
    run_id: str
    assistant_entry_id: str
    tool_index: int
    tool_call_id: str
    tool_name: str
    effective_args: dict[str, Any]
    result_entry_id: str
    replay: Literal["never", "safe"]
    seq: int = 0
    timestamp: int = 0
    type: Literal["tool_started"] = "tool_started"


@dataclass(kw_only=True)
class QueueEnqueuedRecord:
    id: str
    lane: str
    queue: Literal["steer", "followUp", "nextRun"]
    target: ProvisionedEntry
    run_id: str | None = None
    """Required for "steer"/"followUp"; absent for "nextRun"."""
    seq: int = 0
    timestamp: int = 0
    type: Literal["queue_enqueued"] = "queue_enqueued"


@dataclass(kw_only=True)
class QueueCancelledRecord:
    id: str
    lane: str
    entry_id: str
    run_id: str | None = None
    seq: int = 0
    timestamp: int = 0
    type: Literal["queue_cancelled"] = "queue_cancelled"


@dataclass(kw_only=True)
class WriteDeferredRecord:
    id: str
    lane: str
    run_id: str
    target: ProvisionedEntry
    seq: int = 0
    timestamp: int = 0
    type: Literal["write_deferred"] = "write_deferred"


UsageCause = Literal["assistant", "compaction", "branch_summary", "deferred_fetch", "tool", "hook", "adjustment"]


@dataclass(kw_only=True)
class UsageRecord:
    id: str
    lane: str
    usage: Usage
    cause: UsageCause
    run_id: str | None = None
    entry_id: str | None = None
    attempt: int | None = None
    """Present for cause in ("assistant", "compaction", "branch_summary", "deferred_fetch")."""
    stop_reason: SessionStopReason | None = None
    """Present for cause in ("assistant", "compaction", "branch_summary", "deferred_fetch")."""
    tool_call_id: str | None = None
    """Present only for cause == "tool"."""
    details: JsonValue | None = None
    """Present only for cause == "adjustment"."""
    seq: int = 0
    timestamp: int = 0
    type: Literal["usage"] = "usage"


LaneRecord = (
    OperationStartedRecord
    | AbortRequestedRecord
    | OperationFinishedRecord
    | StepAttemptRecord
    | ToolStartedRecord
    | QueueEnqueuedRecord
    | QueueCancelledRecord
    | WriteDeferredRecord
    | UsageRecord
)
"""Discriminated union of every lane record kind. Dispatch on `.type`."""

NewRecord = LaneRecord
"""Alias documenting a caller-supplied record before `seq`/`timestamp` are
storage-assigned. See module docstring."""

RECORD_TYPES: frozenset[str] = frozenset(
    {
        "operation_started",
        "abort_requested",
        "operation_finished",
        "step_attempt",
        "tool_started",
        "queue_enqueued",
        "queue_cancelled",
        "write_deferred",
        "usage",
    }
)


# --------------------------------------------------------------------------
# Queries
# --------------------------------------------------------------------------

EntryOrder = Literal["newestFirst", "oldestFirst"]


@dataclass(kw_only=True)
class EntryCursor:
    after_seq: int


@dataclass(kw_only=True)
class EntryQuery:
    type: str | None = None
    custom_type: str | None = None
    """For `type == "custom"`."""
    order: EntryOrder | None = None
    """Default `"newestFirst"`."""
    limit: int | None = None
    cursor: EntryCursor | None = None


@dataclass(kw_only=True)
class BranchBounds:
    """Bounds of a branch scan. Default: the whole path, leaf to root."""

    start: str | None = None
    """Default: the view's lane leaf."""
    stop_at_type: str | None = None
    """Scan ends after the first match, inclusive."""
    stop_at_id: str | None = None


@dataclass(kw_only=True)
class RecordQuery:
    lane: str | None = None
    """Exact lane match. `None` to query every lane."""
    type: str | None = None
    """Exact record discriminant match. `None` to query every record type."""
    run_id: str | None = None
    """Operation identity. Matches `OperationStartedRecord.id` and the `run_id`
    property of operation-owned records. Records without an operation identity
    do not match."""
    operation_kind: str | None = None
    """Exact operation intent kind. Valid only with `type == "operation_started"`."""
    after_seq: int | None = None
    """Exclusive chronological lower bound: `seq > after_seq`, regardless of order."""
    order: EntryOrder | None = None
    """Default `"newestFirst"`."""
    limit: int | None = None
    """Positive maximum number of matching records."""


@dataclass(kw_only=True)
class SessionMetadata:
    id: str
    created_at: int
    parent_session_id: str | None = None


@dataclass(kw_only=True)
class SessionStats:
    message_count: int = 0
    cached_tokens: int = 0
    uncached_tokens: int = 0
    total_tokens: int = 0
    cost_total: float = 0.0


@dataclass(kw_only=True)
class LanePointer:
    lane: str
    leaf_id: str | None


@dataclass(kw_only=True)
class LogEntryItem:
    seq: int
    entry: Entry
    kind: Literal["entry"] = "entry"


@dataclass(kw_only=True)
class LogRecordItem:
    seq: int
    record: LaneRecord
    kind: Literal["record"] = "record"


@dataclass(kw_only=True)
class LogLaneItem:
    seq: int
    lane: str
    leaf_id: str | None
    kind: Literal["lane"] = "lane"


@dataclass(kw_only=True)
class LogNameFactItem:
    seq: int
    name: str | None
    kind: Literal["fact"] = "fact"
    fact: Literal["name"] = "name"


@dataclass(kw_only=True)
class LogLabelFactItem:
    seq: int
    target_id: str
    label: str | None
    kind: Literal["fact"] = "fact"
    fact: Literal["label"] = "label"


LogItem = LogEntryItem | LogRecordItem | LogLaneItem | LogNameFactItem | LogLabelFactItem


@dataclass(kw_only=True)
class LogOptions:
    after_seq: int | None = None
    limit: int | None = None


@dataclass(kw_only=True)
class SessionCreateOptions:
    id: str | None = None
    parent_session_id: str | None = None


@dataclass(kw_only=True)
class ForkOptions:
    """Fork scope, target, and destination.

    TypeScript models scope/target and destination as two separate
    intersected types: `ForkOptions` (`{ scope?: "branch"; entryId?; position? }
    | { scope: "tree" }`) and `TCreateOptions` (`SessionCreateOptions`, plus
    backend-specific fields such as `cwd`). Python has no intersection types,
    so this single dataclass carries both: `entry_id`/`position` are
    meaningful only when `scope != "tree"` (the default, "branch", scope);
    `id`/`parent_session_id` are the destination session's create options.
    """

    scope: Literal["branch", "tree"] = "branch"
    entry_id: str | None = None
    position: Literal["before", "at"] | None = None
    id: str | None = None
    parent_session_id: str | None = None


class SessionErrorCode:
    NOT_FOUND: Literal["not_found"] = "not_found"
    ALREADY_EXISTS: Literal["already_exists"] = "already_exists"
    INVALID_ENTRY: Literal["invalid_entry"] = "invalid_entry"
    INVALID_PAYLOAD: Literal["invalid_payload"] = "invalid_payload"
    INVALID_LANE: Literal["invalid_lane"] = "invalid_lane"
    INVALID_QUERY: Literal["invalid_query"] = "invalid_query"
    INVALID_FORK_TARGET: Literal["invalid_fork_target"] = "invalid_fork_target"
    STORAGE: Literal["storage"] = "storage"


SessionErrorCodeLiteral = Literal[
    "not_found",
    "already_exists",
    "invalid_entry",
    "invalid_payload",
    "invalid_lane",
    "invalid_query",
    "invalid_fork_target",
    "storage",
]


class SessionError(Exception):
    def __init__(self, code: SessionErrorCodeLiteral, message: str, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.code = code
        if cause is not None:
            self.__cause__ = cause


class SessionTree(Protocol):
    async def get_leaf_id(self) -> str | None: ...
    async def get_entry(self, id: str) -> Entry | None: ...
    async def get_stats(self) -> SessionStats: ...

    async def get_name(self) -> str | None: ...
    async def set_name(self, name: str | None) -> None: ...
    async def get_label(self, target_id: str) -> str | None: ...
    async def set_label(self, target_id: str, label: str | None) -> None: ...

    async def find_entries(self, query: EntryQuery | None = None) -> list[Entry]: ...
    async def find_entry(self, query: EntryQuery | None = None) -> Entry | None: ...

    async def find_entries_on_branch(
        self, start: str | None = None, query: EntryQuery | None = None, bounds: BranchBounds | None = None
    ) -> list[Entry]: ...
    async def find_entry_on_branch(
        self, start: str | None = None, query: EntryQuery | None = None, bounds: BranchBounds | None = None
    ) -> Entry | None: ...

    async def append_message(self, message: AgentMessage) -> str: ...
    async def append_custom_entry(self, custom_type: str, data: Any = None) -> str: ...


class SessionStorage(Protocol):
    async def get_metadata(self) -> SessionMetadata: ...

    async def get_lanes(self) -> list[LanePointer]: ...
    async def create_lane(self, lane: str, at: str | None) -> None: ...
    async def move_lane(self, lane: str, to: str | None) -> None: ...

    async def append_entry(self, entry: ProvisionedEntry, lane: str) -> Entry: ...
    async def append_record(self, record: NewRecord) -> LaneRecord: ...

    async def get_entry(self, id: str) -> Entry | None: ...
    async def find_entries(self, query: EntryQuery | None = None) -> list[Entry]: ...
    async def find_entries_on_branch(
        self, start: str, query: EntryQuery | None = None, bounds: BranchBounds | None = None
    ) -> list[Entry]: ...
    async def find_records(self, query: RecordQuery | None = None) -> list[LaneRecord]: ...
    async def find_open_operations(self, lane: str, limit: int | None = None) -> list[OperationStartedRecord]: ...
    async def get_log(self, options: LogOptions | None = None) -> list[LogItem]: ...

    async def get_name(self) -> str | None: ...
    async def set_name(self, name: str | None) -> None: ...
    async def get_label(self, id: str) -> str | None: ...
    async def set_label(self, id: str, label: str | None) -> None: ...
    async def get_stats(self) -> SessionStats: ...


class SessionRepo(Protocol):
    async def create(self, options: SessionCreateOptions) -> Any: ...

    """Opens the session for writing and acquires any backend writer claim."""

    async def open(self, metadata: SessionMetadata) -> Any: ...

    """Lists session metadata without opening sessions or acquiring writer claims."""

    async def list(self, options: Any = None) -> list[SessionMetadata]: ...
    async def delete(self, metadata: SessionMetadata) -> None: ...
    async def fork(self, source: SessionMetadata, options: Any) -> Any: ...


__all__ = [
    "ENTRY_TYPES",
    "OPERATION_KINDS",
    "RECORD_TYPES",
    "AbortRequestedRecord",
    "ActiveToolsEntry",
    "BranchBounds",
    "BranchSummaryEntry",
    "CompactionEntry",
    "CompactionIntent",
    "CompactionReason",
    "CustomEntry",
    "Entry",
    "EntryCursor",
    "EntryOrder",
    "EntryQuery",
    "ForkOptions",
    "IdGenerator",
    "JsonValue",
    "LanePointer",
    "LaneRecord",
    "LogEntryItem",
    "LogItem",
    "LogLabelFactItem",
    "LogLaneItem",
    "LogNameFactItem",
    "LogOptions",
    "LogRecordItem",
    "MessageEntry",
    "ModelChangeEntry",
    "NavigationIntent",
    "NewRecord",
    "OperationFinishedError",
    "OperationFinishedRecord",
    "OperationIntent",
    "OperationStartedRecord",
    "ProvisionedEntry",
    "QueueCancelledRecord",
    "QueueEnqueuedRecord",
    "RecordQuery",
    "RunIntent",
    "SessionCreateOptions",
    "SessionError",
    "SessionErrorCode",
    "SessionErrorCodeLiteral",
    "SessionMetadata",
    "SessionRepo",
    "SessionStats",
    "SessionStopReason",
    "SessionStorage",
    "SessionTree",
    "StepAttemptRecord",
    "ThinkingLevelEntry",
    "ToolStartedRecord",
    "UsageCause",
    "UsageRecord",
    "WriteDeferredRecord",
]
