"""The durable multi-lane agent harness facade.

Python port of `packages/agent/src/harness/agent-harness.ts`.

Upstream this module is the *declared* surface of the durable harness: the
rejection error taxonomy, the outcome/result unions, the `AgentLane`
interface, and an `AgentHarness` class that implements the configuration
accessors for real while every operation that would need the durable
run/compaction/navigation engine rejects with `HarnessNotImplemented`. That
is upstream's own state, not a gap in this port, so the port keeps the same
shape: accessors behave, operations raise `HarnessNotImplemented`.

TypeScript's `TaggedError` subclasses become `TaggedError(...)`-produced
classes from `pi_agent.harness.result`, and the outcome unions become
dataclasses carrying the same `kind` discriminant.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Protocol

from pi_ai.types import AssistantMessage, DeferredHandle, ImageContent, Model, Usage
from pi_ai.utils.retry import RetryPolicy

from ..types import AgentMessage, AgentTool, QueueMode, ThinkingLevel
from .compaction.compaction import DEFAULT_COMPACTION_SETTINGS, CompactionSettings
from .events import WatchHandle
from .result import Result, ResultNamespace, TaggedError
from .session.session import Session
from .session.types import BranchSummaryEntry, CompactionEntry, Entry, JsonValue, ProvisionedEntry, RecordQuery
from .types import AgentHarnessResources

LaneBusy = TaggedError("LaneBusy")
MissingIdentities = TaggedError("MissingIdentities")
NoActiveRun = TaggedError("NoActiveRun")
NoActiveOperation = TaggedError("NoActiveOperation")
NothingToResume = TaggedError("NothingToResume")
InvalidMessage = TaggedError("InvalidMessage")
UnknownSkill = TaggedError("UnknownSkill")
UnknownTemplate = TaggedError("UnknownTemplate")
UnknownTarget = TaggedError("UnknownTarget")
UnknownQueueItem = TaggedError("UnknownQueueItem")
LaneExists = TaggedError("LaneExists")
InvalidLane = TaggedError("InvalidLane")
NothingToCompact = TaggedError("NothingToCompact")
Closed = TaggedError("Closed")

OperationKind = Literal["run", "compaction", "navigation"]


class HarnessFault(Exception):
    """A harness invariant broke. `cause` carries the originating error."""

    def __init__(self, message: str, cause: object) -> None:
        super().__init__(message)
        self.cause = cause


class HarnessClosed(Exception):
    def __init__(self) -> None:
        super().__init__("AgentHarness was closed while the operation was active")


class HarnessNotImplemented(Exception):
    def __init__(self, operation: str) -> None:
        super().__init__(f"AgentHarness.{operation} is not implemented yet")
        self.operation = operation


@dataclass
class OperationError:
    code: str
    message: str


@dataclass
class RunCompleted:
    leaf_id: str
    final_entry_id: str
    final_message: AssistantMessage
    kind: Literal["completed"] = "completed"


@dataclass
class RunAborted:
    leaf_id: str
    final_entry_id: str
    final_message: AssistantMessage
    kind: Literal["aborted"] = "aborted"


@dataclass
class RunFailed:
    leaf_id: str
    error: OperationError
    final_entry_id: str | None = None
    final_message: AssistantMessage | None = None
    kind: Literal["failed"] = "failed"


@dataclass
class RunSuspended:
    leaf_id: str
    final_entry_id: str
    deferred: DeferredHandle
    kind: Literal["suspended"] = "suspended"


RunOutcome = RunCompleted | RunAborted | RunFailed | RunSuspended


@dataclass
class CompactionCompleted:
    leaf_id: str
    entry: CompactionEntry
    kind: Literal["completed"] = "completed"


@dataclass
class CompactionStopped:
    leaf_id: str
    kind: Literal["declined", "aborted"] = "declined"


@dataclass
class CompactionFailed:
    leaf_id: str
    error: OperationError
    kind: Literal["failed"] = "failed"


CompactionOutcome = CompactionCompleted | CompactionStopped | CompactionFailed


@dataclass
class NavigationCompleted:
    new_leaf_id: str | None
    summary_entry: BranchSummaryEntry | None = None
    kind: Literal["completed"] = "completed"


@dataclass
class NavigationStopped:
    leaf_id: str | None
    kind: Literal["declined", "aborted"] = "declined"


@dataclass
class NavigationFailed:
    leaf_id: str | None
    error: OperationError
    kind: Literal["failed"] = "failed"


NavigationOutcome = NavigationCompleted | NavigationStopped | NavigationFailed


@dataclass
class RunRecord:
    """A `runId` paired with the outcome of that run."""

    run_id: str
    outcome: RunOutcome | CompactionOutcome | NavigationOutcome


RunResult = Result[RunRecord, Any]
CompactionResult = Result[RunRecord, Any]
NavigationResult = Result[RunRecord, Any]


@dataclass
class QueuedEntry:
    entry_id: str


QueueResult = Result[QueuedEntry, Any]


@dataclass
class CancelQueuedOutcome:
    outcome: Literal["cancelled", "already_consumed", "already_cleared"]


CancelQueuedResult = Result[CancelQueuedOutcome, Any]
RecordUsageResult = Result[None, Any]


@dataclass
class AbortOutcome:
    run_id: str
    steer: list[AgentMessage] = field(default_factory=list)
    follow_up: list[AgentMessage] = field(default_factory=list)


AbortResult = Result[AbortOutcome, Any]


@dataclass
class ResumeOutcome:
    operation: OperationKind
    run_id: str
    outcome: RunOutcome | CompactionOutcome | NavigationOutcome


ResumeResult = Result[ResumeOutcome, Any]
CreateLaneResult = Result["AgentLane", Any]


@dataclass
class NavigateOptions:
    summarize: bool | None = None
    custom_instructions: str | None = None
    label: str | None = None


@dataclass
class MissingIdentitySet:
    tools: list[str] = field(default_factory=list)
    models: list[str] = field(default_factory=list)


@dataclass
class AbortingQueues:
    steer: list[AgentMessage] = field(default_factory=list)
    follow_up: list[AgentMessage] = field(default_factory=list)


@dataclass
class SuspendedOperation:
    lane: str
    kind: OperationKind
    id: str
    started_at: int
    reason: Literal["crash", "deferred"]
    missing: MissingIdentitySet = field(default_factory=MissingIdentitySet)
    prompt: list[AgentMessage] | None = None
    deferred: DeferredHandle | None = None
    aborting: AbortingQueues | None = None


@dataclass
class LaneOperation:
    id: str
    kind: OperationKind
    status: Literal["running", "suspended", "aborting"]


@dataclass
class LaneInfo:
    name: str
    leaf_id: str | None = None
    operation: LaneOperation | None = None
    suspended: SuspendedOperation | None = None


@dataclass
class QueuedItem:
    entry_id: str
    message: AgentMessage


@dataclass
class LaneQueues:
    steer: list[QueuedItem] = field(default_factory=list)
    follow_up: list[QueuedItem] = field(default_factory=list)
    next_run: list[QueuedItem] = field(default_factory=list)


@dataclass
class PendingWrite:
    id: str
    entry: ProvisionedEntry


@dataclass
class LaneSnapshot:
    lane: str
    transcript: list[Entry] = field(default_factory=list)
    leaf_id: str | None = None
    operation: LaneOperation | None = None
    queues: LaneQueues = field(default_factory=LaneQueues)
    pending_writes: list[PendingWrite] = field(default_factory=list)
    faulted: bool = False


@dataclass
class SessionSnapshot:
    lanes: list[LaneInfo] = field(default_factory=list)
    faulted: bool = False


HookName = Literal[
    "before_run",
    "before_resume",
    "before_run_end",
    "transform_context",
    "before_request",
    "before_payload",
    "after_response",
    "before_tool",
    "after_tool",
    "before_compaction",
    "before_navigation",
]


@dataclass
class AppendEntryAction:
    entry_type: str
    entry_id: str
    kind: Literal["append_entry"] = "append_entry"


@dataclass
class AppendRecordAction:
    record_type: str
    kind: Literal["append_record"] = "append_record"


@dataclass
class MoveLaneAction:
    to: str | None
    kind: Literal["move_lane"] = "move_lane"


@dataclass
class SetFactAction:
    fact: Literal["name", "label"]
    kind: Literal["set_fact"] = "set_fact"


@dataclass
class TryFinishRunAction:
    outcome: Literal["completed", "failed"]
    kind: Literal["try_finish_run"] = "try_finish_run"


@dataclass
class FinishOperationAction:
    outcome: Literal["completed", "declined", "failed", "aborted"]
    kind: Literal["finish_operation"] = "finish_operation"


@dataclass
class CommitFollowUpAction:
    kind: Literal["commit_follow_up"] = "commit_follow_up"


@dataclass
class ConsumeQueueItemAction:
    queue: Literal["steer", "followUp"]
    entry_id: str
    kind: Literal["consume_queue_item"] = "consume_queue_item"


@dataclass
class ApplyPendingWriteAction:
    entry_id: str
    kind: Literal["apply_pending_write"] = "apply_pending_write"


@dataclass
class StreamAssistantAction:
    step: Literal["assistant", "compaction", "branch_summary"]
    attempt: int
    kind: Literal["stream_assistant"] = "stream_assistant"


@dataclass
class ExecuteToolAction:
    tool_call_id: str
    tool_name: str
    kind: Literal["execute_tool"] = "execute_tool"


@dataclass
class DeferredAction:
    provider: str
    id: str
    kind: Literal["fetch_deferred", "cancel_deferred"] = "fetch_deferred"


@dataclass
class HookAction:
    name: HookName
    kind: Literal["hook"] = "hook"


@dataclass
class SleepAction:
    delay_ms: int
    kind: Literal["sleep"] = "sleep"


ActionInfo = (
    AppendEntryAction
    | AppendRecordAction
    | MoveLaneAction
    | SetFactAction
    | TryFinishRunAction
    | FinishOperationAction
    | CommitFollowUpAction
    | ConsumeQueueItemAction
    | ApplyPendingWriteAction
    | StreamAssistantAction
    | ExecuteToolAction
    | DeferredAction
    | HookAction
    | SleepAction
)

Unsubscribe = Callable[[], None]


class Hooks(Protocol):
    def on(
        self,
        name: HookName,
        handler: Callable[[object], object | Awaitable[object]],
        id: str | None = None,
    ) -> Unsubscribe: ...


class Events(Protocol):
    def on(self, type: str, listener: Callable[[object], Awaitable[None] | None]) -> Unsubscribe: ...


class UnavailableRegistry:
    """A `Hooks`/`Events` registry that always rejects, matching upstream."""

    def __init__(self, operation: str, is_closed: Callable[[], bool]) -> None:
        self._operation = operation
        self._is_closed = is_closed

    def on(self, *_args: object, **_kwargs: object) -> Unsubscribe:
        raise HarnessClosed() if self._is_closed() else HarnessNotImplemented(self._operation)


@dataclass
class HarnessTool:
    """An `AgentTool` plus the harness-only `replay` policy."""

    tool: AgentTool
    replay: Literal["never", "safe"] | None = None

    @property
    def name(self) -> str:
        return self.tool.name


Resources = AgentHarnessResources
StreamOptionsPatch = dict[str, Any]
EntryProjector = Callable[[Entry], list[AgentMessage] | Awaitable[list[AgentMessage]]]


@dataclass(kw_only=True)
class AgentHarnessOptions:
    session: Session
    models: Any
    model: Model
    thinking_level: ThinkingLevel = "off"
    active_tool_names: list[str] | None = None
    tools: list[HarnessTool] | None = None
    tool_context: object | Callable[[], object | Awaitable[object]] | None = None
    system_prompt: str | Callable[[], str | Awaitable[str]] | None = None
    resources: Resources | None = None
    stream_options: dict[str, Any] | None = None
    retry: RetryPolicy | None = None
    compaction: CompactionSettings | None = None
    steering_mode: QueueMode = "one-at-a-time"
    follow_up_mode: QueueMode = "one-at-a-time"
    tool_execution: Literal["sequential", "parallel"] = "sequential"
    drive: Literal["automatic", "manual"] = "automatic"
    to_provider_messages: Callable[[list[AgentMessage]], Any] | None = None
    entry_projectors: dict[str, EntryProjector] | None = None
    context: Any = None


class AgentLane(Protocol):
    """One durable conversation lane inside a harness session."""

    name: str

    async def get_leaf_id(self) -> str | None: ...
    async def prompt(
        self, input: str | AgentMessage | Sequence[AgentMessage], images: list[ImageContent] | None = None
    ) -> RunResult: ...
    async def skill(self, name: str, additional_instructions: str | None = None) -> RunResult: ...
    async def prompt_from_template(self, name: str, args: list[str] | None = None) -> RunResult: ...
    async def compact(self, custom_instructions: str | None = None) -> CompactionResult: ...
    async def navigate_tree(
        self, target_id: str | None, options: NavigateOptions | None = None
    ) -> NavigationResult: ...
    async def resume(self) -> ResumeResult: ...
    async def abort(self) -> AbortResult: ...
    async def steer(self, input: str | AgentMessage, images: list[ImageContent] | None = None) -> QueueResult: ...
    async def follow_up(self, input: str | AgentMessage, images: list[ImageContent] | None = None) -> QueueResult: ...
    async def next_run(self, input: str | AgentMessage, images: list[ImageContent] | None = None) -> QueueResult: ...
    async def cancel_queued(self, entry_id: str) -> CancelQueuedResult: ...
    async def record_usage(
        self, usage: Usage, entry_id: str | None = None, details: JsonValue = None
    ) -> RecordUsageResult: ...
    async def wait_for_idle(self) -> None: ...
    async def run_when_idle(self, callback: Callable[[], Awaitable[None] | None]) -> None: ...
    async def peek_action(self) -> ActionInfo | None: ...
    async def execute_action(self) -> ActionInfo | None: ...
    async def run_to_completion(self) -> None: ...
    async def get_model(self) -> Model: ...
    async def set_model(self, model: Model) -> None: ...
    async def get_thinking_level(self) -> ThinkingLevel: ...
    async def set_thinking_level(self, level: ThinkingLevel) -> None: ...
    async def get_active_tools(self) -> list[str]: ...
    async def set_active_tools(self, names: list[str]) -> None: ...
    async def watch(self) -> WatchHandle[LaneSnapshot]: ...


class AgentHarness:
    """The `main` lane of a durable harness session.

    Configuration accessors are live; every operation that needs the durable
    engine raises `HarnessNotImplemented`, exactly as upstream does today.
    """

    def __init__(self, options: AgentHarnessOptions) -> None:
        self.name = "main"
        self._durable_session = options.session
        self.session = options.session
        self.hooks = UnavailableRegistry("hooks.on", lambda: self._closed)
        self.events = UnavailableRegistry("events.on", lambda: self._closed)
        self._model = options.model
        self._thinking_level: ThinkingLevel = options.thinking_level
        tools = list(options.tools or [])
        names = options.active_tool_names
        self._active_tool_names = list(names) if names is not None else [tool.name for tool in tools]
        self._tools = tools
        source = options.resources
        self._resources = Resources(
            skills=list(source.skills) if source else [],
            prompt_templates=list(source.prompt_templates) if source else [],
        )
        self._stream_options = dict(options.stream_options or {})
        self._retry_policy = options.retry or RetryPolicy(enabled=False, max_retries=0, base_delay_ms=1000)
        self._compaction_settings = options.compaction or DEFAULT_COMPACTION_SETTINGS
        self._steering_mode: QueueMode = options.steering_mode
        self._follow_up_mode: QueueMode = options.follow_up_mode
        self._closed = False

    @staticmethod
    async def create(options: AgentHarnessOptions) -> tuple[AgentHarness, list[SuspendedOperation]]:
        """Open a harness on a session. Restoring an existing session is not implemented upstream."""
        records = await options.session.find_records(RecordQuery(limit=1))
        if records:
            raise HarnessNotImplemented("create.restore")
        return AgentHarness(options), []

    def _unavailable(self, operation: str) -> Any:
        raise HarnessClosed() if self._closed else HarnessNotImplemented(operation)

    async def get_leaf_id(self) -> str | None:
        return await self._durable_session.get_leaf_id()

    async def prompt(
        self, input: str | AgentMessage | Sequence[AgentMessage], images: list[ImageContent] | None = None
    ) -> RunResult:
        return self._unavailable("prompt")

    async def skill(self, name: str, additional_instructions: str | None = None) -> RunResult:
        return self._unavailable("skill")

    async def prompt_from_template(self, name: str, args: list[str] | None = None) -> RunResult:
        return self._unavailable("promptFromTemplate")

    async def compact(self, custom_instructions: str | None = None) -> CompactionResult:
        return self._unavailable("compact")

    async def navigate_tree(self, target_id: str | None, options: NavigateOptions | None = None) -> NavigationResult:
        return self._unavailable("navigateTree")

    async def resume(self) -> ResumeResult:
        return self._unavailable("resume")

    async def abort(self) -> AbortResult:
        return self._unavailable("abort")

    async def steer(self, input: str | AgentMessage, images: list[ImageContent] | None = None) -> QueueResult:
        return self._unavailable("steer")

    async def follow_up(self, input: str | AgentMessage, images: list[ImageContent] | None = None) -> QueueResult:
        return self._unavailable("followUp")

    async def next_run(self, input: str | AgentMessage, images: list[ImageContent] | None = None) -> QueueResult:
        return self._unavailable("nextRun")

    async def cancel_queued(self, entry_id: str) -> CancelQueuedResult:
        return self._unavailable("cancelQueued")

    async def record_usage(
        self, usage: Usage, entry_id: str | None = None, details: JsonValue = None
    ) -> RecordUsageResult:
        return self._unavailable("recordUsage")

    async def wait_for_idle(self) -> None:
        return self._unavailable("waitForIdle")

    async def run_when_idle(self, callback: Callable[[], Awaitable[None] | None]) -> None:
        return self._unavailable("runWhenIdle")

    async def peek_action(self) -> ActionInfo | None:
        return self._unavailable("peekAction")

    async def execute_action(self) -> ActionInfo | None:
        return self._unavailable("executeAction")

    async def run_to_completion(self) -> None:
        return self._unavailable("runToCompletion")

    async def get_model(self) -> Model:
        return self._model

    async def set_model(self, model: Model) -> None:
        self._model = model

    async def get_thinking_level(self) -> ThinkingLevel:
        return self._thinking_level

    async def set_thinking_level(self, level: ThinkingLevel) -> None:
        self._thinking_level = level

    async def get_active_tools(self) -> list[str]:
        return list(self._active_tool_names)

    async def set_active_tools(self, names: list[str]) -> None:
        self._active_tool_names = list(names)

    async def watch(self) -> WatchHandle[LaneSnapshot]:
        return self._unavailable("watch")

    async def lane(self, name: str) -> AgentLane | None:
        return self._unavailable("lane")

    async def create_lane(self, name: str, at: str | None) -> CreateLaneResult:
        return self._unavailable("createLane")

    async def lanes(self) -> list[LaneInfo]:
        return self._unavailable("lanes")

    async def get_tools(self) -> list[HarnessTool]:
        return list(self._tools)

    async def set_tools(self, tools: list[HarnessTool], active_names: list[str] | None = None) -> None:
        self._tools = list(tools)
        self._active_tool_names = list(active_names) if active_names is not None else [tool.name for tool in tools]

    async def get_resources(self) -> Resources:
        return Resources(
            skills=list(self._resources.skills),
            prompt_templates=list(self._resources.prompt_templates),
        )

    async def set_resources(self, resources: Resources) -> None:
        self._resources = Resources(
            skills=list(resources.skills),
            prompt_templates=list(resources.prompt_templates),
        )

    async def get_stream_options(self) -> dict[str, Any]:
        return dict(self._stream_options)

    async def set_stream_options(self, options: dict[str, Any]) -> None:
        self._stream_options = dict(options)

    async def get_retry_policy(self) -> RetryPolicy:
        return RetryPolicy(
            enabled=self._retry_policy.enabled,
            max_retries=self._retry_policy.max_retries,
            base_delay_ms=self._retry_policy.base_delay_ms,
        )

    async def set_retry_policy(self, policy: RetryPolicy) -> None:
        self._retry_policy = replace(policy)

    async def get_compaction_settings(self) -> CompactionSettings:
        return CompactionSettings(
            enabled=self._compaction_settings.enabled,
            reserve_tokens=self._compaction_settings.reserve_tokens,
            keep_recent_tokens=self._compaction_settings.keep_recent_tokens,
        )

    async def set_compaction_settings(self, settings: CompactionSettings) -> None:
        self._compaction_settings = replace(settings)

    async def get_steering_mode(self) -> QueueMode:
        return self._steering_mode

    async def set_steering_mode(self, mode: QueueMode) -> None:
        self._steering_mode = mode

    async def get_follow_up_mode(self) -> QueueMode:
        return self._follow_up_mode

    async def set_follow_up_mode(self, mode: QueueMode) -> None:
        self._follow_up_mode = mode

    async def watch_session(self) -> WatchHandle[SessionSnapshot]:
        return self._unavailable("watchSession")

    async def close(self) -> None:
        self._closed = True


__all__ = [
    "AbortOutcome",
    "AbortResult",
    "ActionInfo",
    "AgentHarness",
    "AgentHarnessOptions",
    "AgentLane",
    "AppendEntryAction",
    "AppendRecordAction",
    "ApplyPendingWriteAction",
    "CancelQueuedOutcome",
    "CancelQueuedResult",
    "Closed",
    "CommitFollowUpAction",
    "CompactionCompleted",
    "CompactionFailed",
    "CompactionOutcome",
    "CompactionResult",
    "CompactionStopped",
    "ConsumeQueueItemAction",
    "CreateLaneResult",
    "DeferredAction",
    "EntryProjector",
    "Events",
    "ExecuteToolAction",
    "FinishOperationAction",
    "HarnessClosed",
    "HarnessFault",
    "HarnessNotImplemented",
    "HarnessTool",
    "HookAction",
    "HookName",
    "Hooks",
    "InvalidLane",
    "InvalidMessage",
    "LaneBusy",
    "LaneExists",
    "LaneInfo",
    "LaneOperation",
    "LaneQueues",
    "LaneSnapshot",
    "MissingIdentities",
    "MissingIdentitySet",
    "MoveLaneAction",
    "NavigateOptions",
    "NavigationCompleted",
    "NavigationFailed",
    "NavigationOutcome",
    "NavigationResult",
    "NavigationStopped",
    "NoActiveOperation",
    "NoActiveRun",
    "NothingToCompact",
    "NothingToResume",
    "OperationError",
    "OperationKind",
    "PendingWrite",
    "QueueResult",
    "QueuedEntry",
    "QueuedItem",
    "RecordUsageResult",
    "Resources",
    "ResultNamespace",
    "ResumeOutcome",
    "ResumeResult",
    "RunAborted",
    "RunCompleted",
    "RunFailed",
    "RunOutcome",
    "RunRecord",
    "RunResult",
    "RunSuspended",
    "SessionSnapshot",
    "SetFactAction",
    "SleepAction",
    "StreamAssistantAction",
    "StreamOptionsPatch",
    "SuspendedOperation",
    "TryFinishRunAction",
    "UnavailableRegistry",
    "UnknownQueueItem",
    "UnknownSkill",
    "UnknownTarget",
    "UnknownTemplate",
    "WatchHandle",
]
