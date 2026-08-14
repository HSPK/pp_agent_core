"""Agent runtime types.

Python port of `packages/agent/src/types.ts`. Callback-shaped configuration
(``convert_to_llm``, ``before_tool_call``, ...) keeps the TypeScript contract:
hooks must not raise, and failures are encoded in the values they return.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from pi_ai import (
    AssistantMessage,
    AssistantMessageEvent,
    AssistantMessageEventStream,
    Context,
    ImageContent,
    Message,
    Model,
    SimpleStreamOptions,
    TextContent,
    Tool,
    ToolCall,
    ToolResultMessage,
    Usage,
)
from pi_ai.utils.abort import AbortSignal

ToolExecutionMode = Literal["sequential", "parallel"]
QueueMode = Literal["all", "one-at-a-time"]
ThinkingLevel = Literal["off", "minimal", "low", "medium", "high", "xhigh", "max"]

AgentToolCall = ToolCall
AgentMessage = Message
"""Union of LLM messages plus any application message type.

TypeScript extends this union through declaration merging. Python callers can
pass their own message objects; the loop only inspects ``role`` and relies on
``convert_to_llm`` to map unknown messages onto LLM messages.
"""

StreamFn = Callable[..., AssistantMessageEventStream | Awaitable[AssistantMessageEventStream]]
"""``(model, context, options) -> AssistantMessageEventStream``.

Must not raise for request/model/runtime failures: those are encoded in the
returned stream as an error event whose message has stop reason ``"error"`` or
``"aborted"``.
"""


@dataclass
class AgentToolResult:
    """Final or partial result produced by a tool."""

    content: list[TextContent | ImageContent] = field(default_factory=list)
    details: Any = None
    usage: Usage | None = None
    added_tool_names: list[str] | None = None
    terminate: bool | None = None
    """Hint to stop after the current tool batch. Honored only when every
    finalized result in the batch sets it."""


AgentToolUpdateCallback = Callable[[AgentToolResult], None]


@dataclass
class AgentTool(Tool):
    """Tool definition used by the agent runtime."""

    label: str = ""
    execute: Callable[..., Awaitable[AgentToolResult]] | None = None
    prepare_arguments: Callable[[Any], dict[str, Any]] | None = None
    execution_mode: ToolExecutionMode | None = None

    def __post_init__(self) -> None:
        if not self.label:
            self.label = self.name


@dataclass
class AgentContext:
    """Context snapshot passed into the low-level agent loop."""

    system_prompt: str = ""
    messages: list[AgentMessage] = field(default_factory=list)
    tools: list[AgentTool] | None = None


@dataclass
class BeforeToolCallResult:
    """Returned from ``before_tool_call``.

    ``block=True`` prevents execution; the loop emits an error tool result whose
    text is ``reason``.
    """

    block: bool | None = None
    reason: str | None = None
    terminate: bool | None = None


@dataclass
class AfterToolCallResult:
    """Field-by-field override of an executed tool result. No deep merge."""

    content: list[TextContent | ImageContent] | None = None
    details: Any = None
    is_error: bool | None = None
    usage: Usage | None = None
    terminate: bool | None = None
    _details_set: bool = False

    def with_details(self, details: Any) -> AfterToolCallResult:
        self.details = details
        self._details_set = True
        return self


@dataclass
class BeforeToolCallContext:
    assistant_message: AssistantMessage
    tool_call: AgentToolCall
    args: Any
    context: AgentContext


@dataclass
class AfterToolCallContext:
    assistant_message: AssistantMessage
    tool_call: AgentToolCall
    args: Any
    result: AgentToolResult
    is_error: bool
    context: AgentContext


@dataclass
class ShouldStopAfterTurnContext:
    message: AssistantMessage
    tool_results: list[ToolResultMessage]
    context: AgentContext
    new_messages: list[AgentMessage]


PrepareNextTurnContext = ShouldStopAfterTurnContext


@dataclass
class AgentLoopTurnUpdate:
    """Replacement runtime state applied before the next provider request."""

    context: AgentContext | None = None
    model: Model | None = None
    thinking_level: ThinkingLevel | None = None


@dataclass
class AgentLoopConfig(SimpleStreamOptions):
    """Configuration for one agent loop run."""

    model: Model | None = None
    convert_to_llm: Callable[[list[AgentMessage]], list[Message] | Awaitable[list[Message]]] | None = None
    transform_context: Callable[..., Awaitable[list[AgentMessage]]] | None = None
    get_api_key: Callable[[str], Any] | None = None
    should_stop_after_turn: Callable[[ShouldStopAfterTurnContext], Any] | None = None
    prepare_next_turn: Callable[[PrepareNextTurnContext], Any] | None = None
    get_steering_messages: Callable[[], Awaitable[list[AgentMessage]]] | None = None
    get_follow_up_messages: Callable[[], Awaitable[list[AgentMessage]]] | None = None
    tool_execution: ToolExecutionMode = "parallel"
    before_tool_call: Callable[..., Awaitable[BeforeToolCallResult | None]] | None = None
    after_tool_call: Callable[..., Awaitable[AfterToolCallResult | None]] | None = None


# --------------------------------------------------------------------------
# Events
# --------------------------------------------------------------------------


@dataclass
class AgentStartEvent:
    type: Literal["agent_start"] = "agent_start"


@dataclass
class AgentEndEvent:
    messages: list[AgentMessage] = field(default_factory=list)
    type: Literal["agent_end"] = "agent_end"


@dataclass
class TurnStartEvent:
    type: Literal["turn_start"] = "turn_start"


@dataclass
class TurnEndEvent:
    message: AgentMessage
    tool_results: list[ToolResultMessage] = field(default_factory=list)
    type: Literal["turn_end"] = "turn_end"


@dataclass
class MessageStartEvent:
    message: AgentMessage
    type: Literal["message_start"] = "message_start"


@dataclass
class MessageUpdateEvent:
    message: AgentMessage
    assistant_message_event: AssistantMessageEvent
    type: Literal["message_update"] = "message_update"


@dataclass
class MessageEndEvent:
    message: AgentMessage
    type: Literal["message_end"] = "message_end"


@dataclass
class ToolExecutionStartEvent:
    tool_call_id: str
    tool_name: str
    args: Any
    type: Literal["tool_execution_start"] = "tool_execution_start"


@dataclass
class ToolExecutionUpdateEvent:
    tool_call_id: str
    tool_name: str
    args: Any
    partial_result: Any
    type: Literal["tool_execution_update"] = "tool_execution_update"


@dataclass
class ToolExecutionEndEvent:
    tool_call_id: str
    tool_name: str
    result: Any
    is_error: bool
    type: Literal["tool_execution_end"] = "tool_execution_end"


AgentEvent = (
    AgentStartEvent
    | AgentEndEvent
    | TurnStartEvent
    | TurnEndEvent
    | MessageStartEvent
    | MessageUpdateEvent
    | MessageEndEvent
    | ToolExecutionStartEvent
    | ToolExecutionUpdateEvent
    | ToolExecutionEndEvent
)


class AgentState(Protocol):
    """Public agent state."""

    system_prompt: str
    model: Model
    thinking_level: ThinkingLevel
    tools: list[AgentTool]
    messages: list[AgentMessage]

    @property
    def is_streaming(self) -> bool: ...

    @property
    def streaming_message(self) -> AgentMessage | None: ...

    @property
    def pending_tool_calls(self) -> Sequence[str]: ...

    @property
    def error_message(self) -> str | None: ...


__all__ = [
    "AbortSignal",
    "AfterToolCallContext",
    "AfterToolCallResult",
    "AgentContext",
    "AgentEndEvent",
    "AgentEvent",
    "AgentLoopConfig",
    "AgentLoopTurnUpdate",
    "AgentMessage",
    "AgentStartEvent",
    "AgentState",
    "AgentTool",
    "AgentToolCall",
    "AgentToolResult",
    "AgentToolUpdateCallback",
    "BeforeToolCallContext",
    "BeforeToolCallResult",
    "Context",
    "MessageEndEvent",
    "MessageStartEvent",
    "MessageUpdateEvent",
    "PrepareNextTurnContext",
    "QueueMode",
    "ShouldStopAfterTurnContext",
    "StreamFn",
    "ThinkingLevel",
    "ToolExecutionEndEvent",
    "ToolExecutionMode",
    "ToolExecutionStartEvent",
    "ToolExecutionUpdateEvent",
    "TurnEndEvent",
    "TurnStartEvent",
]
