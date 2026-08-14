"""Stateful agent facade.

Python port of `packages/agent/src/agent.ts`. :class:`Agent` owns the
transcript, emits lifecycle events, executes tools and exposes the queueing API
for steering and follow-up messages.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from pi_ai import (
    AssistantMessage,
    ImageContent,
    Message,
    Model,
    TextContent,
    ThinkingBudgets,
    Usage,
    UserMessage,
    now_ms,
)
from pi_ai.utils.abort import AbortController, AbortSignal

from .agent_loop import run_agent_loop, run_agent_loop_continue
from .stream_fn import get_default_stream_fn
from .types import (
    AfterToolCallResult,
    AgentContext,
    AgentEndEvent,
    AgentEvent,
    AgentLoopConfig,
    AgentLoopTurnUpdate,
    AgentMessage,
    AgentTool,
    BeforeToolCallResult,
    MessageEndEvent,
    MessageStartEvent,
    PrepareNextTurnContext,
    QueueMode,
    ShouldStopAfterTurnContext,
    StreamFn,
    ThinkingLevel,
    ToolExecutionMode,
    TurnEndEvent,
)

DEFAULT_MODEL = Model(
    id="unknown",
    name="unknown",
    api="unknown",
    provider="unknown",
    base_url="",
    reasoning=False,
    input=[],
    context_window=0,
    max_tokens=0,
)


def default_convert_to_llm(messages: list[AgentMessage]) -> list[Message]:
    return [message for message in messages if message.role in ("user", "assistant", "toolResult")]


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


@dataclass
class MutableAgentState:
    """Mutable agent state. Assigning ``tools``/``messages`` copies the array."""

    system_prompt: str = ""
    model: Model = field(default_factory=lambda: DEFAULT_MODEL)
    thinking_level: ThinkingLevel = "off"
    is_streaming: bool = False
    streaming_message: AgentMessage | None = None
    pending_tool_calls: set[str] = field(default_factory=set)
    error_message: str | None = None
    _tools: list[AgentTool] = field(default_factory=list)
    _messages: list[AgentMessage] = field(default_factory=list)

    @property
    def tools(self) -> list[AgentTool]:
        return self._tools

    @tools.setter
    def tools(self, next_tools: list[AgentTool]) -> None:
        self._tools = list(next_tools)

    @property
    def messages(self) -> list[AgentMessage]:
        return self._messages

    @messages.setter
    def messages(self, next_messages: list[AgentMessage]) -> None:
        self._messages = list(next_messages)


class PendingMessageQueue:
    """Queue of messages waiting to be injected into a run."""

    def __init__(self, mode: QueueMode) -> None:
        self.mode: QueueMode = mode
        self._messages: list[AgentMessage] = []

    def enqueue(self, message: AgentMessage) -> None:
        self._messages.append(message)

    def has_items(self) -> bool:
        return bool(self._messages)

    def drain(self) -> list[AgentMessage]:
        if self.mode == "all":
            drained = list(self._messages)
            self._messages = []
            return drained

        if not self._messages:
            return []
        first = self._messages[0]
        self._messages = self._messages[1:]
        return [first]

    def clear(self) -> None:
        self._messages = []


@dataclass
class _ActiveRun:
    future: asyncio.Future[None]
    abort_controller: AbortController


AgentListener = Callable[[AgentEvent, AbortSignal], Any]


class Agent:
    """Stateful wrapper around the low-level agent loop."""

    def __init__(
        self,
        stream_fn: StreamFn | None = None,
        *,
        initial_state: MutableAgentState | None = None,
        convert_to_llm: Callable[[list[AgentMessage]], Any] | None = None,
        transform_context: Callable[..., Awaitable[list[AgentMessage]]] | None = None,
        get_api_key: Callable[[str], Any] | None = None,
        on_payload: Any = None,
        on_response: Any = None,
        before_tool_call: Callable[..., Awaitable[BeforeToolCallResult | None]] | None = None,
        after_tool_call: Callable[..., Awaitable[AfterToolCallResult | None]] | None = None,
        should_stop_after_turn: Callable[..., Any] | None = None,
        prepare_next_turn: Callable[..., Any] | None = None,
        prepare_next_turn_with_context: Callable[..., Any] | None = None,
        steering_mode: QueueMode = "one-at-a-time",
        follow_up_mode: QueueMode = "one-at-a-time",
        session_id: str | None = None,
        thinking_budgets: ThinkingBudgets | None = None,
        transport: str = "auto",
        max_retry_delay_ms: int | None = None,
        tool_execution: ToolExecutionMode = "parallel",
    ) -> None:
        self._state = initial_state or MutableAgentState()
        self._listeners: list[AgentListener] = []
        self._steering_queue = PendingMessageQueue(steering_mode)
        self._follow_up_queue = PendingMessageQueue(follow_up_mode)
        self._active_run: _ActiveRun | None = None

        self.convert_to_llm = convert_to_llm or default_convert_to_llm
        self.transform_context = transform_context
        self.stream_function = stream_fn if stream_fn is not None else get_default_stream_fn()
        self.get_api_key = get_api_key
        self.on_payload = on_payload
        self.on_response = on_response
        self.before_tool_call = before_tool_call
        self.after_tool_call = after_tool_call
        self.should_stop_after_turn = should_stop_after_turn
        self.prepare_next_turn = prepare_next_turn
        self.prepare_next_turn_with_context = prepare_next_turn_with_context
        self.session_id = session_id
        self.thinking_budgets = thinking_budgets
        self.transport = transport
        self.max_retry_delay_ms = max_retry_delay_ms
        self.tool_execution = tool_execution

    # -- subscription ------------------------------------------------------

    def subscribe(self, listener: AgentListener) -> Callable[[], None]:
        """Subscribe to lifecycle events.

        Listeners are awaited in subscription order and are part of the run's
        settlement: the agent goes idle only after ``agent_end`` listeners
        finish.
        """
        self._listeners.append(listener)

        def unsubscribe() -> None:
            if listener in self._listeners:
                self._listeners.remove(listener)

        return unsubscribe

    @property
    def state(self) -> MutableAgentState:
        return self._state

    # -- queues ------------------------------------------------------------

    @property
    def steering_mode(self) -> QueueMode:
        return self._steering_queue.mode

    @steering_mode.setter
    def steering_mode(self, mode: QueueMode) -> None:
        self._steering_queue.mode = mode

    @property
    def follow_up_mode(self) -> QueueMode:
        return self._follow_up_queue.mode

    @follow_up_mode.setter
    def follow_up_mode(self, mode: QueueMode) -> None:
        self._follow_up_queue.mode = mode

    def steer(self, message: AgentMessage) -> None:
        """Queue a message injected after the current assistant turn finishes."""
        self._steering_queue.enqueue(message)

    def follow_up(self, message: AgentMessage) -> None:
        """Queue a message that runs only after the agent would otherwise stop."""
        self._follow_up_queue.enqueue(message)

    def clear_steering_queue(self) -> None:
        self._steering_queue.clear()

    def clear_follow_up_queue(self) -> None:
        self._follow_up_queue.clear()

    def clear_all_queues(self) -> None:
        self.clear_steering_queue()
        self.clear_follow_up_queue()

    def has_queued_messages(self) -> bool:
        return self._steering_queue.has_items() or self._follow_up_queue.has_items()

    # -- lifecycle ---------------------------------------------------------

    @property
    def signal(self) -> AbortSignal | None:
        return self._active_run.abort_controller.signal if self._active_run else None

    def abort(self) -> None:
        if self._active_run:
            self._active_run.abort_controller.abort()

    async def wait_for_idle(self) -> None:
        """Resolve once the current run and all awaited listeners have finished."""
        if self._active_run:
            await asyncio.shield(self._active_run.future)

    def reset(self) -> None:
        if self._active_run:
            raise RuntimeError("Agent is already processing. Wait for completion before resetting.")

        self._state.messages = []
        self._state.is_streaming = False
        self._state.streaming_message = None
        self._state.pending_tool_calls = set()
        self._state.error_message = None
        self.clear_follow_up_queue()
        self.clear_steering_queue()

    async def prompt(
        self,
        input_value: str | AgentMessage | list[AgentMessage],
        images: list[ImageContent] | None = None,
    ) -> None:
        """Start a new run from text, a single message, or a batch of messages."""
        if self._active_run:
            raise RuntimeError(
                "Agent is already processing a prompt. Use steer() or follow_up() to queue messages, "
                "or wait for completion."
            )
        messages = self._normalize_prompt_input(input_value, images)
        await self._run_prompt_messages(messages)

    async def continue_(self) -> None:
        """Continue from the transcript. The last message must be user or tool result."""
        if self._active_run:
            raise RuntimeError("Agent is already processing. Wait for completion before continuing.")

        if not self._state.messages:
            raise RuntimeError("No messages to continue from")

        last_message = self._state.messages[-1]
        if last_message.role == "assistant":
            queued_steering = self._steering_queue.drain()
            if queued_steering:
                await self._run_prompt_messages(queued_steering, skip_initial_steering_poll=True)
                return

            queued_follow_ups = self._follow_up_queue.drain()
            if queued_follow_ups:
                await self._run_prompt_messages(queued_follow_ups)
                return

            raise RuntimeError("Cannot continue from message role: assistant")

        await self._run_continuation()

    def _normalize_prompt_input(
        self,
        input_value: str | AgentMessage | list[AgentMessage],
        images: list[ImageContent] | None,
    ) -> list[AgentMessage]:
        if isinstance(input_value, list):
            return input_value
        if not isinstance(input_value, str):
            return [input_value]

        content: list[TextContent | ImageContent] = [TextContent(text=input_value)]
        if images:
            content.extend(images)
        return [UserMessage(content=content, timestamp=now_ms())]

    async def _run_prompt_messages(
        self, messages: list[AgentMessage], skip_initial_steering_poll: bool = False
    ) -> None:
        async def executor(signal: AbortSignal) -> None:
            await run_agent_loop(
                messages,
                self._create_context_snapshot(),
                self._create_loop_config(skip_initial_steering_poll),
                self._process_events,
                signal,
                self.stream_function,
            )

        await self._run_with_lifecycle(executor)

    async def _run_continuation(self) -> None:
        async def executor(signal: AbortSignal) -> None:
            await run_agent_loop_continue(
                self._create_context_snapshot(),
                self._create_loop_config(),
                self._process_events,
                signal,
                self.stream_function,
            )

        await self._run_with_lifecycle(executor)

    def _create_context_snapshot(self) -> AgentContext:
        return AgentContext(
            system_prompt=self._state.system_prompt,
            messages=list(self._state.messages),
            tools=list(self._state.tools),
        )

    def _create_loop_config(self, skip_initial_steering_poll: bool = False) -> AgentLoopConfig:
        pending_skip = {"value": skip_initial_steering_poll}

        should_stop_after_turn = self.should_stop_after_turn
        wrapped_should_stop = None
        if should_stop_after_turn is not None:

            async def wrapped_should_stop(context: ShouldStopAfterTurnContext) -> bool:
                return bool(await _maybe_await(should_stop_after_turn(context, self.signal)))

        wrapped_prepare_next_turn = None
        if self.prepare_next_turn_with_context is not None or self.prepare_next_turn is not None:

            async def wrapped_prepare_next_turn(
                context: PrepareNextTurnContext,
            ) -> AgentLoopTurnUpdate | None:
                if self.prepare_next_turn_with_context is not None:
                    return await _maybe_await(self.prepare_next_turn_with_context(context, self.signal))
                if self.prepare_next_turn is not None:
                    return await _maybe_await(self.prepare_next_turn(self.signal))
                return None

        async def get_steering_messages() -> list[AgentMessage]:
            if pending_skip["value"]:
                pending_skip["value"] = False
                return []
            return self._steering_queue.drain()

        async def get_follow_up_messages() -> list[AgentMessage]:
            return self._follow_up_queue.drain()

        return AgentLoopConfig(
            model=self._state.model,
            reasoning=None if self._state.thinking_level == "off" else self._state.thinking_level,
            session_id=self.session_id,
            on_payload=self.on_payload,
            on_response=self.on_response,
            transport=self.transport,
            thinking_budgets=self.thinking_budgets,
            max_retry_delay_ms=self.max_retry_delay_ms,
            tool_execution=self.tool_execution,
            before_tool_call=self.before_tool_call,
            after_tool_call=self.after_tool_call,
            should_stop_after_turn=wrapped_should_stop,
            prepare_next_turn=wrapped_prepare_next_turn,
            convert_to_llm=self.convert_to_llm,
            transform_context=self.transform_context,
            get_api_key=self.get_api_key,
            get_steering_messages=get_steering_messages,
            get_follow_up_messages=get_follow_up_messages,
        )

    async def _run_with_lifecycle(self, executor: Callable[[AbortSignal], Awaitable[None]]) -> None:
        if self._active_run:
            raise RuntimeError("Agent is already processing.")

        abort_controller = AbortController()
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._active_run = _ActiveRun(future=future, abort_controller=abort_controller)

        self._state.is_streaming = True
        self._state.streaming_message = None
        self._state.error_message = None

        try:
            await executor(abort_controller.signal)
        except Exception as error:
            await self._handle_run_failure(error, abort_controller.signal.aborted)
        finally:
            self._finish_run()

    async def _handle_run_failure(self, error: BaseException, aborted: bool) -> None:
        failure_message = AssistantMessage(
            content=[TextContent(text="")],
            api=self._state.model.api,
            provider=self._state.model.provider,
            model=self._state.model.id,
            usage=Usage(),
            stop_reason="aborted" if aborted else "error",
            error_message=str(error),
            timestamp=now_ms(),
        )
        await self._process_events(MessageStartEvent(message=failure_message))
        await self._process_events(MessageEndEvent(message=failure_message))
        await self._process_events(TurnEndEvent(message=failure_message, tool_results=[]))
        await self._process_events(AgentEndEvent(messages=[failure_message]))

    def _finish_run(self) -> None:
        self._state.is_streaming = False
        self._state.streaming_message = None
        self._state.pending_tool_calls = set()
        if self._active_run and not self._active_run.future.done():
            self._active_run.future.set_result(None)
        self._active_run = None

    async def _process_events(self, event: AgentEvent) -> None:
        """Reduce internal state for a loop event, then await listeners.

        ``agent_end`` only means no further loop events arrive; the run becomes
        idle after its listeners settle and ``_finish_run`` clears state.
        """
        if event.type == "message_start" or event.type == "message_update":
            self._state.streaming_message = event.message
        elif event.type == "message_end":
            self._state.streaming_message = None
            self._state.messages.append(event.message)
        elif event.type == "tool_execution_start":
            self._state.pending_tool_calls = {*self._state.pending_tool_calls, event.tool_call_id}
        elif event.type == "tool_execution_end":
            self._state.pending_tool_calls = {
                call_id for call_id in self._state.pending_tool_calls if call_id != event.tool_call_id
            }
        elif event.type == "turn_end":
            if event.message.role == "assistant" and event.message.error_message:
                self._state.error_message = event.message.error_message
        elif event.type == "agent_end":
            self._state.streaming_message = None

        signal = self._active_run.abort_controller.signal if self._active_run else None
        if signal is None:
            raise RuntimeError("Agent listener invoked outside active run")
        for listener in list(self._listeners):
            await _maybe_await(listener(event, signal))
