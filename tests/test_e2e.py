"""Python port of `packages/agent/test/e2e.test.ts`.

Drives the real `Agent` facade against the ported `faux` provider through the
global api-provider registry, exactly as the TypeScript does with
`registerFauxProvider` + `streamSimple`. No network call is made.

`agent.continue()` is spelled `agent.continue_()` in Python (`continue` is a
keyword).
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator

import pytest
from pi_agent.agent import Agent, MutableAgentState
from pi_agent.types import AgentEvent
from pi_ai import (
    AssistantMessage,
    Cost,
    Model,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
    now_ms,
)
from pi_ai.compat import FauxProviderRegistration, register_faux_provider, stream_simple
from pi_ai.providers.faux import (
    FauxModelDefinition,
    FauxTokenSizeOptions,
    RegisterFauxProviderOptions,
    faux_assistant_message,
    faux_text,
    faux_thinking,
    faux_tool_call,
)
from tool_utils import calculate_tool

TIMEOUT = 20


@pytest.fixture
def register_faux() -> Iterator[list[FauxProviderRegistration]]:
    """`createFauxRegistration` plus the TypeScript file's `afterEach` unregister loop."""
    registrations: list[FauxProviderRegistration] = []

    def register(options: RegisterFauxProviderOptions | None = None) -> FauxProviderRegistration:
        registration = register_faux_provider(options)
        registrations.append(registration)
        return registration

    yield register  # type: ignore[misc]

    while registrations:
        registrations.pop().unregister()


def get_text_content(message: AssistantMessage | ToolResultMessage) -> str:
    return "\n".join(block.text for block in message.content if block.type == "text")


def make_agent(model: Model, system_prompt: str, tools: list | None = None) -> Agent:
    state = MutableAgentState(system_prompt=system_prompt, model=model, thinking_level="off")
    state.tools = tools if tools is not None else []
    return Agent(stream_simple, initial_state=state)


def zero_usage() -> Usage:
    return Usage(
        input=0,
        output=0,
        cache_read=0,
        cache_write=0,
        total_tokens=0,
        cost=Cost(input=0, output=0, cache_read=0, cache_write=0, total=0),
    )


# ---------------------------------------------------------------------------
# Agent integration with faux provider
# ---------------------------------------------------------------------------


async def test_handles_a_basic_text_prompt(register_faux) -> None:
    faux = register_faux()
    faux.set_responses([faux_assistant_message("4")])
    agent = make_agent(
        faux.get_model(None),
        "You are a helpful assistant. Keep your responses concise.",
    )

    await asyncio.wait_for(agent.prompt("What is 2+2? Answer with just the number."), timeout=TIMEOUT)

    assert agent.state.is_streaming is False
    assert len(agent.state.messages) == 2
    assert agent.state.messages[0].role == "user"
    assert agent.state.messages[1].role == "assistant"
    assert "4" in get_text_content(agent.state.messages[1])


async def test_executes_tools_and_tracks_pending_tool_calls(register_faux) -> None:
    faux = register_faux()
    faux.set_responses(
        [
            faux_assistant_message(
                [
                    faux_text("Let me calculate that."),
                    faux_tool_call("calculate", {"expression": "123 * 456"}, id="calc-1"),
                ],
                stop_reason="toolUse",
            ),
            faux_assistant_message("The result is 56088."),
        ]
    )
    agent = make_agent(
        faux.get_model(None),
        "You are a helpful assistant. Always use the calculator tool for math.",
        tools=[calculate_tool],
    )

    pending_during_events: list[tuple[str, list[str]]] = []

    def listener(event: AgentEvent, signal) -> None:
        if event.type in ("tool_execution_start", "tool_execution_end"):
            pending_during_events.append((event.type, sorted(agent.state.pending_tool_calls)))

    agent.subscribe(listener)

    await asyncio.wait_for(agent.prompt("Calculate 123 * 456 using the calculator tool."), timeout=TIMEOUT)

    assert agent.state.is_streaming is False
    assert len(agent.state.messages) >= 4
    tool_result = next(message for message in agent.state.messages if message.role == "toolResult")
    assert "123 * 456 = 56088" in get_text_content(tool_result)

    final_message = agent.state.messages[-1]
    assert final_message.role == "assistant"
    assert "56088" in get_text_content(final_message)
    assert len(agent.state.pending_tool_calls) == 0
    assert pending_during_events == [
        ("tool_execution_start", ["calc-1"]),
        ("tool_execution_end", []),
    ]


async def test_handles_abort_during_streaming(register_faux) -> None:
    faux = register_faux(
        RegisterFauxProviderOptions(
            tokens_per_second=20,
            token_size=FauxTokenSizeOptions(min=2, max=2),
        )
    )
    faux.set_responses(
        [
            faux_assistant_message(
                "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen"
            )
        ]
    )
    agent = make_agent(faux.get_model(None), "You are a helpful assistant.")

    async def abort_later() -> None:
        await asyncio.sleep(0.03)
        agent.abort()

    aborter = asyncio.ensure_future(abort_later())
    await asyncio.wait_for(agent.prompt("Count slowly from 1 to 20."), timeout=TIMEOUT)
    await aborter

    assert agent.state.is_streaming is False
    assert len(agent.state.messages) >= 2

    last_message = agent.state.messages[-1]
    assert last_message.role == "assistant"
    assert last_message.stop_reason == "aborted"
    assert last_message.error_message is not None
    assert agent.state.error_message == last_message.error_message


async def test_emits_lifecycle_updates_while_streaming(register_faux) -> None:
    faux = register_faux(RegisterFauxProviderOptions(token_size=FauxTokenSizeOptions(min=1, max=1)))
    faux.set_responses([faux_assistant_message("1 2 3 4 5")])
    agent = make_agent(faux.get_model(None), "You are a helpful assistant.")

    events: list[str] = []
    agent.subscribe(lambda event, signal: events.append(event.type))

    await asyncio.wait_for(agent.prompt("Count from 1 to 5."), timeout=TIMEOUT)

    for expected in (
        "agent_start",
        "turn_start",
        "message_start",
        "message_update",
        "message_end",
        "turn_end",
        "agent_end",
    ):
        assert expected in events
    assert events.index("agent_start") < events.index("message_start")
    assert events.index("message_start") < events.index("message_end")
    assert events.index("message_end") < len(events) - 1 - events[::-1].index("agent_end")

    assert agent.state.is_streaming is False
    assert len(agent.state.messages) == 2


async def test_maintains_context_across_multiple_turns(register_faux) -> None:
    faux = register_faux()

    def second_response(context, stream_options=None, state=None, request_model=None):
        has_alice = any(
            message.role == "user"
            and (
                "Alice" in message.content
                if isinstance(message.content, str)
                else any(block.type == "text" and "Alice" in block.text for block in message.content)
            )
            for message in context.messages
        )
        return faux_assistant_message("Your name is Alice." if has_alice else "I do not know your name.")

    faux.set_responses([faux_assistant_message("Nice to meet you, Alice."), second_response])
    agent = make_agent(faux.get_model(None), "You are a helpful assistant.")

    await asyncio.wait_for(agent.prompt("My name is Alice."), timeout=TIMEOUT)
    assert len(agent.state.messages) == 2

    await asyncio.wait_for(agent.prompt("What is my name?"), timeout=TIMEOUT)
    assert len(agent.state.messages) == 4

    last_message = agent.state.messages[3]
    assert last_message.role == "assistant"
    assert "alice" in get_text_content(last_message).lower()


async def test_preserves_thinking_content_blocks(register_faux) -> None:
    faux = register_faux(RegisterFauxProviderOptions(models=[FauxModelDefinition(id="faux-reasoning", reasoning=True)]))
    faux.set_responses([faux_assistant_message([faux_thinking("step by step"), faux_text("4")])])

    agent = Agent(
        stream_simple,
        initial_state=MutableAgentState(
            system_prompt="You are a helpful assistant.",
            model=faux.get_model(None),
            thinking_level="low",
        ),
    )

    await asyncio.wait_for(agent.prompt("What is 2+2?"), timeout=TIMEOUT)

    assistant_message = agent.state.messages[1]
    assert assistant_message.role == "assistant"
    assert assistant_message.content == [
        ThinkingContent(thinking="step by step"),
        TextContent(text="4"),
    ]


# ---------------------------------------------------------------------------
# Agent.continue() with faux provider
# ---------------------------------------------------------------------------


async def test_continue_throws_when_no_messages_in_context(register_faux) -> None:
    faux = register_faux()
    agent = Agent(
        stream_simple,
        initial_state=MutableAgentState(system_prompt="Test", model=faux.get_model(None)),
    )

    with pytest.raises(RuntimeError, match="No messages to continue from"):
        await agent.continue_()


async def test_continue_throws_when_last_message_is_assistant(register_faux) -> None:
    faux = register_faux()
    model = faux.get_model(None)
    agent = Agent(
        stream_simple,
        initial_state=MutableAgentState(system_prompt="Test", model=model),
    )

    agent.state.messages = [
        AssistantMessage(
            content=[TextContent(text="Hello")],
            api=model.api,
            provider=model.provider,
            model=model.id,
            usage=zero_usage(),
            stop_reason="stop",
            timestamp=now_ms(),
        )
    ]

    with pytest.raises(RuntimeError, match="Cannot continue from message role: assistant"):
        await agent.continue_()


async def test_continues_and_gets_a_response_when_last_message_is_user(register_faux) -> None:
    faux = register_faux()
    faux.set_responses([faux_assistant_message("HELLO WORLD")])
    agent = make_agent(
        faux.get_model(None),
        "You are a helpful assistant. Follow instructions exactly.",
    )

    agent.state.messages = [UserMessage(content=[TextContent(text="Say exactly: HELLO WORLD")], timestamp=now_ms())]

    await asyncio.wait_for(agent.continue_(), timeout=TIMEOUT)

    assert agent.state.is_streaming is False
    assert len(agent.state.messages) == 2
    assert agent.state.messages[0].role == "user"
    assert agent.state.messages[1].role == "assistant"
    assert "HELLO WORLD" in get_text_content(agent.state.messages[1]).upper()


async def test_continues_and_processes_tool_results(register_faux) -> None:
    faux = register_faux()
    model = faux.get_model(None)
    faux.set_responses([faux_assistant_message("The answer is 8.")])
    agent = make_agent(
        model,
        "You are a helpful assistant. After getting a calculation result, state the answer clearly.",
        tools=[calculate_tool],
    )

    agent.state.messages = [
        UserMessage(content=[TextContent(text="What is 5 + 3?")], timestamp=now_ms()),
        AssistantMessage(
            content=[
                TextContent(text="Let me calculate that."),
                ToolCall(id="calc-1", name="calculate", arguments={"expression": "5 + 3"}),
            ],
            api=model.api,
            provider=model.provider,
            model=model.id,
            usage=zero_usage(),
            stop_reason="toolUse",
            timestamp=now_ms(),
        ),
        ToolResultMessage(
            tool_call_id="calc-1",
            tool_name="calculate",
            content=[TextContent(text="5 + 3 = 8")],
            is_error=False,
            timestamp=now_ms(),
        ),
    ]

    await asyncio.wait_for(agent.continue_(), timeout=TIMEOUT)

    assert agent.state.is_streaming is False
    assert len(agent.state.messages) >= 4

    last_message = agent.state.messages[-1]
    assert last_message.role == "assistant"
    assert "8" in get_text_content(last_message)
