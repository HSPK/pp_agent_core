"""Python port of `packages/agent/test/harness/agent-harness-scaffold.test.ts`.

`tests/test_agent_harness.py` already exists in the port with a different (port-only)
suite, so this file keeps the upstream name with a `_scaffold` suffix.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

import pytest
from pi_ai.providers.all import get_builtin_model
from pi_ai.registry import Models
from pi_ai.types import Cost, Model, TextContent, Usage, UserMessage
from pi_ai.utils.retry import RetryPolicy

from pi_agent.harness.agent_harness import (
    AgentHarness,
    AgentHarnessOptions,
    HarnessClosed,
    HarnessNotImplemented,
    HarnessTool,
    Resources,
)
from pi_agent.harness.compaction.compaction import CompactionSettings
from pi_agent.harness.session import InMemorySessionStorage, Session
from pi_agent.harness.session.types import OperationStartedRecord, RunIntent, SessionMetadata
from pi_agent.harness.types import PromptTemplate, Skill
from pi_agent.types import AgentTool


def create_session(id: str = "session") -> Session:
    return Session(InMemorySessionStorage(SessionMetadata(id=id, created_at=1)))


def require_model(provider: str, model_id: str) -> Model:
    model = get_builtin_model(provider, model_id)
    assert model is not None
    return model


async def create_harness(session: Session | None = None) -> AgentHarness:
    harness, _suspended = await AgentHarness.create(
        AgentHarnessOptions(
            session=session if session is not None else create_session(),
            models=Models(),
            model=require_model("google", "gemini-2.5-flash"),
        )
    )
    return harness


def operation_started(id: str) -> OperationStartedRecord:
    return OperationStartedRecord(
        id=id,
        lane="main",
        source_leaf_id=None,
        intent=RunIntent(original_prompt=[], initial_messages=[]),
    )


USER_MESSAGE = UserMessage(content=[TextContent(text="hello")], timestamp=1)

USAGE = Usage(
    input=1,
    output=2,
    cache_read=0,
    cache_write=0,
    total_tokens=3,
    cost=Cost(input=0, output=0, cache_read=0, cache_write=0, total=0),
)


@pytest.mark.asyncio
async def test_opens_only_record_free_sessions_before_restore_is_implemented():
    session = create_session()
    harness, suspended = await AgentHarness.create(
        AgentHarnessOptions(
            session=session,
            models=Models(),
            model=require_model("google", "gemini-2.5-flash"),
        )
    )

    assert suspended == []
    assert harness.name == "main"
    assert harness.session is session
    assert await harness.get_leaf_id() is None
    assert await harness.session.get_leaf_id() is None

    assert await harness.close() is None

    recorded = create_session("recorded")
    await recorded.append_record(operation_started("run"))
    with pytest.raises(HarnessNotImplemented) as caught:
        await AgentHarness.create(
            AgentHarnessOptions(
                session=recorded,
                models=Models(),
                model=require_model("google", "gemini-2.5-flash"),
            )
        )
    assert caught.value.operation == "create.restore"


@pytest.mark.asyncio
async def test_keeps_scaffold_safe_configuration_as_defensive_copies():
    harness = await create_harness()
    model = require_model("anthropic", "claude-sonnet-4-5")
    await harness.set_model(model)
    assert await harness.get_model() is model

    await harness.set_thinking_level("high")
    assert await harness.get_thinking_level() == "high"

    active_tools = ["one"]
    await harness.set_active_tools(active_tools)
    active_tools.append("mutated")
    assert await harness.get_active_tools() == ["one"]
    read_active_tools = await harness.get_active_tools()
    read_active_tools.append("mutated")
    assert await harness.get_active_tools() == ["one"]

    tool = HarnessTool(tool=AgentTool(name="tool", description="", parameters={}, label="Tool"))
    tools = [tool]
    await harness.set_tools(tools)
    tools.append(HarnessTool(tool=AgentTool(name="mutated", description="", parameters={}, label="Mutated")))
    assert [item.name for item in await harness.get_tools()] == ["tool"]
    read_tools = await harness.get_tools()
    read_tools.append(HarnessTool(tool=AgentTool(name="mutated", description="", parameters={}, label="Mutated")))
    assert [item.name for item in await harness.get_tools()] == ["tool"]

    resources = Resources(
        skills=[Skill(name="skill", description="desc", content="body", file_path="/skills/SKILL.md")],
        prompt_templates=[PromptTemplate(name="template", content="body")],
    )
    await harness.set_resources(resources)
    resources.skills.append(Skill(name="mutated", description="desc", content="body", file_path="/skills/OTHER.md"))
    assert [skill.name for skill in (await harness.get_resources()).skills] == ["skill"]
    read_resources = await harness.get_resources()
    read_resources.skills.append(
        Skill(name="mutated", description="desc", content="body", file_path="/skills/OTHER.md")
    )
    assert [skill.name for skill in (await harness.get_resources()).skills] == ["skill"]

    stream_options = {"max_tokens": 10}
    await harness.set_stream_options(stream_options)
    stream_options["max_tokens"] = 20
    assert await harness.get_stream_options() == {"max_tokens": 10}
    read_stream_options = await harness.get_stream_options()
    read_stream_options["max_tokens"] = 30
    assert await harness.get_stream_options() == {"max_tokens": 10}

    retry_policy = RetryPolicy(enabled=True, max_retries=2, base_delay_ms=10)
    await harness.set_retry_policy(retry_policy)
    retry_policy.max_retries = 99
    assert await harness.get_retry_policy() == RetryPolicy(enabled=True, max_retries=2, base_delay_ms=10)

    compaction_settings = CompactionSettings(enabled=False, reserve_tokens=1, keep_recent_tokens=2)
    await harness.set_compaction_settings(compaction_settings)
    compaction_settings.reserve_tokens = 99
    assert await harness.get_compaction_settings() == CompactionSettings(
        enabled=False, reserve_tokens=1, keep_recent_tokens=2
    )

    await harness.set_steering_mode("all")
    assert await harness.get_steering_mode() == "all"
    await harness.set_follow_up_mode("all")
    assert await harness.get_follow_up_mode() == "all"


@pytest.mark.asyncio
async def test_rejects_every_unfinished_public_operation_explicitly():
    harness = await create_harness()
    callback_called = False

    def mark_called() -> None:
        nonlocal callback_called
        callback_called = True

    unfinished: list[tuple[str, Callable[[], Awaitable[object]]]] = [
        ("prompt", lambda: harness.prompt("hello")),
        ("skill", lambda: harness.skill("skill")),
        ("promptFromTemplate", lambda: harness.prompt_from_template("template")),
        ("compact", lambda: harness.compact()),
        ("navigateTree", lambda: harness.navigate_tree(None)),
        ("resume", lambda: harness.resume()),
        ("abort", lambda: harness.abort()),
        ("steer", lambda: harness.steer(USER_MESSAGE)),
        ("followUp", lambda: harness.follow_up(USER_MESSAGE)),
        ("nextRun", lambda: harness.next_run(USER_MESSAGE)),
        ("cancelQueued", lambda: harness.cancel_queued("queued")),
        ("recordUsage", lambda: harness.record_usage(USAGE)),
        ("waitForIdle", lambda: harness.wait_for_idle()),
        ("runWhenIdle", lambda: harness.run_when_idle(mark_called)),
        ("peekAction", lambda: harness.peek_action()),
        ("executeAction", lambda: harness.execute_action()),
        ("runToCompletion", lambda: harness.run_to_completion()),
        ("watch", lambda: harness.watch()),
        ("lane", lambda: harness.lane("main")),
        ("createLane", lambda: harness.create_lane("thread", None)),
        ("lanes", lambda: harness.lanes()),
        ("watchSession", lambda: harness.watch_session()),
    ]

    for operation, invoke in unfinished:
        with pytest.raises(HarnessNotImplemented) as caught:
            await invoke()
        assert caught.value.operation == operation, operation

    assert callback_called is False
    with pytest.raises(HarnessNotImplemented):
        harness.hooks.on("before_run", lambda *_: None)
    with pytest.raises(HarnessNotImplemented):
        harness.events.on("event", lambda *_: None)


@pytest.mark.asyncio
async def test_reports_harness_closed_for_unfinished_operations_after_close():
    harness = await create_harness()
    await harness.close()

    with pytest.raises(HarnessClosed):
        await harness.prompt("hello")
    with pytest.raises(HarnessClosed):
        await harness.wait_for_idle()
    with pytest.raises(HarnessClosed):
        harness.hooks.on("before_run", lambda *_: None)
    with pytest.raises(HarnessClosed):
        harness.events.on("event", lambda *_: None)
