"""Tests for `pi_agent.harness.agent_harness` and `pi_agent.stream_fn`.

No dedicated TS test file exists for `packages/agent/src/agent-harness.ts` or
`stream-fn.ts`. Upstream's harness implements its configuration accessors for
real and rejects every durable operation with `HarnessNotImplemented`; these
tests pin both halves of that contract so the Python port stays in step when
upstream fills the operations in.
"""

from __future__ import annotations

import pytest
from pi_agent.harness.agent_harness import (
    AgentHarness,
    AgentHarnessOptions,
    HarnessClosed,
    HarnessNotImplemented,
    HarnessTool,
    LaneBusy,
    Resources,
)
from pi_agent.harness.compaction.compaction import CompactionSettings
from pi_agent.harness.session import InMemorySessionStorage, Session
from pi_agent.harness.session.types import OperationStartedRecord, RunIntent, SessionMetadata
from pi_agent.stream_fn import get_default_stream_fn, set_default_stream_fn
from pi_agent.types import AgentTool
from pi_ai.types import Model
from pi_ai.utils.retry import RetryPolicy


def make_session() -> Session:
    return Session(InMemorySessionStorage(SessionMetadata(id="session", created_at=1)))


def make_options(**overrides) -> AgentHarnessOptions:
    defaults = dict(session=make_session(), models=None, model=Model(id="m", provider="test"))
    defaults.update(overrides)
    return AgentHarnessOptions(**defaults)


@pytest.mark.asyncio
async def test_create_returns_a_harness_with_no_suspended_operations():
    harness, suspended = await AgentHarness.create(make_options())

    assert harness.name == "main"
    assert suspended == []


@pytest.mark.asyncio
async def test_create_refuses_to_restore_a_session_that_already_has_records():
    session = make_session()
    await session.append_record(OperationStartedRecord(id="op1", lane="main", source_leaf_id=None, intent=RunIntent()))

    with pytest.raises(HarnessNotImplemented, match=r"create\.restore"):
        await AgentHarness.create(make_options(session=session))


@pytest.mark.asyncio
async def test_active_tools_default_to_every_supplied_tool():
    tools = [HarnessTool(tool=AgentTool(name="read", description="", parameters={}, execute=None))]
    harness = AgentHarness(make_options(tools=tools))

    assert await harness.get_active_tools() == ["read"]


@pytest.mark.asyncio
async def test_explicit_active_tool_names_win_over_the_tool_list():
    tools = [HarnessTool(tool=AgentTool(name="read", description="", parameters={}, execute=None))]
    harness = AgentHarness(make_options(tools=tools, active_tool_names=[]))

    assert await harness.get_active_tools() == []


@pytest.mark.asyncio
async def test_configuration_accessors_round_trip():
    harness = AgentHarness(make_options())

    replacement = Model(id="m2", provider="other")
    await harness.set_model(replacement)
    await harness.set_thinking_level("high")
    await harness.set_active_tools(["bash"])
    await harness.set_steering_mode("all")
    await harness.set_follow_up_mode("all")
    await harness.set_stream_options({"temperature": 0.5})
    await harness.set_retry_policy(RetryPolicy(enabled=True, max_retries=3, base_delay_ms=250))
    await harness.set_compaction_settings(CompactionSettings(enabled=False, reserve_tokens=1, keep_recent_tokens=2))

    assert await harness.get_model() is replacement
    assert await harness.get_thinking_level() == "high"
    assert await harness.get_active_tools() == ["bash"]
    assert await harness.get_steering_mode() == "all"
    assert await harness.get_follow_up_mode() == "all"
    assert await harness.get_stream_options() == {"temperature": 0.5}
    assert (await harness.get_retry_policy()).max_retries == 3
    assert (await harness.get_compaction_settings()).enabled is False


@pytest.mark.asyncio
async def test_accessors_return_copies_so_callers_cannot_mutate_harness_state():
    harness = AgentHarness(make_options(stream_options={"temperature": 0.1}))

    (await harness.get_stream_options())["temperature"] = 9.9
    (await harness.get_active_tools()).append("injected")

    assert await harness.get_stream_options() == {"temperature": 0.1}
    assert await harness.get_active_tools() == []


@pytest.mark.asyncio
async def test_resources_round_trip_and_are_copied():
    harness = AgentHarness(make_options())
    await harness.set_resources(Resources(skills=[], prompt_templates=[]))

    resources = await harness.get_resources()
    resources.skills.append("injected")

    assert (await harness.get_resources()).skills == []


@pytest.mark.asyncio
async def test_set_tools_resets_active_names_to_the_new_tools():
    harness = AgentHarness(make_options())
    tools = [
        HarnessTool(tool=AgentTool(name="read", description="", parameters={}, execute=None)),
        HarnessTool(tool=AgentTool(name="write", description="", parameters={}, execute=None), replay="safe"),
    ]

    await harness.set_tools(tools)

    assert await harness.get_active_tools() == ["read", "write"]
    assert [tool.replay for tool in await harness.get_tools()] == [None, "safe"]


@pytest.mark.asyncio
async def test_leaf_id_reads_through_to_the_session():
    session = make_session()
    harness = AgentHarness(make_options(session=session))

    assert await harness.get_leaf_id() == await session.get_leaf_id()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("call", "operation"),
    [
        (lambda h: h.prompt("hi"), "prompt"),
        (lambda h: h.skill("review"), "skill"),
        (lambda h: h.prompt_from_template("t"), "promptFromTemplate"),
        (lambda h: h.compact(), "compact"),
        (lambda h: h.navigate_tree(None), "navigateTree"),
        (lambda h: h.resume(), "resume"),
        (lambda h: h.abort(), "abort"),
        (lambda h: h.steer("hi"), "steer"),
        (lambda h: h.follow_up("hi"), "followUp"),
        (lambda h: h.next_run("hi"), "nextRun"),
        (lambda h: h.cancel_queued("e1"), "cancelQueued"),
        (lambda h: h.wait_for_idle(), "waitForIdle"),
        (lambda h: h.peek_action(), "peekAction"),
        (lambda h: h.execute_action(), "executeAction"),
        (lambda h: h.run_to_completion(), "runToCompletion"),
        (lambda h: h.watch(), "watch"),
        (lambda h: h.lanes(), "lanes"),
        (lambda h: h.watch_session(), "watchSession"),
    ],
)
async def test_durable_operations_are_not_implemented(call, operation):
    harness = AgentHarness(make_options())

    with pytest.raises(HarnessNotImplemented) as info:
        await call(harness)

    assert info.value.operation == operation


@pytest.mark.asyncio
async def test_operations_report_closure_after_close():
    harness = AgentHarness(make_options())
    await harness.close()

    with pytest.raises(HarnessClosed):
        await harness.prompt("hi")


@pytest.mark.asyncio
async def test_hooks_and_events_registration_is_unavailable():
    harness = AgentHarness(make_options())

    with pytest.raises(HarnessNotImplemented, match=r"hooks\.on"):
        harness.hooks.on("before_run", lambda event: None)
    with pytest.raises(HarnessNotImplemented, match=r"events\.on"):
        harness.events.on("run_start", lambda event: None)

    await harness.close()
    with pytest.raises(HarnessClosed):
        harness.hooks.on("before_run", lambda event: None)


def test_rejection_errors_carry_their_tag_and_properties():
    error = LaneBusy(lane="main", operation_id="op1", operation_kind="run", message="busy")

    assert error._tag == "LaneBusy"
    assert error.lane == "main"
    assert str(error) == "busy"


def test_default_stream_fn_round_trips():
    set_default_stream_fn(None)
    with pytest.raises(RuntimeError, match="No default stream function"):
        get_default_stream_fn()

    def fake_stream(*args, **kwargs):
        raise AssertionError("not called")

    set_default_stream_fn(fake_stream)
    try:
        assert get_default_stream_fn() is fake_stream
    finally:
        set_default_stream_fn(None)
