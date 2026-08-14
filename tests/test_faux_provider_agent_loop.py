"""The faux provider drives the real agent loop.

This lived in `pi-ai`'s suite, which made `pp-ai`'s tests depend on
`pp-agent-core` -- and `pp-agent-core` depends on `pp-ai`. Inside one workspace
that cycle is invisible; once each package is its own repository it is not
resolvable, because testing a new `pp-ai` would require a `pp-agent-core` that
pins the *previous* `pp-ai`.

It belongs here instead: this package already depends on `pp-ai`, so the
dependency runs one way and the test always runs.
"""

from __future__ import annotations

from pi_agent import (
    AgentContext,
    AgentLoopConfig,
    AgentTool,
    AgentToolResult,
    agent_loop,
    default_convert_to_llm,
)
from pi_ai.providers.faux import faux_assistant_message, faux_provider, faux_tool_call
from pi_ai.types import TextContent, UserMessage


async def test_faux_provider_drives_agent_loop_prompt_tool_call_final_answer() -> None:
    """Full prompt -> tool call -> tool result -> final answer cycle.

    This proves the faux provider satisfies the same `StreamFn` contract the
    ported agent loop expects from a real provider: `agent_loop` calls
    `stream_fn(model, context, options)` and expects an
    `AssistantMessageEventStream`.
    """
    handle = faux_provider()
    tool_call = faux_tool_call("echo", {"value": "hi"}, id="call-1")
    handle.set_responses(
        [
            faux_assistant_message([tool_call], stop_reason="toolUse"),
            faux_assistant_message("all done"),
        ]
    )

    async def execute(tool_call_id, params, signal=None, on_update=None):
        return AgentToolResult(content=[TextContent(text=f"echo:{params.get('value', '')}")], details={})

    echo_tool = AgentTool(
        name="echo",
        description="Echo a value",
        parameters={"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]},
        label="echo",
        execute=execute,
    )

    config = AgentLoopConfig(model=handle.get_model(), convert_to_llm=default_convert_to_llm)

    def stream_fn(model, context, options=None):
        return handle.provider.stream(model, context, options)

    stream = agent_loop(
        [UserMessage(content="please echo hi")], AgentContext(tools=[echo_tool]), config, None, stream_fn
    )
    events = [event async for event in stream]
    messages = await stream.result()

    assert [event.type for event in events][:2] == ["agent_start", "turn_start"]
    assert events[-1].type == "agent_end"

    tool_results = [m for m in messages if m.role == "toolResult"]
    assert len(tool_results) == 1
    assert tool_results[0].content[0].text == "echo:hi"
    assert tool_results[0].is_error is False

    final_assistant_messages = [m for m in messages if m.role == "assistant"]
    assert final_assistant_messages[-1].content[0].text == "all done"
    assert handle.state.call_count == 2
