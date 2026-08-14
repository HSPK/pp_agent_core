# pi-agent

Stateful agent loop, tool execution, session storage, telemetry, and harness scaffolding. This is the Python port of `@earendil-works/pi-agent-core`.

## Installation

```bash
uv add pi-agent
```

In this workspace, install every package with:

```bash
uv sync --all-packages
```

### Session backends

The Python port ships the in-memory session backend and JSONL v4 session backend in `pi_agent.harness.session`. The TypeScript SQLite session backend is not ported; there is no Python SQLite session backend package.

## Quick Start

```python
import asyncio

from pi_agent import Agent, MutableAgentState
from pi_ai.registry import Models
from pi_ai.providers.anthropic import anthropic_provider


async def main() -> None:
    models = Models()
    models.add(anthropic_provider())
    model = models.get_model("anthropic", "claude-sonnet-4-6")
    if model is None:
        raise RuntimeError("Model not found")

    agent = Agent(
        models.stream_simple,
        initial_state=MutableAgentState(
            system_prompt="You are a helpful assistant.",
            model=model,
        ),
    )

    async def print_text(event, signal) -> None:
        if event.type == "message_update" and event.assistant_message_event.type == "text_delta":
            print(event.assistant_message_event.delta, end="", flush=True)

    agent.subscribe(print_text)
    await agent.prompt("Hello!")


asyncio.run(main())
```

## Core Concepts

### AgentMessage vs LLM Message

The agent works with `AgentMessage`, an alias of `pi_ai.Message`. Python callers may pass custom message objects, but the loop only inspects `role`. Use `convert_to_llm` to filter or transform custom messages before each LLM call.

LLMs only understand `user`, `assistant`, and `toolResult`. The default `default_convert_to_llm()` keeps only those roles.

### Message Flow

```text
AgentMessage list -> transform_context() -> AgentMessage list -> convert_to_llm() -> Message list -> LLM
                         optional                                  required
```

1. **transform_context**: prune old messages or inject external context.
2. **convert_to_llm**: filter UI-only messages or convert custom message objects to LLM messages.

## Event Flow

The agent emits events for UI updates. Understanding the event sequence helps build responsive interfaces.

### prompt() Event Sequence

When you call `await agent.prompt("Hello")`:

```text
prompt("Hello")
├─ agent_start
├─ turn_start
├─ message_start   { message: user_message }      # Your prompt
├─ message_end     { message: user_message }
├─ message_start   { message: assistant_message } # LLM starts responding
├─ message_update  { message: partial... }
├─ message_update  { message: partial... }
├─ message_end     { message: assistant_message } # Complete response
├─ turn_end        { message, tool_results: [] }
└─ agent_end       { messages: [...] }
```

### With Tool Calls

If the assistant calls tools, the loop continues:

```text
prompt("Read config.json")
├─ agent_start
├─ turn_start
├─ message_start/end  { user_message }
├─ message_start      { assistant_message with toolCall }
├─ message_update...
├─ message_end        { assistant_message }
├─ tool_execution_start  { tool_call_id, tool_name, args }
├─ tool_execution_update { partial_result }           # If tool streams
├─ tool_execution_end    { tool_call_id, result }
├─ message_start/end  { toolResult message }
├─ turn_end           { message, tool_results: [tool_result] }
│
├─ turn_start                                        # Next turn
├─ message_start      { assistant_message }          # LLM responds to tool result
├─ message_update...
├─ message_end
├─ turn_end
└─ agent_end
```

Tool execution mode is configurable:

- `parallel` (default): preflight tool calls sequentially, execute allowed tools concurrently, emit `tool_execution_end` as each tool finalizes, then emit `toolResult` messages and `turn_end.tool_results` in assistant source order.
- `sequential`: execute tool calls one by one.

Set the mode globally with `tool_execution` on `Agent` or `AgentLoopConfig`, or per tool with `execution_mode` on `AgentTool`. If any tool call in a batch targets a tool with `execution_mode="sequential"`, the whole batch executes sequentially.

`before_tool_call` runs after `tool_execution_start` and validated argument parsing. It can block execution and attach `terminate=True` to the blocked result. `after_tool_call` runs after tool execution and before final tool events and tool-result messages are emitted.

Tools, blocked `before_tool_call` results, and `after_tool_call` overrides can return `terminate=True` to hint that the automatic follow-up LLM call should be skipped. The loop stops early only when every finalized tool result in that batch sets `terminate=True`.

The `Agent` class accepts `should_stop_after_turn`. Low-level loop callers can set the same hook in `AgentLoopConfig`:

```python
from pi_agent import AgentContext, AgentLoopConfig, agent_loop


def should_compact_before_next_turn(messages) -> bool:
    return len(messages) > 100


async def run(prompts, model, stream_fn) -> None:
    context = AgentContext(system_prompt="You are helpful.", messages=[], tools=[])
    config = AgentLoopConfig(
        model=model,
        convert_to_llm=lambda messages: [m for m in messages if m.role in ("user", "assistant", "toolResult")],
        should_stop_after_turn=lambda turn: should_compact_before_next_turn(turn.context.messages),
    )

    stream = agent_loop(prompts, context, config, None, stream_fn)
    async for event in stream:
        print(event.type)
```

`should_stop_after_turn` runs after `turn_end`, after the assistant response and tool executions complete. If it returns true, the loop emits `agent_end` and exits before polling steering or follow-up queues and before another LLM call.

When you use `Agent`, assistant `message_end` processing is a barrier before tool preflight begins. That means `before_tool_call` sees agent state that already includes the assistant message that requested the tool call.

### continue_() Event Sequence

`continue_()` resumes from existing context without adding a new message. Use it for retries after errors.

```python
await agent.continue_()
```

The last message in context must be `user` or `toolResult`, not `assistant`. Python uses `continue_()` because `continue` is a keyword.

### Event Types

| Event | Description |
|-------|-------------|
| `agent_start` | Agent begins processing |
| `agent_end` | Final event for the run. Awaited subscribers for this event still count toward settlement |
| `turn_start` | New turn begins: one LLM call plus tool executions |
| `turn_end` | Turn completes with assistant message and tool results |
| `message_start` | Any message begins: user, assistant, or toolResult |
| `message_update` | Assistant only. Includes `assistant_message_event` with the delta |
| `message_end` | Message completes |
| `tool_execution_start` | Tool begins |
| `tool_execution_update` | Tool streams progress |
| `tool_execution_end` | Tool completes |

`Agent.subscribe()` listeners are awaited in registration order. `agent_end` means no more loop events will be emitted, but `await agent.wait_for_idle()` and `await agent.prompt(...)` settle only after awaited `agent_end` listeners finish.

## Agent Options

```python
from pi_agent import Agent, MutableAgentState

agent = Agent(
    stream_fn,
    initial_state=MutableAgentState(
        system_prompt="You are a helpful assistant.",
        model=model,
        thinking_level="off",
        tools=[my_tool],
        messages=[],
    ),
    convert_to_llm=lambda messages: [m for m in messages if m.role in ("user", "assistant", "toolResult")],
    transform_context=lambda messages, signal: prune_old_messages(messages),
    steering_mode="one-at-a-time",
    follow_up_mode="one-at-a-time",
    session_id="session-123",
    get_api_key=lambda provider: refresh_token(provider),
    tool_execution="parallel",
    before_tool_call=before_tool_call,
    after_tool_call=after_tool_call,
    should_stop_after_turn=should_stop_after_turn,
    thinking_budgets={"minimal": 128, "low": 512, "medium": 1024, "high": 2048},
)
```

## Agent State

```python
from pi_agent import MutableAgentState

state = MutableAgentState()
state.system_prompt = "You are helpful."
state.thinking_level = "medium"
state.tools = [my_tool]
state.messages = []
```

`agent.state` exposes `MutableAgentState`.

Assigning `agent.state.tools = [...]` or `agent.state.messages = [...]` copies the top-level list before storing it. Mutating the returned list mutates current agent state.

During streaming, `agent.state.streaming_message` contains the current partial assistant message.

`agent.state.is_streaming` remains true until the run fully settles, including awaited `agent_end` subscribers.

## Methods

### Prompting

```python
from pi_ai import ImageContent, TextContent, UserMessage, now_ms

await agent.prompt("Hello")

await agent.prompt(
    "What's in this image?",
    [ImageContent(data=base64_data, mime_type="image/jpeg")],
)

await agent.prompt(UserMessage(content=[TextContent(text="Hello")], timestamp=now_ms()))

await agent.continue_()
```

### State Management

```python
agent.state.system_prompt = "New prompt"
agent.state.model = model
agent.state.thinking_level = "medium"
agent.state.tools = [my_tool]
agent.tool_execution = "sequential"
agent.before_tool_call = before_tool_call
agent.after_tool_call = after_tool_call
agent.should_stop_after_turn = should_stop_after_turn
agent.state.messages = new_messages
agent.state.messages.append(message)
agent.reset()
```

### Session and Thinking Budgets

```python
agent.session_id = "session-123"
agent.thinking_budgets = {
    "minimal": 128,
    "low": 512,
    "medium": 1024,
    "high": 2048,
}
```

### Control

```python
agent.abort()
await agent.wait_for_idle()
```

### Events

```python
async def listener(event, signal) -> None:
    if event.type == "agent_end":
        await flush_session_state(signal)


unsubscribe = agent.subscribe(listener)
unsubscribe()
```

## Steering and Follow-up

Steering messages interrupt the agent after the current assistant turn finishes. Follow-up messages queue work after the agent would otherwise stop.

```python
from pi_ai import TextContent, UserMessage, now_ms

agent.steering_mode = "one-at-a-time"
agent.follow_up_mode = "one-at-a-time"

agent.steer(UserMessage(content=[TextContent(text="Stop! Do this instead.")], timestamp=now_ms()))
agent.follow_up(UserMessage(content=[TextContent(text="Also summarize the result.")], timestamp=now_ms()))

steering_mode = agent.steering_mode
follow_up_mode = agent.follow_up_mode

agent.clear_steering_queue()
agent.clear_follow_up_queue()
agent.clear_all_queues()
```

When steering messages are detected after a turn completes:

1. All tool calls from the current assistant message have already finished.
2. Steering messages are injected.
3. The LLM responds on the next turn.

Follow-up messages are checked only when there are no more tool calls and no steering messages.

## Custom Message Types

Python has no declaration merging. Pass custom message objects if they have the fields your callbacks expect, and provide `convert_to_llm` to map or remove them before the provider call.

```python
from dataclasses import dataclass
from typing import Literal


@dataclass
class NotificationMessage:
    text: str
    timestamp: int
    role: Literal["notification"] = "notification"


def convert_to_llm(messages):
    return [message for message in messages if message.role != "notification"]


agent.convert_to_llm = convert_to_llm
agent.state.messages.append(NotificationMessage(text="Info", timestamp=now_ms()))
```

## Tools

Define tools with `AgentTool`:

```python
from pathlib import Path

from pi_agent import AgentTool, AgentToolResult
from pi_ai import TextContent


async def read_file(tool_call_id, params, signal=None, on_update=None):
    path = Path(params["path"])
    if on_update is not None:
        on_update(AgentToolResult(content=[TextContent(text="Reading...")], details={}))
    content = path.read_text(encoding="utf-8")
    return AgentToolResult(
        content=[TextContent(text=content)],
        details={"path": str(path), "size": len(content)},
    )


read_file_tool = AgentTool(
    name="read_file",
    label="Read File",
    description="Read a file's contents",
    parameters={
        "type": "object",
        "properties": {"path": {"type": "string", "description": "File path"}},
        "required": ["path"],
    },
    execution_mode="sequential",
    execute=read_file,
)

agent.state.tools = [read_file_tool]
```

### Error Handling

Raise an exception when a tool fails. Do not return error messages as successful content.

```python
from pathlib import Path

from pi_agent import AgentToolResult
from pi_ai import TextContent


async def read_required_file(tool_call_id, params, signal=None, on_update=None):
    path = Path(params["path"])
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return AgentToolResult(content=[TextContent(text=path.read_text(encoding="utf-8"))])
```

Thrown exceptions are caught by the agent and reported to the LLM as tool errors with `is_error=True`.

## Proxy Usage

For apps that proxy model calls through a backend:

```python
from pi_agent import Agent, ProxyStreamOptions, stream_proxy


def proxy_stream_fn(model, context, options=None):
    return stream_proxy(
        model,
        context,
        ProxyStreamOptions(
            auth_token="...",
            proxy_url="https://your-server.com",
            signal=getattr(options, "signal", None),
        ),
    )


agent = Agent(proxy_stream_fn)
```

## Low-Level API

For direct control without the `Agent` class:

```python
from pi_agent import AgentContext, AgentLoopConfig, agent_loop, agent_loop_continue

context = AgentContext(system_prompt="You are helpful.", messages=[], tools=[])
config = AgentLoopConfig(
    model=model,
    convert_to_llm=lambda messages: [m for m in messages if m.role in ("user", "assistant", "toolResult")],
    tool_execution="parallel",
    before_tool_call=before_tool_call,
    after_tool_call=after_tool_call,
)

stream = agent_loop([user_message], context, config, None, stream_fn)
async for event in stream:
    print(event.type)

continuation = agent_loop_continue(context, config, None, stream_fn)
async for event in continuation:
    print(event.type)
```

These low-level streams are observational. They preserve event order, but they do not wait for your async event handling to settle before later producer phases continue. If message processing must be a barrier before tool preflight, use `Agent`.

## Harness and sessions

`pi_agent.harness` ports the session entry/record model, in-memory and JSONL v4 session stores, telemetry schema, prompt-template/skill helpers, compaction helpers, and the durable harness surface types.

The durable multi-lane `AgentHarness` runtime is not implemented in either upstream TypeScript or this Python port. Its configuration accessors exist; operations that would need the durable run/compaction/navigation engine raise `HarnessNotImplemented`. See `docs/harness.md` for the design specification.

## License

MIT

---

`pp-agent-core` is developed in [HSPK/pp_agent_core](https://github.com/HSPK/pp_agent_core). It was split out of the `pp` monorepo; sibling packages (`pp-ai`, `pp-agent-core`, `pp-tui`, `pp-coding-agent`, ...) each live in their own
repository and are consumed from PyPI.
