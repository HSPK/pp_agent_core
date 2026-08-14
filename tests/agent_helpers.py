"""Shared fixtures for pi_agent tests."""

from __future__ import annotations

from collections.abc import Callable

from pi_ai import (
    AssistantMessage,
    AssistantMessageEventStream,
    DoneEvent,
    ErrorEvent,
    Model,
    StartEvent,
    TextContent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ToolCall,
    ToolCallEndEvent,
    ToolCallStartEvent,
    Usage,
)

TEST_MODEL = Model(
    id="test-model",
    name="Test Model",
    api="test-api",
    provider="test",
    base_url="https://example.invalid",
    context_window=1000,
    max_tokens=100,
)


def make_assistant_message(
    content: list, stop_reason: str = "stop", error_message: str | None = None
) -> AssistantMessage:
    return AssistantMessage(
        api=TEST_MODEL.api,
        provider=TEST_MODEL.provider,
        model=TEST_MODEL.id,
        content=content,
        usage=Usage(),
        stop_reason=stop_reason,
        error_message=error_message,
    )


def text_response(text: str) -> AssistantMessage:
    return make_assistant_message([TextContent(text=text)], stop_reason="stop")


def tool_call_response(tool_call: ToolCall) -> AssistantMessage:
    return make_assistant_message([tool_call], stop_reason="toolUse")


def error_response(message: str, stop_reason: str = "error") -> AssistantMessage:
    return make_assistant_message([], stop_reason=stop_reason, error_message=message)


def replay_stream(message: AssistantMessage) -> AssistantMessageEventStream:
    """Emit the protocol event sequence that produces ``message``."""
    stream = AssistantMessageEventStream()
    partial = message
    stream.push(StartEvent(partial=partial))

    for index, block in enumerate(message.content):
        if block.type == "text":
            stream.push(TextStartEvent(content_index=index, partial=partial))
            stream.push(TextDeltaEvent(content_index=index, delta=block.text, partial=partial))
            stream.push(TextEndEvent(content_index=index, content=block.text, partial=partial))
        elif block.type == "toolCall":
            stream.push(ToolCallStartEvent(content_index=index, partial=partial))
            stream.push(ToolCallEndEvent(content_index=index, tool_call=block, partial=partial))

    if message.stop_reason in ("error", "aborted"):
        stream.push(ErrorEvent(reason=message.stop_reason, error=message))
    else:
        stream.push(DoneEvent(reason=message.stop_reason, message=message))
    stream.end()
    return stream


def scripted_stream_fn(responses: list[AssistantMessage]) -> Callable:
    """A ``StreamFn`` that replays ``responses`` one per call."""
    remaining = list(responses)
    calls: list = []

    def stream_fn(model, context, options=None):
        calls.append({"model": model, "context": context, "options": options})
        if not remaining:
            raise AssertionError("stream_fn called more times than there are scripted responses")
        return replay_stream(remaining.pop(0))

    stream_fn.calls = calls  # type: ignore[attr-defined]
    return stream_fn
