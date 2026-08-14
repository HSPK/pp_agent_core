"""Tests for `pi_agent.proxy`.

No dedicated TS test file exists for `packages/agent/src/proxy.ts`. These
cover the wire-event reducer directly (which is where all the state rebuilding
lives) plus one end-to-end pass over a canned SSE body served by a stub
transport, so the streaming loop, error mapping and abort path are exercised
without a network.
"""

from __future__ import annotations

import json

import httpx
import pytest
from pi_ai.types import AssistantMessage, Context, Model, TextContent, ThinkingContent, ToolCall, Usage, UserMessage
from pi_ai.utils.abort import AbortSignal

from pi_agent.proxy import (
    ProxyStreamOptions,
    _ProxyState,
    build_proxy_request_options,
    process_proxy_event,
    stream_proxy,
)


def make_state() -> _ProxyState:
    return _ProxyState(
        partial=AssistantMessage(api="openai-completions", provider="test", model="m", content=[], usage=Usage())
    )


def test_text_events_rebuild_the_partial_message():
    state = make_state()

    process_proxy_event({"type": "start"}, state)
    process_proxy_event({"type": "text_start", "contentIndex": 0}, state)
    process_proxy_event({"type": "text_delta", "contentIndex": 0, "delta": "Hel"}, state)
    event = process_proxy_event({"type": "text_delta", "contentIndex": 0, "delta": "lo"}, state)

    assert event.type == "text_delta"
    assert event.delta == "lo"
    assert state.partial.content[0] == TextContent(text="Hello")

    end = process_proxy_event({"type": "text_end", "contentIndex": 0, "contentSignature": "sig"}, state)
    assert end.content == "Hello"
    assert state.partial.content[0].text_signature == "sig"


def test_thinking_events_rebuild_the_partial_message():
    state = make_state()

    process_proxy_event({"type": "thinking_start", "contentIndex": 0}, state)
    process_proxy_event({"type": "thinking_delta", "contentIndex": 0, "delta": "hmm"}, state)
    end = process_proxy_event({"type": "thinking_end", "contentIndex": 0, "contentSignature": "ts"}, state)

    assert state.partial.content[0] == ThinkingContent(thinking="hmm", thinking_signature="ts")
    assert end.content == "hmm"


def test_tool_call_arguments_are_parsed_from_streamed_json():
    state = make_state()

    process_proxy_event({"type": "toolcall_start", "contentIndex": 0, "id": "c1", "toolName": "read"}, state)
    process_proxy_event({"type": "toolcall_delta", "contentIndex": 0, "delta": '{"path": "a'}, state)

    # Partial JSON still yields a usable argument object mid-stream.
    assert state.partial.content[0].arguments == {"path": "a"}

    process_proxy_event({"type": "toolcall_delta", "contentIndex": 0, "delta": '.txt"}'}, state)
    event = process_proxy_event(
        {
            "type": "toolcall_end",
            "contentIndex": 0,
            "toolCall": {"id": "c1", "name": "read", "arguments": {"path": "a.txt"}},
        },
        state,
    )

    assert event.tool_call == ToolCall(id="c1", name="read", arguments={"path": "a.txt"})
    assert state.tool_calls == {}


def test_toolcall_end_without_a_start_is_ignored():
    state = make_state()
    assert process_proxy_event({"type": "toolcall_end", "contentIndex": 3, "toolCall": {}}, state) is None


def test_delta_for_mismatched_content_raises():
    state = make_state()
    process_proxy_event({"type": "text_start", "contentIndex": 0}, state)

    with pytest.raises(Exception, match="non-thinking"):
        process_proxy_event({"type": "thinking_delta", "contentIndex": 0, "delta": "x"}, state)


def test_done_applies_usage_and_stop_reason():
    state = make_state()
    event = process_proxy_event(
        {
            "type": "done",
            "reason": "stop",
            "usage": {"input": 10, "output": 5, "totalTokens": 15, "cost": {"total": 0.5}},
        },
        state,
    )

    assert event.type == "done"
    assert state.partial.stop_reason == "stop"
    assert state.partial.usage.input == 10
    assert state.partial.usage.cost.total == 0.5


def test_error_event_records_message_and_reason():
    state = make_state()
    event = process_proxy_event({"type": "error", "reason": "error", "errorMessage": "boom", "usage": {}}, state)

    assert event.type == "error"
    assert state.partial.stop_reason == "error"
    assert state.partial.error_message == "boom"


def test_unknown_event_types_are_ignored():
    assert process_proxy_event({"type": "not_a_real_event"}, make_state()) is None


def test_request_options_drop_unset_values_and_use_wire_casing():
    options = ProxyStreamOptions(auth_token="t", proxy_url="http://x", max_tokens=100, session_id="s1")

    assert build_proxy_request_options(options) == {"maxTokens": 100, "sessionId": "s1"}


def _sse_body(events: list[dict]) -> bytes:
    return "".join(f"data: {json.dumps(event)}\n\n" for event in events).encode()


def _install_transport(monkeypatch, handler) -> None:
    """Route every `httpx.AsyncClient` in `pi_agent.proxy` at a stub transport."""
    original = httpx.AsyncClient

    def build(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return original(*args, **kwargs)

    monkeypatch.setattr("pi_agent.proxy.httpx.AsyncClient", build)


@pytest.mark.asyncio
async def test_stream_proxy_streams_a_full_message(monkeypatch):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers["Authorization"]
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            content=_sse_body(
                [
                    {"type": "start"},
                    {"type": "text_start", "contentIndex": 0},
                    {"type": "text_delta", "contentIndex": 0, "delta": "hi"},
                    {"type": "text_end", "contentIndex": 0},
                    {"type": "done", "reason": "stop", "usage": {"input": 1, "output": 2, "totalTokens": 3}},
                ]
            ),
        )

    _install_transport(monkeypatch, handler)

    model = Model(id="m", api="openai-completions", provider="test")
    context = Context(messages=[UserMessage(content="hello")])
    stream = stream_proxy(model, context, ProxyStreamOptions(auth_token="tok", proxy_url="http://proxy"))

    types = [event.type async for event in stream]
    message = await stream.result()

    assert types == ["start", "text_start", "text_delta", "text_end", "done"]
    assert message.content == [TextContent(text="hi")]
    assert message.stop_reason == "stop"
    assert message.usage.total_tokens == 3
    assert captured["url"] == "http://proxy/api/stream"
    assert captured["auth"] == "Bearer tok"
    assert captured["body"]["model"]["id"] == "m"


@pytest.mark.asyncio
async def test_stream_proxy_reports_a_server_error_as_an_error_event(monkeypatch):
    _install_transport(monkeypatch, lambda request: httpx.Response(500, json={"error": "upstream exploded"}))

    stream = stream_proxy(
        Model(id="m", provider="test"),
        Context(messages=[]),
        ProxyStreamOptions(auth_token="t", proxy_url="http://proxy"),
    )

    events = [event async for event in stream]

    assert [event.type for event in events] == ["error"]
    assert events[0].reason == "error"
    assert "upstream exploded" in events[0].error.error_message


@pytest.mark.asyncio
async def test_stream_proxy_reports_abort_when_the_signal_fires(monkeypatch):
    signal = AbortSignal()

    def handler(request: httpx.Request) -> httpx.Response:
        signal.abort()
        raise httpx.ReadError("connection dropped")

    _install_transport(monkeypatch, handler)

    stream = stream_proxy(
        Model(id="m", provider="test"),
        Context(messages=[]),
        ProxyStreamOptions(auth_token="t", proxy_url="http://proxy", signal=signal),
    )

    events = [event async for event in stream]

    assert [event.type for event in events] == ["error"]
    assert events[0].reason == "aborted"
