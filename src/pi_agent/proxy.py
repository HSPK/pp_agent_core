"""Proxy stream function for apps that route LLM calls through a server.

Python port of `packages/agent/src/proxy.ts`.

The server owns provider auth and forwards each request upstream, streaming
the assistant events back over SSE with the `partial` field stripped to save
bandwidth. This module rebuilds the partial `AssistantMessage` client-side so
consumers see exactly the same `AssistantMessageEvent` sequence they would
get from a direct provider call.

TypeScript's `ReadableStreamDefaultReader` + `TextDecoder` loop becomes an
`httpx` streaming response iterated line by line, and the fire-and-forget
async IIFE becomes an `asyncio` task owned by the returned stream.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import httpx
from pi_ai.types import (
    AssistantMessage,
    AssistantMessageEvent,
    Context,
    Cost,
    DoneEvent,
    ErrorEvent,
    Model,
    StartEvent,
    TextContent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ThinkingContent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ThinkingStartEvent,
    ToolCall,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    Usage,
    now_ms,
)
from pi_ai.utils.abort import AbortSignal
from pi_ai.utils.event_stream import AssistantMessageEventStream
from pi_ai.utils.http import build_timeout
from pi_ai.utils.json_parse import parse_streaming_json

PROXY_SERIALIZABLE_OPTION_KEYS = (
    "temperature",
    "samplingParams",
    "maxTokens",
    "reasoning",
    "cacheRetention",
    "sessionId",
    "headers",
    "metadata",
    "transport",
    "thinkingBudgets",
    "maxRetryDelayMs",
)
"""The `SimpleStreamOptions` subset the proxy protocol puts on the wire.

Wire keys stay camelCase so a Python client interoperates with the TypeScript
proxy server unchanged.
"""


class ProxyMessageEventStream(AssistantMessageEventStream):
    """The stream `stream_proxy` returns. Same completion contract as a direct call."""


@dataclass
class ProxyStreamOptions:
    """Options for :func:`stream_proxy`.

    `auth_token` and `proxy_url` are proxy-specific; every other field is a
    pass-through `SimpleStreamOptions` value forwarded to the server.
    """

    auth_token: str
    proxy_url: str
    signal: AbortSignal | None = None
    temperature: float | None = None
    sampling_params: dict[str, Any] | None = None
    max_tokens: int | None = None
    reasoning: str | None = None
    cache_retention: str | None = None
    session_id: str | None = None
    headers: dict[str, str] | None = None
    metadata: dict[str, Any] | None = None
    transport: str | None = None
    thinking_budgets: dict[str, int] | None = None
    max_retry_delay_ms: int | None = None
    timeout_ms: int | None = None


class ProxyError(Exception):
    """The proxy server rejected the request or the transport failed."""


def build_proxy_request_options(options: ProxyStreamOptions) -> dict[str, Any]:
    """The serializable `SimpleStreamOptions` subset, in the wire's camelCase."""
    values = {
        "temperature": options.temperature,
        "samplingParams": options.sampling_params,
        "maxTokens": options.max_tokens,
        "reasoning": options.reasoning,
        "cacheRetention": options.cache_retention,
        "sessionId": options.session_id,
        "headers": options.headers,
        "metadata": options.metadata,
        "transport": options.transport,
        "thinkingBudgets": options.thinking_budgets,
        "maxRetryDelayMs": options.max_retry_delay_ms,
    }
    return {key: value for key, value in values.items() if value is not None}


@dataclass
class _PartialToolCall:
    """A tool call still accumulating streamed JSON.

    TypeScript smuggles `partialJson` onto the `ToolCall` object and deletes it
    on `toolcall_end`; Python keeps it beside the call instead.
    """

    call: ToolCall
    partial_json: str = ""


@dataclass
class _ProxyState:
    partial: AssistantMessage
    tool_calls: dict[int, _PartialToolCall] = field(default_factory=dict)


def _usage_from_wire(value: Any) -> Usage:
    if not isinstance(value, dict):
        return Usage()
    cost = value.get("cost") or {}
    return Usage(
        input=value.get("input", 0),
        output=value.get("output", 0),
        cache_read=value.get("cacheRead", 0),
        cache_write=value.get("cacheWrite", 0),
        cache_write_1h=value.get("cacheWrite1h"),
        reasoning=value.get("reasoning"),
        total_tokens=value.get("totalTokens", 0),
        cost=Cost(
            input=cost.get("input", 0.0),
            output=cost.get("output", 0.0),
            cache_read=cost.get("cacheRead", 0.0),
            cache_write=cost.get("cacheWrite", 0.0),
            total=cost.get("total", 0.0),
        ),
    )


def _set_content(partial: AssistantMessage, index: int, value: Any) -> None:
    """Assign at `index`, growing the list like JavaScript's sparse arrays do."""
    while len(partial.content) <= index:
        partial.content.append(TextContent(text=""))
    partial.content[index] = value


def process_proxy_event(proxy_event: dict[str, Any], state: _ProxyState) -> AssistantMessageEvent | None:
    """Apply one wire event to the partial message and return the local event."""
    partial = state.partial
    event_type = proxy_event.get("type")
    index = proxy_event.get("contentIndex", 0)

    if event_type == "start":
        return StartEvent(partial=partial)

    if event_type == "text_start":
        _set_content(partial, index, TextContent(text=""))
        return TextStartEvent(content_index=index, partial=partial)

    if event_type == "text_delta":
        content = partial.content[index] if index < len(partial.content) else None
        if not isinstance(content, TextContent):
            raise ProxyError("Received text_delta for non-text content")
        content.text += proxy_event.get("delta", "")
        return TextDeltaEvent(content_index=index, delta=proxy_event.get("delta", ""), partial=partial)

    if event_type == "text_end":
        content = partial.content[index] if index < len(partial.content) else None
        if not isinstance(content, TextContent):
            raise ProxyError("Received text_end for non-text content")
        content.text_signature = proxy_event.get("contentSignature")
        return TextEndEvent(content_index=index, content=content.text, partial=partial)

    if event_type == "thinking_start":
        _set_content(partial, index, ThinkingContent(thinking=""))
        return ThinkingStartEvent(content_index=index, partial=partial)

    if event_type == "thinking_delta":
        content = partial.content[index] if index < len(partial.content) else None
        if not isinstance(content, ThinkingContent):
            raise ProxyError("Received thinking_delta for non-thinking content")
        content.thinking += proxy_event.get("delta", "")
        return ThinkingDeltaEvent(content_index=index, delta=proxy_event.get("delta", ""), partial=partial)

    if event_type == "thinking_end":
        content = partial.content[index] if index < len(partial.content) else None
        if not isinstance(content, ThinkingContent):
            raise ProxyError("Received thinking_end for non-thinking content")
        content.thinking_signature = proxy_event.get("contentSignature")
        return ThinkingEndEvent(content_index=index, content=content.thinking, partial=partial)

    if event_type == "toolcall_start":
        call = ToolCall(id=proxy_event.get("id", ""), name=proxy_event.get("toolName", ""), arguments={})
        state.tool_calls[index] = _PartialToolCall(call=call)
        _set_content(partial, index, call)
        return ToolCallStartEvent(content_index=index, partial=partial)

    if event_type == "toolcall_delta":
        pending = state.tool_calls.get(index)
        if pending is None:
            raise ProxyError("Received toolcall_delta for non-toolCall content")
        pending.partial_json += proxy_event.get("delta", "")
        pending.call.arguments = parse_streaming_json(pending.partial_json) or {}
        return ToolCallDeltaEvent(content_index=index, delta=proxy_event.get("delta", ""), partial=partial)

    if event_type == "toolcall_end":
        pending = state.tool_calls.pop(index, None)
        if pending is None:
            return None
        final = proxy_event.get("toolCall") or {}
        pending.call.id = final.get("id", pending.call.id)
        pending.call.name = final.get("name", pending.call.name)
        pending.call.arguments = final.get("arguments", pending.call.arguments)
        pending.call.thought_signature = final.get("thoughtSignature", pending.call.thought_signature)
        pending.call.namespace = final.get("namespace", pending.call.namespace)
        return ToolCallEndEvent(content_index=index, tool_call=pending.call, partial=partial)

    if event_type == "done":
        partial.stop_reason = proxy_event.get("reason", "stop")
        partial.usage = _usage_from_wire(proxy_event.get("usage"))
        return DoneEvent(reason=partial.stop_reason, message=partial)

    if event_type == "error":
        partial.stop_reason = proxy_event.get("reason", "error")
        partial.error_message = proxy_event.get("errorMessage")
        partial.usage = _usage_from_wire(proxy_event.get("usage"))
        return ErrorEvent(reason=partial.stop_reason, error=partial)

    return None


def stream_proxy(model: Model, context: Context, options: ProxyStreamOptions) -> ProxyMessageEventStream:
    """Stream an assistant message through a proxy server instead of a provider.

    Pass this as an `Agent`'s `stream_fn` when requests must be brokered by a
    server that holds the provider credentials.
    """
    stream = ProxyMessageEventStream()
    state = _ProxyState(
        partial=AssistantMessage(
            api=model.api,
            provider=model.provider,
            model=model.id,
            content=[],
            usage=Usage(),
            stop_reason="pending",
            timestamp=now_ms(),
        )
    )

    async def run() -> None:
        try:
            payload = {
                "model": asdict(model),
                "context": asdict(context),
                "options": build_proxy_request_options(options),
            }
            headers = {
                "Authorization": f"Bearer {options.auth_token}",
                "Content-Type": "application/json",
            }
            timeout = build_timeout(options.timeout_ms)
            async with (
                httpx.AsyncClient(timeout=timeout) as client,
                client.stream("POST", f"{options.proxy_url}/api/stream", headers=headers, json=payload) as response,
            ):
                if response.status_code >= 400:
                    body = await response.aread()
                    message = f"Proxy error: {response.status_code} {response.reason_phrase}"
                    try:
                        error_data = json.loads(body)
                        if isinstance(error_data, dict) and error_data.get("error"):
                            message = f"Proxy error: {error_data['error']}"
                    except ValueError:
                        pass
                    raise ProxyError(message)

                async for line in response.aiter_lines():
                    if options.signal is not None and options.signal.aborted:
                        raise ProxyError("Request aborted by user")
                    if not line.startswith("data: "):
                        continue
                    data = line[6:].strip()
                    if not data:
                        continue
                    event = process_proxy_event(json.loads(data), state)
                    if event is not None:
                        stream.push(event)

            if options.signal is not None and options.signal.aborted:
                raise ProxyError("Request aborted by user")
            stream.end()
        except asyncio.CancelledError:
            _fail(stream, state, "Request aborted by user", "aborted")
            raise
        except Exception as error:
            aborted = options.signal is not None and options.signal.aborted
            _fail(stream, state, str(error), "aborted" if aborted else "error")

    task = asyncio.ensure_future(run())

    if options.signal is not None:
        signal = options.signal

        async def cancel_on_abort() -> None:
            """TypeScript attaches an `abort` listener; Python waits on the event instead."""
            await signal.wait()
            if not task.done():
                task.cancel()

        watcher = asyncio.ensure_future(cancel_on_abort())
        task.add_done_callback(lambda _: watcher.cancel())

    return stream


def _fail(
    stream: ProxyMessageEventStream,
    state: _ProxyState,
    message: str,
    reason: Literal["aborted", "error"],
) -> None:
    state.partial.stop_reason = reason
    state.partial.error_message = message
    stream.push(ErrorEvent(reason=reason, error=state.partial))
    stream.end()


__all__ = [
    "PROXY_SERIALIZABLE_OPTION_KEYS",
    "ProxyError",
    "ProxyMessageEventStream",
    "ProxyStreamOptions",
    "build_proxy_request_options",
    "process_proxy_event",
    "stream_proxy",
]
