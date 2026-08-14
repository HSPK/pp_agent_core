"""Python port of `packages/agent/test/proxy.test.ts`.

Named ``test_proxy_port`` because ``test_proxy.py`` already holds this port's
own proxy suite; this file is the direct translation of the upstream test.

TypeScript stubs the global ``fetch``; the Python port routes
``pi_agent.proxy``'s ``httpx.AsyncClient`` at an in-process mock transport, so
no network call is made either.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from pi_agent.proxy import ProxyStreamOptions, stream_proxy
from pi_ai.types import Context, Model, ModelCost, Usage

MODEL = Model(
    id="gpt-5.4",
    name="GPT-5.4",
    api="openai-responses",
    provider="openai",
    base_url="https://api.openai.com/v1",
    reasoning=True,
    input=["text"],
    cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
    context_window=400000,
    max_tokens=128000,
)

USAGE: dict[str, Any] = {
    "input": 0,
    "output": 0,
    "cacheRead": 0,
    "cacheWrite": 0,
    "totalTokens": 0,
    "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0},
}


def _install_transport(monkeypatch: pytest.MonkeyPatch, body: bytes) -> None:
    original = httpx.AsyncClient

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    def build(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = httpx.MockTransport(handler)
        return original(*args, **kwargs)

    monkeypatch.setattr("pi_agent.proxy.httpx.AsyncClient", build)


@pytest.mark.asyncio
async def test_preserves_tool_call_metadata_received_only_on_toolcall_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proxy_events: list[dict[str, Any]] = [
        {"type": "start"},
        {"type": "toolcall_start", "contentIndex": 0, "id": "call_test|fc_test", "toolName": "lookup"},
        {"type": "toolcall_delta", "contentIndex": 0, "delta": '{"value":"hello"}'},
        {
            "type": "toolcall_end",
            "contentIndex": 0,
            "toolCall": {
                "type": "toolCall",
                "id": "call_test|fc_test",
                "name": "lookup",
                "arguments": {"value": "hello"},
                "namespace": "dynamic_tools",
            },
        },
        {"type": "done", "reason": "toolUse", "usage": USAGE},
    ]
    body = "".join(f"data: {json.dumps(event)}\n\n" for event in proxy_events).encode()
    _install_transport(monkeypatch, body)

    stream = stream_proxy(
        MODEL,
        Context(system_prompt="", messages=[]),
        ProxyStreamOptions(auth_token="test-token", proxy_url="https://proxy.example.com"),
    )

    events = [event async for event in stream]
    result = await stream.result()
    end_event = next(event for event in events if event.type == "toolcall_end")

    assert end_event.type == "toolcall_end"
    assert end_event.tool_call.namespace == "dynamic_tools"

    assert result.content[0].type == "toolCall"
    assert result.content[0].arguments == {"value": "hello"}
    assert result.content[0].namespace == "dynamic_tools"
    assert result.usage == Usage()
