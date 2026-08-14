"""Shared constructors and assertions for the session backend conformance suite.

Python port of the helper functions in
`packages/agent/src/harness/session/testing/conformance.ts` (lines 1-70). The
conformance *cases* themselves live in `test_session_conformance.py`.
"""

from __future__ import annotations

from collections.abc import Coroutine
from typing import Any, Literal

import pytest
from pi_ai import AssistantMessage, Cost, TextContent, Usage, UserMessage

from pi_agent.harness.session import (
    CompactionIntent,
    NavigationIntent,
    OperationStartedRecord,
    RunIntent,
    SessionError,
)


def create_user_message(text: str) -> UserMessage:
    return UserMessage(content=[TextContent(text=text)], timestamp=1)


def create_assistant_message(text: str) -> AssistantMessage:
    return AssistantMessage(
        content=[TextContent(text=text)],
        api="anthropic-messages",
        provider="anthropic",
        model="claude-sonnet-4-5",
        usage=Usage(input=0, output=0, cache_read=0, cache_write=0, total_tokens=0, cost=Cost()),
        stop_reason="stop",
        timestamp=1,
    )


def operation_started(
    id: str, *, lane: str, kind: Literal["run", "compaction", "navigation"]
) -> OperationStartedRecord:
    if kind == "run":
        intent: Any = RunIntent(original_prompt=[], initial_messages=[])
    elif kind == "compaction":
        intent = CompactionIntent(result_entry_id=f"{id}-result")
    else:
        intent = NavigationIntent(target_id=None, summarize=False)
    return OperationStartedRecord(id=id, lane=lane, source_leaf_id=None, intent=intent)


async def entry_ids(entries_coro: Coroutine[Any, Any, list[Any]]) -> list[str]:
    entries = await entries_coro
    return [entry.id for entry in entries]


async def assert_rejects_with_code(operation: Coroutine[Any, Any, Any], code: str) -> None:
    with pytest.raises(SessionError) as exc_info:
        await operation
    assert exc_info.value.code == code, f"Expected SessionError with code {code!r}, got {exc_info.value.code!r}"
