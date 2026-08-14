"""Python port of `packages/agent/test/harness/session/context.test.ts`."""

from __future__ import annotations

from typing import Any

from pi_ai.types import (
    AssistantMessage,
    Cost,
    DeferredHandle,
    TextContent,
    Usage,
    UserMessage,
)

from pi_agent.harness.session.context import (
    SessionContextBuildOptions,
    SessionContextModel,
    build_session_context,
)
from pi_agent.harness.session.types import (
    BranchSummaryEntry,
    CompactionEntry,
    CustomEntry,
    Entry,
    MessageEntry,
    ModelChangeEntry,
    ThinkingLevelEntry,
)


def user_message(text: str) -> UserMessage:
    return UserMessage(content=[TextContent(text=text)], timestamp=1)


def assistant_message(text: str) -> AssistantMessage:
    return AssistantMessage(
        content=[TextContent(text=text)],
        api="anthropic-messages",
        provider="anthropic",
        model="claude-sonnet-4-5",
        usage=Usage(
            input=0,
            output=0,
            cache_read=0,
            cache_write=0,
            total_tokens=0,
            cost=Cost(input=0, output=0, cache_read=0, cache_write=0, total=0),
        ),
        stop_reason="stop",
        timestamp=1,
    )


def with_storage(entry: Entry, seq: int) -> Entry:
    """Fill in the storage-assigned fields the TypeScript helper stamps on."""
    entry.seq = seq
    entry.timestamp = seq
    return entry


def test_starts_at_the_latest_compaction_and_materializes_its_retained_tail() -> None:
    entries: list[Entry] = [
        with_storage(MessageEntry(id="old", parent_id=None, message=user_message("old")), 1),
        with_storage(
            CompactionEntry(
                id="compact",
                parent_id="old",
                summary="summary",
                retained_tail=[user_message("retained"), assistant_message("answer")],
                tokens_before=100,
            ),
            2,
        ),
        with_storage(ModelChangeEntry(id="model", parent_id="compact", provider="openai", model_id="gpt-5"), 3),
        with_storage(ThinkingLevelEntry(id="thinking", parent_id="model", thinking_level="high"), 4),
        with_storage(MessageEntry(id="tail", parent_id="thinking", message=user_message("tail")), 5),
    ]

    context = build_session_context(entries)
    assert [message.role for message in context.messages] == [
        "compactionSummary",
        "user",
        "assistant",
        "user",
    ]
    assert context.model == SessionContextModel(provider="openai", model_id="gpt-5")
    assert context.thinking_level == "high"


def test_applies_caller_transforms_after_the_compaction_boundary() -> None:
    entries: list[Entry] = [
        with_storage(MessageEntry(id="old", parent_id=None, message=user_message("old")), 1),
        with_storage(
            CompactionEntry(id="compact", parent_id="old", summary="summary", retained_tail=[], tokens_before=100),
            2,
        ),
        with_storage(
            BranchSummaryEntry(id="branch", parent_id="compact", from_id="abandoned", summary="branch summary"),
            3,
        ),
        with_storage(MessageEntry(id="tail", parent_id="branch", message=user_message("tail")), 4),
    ]

    context = build_session_context(
        entries,
        SessionContextBuildOptions(
            entry_transforms=[
                lambda context_entries: [candidate for candidate in context_entries if candidate.type != "compaction"]
            ]
        ),
    )
    assert [message.role for message in context.messages] == ["branchSummary", "user"]


def test_projects_custom_entries_and_omits_deferred_assistant_handles() -> None:
    deferred = assistant_message("")
    deferred.content = []
    deferred.stop_reason = "deferred"
    deferred.deferred = DeferredHandle(provider="openai", model_id="gpt-5", api="openai-responses", id="response-1")

    entries: list[Entry] = [
        with_storage(MessageEntry(id="user", parent_id=None, message=user_message("hello")), 1),
        with_storage(MessageEntry(id="deferred", parent_id="user", message=deferred), 2),
        with_storage(CustomEntry(id="custom", parent_id="deferred", custom_type="note", data="project me"), 3),
    ]

    def project_note(custom: CustomEntry, index: int, all_entries: Any) -> list[UserMessage]:
        return [user_message(f"note: {custom.data}")]

    context = build_session_context(entries, SessionContextBuildOptions(entry_projectors={"note": project_note}))
    assert [message.role for message in context.messages] == ["user", "user"]
    assert context.messages[1].content == [TextContent(text="note: project me")]
