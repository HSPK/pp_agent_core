"""Tests for `pi_agent.harness.compaction.compaction`.

Ported from `packages/agent/test/harness/compaction.test.ts`. The TypeScript
suite drives summarization through `fauxProvider`/`createModels`; this port's
`compact`/`generate_summary`/`generate_summary_with_usage` take a `StreamFn`
plus a `pi_ai.types.Model` instead (see `compaction.py`'s module docstring),
so these tests use `agent_helpers.scripted_stream_fn` and build `Model` instances
directly rather than a `Models` registry.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
from agent_helpers import TEST_MODEL, scripted_stream_fn
from pi_ai.types import (
    AssistantMessage,
    ImageContent,
    Model,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)

from pi_agent.harness.compaction.compaction import (
    DEFAULT_COMPACTION_SETTINGS,
    CompactionPreparation,
    CompactionSettings,
    calculate_context_tokens,
    compact,
    estimate_context_tokens,
    estimate_tokens,
    find_cut_point,
    find_turn_start_index,
    generate_summary,
    generate_summary_with_usage,
    get_last_assistant_usage,
    prepare_compaction,
    serialize_conversation,
    should_compact,
)
from pi_agent.harness.compaction.utils import FileOperations
from pi_agent.harness.messages import BashExecutionMessage, CustomMessage
from pi_agent.harness.session.context import BranchSummaryMessage, CompactionSummaryMessage, build_session_context
from pi_agent.harness.session.types import (
    BranchSummaryEntry,
    CompactionEntry,
    Entry,
    MessageEntry,
    ModelChangeEntry,
    ThinkingLevelEntry,
)
from pi_agent.harness.types import get_or_throw

_next_id = 0


def _create_id() -> str:
    global _next_id
    _next_id += 1
    return f"entry-{_next_id}"


def _create_mock_usage(input_: int, output: int, cache_read: int = 0, cache_write: int = 0) -> Usage:
    return Usage(
        input=input_,
        output=output,
        cache_read=cache_read,
        cache_write=cache_write,
        total_tokens=input_ + output + cache_read + cache_write,
    )


def _create_user_message(text: str) -> UserMessage:
    return UserMessage(content=[TextContent(text=text)], timestamp=1)


def _create_assistant_message(text: str, usage: Usage | None = None) -> AssistantMessage:
    return AssistantMessage(
        api="anthropic-messages",
        provider="anthropic",
        model="claude-sonnet-4-5",
        content=[TextContent(text=text)],
        usage=usage if usage is not None else _create_mock_usage(100, 50),
        stop_reason="stop",
    )


def _create_message_entry(message, parent_id: str | None = None) -> MessageEntry:
    return MessageEntry(id=_create_id(), parent_id=parent_id, seq=_next_id, timestamp=1, message=message)


def _create_compaction_entry(summary: str, parent_id: str | None = None, retained_tail=None) -> CompactionEntry:
    return CompactionEntry(
        id=_create_id(),
        parent_id=parent_id,
        seq=_next_id,
        timestamp=1,
        summary=summary,
        tokens_before=1234,
        retained_tail=retained_tail or [],
    )


def _create_thinking_level_entry(level: str, parent_id: str | None = None) -> ThinkingLevelEntry:
    return ThinkingLevelEntry(id=_create_id(), parent_id=parent_id, seq=_next_id, timestamp=1, thinking_level=level)


def _create_model_change_entry(provider: str, model_id: str, parent_id: str | None = None) -> ModelChangeEntry:
    return ModelChangeEntry(
        id=_create_id(), parent_id=parent_id, seq=_next_id, timestamp=1, provider=provider, model_id=model_id
    )


def _make_model(reasoning: bool = False, max_tokens: int = 8192, context_window: int = 200000) -> Model:
    return Model(
        id="reasoning-model" if reasoning else "non-reasoning-model",
        api="test-api",
        provider="test",
        reasoning=reasoning,
        context_window=context_window,
        max_tokens=max_tokens,
    )


def _text_response_with_usage(text: str, usage: Usage) -> AssistantMessage:
    return AssistantMessage(
        api=TEST_MODEL.api,
        provider=TEST_MODEL.provider,
        model=TEST_MODEL.id,
        content=[TextContent(text=text)],
        usage=usage,
        stop_reason="stop",
    )


@pytest.fixture(autouse=True)
def _reset_id_counter():
    global _next_id
    _next_id = 0
    yield


def test_calculates_total_context_tokens_from_usage():
    assert calculate_context_tokens(_create_mock_usage(1000, 500, 200, 100)) == 1800
    assert calculate_context_tokens(_create_mock_usage(0, 0, 0, 0)) == 0


def test_checks_compaction_threshold():
    settings = CompactionSettings(enabled=True, reserve_tokens=10000, keep_recent_tokens=20000)
    assert should_compact(95000, 100000, settings) is True
    assert should_compact(89000, 100000, settings) is False
    assert should_compact(95000, 100000, replace(settings, enabled=False)) is False


def test_finds_a_cut_point_based_on_token_differences():
    entries: list[Entry] = []
    parent_id: str | None = None
    for i in range(10):
        user = _create_message_entry(_create_user_message(f"User {i}"), parent_id)
        entries.append(user)
        assistant = _create_message_entry(
            _create_assistant_message(f"Assistant {i}", _create_mock_usage(0, 100, (i + 1) * 1000, 0)), user.id
        )
        entries.append(assistant)
        parent_id = assistant.id

    result = find_cut_point(entries, 0, len(entries), 2500)
    assert entries[result.first_kept_entry_index].type == "message"


def test_covers_cut_point_and_turn_start_edge_cases():
    thinking = _create_thinking_level_entry("high")
    model_change = _create_model_change_entry("openai", "gpt-4", thinking.id)
    result = find_cut_point([thinking, model_change], 0, 2, 1)
    assert result.first_kept_entry_index == 0
    assert result.turn_start_index == -1
    assert result.is_split_turn is False

    branch_summary = BranchSummaryEntry(
        id=_create_id(),
        parent_id=model_change.id,
        seq=_next_id,
        timestamp=1,
        from_id="branch",
        summary="branch summary",
    )
    assert find_turn_start_index([thinking, branch_summary], 1, 0) == 1
    assert find_turn_start_index([thinking, model_change], 1, 0) == -1

    result2 = find_cut_point([thinking, branch_summary], 0, 2, 1)
    assert result2.first_kept_entry_index == 0

    tool_result = _create_message_entry(
        ToolResultMessage(
            tool_call_id="call-1", tool_name="read", content=[TextContent(text="tool output")], is_error=False
        )
    )
    result3 = find_cut_point([tool_result], 0, 1, 1)
    assert result3.first_kept_entry_index == 0
    assert result3.turn_start_index == -1
    assert result3.is_split_turn is False

    user = _create_message_entry(_create_user_message("user"))
    compaction = _create_compaction_entry("summary", user.id)
    assistant = _create_message_entry(_create_assistant_message("assistant"), compaction.id)
    assert find_cut_point([user, compaction, assistant], 0, 3, 1).first_kept_entry_index == 2


def test_estimates_tokens_and_context_usage_across_supported_message_roles():
    usage = _create_mock_usage(10, 5, 3, 2)
    assistant = _create_assistant_message("assistant", usage)
    assistant_with_thinking_and_tool = replace(
        assistant,
        content=[
            ThinkingContent(thinking="thinking"),
            ToolCall(id="call-1", name="read", arguments={"path": "file.ts"}),
        ],
    )
    custom_string = CustomMessage(custom_type="note", content="custom text", display=True, timestamp=1)
    tool_result_with_image = ToolResultMessage(
        tool_call_id="call-1",
        tool_name="read",
        content=[
            TextContent(text="tool text"),
            ImageContent(mime_type="image/png", data="abc"),
        ],
        is_error=False,
    )
    bash_execution = BashExecutionMessage(
        command="npm run check", output="ok", exit_code=0, cancelled=False, truncated=False, timestamp=1
    )
    branch_summary_message = BranchSummaryMessage(summary="branch", from_id="x", timestamp=1)
    compaction_summary_message = CompactionSummaryMessage(summary="compact", tokens_before=123, timestamp=1)

    assert estimate_tokens(UserMessage(content="plain user", timestamp=1)) > 0
    assert estimate_tokens(assistant_with_thinking_and_tool) > 0
    assert estimate_tokens(custom_string) > 0
    assert estimate_tokens(tool_result_with_image) > 1000
    assert estimate_tokens(bash_execution) > 0
    assert estimate_tokens(branch_summary_message) > 0
    assert estimate_tokens(compaction_summary_message) > 0
    # TypeScript casts `{ role: "unknown" }` through `AgentMessage` to reach the
    # fallback branch; `SimpleNamespace` is the closest Python equivalent.
    assert estimate_tokens(SimpleNamespace(role="unknown", timestamp=1)) == 0  # type: ignore[arg-type]

    assert (
        get_last_assistant_usage(
            [_create_message_entry(_create_user_message("user")), _create_message_entry(assistant)]
        )
        is usage
    )
    assert (
        get_last_assistant_usage(
            [
                _create_message_entry(replace(assistant, stop_reason="aborted")),
                _create_message_entry(replace(assistant, stop_reason="error")),
            ]
        )
        is None
    )
    assert (
        get_last_assistant_usage(
            [
                _create_message_entry(_create_user_message("user")),
                _create_message_entry(assistant),
                _create_message_entry(_create_assistant_message("partial", _create_mock_usage(0, 0))),
            ]
        )
        is usage
    )

    assert estimate_context_tokens([_create_user_message("no usage")]).last_usage_index is None
    estimate1 = estimate_context_tokens([assistant, _create_user_message("tail")])
    assert estimate1.usage_tokens == 20
    assert estimate1.last_usage_index == 0

    estimate2 = estimate_context_tokens(
        [
            _create_user_message("Hello"),
            assistant,
            _create_user_message("continue"),
            _create_assistant_message("Partial thinking", _create_mock_usage(0, 0)),
        ]
    )
    assert estimate2.usage_tokens == 20
    assert estimate2.last_usage_index == 1
    assert estimate2.trailing_tokens > 0
    assert estimate2.tokens == 20 + estimate2.trailing_tokens


def test_builds_session_context_with_a_compaction_entry():
    u1 = _create_message_entry(_create_user_message("1"))
    a1 = _create_message_entry(_create_assistant_message("a"), u1.id)
    u2 = _create_message_entry(_create_user_message("2"), a1.id)
    a2 = _create_message_entry(_create_assistant_message("b"), u2.id)
    compaction = _create_compaction_entry(
        "Summary of 1,a,2,b", a2.id, [_create_user_message("2"), _create_assistant_message("b")]
    )
    u3 = _create_message_entry(_create_user_message("3"), compaction.id)
    a3 = _create_message_entry(_create_assistant_message("c"), u3.id)
    loaded = build_session_context([u1, a1, u2, a2, compaction, u3, a3])
    assert len(loaded.messages) == 5
    assert loaded.messages[0].role == "compactionSummary"
    assert [message.role for message in loaded.messages] == [
        "compactionSummary",
        "user",
        "assistant",
        "user",
        "assistant",
    ]


def test_tracks_model_and_thinking_level_changes_in_built_context():
    user = _create_message_entry(_create_user_message("1"))
    model_change = _create_model_change_entry("openai", "gpt-4", user.id)
    assistant = _create_message_entry(_create_assistant_message("a"), model_change.id)
    thinking_change = _create_thinking_level_entry("high", assistant.id)
    loaded = build_session_context([user, model_change, assistant, thinking_change])
    assert loaded.model is not None
    assert loaded.model.provider == "anthropic"
    assert loaded.model.model_id == "claude-sonnet-4-5"
    assert loaded.thinking_level == "high"


def test_prepares_compaction_using_latest_compaction_summary_as_previous_summary():
    u1 = _create_message_entry(_create_user_message("user msg 1"))
    a1 = _create_message_entry(_create_assistant_message("assistant msg 1"), u1.id)
    u2 = _create_message_entry(_create_user_message("user msg 2"), a1.id)
    a2 = _create_message_entry(_create_assistant_message("assistant msg 2", _create_mock_usage(5000, 1000)), u2.id)
    compaction1 = _create_compaction_entry("First summary", a2.id)
    u3 = _create_message_entry(_create_user_message("user msg 3"), compaction1.id)
    a3 = _create_message_entry(_create_assistant_message("assistant msg 3", _create_mock_usage(8000, 2000)), u3.id)
    path_entries = [u1, a1, u2, a2, compaction1, u3, a3]
    preparation = get_or_throw(prepare_compaction(path_entries, DEFAULT_COMPACTION_SETTINGS))

    assert preparation is not None
    assert preparation.previous_summary == "First summary"
    assert len(preparation.retained_tail) > 0
    assert preparation.tokens_before == estimate_context_tokens(build_session_context(path_entries).messages).tokens


def test_carries_a_previous_compactions_retained_tail_into_next_preparation():
    retained_user = _create_user_message("retained user")
    retained_assistant = _create_assistant_message("retained assistant")
    compaction = _create_compaction_entry("previous summary", None, [retained_user, retained_assistant])
    user = _create_message_entry(_create_user_message("new user"), compaction.id)
    assistant = _create_message_entry(_create_assistant_message("new assistant"), user.id)

    preparation = get_or_throw(
        prepare_compaction(
            [compaction, user, assistant],
            CompactionSettings(enabled=True, reserve_tokens=100, keep_recent_tokens=1),
        )
    )

    assert preparation is not None
    assert preparation.previous_summary == "previous summary"
    assert [
        *preparation.messages_to_summarize,
        *preparation.turn_prefix_messages,
        *preparation.retained_tail,
    ] == [retained_user, retained_assistant, user.message, assistant.message]


def test_prepares_split_turn_compaction_with_prior_file_operation_details():
    u1 = _create_message_entry(_create_user_message("user msg 1"))
    assistant_message = replace(
        _create_assistant_message("assistant msg 1"),
        content=[ToolCall(id="tool-1", name="write", arguments={"path": "written.ts"})],
    )
    a1 = _create_message_entry(assistant_message, u1.id)
    compaction1 = replace(
        _create_compaction_entry("First summary", a1.id),
        details={"readFiles": ["old-read.ts"], "modifiedFiles": ["old-edit.ts", "written.ts"]},
    )
    u2 = _create_message_entry(_create_user_message("large turn"), compaction1.id)
    a2 = _create_message_entry(_create_assistant_message("large assistant message"), u2.id)
    preparation = get_or_throw(
        prepare_compaction(
            [u1, a1, compaction1, u2, a2],
            CompactionSettings(enabled=True, reserve_tokens=100, keep_recent_tokens=1),
        )
    )

    assert preparation is not None
    assert preparation.previous_summary == "First summary"
    assert preparation.is_split_turn is True
    assert [message.role for message in preparation.turn_prefix_messages] == ["user"]
    assert "old-read.ts" in preparation.file_ops.read
    assert "old-edit.ts" in preparation.file_ops.edited
    assert "written.ts" in preparation.file_ops.edited


def test_does_not_prepare_compaction_when_there_is_nothing_valid_to_compact():
    compaction = _create_compaction_entry("already compacted")
    assert get_or_throw(prepare_compaction([compaction], DEFAULT_COMPACTION_SETTINGS)) is None
    assert get_or_throw(prepare_compaction([], DEFAULT_COMPACTION_SETTINGS)) is None


def test_serializes_conversation_with_truncated_tool_results():
    long_content = "x" * 5000
    messages = [
        ToolResultMessage(
            tool_call_id="tc1", tool_name="read", content=[TextContent(text=long_content)], is_error=False
        )
    ]
    result = serialize_conversation(messages)
    assert "[Tool result]:" in result
    assert "[... 3000 more characters truncated]" in result


async def test_passes_reasoning_through_generate_summary_only_for_reasoning_models_with_thinking_enabled():
    messages = [_create_user_message("Summarize this.")]

    reasoning_model = _make_model(reasoning=True)
    stream_fn = scripted_stream_fn([_text_response_with_usage("## Goal\nTest summary", Usage())])
    get_or_throw(await generate_summary(messages, stream_fn, reasoning_model, 2000, None, None, None, "medium"))
    assert stream_fn.calls[0]["options"].reasoning == "medium"

    off_model = _make_model(reasoning=True)
    stream_fn_off = scripted_stream_fn([_text_response_with_usage("## Goal\nTest summary", Usage())])
    get_or_throw(await generate_summary(messages, stream_fn_off, off_model, 2000, None, None, None, "off"))
    assert stream_fn_off.calls[0]["options"].reasoning is None

    non_reasoning_model = _make_model(reasoning=False)
    stream_fn_non = scripted_stream_fn([_text_response_with_usage("## Goal\nTest summary", Usage())])
    get_or_throw(await generate_summary(messages, stream_fn_non, non_reasoning_model, 2000, None, None, None, "medium"))
    assert stream_fn_non.calls[0]["options"].reasoning is None


async def test_includes_previous_summaries_and_custom_instructions_in_generate_summary_prompts():
    messages = [_create_user_message("Summarize this.")]
    model = _make_model(reasoning=False)
    stream_fn = scripted_stream_fn([_text_response_with_usage("## Goal\nTest summary", _create_mock_usage(10, 5))])

    summary = get_or_throw(
        await generate_summary_with_usage(messages, stream_fn, model, 2000, None, "focus", "old summary")
    )

    text, usage = summary
    assert "Test summary" in text
    assert usage.input > 0
    assert usage.output > 0
    assert usage.total_tokens == usage.input + usage.output + usage.cache_read + usage.cache_write

    prompt_text = stream_fn.calls[0]["context"].messages[0].content[0].text
    assert "<previous-summary>\nold summary\n</previous-summary>" in prompt_text
    assert "Additional focus: focus" in prompt_text


async def test_preserves_the_string_result_from_generate_summary():
    messages = [_create_user_message("Summarize this.")]
    model = _make_model(reasoning=False)
    stream_fn = scripted_stream_fn([_text_response_with_usage("## Goal\nTest summary", Usage())])

    result = get_or_throw(await generate_summary(messages, stream_fn, model, 2000))
    assert result == "## Goal\nTest summary"


async def test_returns_error_results_for_failed_or_aborted_summary_generations():
    messages = [_create_user_message("Summarize this.")]
    error_model = _make_model(reasoning=False)
    error_stream_fn = scripted_stream_fn(
        [
            AssistantMessage(
                api=TEST_MODEL.api,
                provider=TEST_MODEL.provider,
                model=TEST_MODEL.id,
                content=[],
                usage=Usage(),
                stop_reason="error",
                error_message="boom",
            )
        ]
    )
    error_result = await generate_summary(messages, error_stream_fn, error_model, 2000)
    assert error_result.ok is False
    assert error_result.error.code == "summarization_failed"
    assert error_result.error.args[0] == "Summarization failed: boom"

    aborted_model = _make_model(reasoning=False)
    aborted_stream_fn = scripted_stream_fn(
        [
            AssistantMessage(
                api=TEST_MODEL.api,
                provider=TEST_MODEL.provider,
                model=TEST_MODEL.id,
                content=[],
                usage=Usage(),
                stop_reason="aborted",
                error_message="stopped",
            )
        ]
    )
    aborted_result = await generate_summary(messages, aborted_stream_fn, aborted_model, 2000)
    assert aborted_result.ok is False
    assert aborted_result.error.code == "aborted"
    assert aborted_result.error.args[0] == "stopped"


async def test_clamps_compaction_summary_max_tokens_to_the_model_output_cap():
    messages = [_create_user_message("Summarize this.")]
    model = _make_model(reasoning=False, max_tokens=128000)
    stream_fn = scripted_stream_fn(
        [
            _text_response_with_usage("## Goal\nTest summary", Usage()),
            _text_response_with_usage("## Goal\nTest summary", Usage()),
        ]
    )
    preparation = CompactionPreparation(
        messages_to_summarize=messages,
        turn_prefix_messages=messages,
        retained_tail=messages,
        is_split_turn=True,
        tokens_before=600000,
        file_ops=FileOperations(),
        settings=CompactionSettings(enabled=True, reserve_tokens=500000, keep_recent_tokens=20000),
    )

    get_or_throw(await compact(preparation, stream_fn, model))

    seen_options = [call["options"] for call in stream_fn.calls]
    assert [options.max_tokens for options in seen_options] == [128000, 128000]
    assert [options.cache_retention for options in seen_options] == ["none", "none"]
    assert seen_options[0].session_id != seen_options[1].session_id


async def test_returns_compaction_error_results_without_throwing():
    messages = [_create_user_message("Summarize this.")]
    preparation = CompactionPreparation(
        messages_to_summarize=messages,
        turn_prefix_messages=[],
        retained_tail=messages,
        is_split_turn=False,
        tokens_before=100,
        file_ops=FileOperations(),
        settings=CompactionSettings(enabled=True, reserve_tokens=2000, keep_recent_tokens=20),
    )
    history_model = _make_model(reasoning=False)
    history_stream_fn = scripted_stream_fn(
        [
            AssistantMessage(
                api=TEST_MODEL.api,
                provider=TEST_MODEL.provider,
                model=TEST_MODEL.id,
                content=[],
                usage=Usage(),
                stop_reason="error",
                error_message="history failed",
            )
        ]
    )
    result = await compact(preparation, history_stream_fn, history_model)
    assert result.ok is False
    assert result.error.code == "summarization_failed"
    assert result.error.args[0] == "Summarization failed: history failed"


async def test_combines_usage_for_split_turn_compaction_summaries():
    messages = [_create_user_message("Summarize this.")]
    model = _make_model(reasoning=False)
    history_usage = _create_mock_usage(1, 2, 3, 4)
    turn_prefix_usage = _create_mock_usage(5, 6, 7, 8)
    usage_stream_fn = scripted_stream_fn(
        [
            _text_response_with_usage("history summary", history_usage),
            _text_response_with_usage("turn prefix summary", turn_prefix_usage),
        ]
    )
    preparation = CompactionPreparation(
        messages_to_summarize=messages,
        turn_prefix_messages=messages,
        is_split_turn=True,
        tokens_before=100,
        retained_tail=messages,
        file_ops=FileOperations(),
        settings=CompactionSettings(enabled=True, reserve_tokens=2000, keep_recent_tokens=20),
    )

    result = get_or_throw(await compact(preparation, usage_stream_fn, model))

    assert result.usage == _create_mock_usage(6, 8, 10, 12)


async def test_passes_reasoning_through_turn_prefix_summaries_when_enabled():
    messages = [_create_user_message("Summarize this.")]
    model = _make_model(reasoning=True)
    stream_fn = scripted_stream_fn([_text_response_with_usage("## Original Request\nTest summary", Usage())])
    preparation = CompactionPreparation(
        messages_to_summarize=[],
        turn_prefix_messages=messages,
        retained_tail=messages,
        is_split_turn=True,
        tokens_before=100,
        file_ops=FileOperations(),
        settings=CompactionSettings(enabled=True, reserve_tokens=2000, keep_recent_tokens=20),
    )

    get_or_throw(await compact(preparation, stream_fn, model, None, None, "high"))

    assert stream_fn.calls[0]["options"].reasoning == "high"


async def test_returns_turn_prefix_compaction_errors_without_throwing():
    messages = [_create_user_message("Summarize this.")]
    preparation = CompactionPreparation(
        messages_to_summarize=[],
        turn_prefix_messages=messages,
        retained_tail=messages,
        is_split_turn=True,
        tokens_before=100,
        file_ops=FileOperations(),
        settings=CompactionSettings(enabled=True, reserve_tokens=2000, keep_recent_tokens=20),
    )
    model = _make_model(reasoning=False)
    stream_fn = scripted_stream_fn(
        [
            AssistantMessage(
                api=TEST_MODEL.api,
                provider=TEST_MODEL.provider,
                model=TEST_MODEL.id,
                content=[],
                usage=Usage(),
                stop_reason="error",
                error_message="prefix failed",
            )
        ]
    )
    result = await compact(preparation, stream_fn, model)
    assert result.ok is False
    assert result.error.code == "summarization_failed"
    assert result.error.args[0] == "Turn prefix summarization failed: prefix failed"

    aborted_model = _make_model(reasoning=False)
    aborted_stream_fn = scripted_stream_fn(
        [
            AssistantMessage(
                api=TEST_MODEL.api,
                provider=TEST_MODEL.provider,
                model=TEST_MODEL.id,
                content=[],
                usage=Usage(),
                stop_reason="aborted",
                error_message="prefix stopped",
            )
        ]
    )
    aborted_result = await compact(preparation, aborted_stream_fn, aborted_model)
    assert aborted_result.ok is False
    assert aborted_result.error.code == "aborted"
    assert aborted_result.error.args[0] == "prefix stopped"


async def test_returns_a_compaction_result_with_file_details():
    u1 = _create_message_entry(_create_user_message("read a file"))
    assistant_message = replace(
        _create_assistant_message("calling tool", _create_mock_usage(1000, 200)),
        content=[ToolCall(id="tool-1", name="read", arguments={"path": "src/index.ts"})],
    )
    a1 = _create_message_entry(assistant_message, u1.id)
    u2 = _create_message_entry(_create_user_message("continue"), a1.id)
    a2 = _create_message_entry(_create_assistant_message("done", _create_mock_usage(4000, 500)), u2.id)
    preparation = get_or_throw(prepare_compaction([u1, a1, u2, a2], DEFAULT_COMPACTION_SETTINGS))
    assert preparation is not None

    model = _make_model(reasoning=False)
    stream_fn = scripted_stream_fn([_text_response_with_usage("## Goal\nTest summary", _create_mock_usage(10, 5))])
    result = get_or_throw(await compact(preparation, stream_fn, model))

    assert len(result.summary) > 0
    assert result.usage is not None
    assert result.usage.total_tokens > 0
    assert len(result.retained_tail) > 0
    assert result.details is not None
