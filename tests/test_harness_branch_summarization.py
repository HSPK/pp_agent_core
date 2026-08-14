"""Tests for `pi_agent.harness.compaction.branch_summarization`.

Ported from `packages/agent/test/harness/branch-summarization.test.ts`, which
only covers `collect_entries_for_branch_summary`. `prepare_branch_entries` and
`generate_branch_summary` have no dedicated TypeScript suite, so the tests for
them below assert the behaviour written in
`packages/agent/src/harness/compaction/branch-summarization.ts` directly. Like
`test_harness_compaction.py` they drive summarization through
`agent_helpers.scripted_stream_fn` plus a `pi_ai.types.Model`, because this port's
`generate_branch_summary` takes a `StreamFn` instead of a `Models` registry
(see `branch_summarization.py`'s module docstring).
"""

from __future__ import annotations

import asyncio

import pytest
from agent_helpers import TEST_MODEL, scripted_stream_fn
from pi_agent.harness.compaction.branch_summarization import (
    GenerateBranchSummaryOptions,
    collect_entries_for_branch_summary,
    generate_branch_summary,
    prepare_branch_entries,
)
from pi_agent.harness.session import InMemorySessionStorage, Session
from pi_agent.harness.session.types import (
    ActiveToolsEntry,
    BranchSummaryEntry,
    CompactionEntry,
    CustomEntry,
    Entry,
    MessageEntry,
    ModelChangeEntry,
    SessionError,
    SessionMetadata,
    ThinkingLevelEntry,
)
from pi_ai.types import (
    AssistantMessage,
    Model,
    TextContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)

TIMEOUT = 5.0


def _message(text: str) -> UserMessage:
    return UserMessage(content=[TextContent(text=text)], timestamp=1)


class _SequentialIdGenerator:
    def __init__(self) -> None:
        self._next = 0

    def next(self) -> str:
        self._next += 1
        return f"entry-{self._next}"


class _StubSession:
    """Minimal `Session` stand-in exposing only what `collect_entries_for_branch_summary` calls."""

    def __init__(self, entries: dict[str, Entry], branches: dict[str, list[Entry]]) -> None:
        self._entries = entries
        self._branches = branches

    async def find_entries_on_branch(self, query=None, bounds=None) -> list[Entry]:
        return self._branches[bounds.start]

    async def get_entry(self, id: str) -> Entry | None:
        return self._entries.get(id)


_next_id = 0


def _create_id() -> str:
    global _next_id
    _next_id += 1
    return f"entry-{_next_id}"


@pytest.fixture(autouse=True)
def _reset_id_counter():
    global _next_id
    _next_id = 0
    yield


def _message_entry(message, parent_id: str | None = None) -> MessageEntry:
    return MessageEntry(id=_create_id(), parent_id=parent_id, seq=_next_id, timestamp=1, message=message)


def _assistant_with_tool_calls(*tool_calls: ToolCall) -> AssistantMessage:
    return AssistantMessage(
        api=TEST_MODEL.api,
        provider=TEST_MODEL.provider,
        model=TEST_MODEL.id,
        content=[TextContent(text="working"), *tool_calls],
        usage=Usage(),
        stop_reason="toolUse",
        timestamp=1,
    )


def _text_response(text: str, usage: Usage | None = None) -> AssistantMessage:
    return AssistantMessage(
        api=TEST_MODEL.api,
        provider=TEST_MODEL.provider,
        model=TEST_MODEL.id,
        content=[TextContent(text=text)],
        usage=usage if usage is not None else Usage(),
        stop_reason="stop",
        timestamp=1,
    )


def _make_model(context_window: int = 200000) -> Model:
    return Model(
        id="summarizer",
        api="test-api",
        provider="test",
        context_window=context_window,
        max_tokens=8192,
    )


# --------------------------------------------------------------------------
# collect_entries_for_branch_summary
# --------------------------------------------------------------------------


async def test_collects_abandoned_side_of_branch_in_chronological_order():
    session = Session(
        InMemorySessionStorage(SessionMetadata(id="session", created_at=1)),
        id_generator=_SequentialIdGenerator(),
    )
    root_id = await session.append_message(_message("root"))
    common_id = await session.append_message(_message("common"))
    abandoned_ids = [
        await session.append_message(_message("abandoned 1")),
        await session.append_message(_message("abandoned 2")),
    ]
    await session.create_lane("target", common_id)
    target_id = await session.view("target").append_message(_message("target"))

    result = await collect_entries_for_branch_summary(session, abandoned_ids[-1], target_id)

    assert result.common_ancestor_id == common_id
    assert [entry.id for entry in result.entries] == abandoned_ids
    assert not any(entry.id == root_id for entry in result.entries)


async def test_returns_no_entries_when_there_was_no_previous_leaf():
    session = Session(InMemorySessionStorage(SessionMetadata(id="session", created_at=1)))
    target_id = await session.append_message(_message("target"))

    result = await collect_entries_for_branch_summary(session, None, target_id)

    assert result.entries == []
    assert result.common_ancestor_id is None


async def test_collects_the_whole_branch_when_paths_share_no_ancestor():
    # Two unrelated roots: `commonAncestorId` stays null and the walk runs all
    # the way up to the abandoned branch's own root.
    session = Session(
        InMemorySessionStorage(SessionMetadata(id="session", created_at=1)),
        id_generator=_SequentialIdGenerator(),
    )
    first_id = await session.append_message(_message("abandoned root"))
    second_id = await session.append_message(_message("abandoned leaf"))
    await session.create_lane("other", None)
    target_id = await session.view("other").append_message(_message("target root"))

    result = await asyncio.wait_for(collect_entries_for_branch_summary(session, second_id, target_id), timeout=TIMEOUT)

    assert result.common_ancestor_id is None
    assert [entry.id for entry in result.entries] == [first_id, second_id]


async def test_collects_nothing_when_the_previous_leaf_is_the_navigation_target():
    session = Session(
        InMemorySessionStorage(SessionMetadata(id="session", created_at=1)),
        id_generator=_SequentialIdGenerator(),
    )
    await session.append_message(_message("root"))
    leaf_id = await session.append_message(_message("leaf"))

    result = await asyncio.wait_for(collect_entries_for_branch_summary(session, leaf_id, leaf_id), timeout=TIMEOUT)

    assert result.common_ancestor_id == leaf_id
    assert result.entries == []


async def test_raises_when_an_entry_on_the_abandoned_path_is_missing():
    # TS: `if (!entry) throw new SessionError("invalid_entry", ...)`. The
    # in-memory store never loses an entry, so drive the walk through a stub
    # whose `getEntry` returns nothing for the dangling parent.
    leaf = MessageEntry(id="leaf", parent_id="gone", seq=2, timestamp=1, message=_message("leaf"))
    session = _StubSession({"leaf": leaf}, {"leaf": [leaf], "target": []})

    with pytest.raises(SessionError) as excinfo:
        await asyncio.wait_for(collect_entries_for_branch_summary(session, "leaf", "target"), timeout=TIMEOUT)

    assert excinfo.value.code == "invalid_entry"
    assert "Entry gone not found" in str(excinfo.value)


# --------------------------------------------------------------------------
# prepare_branch_entries
# --------------------------------------------------------------------------


def test_prepare_branch_entries_returns_nothing_for_an_empty_branch():
    preparation = prepare_branch_entries([])

    assert preparation.messages == []
    assert preparation.total_tokens == 0
    assert preparation.file_ops.read == set()
    assert preparation.file_ops.written == set()
    assert preparation.file_ops.edited == set()


def test_prepare_branch_entries_keeps_a_single_entry_branch():
    entry = _message_entry(_message("only message"))

    preparation = prepare_branch_entries([entry])

    assert preparation.messages == [entry.message]
    assert preparation.total_tokens > 0


def test_prepare_branch_entries_skips_tool_results_and_non_message_entries():
    # TS `getMessageFromEntry` returns undefined for toolResult messages and
    # for thinking_level_change / model_change / active_tools_change / custom.
    user = _message_entry(_message("keep me"))
    tool_result = _message_entry(
        ToolResultMessage(
            tool_call_id="c1",
            tool_name="read",
            content=[TextContent(text="drop me")],
            is_error=False,
            timestamp=1,
        )
    )
    entries: list[Entry] = [
        user,
        tool_result,
        ThinkingLevelEntry(id=_create_id(), seq=_next_id, timestamp=1, thinking_level="high"),
        ModelChangeEntry(id=_create_id(), seq=_next_id, timestamp=1, provider="openai", model_id="gpt-5"),
        ActiveToolsEntry(id=_create_id(), seq=_next_id, timestamp=1, active_tool_names=["read"]),
        CustomEntry(id=_create_id(), seq=_next_id, timestamp=1, custom_type="note", data={"x": 1}),
    ]

    preparation = prepare_branch_entries(entries)

    assert preparation.messages == [user.message]


def test_prepare_branch_entries_converts_summary_entries_to_summary_messages():
    compaction = CompactionEntry(
        id=_create_id(), seq=_next_id, timestamp=11, summary="compacted", tokens_before=99, retained_tail=[]
    )
    branch_summary = BranchSummaryEntry(id=_create_id(), seq=_next_id, timestamp=12, from_id="e-x", summary="branched")

    preparation = prepare_branch_entries([compaction, branch_summary])

    assert [m.role for m in preparation.messages] == ["compactionSummary", "branchSummary"]
    assert preparation.messages[0].summary == "compacted"
    assert preparation.messages[0].tokens_before == 99
    assert preparation.messages[1].summary == "branched"
    assert preparation.messages[1].from_id == "e-x"
    assert preparation.messages[1].timestamp == 12


def test_prepare_branch_entries_seeds_file_ops_from_branch_summary_details():
    branch_summary = BranchSummaryEntry(
        id=_create_id(),
        seq=_next_id,
        timestamp=1,
        from_id="e-x",
        summary="earlier branch",
        details={"readFiles": ["seeded-read.py"], "modifiedFiles": ["seeded-edit.py"]},
    )

    preparation = prepare_branch_entries([branch_summary])

    assert preparation.file_ops.read == {"seeded-read.py"}
    assert preparation.file_ops.edited == {"seeded-edit.py"}


def test_prepare_branch_entries_ignores_branch_summary_details_that_are_not_objects():
    branch_summary = BranchSummaryEntry(
        id=_create_id(), seq=_next_id, timestamp=1, from_id="e-x", summary="earlier", details="not an object"
    )

    preparation = prepare_branch_entries([branch_summary])

    assert preparation.file_ops.read == set()
    assert preparation.file_ops.edited == set()


def test_prepare_branch_entries_extracts_file_ops_from_assistant_tool_calls():
    entry = _message_entry(
        _assistant_with_tool_calls(
            ToolCall(id="c1", name="read", arguments={"path": "read.py"}),
            ToolCall(id="c2", name="write", arguments={"path": "written.py"}),
            ToolCall(id="c3", name="edit", arguments={"path": "edited.py"}),
        )
    )

    preparation = prepare_branch_entries([entry])

    assert preparation.file_ops.read == {"read.py"}
    assert preparation.file_ops.written == {"written.py"}
    assert preparation.file_ops.edited == {"edited.py"}


def test_prepare_branch_entries_drops_oldest_messages_over_the_token_budget():
    # Newest-first walk: 40 characters is 10 estimated tokens per message, so a
    # 25-token budget fits the two newest and stops before the oldest.
    entries = [_message_entry(_message("a" * 40)) for _ in range(3)]

    preparation = prepare_branch_entries(entries, 25)

    assert preparation.messages == [entries[1].message, entries[2].message]
    assert preparation.total_tokens == 20


def test_prepare_branch_entries_keeps_an_over_budget_summary_entry_under_the_90_percent_mark():
    compaction = CompactionEntry(
        id=_create_id(), seq=_next_id, timestamp=1, summary="s" * 40, tokens_before=1, retained_tail=[]
    )
    tail = _message_entry(_message("a" * 40))

    preparation = prepare_branch_entries([compaction, tail], 15)

    assert [m.role for m in preparation.messages] == ["compactionSummary", "user"]
    assert preparation.total_tokens == 20


def test_prepare_branch_entries_drops_an_over_budget_summary_entry_past_the_90_percent_mark():
    compaction = CompactionEntry(
        id=_create_id(), seq=_next_id, timestamp=1, summary="s" * 40, tokens_before=1, retained_tail=[]
    )
    tail = _message_entry(_message("a" * 40))

    preparation = prepare_branch_entries([compaction, tail], 10)

    assert [m.role for m in preparation.messages] == ["user"]
    assert preparation.total_tokens == 10


def test_prepare_branch_entries_ignores_the_budget_when_it_is_zero():
    entries = [_message_entry(_message("a" * 400)) for _ in range(3)]

    preparation = prepare_branch_entries(entries)

    assert len(preparation.messages) == 3


# --------------------------------------------------------------------------
# generate_branch_summary
# --------------------------------------------------------------------------


async def test_generate_branch_summary_returns_a_placeholder_for_an_empty_branch():
    stream_fn = scripted_stream_fn([])

    result = await asyncio.wait_for(
        generate_branch_summary([], GenerateBranchSummaryOptions(stream_fn=stream_fn, model=_make_model())),
        timeout=TIMEOUT,
    )

    assert result.ok is True
    assert result.value.summary == "No content to summarize"
    assert result.value.read_files == []
    assert result.value.modified_files == []
    assert stream_fn.calls == []


async def test_generate_branch_summary_prefixes_the_preamble_and_appends_file_operations():
    entries = [
        _message_entry(_message("please explore")),
        _message_entry(
            _assistant_with_tool_calls(
                ToolCall(id="c1", name="read", arguments={"path": "read-only.py"}),
                ToolCall(id="c2", name="edit", arguments={"path": "changed.py"}),
            )
        ),
    ]
    usage = Usage(input=11, output=7, total_tokens=18)
    stream_fn = scripted_stream_fn([_text_response("## Goal\nExplore the parser", usage)])

    result = await asyncio.wait_for(
        generate_branch_summary(entries, GenerateBranchSummaryOptions(stream_fn=stream_fn, model=_make_model())),
        timeout=TIMEOUT,
    )

    assert result.ok is True
    summary = result.value.summary
    assert summary.startswith("The user explored a different conversation branch before returning here.")
    assert "## Goal\nExplore the parser" in summary
    assert summary.endswith(
        "<read-files>\nread-only.py\n</read-files>\n\n<modified-files>\nchanged.py\n</modified-files>"
    )
    assert result.value.read_files == ["read-only.py"]
    assert result.value.modified_files == ["changed.py"]
    assert result.value.usage is usage


async def test_generate_branch_summary_sends_the_default_prompt_and_conversation():
    entries = [_message_entry(_message("please explore"))]
    stream_fn = scripted_stream_fn([_text_response("summary")])

    await asyncio.wait_for(
        generate_branch_summary(entries, GenerateBranchSummaryOptions(stream_fn=stream_fn, model=_make_model())),
        timeout=TIMEOUT,
    )

    call = stream_fn.calls[0]
    prompt = call["context"].messages[0].content[0].text
    assert prompt.startswith("<conversation>\n[User]: please explore\n</conversation>\n\n")
    assert "Create a structured summary of this conversation branch" in prompt
    assert "Additional focus" not in prompt
    assert call["context"].system_prompt.startswith("You are a context summarization assistant")
    assert call["options"].max_tokens == 2048


async def test_generate_branch_summary_appends_custom_instructions_by_default():
    entries = [_message_entry(_message("please explore"))]
    stream_fn = scripted_stream_fn([_text_response("summary")])

    await asyncio.wait_for(
        generate_branch_summary(
            entries,
            GenerateBranchSummaryOptions(
                stream_fn=stream_fn, model=_make_model(), custom_instructions="mention the parser"
            ),
        ),
        timeout=TIMEOUT,
    )

    prompt = stream_fn.calls[0]["context"].messages[0].content[0].text
    assert "Create a structured summary of this conversation branch" in prompt
    assert prompt.endswith("Additional focus: mention the parser")


async def test_generate_branch_summary_replaces_the_prompt_when_asked():
    entries = [_message_entry(_message("please explore"))]
    stream_fn = scripted_stream_fn([_text_response("summary")])

    await asyncio.wait_for(
        generate_branch_summary(
            entries,
            GenerateBranchSummaryOptions(
                stream_fn=stream_fn,
                model=_make_model(),
                custom_instructions="just the file list",
                replace_instructions=True,
            ),
        ),
        timeout=TIMEOUT,
    )

    prompt = stream_fn.calls[0]["context"].messages[0].content[0].text
    assert prompt.endswith("</conversation>\n\njust the file list")
    assert "Create a structured summary of this conversation branch" not in prompt


async def test_generate_branch_summary_ignores_replace_instructions_without_custom_instructions():
    entries = [_message_entry(_message("please explore"))]
    stream_fn = scripted_stream_fn([_text_response("summary")])

    await asyncio.wait_for(
        generate_branch_summary(
            entries,
            GenerateBranchSummaryOptions(stream_fn=stream_fn, model=_make_model(), replace_instructions=True),
        ),
        timeout=TIMEOUT,
    )

    prompt = stream_fn.calls[0]["context"].messages[0].content[0].text
    assert "Create a structured summary of this conversation branch" in prompt


async def test_generate_branch_summary_applies_the_reserve_token_budget():
    # A 1000-token context window minus a 990-token reserve leaves 10 tokens,
    # so only the newest 40-character (10-token) message survives.
    entries = [_message_entry(_message("a" * 40)) for _ in range(3)]
    stream_fn = scripted_stream_fn([_text_response("summary")])

    await asyncio.wait_for(
        generate_branch_summary(
            entries,
            GenerateBranchSummaryOptions(
                stream_fn=stream_fn, model=_make_model(context_window=1000), reserve_tokens=990
            ),
        ),
        timeout=TIMEOUT,
    )

    prompt = stream_fn.calls[0]["context"].messages[0].content[0].text
    assert prompt.count("[User]: ") == 1


async def test_generate_branch_summary_reports_summarization_failures():
    entries = [_message_entry(_message("please explore"))]
    stream_fn = scripted_stream_fn(
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

    result = await asyncio.wait_for(
        generate_branch_summary(entries, GenerateBranchSummaryOptions(stream_fn=stream_fn, model=_make_model())),
        timeout=TIMEOUT,
    )

    assert result.ok is False
    assert result.error.code == "summarization_failed"
    assert result.error.args[0] == "Branch summary failed: boom"


async def test_generate_branch_summary_falls_back_to_unknown_error_text():
    entries = [_message_entry(_message("please explore"))]
    stream_fn = scripted_stream_fn(
        [
            AssistantMessage(
                api=TEST_MODEL.api,
                provider=TEST_MODEL.provider,
                model=TEST_MODEL.id,
                content=[],
                usage=Usage(),
                stop_reason="error",
            )
        ]
    )

    result = await asyncio.wait_for(
        generate_branch_summary(entries, GenerateBranchSummaryOptions(stream_fn=stream_fn, model=_make_model())),
        timeout=TIMEOUT,
    )

    assert result.ok is False
    assert result.error.args[0] == "Branch summary failed: Unknown error"


async def test_generate_branch_summary_reports_aborts():
    entries = [_message_entry(_message("please explore"))]
    stream_fn = scripted_stream_fn(
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

    result = await asyncio.wait_for(
        generate_branch_summary(entries, GenerateBranchSummaryOptions(stream_fn=stream_fn, model=_make_model())),
        timeout=TIMEOUT,
    )

    assert result.ok is False
    assert result.error.code == "aborted"
    assert result.error.args[0] == "stopped"


async def test_generate_branch_summary_uses_a_default_abort_message():
    entries = [_message_entry(_message("please explore"))]
    stream_fn = scripted_stream_fn(
        [
            AssistantMessage(
                api=TEST_MODEL.api,
                provider=TEST_MODEL.provider,
                model=TEST_MODEL.id,
                content=[],
                usage=Usage(),
                stop_reason="aborted",
            )
        ]
    )

    result = await asyncio.wait_for(
        generate_branch_summary(entries, GenerateBranchSummaryOptions(stream_fn=stream_fn, model=_make_model())),
        timeout=TIMEOUT,
    )

    assert result.ok is False
    assert result.error.args[0] == "Branch summary aborted"


async def test_generate_branch_summary_defaults_the_context_window_when_the_model_has_none():
    # TS: `model.contextWindow || 128000`, so a missing window still leaves a
    # workable budget rather than a negative one.
    entries = [_message_entry(_message("a" * 40)) for _ in range(3)]
    model = Model(id="no-window", api="test-api", provider="test", max_tokens=8192)
    stream_fn = scripted_stream_fn([_text_response("summary")])

    await asyncio.wait_for(
        generate_branch_summary(entries, GenerateBranchSummaryOptions(stream_fn=stream_fn, model=model)),
        timeout=TIMEOUT,
    )

    prompt = stream_fn.calls[0]["context"].messages[0].content[0].text
    assert prompt.count("[User]: ") == 3
