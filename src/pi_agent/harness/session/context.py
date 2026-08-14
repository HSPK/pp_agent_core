"""Session-entries-to-context-messages projection.

Python port of `packages/agent/src/harness/session/context.ts`. Turns the
path of entries from the current leaf back to the session root into the
`AgentMessage` list an agent loop sends to a model, folding in the derived
"current" thinking level / model / active tools.

Scoped dependency: `context.ts` imports `createBranchSummaryMessage` and
`createCompactionSummaryMessage` (plus their message shapes) from
`../messages.ts`. Porting all of `messages.ts` (bash-execution-to-text
conversion, `convertToLlm`, TypeScript declaration merging for
`CustomAgentMessages`, ...) is out of scope for the session storage layer, so
only those two factories and their two message dataclasses are ported here,
directly. `pi_agent.types.AgentMessage` remains `pi_ai.types.Message` only
(user/assistant/toolResult); `BranchSummaryMessage`/`CompactionSummaryMessage`
are additional message shapes this module produces but that are not part of
that union, mirroring how TypeScript widens `AgentMessage` via declaration
merging in `messages.ts` in a way Python has no equivalent for.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from ...types import AgentMessage
from .types import CompactionEntry, CustomEntry, Entry

COMPACTION_SUMMARY_PREFIX = """The conversation history before this point was compacted into the following summary:

<summary>
"""

COMPACTION_SUMMARY_SUFFIX = """
</summary>"""

BRANCH_SUMMARY_PREFIX = """The following is a summary of a branch that this conversation came back from:

<summary>
"""

BRANCH_SUMMARY_SUFFIX = "</summary>"


@dataclass(kw_only=True)
class BranchSummaryMessage:
    summary: str
    from_id: str
    timestamp: int
    role: Literal["branchSummary"] = "branchSummary"


@dataclass(kw_only=True)
class CompactionSummaryMessage:
    summary: str
    tokens_before: int
    timestamp: int
    role: Literal["compactionSummary"] = "compactionSummary"


ContextMessage = AgentMessage | BranchSummaryMessage | CompactionSummaryMessage


def create_branch_summary_message(summary: str, from_id: str, timestamp: int) -> BranchSummaryMessage:
    return BranchSummaryMessage(summary=summary, from_id=from_id, timestamp=timestamp)


def create_compaction_summary_message(summary: str, tokens_before: int, timestamp: int) -> CompactionSummaryMessage:
    return CompactionSummaryMessage(summary=summary, tokens_before=tokens_before, timestamp=timestamp)


@dataclass(kw_only=True)
class SessionContextModel:
    provider: str
    model_id: str


@dataclass(kw_only=True)
class SessionContext:
    messages: list[ContextMessage] = field(default_factory=list)
    thinking_level: str = "off"
    model: SessionContextModel | None = None
    active_tool_names: list[str] | None = None


ContextEntryTransform = Callable[[Sequence[Entry]], Sequence[Entry]]
CustomEntryContextMessageProjector = Callable[[CustomEntry, int, Sequence[Entry]], Sequence[ContextMessage] | None]


@dataclass(kw_only=True)
class SessionContextBuildOptions:
    entry_transforms: Sequence[ContextEntryTransform] = field(default_factory=tuple)
    entry_projectors: Mapping[str, CustomEntryContextMessageProjector] = field(default_factory=dict)


def _derive_session_context_state(
    path_entries: Sequence[Entry],
) -> tuple[str, SessionContextModel | None, list[str] | None]:
    thinking_level = "off"
    model: SessionContextModel | None = None
    active_tool_names: list[str] | None = None

    for entry in path_entries:
        if entry.type == "thinking_level_change":
            thinking_level = entry.thinking_level
        elif entry.type == "model_change":
            model = SessionContextModel(provider=entry.provider, model_id=entry.model_id)
        elif entry.type == "message" and entry.message.role == "assistant":
            model = SessionContextModel(provider=entry.message.provider, model_id=entry.message.model)
        elif entry.type == "active_tools_change":
            active_tool_names = list(entry.active_tool_names)

    return thinking_level, model, active_tool_names


def default_context_entry_transform(path_entries: Sequence[Entry]) -> list[Entry]:
    compaction: CompactionEntry | None = None
    compaction_index = -1
    for index in range(len(path_entries) - 1, -1, -1):
        entry = path_entries[index]
        if isinstance(entry, CompactionEntry):
            compaction = entry
            compaction_index = index
            break
    if compaction is None:
        return list(path_entries)
    return [compaction, *path_entries[compaction_index + 1 :]]


def build_context_entries(
    path_entries: Sequence[Entry], options: SessionContextBuildOptions | None = None
) -> list[Entry]:
    options = options or SessionContextBuildOptions()
    entries: Sequence[Entry] = default_context_entry_transform(path_entries)
    for transform in options.entry_transforms:
        entries = list(transform(entries))
    return list(entries)


def session_entry_to_context_messages(
    entry: Entry,
    index: int,
    entries: Sequence[Entry],
    options: SessionContextBuildOptions | None = None,
) -> list[ContextMessage]:
    options = options or SessionContextBuildOptions()
    if entry.type == "message":
        if entry.message.role == "assistant" and entry.message.stop_reason == "deferred":
            return []
        return [entry.message]
    if entry.type == "compaction":
        return [
            create_compaction_summary_message(entry.summary, entry.tokens_before, entry.timestamp),
            *entry.retained_tail,
        ]
    if entry.type == "branch_summary" and entry.summary:
        return [create_branch_summary_message(entry.summary, entry.from_id, entry.timestamp)]
    if entry.type == "custom":
        projector = options.entry_projectors.get(entry.custom_type)
        projected = projector(entry, index, entries) if projector is not None else None
        return list(projected) if projected is not None else []
    return []


def build_session_context(
    path_entries: Sequence[Entry], options: SessionContextBuildOptions | None = None
) -> SessionContext:
    options = options or SessionContextBuildOptions()
    thinking_level, model, active_tool_names = _derive_session_context_state(path_entries)
    context_entries = build_context_entries(path_entries, options)
    messages: list[ContextMessage] = []
    for index, entry in enumerate(context_entries):
        messages.extend(session_entry_to_context_messages(entry, index, context_entries, options))
    return SessionContext(
        messages=messages, thinking_level=thinking_level, model=model, active_tool_names=active_tool_names
    )


__all__ = [
    "BRANCH_SUMMARY_PREFIX",
    "BRANCH_SUMMARY_SUFFIX",
    "COMPACTION_SUMMARY_PREFIX",
    "COMPACTION_SUMMARY_SUFFIX",
    "BranchSummaryMessage",
    "CompactionSummaryMessage",
    "ContextEntryTransform",
    "ContextMessage",
    "CustomEntryContextMessageProjector",
    "SessionContext",
    "SessionContextBuildOptions",
    "SessionContextModel",
    "build_context_entries",
    "build_session_context",
    "create_branch_summary_message",
    "create_compaction_summary_message",
    "default_context_entry_transform",
    "session_entry_to_context_messages",
]
