"""Compaction: summarizing old conversation history to keep context size bounded.

Python port of `packages/agent/src/harness/compaction/compaction.ts`.

TypeScript threads a `Models` registry plus `Model<Api>` through
`completeSimpleWithRetries` so it can call `models.completeSimple(...)`.
`pi_ai` has no ported `Models`/`completeSimple` registry yet (see
`pi_agent.agent_loop`, which already made the same call for the main agent
loop). This port instead takes a `StreamFn` (as `pi_agent.types.StreamFn`)
plus a `pi_ai.types.Model`, and drives it the same way
`agent_loop._stream_assistant_response` does: call the stream function, then
await `stream.result()` for the final `AssistantMessage`. This keeps
compaction trivially testable with `conftest.scripted_stream_fn` instead of
needing a ported provider registry.

Token estimation reuses `pi_ai.utils.estimate.estimate_message_tokens`
(assistant messages) and `estimate_text_and_image_content_tokens`
(user/toolResult/custom messages, which all use the same text+image
character heuristic) for the roles `pi_ai` already understands, and extends
that inline only for the additional harness-only roles (`bashExecution`,
`branchSummary`, `compactionSummary`) that `estimate_tokens` here must also
support, mirroring `compaction.ts`'s `estimateTokens`.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Awaitable
from dataclasses import dataclass, replace
from typing import Any

from pi_ai.types import AssistantMessage, Context, Model, SimpleStreamOptions, Usage
from pi_ai.utils.abort import AbortSignal
from pi_ai.utils.estimate import ESTIMATED_IMAGE_CHARS, estimate_message_tokens, estimate_text_and_image_content_tokens
from pi_ai.utils.retry import RetryCallbacks, RetryPolicy, retry_assistant_call
from pi_ai.utils.text import content_text
from pi_ai.utils.uuid import uuidv7

from ...types import ThinkingLevel
from ..messages import HarnessMessage, convert_to_llm
from ..session.context import build_session_context
from ..session.types import CompactionEntry, Entry
from ..types import CompactionError, Result, err, ok
from .utils import (
    FileOperations,
    compute_file_lists,
    create_file_ops,
    extract_file_ops_from_message,
    format_file_operations,
    serialize_conversation,
)


@dataclass(kw_only=True)
class CompactionDetails:
    """File-operation details stored on generated compaction entries."""

    read_files: list[str]
    """Files read in the compacted history."""
    modified_files: list[str]
    """Files modified in the compacted history."""


def _extract_file_operations(
    messages: list[HarnessMessage], entries: list[Entry], prev_compaction_index: int
) -> FileOperations:
    file_ops = create_file_ops()
    if prev_compaction_index >= 0:
        prev_compaction = entries[prev_compaction_index]
        if isinstance(prev_compaction, CompactionEntry) and isinstance(prev_compaction.details, dict):
            details = prev_compaction.details
            for f in details.get("readFiles") or details.get("read_files") or []:
                file_ops.read.add(f)
            for f in details.get("modifiedFiles") or details.get("modified_files") or []:
                file_ops.edited.add(f)

    for msg in messages:
        extract_file_ops_from_message(msg, file_ops)

    return file_ops


def _get_message_from_entry(entry: Entry) -> HarnessMessage | None:
    if entry.type == "message":
        return entry.message
    if entry.type == "branch_summary":
        from ..session.context import create_branch_summary_message

        return create_branch_summary_message(entry.summary, entry.from_id, entry.timestamp)
    if entry.type == "compaction":
        from ..session.context import create_compaction_summary_message

        return create_compaction_summary_message(entry.summary, entry.tokens_before, entry.timestamp)
    return None


def _get_message_from_entry_for_compaction(entry: Entry) -> HarnessMessage | None:
    if entry.type == "compaction":
        return None
    return _get_message_from_entry(entry)


@dataclass(kw_only=True)
class CompactResult:
    """Generated compaction data ready to be persisted as a compaction entry."""

    summary: str
    """Summary text that replaces compacted history in future context."""
    tokens_before: int
    """Estimated context tokens before compaction."""
    retained_tail: list[HarnessMessage]
    """Retained recent messages stored directly on the compaction entry."""
    usage: Usage | None = None
    """Usage from the LLM call(s) that generated this summary, if available."""
    details: Any = None
    """Optional implementation-specific details stored with the compaction entry."""


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


async def complete_simple_with_retries(
    stream_fn: Any,
    model: Model,
    context: Context,
    options: SimpleStreamOptions,
    retry: RetryPolicy | None = None,
    callbacks: RetryCallbacks | None = None,
) -> AssistantMessage:
    """Run one summarization request through `stream_fn`, with retries.

    See module docstring for why this takes a `StreamFn` instead of a
    `Models` registry. Summaries are standalone requests, so isolate routing
    and avoid cache writes that cannot be reused.
    """
    request_options = replace(options, cache_retention="none", session_id=uuidv7())

    async def produce() -> AssistantMessage:
        stream: Awaitable[Any] | Any = stream_fn(model, context, request_options)
        stream = await _maybe_await(stream)
        return await stream.result()

    return await retry_assistant_call(produce, retry, request_options.signal, callbacks)


def combine_usage(first: Usage, second: Usage) -> Usage:
    return Usage(
        input=first.input + second.input,
        output=first.output + second.output,
        cache_read=first.cache_read + second.cache_read,
        cache_write=first.cache_write + second.cache_write,
        cache_write_1h=(
            (first.cache_write_1h or 0) + (second.cache_write_1h or 0)
            if first.cache_write_1h is not None or second.cache_write_1h is not None
            else None
        ),
        reasoning=(
            (first.reasoning or 0) + (second.reasoning or 0)
            if first.reasoning is not None or second.reasoning is not None
            else None
        ),
        total_tokens=first.total_tokens + second.total_tokens,
        cost=replace(
            first.cost,
            input=first.cost.input + second.cost.input,
            output=first.cost.output + second.cost.output,
            cache_read=first.cost.cache_read + second.cost.cache_read,
            cache_write=first.cost.cache_write + second.cost.cache_write,
            total=first.cost.total + second.cost.total,
        ),
    )


@dataclass(kw_only=True)
class CompactionSettings:
    """Compaction thresholds and retention settings."""

    enabled: bool
    """Enable automatic compaction decisions."""
    reserve_tokens: int
    """Tokens reserved for summary prompt and output."""
    keep_recent_tokens: int
    """Approximate recent-context tokens to keep after compaction."""


DEFAULT_COMPACTION_SETTINGS = CompactionSettings(enabled=True, reserve_tokens=16384, keep_recent_tokens=20000)
"""Default compaction settings used by the harness."""


def calculate_context_tokens(usage: Usage) -> int:
    """Calculate total context tokens from provider usage."""
    return usage.total_tokens or usage.input + usage.output + usage.cache_read + usage.cache_write


def _get_assistant_usage(msg: HarnessMessage) -> Usage | None:
    if (
        msg.role == "assistant"
        and msg.stop_reason not in ("aborted", "error")
        and msg.usage
        and calculate_context_tokens(msg.usage) > 0
    ):
        return msg.usage
    return None


def get_last_assistant_usage(entries: list[Entry]) -> Usage | None:
    """Return usage from the last valid assistant message in session entries."""
    for entry in reversed(entries):
        if entry.type == "message":
            usage = _get_assistant_usage(entry.message)
            if usage:
                return usage
    return None


@dataclass(kw_only=True)
class ContextUsageEstimate:
    """Estimated context-token usage for a message list."""

    tokens: int
    """Estimated total context tokens."""
    usage_tokens: int
    """Tokens reported by the most recent assistant usage block."""
    trailing_tokens: int
    """Estimated tokens after the most recent assistant usage block."""
    last_usage_index: int | None
    """Index of the message that provided usage, or None when none exists."""


def _get_last_assistant_usage_info(messages: list[HarnessMessage]) -> tuple[Usage, int] | None:
    for i in range(len(messages) - 1, -1, -1):
        usage = _get_assistant_usage(messages[i])
        if usage:
            return usage, i
    return None


def estimate_context_tokens(messages: list[HarnessMessage]) -> ContextUsageEstimate:
    """Estimate context tokens for messages using provider usage when available."""
    usage_info = _get_last_assistant_usage_info(messages)

    if usage_info is None:
        estimated = sum(estimate_tokens(message) for message in messages)
        return ContextUsageEstimate(tokens=estimated, usage_tokens=0, trailing_tokens=estimated, last_usage_index=None)

    usage, index = usage_info
    usage_tokens = calculate_context_tokens(usage)
    trailing_tokens = sum(estimate_tokens(message) for message in messages[index + 1 :])

    return ContextUsageEstimate(
        tokens=usage_tokens + trailing_tokens,
        usage_tokens=usage_tokens,
        trailing_tokens=trailing_tokens,
        last_usage_index=index,
    )


def should_compact(context_tokens: int, context_window: int, settings: CompactionSettings) -> bool:
    """Return whether context usage exceeds the configured compaction threshold."""
    if not settings.enabled:
        return False
    return context_tokens > context_window - settings.reserve_tokens


def estimate_tokens(message: HarnessMessage) -> int:
    """Estimate token count for one message using a conservative character heuristic.

    See module docstring: delegates to `pi_ai.utils.estimate` for roles it
    already understands and only implements the harness-only roles inline.
    """
    if message.role == "assistant":
        return estimate_message_tokens(message)
    if message.role in ("user", "toolResult", "custom"):
        return estimate_text_and_image_content_tokens(message.content)
    if message.role == "bashExecution":
        chars = len(message.command) + len(message.output)
        return math.ceil(chars / 4)
    if message.role in ("branchSummary", "compactionSummary"):
        chars = len(message.summary)
        return math.ceil(chars / 4)

    return 0


def _find_valid_cut_points(entries: list[Entry], start_index: int, end_index: int) -> list[int]:
    cut_points: list[int] = []
    for i in range(start_index, end_index):
        entry = entries[i]
        if entry.type == "message":
            role = entry.message.role
            if role in ("bashExecution", "custom", "branchSummary", "compactionSummary", "user", "assistant"):
                cut_points.append(i)
        if entry.type == "branch_summary":
            cut_points.append(i)
    return cut_points


def find_turn_start_index(entries: list[Entry], entry_index: int, start_index: int) -> int:
    """Find the user-visible message that starts the turn containing an entry."""
    for i in range(entry_index, start_index - 1, -1):
        entry = entries[i]
        if entry.type == "branch_summary":
            return i
        if entry.type == "message" and entry.message.role in ("user", "bashExecution"):
            return i
    return -1


@dataclass(kw_only=True)
class CutPointResult:
    """Cut point selected for compaction."""

    first_kept_entry_index: int
    """Index of the first entry retained after compaction."""
    turn_start_index: int
    """Index of the turn-start entry when the cut splits a turn, otherwise -1."""
    is_split_turn: bool
    """Whether the selected cut point splits an in-progress turn."""


def find_cut_point(entries: list[Entry], start_index: int, end_index: int, keep_recent_tokens: int) -> CutPointResult:
    """Find the compaction cut point that keeps approximately the requested recent-token budget."""
    cut_points = _find_valid_cut_points(entries, start_index, end_index)

    if not cut_points:
        return CutPointResult(first_kept_entry_index=start_index, turn_start_index=-1, is_split_turn=False)

    accumulated_tokens = 0
    cut_index = cut_points[0]

    for i in range(end_index - 1, start_index - 1, -1):
        entry = entries[i]
        if entry.type != "message":
            continue
        message_tokens = estimate_tokens(entry.message)
        accumulated_tokens += message_tokens
        if accumulated_tokens >= keep_recent_tokens:
            for candidate in cut_points:
                if candidate >= i:
                    cut_index = candidate
                    break
            break

    while cut_index > start_index:
        prev_entry = entries[cut_index - 1]
        if prev_entry.type in ("compaction", "message"):
            break
        cut_index -= 1

    cut_entry = entries[cut_index]
    is_user_message = cut_entry.type == "message" and cut_entry.message.role == "user"
    turn_start_index = -1 if is_user_message else find_turn_start_index(entries, cut_index, start_index)

    return CutPointResult(
        first_kept_entry_index=cut_index,
        turn_start_index=turn_start_index,
        is_split_turn=(not is_user_message) and turn_start_index != -1,
    )


SUMMARIZATION_SYSTEM_PROMPT = """You are a context summarization assistant. Your task is to read a conversation between a user and an AI assistant, then produce a structured summary following the exact format specified.

Do NOT continue the conversation. Do NOT respond to any questions in the conversation. ONLY output the structured summary."""

_SUMMARIZATION_PROMPT = """The messages above are a conversation to summarize. Create a structured context checkpoint summary that another LLM will use to continue the work.

Use this EXACT format:

## Goal
[What is the user trying to accomplish? Can be multiple items if the session covers different tasks.]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned by user]
- [Or "(none)" if none were mentioned]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Current work]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [Ordered list of what should happen next]

## Critical Context
- [Any data, examples, or references needed to continue]
- [Or "(none)" if not applicable]

Keep each section concise. Preserve exact file paths, function names, and error messages."""

_UPDATE_SUMMARIZATION_PROMPT = """The messages above are NEW conversation messages to incorporate into the existing summary provided in <previous-summary> tags.

Update the existing structured summary with new information. RULES:
- PRESERVE all existing information from the previous summary
- ADD new progress, decisions, and context from the new messages
- UPDATE the Progress section: move items from "In Progress" to "Done" when completed
- UPDATE "Next Steps" based on what was accomplished
- PRESERVE exact file paths, function names, and error messages
- If something is no longer relevant, you may remove it

Use this EXACT format:

## Goal
[Preserve existing goals, add new ones if the task expanded]

## Constraints & Preferences
- [Preserve existing, add new ones discovered]

## Progress
### Done
- [x] [Include previously done items AND newly completed items]

### In Progress
- [ ] [Current work - update based on progress]

### Blocked
- [Current blockers - remove if resolved]

## Key Decisions
- **[Decision]**: [Brief rationale] (preserve all previous, add new)

## Next Steps
1. [Update based on current state]

## Critical Context
- [Preserve important context, add new if needed]

Keep each section concise. Preserve exact file paths, function names, and error messages."""


async def generate_summary(
    current_messages: list[HarnessMessage],
    stream_fn: Any,
    model: Model,
    reserve_tokens: int,
    signal: AbortSignal | None = None,
    custom_instructions: str | None = None,
    previous_summary: str | None = None,
    thinking_level: ThinkingLevel | None = None,
    retry: RetryPolicy | None = None,
    callbacks: RetryCallbacks | None = None,
) -> Result[str, CompactionError]:
    """Generate or update a conversation summary for compaction."""
    result = await generate_summary_with_usage(
        current_messages,
        stream_fn,
        model,
        reserve_tokens,
        signal,
        custom_instructions,
        previous_summary,
        thinking_level,
        retry,
        callbacks,
    )
    return ok(result.value[0]) if result.ok else err(result.error)


async def generate_summary_with_usage(
    current_messages: list[HarnessMessage],
    stream_fn: Any,
    model: Model,
    reserve_tokens: int,
    signal: AbortSignal | None = None,
    custom_instructions: str | None = None,
    previous_summary: str | None = None,
    thinking_level: ThinkingLevel | None = None,
    retry: RetryPolicy | None = None,
    callbacks: RetryCallbacks | None = None,
) -> Result[tuple[str, Usage], CompactionError]:
    """Generate or update a conversation summary and return its provider usage."""
    max_tokens = math.floor(0.8 * reserve_tokens)
    if model.max_tokens > 0:
        max_tokens = min(max_tokens, model.max_tokens)
    base_prompt = _UPDATE_SUMMARIZATION_PROMPT if previous_summary else _SUMMARIZATION_PROMPT
    if custom_instructions:
        base_prompt = f"{base_prompt}\n\nAdditional focus: {custom_instructions}"
    llm_messages = convert_to_llm(current_messages)
    conversation_text = serialize_conversation(llm_messages)
    prompt_text = f"<conversation>\n{conversation_text}\n</conversation>\n\n"
    if previous_summary:
        prompt_text += f"<previous-summary>\n{previous_summary}\n</previous-summary>\n\n"
    prompt_text += base_prompt

    from pi_ai.types import TextContent, UserMessage

    summarization_messages = [UserMessage(content=[TextContent(text=prompt_text)])]

    completion_kwargs: dict[str, Any] = {"max_tokens": max_tokens, "signal": signal}
    if model.reasoning and thinking_level and thinking_level != "off":
        completion_kwargs["reasoning"] = thinking_level
    completion_options = SimpleStreamOptions(**completion_kwargs)

    response = await complete_simple_with_retries(
        stream_fn,
        model,
        Context(system_prompt=SUMMARIZATION_SYSTEM_PROMPT, messages=summarization_messages),
        completion_options,
        retry,
        callbacks,
    )
    if response.stop_reason == "aborted":
        return err(CompactionError("aborted", response.error_message or "Summarization aborted"))
    if response.stop_reason == "error":
        return err(
            CompactionError(
                "summarization_failed", f"Summarization failed: {response.error_message or 'Unknown error'}"
            )
        )

    text_content = content_text(response.content)

    return ok((text_content, response.usage))


@dataclass(kw_only=True)
class CompactionPreparation:
    """Prepared inputs for a compaction run."""

    messages_to_summarize: list[HarnessMessage]
    """Messages summarized into the history summary."""
    turn_prefix_messages: list[HarnessMessage]
    """Prefix messages summarized separately when compaction splits a turn."""
    retained_tail: list[HarnessMessage]
    """Recent messages retained after compaction and stored on the compaction entry."""
    is_split_turn: bool
    """Whether compaction splits a turn."""
    tokens_before: int
    """Estimated context tokens before compaction."""
    file_ops: FileOperations
    """File operations extracted from summarized history."""
    settings: CompactionSettings
    """Settings used to prepare compaction."""
    previous_summary: str | None = None
    """Previous compaction summary used for iterative updates."""


def prepare_compaction(
    path_entries: list[Entry], settings: CompactionSettings
) -> Result[CompactionPreparation | None, CompactionError]:
    """Prepare session entries for compaction, or return None when compaction is not applicable."""
    if not path_entries or path_entries[-1].type == "compaction":
        return ok(None)

    prev_compaction_index = -1
    for i in range(len(path_entries) - 1, -1, -1):
        if path_entries[i].type == "compaction":
            prev_compaction_index = i
            break

    previous_summary: str | None = None
    compactable_entries = path_entries
    if prev_compaction_index >= 0:
        prev_compaction = path_entries[prev_compaction_index]
        assert isinstance(prev_compaction, CompactionEntry)
        previous_summary = prev_compaction.summary
        virtual_retained_entries: list[Entry] = []
        for index, message in enumerate(prev_compaction.retained_tail):
            from ..session.types import MessageEntry

            virtual_retained_entries.append(
                MessageEntry(
                    id=f"{prev_compaction.id}:retained:{index}",
                    parent_id=(prev_compaction.id if index == 0 else f"{prev_compaction.id}:retained:{index - 1}"),
                    seq=prev_compaction.seq,
                    timestamp=message.timestamp,
                    message=message,
                )
            )
        compactable_entries = [*virtual_retained_entries, *path_entries[prev_compaction_index + 1 :]]
    boundary_end = len(compactable_entries)

    tokens_before = estimate_context_tokens(build_session_context(path_entries).messages).tokens

    cut_point = find_cut_point(compactable_entries, 0, boundary_end, settings.keep_recent_tokens)
    history_end = cut_point.turn_start_index if cut_point.is_split_turn else cut_point.first_kept_entry_index
    messages_to_summarize: list[HarnessMessage] = []
    for i in range(history_end):
        msg = _get_message_from_entry_for_compaction(compactable_entries[i])
        if msg:
            messages_to_summarize.append(msg)
    turn_prefix_messages: list[HarnessMessage] = []
    if cut_point.is_split_turn:
        for i in range(cut_point.turn_start_index, cut_point.first_kept_entry_index):
            msg = _get_message_from_entry_for_compaction(compactable_entries[i])
            if msg:
                turn_prefix_messages.append(msg)
    retained_tail: list[HarnessMessage] = []
    for i in range(cut_point.first_kept_entry_index, boundary_end):
        msg = _get_message_from_entry_for_compaction(compactable_entries[i])
        if msg:
            retained_tail.append(msg)
    file_ops = _extract_file_operations(messages_to_summarize, path_entries, prev_compaction_index)
    if cut_point.is_split_turn:
        for msg in turn_prefix_messages:
            extract_file_ops_from_message(msg, file_ops)

    return ok(
        CompactionPreparation(
            messages_to_summarize=messages_to_summarize,
            turn_prefix_messages=turn_prefix_messages,
            retained_tail=retained_tail,
            is_split_turn=cut_point.is_split_turn,
            tokens_before=tokens_before,
            previous_summary=previous_summary,
            file_ops=file_ops,
            settings=settings,
        )
    )


_TURN_PREFIX_SUMMARIZATION_PROMPT = """This is the PREFIX of a turn that was too large to keep. The SUFFIX (recent work) is retained.

Summarize the prefix to provide context for the retained suffix:

## Original Request
[What did the user ask for in this turn?]

## Early Progress
- [Key decisions and work done in the prefix]

## Context for Suffix
- [Information needed to understand the retained recent work]

Be concise. Focus on what's needed to understand the kept suffix."""


async def compact(
    preparation: CompactionPreparation,
    stream_fn: Any,
    model: Model,
    custom_instructions: str | None = None,
    signal: AbortSignal | None = None,
    thinking_level: ThinkingLevel | None = None,
    retry: RetryPolicy | None = None,
    callbacks: RetryCallbacks | None = None,
) -> Result[CompactResult, CompactionError]:
    """Generate compaction summary data from prepared session history."""
    messages_to_summarize = preparation.messages_to_summarize
    turn_prefix_messages = preparation.turn_prefix_messages
    retained_tail = preparation.retained_tail
    is_split_turn = preparation.is_split_turn
    tokens_before = preparation.tokens_before
    previous_summary = preparation.previous_summary
    file_ops = preparation.file_ops
    settings = preparation.settings

    if is_split_turn and turn_prefix_messages:
        history_text = "No prior history."
        history_usage: Usage | None = None
        if messages_to_summarize:
            history_result = await generate_summary_with_usage(
                messages_to_summarize,
                stream_fn,
                model,
                settings.reserve_tokens,
                signal,
                custom_instructions,
                previous_summary,
                thinking_level,
                retry,
                callbacks,
            )
            if not history_result.ok:
                return err(history_result.error)
            history_text, history_usage = history_result.value
        turn_prefix_result = await _generate_turn_prefix_summary(
            turn_prefix_messages, stream_fn, model, settings.reserve_tokens, signal, thinking_level, retry, callbacks
        )
        if not turn_prefix_result.ok:
            return err(turn_prefix_result.error)
        turn_prefix_text, turn_prefix_usage = turn_prefix_result.value
        summary = f"{history_text}\n\n---\n\n**Turn Context (split turn):**\n\n{turn_prefix_text}"
        summary_usage = combine_usage(history_usage, turn_prefix_usage) if history_usage else turn_prefix_usage
    else:
        summary_result = await generate_summary_with_usage(
            messages_to_summarize,
            stream_fn,
            model,
            settings.reserve_tokens,
            signal,
            custom_instructions,
            previous_summary,
            thinking_level,
            retry,
            callbacks,
        )
        if not summary_result.ok:
            return err(summary_result.error)
        summary, summary_usage = summary_result.value

    read_files, modified_files = compute_file_lists(file_ops)
    summary += format_file_operations(read_files, modified_files)

    return ok(
        CompactResult(
            summary=summary,
            tokens_before=tokens_before,
            usage=summary_usage,
            retained_tail=retained_tail,
            details=CompactionDetails(read_files=read_files, modified_files=modified_files),
        )
    )


async def _generate_turn_prefix_summary(
    messages: list[HarnessMessage],
    stream_fn: Any,
    model: Model,
    reserve_tokens: int,
    signal: AbortSignal | None = None,
    thinking_level: ThinkingLevel | None = None,
    retry: RetryPolicy | None = None,
    callbacks: RetryCallbacks | None = None,
) -> Result[tuple[str, Usage], CompactionError]:
    max_tokens = math.floor(0.5 * reserve_tokens)
    if model.max_tokens > 0:
        max_tokens = min(max_tokens, model.max_tokens)
    llm_messages = convert_to_llm(messages)
    conversation_text = serialize_conversation(llm_messages)
    prompt_text = f"<conversation>\n{conversation_text}\n</conversation>\n\n{_TURN_PREFIX_SUMMARIZATION_PROMPT}"

    from pi_ai.types import TextContent, UserMessage

    summarization_messages = [UserMessage(content=[TextContent(text=prompt_text)])]

    completion_kwargs: dict[str, Any] = {"max_tokens": max_tokens, "signal": signal}
    if model.reasoning and thinking_level and thinking_level != "off":
        completion_kwargs["reasoning"] = thinking_level
    completion_options = SimpleStreamOptions(**completion_kwargs)

    response = await complete_simple_with_retries(
        stream_fn,
        model,
        Context(system_prompt=SUMMARIZATION_SYSTEM_PROMPT, messages=summarization_messages),
        completion_options,
        retry,
        callbacks,
    )
    if response.stop_reason == "aborted":
        return err(CompactionError("aborted", response.error_message or "Turn prefix summarization aborted"))
    if response.stop_reason == "error":
        return err(
            CompactionError(
                "summarization_failed",
                f"Turn prefix summarization failed: {response.error_message or 'Unknown error'}",
            )
        )

    return ok((content_text(response.content), response.usage))


__all__ = [
    "DEFAULT_COMPACTION_SETTINGS",
    "ESTIMATED_IMAGE_CHARS",
    "SUMMARIZATION_SYSTEM_PROMPT",
    "CompactResult",
    "CompactionDetails",
    "CompactionPreparation",
    "CompactionSettings",
    "ContextUsageEstimate",
    "CutPointResult",
    "calculate_context_tokens",
    "combine_usage",
    "compact",
    "complete_simple_with_retries",
    "estimate_context_tokens",
    "estimate_tokens",
    "find_cut_point",
    "find_turn_start_index",
    "generate_summary",
    "generate_summary_with_usage",
    "get_last_assistant_usage",
    "prepare_compaction",
    "serialize_conversation",
    "should_compact",
]
