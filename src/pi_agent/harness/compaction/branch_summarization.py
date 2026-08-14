"""Branch summarization: summarize an abandoned conversation branch on navigation.

Python port of `packages/agent/src/harness/compaction/branch-summarization.ts`.

Like `compaction.py`, `generate_branch_summary` takes a `StreamFn` plus a
`pi_ai.types.Model` instead of a `Models` registry (see that module's
docstring for the rationale).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pi_ai.types import Context, Model, SimpleStreamOptions, TextContent, Usage, UserMessage
from pi_ai.utils.abort import AbortSignal
from pi_ai.utils.retry import RetryCallbacks, RetryPolicy
from pi_ai.utils.text import content_text

from ..messages import HarnessMessage, convert_to_llm
from ..session import Entry, Session, SessionError
from ..session.context import create_branch_summary_message, create_compaction_summary_message
from ..types import BranchSummaryError, Result, err, ok
from .compaction import SUMMARIZATION_SYSTEM_PROMPT, complete_simple_with_retries, estimate_tokens
from .utils import (
    FileOperations,
    compute_file_lists,
    create_file_ops,
    extract_file_ops_from_message,
    format_file_operations,
    serialize_conversation,
)


@dataclass(kw_only=True)
class BranchSummaryResult:
    """Generated branch summary data ready to be persisted as a branch-summary entry."""

    summary: str
    read_files: list[str]
    modified_files: list[str]
    usage: Usage | None = None


@dataclass(kw_only=True)
class BranchSummaryDetails:
    """File-operation details stored on generated branch summary entries."""

    read_files: list[str]
    """Files read while exploring the summarized branch."""
    modified_files: list[str]
    """Files modified while exploring the summarized branch."""


@dataclass(kw_only=True)
class BranchPreparation:
    """Prepared branch content for summarization."""

    messages: list[HarnessMessage]
    """Messages selected for the branch summary."""
    file_ops: FileOperations
    """File operations extracted from the branch."""
    total_tokens: int
    """Estimated token count for selected messages."""


@dataclass(kw_only=True)
class CollectEntriesResult:
    """Entries selected for branch summarization."""

    entries: list[Entry]
    """Entries to summarize in chronological order."""
    common_ancestor_id: str | None
    """Deepest common ancestor between the previous leaf and target entry."""


@dataclass(kw_only=True)
class GenerateBranchSummaryOptions:
    """Options for generating a branch summary."""

    stream_fn: Any
    """`StreamFn` the summarization request goes through."""
    model: Model
    """Model used for summarization."""
    signal: AbortSignal | None = None
    """Abort signal for the summarization request."""
    custom_instructions: str | None = None
    """Optional instructions appended to or replacing the default prompt."""
    replace_instructions: bool = False
    """Replace the default prompt with custom instructions instead of appending them."""
    reserve_tokens: int = 16384
    """Tokens reserved for prompt and model output. Defaults to 16384."""
    retry: RetryPolicy | None = None
    """Optional retry policy for transient summarization errors."""
    callbacks: RetryCallbacks | None = None
    """Optional callbacks for retry reporting."""


async def collect_entries_for_branch_summary(
    session: Session, old_leaf_id: str | None, target_id: str
) -> CollectEntriesResult:
    """Collect entries that should be summarized before navigating to a different session tree entry."""
    if not old_leaf_id:
        return CollectEntriesResult(entries=[], common_ancestor_id=None)

    from ..session.types import BranchBounds

    old_path = {entry.id for entry in await session.find_entries_on_branch(bounds=BranchBounds(start=old_leaf_id))}
    target_path = await session.find_entries_on_branch(bounds=BranchBounds(start=target_id))
    common_ancestor_id: str | None = None
    for entry in target_path:
        if entry.id in old_path:
            common_ancestor_id = entry.id
            break

    entries: list[Entry] = []
    current: str | None = old_leaf_id

    while current and current != common_ancestor_id:
        entry = await session.get_entry(current)
        if entry is None:
            raise SessionError("invalid_entry", f"Entry {current} not found")
        entries.append(entry)
        current = entry.parent_id
    entries.reverse()

    return CollectEntriesResult(entries=entries, common_ancestor_id=common_ancestor_id)


def _get_message_from_entry(entry: Entry) -> HarnessMessage | None:
    if entry.type == "message":
        if entry.message.role == "toolResult":
            return None
        return entry.message
    if entry.type == "branch_summary":
        return create_branch_summary_message(entry.summary, entry.from_id, entry.timestamp)
    if entry.type == "compaction":
        return create_compaction_summary_message(entry.summary, entry.tokens_before, entry.timestamp)
    return None


def prepare_branch_entries(entries: list[Entry], token_budget: int = 0) -> BranchPreparation:
    """Prepare branch entries for summarization within an optional token budget."""
    messages: list[HarnessMessage] = []
    file_ops = create_file_ops()
    total_tokens = 0
    for entry in entries:
        if entry.type == "branch_summary" and isinstance(entry.details, dict):
            details = entry.details
            for f in details.get("readFiles") or details.get("read_files") or []:
                file_ops.read.add(f)
            for f in details.get("modifiedFiles") or details.get("modified_files") or []:
                file_ops.edited.add(f)

    for entry in reversed(entries):
        message = _get_message_from_entry(entry)
        if message is None:
            continue
        extract_file_ops_from_message(message, file_ops)

        tokens = estimate_tokens(message)
        if token_budget > 0 and total_tokens + tokens > token_budget:
            if entry.type in ("compaction", "branch_summary") and total_tokens < token_budget * 0.9:
                messages.insert(0, message)
                total_tokens += tokens
            break

        messages.insert(0, message)
        total_tokens += tokens

    return BranchPreparation(messages=messages, file_ops=file_ops, total_tokens=total_tokens)


_BRANCH_SUMMARY_PREAMBLE = """The user explored a different conversation branch before returning here.
Summary of that exploration:

"""

_BRANCH_SUMMARY_PROMPT = """Create a structured summary of this conversation branch for context when returning later.

Use this EXACT format:

## Goal
[What was the user trying to accomplish in this branch?]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned]
- [Or "(none)" if none were mentioned]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Work that was started but not finished]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [What should happen next to continue this work]

Keep each section concise. Preserve exact file paths, function names, and error messages."""


async def generate_branch_summary(
    entries: list[Entry], options: GenerateBranchSummaryOptions
) -> Result[BranchSummaryResult, BranchSummaryError]:
    """Generate a summary for abandoned branch entries."""
    stream_fn = options.stream_fn
    model = options.model
    signal = options.signal
    custom_instructions = options.custom_instructions
    replace_instructions = options.replace_instructions
    reserve_tokens = options.reserve_tokens
    retry = options.retry
    callbacks = options.callbacks

    context_window = model.context_window or 128000
    token_budget = context_window - reserve_tokens

    preparation = prepare_branch_entries(entries, token_budget)
    messages = preparation.messages
    file_ops = preparation.file_ops

    if not messages:
        return ok(BranchSummaryResult(summary="No content to summarize", read_files=[], modified_files=[]))

    llm_messages = convert_to_llm(messages)
    conversation_text = serialize_conversation(llm_messages)
    if replace_instructions and custom_instructions:
        instructions = custom_instructions
    elif custom_instructions:
        instructions = f"{_BRANCH_SUMMARY_PROMPT}\n\nAdditional focus: {custom_instructions}"
    else:
        instructions = _BRANCH_SUMMARY_PROMPT
    prompt_text = f"<conversation>\n{conversation_text}\n</conversation>\n\n{instructions}"

    summarization_messages = [UserMessage(content=[TextContent(text=prompt_text)])]
    response = await complete_simple_with_retries(
        stream_fn,
        model,
        Context(system_prompt=SUMMARIZATION_SYSTEM_PROMPT, messages=summarization_messages),
        SimpleStreamOptions(signal=signal, max_tokens=2048),
        retry,
        callbacks,
    )
    if response.stop_reason == "aborted":
        return err(BranchSummaryError("aborted", response.error_message or "Branch summary aborted"))
    if response.stop_reason == "error":
        return err(
            BranchSummaryError(
                "summarization_failed", f"Branch summary failed: {response.error_message or 'Unknown error'}"
            )
        )

    summary = content_text(response.content)
    summary = _BRANCH_SUMMARY_PREAMBLE + summary
    read_files, modified_files = compute_file_lists(file_ops)
    summary += format_file_operations(read_files, modified_files)

    return ok(
        BranchSummaryResult(
            summary=summary or "No summary generated",
            usage=response.usage,
            read_files=read_files,
            modified_files=modified_files,
        )
    )


__all__ = [
    "BranchPreparation",
    "BranchSummaryDetails",
    "BranchSummaryResult",
    "CollectEntriesResult",
    "FileOperations",
    "GenerateBranchSummaryOptions",
    "collect_entries_for_branch_summary",
    "generate_branch_summary",
    "prepare_branch_entries",
]
