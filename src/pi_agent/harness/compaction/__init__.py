"""Compaction: history summarization and branch summarization.

Python port of `packages/agent/src/harness/compaction/index.ts` (no such
barrel file exists upstream; `compaction.ts`/`branch-summarization.ts` are
imported directly by their callers there). This `__init__.py` re-exports
both modules plus the shared `utils.py` helpers for convenience.
"""

from __future__ import annotations

from .branch_summarization import (
    BranchPreparation,
    BranchSummaryDetails,
    BranchSummaryResult,
    CollectEntriesResult,
    GenerateBranchSummaryOptions,
    collect_entries_for_branch_summary,
    generate_branch_summary,
    prepare_branch_entries,
)
from .compaction import (
    DEFAULT_COMPACTION_SETTINGS,
    ESTIMATED_IMAGE_CHARS,
    SUMMARIZATION_SYSTEM_PROMPT,
    CompactionDetails,
    CompactionPreparation,
    CompactionSettings,
    CompactResult,
    ContextUsageEstimate,
    CutPointResult,
    calculate_context_tokens,
    combine_usage,
    compact,
    complete_simple_with_retries,
    estimate_context_tokens,
    estimate_tokens,
    find_cut_point,
    find_turn_start_index,
    generate_summary,
    generate_summary_with_usage,
    get_last_assistant_usage,
    prepare_compaction,
    should_compact,
)
from .utils import (
    TOOL_RESULT_MAX_CHARS,
    FileOperations,
    compute_file_lists,
    create_file_ops,
    extract_file_ops_from_message,
    format_file_operations,
    serialize_conversation,
)

__all__ = [
    "DEFAULT_COMPACTION_SETTINGS",
    "ESTIMATED_IMAGE_CHARS",
    "SUMMARIZATION_SYSTEM_PROMPT",
    "TOOL_RESULT_MAX_CHARS",
    "BranchPreparation",
    "BranchSummaryDetails",
    "BranchSummaryResult",
    "CollectEntriesResult",
    "CompactResult",
    "CompactionDetails",
    "CompactionPreparation",
    "CompactionSettings",
    "ContextUsageEstimate",
    "CutPointResult",
    "FileOperations",
    "GenerateBranchSummaryOptions",
    "calculate_context_tokens",
    "collect_entries_for_branch_summary",
    "combine_usage",
    "compact",
    "complete_simple_with_retries",
    "compute_file_lists",
    "create_file_ops",
    "estimate_context_tokens",
    "estimate_tokens",
    "extract_file_ops_from_message",
    "find_cut_point",
    "find_turn_start_index",
    "format_file_operations",
    "generate_branch_summary",
    "generate_summary",
    "generate_summary_with_usage",
    "get_last_assistant_usage",
    "prepare_branch_entries",
    "prepare_compaction",
    "serialize_conversation",
    "should_compact",
]
