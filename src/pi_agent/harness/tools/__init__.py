"""Built-in execution tools for the agent harness.

Python port of `packages/agent/src/harness/tools/`. Each tool takes its
filesystem/shell capability from the `ExecutionToolContext` it is handed at
execution time, so it works against any `ExecutionEnv` (see
`pi_agent.harness.env.LocalExecutionEnv` for the local backend).

`pi_coding_agent.tools` contains a separate, richer set of tools ported from
`packages/coding-agent/src/core/tools/`; these are the slimmer harness
versions that the TypeScript package also ships on its own.
"""

from __future__ import annotations

from .bash import (
    BASH_SCHEMA,
    BashExecution,
    BashPrepare,
    BashToolDetails,
    create_bash_tool,
)
from .edit import EDIT_SCHEMA, EditToolDetails, create_edit_tool, prepare_edit_arguments
from .edit_diff import (
    AppliedEditsResult,
    Edit,
    apply_edits_to_normalized_content,
    detect_line_ending,
    generate_diff_string,
    generate_unified_patch,
    normalize_to_lf,
    restore_line_endings,
    strip_bom,
)
from .file_mutation_queue import with_file_mutation_queue
from .image import detect_supported_image_mime_type, encode_base64
from .path_utils import normalize_tool_path, resolve_read_tool_path, resolve_tool_path
from .read import (
    READ_SCHEMA,
    ReadImageProcessor,
    ReadImageProcessorResult,
    ReadToolDetails,
    create_read_tool,
)
from .tool_context import ExecutionToolContext
from .write import WRITE_SCHEMA, create_write_tool

__all__ = [
    "BASH_SCHEMA",
    "EDIT_SCHEMA",
    "READ_SCHEMA",
    "WRITE_SCHEMA",
    "AppliedEditsResult",
    "BashExecution",
    "BashPrepare",
    "BashToolDetails",
    "Edit",
    "EditToolDetails",
    "ExecutionToolContext",
    "ReadImageProcessor",
    "ReadImageProcessorResult",
    "ReadToolDetails",
    "apply_edits_to_normalized_content",
    "create_bash_tool",
    "create_edit_tool",
    "create_read_tool",
    "create_write_tool",
    "detect_line_ending",
    "detect_supported_image_mime_type",
    "encode_base64",
    "generate_diff_string",
    "generate_unified_patch",
    "normalize_to_lf",
    "normalize_tool_path",
    "prepare_edit_arguments",
    "resolve_read_tool_path",
    "resolve_tool_path",
    "restore_line_endings",
    "strip_bom",
    "with_file_mutation_queue",
]
