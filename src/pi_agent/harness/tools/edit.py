"""The harness `edit` tool.

Python port of `packages/agent/src/harness/tools/edit.ts`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from pi_ai.types import TextContent
from pi_ai.utils.abort import AbortSignal

from ..types import AgentHarnessTool, AgentToolResult, AgentToolUpdateCallback, FileError
from .edit_diff import (
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
from .path_utils import resolve_tool_path
from .tool_context import ExecutionToolContext

REPLACE_EDIT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "oldText": {
            "type": "string",
            "description": (
                "Exact text for one targeted replacement. It must be unique in the original file and must not "
                "overlap with any other edits[].oldText in the same call."
            ),
        },
        "newText": {"type": "string", "description": "Replacement text for this targeted edit."},
    },
    "required": ["oldText", "newText"],
}

EDIT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "Path to the file to edit (relative or absolute)"},
        "edits": {
            "type": "array",
            "items": REPLACE_EDIT_SCHEMA,
            "description": (
                "One or more targeted replacements. Each edit is matched against the original file, not "
                "incrementally. Do not include overlapping or nested edits. If two changes touch the same block "
                "or nearby lines, merge them into one edit instead."
            ),
        },
    },
    "required": ["path", "edits"],
}


@dataclass
class EditToolDetails:
    diff: str
    patch: str
    first_changed_line: int | None = None


def prepare_edit_arguments(args: Any) -> dict[str, Any]:
    """Accept a JSON-encoded `edits` string and the legacy flat `oldText`/`newText` shape."""
    if not isinstance(args, dict):
        return args
    prepared = dict(args)
    if isinstance(prepared.get("edits"), str):
        try:
            parsed = json.loads(prepared["edits"])
        except ValueError:
            parsed = None
        if isinstance(parsed, list):
            prepared["edits"] = parsed

    if not isinstance(prepared.get("oldText"), str) or not isinstance(prepared.get("newText"), str):
        return prepared
    edits = list(prepared["edits"]) if isinstance(prepared.get("edits"), list) else []
    edits.append({"oldText": prepared["oldText"], "newText": prepared["newText"]})
    prepared.pop("oldText")
    prepared.pop("newText")
    prepared["edits"] = edits
    return prepared


def _validate_edit_input(args: dict[str, Any]) -> tuple[str, list[Edit]]:
    edits = args.get("edits")
    if not isinstance(edits, list) or len(edits) == 0:
        raise ValueError("Edit tool input is invalid. edits must contain at least one replacement.")
    return args["path"], [Edit(old_text=edit["oldText"], new_text=edit["newText"]) for edit in edits]


def _edit_access_error(path: str, error: FileError) -> Exception:
    failure = RuntimeError(f"Could not edit file: {path}. Error code: {error.code}.")
    failure.__cause__ = error
    return failure


def create_edit_tool() -> AgentHarnessTool[ExecutionToolContext]:
    """Create the `edit` tool: exact, non-overlapping text replacements against the original file."""

    async def execute(
        _tool_call_id: str,
        args: dict[str, Any],
        signal: AbortSignal | None,
        _on_update: AgentToolUpdateCallback | None,
        context: ExecutionToolContext,
    ) -> AgentToolResult:
        path, edits = _validate_edit_input(args)
        env = context.env
        absolute_path = await resolve_tool_path(env, path, signal)

        async def apply() -> AgentToolResult:
            if signal is not None and signal.aborted:
                raise RuntimeError("Operation aborted")
            info = await env.file_info(absolute_path, signal)
            if not info.ok:
                raise _edit_access_error(path, info.error)
            if info.value.kind not in ("file", "symlink"):
                raise RuntimeError(f"Could not edit file: {path}. Path is not a file.")

            read_result = await env.read_text_file(absolute_path, signal)
            if not read_result.ok:
                raise _edit_access_error(path, read_result.error)
            if signal is not None and signal.aborted:
                raise RuntimeError("Operation aborted")

            bom, content = strip_bom(read_result.value)
            original_ending = detect_line_ending(content)
            normalized_content = normalize_to_lf(content)
            applied = apply_edits_to_normalized_content(normalized_content, edits, path)
            if signal is not None and signal.aborted:
                raise RuntimeError("Operation aborted")

            final_content = bom + restore_line_endings(applied.new_content, original_ending)
            write_result = await env.write_file(absolute_path, final_content, signal)
            if not write_result.ok:
                raise _edit_access_error(path, write_result.error)
            if signal is not None and signal.aborted:
                raise RuntimeError("Operation aborted")

            diff, first_changed_line = generate_diff_string(applied.base_content, applied.new_content)
            return AgentToolResult(
                content=[TextContent(text=f"Successfully replaced {len(edits)} block(s) in {path}.")],
                details=EditToolDetails(
                    diff=diff,
                    patch=generate_unified_patch(path, applied.base_content, applied.new_content),
                    first_changed_line=first_changed_line,
                ),
            )

        return await with_file_mutation_queue(env, absolute_path, apply)

    return AgentHarnessTool(
        name="edit",
        label="edit",
        description=(
            "Edit a single file using exact text replacement. Every edits[].oldText must match a unique, "
            "non-overlapping region of the original file. If two changes affect the same block or nearby lines, "
            "merge them into one edit instead of emitting overlapping edits. Do not include large unchanged "
            "regions just to connect distant changes."
        ),
        parameters=EDIT_SCHEMA,
        prepare_arguments=prepare_edit_arguments,
        execute_with_context=execute,
    )


__all__ = [
    "EDIT_SCHEMA",
    "REPLACE_EDIT_SCHEMA",
    "EditToolDetails",
    "create_edit_tool",
    "prepare_edit_arguments",
]
