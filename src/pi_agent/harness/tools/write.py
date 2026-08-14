"""The harness `write` tool.

Python port of `packages/agent/src/harness/tools/write.ts`.
"""

from __future__ import annotations

from typing import Any

from pi_ai.types import TextContent
from pi_ai.utils.abort import AbortSignal

from ..types import AgentHarnessTool, AgentToolResult, AgentToolUpdateCallback, get_or_throw
from .file_mutation_queue import with_file_mutation_queue
from .path_utils import resolve_tool_path
from .tool_context import ExecutionToolContext

WRITE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "Path to the file to write (relative or absolute)"},
        "content": {"type": "string", "description": "Content to write to the file"},
    },
    "required": ["path", "content"],
}


def create_write_tool() -> AgentHarnessTool[ExecutionToolContext]:
    """Create the `write` tool: overwrite a file, creating parents as needed."""

    async def execute(
        _tool_call_id: str,
        args: dict[str, Any],
        signal: AbortSignal | None,
        _on_update: AgentToolUpdateCallback | None,
        context: ExecutionToolContext,
    ) -> AgentToolResult:
        path = args["path"]
        content = args["content"]
        env = context.env
        absolute_path = await resolve_tool_path(env, path, signal)

        async def write() -> AgentToolResult:
            if signal is not None and signal.aborted:
                raise RuntimeError("Operation aborted")
            get_or_throw(await env.write_file(absolute_path, content, signal))
            if signal is not None and signal.aborted:
                raise RuntimeError("Operation aborted")
            return AgentToolResult(
                content=[TextContent(text=f"Successfully wrote {len(content)} bytes to {path}")],
                details=None,
            )

        return await with_file_mutation_queue(env, absolute_path, write)

    return AgentHarnessTool(
        name="write",
        label="write",
        description=(
            "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. "
            "Automatically creates parent directories."
        ),
        parameters=WRITE_SCHEMA,
        execute_with_context=execute,
    )


__all__ = ["WRITE_SCHEMA", "create_write_tool"]
