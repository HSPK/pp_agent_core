"""The harness `read` tool.

Python port of `packages/agent/src/harness/tools/read.ts`.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from pi_ai.types import ImageContent, TextContent
from pi_ai.utils.abort import AbortSignal

from ..types import AgentHarnessTool, AgentToolResult, AgentToolUpdateCallback, get_or_throw
from ..utils.truncate import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, TruncationResult, format_size, truncate_head
from .image import detect_supported_image_mime_type, encode_base64
from .path_utils import resolve_read_tool_path
from .tool_context import ExecutionToolContext

READ_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "Path to the file to read (relative or absolute)"},
        "offset": {"type": "number", "description": "Line number to start reading from (1-indexed)"},
        "limit": {"type": "number", "description": "Maximum number of lines to read"},
    },
    "required": ["path"],
}


@dataclass
class ReadToolDetails:
    truncation: TruncationResult | None = None


@dataclass
class ReadImageProcessorResult:
    """Either a converted image (`ok=True`) or a human-readable failure message."""

    ok: bool
    data: str = ""
    mime_type: str = ""
    hints: list[str] | None = None
    message: str = ""


ReadImageProcessor = Callable[[bytes, str, bool], Awaitable[ReadImageProcessorResult]]
"""`(bytes, mime_type, auto_resize_images) -> ReadImageProcessorResult`."""


def create_read_tool(
    auto_resize_images: bool = True, image_processor: ReadImageProcessor | None = None
) -> AgentHarnessTool[ExecutionToolContext]:
    """Create the `read` tool: read text with offset/limit, or sniff and attach images."""

    async def execute(
        _tool_call_id: str,
        args: dict[str, Any],
        signal: AbortSignal | None,
        _on_update: AgentToolUpdateCallback | None,
        context: ExecutionToolContext,
    ) -> AgentToolResult:
        path = args["path"]
        offset = args.get("offset")
        limit = args.get("limit")
        env = context.env
        absolute_path = await resolve_read_tool_path(env, path, signal)
        data = get_or_throw(await env.read_binary_file(absolute_path, signal))
        mime_type = detect_supported_image_mime_type(data)
        if mime_type is not None:
            return await _read_image(data, mime_type, auto_resize_images, image_processor)

        text_content = data.decode("utf-8", errors="replace")
        all_lines = text_content.split("\n")
        total_file_lines = len(all_lines)
        start_line = max(0, int(offset) - 1) if offset else 0
        start_line_display = start_line + 1
        if start_line >= len(all_lines):
            raise ValueError(f"Offset {offset} is beyond end of file ({len(all_lines)} lines total)")

        user_limited_lines: int | None = None
        if limit is not None:
            end_line = min(start_line + int(limit), len(all_lines))
            selected_content = "\n".join(all_lines[start_line:end_line])
            user_limited_lines = end_line - start_line
        else:
            selected_content = "\n".join(all_lines[start_line:])

        truncation = truncate_head(selected_content)
        details: ReadToolDetails | None = None
        if truncation.first_line_exceeds_limit:
            first_line_size = format_size(len(all_lines[start_line].encode("utf-8")))
            output_text = (
                f"[Line {start_line_display} is {first_line_size}, exceeds {format_size(DEFAULT_MAX_BYTES)} limit. "
                f"Use bash: sed -n '{start_line_display}p' {path} | head -c {DEFAULT_MAX_BYTES}]"
            )
            details = ReadToolDetails(truncation=truncation)
        elif truncation.truncated:
            end_line_display = start_line_display + truncation.output_lines - 1
            next_offset = end_line_display + 1
            output_text = truncation.content
            if truncation.truncated_by == "lines":
                output_text += (
                    f"\n\n[Showing lines {start_line_display}-{end_line_display} of {total_file_lines}. "
                    f"Use offset={next_offset} to continue.]"
                )
            else:
                output_text += (
                    f"\n\n[Showing lines {start_line_display}-{end_line_display} of {total_file_lines} "
                    f"({format_size(DEFAULT_MAX_BYTES)} limit). Use offset={next_offset} to continue.]"
                )
            details = ReadToolDetails(truncation=truncation)
        elif user_limited_lines is not None and start_line + user_limited_lines < len(all_lines):
            remaining = len(all_lines) - (start_line + user_limited_lines)
            next_offset = start_line + user_limited_lines + 1
            output_text = (
                f"{truncation.content}\n\n[{remaining} more lines in file. Use offset={next_offset} to continue.]"
            )
        else:
            output_text = truncation.content

        return AgentToolResult(content=[TextContent(text=output_text)], details=details)

    return AgentHarnessTool(
        name="read",
        label="read",
        description=(
            "Read the contents of a file. Supports text files and images (jpg, png, gif, webp, bmp). "
            "Images are sent as attachments. For text files, output is truncated to "
            f"{DEFAULT_MAX_LINES} lines or {DEFAULT_MAX_BYTES // 1024}KB (whichever is hit first). "
            "Use offset/limit for large files. When you need the full file, continue with offset until complete."
        ),
        parameters=READ_SCHEMA,
        execute_with_context=execute,
    )


async def _read_image(
    data: bytes,
    mime_type: str,
    auto_resize_images: bool,
    image_processor: ReadImageProcessor | None,
) -> AgentToolResult:
    if image_processor is not None:
        processed = await image_processor(data, mime_type, auto_resize_images)
        if not processed.ok:
            return AgentToolResult(
                content=[TextContent(text=f"Read image file [{mime_type}]\n{processed.message}")],
                details=None,
            )
        hints = "\n" + "\n".join(processed.hints) if processed.hints else ""
        return AgentToolResult(
            content=[
                TextContent(text=f"Read image file [{processed.mime_type}]{hints}"),
                ImageContent(data=processed.data, mime_type=processed.mime_type),
            ],
            details=None,
        )
    if mime_type == "image/bmp":
        return AgentToolResult(
            content=[
                TextContent(
                    text=(
                        "Read image file [image/bmp]\n"
                        "[Image omitted: configure an imageProcessor to convert BMP images.]"
                    )
                )
            ],
            details=None,
        )
    return AgentToolResult(
        content=[
            TextContent(text=f"Read image file [{mime_type}]"),
            ImageContent(data=encode_base64(data), mime_type=mime_type),
        ],
        details=None,
    )


__all__ = [
    "READ_SCHEMA",
    "ReadImageProcessor",
    "ReadImageProcessorResult",
    "ReadToolDetails",
    "create_read_tool",
]
