"""File-operation tracking and conversation serialization helpers for compaction.

Python port of `packages/agent/src/harness/compaction/utils.ts`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from pi_ai.types import Message
from pi_ai.utils.text import content_text

from ..messages import HarnessMessage

TOOL_RESULT_MAX_CHARS = 2000


@dataclass
class FileOperations:
    """File paths touched by a session branch or compaction range."""

    read: set[str] = field(default_factory=set)
    """Files read but not necessarily modified."""
    written: set[str] = field(default_factory=set)
    """Files written by full-file write operations."""
    edited: set[str] = field(default_factory=set)
    """Files modified by edit operations."""


def create_file_ops() -> FileOperations:
    """Create an empty file-operation accumulator."""
    return FileOperations()


def extract_file_ops_from_message(message: HarnessMessage, file_ops: FileOperations) -> None:
    """Add file operations from assistant tool calls to an accumulator."""
    if message.role != "assistant":
        return

    for block in message.content:
        if block.type != "toolCall":
            continue
        args = block.arguments
        if not args:
            continue
        path = args.get("path")
        if not isinstance(path, str) or not path:
            continue

        if block.name == "read":
            file_ops.read.add(path)
        elif block.name == "write":
            file_ops.written.add(path)
        elif block.name == "edit":
            file_ops.edited.add(path)


def compute_file_lists(file_ops: FileOperations) -> tuple[list[str], list[str]]:
    """Compute sorted read-only and modified file lists from accumulated operations.

    Returns `(read_files, modified_files)`.
    """
    modified = file_ops.edited | file_ops.written
    read_only = sorted(f for f in file_ops.read if f not in modified)
    modified_files = sorted(modified)
    return read_only, modified_files


def format_file_operations(read_files: list[str], modified_files: list[str]) -> str:
    """Format file lists as summary metadata tags."""
    sections: list[str] = []
    if read_files:
        sections.append("<read-files>\n" + "\n".join(read_files) + "\n</read-files>")
    if modified_files:
        sections.append("<modified-files>\n" + "\n".join(modified_files) + "\n</modified-files>")
    if not sections:
        return ""
    return "\n\n" + "\n\n".join(sections)


def _safe_json_stringify(value: object) -> str:
    try:
        return json.dumps(value)
    except (TypeError, ValueError):
        return "[unserializable]"


def _truncate_for_summary(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    truncated_chars = len(text) - max_chars
    return f"{text[:max_chars]}\n\n[... {truncated_chars} more characters truncated]"


def serialize_conversation(messages: list[Message]) -> str:
    """Serialize LLM messages to plain text for summarization prompts."""
    parts: list[str] = []

    for msg in messages:
        if msg.role == "user":
            content = content_text(msg.content, "")
            if content:
                parts.append(f"[User]: {content}")
        elif msg.role == "assistant":
            thinking_parts: list[str] = []
            tool_calls: list[str] = []

            for block in msg.content:
                if block.type == "thinking":
                    thinking_parts.append(block.thinking)
                elif block.type == "toolCall":
                    args_str = ", ".join(f"{k}={_safe_json_stringify(v)}" for k, v in block.arguments.items())
                    tool_calls.append(f"{block.name}({args_str})")

            if thinking_parts:
                parts.append(f"[Assistant thinking]: {chr(10).join(thinking_parts)}")
            if any(block.type == "text" for block in msg.content):
                parts.append(f"[Assistant]: {content_text(msg.content)}")
            if tool_calls:
                parts.append(f"[Assistant tool calls]: {'; '.join(tool_calls)}")
        elif msg.role == "toolResult":
            content = content_text(msg.content, "")
            if content:
                parts.append(f"[Tool result]: {_truncate_for_summary(content, TOOL_RESULT_MAX_CHARS)}")

    return "\n\n".join(parts)


__all__ = [
    "TOOL_RESULT_MAX_CHARS",
    "FileOperations",
    "compute_file_lists",
    "create_file_ops",
    "extract_file_ops_from_message",
    "format_file_operations",
    "serialize_conversation",
]
