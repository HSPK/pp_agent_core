"""Harness message shapes and LLM conversion.

Python port of `packages/agent/src/harness/messages.ts`.

TypeScript widens `AgentMessage` (declared in `../types.ts`) via declaration
merging, adding `bashExecution`/`custom`/`branchSummary`/`compactionSummary`
roles as `CustomAgentMessages` members. Python has no equivalent mechanism:
`pi_agent.types.AgentMessage` stays exactly `pi_ai.types.Message`
(user/assistant/toolResult). `HarnessMessage` here is the explicit union that
plays the role TypeScript's widened `AgentMessage` plays for the functions in
this module: `AgentMessage` plus the four additional message dataclasses.
`BranchSummaryMessage`/`CompactionSummaryMessage` and their factories were
already ported to `pi_agent.harness.session.context` for the session
storage layer (see that module's docstring) and are re-exported here rather
than redefined, per this port's convention of reusing the session model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pi_ai.types import ImageContent, Message, TextContent, UserMessage

from ..types import AgentMessage
from .session.context import (
    BRANCH_SUMMARY_PREFIX,
    BRANCH_SUMMARY_SUFFIX,
    COMPACTION_SUMMARY_PREFIX,
    COMPACTION_SUMMARY_SUFFIX,
    BranchSummaryMessage,
    CompactionSummaryMessage,
    create_branch_summary_message,
    create_compaction_summary_message,
)


@dataclass(kw_only=True)
class BashExecutionMessage:
    command: str
    output: str
    exit_code: int | None
    cancelled: bool
    truncated: bool
    timestamp: int
    full_output_path: str | None = None
    exclude_from_context: bool = False
    role: Literal["bashExecution"] = "bashExecution"


@dataclass(kw_only=True)
class CustomMessage:
    custom_type: str
    content: str | list[TextContent | ImageContent]
    display: bool
    timestamp: int
    details: object = None
    role: Literal["custom"] = "custom"


HarnessMessage = AgentMessage | BashExecutionMessage | CustomMessage | BranchSummaryMessage | CompactionSummaryMessage
"""Union played by TypeScript's declaration-merged `AgentMessage` for this module. See module docstring."""


def bash_execution_to_text(msg: BashExecutionMessage) -> str:
    text = f"Ran `{msg.command}`\n"
    if msg.output:
        text += f"```\n{msg.output}\n```"
    else:
        text += "(no output)"
    if msg.cancelled:
        text += "\n\n(command cancelled)"
    elif msg.exit_code is not None and msg.exit_code != 0:
        text += f"\n\nCommand exited with code {msg.exit_code}"
    if msg.truncated and msg.full_output_path:
        text += f"\n\n[Output truncated. Full output: {msg.full_output_path}]"
    return text


def create_custom_message(
    custom_type: str,
    content: str | list[TextContent | ImageContent],
    display: bool,
    details: object,
    timestamp: int,
) -> CustomMessage:
    return CustomMessage(
        custom_type=custom_type, content=content, display=display, details=details, timestamp=timestamp
    )


def convert_to_llm(messages: list[HarnessMessage]) -> list[Message]:
    result: list[Message] = []
    for message in messages:
        if message.role == "bashExecution":
            if message.exclude_from_context:
                continue
            result.append(
                UserMessage(content=[TextContent(text=bash_execution_to_text(message))], timestamp=message.timestamp)
            )
        elif message.role == "custom":
            content = [TextContent(text=message.content)] if isinstance(message.content, str) else message.content
            result.append(UserMessage(content=content, timestamp=message.timestamp))
        elif message.role == "branchSummary":
            text = BRANCH_SUMMARY_PREFIX + message.summary + BRANCH_SUMMARY_SUFFIX
            result.append(UserMessage(content=[TextContent(text=text)], timestamp=message.timestamp))
        elif message.role == "compactionSummary":
            text = COMPACTION_SUMMARY_PREFIX + message.summary + COMPACTION_SUMMARY_SUFFIX
            result.append(UserMessage(content=[TextContent(text=text)], timestamp=message.timestamp))
        elif message.role in ("user", "assistant", "toolResult"):
            result.append(message)
    return result


__all__ = [
    "BRANCH_SUMMARY_PREFIX",
    "BRANCH_SUMMARY_SUFFIX",
    "COMPACTION_SUMMARY_PREFIX",
    "COMPACTION_SUMMARY_SUFFIX",
    "BashExecutionMessage",
    "BranchSummaryMessage",
    "CompactionSummaryMessage",
    "CustomMessage",
    "HarnessMessage",
    "bash_execution_to_text",
    "convert_to_llm",
    "create_branch_summary_message",
    "create_compaction_summary_message",
    "create_custom_message",
]
