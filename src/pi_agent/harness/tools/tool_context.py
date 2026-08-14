"""Filesystem and shell context required by the built-in execution tools.

Python port of `packages/agent/src/harness/tools/tool-context.ts`.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..types import ExecutionEnv


@dataclass
class ExecutionToolContext:
    """The context object the built-in execution tools receive."""

    env: ExecutionEnv


__all__ = ["ExecutionToolContext"]
