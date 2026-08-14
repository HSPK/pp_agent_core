"""Path normalization shared by the harness tools.

Python port of `packages/agent/src/harness/tools/path-utils.ts`.
"""

from __future__ import annotations

import re
import unicodedata

from pi_ai.utils.abort import AbortSignal

from ..types import ExecutionEnv, get_or_throw

_UNICODE_SPACES = re.compile("[\u00a0\u2000-\u200a\u202f\u205f\u3000]")
NARROW_NO_BREAK_SPACE = "\u202f"
_AM_PM = re.compile(r" (AM|PM)\.", re.IGNORECASE)


def normalize_tool_path(path: str) -> str:
    normalized = _UNICODE_SPACES.sub(" ", path)
    return normalized[1:] if normalized.startswith("@") else normalized


async def resolve_tool_path(env: ExecutionEnv, path: str, signal: AbortSignal | None = None) -> str:
    return get_or_throw(await env.absolute_path(normalize_tool_path(path), signal))


async def resolve_read_tool_path(env: ExecutionEnv, path: str, signal: AbortSignal | None = None) -> str:
    """Probe common macOS/Unicode spellings of `path` before giving up on the literal one."""
    resolved = await resolve_tool_path(env, path, signal)
    variants = [
        resolved,
        _AM_PM.sub(NARROW_NO_BREAK_SPACE + r"\1.", resolved),
        unicodedata.normalize("NFD", resolved),
        resolved.replace("'", "\u2019"),
        unicodedata.normalize("NFD", resolved).replace("'", "\u2019"),
    ]

    seen: set[str] = set()
    for variant in variants:
        if variant in seen:
            continue
        seen.add(variant)
        if get_or_throw(await env.exists(variant, signal)):
            return variant
    return resolved


__all__ = ["normalize_tool_path", "resolve_read_tool_path", "resolve_tool_path"]
