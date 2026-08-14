"""JSONL v4 session storage data types.

Python port of `packages/agent/src/harness/session/jsonl/types.ts`. The
TypeScript version parameterizes storage over an injectable `FileSystem`
abstraction (`packages/agent/src/harness/types.ts`); that abstraction is out
of scope for this port, so `JsonlSessionStorage`/`JsonlSessionRepo` operate
directly on `pathlib.Path` with plain synchronous file IO instead of taking a
`fs` dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ..types import JsonValue, SessionCreateOptions, SessionMetadata


@dataclass(kw_only=True)
class JsonlSessionRepoOptions:
    sessions_root: str | Path
    """Root containing coding-agent-compatible cwd-encoded session directories."""


@dataclass(kw_only=True)
class JsonlSessionMetadata(SessionMetadata):
    cwd: str = ""
    path: str = ""
    modified_at: int = 0
    """Filesystem modification time as milliseconds since Unix epoch."""
    source_format: Literal[3, 4] = 4
    legacy_parent_session_path: str | None = None
    """Present only when a v3 parent path could not be resolved to a session id."""
    metadata: dict[str, JsonValue] | None = None
    """Opaque application-owned metadata."""


@dataclass(kw_only=True)
class JsonlSessionCreateOptions(SessionCreateOptions):
    cwd: str = ""
    metadata: dict[str, JsonValue] | None = None


@dataclass(kw_only=True)
class JsonlSessionListOptions:
    cwd: str | None = None


@dataclass(kw_only=True)
class JsonlForkOptions:
    """Fork destination and scope for `JsonlSessionRepo.fork`.

    Combines `ForkOptions` (scope/target) with `JsonlSessionCreateOptions`
    (destination `id`/`cwd`/`parent_session_id`/`metadata`) into one dataclass;
    see `ForkOptions` for why Python flattens the TypeScript intersection type.
    """

    scope: Literal["branch", "tree"] = "branch"
    entry_id: str | None = None
    position: Literal["before", "at"] | None = None
    id: str | None = None
    parent_session_id: str | None = None
    cwd: str = ""
    metadata: dict[str, JsonValue] | None = None


@dataclass(kw_only=True)
class JsonlV4Header:
    id: str
    created_at: int
    cwd: str
    parent_session_id: str | None = None
    legacy_parent_session_path: str | None = None
    """Preserved only when a v3 parent path could not be resolved to a session id."""
    metadata: dict[str, JsonValue] | None = None
    kind: Literal["header"] = "header"
    version: Literal[4] = 4


__all__ = [
    "JsonlSessionCreateOptions",
    "JsonlSessionListOptions",
    "JsonlSessionMetadata",
    "JsonlSessionRepoOptions",
    "JsonlV4Header",
]
