"""Session search interfaces.

Python port of `packages/agent/src/search/index.ts`.

Upstream moved search out of `harness/session/` into its own top-level module
(`refactor: search`, #7797) and reshaped the API at the same time. Two changes
matter to callers:

* ``search`` takes the query text as an argument and yields hits **lazily**.
  The old shape built the whole result list before returning, so a
  search-as-you-type caller paid for every session on every keystroke.
* ``SessionSearchHit`` carries only the identifiers. A backend that can afford
  richer hits declares its own subtype, the way
  :class:`~pi_agent.search.scanning.ScanningSessionSearchHit` adds
  ``timestamp`` and ``snippet``.
"""

from __future__ import annotations

from collections.abc import AsyncIterable, Sequence
from dataclasses import dataclass
from typing import Protocol

from pi_ai.utils.abort import AbortSignal

from .scanning import (
    ScanningReadable,
    ScanningReadableOptions,
    ScanningReadableSource,
    ScanningSearchTextProjector,
    ScanningSessionSearchHit,
    ScanningSessionSearchOptions,
    SessionSearchCandidate,
    create_scanning_session_search,
    scanning_entries,
)


@dataclass(kw_only=True)
class SessionSearchOptions:
    entry_types: Sequence[str] | None = None
    """Restrict results to specific canonical entry types."""
    limit: int | None = None
    """Maximum number of hits to return."""
    signal: AbortSignal | None = None
    """Abort signal for cancellation, e.g. search-as-you-type."""


@dataclass(kw_only=True)
class SessionSearchHit:
    session_id: str
    """Logical identifier of the session that owns the entry."""
    entry_id: str
    """Logical identifier of the entry within that session."""


class SessionSearch(Protocol):
    def search(self, text: str, options: SessionSearchOptions | None = None) -> AsyncIterable[SessionSearchHit]: ...


__all__ = [
    "ScanningReadable",
    "ScanningReadableOptions",
    "ScanningReadableSource",
    "ScanningSearchTextProjector",
    "ScanningSessionSearchHit",
    "ScanningSessionSearchOptions",
    "SessionSearch",
    "SessionSearchCandidate",
    "SessionSearchHit",
    "SessionSearchOptions",
    "create_scanning_session_search",
    "scanning_entries",
]
