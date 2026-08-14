"""Scanning session search: no index, one pass over every entry.

Python port of `packages/agent/src/search/scanning.ts`.

Every entry of every session is read and matched with a case-insensitive
substring test, so this is only suitable for small stores. It exists because it
needs nothing from the storage backend beyond reading, which makes it the
fallback any backend gets for free.

The scan is lazy at both levels -- sessions are pulled from the source one at a
time, and each session's entries are paged -- so a caller that stops early (a
`limit`, or abandoning an async iterator) never reads the rest of the store.
"""

from __future__ import annotations

from collections.abc import AsyncIterable, Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from pi_ai.utils.abort import AbortSignal

from ..harness.session.jsonl.codec import entry_to_wire
from ..harness.session.types import Entry, EntryCursor, EntryQuery, SessionMetadata

_DEFAULT_PAGE_SIZE = 100


@dataclass(kw_only=True)
class SessionSearchCandidate:
    entry_id: str
    seq: int
    type: str
    timestamp: int
    text: str
    fields: dict[str, Any] | None = None


class ScanningReadable(Protocol):
    """The read-only slice of `SessionStorage` a scan needs."""

    async def get_metadata(self) -> SessionMetadata: ...
    async def find_entries(self, query: EntryQuery | None = None) -> list[Entry]: ...
    async def get_label(self, id: str) -> str | None: ...


ScanningReadableSource = Callable[[Any], AsyncIterable[ScanningReadable]]
"""Yields the sessions to scan, given the backend-specific source options."""

ScanningSearchTextProjector = Callable[[SessionMetadata, Entry, str | None], str]
"""Builds the text a candidate is matched against."""


@dataclass(kw_only=True)
class ScanningReadableOptions:
    project_text: ScanningSearchTextProjector | None = None
    page_size: int | None = None


@dataclass(kw_only=True)
class ScanningSessionSearchHit:
    session_id: str
    entry_id: str
    timestamp: int
    snippet: str


@dataclass(kw_only=True)
class ScanningSessionSearchOptions(ScanningReadableOptions):
    source_options: Callable[[str, Any], Any] | None = None
    match: Callable[[str, SessionSearchCandidate, SessionMetadata], bool] | None = None
    create_hit: Callable[[SessionMetadata, SessionSearchCandidate], Any] | None = None


def _default_search_text(_metadata: SessionMetadata, entry: Entry, label: str | None) -> str:
    # `entry_to_wire` is the Python stand-in for TypeScript's `JSON.stringify(entry)`:
    # the TS entry already *is* the wire shape, while this port keeps dataclasses.
    import json

    payload = json.dumps(entry_to_wire(entry))
    return payload if label is None else f"{payload} {label}"


async def _scan_readable_entries(
    readable: ScanningReadable,
    metadata: SessionMetadata,
    options: ScanningReadableOptions,
    *,
    after_seq: int = 0,
    limit: int | None = None,
    entry_types: Sequence[str] | None = None,
) -> AsyncIterable[SessionSearchCandidate]:
    project_text = options.project_text or _default_search_text
    page_size = limit if limit is not None else (options.page_size or _DEFAULT_PAGE_SIZE)
    wanted_types = None if entry_types is None else set(entry_types)
    while True:
        entries = await readable.find_entries(
            EntryQuery(
                order="oldestFirst",
                limit=page_size,
                cursor=EntryCursor(after_seq=after_seq),
                # Only a single requested type can be pushed down to the query;
                # anything wider is filtered below.
                type=entry_types[0] if entry_types is not None and len(entry_types) == 1 else None,
            )
        )
        if not entries:
            break
        for entry in entries:
            if wanted_types is not None and entry.type not in wanted_types:
                continue
            label = await readable.get_label(entry.id)
            yield SessionSearchCandidate(
                entry_id=entry.id,
                seq=entry.seq,
                type=entry.type,
                timestamp=entry.timestamp,
                text=project_text(metadata, entry, label),
                fields=None if label is None else {"label": label},
            )
        after_seq = entries[-1].seq
        if len(entries) < page_size:
            break


async def scanning_entries(
    readable: ScanningReadable, options: ScanningReadableOptions | None = None
) -> AsyncIterable[SessionSearchCandidate]:
    """Every entry of one session, projected to search candidates."""
    resolved = options if options is not None else ScanningReadableOptions()
    metadata = await readable.get_metadata()
    async for candidate in _scan_readable_entries(readable, metadata, resolved):
        yield candidate


async def _array_source(readables: Sequence[ScanningReadable]) -> AsyncIterable[ScanningReadable]:
    for readable in readables:
        yield readable


def _readables_for(
    source: Sequence[ScanningReadable] | ScanningReadableSource, source_options: Any
) -> AsyncIterable[ScanningReadable]:
    return source(source_options) if callable(source) else _array_source(source)


def _default_match(query_text: str, candidate: SessionSearchCandidate) -> bool:
    return query_text in candidate.text.lower()


def _create_default_scanning_hit(
    metadata: SessionMetadata, candidate: SessionSearchCandidate
) -> ScanningSessionSearchHit:
    return ScanningSessionSearchHit(
        session_id=metadata.id,
        entry_id=candidate.entry_id,
        timestamp=candidate.timestamp,
        snippet=candidate.text,
    )


def create_scanning_session_search(
    source: Sequence[ScanningReadable] | ScanningReadableSource,
    options: ScanningSessionSearchOptions | None = None,
) -> Any:
    """A `SessionSearch` that scans `source`.

    `source` is either a fixed list of sessions or a callable producing them,
    which lets a backend push its own filtering (a `cwd`, a date range) into
    the listing instead of reading every session and discarding most of them.
    """
    resolved = options if options is not None else ScanningSessionSearchOptions()
    create_hit = resolved.create_hit or _create_default_scanning_hit

    class _ScanningSessionSearch:
        async def search(self, text: str, search_options: Any = None) -> AsyncIterable[Any]:
            opts = search_options
            normalized_text = text.strip().lower()
            limit = getattr(opts, "limit", None) if opts is not None else None
            entry_types = getattr(opts, "entry_types", None) if opts is not None else None
            signal: AbortSignal | None = getattr(opts, "signal", None) if opts is not None else None

            if not normalized_text or (limit is not None and limit <= 0):
                return
            # An explicitly empty type list selects nothing, which is not the
            # same as `None` (no filter).
            if entry_types is not None and len(entry_types) == 0:
                return

            hit_count = 0
            seen_session_ids: set[str] = set()
            wanted_types = None if entry_types is None else set(entry_types)
            source_options = resolved.source_options(normalized_text, opts) if resolved.source_options else None

            async for readable in _readables_for(source, source_options):
                if signal is not None:
                    signal.throw_if_aborted()
                metadata = await readable.get_metadata()
                # Scanning the same session twice would double every hit from
                # it, so a source that repeats itself is a bug, not a warning.
                if metadata.id in seen_session_ids:
                    raise ValueError(f"Duplicate sessionId: {metadata.id}")
                seen_session_ids.add(metadata.id)

                async for candidate in _scan_readable_entries(readable, metadata, resolved, entry_types=entry_types):
                    if signal is not None:
                        signal.throw_if_aborted()
                    if wanted_types is not None and candidate.type not in wanted_types:
                        continue
                    matches = (
                        resolved.match(normalized_text, candidate, metadata)
                        if resolved.match
                        else _default_match(normalized_text, candidate)
                    )
                    if not matches:
                        continue
                    yield create_hit(metadata, candidate)
                    hit_count += 1
                    if limit is not None and hit_count >= limit:
                        return

    return _ScanningSessionSearch()


__all__ = [
    "ScanningReadable",
    "ScanningReadableOptions",
    "ScanningReadableSource",
    "ScanningSearchTextProjector",
    "ScanningSessionSearchHit",
    "ScanningSessionSearchOptions",
    "SessionSearchCandidate",
    "create_scanning_session_search",
    "scanning_entries",
]
