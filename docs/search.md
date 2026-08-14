# Session Search

This document describes shipped Python behaviour. The current Python port provides a synchronous-result scanning search over committed session entries; the newer TypeScript design for async-iterable hits, SQLite FTS, and external indexes is not ported.

Pi search is a small query interface over committed session entries. The shared Python contract returns hits with session metadata, stable entry identity, timestamp, and optional display fields.

## Core API

```python
from dataclasses import dataclass
from typing import Protocol

from pi_agent.harness.session import SessionMetadata


@dataclass(kw_only=True)
class SessionSearchOptions:
    text: str
    cwd: str | None = None


@dataclass(kw_only=True)
class SessionSearchHit:
    metadata: SessionMetadata
    entry_id: str
    timestamp: str
    snippet: str | None = None
    score: float | None = None


class SessionSearch(Protocol):
    async def search(self, options: SessionSearchOptions) -> list[SessionSearchHit]: ...
```

`metadata.id` and `entry_id` are the portable hit identity. Python hits also carry an ISO-8601 UTC `timestamp`, the matched JSON `snippet`, and a `score` field for future implementations. The scanning implementation always leaves `score` as `None`.

`SessionSearchOptions.cwd` filters against `metadata.cwd` when the repository metadata has that field. `InMemorySessionRepo` metadata has no `cwd`, so a non-`None` `cwd` filter excludes in-memory sessions.

## Why async iterable

The TypeScript source specifies an `AsyncIterable` API so consumers can render early results and cancel in-flight search. That API is not ported. Python `SessionSearch.search()` returns a complete `list[SessionSearchHit]` and has no `AbortSignal` option.

For search-as-you-type, cancel the task that is awaiting `search()`:

```python
import asyncio

from pi_agent.harness.session import SessionSearchOptions

current_task: asyncio.Task | None = None


def render(hit) -> None:
    print(hit.entry_id)


async def update_results(search, query: str) -> None:
    global current_task
    if current_task is not None:
        current_task.cancel()

    async def run_search() -> None:
        hits = await search.search(SessionSearchOptions(text=query))
        for hit in hits[:10]:
            render(hit)

    current_task = asyncio.create_task(run_search())
    try:
        await current_task
    except asyncio.CancelledError:
        pass
```

## Default implementations

### Scanning search

`ScanningSessionSearch` adapts a repository-like source with `list()` and `open(metadata)` methods. It scans every opened session's committed entries in oldest-first order and matches a case-insensitive substring against each entry's JSONL wire serialization.

```python
from typing import Protocol

from pi_agent.harness.session import Session, SessionMetadata


class ScanningSessionSearchSource(Protocol):
    async def list(self) -> list[SessionMetadata]: ...
    async def open(self, metadata: SessionMetadata) -> Session: ...
```

The scanner searches the same camelCase wire field names that JSONL writes (`parentId`, `customType`, `toolCallId`, `stopReason`, and similar), because Python converts dataclasses through `entry_to_wire()` before matching.

Already-open repositories can be scanned directly:

```python
from pi_agent.harness.session import InMemorySessionRepo, SessionSearchOptions, create_scanning_session_search


async def main() -> None:
    repo = InMemorySessionRepo()
    search = create_scanning_session_search(repo)

    hits = await search.search(SessionSearchOptions(text="authentication"))
    for hit in hits[:10]:
        session = await repo.open(hit.metadata)
        entry = await session.get_entry(hit.entry_id)
        print(entry)
```

JSONL-backed code uses the same scanner over `JsonlSessionRepo`:

```python
from pi_agent.harness.session import JsonlSessionRepo, SessionSearchOptions, create_scanning_session_search
from pi_agent.harness.session.jsonl.types import JsonlSessionRepoOptions


async def main() -> None:
    repo = JsonlSessionRepo(JsonlSessionRepoOptions(sessions_root=".pi-sessions"))
    search = create_scanning_session_search(repo)

    hits = await search.search(SessionSearchOptions(text="authentication", cwd="/workspace/project"))
    for hit in hits:
        print(hit.metadata.id, hit.entry_id, hit.timestamp)
```

A scanning source opens each session through the source's `open()` method. The current Python JSONL repo has no writer lease, so this does not claim a SQLite-style writer lease. If a future backend adds exclusive write claims, provide a read-only source rather than scanning through a harness-owned writer.

### SQLite FTS

Not ported. The Python `pi-agent` package has no SQLite session backend and no `create_sqlite_session_search()` equivalent.

## Indexed backends

Search indexing is backend-owned derived state. The Python shared package only exports the scanning query API; applications may define their own writer/feed contracts when they need explicit index maintenance.

### JSONL sessions with Elasticsearch

Not ported. The TypeScript source includes an application-owned Elasticsearch adapter sketch. Python does not ship Elasticsearch dependencies or an index writer. Use the scanning API above, or build an application-local index from `Session.find_entries()` / `JsonlSessionRepo.list()`.

A minimal local feed loop looks like this:

```python
from pi_agent.harness.session import EntryQuery, JsonlSessionRepo
from pi_agent.harness.session.jsonl.types import JsonlSessionRepoOptions
from pi_agent.harness.session.jsonl import entry_to_wire


async def iter_jsonl_entry_documents(root: str):
    repo = JsonlSessionRepo(JsonlSessionRepoOptions(sessions_root=root))
    for metadata in await repo.list():
        session = await repo.open(metadata)
        entries = await session.find_entries(EntryQuery(order="oldestFirst"))
        for entry in entries:
            yield {
                "session_id": metadata.id,
                "entry_id": entry.id,
                "timestamp": entry.timestamp,
                "text": entry_to_wire(entry),
                "metadata": metadata,
            }
```

## Correctness and failure boundaries

Scanning search has no index and no durable cursor. It reflects committed entries visible through the source at the time each session is opened. It may be slow on large stores.

The scanner returns no hits for an empty or whitespace-only query and does not touch the source in that case.

Search indexes, if an application adds one, are derived state. Applications can retry, rebuild, or mark search stale. Index failures must not affect session commits.

The TypeScript follow-up design for a no-op-by-default search index sink is not ported.
