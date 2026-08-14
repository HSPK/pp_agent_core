"""Python port of `packages/agent/test/harness/session/search.test.ts`.

The four upstream cases come first; the rest pin behaviour the TypeScript
suite leaves to its source (empty/whitespace queries, `limit`, paging, a
repeated session id), which this port had covered before upstream grew a
dedicated suite.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from pi_ai.utils.abort import AbortController, AbortError
from session_conformance_helpers import create_user_message

from pi_agent.harness.session import InMemorySessionRepo, JsonlSessionCreateOptions, JsonlSessionRepo
from pi_agent.harness.session.jsonl.repo import list_jsonl_session_metadata, load_jsonl_session_storage
from pi_agent.harness.session.jsonl.types import JsonlSessionRepoOptions
from pi_agent.harness.session.types import SessionCreateOptions
from pi_agent.search import (
    ScanningReadableOptions,
    ScanningSessionSearchOptions,
    SessionSearchOptions,
    create_scanning_session_search,
    scanning_entries,
)


async def _collect(iterable: Any) -> list[Any]:
    items: list[Any] = []
    async for item in iterable:
        items.append(item)
    return items


@pytest.fixture
def repo() -> InMemorySessionRepo:
    return InMemorySessionRepo()


async def _session(repo: InMemorySessionRepo, session_id: str, texts: list[str]):
    # The new API has no `cwd` filter, so the in-memory cases need no cwd; the
    # JSONL case below still sets one, because its repo lists by cwd.
    session = await repo.create(SessionCreateOptions(id=session_id))
    entry_ids = [await session.append_message(create_user_message(text)) for text in texts]
    return session, entry_ids


# --------------------------------------------------------------------------
# Ported from search.test.ts
# --------------------------------------------------------------------------


async def test_scans_an_arbitrary_in_memory_projected_source(repo: InMemorySessionRepo):
    root, _ = await _session(repo, "root", ["fix auth flow"])
    other, _ = await _session(repo, "other", ["auth in another workspace"])
    search = create_scanning_session_search([root, other])

    hits = await _collect(search.search("auth"))
    assert [hit.session_id for hit in hits] == ["root", "other"]
    assert await _collect(search.search("missing")) == []


async def test_includes_labels_in_memory_scanning_projections(repo: InMemorySessionRepo):
    session, entry_ids = await _session(repo, "session", ["plain body"])
    await session.set_label(entry_ids[0], "important label")
    search = create_scanning_session_search([session])

    hits = await _collect(search.search("important"))
    assert [(hit.session_id, hit.entry_id) for hit in hits] == [("session", entry_ids[0])]


async def test_honors_entry_type_filters_and_abort_signals(repo: InMemorySessionRepo):
    session, entry_ids = await _session(repo, "session", ["auth message"])
    await session.append_custom_entry("note", {"text": "auth custom"})
    search = create_scanning_session_search([session])

    hits = await _collect(search.search("auth", SessionSearchOptions(entry_types=["message"])))
    assert [(hit.session_id, hit.entry_id) for hit in hits] == [("session", entry_ids[0])]

    controller = AbortController()
    controller.abort()
    with pytest.raises(AbortError):
        await _collect(search.search("auth", SessionSearchOptions(signal=controller.signal)))


async def test_scans_jsonl_sessions_from_disk(tmp_path):
    options = JsonlSessionRepoOptions(sessions_root=str(tmp_path))
    repository = JsonlSessionRepo(options)
    cwd = str(tmp_path / "workspace")
    other_cwd = str(tmp_path / "other")

    session = await repository.create(JsonlSessionCreateOptions(id="jsonl", cwd=cwd))
    entry_id = await session.append_message(create_user_message("jsonl backed auth entry"))
    await session.set_label(entry_id, "disk label")
    other = await repository.create(JsonlSessionCreateOptions(id="other", cwd=other_cwd))
    other_entry_id = await other.append_message(create_user_message("jsonl backed auth entry in another cwd"))

    async def jsonl_readables(query: Any = None):
        for metadata in await list_jsonl_session_metadata(options, query):
            yield await load_jsonl_session_storage(options, metadata)

    search = create_scanning_session_search(jsonl_readables)

    auth_hits = await _collect(search.search("auth"))
    assert len(auth_hits) == 2
    assert {(hit.session_id, hit.entry_id) for hit in auth_hits} == {
        ("jsonl", entry_id),
        ("other", other_entry_id),
    }

    disk_hits = await _collect(search.search("disk"))
    assert [(hit.session_id, hit.entry_id) for hit in disk_hits] == [("jsonl", entry_id)]


# --------------------------------------------------------------------------
# Behaviour the TypeScript suite leaves to its source
# --------------------------------------------------------------------------


class _ExplodingReadable:
    """Fails if touched, proving a query short-circuits before any read."""

    async def get_metadata(self):
        raise AssertionError("get_metadata() must not be called")

    async def find_entries(self, query=None):
        raise AssertionError("find_entries() must not be called")

    async def get_label(self, id: str):
        raise AssertionError("get_label() must not be called")


@pytest.mark.parametrize(
    "text,options",
    [
        ("", None),
        ("   ", None),
        ("auth", SessionSearchOptions(limit=0)),
        ("auth", SessionSearchOptions(entry_types=[])),
    ],
)
async def test_a_query_that_can_match_nothing_reads_nothing(text: str, options):
    search = create_scanning_session_search([_ExplodingReadable()])

    assert await _collect(search.search(text, options)) == []


async def test_limit_stops_the_scan(repo: InMemorySessionRepo):
    session, entry_ids = await _session(repo, "s", ["auth one", "auth two", "auth three"])
    search = create_scanning_session_search([session])

    hits = await _collect(search.search("auth", SessionSearchOptions(limit=2)))
    assert [hit.entry_id for hit in hits] == entry_ids[:2]


async def test_the_default_hit_carries_the_timestamp_and_snippet(repo: InMemorySessionRepo):
    session, entry_ids = await _session(repo, "s", ["auth body"])
    search = create_scanning_session_search([session])

    hit = (await _collect(search.search("auth")))[0]
    assert hit.entry_id == entry_ids[0]
    assert isinstance(hit.timestamp, int)
    # The snippet is the wire JSON, which is what the match ran against.
    assert "auth body" in hit.snippet
    assert json.loads(hit.snippet)["type"] == "message"


async def test_a_repeated_session_id_is_an_error(repo: InMemorySessionRepo):
    """Scanning one session twice would double every hit it produces."""
    session, _ = await _session(repo, "dup", ["auth"])
    search = create_scanning_session_search([session, session])

    with pytest.raises(ValueError, match="Duplicate sessionId: dup"):
        await _collect(search.search("auth"))


async def test_paging_reads_every_entry(repo: InMemorySessionRepo):
    """`page_size` is an internal detail: the result must not depend on it."""
    session, entry_ids = await _session(repo, "s", [f"auth {index}" for index in range(7)])
    search = create_scanning_session_search([session], ScanningSessionSearchOptions(page_size=2))

    hits = await _collect(search.search("auth"))
    assert [hit.entry_id for hit in hits] == entry_ids


async def test_a_custom_projector_replaces_the_matched_text(repo: InMemorySessionRepo):
    session, entry_ids = await _session(repo, "s", ["body text"])
    search = create_scanning_session_search(
        [session],
        ScanningSessionSearchOptions(project_text=lambda metadata, entry, label: f"projected {metadata.id}"),
    )

    # The entry body is no longer searchable; the projection is.
    assert await _collect(search.search("body")) == []
    hits = await _collect(search.search("projected"))
    assert [hit.entry_id for hit in hits] == entry_ids


async def test_scanning_entries_yields_candidates_for_one_session(repo: InMemorySessionRepo):
    session, entry_ids = await _session(repo, "s", ["first", "second"])

    candidates = await _collect(scanning_entries(session, ScanningReadableOptions()))
    assert [candidate.entry_id for candidate in candidates] == entry_ids
    assert all(candidate.type == "message" for candidate in candidates)
