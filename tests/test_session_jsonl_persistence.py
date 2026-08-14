"""Python port of the "JSONL v4 persistence" suite in
`packages/agent/test/harness/session/jsonl.test.ts`.

The `JsonlSessionRepo conformance` describe in that file is ported separately:
`test_session_conformance.py` parametrizes the whole conformance suite over the
in-memory and JSONL backends, exactly as `memory.test.ts` and `jsonl.test.ts`
each do in TypeScript.

TypeScript injects a `NodeExecutionEnv` into `JsonlSessionRepo` and stubs its
`writeFile`/`appendFile`/`renameFile` with `vi.spyOn` to simulate filesystem
failures. This port has no such abstraction (see `jsonl/types.py`), so the
equivalent failures are injected by monkeypatching the concrete IO call the
port makes at that point (`Path.write_text`, `os.replace`,
`JsonlSessionStorage._append_mutation`).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pi_ai import TextContent, UserMessage
from session_conformance_helpers import operation_started

from pi_agent.harness.session import (
    CustomEntry,
    EntryQuery,
    JsonlForkOptions,
    JsonlSessionCreateOptions,
    JsonlSessionListOptions,
    JsonlSessionMetadata,
    JsonlSessionRepo,
    JsonlSessionRepoOptions,
    OperationFinishedRecord,
    RecordQuery,
    SessionError,
)
from pi_agent.harness.session.jsonl import storage as storage_module


def create_repository(root: Path) -> JsonlSessionRepo:
    return JsonlSessionRepo(JsonlSessionRepoOptions(sessions_root=root))


def expected_session_path(root: Path, cwd: str, created_at: int, id: str) -> str:
    directory = "--" + cwd.lstrip("/\\").replace("/", "-").replace("\\", "-").replace(":", "-") + "--"
    timestamp = (
        datetime.fromtimestamp(created_at / 1000, tz=UTC)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
        .replace(":", "-")
        .replace(".", "-")
    )
    return str(root / directory / f"{timestamp}_{id}.jsonl")


def write_raw_session(root: Path, id: str, mutations: list[dict[str, Any]]) -> JsonlSessionMetadata:
    path = root / f"{id}.jsonl"
    created_at = 1
    header = {"kind": "header", "version": 4, "id": id, "createdAt": created_at, "cwd": str(root)}
    path.write_text("\n".join(json.dumps(line) for line in [header, *mutations]) + "\n")
    return JsonlSessionMetadata(
        id=id,
        created_at=created_at,
        cwd=str(root),
        path=str(path),
        modified_at=path.stat().st_mtime_ns // 1_000_000,
        source_format=4,
    )


def user_message(text: str, timestamp: int) -> UserMessage:
    return UserMessage(content=[TextContent(text=text)], timestamp=timestamp)


def read_lines(path: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().rstrip("\n").split("\n")]


async def test_exposes_the_complete_metadata_contract(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    repository = create_repository(root)
    cwd = str(root / "workspace" / "project")
    session = await repository.create(
        JsonlSessionCreateOptions(
            id="metadata",
            cwd=cwd,
            parent_session_id="parent",
            metadata={"owner": "agent", "nested": {"enabled": True}},
        )
    )
    metadata = await session.get_metadata()

    # TypeScript uses a whole-object `toEqual` with `expect.any(Number)` for
    # `createdAt`; asserting full dataclass equality keeps the "and no other
    # field is set" half of that claim (notably legacy_parent_session_path).
    assert isinstance(metadata.created_at, int)
    assert metadata == JsonlSessionMetadata(
        id="metadata",
        created_at=metadata.created_at,
        parent_session_id="parent",
        path=expected_session_path(root, metadata.cwd, metadata.created_at, metadata.id),
        cwd=cwd,
        modified_at=Path(metadata.path).stat().st_mtime_ns // 1_000_000,
        source_format=4,
        metadata={"owner": "agent", "nested": {"enabled": True}},
    )
    assert await repository.list(_list(cwd)) == [metadata]
    assert await repository.list(_list(str(root / "other" / "project"))) == []


def _list(cwd: str | None = None) -> JsonlSessionListOptions:
    return JsonlSessionListOptions(cwd=cwd)


async def test_rejects_a_malformed_json_header_on_open_and_skips_it_when_listing(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    repository = create_repository(root)
    await repository.create(JsonlSessionCreateOptions(id="valid", cwd=str(root)))
    session = await repository.create(JsonlSessionCreateOptions(id="malformed-header", cwd=str(root)))
    metadata = await session.get_metadata()
    malformed = "not json\n"
    Path(metadata.path).write_text(malformed)

    with pytest.raises(SessionError) as exc_info:
        await repository.open(metadata)
    assert exc_info.value.code == "invalid_entry"
    listed = await repository.list(_list(str(root)))
    assert [entry.id for entry in listed] == ["valid"]
    assert Path(metadata.path).read_text() == malformed


async def test_rejects_non_object_header_metadata_on_open_and_skips_it_when_listing(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    repository = create_repository(root)
    await repository.create(JsonlSessionCreateOptions(id="valid", cwd=str(root)))
    session = await repository.create(JsonlSessionCreateOptions(id="invalid-header-metadata", cwd=str(root)))
    metadata = await session.get_metadata()
    malformed = (
        json.dumps(
            {
                "kind": "header",
                "version": 4,
                "id": metadata.id,
                "createdAt": metadata.created_at,
                "cwd": metadata.cwd,
                "metadata": "invalid",
            }
        )
        + "\n"
    )
    Path(metadata.path).write_text(malformed)

    with pytest.raises(SessionError) as exc_info:
        await repository.open(metadata)
    assert exc_info.value.code == "invalid_entry"
    listed = await repository.list(_list(str(root)))
    assert [entry.id for entry in listed] == ["valid"]
    assert Path(metadata.path).read_text() == malformed


async def test_rejects_session_ids_that_cannot_be_used_in_coding_agent_filenames(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    repository = create_repository(root)

    with pytest.raises(SessionError) as exc_info:
        await repository.create(JsonlSessionCreateOptions(id="../escape", cwd=str(root)))
    assert exc_info.value.code == "invalid_payload"


async def test_allows_the_same_explicit_session_id_in_different_working_directories(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    repository = create_repository(root)
    first_cwd = str(root / "workspaces" / "first")
    second_cwd = str(root / "workspaces" / "second")

    first = await repository.create(JsonlSessionCreateOptions(id="shared", cwd=first_cwd))
    second = await repository.create(JsonlSessionCreateOptions(id="shared", cwd=second_cwd))

    assert (await first.get_metadata()).cwd == first_cwd
    assert (await second.get_metadata()).cwd == second_cwd
    assert [entry.id for entry in await repository.list()] == ["shared", "shared"]


@pytest.mark.parametrize(
    ("first_kind", "second_kind"),
    [("create", "create"), ("create", "fork"), ("fork", "fork")],
)
async def test_rejects_concurrent_calls_for_the_same_destination(
    tmp_path: Path, first_kind: str, second_kind: str
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    repository = create_repository(root)
    cwd = str(root / "workspace")
    source = await repository.create(JsonlSessionCreateOptions(id="source", cwd=cwd))
    source_metadata = await source.get_metadata()

    def run(kind: str) -> Any:
        if kind == "create":
            return repository.create(JsonlSessionCreateOptions(id="same", cwd=cwd))
        return repository.fork(source_metadata, JsonlForkOptions(id="same", cwd=cwd))

    results = await asyncio.gather(run(first_kind), run(second_kind), return_exceptions=True)
    successes = [result for result in results if not isinstance(result, BaseException)]
    failures = [result for result in results if isinstance(result, BaseException)]

    assert len(successes) == 1
    assert len(failures) == 1
    assert isinstance(failures[0], SessionError)
    assert failures[0].code == "already_exists"
    listed = await repository.list(_list(cwd))
    assert len([entry for entry in listed if entry.id == "same"]) == 1


@pytest.mark.parametrize("kind", ["create", "fork"])
async def test_releases_a_destination_reservation_after_a_failed_operation(
    tmp_path: Path, kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    repository = create_repository(root)
    cwd = str(root / "workspace")
    source = await repository.create(JsonlSessionCreateOptions(id="source", cwd=cwd))
    source_metadata = await source.get_metadata()

    def run() -> Any:
        if kind == "create":
            return repository.create(JsonlSessionCreateOptions(id="retry", cwd=cwd))
        return repository.fork(source_metadata, JsonlForkOptions(id="retry", cwd=cwd))

    remaining = [1]
    if kind == "create":
        original_write_text = Path.write_text

        def failing_write_text(self: Path, *args: Any, **kwargs: Any) -> int:
            if remaining and remaining.pop():
                raise OSError("injected creation failure")
            return original_write_text(self, *args, **kwargs)

        monkeypatch.setattr(Path, "write_text", failing_write_text)
    else:
        original_replace = os.replace

        def failing_replace(src: Any, dst: Any, **kwargs: Any) -> None:
            if remaining and remaining.pop():
                raise OSError("injected fork failure")
            original_replace(src, dst, **kwargs)

        monkeypatch.setattr(storage_module.os, "replace", failing_replace)

    with pytest.raises(SessionError) as exc_info:
        await run()
    assert exc_info.value.code == "storage"
    assert await run() is not None
    listed = await repository.list(_list(cwd))
    assert len([entry for entry in listed if entry.id == "retry"]) == 1


async def test_sorts_listed_sessions_by_current_filesystem_modification_time(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    repository = create_repository(root)
    newest_cwd = str(root / "workspaces" / "newest")
    oldest_cwd = str(root / "workspaces" / "oldest")
    newest = await repository.create(JsonlSessionCreateOptions(id="newest", cwd=newest_cwd))
    newest_metadata = await newest.get_metadata()
    oldest = await repository.create(JsonlSessionCreateOptions(id="oldest", cwd=oldest_cwd))
    oldest_metadata = await oldest.get_metadata()
    os.utime(newest_metadata.path, (1_700_000_002, 1_700_000_002))
    os.utime(oldest_metadata.path, (1_700_000_001, 1_700_000_001))

    listed = await repository.list()

    assert [entry.id for entry in listed] == ["newest", "oldest"]
    assert [entry.id for entry in await repository.list(_list(newest_cwd))] == ["newest"]
    assert [entry.modified_at for entry in listed] == [
        int(Path(newest_metadata.path).stat().st_mtime * 1000),
        int(Path(oldest_metadata.path).stat().st_mtime * 1000),
    ]


async def test_writes_one_line_per_mutation_and_restores_the_shared_sequence(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    repository = create_repository(root)
    session = await repository.create(JsonlSessionCreateOptions(id="session", cwd=str(root)))
    metadata = await session.get_metadata()
    entry_id = await session.append_custom_entry("note", {"value": 1})
    await session.create_lane("thread", entry_id)
    await session.append_record(operation_started("run", lane="thread", kind="run"))
    await session.set_name("Example")
    await session.set_label(entry_id, "checkpoint")
    await session.move_lane("main", None)

    lines = read_lines(metadata.path)
    assert [line["kind"] for line in lines] == ["header", "entry", "lane", "record", "fact", "fact", "lane"]
    assert [line["seq"] for line in lines[1:]] == [1, 2, 3, 4, 5, 6]

    reopened = await create_repository(root).open(metadata)
    lanes = await reopened.get_lanes()
    assert [(pointer.lane, pointer.leaf_id) for pointer in lanes] == [("main", None), ("thread", entry_id)]
    assert await reopened.get_name() == "Example"
    assert await reopened.get_label(entry_id) == "checkpoint"
    assert [record.id for record in await reopened.find_records()] == ["run"]
    filtered = await reopened.find_records(RecordQuery(type="operation_started", operation_kind="run"))
    assert [record.id for record in filtered] == ["run"]
    assert [record.id for record in await reopened.find_open_operations("thread", 2)] == ["run"]
    assert [item.seq for item in await reopened.get_log()] == [1, 2, 3, 4, 5, 6]
    finished = await reopened.append_record(
        OperationFinishedRecord(id="finish", lane="thread", run_id="run", outcome="completed")
    )
    assert finished.seq == 7
    assert await reopened.find_open_operations("thread", 2) == []


async def test_recomputes_fork_message_counts_when_reopening(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    repository = create_repository(root)
    source = await repository.create(JsonlSessionCreateOptions(id="source", cwd=str(root)))
    await source.append_message(user_message("one", 1))
    await source.append_message(user_message("two", 2))
    fork = await repository.fork(await source.get_metadata(), JsonlForkOptions(id="fork", cwd=str(root)))
    metadata = await fork.get_metadata()

    reopened = await create_repository(root).open(metadata)
    assert (await reopened.get_stats()).message_count == 2
    await reopened.append_message(user_message("three", 3))
    assert (await reopened.get_stats()).message_count == 3

    verified = await create_repository(root).open(metadata)
    assert (await verified.get_stats()).message_count == 3


async def test_reopens_a_tree_fork_with_its_lanes_and_facts(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    repository = create_repository(root)
    source = await repository.create(JsonlSessionCreateOptions(id="source", cwd=str(root)))
    root_id = await source.append_custom_entry("root")
    await source.create_lane("thread", root_id)
    main_id = await source.append_custom_entry("main")
    thread_entry = await source.append_entry(
        CustomEntry(id="thread", custom_type="thread"),
        "thread",
    )
    thread_id = thread_entry.id
    await source.set_name("Source")
    await source.set_label(thread_id, "tip")
    fork = await repository.fork(await source.get_metadata(), JsonlForkOptions(scope="tree", id="fork", cwd=str(root)))
    metadata = await fork.get_metadata()

    imported_entry_lines = [line for line in read_lines(metadata.path) if line["kind"] == "entry"]
    assert [("lane" in line) for line in imported_entry_lines] == [False, False, False]

    reopened = await create_repository(root).open(metadata)
    entries = await reopened.find_entries(EntryQuery(order="oldestFirst"))
    assert [entry.id for entry in entries] == [root_id, main_id, thread_id]
    lanes = await reopened.get_lanes()
    assert [(pointer.lane, pointer.leaf_id) for pointer in lanes] == [("main", main_id), ("thread", thread_id)]
    assert await reopened.get_name() == "Source"
    assert await reopened.get_label(thread_id) == "tip"
    assert await reopened.find_records() == []


async def test_does_not_publish_a_partial_fork_when_staging_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    repository = create_repository(root)
    source = await repository.create(JsonlSessionCreateOptions(id="source", cwd=str(root)))
    await source.append_message(user_message("one", 1))
    await source.append_message(user_message("two", 2))
    source_metadata = await source.get_metadata()

    original_append = storage_module.JsonlSessionStorage._append_mutation
    calls = [0]

    async def failing_append(self: Any, mutation: Any) -> None:
        calls[0] += 1
        if calls[0] == 2:
            raise SessionError("storage", "injected staging failure")
        await original_append(self, mutation)

    monkeypatch.setattr(storage_module.JsonlSessionStorage, "_append_mutation", failing_append)

    with pytest.raises(SessionError) as exc_info:
        await repository.fork(source_metadata, JsonlForkOptions(id="fork", cwd=str(root)))
    assert exc_info.value.code == "storage"

    monkeypatch.undo()
    assert [entry.id for entry in await repository.list()] == ["source"]
    directory = Path(source_metadata.path).parent
    assert [name.name for name in directory.iterdir() if name.name.endswith(".tmp")] == []


async def test_does_not_publish_a_fork_when_atomic_rename_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    repository = create_repository(root)
    source = await repository.create(JsonlSessionCreateOptions(id="source", cwd=str(root)))
    await source.append_message(user_message("one", 1))
    source_metadata = await source.get_metadata()

    original_replace = os.replace
    remaining = [1]

    def failing_replace(src: Any, dst: Any, **kwargs: Any) -> None:
        if remaining and remaining.pop():
            raise OSError("injected rename failure")
        original_replace(src, dst, **kwargs)

    monkeypatch.setattr(storage_module.os, "replace", failing_replace)

    with pytest.raises(SessionError) as exc_info:
        await repository.fork(source_metadata, JsonlForkOptions(id="fork", cwd=str(root)))
    assert exc_info.value.code == "storage"

    monkeypatch.undo()
    assert [entry.id for entry in await repository.list()] == ["source"]
    directory = Path(source_metadata.path).parent
    assert [name.name for name in directory.iterdir() if name.name.endswith(".tmp")] == []


async def test_repairs_a_valid_final_line_missing_its_newline(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    repository = create_repository(root)
    session = await repository.create(JsonlSessionCreateOptions(id="session", cwd=str(root)))
    metadata = await session.get_metadata()
    first_id = await session.append_custom_entry("first")
    unterminated = Path(metadata.path).read_text().rstrip("\n")
    Path(metadata.path).write_text(unterminated)

    reopened = await create_repository(root).open(metadata)
    assert Path(metadata.path).read_text() == f"{unterminated}\n"
    second_id = await reopened.append_custom_entry("second")

    verified = await create_repository(root).open(metadata)
    entries = await verified.find_entries(EntryQuery(order="oldestFirst"))
    assert [entry.id for entry in entries] == [first_id, second_id]


async def test_fails_to_open_when_repairing_a_missing_final_newline_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    repository = create_repository(root)
    session = await repository.create(JsonlSessionCreateOptions(id="session", cwd=str(root)))
    metadata = await session.get_metadata()
    await session.append_custom_entry("first")
    Path(metadata.path).write_text(Path(metadata.path).read_text().rstrip("\n"))

    original_open = Path.open

    def failing_open(self: Path, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        if "a" in mode and str(self) == metadata.path:
            raise PermissionError("repair denied")
        return original_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", failing_open)

    with pytest.raises(SessionError) as exc_info:
        await create_repository(root).open(metadata)
    assert exc_info.value.code == "storage"
    assert isinstance(exc_info.value.__cause__, PermissionError)


async def test_truncates_a_malformed_final_line(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    repository = create_repository(root)
    session = await repository.create(JsonlSessionCreateOptions(id="session", cwd=str(root)))
    metadata = await session.get_metadata()
    await session.append_custom_entry("note", {"value": "kept"})
    valid_prefix = Path(metadata.path).read_text()
    with Path(metadata.path).open("a") as handle:
        handle.write('{"kind":"entry"')

    reopened = await create_repository(root).open(metadata)
    assert len(await reopened.find_entries()) == 1
    assert Path(metadata.path).read_text() == valid_prefix
    appended_id = await reopened.append_custom_entry("after-recovery")
    entry = await reopened.get_entry(appended_id)
    assert entry is not None
    assert entry.seq == 2


async def test_rejects_a_complete_invalid_final_mutation_without_modifying_the_file(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    metadata = write_raw_session(root, "invalid-final-mutation", [{"kind": "unknown", "seq": 1}])
    corrupted = Path(metadata.path).read_text()

    with pytest.raises(SessionError) as exc_info:
        await create_repository(root).open(metadata)
    assert exc_info.value.code == "invalid_entry"
    assert Path(metadata.path).read_text() == corrupted


async def test_rejects_a_malformed_middle_line_without_modifying_the_file(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    repository = create_repository(root)
    session = await repository.create(JsonlSessionCreateOptions(id="session", cwd=str(root)))
    metadata = await session.get_metadata()
    await session.append_custom_entry("first")
    await session.append_custom_entry("second")
    lines = Path(metadata.path).read_text().rstrip("\n").split("\n")
    corrupted = f"{lines[0]}\n{lines[1]}\nnot-json\n{lines[2]}\n"
    Path(metadata.path).write_text(corrupted)

    with pytest.raises(SessionError) as exc_info:
        await create_repository(root).open(metadata)
    assert exc_info.value.code == "invalid_entry"
    assert Path(metadata.path).read_text() == corrupted


async def test_rejects_an_imported_entry_that_references_a_missing_parent(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    path = root / "session-missing-parent.jsonl"
    header = {"kind": "header", "version": 4, "id": "missing-parent", "createdAt": 1, "cwd": str(root)}
    entry = {
        "kind": "entry",
        "type": "custom",
        "id": "orphan",
        "customType": "note",
        "parentId": "missing",
        "seq": 1,
        "timestamp": 1,
    }
    path.write_text(f"{json.dumps(header)}\n{json.dumps(entry)}\n")
    metadata = JsonlSessionMetadata(
        id="missing-parent",
        created_at=1,
        path=str(path),
        cwd=str(root),
        modified_at=path.stat().st_mtime_ns // 1_000_000,
        source_format=4,
    )

    with pytest.raises(SessionError) as exc_info:
        await create_repository(root).open(metadata)
    assert exc_info.value.code == "invalid_entry"
    assert str(exc_info.value) == (
        f"Invalid JSONL v4 session {path}: line 2 Invalid session mutation: references missing parent missing"
    )


async def test_rejects_a_lane_bound_entry_that_does_not_chain_to_the_lane_leaf(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    repository = create_repository(root)
    session = await repository.create(JsonlSessionCreateOptions(id="session", cwd=str(root)))
    metadata = await session.get_metadata()
    await session.append_custom_entry("first")
    await session.append_custom_entry("second")

    lines = read_lines(metadata.path)
    lines[2]["parentId"] = None
    Path(metadata.path).write_text("\n".join(json.dumps(line) for line in lines) + "\n")

    with pytest.raises(SessionError) as exc_info:
        await create_repository(root).open(metadata)
    assert exc_info.value.code == "invalid_entry"
    assert "does not chain to the lane leaf" in str(exc_info.value)


async def test_does_not_move_a_lane_for_an_imported_entry_without_lane_metadata(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    path = root / "session-import.jsonl"
    header = {"kind": "header", "version": 4, "id": "import", "createdAt": 1, "cwd": str(root)}
    imported_entry = {
        "kind": "entry",
        "type": "custom",
        "id": "imported",
        "customType": "note",
        "parentId": None,
        "seq": 1,
        "timestamp": 1,
    }
    path.write_text(f"{json.dumps(header)}\n{json.dumps(imported_entry)}\n")
    metadata = JsonlSessionMetadata(
        id="import",
        created_at=1,
        path=str(path),
        cwd=str(root),
        modified_at=path.stat().st_mtime_ns // 1_000_000,
        source_format=4,
    )

    imported = await create_repository(root).open(metadata)
    assert await imported.get_leaf_id() is None
    assert [entry.id for entry in await imported.find_entries()] == ["imported"]

    with path.open("a") as handle:
        handle.write(json.dumps({"kind": "lane", "seq": 2, "lane": "main", "leafId": "imported"}) + "\n")
    moved = await create_repository(root).open(metadata)
    assert await moved.get_leaf_id() == "imported"


@dataclass(frozen=True)
class ReplayCase:
    name: str
    message: str
    mutations: list[dict[str, Any]]


REPLAY_CASES = [
    ReplayCase(
        name="a non-consecutive sequence",
        message="non-consecutive seq",
        mutations=[
            {
                "kind": "entry",
                "type": "custom",
                "id": "entry",
                "customType": "note",
                "parentId": None,
                "seq": 2,
                "timestamp": 1,
            }
        ],
    ),
    ReplayCase(
        name="a duplicate entry/record id",
        message="duplicate id",
        mutations=[
            {
                "kind": "entry",
                "type": "custom",
                "id": "duplicate",
                "customType": "note",
                "parentId": None,
                "seq": 1,
                "timestamp": 1,
            },
            {
                "kind": "record",
                "type": "operation_started",
                "id": "duplicate",
                "lane": "main",
                "seq": 2,
                "timestamp": 2,
                "sourceLeafId": None,
                "intent": {"kind": "run", "originalPrompt": [], "initialMessages": []},
            },
        ],
    ),
    ReplayCase(
        name="an entry with a missing parent",
        message="missing parent",
        mutations=[
            {
                "kind": "entry",
                "type": "custom",
                "id": "entry",
                "customType": "note",
                "parentId": "missing",
                "seq": 1,
                "timestamp": 1,
            }
        ],
    ),
    ReplayCase(
        name="an entry referencing a missing lane",
        message="missing lane",
        mutations=[
            {
                "kind": "entry",
                "lane": "thread",
                "type": "custom",
                "id": "entry",
                "customType": "note",
                "parentId": None,
                "seq": 1,
                "timestamp": 1,
            }
        ],
    ),
    ReplayCase(
        name="a record referencing a missing lane",
        message="missing lane",
        mutations=[
            {
                "kind": "record",
                "type": "operation_started",
                "id": "run",
                "lane": "thread",
                "seq": 1,
                "timestamp": 1,
                "sourceLeafId": None,
                "intent": {"kind": "run", "originalPrompt": [], "initialMessages": []},
            }
        ],
    ),
    ReplayCase(
        name="a lane move referencing a missing entry",
        message="missing lane target",
        mutations=[{"kind": "lane", "lane": "thread", "leafId": "missing", "seq": 1}],
    ),
    ReplayCase(
        name="a label referencing a missing entry",
        message="missing label target",
        mutations=[{"kind": "fact", "fact": "label", "targetId": "missing", "label": "checkpoint", "seq": 1}],
    ),
]


@pytest.mark.parametrize("case", REPLAY_CASES, ids=lambda case: case.name)
async def test_rejects_invalid_mutations_during_replay(tmp_path: Path, case: ReplayCase) -> None:
    root = tmp_path / "root"
    root.mkdir()
    metadata = write_raw_session(root, re.sub(r"[^A-Za-z0-9._-]", "-", case.name), case.mutations)

    with pytest.raises(SessionError) as exc_info:
        await create_repository(root).open(metadata)
    assert exc_info.value.code == "invalid_entry"
    assert case.message in str(exc_info.value)


async def test_rejects_a_complete_malformed_interior_mutation_without_modifying_the_file(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    metadata = write_raw_session(
        root,
        "malformed-interior",
        [
            {
                "kind": "record",
                "type": "operation_started",
                "id": "run",
                "lane": "main",
                "seq": 1,
                "timestamp": 1,
                "sourceLeafId": None,
            },
            {"kind": "fact", "fact": "name", "name": "after", "seq": 2},
        ],
    )
    corrupted = Path(metadata.path).read_text()

    with pytest.raises(SessionError) as exc_info:
        await create_repository(root).open(metadata)
    assert exc_info.value.code == "invalid_entry"
    assert Path(metadata.path).read_text() == corrupted


async def test_preserves_the_session_when_staging_torn_tail_repair_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    repository = create_repository(root)
    session = await repository.create(JsonlSessionCreateOptions(id="repair-failure", cwd=str(root)))
    metadata = await session.get_metadata()
    await session.append_custom_entry("kept")
    with Path(metadata.path).open("a") as handle:
        handle.write('{"kind":"entry"')
    original = Path(metadata.path).read_text()

    original_write_text = Path.write_text

    def failing_write_text(self: Path, *args: Any, **kwargs: Any) -> int:
        if str(self) == f"{metadata.path}.tmp":
            original_write_text(self, "")
            raise OSError("repair interrupted after truncation")
        return original_write_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", failing_write_text)

    with pytest.raises(SessionError) as exc_info:
        await create_repository(root).open(metadata)
    assert exc_info.value.code == "storage"

    monkeypatch.undo()
    assert Path(metadata.path).read_text() == original
    assert not Path(f"{metadata.path}.tmp").exists()


async def test_preserves_the_session_when_torn_tail_repair_cannot_be_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    repository = create_repository(root)
    session = await repository.create(JsonlSessionCreateOptions(id="repair-rename-failure", cwd=str(root)))
    metadata = await session.get_metadata()
    await session.append_custom_entry("kept")
    with Path(metadata.path).open("a") as handle:
        handle.write('{"kind":"entry"')
    original = Path(metadata.path).read_text()

    original_replace = os.replace
    remaining = [1]

    def failing_replace(src: Any, dst: Any, **kwargs: Any) -> None:
        if remaining and remaining.pop():
            raise OSError("injected repair rename failure")
        original_replace(src, dst, **kwargs)

    monkeypatch.setattr(storage_module.os, "replace", failing_replace)

    with pytest.raises(SessionError) as exc_info:
        await create_repository(root).open(metadata)
    assert exc_info.value.code == "storage"

    monkeypatch.undo()
    assert Path(metadata.path).read_text() == original
    assert not Path(f"{metadata.path}.tmp").exists()
