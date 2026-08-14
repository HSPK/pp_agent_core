"""Append-only JSONL v4 session storage.

Python port of `packages/agent/src/harness/session/jsonl/storage.ts`. Each
session is one `.jsonl` file: a header line followed by one mutation per
line. `JsonlSessionStorage` replays the file into a `SessionState` on load
and appends new mutations durably (open in append mode, write one line,
flush) as they are recorded.

The TypeScript version is parameterized over an injectable `FileSystem`
(`packages/agent/src/harness/types.ts`); that abstraction is out of scope for
this port (see `jsonl/types.py`), so this module uses `pathlib.Path` with
plain synchronous file IO wrapped in `async def` methods, and raises
`SessionError`/`JsonlDecodeError` directly instead of unwrapping a
`Result<T, FileError>`.

Concurrency: TypeScript serializes mutating calls with an in-process promise
chain (`this.tail`). Python's `async def` methods can likewise interleave at
`await` points within a single event loop, so this port uses an `asyncio.Lock`
per storage instance to serialize mutating operations the same way. Neither
mechanism protects a session file against a second OS *process* writing to it
concurrently; `jsonl.lockfile.FileLock` adds that cross-process guard and is
used here around every append and around atomic publish, as a Python-specific
durability addition (see `jsonl/lockfile.py`).
"""

from __future__ import annotations

import asyncio
import copy
import os
from collections.abc import Awaitable, Callable
from dataclasses import replace
from pathlib import Path
from typing import TypeVar

from pi_ai.types import now_ms

from ..state import (
    EntryMutation,
    LabelFactMutation,
    LaneMutation,
    NameFactMutation,
    RecordMutation,
    SessionMutation,
    SessionState,
)
from ..types import (
    BranchBounds,
    Entry,
    EntryQuery,
    ForkOptions,
    LanePointer,
    LaneRecord,
    LogItem,
    LogOptions,
    NewRecord,
    OperationStartedRecord,
    ProvisionedEntry,
    RecordQuery,
    SessionError,
    SessionStats,
)
from .codec import decode_header, decode_mutation, encode_header, encode_mutation, metadata_from_header
from .errors import JsonlDecodeError, file_error_guard, invalid_file
from .lockfile import FileLock
from .types import JsonlSessionMetadata, JsonlV4Header

T = TypeVar("T")


async def _to_thread(func: Callable[[], T]) -> T:
    return await asyncio.get_running_loop().run_in_executor(None, func)


def _publish_file_atomically_sync(destination_path: Path, populate: Callable[[Path], None]) -> None:
    """Build a sibling temporary file, then atomically rename it over the destination.

    `populate` must create or overwrite `temp_path` with the complete file
    contents. The destination is untouched until the rename commits, so a
    process crash while populating can leave only the ignored `.tmp` file
    behind. Callers must serialize publications to the same destination
    because they share its deterministic `.tmp` path.
    """
    temp_path = destination_path.with_name(f"{destination_path.name}.tmp")
    try:
        populate(temp_path)
        with file_error_guard(f"Failed to publish staged file {destination_path}"):
            os.replace(temp_path, destination_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


async def _publish_file_atomically(destination_path: Path, populate: Callable[[Path], None]) -> None:
    await _to_thread(lambda: _publish_file_atomically_sync(destination_path, populate))


async def _publish_async_atomically(destination_path: Path, populate: Callable[[Path], Awaitable[None]]) -> None:
    """Async-callback variant of `_publish_file_atomically` for populate steps that await."""
    temp_path = destination_path.with_name(f"{destination_path.name}.tmp")

    def publish() -> None:
        with file_error_guard(f"Failed to publish staged file {destination_path}"):
            os.replace(temp_path, destination_path)

    try:
        await populate(temp_path)
        await _to_thread(publish)
    except Exception:
        await _to_thread(lambda: temp_path.unlink(missing_ok=True))
        raise


class JsonlSessionStorage:
    def __init__(self, path: str | Path, metadata: JsonlSessionMetadata) -> None:
        self._path = Path(path)
        self._metadata = copy.deepcopy(metadata)
        self._state = SessionState()
        self._tail = asyncio.Lock()

    @staticmethod
    async def create(path: str | Path, header: JsonlV4Header) -> JsonlSessionStorage:
        path = Path(path)

        def populate() -> int:
            with file_error_guard(f"Failed to initialize session {path}"):
                path.write_text(encode_header(header))
            with file_error_guard(f"Failed to read session metadata {path}"):
                return path.stat().st_mtime_ns // 1_000_000

        modified_at = await _to_thread(populate)
        return JsonlSessionStorage(path, metadata_from_header(header, str(path), modified_at))

    @staticmethod
    async def load(path: str | Path) -> JsonlSessionStorage:
        path = Path(path)

        def read() -> str:
            with file_error_guard(f"Failed to read session {path}"):
                return path.read_text()

        content = await _to_thread(read)
        physical_lines = content.split("\n")
        if physical_lines and physical_lines[-1] == "":
            physical_lines.pop()
        if not physical_lines or not physical_lines[0]:
            raise invalid_file(str(path), 1, JsonlDecodeError("schema", "is missing a header"))
        try:
            header = decode_header(physical_lines[0])
        except JsonlDecodeError as error:
            raise invalid_file(str(path), 1, error) from error

        def stat_mtime() -> int:
            with file_error_guard(f"Failed to read session metadata {path}"):
                return path.stat().st_mtime_ns // 1_000_000

        modified_at = await _to_thread(stat_mtime)
        storage = JsonlSessionStorage(path, metadata_from_header(header, str(path), modified_at))
        for index in range(1, len(physical_lines)):
            line = physical_lines[index]
            try:
                mutation = decode_mutation(line)
            except JsonlDecodeError as error:
                is_torn_tail = index == len(physical_lines) - 1 and error.kind == "syntax"
                if is_torn_tail:
                    # Drop the unacknowledged partial append by atomically publishing the valid prefix.
                    valid_prefix = "\n".join(physical_lines[:index]) + "\n"

                    def repair(temp_path: Path, valid_prefix: str = valid_prefix) -> None:
                        with file_error_guard(f"Failed to stage torn-tail repair {path}"):
                            temp_path.write_text(valid_prefix)

                    await _publish_file_atomically(path, repair)
                    return storage
                raise invalid_file(str(path), index + 1, error) from error
            try:
                storage._apply_mutation(mutation)
            except SessionError as error:
                if error.code == "invalid_entry":
                    raise invalid_file(str(path), index + 1, error) from error
                raise
        if not content.endswith("\n"):

            def repair_tail() -> None:
                with file_error_guard(f"Failed to repair unterminated session tail {path}"), path.open("a") as handle:
                    handle.write("\n")
                    handle.flush()

            await _to_thread(repair_tail)
        return storage

    async def fork(self, path: str | Path, header: JsonlV4Header, options: ForkOptions) -> JsonlSessionStorage:
        path = Path(path)
        mutations = self._state.create_fork_mutations(options)

        async def populate(temp_path: Path) -> None:
            target_storage = await JsonlSessionStorage.create(temp_path, header)
            for mutation in mutations:
                await target_storage._append_mutation(mutation)
                target_storage._apply_mutation(mutation)

        await _publish_async_atomically(path, populate)
        return await JsonlSessionStorage.load(path)

    async def drain(self) -> None:
        async with self._tail:
            return

    async def get_metadata(self) -> JsonlSessionMetadata:
        return copy.deepcopy(self._metadata)

    async def get_lanes(self) -> list[LanePointer]:
        return self._state.get_lanes()

    async def create_lane(self, lane: str, at: str | None) -> None:
        async def operation() -> None:
            self._state.validate_new_lane(lane)
            self._state.validate_target(at)
            mutation = _lane_mutation(self._state.next_sequence, lane, at)
            await self._append_mutation(mutation)
            self._apply_mutation(mutation)

        await self._enqueue(operation)

    async def move_lane(self, lane: str, to: str | None) -> None:
        async def operation() -> None:
            self._state.require_lane(lane)
            self._state.validate_target(to)
            mutation = _lane_mutation(self._state.next_sequence, lane, to)
            await self._append_mutation(mutation)
            self._apply_mutation(mutation)

        await self._enqueue(operation)

    async def append_entry(self, new_entry: ProvisionedEntry, lane: str) -> Entry:
        async def operation() -> Entry:
            parent_id = self._state.require_lane(lane)
            self._state.validate_unused_id(new_entry.id)
            entry = replace(
                copy.deepcopy(new_entry), parent_id=parent_id, seq=self._state.next_sequence, timestamp=now_ms()
            )
            mutation = _entry_mutation(lane, entry)
            await self._append_mutation(mutation)
            self._apply_mutation(mutation)
            return copy.deepcopy(entry)

        return await self._enqueue(operation)

    async def append_record(self, new_record: NewRecord) -> LaneRecord:
        async def operation() -> LaneRecord:
            self._state.require_lane(new_record.lane)
            self._state.validate_unused_id(new_record.id)
            open_operations = self._state.find_open_operations(new_record.lane, limit=1)
            current_open_operation_id = open_operations[0].id if open_operations else None
            if new_record.type == "operation_started" and current_open_operation_id is not None:
                raise SessionError(
                    "storage",
                    f"Lane {new_record.lane} already has an open operation {current_open_operation_id}",
                )
            record = replace(copy.deepcopy(new_record), seq=self._state.next_sequence, timestamp=now_ms())
            mutation = _record_mutation(record)
            await self._append_mutation(mutation)
            self._apply_mutation(mutation)
            return copy.deepcopy(record)

        return await self._enqueue(operation)

    async def get_entry(self, id: str) -> Entry | None:
        entry = self._state.get_entry(id)
        return None if entry is None else copy.deepcopy(entry)

    async def find_entries(self, query: EntryQuery | None = None) -> list[Entry]:
        return copy.deepcopy(self._state.find_entries(query))

    async def find_entries_on_branch(
        self, start: str, query: EntryQuery | None = None, bounds: BranchBounds | None = None
    ) -> list[Entry]:
        return copy.deepcopy(self._state.find_entries_on_branch(start, query, bounds))

    async def find_records(self, query: RecordQuery | None = None) -> list[LaneRecord]:
        return copy.deepcopy(self._state.find_records(query))

    async def find_open_operations(self, lane: str, limit: int | None = None) -> list[OperationStartedRecord]:
        return copy.deepcopy(self._state.find_open_operations(lane, limit))

    async def get_log(self, options: LogOptions | None = None) -> list[LogItem]:
        return copy.deepcopy(self._state.get_log(options))

    async def get_name(self) -> str | None:
        return self._state.get_name()

    async def set_name(self, name: str | None) -> None:
        async def operation() -> None:
            mutation = _name_mutation(self._state.next_sequence, name)
            await self._append_mutation(mutation)
            self._apply_mutation(mutation)

        await self._enqueue(operation)

    async def get_label(self, id: str) -> str | None:
        return self._state.get_label(id)

    async def set_label(self, id: str, label: str | None) -> None:
        async def operation() -> None:
            self._state.validate_target(id)
            mutation = _label_mutation(self._state.next_sequence, id, label)
            await self._append_mutation(mutation)
            self._apply_mutation(mutation)

        await self._enqueue(operation)

    async def get_stats(self) -> SessionStats:
        return copy.deepcopy(self._state.get_stats())

    async def _enqueue(self, operation: Callable[[], Awaitable[T]]) -> T:
        async with self._tail:
            return await operation()

    async def _append_mutation(self, mutation: SessionMutation) -> None:
        line = encode_mutation(mutation)

        def write() -> None:
            with FileLock(self._path), self._path.open("a") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())

        with file_error_guard(f"Failed to append session {self._path}"):
            await _to_thread(write)

    def _apply_mutation(self, mutation: SessionMutation) -> None:
        self._state.apply_mutation(mutation)


def _lane_mutation(seq: int, lane: str, leaf_id: str | None) -> SessionMutation:
    return LaneMutation(seq=seq, lane=lane, leaf_id=leaf_id)


def _entry_mutation(lane: str, entry: Entry) -> SessionMutation:
    return EntryMutation(lane=lane, entry=entry)


def _record_mutation(record: LaneRecord) -> SessionMutation:
    return RecordMutation(record=record)


def _name_mutation(seq: int, name: str | None) -> SessionMutation:
    return NameFactMutation(seq=seq, name=name)


def _label_mutation(seq: int, target_id: str, label: str | None) -> SessionMutation:
    return LabelFactMutation(seq=seq, target_id=target_id, label=label)


__all__ = ["JsonlSessionStorage"]
