"""Session facade over a `SessionStorage` backend.

Python port of `packages/agent/src/harness/session/session.ts`. `Session`
provides the `SessionTree` (single-lane, tree-relative) view used by callers
that only care about "the branch I'm on", plus lane/record management for
callers that need multi-lane and record-level control.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Any

from pi_ai.utils.uuid import uuidv7

from ...types import AgentMessage
from .types import (
    BranchBounds,
    CustomEntry,
    Entry,
    EntryQuery,
    IdGenerator,
    LanePointer,
    LaneRecord,
    LogItem,
    LogOptions,
    MessageEntry,
    NewRecord,
    OperationStartedRecord,
    ProvisionedEntry,
    RecordQuery,
    SessionError,
    SessionMetadata,
    SessionStats,
    SessionStorage,
)


def _assert_valid_limit(limit: int | None) -> None:
    if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0):
        raise SessionError("invalid_query", "limit must be a positive integer")


def _assert_valid_cursor(after_seq: int | None) -> None:
    if after_seq is not None and (not isinstance(after_seq, int) or isinstance(after_seq, bool) or after_seq < 0):
        raise SessionError("invalid_query", "cursor sequence must be a non-negative integer")


def _invalid_payload(reason: str) -> None:
    raise SessionError("invalid_payload", f"Durable payload {reason}")


def assert_json_serializable(value: Any) -> None:
    """Reject a durable payload that JSON cannot round-trip.

    TypeScript walks a plain JS object/array structurally. Python's storage
    payloads are dataclass instances (entries/records) that may embed
    arbitrary `Any`-typed values (`data`, `details`, `effective_args`,
    `resume_data`, ...), so this walk also descends into dataclass fields,
    treating them like TypeScript's plain objects.
    """
    active: set[int] = set()
    stack: list[Any] = [value]
    exiting: list[int] = []

    def push_exit(obj: Any) -> None:
        exiting.append(id(obj))
        stack.append(_Exit(id(obj)))

    while stack:
        frame = stack.pop()
        if isinstance(frame, _Exit):
            active.discard(frame.marker)
            continue
        candidate = frame

        if candidate is None or isinstance(candidate, (str, bool)):
            continue
        if isinstance(candidate, int):
            continue
        if isinstance(candidate, float):
            if not math.isfinite(candidate):
                _invalid_payload("contains a non-finite number")
            continue

        if isinstance(candidate, list):
            marker = id(candidate)
            if marker in active:
                _invalid_payload("contains a cycle")
            active.add(marker)
            push_exit(candidate)
            for item in reversed(candidate):
                stack.append(item)
            continue

        if isinstance(candidate, dict):
            marker = id(candidate)
            if marker in active:
                _invalid_payload("contains a cycle")
            active.add(marker)
            push_exit(candidate)
            for key, item in reversed(list(candidate.items())):
                if not isinstance(key, str):
                    _invalid_payload("contains a non-string key")
                stack.append(item)
            continue

        if dataclasses.is_dataclass(candidate) and not isinstance(candidate, type):
            marker = id(candidate)
            if marker in active:
                _invalid_payload("contains a cycle")
            active.add(marker)
            push_exit(candidate)
            for entry_field in reversed(dataclasses.fields(candidate)):
                stack.append(getattr(candidate, entry_field.name))
            continue

        _invalid_payload(f"contains {type(candidate).__name__}")


@dataclass
class _Exit:
    marker: int


class _LaneView:
    """Lane-scoped `SessionTree` returned by `Session.view()` for non-"main" lanes."""

    def __init__(self, session: Session, lane: str) -> None:
        self._session = session
        self._lane = lane

    async def get_leaf_id(self) -> str | None:
        return await self._session._get_leaf_id_for_lane(self._lane)

    async def get_entry(self, id: str) -> Entry | None:
        return await self._session.get_entry(id)

    async def get_stats(self) -> SessionStats:
        return await self._session.get_stats()

    async def get_name(self) -> str | None:
        return await self._session.get_name()

    async def set_name(self, name: str | None) -> None:
        await self._session.set_name(name)

    async def get_label(self, target_id: str) -> str | None:
        return await self._session.get_label(target_id)

    async def set_label(self, target_id: str, label: str | None) -> None:
        await self._session.set_label(target_id, label)

    async def find_entries(self, query: EntryQuery | None = None) -> list[Entry]:
        return await self._session._query_entries(query)

    async def find_entry(self, query: EntryQuery | None = None) -> Entry | None:
        results = await self._session._query_entries(query, result_limit=1)
        return results[0] if results else None

    async def find_entries_on_branch(
        self, query: EntryQuery | None = None, bounds: BranchBounds | None = None
    ) -> list[Entry]:
        return await self._session._query_branch_entries(self._lane, query, bounds)

    async def find_entry_on_branch(
        self, query: EntryQuery | None = None, bounds: BranchBounds | None = None
    ) -> Entry | None:
        results = await self._session._query_branch_entries(self._lane, query, bounds, result_limit=1)
        return results[0] if results else None

    async def append_message(self, message: AgentMessage) -> str:
        return await self._session._append_message_to_lane(self._lane, message)

    async def append_custom_entry(self, custom_type: str, data: Any = None) -> str:
        return await self._session._append_custom_entry_to_lane(self._lane, custom_type, data)


class Session:
    def __init__(self, storage: SessionStorage, id_generator: IdGenerator | None = None) -> None:
        self._storage = storage
        self.id_generator: IdGenerator = id_generator or _UuidV7IdGenerator()

    async def get_metadata(self) -> SessionMetadata:
        return await self._storage.get_metadata()

    def view(self, lane: str) -> Any:
        if lane == "main":
            return self
        return _LaneView(self, lane)

    async def get_leaf_id(self) -> str | None:
        return await self._get_leaf_id_for_lane("main")

    async def get_entry(self, id: str) -> Entry | None:
        return await self._storage.get_entry(id)

    async def get_stats(self) -> SessionStats:
        return await self._storage.get_stats()

    async def get_name(self) -> str | None:
        return await self._storage.get_name()

    async def set_name(self, name: str | None) -> None:
        await self._storage.set_name(name)

    async def get_label(self, target_id: str) -> str | None:
        return await self._storage.get_label(target_id)

    async def set_label(self, target_id: str, label: str | None) -> None:
        await self._storage.set_label(target_id, label)

    async def find_entries(self, query: EntryQuery | None = None) -> list[Entry]:
        return await self._query_entries(query)

    async def find_entry(self, query: EntryQuery | None = None) -> Entry | None:
        results = await self._query_entries(query, result_limit=1)
        return results[0] if results else None

    async def find_entries_on_branch(
        self, query: EntryQuery | None = None, bounds: BranchBounds | None = None
    ) -> list[Entry]:
        return await self._query_branch_entries("main", query, bounds)

    async def find_entry_on_branch(
        self, query: EntryQuery | None = None, bounds: BranchBounds | None = None
    ) -> Entry | None:
        results = await self._query_branch_entries("main", query, bounds, result_limit=1)
        return results[0] if results else None

    async def append_message(self, message: AgentMessage) -> str:
        return await self._append_message_to_lane("main", message)

    async def append_custom_entry(self, custom_type: str, data: Any = None) -> str:
        return await self._append_custom_entry_to_lane("main", custom_type, data)

    async def get_lanes(self) -> list[LanePointer]:
        return await self._storage.get_lanes()

    async def create_lane(self, lane: str, at: str | None) -> None:
        await self._storage.create_lane(lane, at)

    async def move_lane(self, lane: str, to: str | None) -> None:
        await self._storage.move_lane(lane, to)

    async def append_entry(self, entry: ProvisionedEntry, lane: str) -> Entry:
        return await self._commit_entry(entry, lane)

    async def append_record(self, record: NewRecord) -> LaneRecord:
        return await self._commit_record(record)

    async def find_records(self, query: RecordQuery | None = None) -> list[LaneRecord]:
        return await self._query_records(query)

    async def find_open_operations(self, lane: str, limit: int | None = None) -> list[OperationStartedRecord]:
        _assert_valid_limit(limit)
        return await self._storage.find_open_operations(lane, limit)

    async def get_log(self, options: LogOptions | None = None) -> list[LogItem]:
        return await self._query_log(options)

    async def _get_leaf_id_for_lane(self, lane: str) -> str | None:
        """Returns the lane's current leaf, or `None` when empty. Raises when the lane does not exist."""
        for pointer in await self.get_lanes():
            if pointer.lane == lane:
                return pointer.leaf_id
        raise SessionError("invalid_lane", f"Lane not found: {lane}")

    async def _query_entries(self, query: EntryQuery | None = None, result_limit: int | None = -1) -> list[Entry]:
        query = query or EntryQuery()
        _assert_valid_limit(query.limit)
        _assert_valid_cursor(query.cursor.after_seq if query.cursor is not None else None)
        effective_limit = query.limit if result_limit == -1 else result_limit
        return await self._storage.find_entries(
            query if effective_limit == query.limit else dataclasses.replace(query, limit=effective_limit)
        )

    async def _query_branch_entries(
        self,
        default_lane: str,
        query: EntryQuery | None = None,
        bounds: BranchBounds | None = None,
        result_limit: int | None = -1,
    ) -> list[Entry]:
        query = query or EntryQuery()
        bounds = bounds or BranchBounds()
        _assert_valid_limit(query.limit)
        _assert_valid_cursor(query.cursor.after_seq if query.cursor is not None else None)
        start = bounds.start if bounds.start is not None else await self._get_leaf_id_for_lane(default_lane)
        if start is None:
            return []
        effective_limit = query.limit if result_limit == -1 else result_limit
        storage_query = query if effective_limit == query.limit else dataclasses.replace(query, limit=effective_limit)
        return await self._storage.find_entries_on_branch(start, storage_query, bounds)

    async def _query_records(self, query: RecordQuery | None = None) -> list[LaneRecord]:
        query = query or RecordQuery()
        _assert_valid_limit(query.limit)
        _assert_valid_cursor(query.after_seq)
        if query.operation_kind is not None and query.type != "operation_started":
            raise SessionError("invalid_query", 'operation_kind requires type "operation_started"')
        return await self._storage.find_records(query)

    async def _query_log(self, options: LogOptions | None = None) -> list[LogItem]:
        options = options or LogOptions()
        _assert_valid_limit(options.limit)
        _assert_valid_cursor(options.after_seq)
        return await self._storage.get_log(options)

    async def _append_message_to_lane(self, lane: str, message: AgentMessage) -> str:
        entry = await self._commit_entry(MessageEntry(id=self.id_generator.next(), message=message), lane)
        return entry.id

    async def _append_custom_entry_to_lane(self, lane: str, custom_type: str, data: Any = None) -> str:
        entry = await self._commit_entry(
            CustomEntry(id=self.id_generator.next(), custom_type=custom_type, data=data), lane
        )
        return entry.id

    async def _commit_entry(self, entry: ProvisionedEntry, lane: str) -> Entry:
        assert_json_serializable(entry)
        return await self._storage.append_entry(entry, lane)

    async def _commit_record(self, record: NewRecord) -> LaneRecord:
        assert_json_serializable(record)
        return await self._storage.append_record(record)


class _UuidV7IdGenerator:
    def next(self) -> str:
        return uuidv7()


__all__ = ["Session", "assert_json_serializable"]
