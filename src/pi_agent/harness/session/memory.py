"""In-memory session storage and repository.

Python port of `packages/agent/src/harness/session/memory.ts`. Useful for
tests and any caller that does not need durability across process restarts.
"""

from __future__ import annotations

import copy
from dataclasses import replace

from pi_ai.types import now_ms
from pi_ai.utils.uuid import uuidv7

from .session import Session
from .state import EntryMutation, LabelFactMutation, LaneMutation, NameFactMutation, RecordMutation, SessionState
from .types import (
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
    SessionCreateOptions,
    SessionError,
    SessionMetadata,
    SessionStats,
)


class InMemorySessionStorage:
    def __init__(self, metadata: SessionMetadata) -> None:
        self._metadata = copy.deepcopy(metadata)
        self._state = SessionState()

    def fork(self, metadata: SessionMetadata, options: ForkOptions) -> InMemorySessionStorage:
        storage = InMemorySessionStorage(metadata)
        for mutation in self._state.create_fork_mutations(options):
            storage._state.apply_mutation(mutation)
        return storage

    async def get_metadata(self) -> SessionMetadata:
        return copy.deepcopy(self._metadata)

    async def get_lanes(self) -> list[LanePointer]:
        return self._state.get_lanes()

    async def create_lane(self, lane: str, at: str | None) -> None:
        self._state.validate_new_lane(lane)
        self._state.validate_target(at)
        self._state.apply_mutation(LaneMutation(seq=self._state.next_sequence, lane=lane, leaf_id=at))

    async def move_lane(self, lane: str, to: str | None) -> None:
        self._state.require_lane(lane)
        self._state.validate_target(to)
        self._state.apply_mutation(LaneMutation(seq=self._state.next_sequence, lane=lane, leaf_id=to))

    async def append_entry(self, new_entry: ProvisionedEntry, lane: str) -> Entry:
        parent_id = self._state.require_lane(lane)
        self._state.validate_unused_id(new_entry.id)
        entry = replace(
            copy.deepcopy(new_entry), parent_id=parent_id, seq=self._state.next_sequence, timestamp=now_ms()
        )
        self._state.apply_mutation(EntryMutation(lane=lane, entry=entry))
        return copy.deepcopy(entry)

    async def append_record(self, new_record: NewRecord) -> LaneRecord:
        self._state.require_lane(new_record.lane)
        self._state.validate_unused_id(new_record.id)
        open_operations = self._state.find_open_operations(new_record.lane, limit=1)
        current_open_operation_id = open_operations[0].id if open_operations else None
        if new_record.type == "operation_started" and current_open_operation_id is not None:
            raise SessionError(
                "storage", f"Lane {new_record.lane} already has an open operation {current_open_operation_id}"
            )
        record = replace(copy.deepcopy(new_record), seq=self._state.next_sequence, timestamp=now_ms())
        self._state.apply_mutation(RecordMutation(record=record))
        return copy.deepcopy(record)

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
        self._state.apply_mutation(NameFactMutation(seq=self._state.next_sequence, name=name))

    async def get_label(self, id: str) -> str | None:
        return self._state.get_label(id)

    async def set_label(self, id: str, label: str | None) -> None:
        self._state.validate_target(id)
        self._state.apply_mutation(LabelFactMutation(seq=self._state.next_sequence, target_id=id, label=label))

    async def get_stats(self) -> SessionStats:
        return copy.deepcopy(self._state.get_stats())


class InMemorySessionRepo:
    def __init__(self) -> None:
        self._sessions: dict[str, InMemorySessionStorage] = {}

    async def create(self, options: SessionCreateOptions | None = None) -> Session:
        options = options or SessionCreateOptions()
        id = options.id if options.id is not None else uuidv7()
        if id in self._sessions:
            raise SessionError("already_exists", f"Session already exists: {id}")
        storage = InMemorySessionStorage(
            SessionMetadata(id=id, created_at=now_ms(), parent_session_id=options.parent_session_id)
        )
        self._sessions[id] = storage
        return Session(storage)

    async def open(self, metadata: SessionMetadata) -> Session:
        return Session(self._require_storage(metadata.id))

    async def list(self, options: None = None) -> list[SessionMetadata]:
        return [await storage.get_metadata() for storage in self._sessions.values()]

    async def delete(self, metadata: SessionMetadata) -> None:
        self._sessions.pop(metadata.id, None)

    async def fork(self, source: SessionMetadata, options: ForkOptions | None = None) -> Session:
        options = options or ForkOptions()
        source_storage = self._require_storage(source.id)
        id = options.id if options.id is not None else uuidv7()
        if id in self._sessions:
            raise SessionError("already_exists", f"Session already exists: {id}")
        parent_session_id = options.parent_session_id if options.parent_session_id is not None else source.id
        storage = source_storage.fork(
            SessionMetadata(id=id, created_at=now_ms(), parent_session_id=parent_session_id),
            options,
        )
        self._sessions[id] = storage
        return Session(storage)

    def _require_storage(self, id: str) -> InMemorySessionStorage:
        storage = self._sessions.get(id)
        if storage is None:
            raise SessionError("not_found", f"Session not found: {id}")
        return storage


__all__ = ["InMemorySessionRepo", "InMemorySessionStorage"]
