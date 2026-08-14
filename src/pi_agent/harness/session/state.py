"""Derived-state projection shared by every session storage backend.

Python port of `packages/agent/src/harness/session/state.ts`. `SessionState`
is the single source of truth for entries, records, lanes, facts, and
statistics; both `InMemorySessionStorage` and `JsonlSessionStorage` replay the
same `SessionMutation` stream through it so their behaviour, including error
messages and validation order, stays identical.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace
from typing import Literal

from .types import (
    BranchBounds,
    Entry,
    EntryOrder,
    EntryQuery,
    ForkOptions,
    LanePointer,
    LaneRecord,
    LogEntryItem,
    LogItem,
    LogLabelFactItem,
    LogLaneItem,
    LogNameFactItem,
    LogOptions,
    LogRecordItem,
    OperationStartedRecord,
    RecordQuery,
    SessionError,
    SessionStats,
)


@dataclass(kw_only=True)
class EntryMutation:
    entry: Entry
    lane: str | None = None
    kind: Literal["entry"] = "entry"


@dataclass(kw_only=True)
class RecordMutation:
    record: LaneRecord
    kind: Literal["record"] = "record"


@dataclass(kw_only=True)
class LaneMutation:
    seq: int
    lane: str
    leaf_id: str | None
    kind: Literal["lane"] = "lane"


@dataclass(kw_only=True)
class NameFactMutation:
    seq: int
    name: str | None
    kind: Literal["fact"] = "fact"
    fact: Literal["name"] = "name"


@dataclass(kw_only=True)
class LabelFactMutation:
    seq: int
    target_id: str
    label: str | None
    kind: Literal["fact"] = "fact"
    fact: Literal["label"] = "label"


SessionMutation = EntryMutation | RecordMutation | LaneMutation | NameFactMutation | LabelFactMutation


def _mutation_seq(mutation: SessionMutation) -> int:
    if isinstance(mutation, EntryMutation):
        return mutation.entry.seq
    if isinstance(mutation, RecordMutation):
        return mutation.record.seq
    return mutation.seq


def _invalid_mutation(message: str) -> None:
    raise SessionError("invalid_entry", f"Invalid session mutation: {message}")


def _assert_valid_limit(limit: int | None) -> None:
    if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0):
        raise SessionError("invalid_query", "limit must be a positive integer")


def _assert_valid_cursor(after_seq: int | None) -> None:
    if after_seq is not None and (not isinstance(after_seq, int) or isinstance(after_seq, bool) or after_seq < 0):
        raise SessionError("invalid_query", "cursor sequence must be a non-negative integer")


def _ordered(items: list[Entry] | list[LaneRecord], order: EntryOrder | None) -> Iterator:
    if order == "oldestFirst":
        yield from items
        return
    for index in range(len(items) - 1, -1, -1):
        yield items[index]


class SessionState:
    def __init__(self) -> None:
        self._sequence = 0
        self._used_ids: set[str] = set()
        self._entries: list[Entry] = []
        self._entries_by_id: dict[str, Entry] = {}
        self._records: list[LaneRecord] = []
        self._open_operations_by_lane: dict[str, dict[str, OperationStartedRecord]] = {}
        self._lanes: dict[str, str | None] = {"main": None}
        self._log: list[LogItem] = []
        self._stats = SessionStats()
        self._name: str | None = None
        self._labels: dict[str, str] = {}

    @property
    def next_sequence(self) -> int:
        return self._sequence + 1

    def get_lanes(self) -> list[LanePointer]:
        return [LanePointer(lane=lane, leaf_id=leaf_id) for lane, leaf_id in self._lanes.items()]

    def require_lane(self, lane: str) -> str | None:
        if lane not in self._lanes:
            raise SessionError("invalid_lane", f"Lane not found: {lane}")
        return self._lanes[lane]

    def validate_new_lane(self, lane: str) -> None:
        if lane in self._lanes:
            raise SessionError("already_exists", f"Lane already exists: {lane}")

    def validate_target(self, target_id: str | None) -> None:
        if target_id is not None and target_id not in self._entries_by_id:
            raise SessionError("not_found", f"Entry not found: {target_id}")

    def validate_unused_id(self, id: str) -> None:
        if id in self._used_ids:
            raise SessionError("already_exists", f"Session id already exists: {id}")

    def apply_mutation(self, mutation: SessionMutation) -> None:
        seq = _mutation_seq(mutation)
        if seq != self._sequence + 1:
            _invalid_mutation(f"has non-consecutive seq {seq}")

        if isinstance(mutation, EntryMutation):
            entry = mutation.entry
            if entry.id in self._used_ids:
                _invalid_mutation(f"contains duplicate id {entry.id}")
            if mutation.lane is not None:
                if mutation.lane not in self._lanes:
                    _invalid_mutation(f"references missing lane {mutation.lane}")
                if entry.parent_id != self._lanes[mutation.lane]:
                    _invalid_mutation("does not chain to the lane leaf")
            if entry.parent_id is not None and entry.parent_id not in self._entries_by_id:
                _invalid_mutation(f"references missing parent {entry.parent_id}")
            self._sequence = seq
            self._used_ids.add(entry.id)
            self._entries.append(entry)
            self._entries_by_id[entry.id] = entry
            if mutation.lane is not None:
                self._lanes[mutation.lane] = entry.id
            self._log.append(LogEntryItem(seq=seq, entry=entry))
            if entry.type == "message":
                self._stats.message_count += 1
            return

        if isinstance(mutation, RecordMutation):
            record = mutation.record
            if record.lane not in self._lanes:
                _invalid_mutation(f"references missing lane {record.lane}")
            if record.id in self._used_ids:
                _invalid_mutation(f"contains duplicate id {record.id}")
            self._sequence = seq
            self._used_ids.add(record.id)
            self._records.append(record)
            if record.type == "operation_started":
                open_operations = self._open_operations_by_lane.setdefault(record.lane, {})
                open_operations[record.id] = record
            elif record.type == "operation_finished":
                self._open_operations_by_lane.get(record.lane, {}).pop(record.run_id, None)
            self._log.append(LogRecordItem(seq=seq, record=record))
            if record.type == "usage":
                self._stats.cached_tokens += record.usage.cache_read
                self._stats.uncached_tokens += record.usage.input + record.usage.cache_write
                self._stats.total_tokens += record.usage.total_tokens
                self._stats.cost_total += record.usage.cost.total
            return

        if isinstance(mutation, LaneMutation):
            if mutation.leaf_id is not None and mutation.leaf_id not in self._entries_by_id:
                _invalid_mutation(f"references missing lane target {mutation.leaf_id}")
            self._sequence = seq
            self._lanes[mutation.lane] = mutation.leaf_id
            self._log.append(LogLaneItem(seq=seq, lane=mutation.lane, leaf_id=mutation.leaf_id))
            return

        if isinstance(mutation, NameFactMutation):
            self._sequence = seq
            self._name = mutation.name
            self._log.append(LogNameFactItem(seq=seq, name=mutation.name))
            return

        if isinstance(mutation, LabelFactMutation):
            if mutation.target_id not in self._entries_by_id:
                _invalid_mutation(f"references missing label target {mutation.target_id}")
            self._sequence = seq
            if mutation.label is None:
                self._labels.pop(mutation.target_id, None)
            else:
                self._labels[mutation.target_id] = mutation.label
            self._log.append(LogLabelFactItem(seq=seq, target_id=mutation.target_id, label=mutation.label))
            return

    def get_entry(self, id: str) -> Entry | None:
        return self._entries_by_id.get(id)

    def find_entries(self, query: EntryQuery | None = None) -> list[Entry]:
        query = query or EntryQuery()
        _assert_valid_limit(query.limit)
        _assert_valid_cursor(query.cursor.after_seq if query.cursor is not None else None)
        results: list[Entry] = []
        for entry in _ordered(self._entries, query.order):
            if not self._matches_entry_query(entry, query):
                continue
            results.append(entry)
            if len(results) == query.limit:
                break
        return results

    def find_entries_on_branch(
        self, start: str, query: EntryQuery | None = None, bounds: BranchBounds | None = None
    ) -> list[Entry]:
        query = query or EntryQuery()
        bounds = bounds or BranchBounds()
        _assert_valid_limit(query.limit)
        _assert_valid_cursor(query.cursor.after_seq if query.cursor is not None else None)
        results: list[Entry] = []
        if query.order == "oldestFirst":
            for entry in reversed(list(self._walk_to_root(start))):
                reached_bound = entry.id == bounds.stop_at_id or entry.type == bounds.stop_at_type
                if self._matches_entry_query(entry, query):
                    results.append(entry)
                if reached_bound or len(results) == query.limit:
                    break
        else:
            for entry in self._walk_to_root(start, bounds):
                if self._matches_entry_query(entry, query):
                    results.append(entry)
                if len(results) == query.limit:
                    break
        return results

    def find_records(self, query: RecordQuery | None = None) -> list[LaneRecord]:
        query = query or RecordQuery()
        _assert_valid_limit(query.limit)
        _assert_valid_cursor(query.after_seq)
        results: list[LaneRecord] = []
        for record in _ordered(self._records, query.order):
            if not self._matches_record_query(record, query):
                continue
            results.append(record)
            if len(results) == query.limit:
                break
        return results

    def find_open_operations(self, lane: str, limit: int | None = None) -> list[OperationStartedRecord]:
        _assert_valid_limit(limit)
        open_operations_by_id = self._open_operations_by_lane.get(lane)
        open_operations = list(reversed(list(open_operations_by_id.values()))) if open_operations_by_id else []
        return open_operations if limit is None else open_operations[:limit]

    def get_log(self, options: LogOptions | None = None) -> list[LogItem]:
        options = options or LogOptions()
        _assert_valid_limit(options.limit)
        _assert_valid_cursor(options.after_seq)
        results: list[LogItem] = []
        for item in self._log:
            if options.after_seq is not None and item.seq <= options.after_seq:
                continue
            results.append(item)
            if len(results) == options.limit:
                break
        return results

    def get_name(self) -> str | None:
        return self._name

    def get_label(self, id: str) -> str | None:
        return self._labels.get(id)

    def get_stats(self) -> SessionStats:
        return self._stats

    def create_fork_mutations(self, options: ForkOptions) -> list[SessionMutation]:
        if options.scope == "tree":
            copied_entries = self.find_entries(EntryQuery(order="oldestFirst"))
            fork_lanes = self.get_lanes()
        else:
            selected_entry_id = options.entry_id if options.entry_id is not None else self.require_lane("main")
            target_id: str | None = None
            if selected_entry_id is not None:
                entry = self.get_entry(selected_entry_id)
                if entry is None or entry.type != "message":
                    raise SessionError(
                        "invalid_fork_target", f"Fork target is not a message entry: {selected_entry_id}"
                    )
                position = (
                    options.position
                    if options.position is not None
                    else ("before" if options.entry_id is not None else "at")
                )
                target_id = entry.id if position == "at" else entry.parent_id
            copied_entries = (
                [] if target_id is None else self.find_entries_on_branch(target_id, EntryQuery(order="oldestFirst"))
            )
            fork_lanes = [LanePointer(lane="main", leaf_id=target_id)]

        mutations: list[SessionMutation] = []
        sequence = 1
        for source_entry in copied_entries:
            mutations.append(EntryMutation(entry=replace(source_entry, seq=sequence)))
            sequence += 1
        for pointer in fork_lanes:
            mutations.append(LaneMutation(seq=sequence, lane=pointer.lane, leaf_id=pointer.leaf_id))
            sequence += 1
        if self._name is not None:
            mutations.append(NameFactMutation(seq=sequence, name=self._name))
            sequence += 1
        for entry in copied_entries:
            label = self._labels.get(entry.id)
            if label is not None:
                mutations.append(LabelFactMutation(seq=sequence, target_id=entry.id, label=label))
                sequence += 1
        return mutations

    def _walk_to_root(self, start: str | None, bounds: BranchBounds | None = None) -> Iterator[Entry]:
        if start is None:
            return
        bounds = bounds or BranchBounds()
        visited: set[str] = set()
        current = self._entries_by_id.get(start)
        if current is None:
            raise SessionError("not_found", f"Entry not found: {start}")
        while current is not None:
            if current.id in visited:
                raise SessionError("invalid_entry", f"Session branch contains a cycle at {current.id}")
            visited.add(current.id)
            yield current
            if current.id == bounds.stop_at_id or current.type == bounds.stop_at_type or current.parent_id is None:
                break
            parent_id = current.parent_id
            current = self._entries_by_id.get(parent_id)
            if current is None:
                raise SessionError("invalid_entry", f"Entry not found: {parent_id}")

    def _matches_entry_query(self, entry: Entry, query: EntryQuery) -> bool:
        if query.type is not None and entry.type != query.type:
            return False
        if query.custom_type is not None and not (entry.type == "custom" and entry.custom_type == query.custom_type):
            return False
        if query.cursor is not None:
            if query.order == "oldestFirst":
                if not entry.seq > query.cursor.after_seq:
                    return False
            elif not entry.seq < query.cursor.after_seq:
                return False
        return True

    def _matches_record_query(self, record: LaneRecord, query: RecordQuery) -> bool:
        if query.lane is not None and record.lane != query.lane:
            return False
        if query.type is not None and record.type != query.type:
            return False
        if query.run_id is not None:
            if record.type == "operation_started":
                if record.id != query.run_id:
                    return False
            elif getattr(record, "run_id", None) != query.run_id:
                return False
        if query.operation_kind is not None and not (
            record.type == "operation_started" and record.intent.kind == query.operation_kind
        ):
            return False
        return not (query.after_seq is not None and record.seq <= query.after_seq)


__all__ = [
    "EntryMutation",
    "LabelFactMutation",
    "LaneMutation",
    "NameFactMutation",
    "RecordMutation",
    "SessionMutation",
    "SessionState",
]
