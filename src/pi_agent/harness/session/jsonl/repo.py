"""JSONL-backed session repository.

Python port of `packages/agent/src/harness/session/jsonl/repo.ts`.
`JsonlSessionRepo` manages a directory tree of per-cwd session directories,
each containing one `.jsonl` file per session (see `jsonl/storage.py` for the
file format).

The TypeScript version takes an injectable `FileSystem`; this port operates
directly on `pathlib.Path` (see `jsonl/types.py` and `jsonl/storage.py` for
why). `fs.absolutePath`/`fs.joinPath` become `Path.resolve()`/`Path.__truediv__`,
and `Result<T, FileError>` failures become `SessionError`/`OSError` raised
directly instead of unwrapped via `fileResult`.
"""

from __future__ import annotations

import re
from collections.abc import Coroutine
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pi_ai.types import now_ms
from pi_ai.utils.uuid import uuidv7

from ..session import Session, assert_json_serializable
from ..types import SessionError
from .codec import metadata_from_header, parse_header
from .errors import JsonlDecodeError
from .storage import JsonlSessionStorage
from .types import (
    JsonlForkOptions,
    JsonlSessionCreateOptions,
    JsonlSessionListOptions,
    JsonlSessionMetadata,
    JsonlSessionRepoOptions,
    JsonlV4Header,
)

_SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$")


def _validate_session_id(id: str) -> None:
    if not _SESSION_ID_PATTERN.match(id):
        raise SessionError(
            "invalid_payload",
            "Session id must be non-empty, contain only alphanumeric characters, '-', '_', and '.', "
            "and start and end with an alphanumeric character",
        )


def _session_directory_name(cwd: str) -> str:
    name = re.sub(r"^[/\\]", "", cwd)
    name = re.sub(r"[/\\:]", "-", name)
    return f"--{name}--"


def _session_file_name(created_at: int, id: str) -> str:
    timestamp = (
        datetime.fromtimestamp(created_at / 1000, tz=UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    )
    timestamp = timestamp.replace(":", "-").replace(".", "-")
    return f"{timestamp}_{id}.jsonl"


@dataclass(kw_only=True)
class _Destination:
    id: str
    cwd: str


async def list_jsonl_session_metadata(
    options: JsonlSessionRepoOptions, query: JsonlSessionListOptions | None = None
) -> list[JsonlSessionMetadata]:
    """Module-level session listing, port of `listJsonlSessionMetadata`.

    Upstream lifted this out of the repo class so callers can enumerate
    sessions without constructing one (`jsonl/repo.ts:65`). The behaviour is
    the repo's own listing; this is the same code path, not a second one.
    """
    return await JsonlSessionRepo(options)._list_direct(query or JsonlSessionListOptions())


async def load_jsonl_session_storage(
    options: JsonlSessionRepoOptions, metadata: JsonlSessionMetadata
) -> JsonlSessionStorage:
    """Module-level storage load, port of `loadJsonlSessionStorage` (`jsonl/repo.ts:89`)."""
    return await JsonlSessionRepo(options)._load_storage(metadata)


class JsonlSessionRepo:
    def __init__(self, options: JsonlSessionRepoOptions) -> None:
        self._sessions_root_input = options.sessions_root
        self._active_create_destinations: set[str] = set()
        self._root: Path | None = None

    async def create(self, options: JsonlSessionCreateOptions | None = None) -> Session:
        options = options or JsonlSessionCreateOptions()
        destination = self._resolve_create_destination(options)
        return await self._claim_create_destination(destination, self._do_create(destination, options))

    async def open(self, metadata: JsonlSessionMetadata) -> Session:
        return Session(await self._load_storage(metadata))

    async def list(self, options: JsonlSessionListOptions | None = None) -> list[JsonlSessionMetadata]:
        options = options or JsonlSessionListOptions()
        return await self._list_direct(options)

    async def delete(self, metadata: JsonlSessionMetadata) -> None:
        Path(metadata.path).unlink(missing_ok=True)

    async def fork(self, source: JsonlSessionMetadata, options: JsonlForkOptions | None = None) -> Session:
        options = options or JsonlForkOptions()
        source_storage = await self._load_storage(source)
        parent_session_id = options.parent_session_id if options.parent_session_id is not None else source.id
        create_options = JsonlSessionCreateOptions(
            id=options.id,
            parent_session_id=parent_session_id,
            cwd=options.cwd,
            metadata=options.metadata,
        )
        destination = self._resolve_create_destination(create_options)
        return await self._claim_create_destination(
            destination, self._do_fork(destination, create_options, source_storage, options)
        )

    async def _do_create(self, destination: _Destination, options: JsonlSessionCreateOptions) -> Session:
        header, path = self._prepare_create(destination, options)
        return Session(await JsonlSessionStorage.create(path, header))

    async def _do_fork(
        self,
        destination: _Destination,
        options: JsonlSessionCreateOptions,
        source_storage: JsonlSessionStorage,
        fork_options: JsonlForkOptions,
    ) -> Session:
        header, path = self._prepare_create(destination, options)
        return Session(await source_storage.fork(path, header, fork_options))

    async def _load_storage(self, metadata: JsonlSessionMetadata) -> JsonlSessionStorage:
        path = Path(metadata.path)
        if not path.exists():
            raise SessionError("not_found", f"Session not found: {metadata.id}")
        storage = await JsonlSessionStorage.load(path)
        loaded_metadata = await storage.get_metadata()
        if loaded_metadata.id != metadata.id:
            raise SessionError("invalid_entry", f"Session id does not match header: {metadata.id}")
        return storage

    def _resolve_create_destination(self, options: JsonlSessionCreateOptions) -> _Destination:
        id = options.id if options.id is not None else uuidv7()
        _validate_session_id(id)
        cwd = str(Path(options.cwd).resolve())
        return _Destination(id=id, cwd=cwd)

    async def _claim_create_destination(
        self, destination: _Destination, operation: Coroutine[Any, Any, Session]
    ) -> Session:
        """Prevent same-process create/fork races for one logical destination.

        The durable filename includes a timestamp, so the async filesystem
        existence check alone can let two concurrent calls both decide the
        same `{cwd, id}` is free and publish duplicate sessions.
        """
        key = f"{destination.cwd}\0{destination.id}"
        if key in self._active_create_destinations:
            # Close the not-yet-started coroutine so rejecting the duplicate does not
            # leave an un-awaited coroutine behind.
            operation.close()
            raise SessionError("already_exists", f"Session already exists: {destination.id}")
        self._active_create_destinations.add(key)
        try:
            return await operation
        finally:
            self._active_create_destinations.discard(key)

    def _prepare_create(
        self, destination: _Destination, options: JsonlSessionCreateOptions
    ) -> tuple[JsonlV4Header, Path]:
        id, cwd = destination.id, destination.cwd
        if self._session_id_exists(id, cwd):
            raise SessionError("already_exists", f"Session already exists: {id}")

        created_at = now_ms()
        session_directory = self._session_directory(cwd)
        path = session_directory / _session_file_name(created_at, id)
        if options.metadata is not None:
            assert_json_serializable(options.metadata)
        header = JsonlV4Header(
            id=id,
            created_at=created_at,
            cwd=cwd,
            parent_session_id=options.parent_session_id,
            metadata=options.metadata,
        )
        session_directory.mkdir(parents=True, exist_ok=True)
        return header, path

    async def _list_direct(self, options: JsonlSessionListOptions) -> list[JsonlSessionMetadata]:
        directories = self._session_directories(options.cwd)
        metadata: list[JsonlSessionMetadata] = []
        for directory in directories:
            files = sorted(p for p in directory.iterdir() if p.is_file() and p.name.endswith(".jsonl"))
            for file in files:
                first_line = self._read_first_line(file)
                if not first_line:
                    continue
                try:
                    header = parse_header(first_line)
                except JsonlDecodeError:
                    continue
                modified_at = int(file.stat().st_mtime * 1000)
                metadata.append(metadata_from_header(header, str(file), modified_at))
        metadata.sort(key=lambda entry: entry.modified_at, reverse=True)
        return metadata

    def _read_first_line(self, path: Path) -> str | None:
        with path.open("r") as handle:
            line = handle.readline()
        if not line:
            return None
        return line[:-1] if line.endswith("\n") else line

    def _session_id_exists(self, id: str, cwd: str) -> bool:
        suffix = f"_{id}.jsonl"
        directory = self._session_directory(cwd)
        if not directory.exists():
            return False
        return any(p.is_file() and p.name.endswith(suffix) for p in directory.iterdir())

    def _session_directories(self, cwd: str | None = None) -> list[Path]:
        root = self._resolve_root()
        if cwd is not None:
            resolved_cwd = str(Path(cwd).resolve())
            directory = self._session_directory(resolved_cwd)
            return [directory] if directory.exists() else []
        if not root.exists():
            return []
        return [p for p in root.iterdir() if p.is_dir()]

    def _session_directory(self, cwd: str) -> Path:
        return self._resolve_root() / _session_directory_name(cwd)

    def _resolve_root(self) -> Path:
        if self._root is None:
            self._root = Path(self._sessions_root_input).resolve()
        return self._root


__all__ = ["JsonlSessionRepo"]
