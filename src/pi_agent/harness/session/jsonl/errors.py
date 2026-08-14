"""JSONL decode errors.

Python port of `packages/agent/src/harness/session/jsonl/errors.ts`. The
TypeScript version's `fileResult` unwraps the injected `FileSystem`'s
`Result<T, FileError>` return values and converts a failure into a
`SessionError` with code `not_found` or `storage`. This port has no such
abstraction (see `jsonl/types.py`) and raises `OSError` directly, so
`file_error_guard` is the equivalent: a context manager that performs the same
classification on any `OSError` raised inside it.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

from ..types import SessionError


class JsonlDecodeError(Exception):
    def __init__(self, kind: str, message: str, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.kind = kind
        if cause is not None:
            self.__cause__ = cause


@contextlib.contextmanager
def file_error_guard(message: str) -> Iterator[None]:
    """Convert filesystem failures into `SessionError`, mirroring `fileResult`."""
    try:
        yield
    except OSError as error:
        code = "not_found" if isinstance(error, FileNotFoundError) else "storage"
        raise SessionError(code, f"{message}: {error}", error) from error


def invalid_file(path: str, line: int, cause: Exception) -> SessionError:
    return SessionError("invalid_entry", f"Invalid JSONL v4 session {path}: line {line} {cause}", cause)


__all__ = ["JsonlDecodeError", "file_error_guard", "invalid_file"]
