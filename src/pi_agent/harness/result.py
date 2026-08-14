"""Generic result and tagged-error helpers.

Python port of `packages/agent/src/harness/result.ts`.

`TaggedError` is a TypeScript factory function that builds a subclass of
`Error` carrying a `_tag` discriminant and an `is()` type guard, then relies
on structural typing plus `Object.assign` to attach arbitrary extra
properties from the constructor's `props` object. Python has no equivalent to
constructing a fresh subclass at runtime with type-parameterized extra
properties attached via prototype-less assignment; this port instead exposes
`TaggedError(tag)` as a factory that returns a plain `Exception` subclass
whose instances take `message` plus arbitrary keyword properties (mirroring
`Object.assign`) and expose the same `_tag`/`is`/`to_json` behavior.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

TValue = TypeVar("TValue")
TError = TypeVar("TError")


@dataclass
class _Ok(Generic[TValue]):
    value: TValue
    ok: bool = True


@dataclass
class _Err(Generic[TError]):
    error: TError
    ok: bool = False


Result = _Ok[TValue] | _Err[TError]
"""Result of a fallible operation. Expected failures are returned as `ok=False` instead of raised."""


class _ResultNamespace:
    @staticmethod
    def ok(value: TValue) -> Result[TValue, Any]:
        return _Ok(value=value)

    @staticmethod
    def err(error: TError) -> Result[Any, TError]:
        return _Err(error=error)

    @staticmethod
    def is_ok(result: Result[TValue, TError]) -> bool:
        return result.ok

    @staticmethod
    def is_err(result: Result[TValue, TError]) -> bool:
        return not result.ok


ResultNamespace = _ResultNamespace()
"""Namespace object mirroring the TypeScript `Result` value (`Result.ok`, `Result.err`, ...)."""


class TaggedErrorValue(Exception):
    """Base class returned by :func:`TaggedError`. Instances carry a `_tag` plus arbitrary properties."""

    _tag: str = ""

    def __init__(self, **props: Any) -> None:
        message = props.get("message", "")
        super().__init__(message)
        for key, value in props.items():
            setattr(self, key, value)

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key, value in vars(self).items():
            if key != "_tag":
                payload[key] = value
        return {"_tag": self._tag, "message": self.args[0] if self.args else "", **payload}


def TaggedError(tag: str) -> type[TaggedErrorValue]:
    """Build a `TaggedErrorValue` subclass carrying the given `_tag`."""

    class _TaggedErrorClass(TaggedErrorValue):
        _tag = tag

        @classmethod
        def is_(cls, value: object) -> bool:
            return isinstance(value, _TaggedErrorClass)

    _TaggedErrorClass.__name__ = tag
    _TaggedErrorClass.__qualname__ = tag
    return _TaggedErrorClass


ErrorMatchers = dict[str, Callable[[TaggedErrorValue], Any]]


def match_error(error: TaggedErrorValue, matchers: ErrorMatchers) -> Any:
    return matchers[error._tag](error)


__all__ = [
    "ErrorMatchers",
    "Result",
    "ResultNamespace",
    "TaggedError",
    "TaggedErrorValue",
    "match_error",
]
