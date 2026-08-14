"""The process-wide fallback stream function.

Python port of `packages/agent/src/stream-fn.ts`.

Hosts that own a model runtime install its stream function here so `Agent`
and the low-level loops can call a model without `pi_agent` itself depending
on a provider catalog or compatibility layer.
"""

from __future__ import annotations

from .types import StreamFn

_default_stream_fn: StreamFn | None = None


def set_default_stream_fn(stream_fn: StreamFn | None) -> None:
    """Install (or clear, with `None`) the fallback used when callers omit `stream_fn`."""
    global _default_stream_fn
    _default_stream_fn = stream_fn


def get_default_stream_fn() -> StreamFn:
    """The installed fallback. Raises if no host configured one."""
    if _default_stream_fn is None:
        raise RuntimeError(
            "No default stream function configured. Pass stream_fn explicitly or call set_default_stream_fn()."
        )
    return _default_stream_fn


__all__ = ["get_default_stream_fn", "set_default_stream_fn"]
