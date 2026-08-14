"""Harness-level result, error, filesystem, and shell type contracts.

Python port of `packages/agent/src/harness/types.ts`.

`FileSystem`/`Shell`/`ExecutionEnv` are ported as `Protocol` classes: they are
pure capability interfaces with no TypeScript-side implementation in this
file (the concrete Node implementation lives in `harness/env/nodejs.ts`,
which is Node-specific and out of scope for this port; see
`pi_agent.harness.skills`/`pi_agent.harness.prompt_templates` for how the
higher-level modules that used to depend on `ExecutionEnv` are ported
directly against `pathlib` instead of an injected environment).

`Static`/`TSchema` (typebox runtime schema types) have no Python port: this
file's `AgentHarnessTool` therefore drops the `TParameters`/`Static<TParameters>`
generic parameterization from `AgentToolParameters extends TSchema` and works
with `dict[str, Any]`-shaped tool arguments directly, consistent with how
`pi_agent.types.AgentTool` already models tool parameters.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, Literal, Protocol, TypeVar

from pi_ai.types import CacheRetention, Transport
from pi_ai.utils.abort import AbortSignal

from ..types import AgentTool, AgentToolResult, AgentToolUpdateCallback

TValue = TypeVar("TValue")
TError = TypeVar("TError")


@dataclass
class _Ok(Generic[TValue]):
    value: TValue
    ok: Literal[True] = True


@dataclass
class _Err(Generic[TError]):
    error: TError
    ok: Literal[False] = False


Result = _Ok[TValue] | _Err[TError]
"""Result of a fallible operation. Expected failures are returned as `ok=False` instead of raised."""


def ok(value: TValue) -> Result[TValue, Any]:
    """Create a successful :data:`Result`."""
    return _Ok(value=value)


def err(error: TError) -> Result[Any, TError]:
    """Create a failed :data:`Result`."""
    return _Err(error=error)


def get_or_throw(result: Result[TValue, BaseException]) -> TValue:
    """Return the success value or raise the failure error.

    Intended for tests and explicit adapter boundaries.
    """
    if not result.ok:
        raise result.error
    return result.value


def get_or_undefined(result: Result[TValue, Any]) -> TValue | None:
    """Return the success value or `None`."""
    return result.value if result.ok else None


def to_error(error: object) -> Exception:
    """Normalize an unknown raised value into an `Exception` before using it as a typed error cause."""
    if isinstance(error, Exception):
        return error
    if isinstance(error, str):
        return Exception(error)
    try:
        import json

        return Exception(json.dumps(error))
    except (TypeError, ValueError):
        return Exception(str(error))


@dataclass(kw_only=True)
class Skill:
    """Skill loaded from a `SKILL.md` file or provided by an application.

    `name`, `description`, and `file_path` are inserted into the system
    prompt in an XML-formatted block as suggested by agentskills.io. Use
    `format_skills_for_system_prompt` to generate the spec-compatible system
    prompt block.
    """

    name: str
    """Stable skill name used for lookup and model-visible listings."""
    description: str
    """Short model-visible description of when to use the skill."""
    content: str
    """Full skill instructions."""
    file_path: str
    """Absolute path to the skill file. Used for model-visible location and resolving relative references."""
    disable_model_invocation: bool = False
    """Exclude this skill from model-visible skill lists while still allowing explicit application invocation."""


@dataclass(kw_only=True)
class PromptTemplate:
    """Prompt template that can be formatted into a prompt for explicit invocation."""

    name: str
    """Stable template name used for lookup or application command routing."""
    content: str
    """Template content. Argument placeholders are formatted by `format_prompt_template_invocation`."""
    description: str | None = None
    """Optional description for command lists or autocomplete."""


@dataclass(kw_only=True)
class AgentHarnessResources:
    """Resources made available to explicit invocation methods and system-prompt callbacks."""

    prompt_templates: list[PromptTemplate] = field(default_factory=list)
    """Prompt templates available for explicit invocation."""
    skills: list[Skill] = field(default_factory=list)
    """Skills available to the model and explicit skill invocation."""


TContext = TypeVar("TContext")


@dataclass(kw_only=True)
class AgentHarnessTool(AgentTool, Generic[TContext]):
    """Tool definition executed by an agent harness with an application-defined context."""

    execute_with_context: (
        Callable[
            [str, dict[str, Any], AbortSignal | None, AgentToolUpdateCallback | None, TContext],
            Awaitable[AgentToolResult],
        ]
        | None
    ) = None
    """Execute the tool call with the context resolved for the current turn snapshot."""


AgentHarnessToolContextSource = TContext | Callable[[], TContext | Awaitable[TContext]]
"""Static tool context or zero-argument provider resolved for each turn snapshot."""


@dataclass(kw_only=True)
class AgentHarnessStreamOptions:
    """Curated provider request options owned by the harness and snapshotted per turn."""

    transport: Transport | None = None
    """Preferred transport forwarded to the stream function."""
    timeout_ms: float | None = None
    """Provider request timeout in milliseconds."""
    max_retries: int | None = None
    """Maximum provider retry attempts."""
    max_retry_delay_ms: float | None = None
    """Optional cap for provider-requested retry delays."""
    headers: dict[str, str] = field(default_factory=dict)
    """Additional request headers merged with auth and lifecycle headers."""
    metadata: dict[str, Any] | None = None
    """Provider metadata forwarded with requests."""
    cache_retention: CacheRetention | None = None
    """Provider cache retention hint."""


@dataclass(kw_only=True)
class AgentHarnessStreamOptionsPatch:
    """Per-request stream option patch returned by provider hooks."""

    transport: Transport | None = None
    timeout_ms: float | None = None
    max_retries: int | None = None
    max_retry_delay_ms: float | None = None
    headers: dict[str, str | None] | None = None
    """Header patch. `None` values delete keys; explicit `headers=None` clears all headers."""
    metadata: dict[str, Any | None] | None = None
    """Metadata patch. `None` values delete keys; explicit `metadata=None` clears all metadata."""
    cache_retention: CacheRetention | None = None


FileKind = Literal["file", "directory", "symlink"]
"""Kind of filesystem object as addressed by a `FileSystem`. Symlinks are not followed automatically."""

FileErrorCode = Literal[
    "aborted",
    "not_found",
    "permission_denied",
    "not_directory",
    "is_directory",
    "invalid",
    "not_supported",
    "unknown",
]
"""Stable, backend-independent file error codes returned by `FileSystem` file operations."""


class FileError(Exception):
    """Error returned by `FileSystem` file operations."""

    def __init__(
        self, code: FileErrorCode, message: str, path: str | None = None, cause: Exception | None = None
    ) -> None:
        super().__init__(message)
        self.code = code
        self.path = path
        if cause is not None:
            self.__cause__ = cause


ExecutionErrorCode = Literal["aborted", "timeout", "shell_unavailable", "spawn_error", "callback_error", "unknown"]
"""Stable, backend-independent execution error codes returned by `ExecutionEnv.exec`."""


class ExecutionError(Exception):
    """Error returned by `ExecutionEnv.exec`."""

    def __init__(self, code: ExecutionErrorCode, message: str, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.code = code
        if cause is not None:
            self.__cause__ = cause


CompactionErrorCode = Literal["aborted", "summarization_failed"]
"""Stable compaction error codes returned by compaction helpers."""


class CompactionError(Exception):
    """Error returned by compaction helpers."""

    def __init__(self, code: CompactionErrorCode, message: str, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.code = code
        if cause is not None:
            self.__cause__ = cause


BranchSummaryErrorCode = Literal["aborted", "summarization_failed"]
"""Stable branch-summary error codes returned by branch summarization helpers."""


class BranchSummaryError(Exception):
    """Error returned by branch summarization helpers."""

    def __init__(self, code: BranchSummaryErrorCode, message: str, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.code = code
        if cause is not None:
            self.__cause__ = cause


@dataclass(kw_only=True)
class FileInfo:
    """Metadata for one filesystem object in a `FileSystem`."""

    name: str
    """Basename of `path`."""
    path: str
    """Absolute, syntactically normalized addressed path in the execution environment. Symlinks are not followed."""
    kind: FileKind
    """Object kind. Symlink targets are not followed; use `FileSystem.canonical_path` explicitly."""
    size: int
    """Size in bytes for the addressed filesystem object."""
    mtime_ms: int
    """Modification time as milliseconds since Unix epoch."""


class FileSystem(Protocol):
    """Filesystem capability used by the harness.

    Paths passed to methods may be absolute or relative to `cwd`. Paths
    returned by file operations are addressed paths in the filesystem
    namespace, but are not canonicalized through symlinks unless returned by
    `canonical_path`.

    Operation methods must never raise. All filesystem failures, including
    unexpected backend failures, must be encoded in the returned `Result`.
    Implementations must preserve this invariant.
    """

    cwd: str
    """Current working directory for relative paths."""

    async def absolute_path(self, path: str, abort_signal: AbortSignal | None = None) -> Result[str, FileError]: ...
    async def join_path(self, parts: list[str], abort_signal: AbortSignal | None = None) -> Result[str, FileError]: ...
    async def read_text_file(self, path: str, abort_signal: AbortSignal | None = None) -> Result[str, FileError]: ...
    async def read_text_lines(
        self, path: str, max_lines: int | None = None, abort_signal: AbortSignal | None = None
    ) -> Result[list[str], FileError]: ...
    async def read_binary_file(
        self, path: str, abort_signal: AbortSignal | None = None
    ) -> Result[bytes, FileError]: ...
    async def write_file(
        self, path: str, content: str | bytes, abort_signal: AbortSignal | None = None
    ) -> Result[None, FileError]: ...
    async def append_file(
        self, path: str, content: str | bytes, abort_signal: AbortSignal | None = None
    ) -> Result[None, FileError]: ...
    async def rename_file(
        self, source_path: str, destination_path: str, abort_signal: AbortSignal | None = None
    ) -> Result[None, FileError]: ...
    async def file_info(self, path: str, abort_signal: AbortSignal | None = None) -> Result[FileInfo, FileError]: ...
    async def list_dir(
        self, path: str, abort_signal: AbortSignal | None = None
    ) -> Result[list[FileInfo], FileError]: ...
    async def canonical_path(self, path: str, abort_signal: AbortSignal | None = None) -> Result[str, FileError]: ...
    async def exists(self, path: str, abort_signal: AbortSignal | None = None) -> Result[bool, FileError]: ...
    async def create_dir(
        self, path: str, recursive: bool = True, abort_signal: AbortSignal | None = None
    ) -> Result[None, FileError]: ...
    async def remove(
        self,
        path: str,
        recursive: bool = False,
        force: bool = False,
        abort_signal: AbortSignal | None = None,
    ) -> Result[None, FileError]: ...
    async def create_temp_dir(
        self, prefix: str = "tmp-", abort_signal: AbortSignal | None = None
    ) -> Result[str, FileError]: ...
    async def create_temp_file(
        self, prefix: str = "", suffix: str = "", abort_signal: AbortSignal | None = None
    ) -> Result[str, FileError]: ...
    async def cleanup(self) -> None: ...


@dataclass(kw_only=True)
class ShellExecOptions:
    """Options for `Shell.exec`."""

    cwd: str | None = None
    """Working directory for the command. Relative paths are resolved against `ExecutionEnv.cwd`. Defaults to `cwd`."""
    env: dict[str, str] | None = None
    """Environment variables for the command. Values override inherited defaults when `inherit_env` is true."""
    inherit_env: bool = True
    """Whether to inherit the execution environment's default variables."""
    timeout: float | None = None
    """Timeout in seconds. Implementations should return a timeout error when the command exceeds this duration."""
    abort_signal: AbortSignal | None = None
    """Abort signal used to terminate the command."""
    on_stdout: Callable[[str], None] | None = None
    """Called with stdout chunks as they are produced."""
    on_stderr: Callable[[str], None] | None = None
    """Called with stderr chunks as they are produced."""


@dataclass(kw_only=True)
class ShellExecResult:
    stdout: str
    stderr: str
    exit_code: int


class Shell(Protocol):
    """Shell execution capability used by the harness."""

    async def exec(
        self, command: str, options: ShellExecOptions | None = None
    ) -> Result[ShellExecResult, ExecutionError]:
        """Execute a shell command in `FileSystem.cwd` unless `options.cwd` is provided."""
        ...

    async def cleanup(self) -> None:
        """Release shell resources. Must be best-effort and must not raise."""
        ...


class ExecutionEnv(FileSystem, Shell, Protocol):
    """Filesystem and process execution environment used by the harness."""


__all__: Sequence[str] = [
    "AgentHarnessResources",
    "AgentHarnessStreamOptions",
    "AgentHarnessStreamOptionsPatch",
    "AgentHarnessTool",
    "AgentHarnessToolContextSource",
    "BranchSummaryError",
    "BranchSummaryErrorCode",
    "CompactionError",
    "CompactionErrorCode",
    "ExecutionEnv",
    "ExecutionError",
    "ExecutionErrorCode",
    "FileError",
    "FileErrorCode",
    "FileInfo",
    "FileKind",
    "FileSystem",
    "PromptTemplate",
    "Result",
    "Shell",
    "ShellExecOptions",
    "ShellExecResult",
    "Skill",
    "err",
    "get_or_throw",
    "get_or_undefined",
    "ok",
    "to_error",
]
