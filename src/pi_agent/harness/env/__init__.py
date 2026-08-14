"""Concrete `ExecutionEnv` backends.

Python port of `packages/agent/src/harness/env/`. Only the local backend
exists; see `pi_agent.harness.env.local` for what the Node original does that
this port deliberately drops.
"""

from __future__ import annotations

from .local import (
    EXIT_STDIO_GRACE_SECONDS,
    MAX_TIMEOUT_MS,
    MAX_TIMEOUT_SECONDS,
    LocalExecutionEnv,
    ShellConfig,
    get_shell_config,
    get_shell_env,
    kill_process_tree,
    resolve_path,
    to_file_error,
)

__all__ = [
    "EXIT_STDIO_GRACE_SECONDS",
    "MAX_TIMEOUT_MS",
    "MAX_TIMEOUT_SECONDS",
    "LocalExecutionEnv",
    "ShellConfig",
    "get_shell_config",
    "get_shell_env",
    "kill_process_tree",
    "resolve_path",
    "to_file_error",
]
