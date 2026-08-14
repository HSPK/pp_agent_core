"""Shared fixtures for pi_agent tests.

The helpers themselves live in `agent_helpers.py`: several packages in this
workspace ship a `tests/conftest.py`, and pytest's prepend import mode puts
every test directory on `sys.path`, so a plain `from conftest import ...` in a
multi-package run can resolve to another package's conftest. Importing from a
uniquely named module keeps `uv run pytest packages/pi-agent packages/pi-server`
working.
"""

from __future__ import annotations
