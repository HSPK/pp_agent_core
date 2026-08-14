"""Tests for `pi_agent.harness.result`.

No dedicated TS test file exists for `harness/result.ts`; this suite covers
`ResultNamespace` (the `Result` namespace object), and `TaggedError` /
`match_error` behavior directly.
"""

from __future__ import annotations

import pytest
from pi_agent.harness.result import ResultNamespace, TaggedError, match_error


def test_ok_creates_a_successful_result():
    result = ResultNamespace.ok(42)
    assert result.ok is True
    assert result.value == 42
    assert ResultNamespace.is_ok(result) is True
    assert ResultNamespace.is_err(result) is False


def test_err_creates_a_failed_result():
    result = ResultNamespace.err("boom")
    assert result.ok is False
    assert result.error == "boom"
    assert ResultNamespace.is_ok(result) is False
    assert ResultNamespace.is_err(result) is True


def test_tagged_error_carries_tag_and_props():
    NotFoundError = TaggedError("not_found")
    error = NotFoundError(message="missing", path="/tmp/foo")

    assert error._tag == "not_found"
    assert str(error) == "missing"
    assert error.path == "/tmp/foo"
    assert NotFoundError.is_(error) is True


def test_tagged_error_is_guard_rejects_other_tags():
    NotFoundError = TaggedError("not_found")
    OtherError = TaggedError("other")
    error = OtherError(message="oops")

    assert NotFoundError.is_(error) is False


def test_tagged_error_to_json_includes_tag_message_and_props():
    NotFoundError = TaggedError("not_found")
    error = NotFoundError(message="missing", path="/tmp/foo")

    assert error.to_json() == {"_tag": "not_found", "message": "missing", "path": "/tmp/foo"}


def test_tagged_error_is_raisable_as_an_exception():
    NotFoundError = TaggedError("not_found")

    with pytest.raises(NotFoundError) as excinfo:
        raise NotFoundError(message="missing")
    assert str(excinfo.value) == "missing"


def test_match_error_dispatches_to_the_matching_tag_handler():
    NotFoundError = TaggedError("not_found")

    error = NotFoundError(message="missing")
    result = match_error(
        error,
        {
            "not_found": lambda e: f"not found: {e}",
            "permission_denied": lambda e: f"denied: {e}",
        },
    )

    assert result == "not found: missing"
