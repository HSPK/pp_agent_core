"""Tests for gitignore matching in skill discovery.

The expected results are git's own, established by running `git check-ignore`
against a real repository with the same `.gitignore`. Getting this wrong is not
cosmetic: a pattern that fails to match directories means skills are discovered
inside `node_modules`, `build` and other excluded trees and injected into the
model's skill set.
"""

from __future__ import annotations

import pytest
from pi_agent.harness.skills import _pattern_to_regex, load_skills


def matches(patterns: list[str], path: str, is_dir: bool) -> bool:
    probe = f"{path}/" if is_dir else path
    return any(regex.match(probe) for regex, _ in (_pattern_to_regex(p) for p in patterns))


# (pattern, path, is_dir, git's answer)
GIT_CASES = [
    # A plain pattern matches both a file and a directory of that name.
    ("build", "build", True, True),
    ("build", "build", False, True),
    ("build", "build/sub/file.md", False, True),
    ("node_modules", "node_modules", True, True),
    ("node_modules", "pkg/node_modules", True, True),
    # A directory-only pattern must NOT match a file of the same name.
    ("logs/", "logs", True, True),
    ("logs/", "logs/file.md", False, True),
    ("logs/", "logs", False, False),
    ("logs/", "other/logs", False, False),
    # Globs.
    ("*.tmp", "x.tmp", False, True),
    ("*.tmp", "dir/x.tmp", False, True),
    ("*.tmp", "x.txt", False, False),
    # "**" collapses to zero segments.
    ("a/**/b", "a/b/file.md", False, True),
    ("a/**/b", "a/x/b/file.md", False, True),
    ("a/**/b", "a/x/y/b/file.md", False, True),
    # Anchored patterns only match at the root.
    ("/root", "root", True, True),
    ("/root", "sub/root", True, False),
    # Unrelated paths are kept.
    ("build", "real/file.md", False, False),
]


@pytest.mark.parametrize(("pattern", "path", "is_dir", "expected"), GIT_CASES)
def test_matches_git_behaviour(pattern, path, is_dir, expected):
    assert matches([pattern], path, is_dir) is expected


def test_plain_pattern_matches_the_directory_probe():
    """Regression: `build` compiled to a regex that could not match `build/`,
    so no directory was ever excluded."""
    regex, directory_only = _pattern_to_regex("build")
    assert directory_only is False
    assert regex.match("build/") is not None
    assert regex.match("build") is not None


def test_directory_only_pattern_rejects_a_file_probe():
    """Regression: `logs/` matched the bare file path `logs`."""
    regex, directory_only = _pattern_to_regex("logs/")
    assert directory_only is True
    assert regex.match("logs/") is not None
    assert regex.match("logs") is None


def test_double_star_collapses_to_zero_segments():
    """Regression: `a/**/b` could not match `a/b`."""
    regex, _ = _pattern_to_regex("a/**/b")
    assert regex.match("a/b") is not None
    assert regex.match("a/x/b") is not None
    assert regex.match("a/x/y/b") is not None


# --------------------------------------------------------------------------
# end-to-end discovery
# --------------------------------------------------------------------------


def write_skill(root, relative: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "---\nname: demo\ndescription: a demo skill\n---\n\nbody\n",
        encoding="utf-8",
    )


async def test_ignored_directories_are_not_scanned_for_skills(tmp_path):
    (tmp_path / ".gitignore").write_text("build\nvendor/\n", encoding="utf-8")
    write_skill(tmp_path, "real/SKILL.md")
    write_skill(tmp_path, "build/sub/SKILL.md")
    write_skill(tmp_path, "vendor/sub/SKILL.md")

    skills, _diagnostics = await load_skills([str(tmp_path)])

    # Only the skill outside the ignored trees is discovered.
    assert len(skills) == 1


async def test_unignored_directories_are_still_scanned(tmp_path):
    (tmp_path / ".gitignore").write_text("build\n", encoding="utf-8")
    write_skill(tmp_path, "a/SKILL.md")
    write_skill(tmp_path, "b/nested/SKILL.md")

    skills, _diagnostics = await load_skills([str(tmp_path)])
    assert len(skills) == 2
