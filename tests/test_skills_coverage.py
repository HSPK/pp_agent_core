"""Additional coverage tests for `pi_agent.harness.skills`.

Targets uncovered lines: 102-104, 107, 120, 161-166, 183-185, 187, 223,
229-231, 237, 240, 253, 269->271, 286-288, 294->280, 303, 308-309, 311, 313,
322-324, 328-330, 362, 364, 366, 368, 377, 384, 387, 394-398, 408.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from pi_agent.harness.skills import (
    _add_ignore_rules,
    _dirname_env_path,
    _IgnoreMatcher,
    _load_skill_from_file,
    _load_skills_from_dir,
    _pattern_to_regex,
    _prefix_ignore_pattern,
    _relative_env_path,
    _validate_description,
    _validate_name,
    format_skill_invocation,
    load_skills,
    load_sourced_skills,
)
from pi_agent.harness.types import Skill


def make_skill_dir(tmp_path: Path, name: str, description: str = "A skill", content: str = "Skill content") -> Path:
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {description}\n---\n{content}")
    return d


# --------------------------------------------------------------------------
# load_skills — error and edge cases (lines 102-107, 120)
# --------------------------------------------------------------------------


async def test_load_skills_skips_nonexistent_dir(tmp_path: Path):
    skills, diagnostics = await load_skills(tmp_path / "does_not_exist")
    assert skills == []
    assert diagnostics == []


async def test_load_skills_file_info_failed(tmp_path: Path):
    with patch.object(Path, "exists", side_effect=OSError("stat failed")):
        skills, diagnostics = await load_skills(tmp_path / "dir")
    assert skills == []
    assert any(d.code == "file_info_failed" for d in diagnostics)


async def test_load_skills_skips_plain_file_input(tmp_path: Path):
    f = tmp_path / "file.txt"
    f.write_text("not a dir")
    skills, diagnostics = await load_skills(f)
    assert skills == []
    assert diagnostics == []


async def test_load_skills_multiple_dirs(tmp_path: Path):
    make_skill_dir(tmp_path / "a", "my-skill")
    make_skill_dir(tmp_path / "b", "other-skill")
    skills, diagnostics = await load_skills([tmp_path / "a", tmp_path / "b"])
    assert diagnostics == []
    assert {s.name for s in skills} == {"my-skill", "other-skill"}


# --------------------------------------------------------------------------
# _load_skills_from_dir — list_failed (lines 161-166)
# --------------------------------------------------------------------------


def test_load_skills_from_dir_list_failed(tmp_path: Path):
    with patch.object(Path, "iterdir", side_effect=OSError("list failed")):
        skills, diagnostics = _load_skills_from_dir(tmp_path, True, _IgnoreMatcher(), tmp_path)
    assert skills == []
    assert any(d.code == "list_failed" for d in diagnostics)


# --------------------------------------------------------------------------
# _load_skills_from_dir — SKILL.md is not a file (lines 183-185, 187)
# --------------------------------------------------------------------------


def test_load_skills_from_dir_skill_md_is_directory(tmp_path: Path):
    skill_dir = tmp_path / "skill-name"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").mkdir()  # SKILL.md is a directory
    skills, _diagnostics = _load_skills_from_dir(tmp_path, True, _IgnoreMatcher(), tmp_path)
    assert skills == []


# --------------------------------------------------------------------------
# _load_skills_from_dir — ignored entries (line 223, 229-231, 237, 240)
# --------------------------------------------------------------------------


async def test_skills_respects_gitignore_patterns(tmp_path: Path):
    (tmp_path / ".gitignore").write_text("ignored/\n")
    ignored_dir = tmp_path / "ignored"
    ignored_dir.mkdir()
    (ignored_dir / "SKILL.md").write_text("---\nname: ignored\ndescription: Ignored\n---\nContent")
    visible_dir = tmp_path / "visible"
    visible_dir.mkdir()
    (visible_dir / "SKILL.md").write_text("---\nname: visible\ndescription: Visible\n---\nContent")

    skills, _diagnostics = await load_skills(tmp_path)
    assert [s.name for s in skills] == ["visible"]


async def test_skills_skips_hidden_dirs(tmp_path: Path):
    hidden = tmp_path / ".hidden"
    hidden.mkdir()
    (hidden / "SKILL.md").write_text("---\nname: hidden\ndescription: Hidden\n---\nContent")
    skills, _ = await load_skills(tmp_path)
    assert skills == []


async def test_skills_skips_node_modules(tmp_path: Path):
    nm = tmp_path / "node_modules"
    nm.mkdir()
    (nm / "SKILL.md").write_text("---\nname: node-modules\ndescription: NM\n---\nContent")
    skills, _ = await load_skills(tmp_path)
    assert skills == []


async def test_skills_skips_ignored_file(tmp_path: Path):
    (tmp_path / ".gitignore").write_text("skip.md\n")
    (tmp_path / "skip.md").write_text("---\ndescription: Skipped\n---\nContent")
    skills, _ = await load_skills(tmp_path)
    assert skills == []


# --------------------------------------------------------------------------
# _load_skill_from_file — error paths (lines 253, 269->271, 286-288)
# --------------------------------------------------------------------------


def test_load_skill_from_file_read_failed(tmp_path: Path):
    p = tmp_path / "SKILL.md"
    p.write_text("content")
    with patch.object(Path, "read_text", side_effect=OSError("read failed")):
        skill, diagnostics = _load_skill_from_file(p, "parent")
    assert skill is None
    assert diagnostics[0].code == "read_failed"


def test_load_skill_from_file_parse_failed(tmp_path: Path):
    p = tmp_path / "SKILL.md"
    p.write_text("---\ndescription: [unterminated\n---\nBody")
    skill, diagnostics = _load_skill_from_file(p, "parent")
    assert skill is None
    assert diagnostics[0].code == "parse_failed"


def test_load_skill_from_file_non_string_description(tmp_path: Path):
    p = tmp_path / "SKILL.md"
    p.write_text("---\ndescription: 12345\n---\nBody")
    skill, diagnostics = _load_skill_from_file(p, "parent")
    assert skill is None
    assert any(d.code == "invalid_metadata" for d in diagnostics)


def test_load_skill_from_file_non_string_name_uses_parent(tmp_path: Path):
    # name: 123 is an integer, not a string — frontmatter_name falls back to None,
    # so the skill name becomes parent_dir_name which passes _validate_name.
    p = tmp_path / "SKILL.md"
    p.write_text("---\nname: 123\ndescription: Valid desc\n---\nBody")
    skill, _diagnostics = _load_skill_from_file(p, "parent")
    # No name mismatch because non-string name is ignored; "parent" matches "parent"
    assert skill is not None
    assert skill.name == "parent"


def test_load_skill_from_file_disable_model_invocation_true(tmp_path: Path):
    p = tmp_path / "SKILL.md"
    p.write_text("---\nname: parent\ndescription: Valid\ndisable-model-invocation: true\n---\nBody")
    skill, _diagnostics = _load_skill_from_file(p, "parent")
    assert skill is not None
    assert skill.disable_model_invocation is True


def test_load_skill_from_file_empty_description_returns_none(tmp_path: Path):
    p = tmp_path / "SKILL.md"
    p.write_text("---\nname: parent\ndescription: \n---\nBody")
    skill, diagnostics = _load_skill_from_file(p, "parent")
    assert skill is None
    assert any("description is required" in d.message for d in diagnostics)


# --------------------------------------------------------------------------
# _validate_name (lines 294->280, 303, 308-309, 311, 313)
# --------------------------------------------------------------------------


def test_validate_name_mismatch():
    errors = _validate_name("other", "parent")
    assert any("does not match" in e for e in errors)


def test_validate_name_too_long():
    long_name = "a" * 65
    errors = _validate_name(long_name, long_name)
    assert any("exceeds" in e for e in errors)


def test_validate_name_invalid_chars():
    errors = _validate_name("Bad_Name", "Bad_Name")
    assert any("invalid characters" in e for e in errors)


def test_validate_name_starts_with_hyphen():
    errors = _validate_name("-name", "-name")
    assert any("must not start" in e for e in errors)


def test_validate_name_ends_with_hyphen():
    errors = _validate_name("name-", "name-")
    assert any("must not start or end" in e for e in errors)


def test_validate_name_consecutive_hyphens():
    errors = _validate_name("na--me", "na--me")
    assert any("consecutive" in e for e in errors)


def test_validate_name_valid():
    errors = _validate_name("valid-name", "valid-name")
    assert errors == []


# --------------------------------------------------------------------------
# _validate_description (lines 322-324, 328-330)
# --------------------------------------------------------------------------


def test_validate_description_none():
    errors = _validate_description(None)
    assert errors == ["description is required"]


def test_validate_description_empty():
    errors = _validate_description("")
    assert errors == ["description is required"]


def test_validate_description_whitespace_only():
    errors = _validate_description("   ")
    assert errors == ["description is required"]


def test_validate_description_too_long():
    errors = _validate_description("x" * 1025)
    assert any("exceeds" in e for e in errors)


def test_validate_description_valid():
    errors = _validate_description("A valid description")
    assert errors == []


# --------------------------------------------------------------------------
# format_skill_invocation (lines 362, 364, 366, 368, 377, 384, 387)
# --------------------------------------------------------------------------


def test_format_skill_invocation_without_additional_instructions():
    skill = Skill(
        name="my-skill",
        description="desc",
        content="Do X.",
        file_path="/path/to/SKILL.md",
        disable_model_invocation=False,
    )
    result = format_skill_invocation(skill)
    assert '<skill name="my-skill"' in result
    assert "Do X." in result
    assert "additional" not in result.lower() or "additional_instructions" not in result


def test_format_skill_invocation_with_additional_instructions():
    skill = Skill(
        name="my-skill",
        description="desc",
        content="Do X.",
        file_path="/path/to/SKILL.md",
        disable_model_invocation=False,
    )
    result = format_skill_invocation(skill, "Also do Y.")
    assert "Also do Y." in result


# --------------------------------------------------------------------------
# _dirname_env_path (lines 394-398, 408)
# --------------------------------------------------------------------------


def test_dirname_env_path_unix():
    assert _dirname_env_path("/a/b/c") == "/a/b"


def test_dirname_env_path_root():
    assert _dirname_env_path("/file") == "/"


def test_dirname_env_path_windows_drive():
    assert _dirname_env_path("C:\\file.txt") == "C:\\"


def test_dirname_env_path_no_separator():
    assert _dirname_env_path("file") == "/"


def test_dirname_env_path_trailing_slash_stripped():
    assert _dirname_env_path("/a/b/") == "/a"


# --------------------------------------------------------------------------
# _relative_env_path
# --------------------------------------------------------------------------


def test_relative_env_path_same():
    root = Path("/a/b")
    assert _relative_env_path(root, root) == ""


def test_relative_env_path_child():
    root = Path("/a/b")
    child = Path("/a/b/c/d")
    assert _relative_env_path(root, child) == "c/d"


def test_relative_env_path_outside_root():
    root = Path("/a/b")
    outside = Path("/c/d")
    result = _relative_env_path(root, outside)
    assert "c/d" in result


# --------------------------------------------------------------------------
# _pattern_to_regex edge cases
# --------------------------------------------------------------------------


def test_pattern_to_regex_directory_only():
    regex, dir_only = _pattern_to_regex("logs/")
    assert dir_only is True
    assert regex.match("logs/")
    assert not regex.match("logs")  # file probe must not match


def test_pattern_to_regex_anchored():
    regex, _ = _pattern_to_regex("/build")
    assert regex.match("build")
    assert regex.match("build/anything")


def test_pattern_to_regex_double_star():
    regex, _ = _pattern_to_regex("a/**/b")
    assert regex.match("a/b")
    assert regex.match("a/x/y/b")


def test_pattern_to_regex_question_mark():
    regex, _ = _pattern_to_regex("file?.md")
    assert regex.match("fileA.md")
    assert not regex.match("file/b.md")


# --------------------------------------------------------------------------
# _IgnoreMatcher — negation
# --------------------------------------------------------------------------


def test_ignore_matcher_negation():
    ig = _IgnoreMatcher()
    ig.add(["*.log", "!important.log"])
    assert ig.ignores("debug.log")
    assert not ig.ignores("important.log")


def test_ignore_matcher_empty():
    ig = _IgnoreMatcher()
    assert not ig.ignores("anything")


# --------------------------------------------------------------------------
# _prefix_ignore_pattern
# --------------------------------------------------------------------------


def test_prefix_ignore_pattern_blank_returns_none():
    assert _prefix_ignore_pattern("", "") is None
    assert _prefix_ignore_pattern("   ", "") is None


def test_prefix_ignore_pattern_comment_returns_none():
    assert _prefix_ignore_pattern("# comment", "") is None


def test_prefix_ignore_pattern_escaped_comment_kept():
    # A line starting with \# is NOT treated as a comment; the backslash is preserved.
    result = _prefix_ignore_pattern("\\#not-comment", "")
    assert result == "\\#not-comment"


def test_prefix_ignore_pattern_negated():
    result = _prefix_ignore_pattern("!file.md", "dir/")
    assert result == "!dir/file.md"


def test_prefix_ignore_pattern_leading_slash_stripped():
    result = _prefix_ignore_pattern("/file.md", "dir/")
    assert result == "dir/file.md"


def test_prefix_ignore_pattern_escaped_exclamation():
    result = _prefix_ignore_pattern("\\!file.md", "")
    assert result == "!file.md"


# --------------------------------------------------------------------------
# _add_ignore_rules — read failure creates diagnostic
# --------------------------------------------------------------------------


def test_add_ignore_rules_read_failure(tmp_path: Path):
    gi = tmp_path / ".gitignore"
    gi.write_text("*.log")
    ig = _IgnoreMatcher()
    diagnostics = []
    with patch.object(Path, "read_text", side_effect=OSError("cannot read")):
        _add_ignore_rules(ig, tmp_path, tmp_path, diagnostics)
    assert any(d.code == "read_failed" for d in diagnostics)


# --------------------------------------------------------------------------
# load_sourced_skills — map_skill callback
# --------------------------------------------------------------------------


async def test_load_sourced_skills_applies_map(tmp_path: Path):
    make_skill_dir(tmp_path, "my-skill")

    def mapper(skill, source):
        return Skill(
            name=skill.name,
            description=skill.description + " (mapped)",
            content=skill.content,
            file_path=skill.file_path,
            disable_model_invocation=skill.disable_model_invocation,
        )

    skills, diagnostics = await load_sourced_skills([(tmp_path, "src")], mapper)
    assert diagnostics == []
    assert len(skills) == 1
    assert "(mapped)" in skills[0].skill.description
    assert skills[0].source == "src"
