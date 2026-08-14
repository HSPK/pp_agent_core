"""Additional coverage tests for `pi_agent.harness.prompt_templates`.

Targets the branches and statements not covered by test_harness_prompt_templates.py:
lines 82-84, 86, 89-91, 96->79, 131-133, 139->141, 149-151, 164, 180, 188-208, 224, 227.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from pi_agent.harness.prompt_templates import (
    _load_template_from_file,
    _load_templates_from_dir,
    _normalize_paths,
    _parse_frontmatter,
    format_prompt_template_invocation,
    load_prompt_templates,
    load_sourced_prompt_templates,
    parse_command_args,
    substitute_args,
)
from pi_agent.harness.types import PromptTemplate

# --------------------------------------------------------------------------
# _normalize_paths
# --------------------------------------------------------------------------


def test_normalize_paths_single_string():
    result = _normalize_paths("some/path")
    assert result == [Path("some/path")]


def test_normalize_paths_single_path():
    result = _normalize_paths(Path("/tmp/x"))
    assert result == [Path("/tmp/x")]


def test_normalize_paths_list():
    result = _normalize_paths(["a", "b"])
    assert result == [Path("a"), Path("b")]


# --------------------------------------------------------------------------
# load_prompt_templates — error paths (lines 82-91, 96->79)
# --------------------------------------------------------------------------


async def test_load_prompt_templates_file_info_failed_on_exists(tmp_path: Path):
    """OSError on path.exists() produces a file_info_failed diagnostic."""
    p = tmp_path / "x.md"
    with patch.object(Path, "exists", side_effect=OSError("stat failed")):
        templates, diagnostics = await load_prompt_templates(p)
    assert templates == []
    assert len(diagnostics) == 1
    assert diagnostics[0].code == "file_info_failed"
    assert "stat failed" in diagnostics[0].message


async def test_load_prompt_templates_file_info_failed_on_is_dir(tmp_path: Path):
    """OSError on path.is_dir() after exists() produces a file_info_failed diagnostic."""
    p = tmp_path / "dir"
    p.mkdir()

    def patched_is_dir(self):
        raise OSError("is_dir failed")

    with patch.object(Path, "is_dir", patched_is_dir):
        templates, diagnostics = await load_prompt_templates(p)
    assert templates == []
    assert any(d.code == "file_info_failed" for d in diagnostics)


async def test_load_prompt_templates_nonexistent_path_is_skipped(tmp_path: Path):
    templates, diagnostics = await load_prompt_templates(tmp_path / "does_not_exist.md")
    assert templates == []
    assert diagnostics == []


async def test_load_prompt_templates_non_md_file_is_skipped(tmp_path: Path):
    f = tmp_path / "file.txt"
    f.write_text("ignored")
    templates, diagnostics = await load_prompt_templates(f)
    assert templates == []
    assert diagnostics == []


# --------------------------------------------------------------------------
# _load_templates_from_dir — OSError on iterdir (lines 131-133)
# --------------------------------------------------------------------------


def test_load_templates_from_dir_list_failed(tmp_path: Path):
    d = tmp_path / "dir"
    d.mkdir()
    with patch.object(Path, "iterdir", side_effect=OSError("list failed")):
        templates, diagnostics = _load_templates_from_dir(d)
    assert templates == []
    assert len(diagnostics) == 1
    assert diagnostics[0].code == "list_failed"


def test_load_templates_from_dir_skips_non_md_files(tmp_path: Path):
    (tmp_path / "ignore.txt").write_text("ignored")
    (tmp_path / "valid.md").write_text("---\ndescription: Valid\n---\nBody")
    templates, diagnostics = _load_templates_from_dir(tmp_path)
    assert [t.name for t in templates] == ["valid"]
    assert diagnostics == []


def test_load_templates_from_dir_skips_non_file_entries(tmp_path: Path):
    (tmp_path / "subdir").mkdir()
    templates, diagnostics = _load_templates_from_dir(tmp_path)
    assert templates == []
    assert diagnostics == []


# --------------------------------------------------------------------------
# _load_template_from_file — error paths (lines 139->141, 149-151, 164, 180)
# --------------------------------------------------------------------------


def test_load_template_from_file_read_failed(tmp_path: Path):
    p = tmp_path / "a.md"
    p.write_text("body")
    with patch.object(Path, "read_text", side_effect=OSError("read failed")):
        template, diagnostics = _load_template_from_file(p)
    assert template is None
    assert diagnostics[0].code == "read_failed"


def test_load_template_from_file_parse_failed_yaml_error(tmp_path: Path):
    p = tmp_path / "bad.md"
    p.write_text("---\ndescription: [unterminated\n---\nBody")
    template, diagnostics = _load_template_from_file(p)
    assert template is None
    assert diagnostics[0].code == "parse_failed"


def test_load_template_from_file_description_from_first_line(tmp_path: Path):
    p = tmp_path / "nodesc.md"
    p.write_text("The quick brown fox jumps over the lazy dog\nMore body")
    template, diagnostics = _load_template_from_file(p)
    assert diagnostics == []
    assert template is not None
    assert template.description == "The quick brown fox jumps over the lazy dog"


def test_load_template_from_file_long_first_line_truncated(tmp_path: Path):
    long_line = "A" * 80
    p = tmp_path / "long.md"
    p.write_text(long_line)
    template, _diagnostics = _load_template_from_file(p)
    assert template is not None
    assert template.description == "A" * 60 + "..."


def test_load_template_from_file_empty_body_no_first_line(tmp_path: Path):
    p = tmp_path / "empty.md"
    p.write_text("---\ndescription: Explicit\n---\n\n")
    template, _diagnostics = _load_template_from_file(p)
    assert template is not None
    assert template.description == "Explicit"


def test_load_template_from_file_no_description_and_blank_body(tmp_path: Path):
    p = tmp_path / "blank.md"
    p.write_text("\n\n\n")
    template, _diagnostics = _load_template_from_file(p)
    assert template is not None
    assert template.description == ""


# --------------------------------------------------------------------------
# _parse_frontmatter edge cases (lines 188-208)
# --------------------------------------------------------------------------


def test_parse_frontmatter_no_frontmatter():
    fm, body = _parse_frontmatter("Hello world")
    assert fm == {}
    assert body == "Hello world"


def test_parse_frontmatter_unterminated_returns_raw():
    content = "---\nkey: value\nno closing"
    fm, body = _parse_frontmatter(content)
    assert fm == {}
    assert body == content


def test_parse_frontmatter_with_valid_yaml():
    content = "---\ndescription: Hi\nauthor: alice\n---\nBody text"
    fm, body = _parse_frontmatter(content)
    assert fm == {"description": "Hi", "author": "alice"}
    assert body == "Body text"


def test_parse_frontmatter_empty_yaml_block():
    content = "---\n---\nBody"
    fm, body = _parse_frontmatter(content)
    assert fm == {}
    assert body == "Body"


def test_parse_frontmatter_strips_body_whitespace():
    content = "---\ndesc: x\n---\n\n  Body\n"
    _fm, body = _parse_frontmatter(content)
    assert body == "Body"


def test_parse_frontmatter_crlf_normalized():
    content = "---\r\ndescription: Test\r\n---\r\nBody"
    fm, body = _parse_frontmatter(content)
    assert fm == {"description": "Test"}
    assert body == "Body"


# --------------------------------------------------------------------------
# parse_command_args edge cases (lines 224, 227)
# --------------------------------------------------------------------------


def test_parse_command_args_single_quotes():
    result = parse_command_args("'hello world' foo")
    assert result == ["hello world", "foo"]


def test_parse_command_args_double_quotes():
    result = parse_command_args('"hello world" foo')
    assert result == ["hello world", "foo"]


def test_parse_command_args_mixed_quotes():
    parse_command_args("'it\\'s' \"nice\"")
    # single quote closes on the second quote, not on backslash-escape
    # simple parser: in_quote='\'', sees \\, adds backslash, sees ', then adds
    # that closes the quote...actually the parser sees \ as plain char
    result2 = parse_command_args("'abc' \"def ghi\"")
    assert result2 == ["abc", "def ghi"]


def test_parse_command_args_tab_as_separator():
    result = parse_command_args("a\tb\tc")
    assert result == ["a", "b", "c"]


def test_parse_command_args_empty_string():
    result = parse_command_args("")
    assert result == []


def test_parse_command_args_trailing_token():
    result = parse_command_args("hello")
    assert result == ["hello"]


# --------------------------------------------------------------------------
# substitute_args edge cases
# --------------------------------------------------------------------------


def test_substitute_args_positional_out_of_range():
    result = substitute_args("$1 $2 $3", ["only_one"])
    assert result == "only_one  "


def test_substitute_args_dollar_at_all_args():
    result = substitute_args("$@ $ARGUMENTS", ["a", "b"])
    assert result == "a b a b"


def test_substitute_args_slice_with_length():
    result = substitute_args("${@:2:2}", ["a", "b", "c", "d"])
    assert result == "b c"


def test_substitute_args_slice_negative_start_clamped():
    # ${@:0} with 0 -> start=0-1=-1 -> clamped to 0
    result = substitute_args("${@:1}", ["x", "y"])
    assert result == "x y"


def test_substitute_args_no_args():
    result = substitute_args("$1 $@ $ARGUMENTS", [])
    assert result == "  "


# --------------------------------------------------------------------------
# load_sourced_prompt_templates — map_prompt_template callback
# --------------------------------------------------------------------------


async def test_load_sourced_prompt_templates_applies_map(tmp_path: Path):
    (tmp_path / "a.md").write_text("---\ndescription: A\n---\nContent")

    def mapper(template, source):
        return PromptTemplate(
            name=template.name + "-mapped", description=template.description, content=template.content
        )

    templates, diagnostics = await load_sourced_prompt_templates([(tmp_path / "a.md", "src")], mapper)
    assert diagnostics == []
    assert templates[0].prompt_template.name == "a-mapped"
    assert templates[0].source == "src"


async def test_load_sourced_prompt_templates_no_map(tmp_path: Path):
    (tmp_path / "b.md").write_text("---\ndescription: B\n---\nBody")
    templates, _diagnostics = await load_sourced_prompt_templates([(tmp_path / "b.md", 42)])
    assert len(templates) == 1
    assert templates[0].source == 42


# --------------------------------------------------------------------------
# format_prompt_template_invocation with no args
# --------------------------------------------------------------------------


def test_format_prompt_template_invocation_no_args():
    t = PromptTemplate(name="t", description="", content="Hello $1")
    result = format_prompt_template_invocation(t)
    assert result == "Hello "
