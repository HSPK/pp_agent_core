"""Tests for `pi_agent.harness.prompt_templates`.

Ported from `packages/agent/test/harness/prompt-templates.test.ts`. The
TypeScript suite drives loading through `NodeExecutionEnv`; this port reads
directly from `pathlib.Path` (see `prompt_templates.py`'s module docstring),
so the fixtures below write files directly under `tmp_path` instead of going
through an execution-environment abstraction.
"""

from __future__ import annotations

from pathlib import Path

from pi_agent.harness.prompt_templates import (
    format_prompt_template_invocation,
    load_prompt_templates,
    load_sourced_prompt_templates,
)
from pi_agent.harness.types import PromptTemplate


async def test_loads_markdown_templates_non_recursively_from_one_or_more_dirs(tmp_path: Path):
    (tmp_path / "a" / "nested").mkdir(parents=True)
    (tmp_path / "b").mkdir(parents=True)
    (tmp_path / "a" / "one.md").write_text("---\ndescription: One template\n---\nHello $1")
    (tmp_path / "a" / "nested" / "ignored.md").write_text("Ignored")
    (tmp_path / "b" / "two.md").write_text("First line description\nBody")

    prompt_templates, diagnostics = await load_prompt_templates([tmp_path / "a", tmp_path / "b"])

    assert diagnostics == []
    assert prompt_templates == [
        PromptTemplate(name="one", description="One template", content="Hello $1"),
        PromptTemplate(name="two", description="First line description", content="First line description\nBody"),
    ]


async def test_preserves_source_info_for_sourced_prompt_templates(tmp_path: Path):
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "example.md").write_text("---\ndescription: Example\n---\nExample body")

    prompt_templates, diagnostics = await load_sourced_prompt_templates([(tmp_path / "prompts", {"type": "project"})])

    assert diagnostics == []
    assert len(prompt_templates) == 1
    assert prompt_templates[0].prompt_template == PromptTemplate(
        name="example", description="Example", content="Example body"
    )
    assert prompt_templates[0].source == {"type": "project"}


async def test_attaches_source_info_to_diagnostics(tmp_path: Path):
    (tmp_path / "broken.md").write_text("---\ndescription: [unterminated\n---\nBody")

    prompt_templates, diagnostics = await load_sourced_prompt_templates([(tmp_path / "broken.md", {"type": "user"})])

    assert prompt_templates == []
    assert len(diagnostics) == 1
    assert diagnostics[0].diagnostic.type == "warning"
    assert diagnostics[0].diagnostic.path == str(tmp_path / "broken.md")
    assert diagnostics[0].source == {"type": "user"}


async def test_loads_explicit_markdown_files_and_symlinked_files(tmp_path: Path):
    target = tmp_path / "target.md"
    target.write_text("---\ndescription: Target\n---\nTarget body")
    link = tmp_path / "link.md"
    link.symlink_to(target)

    prompt_templates, _diagnostics = await load_prompt_templates([target, link])

    assert prompt_templates == [
        PromptTemplate(name="target", description="Target", content="Target body"),
        PromptTemplate(name="link", description="Target", content="Target body"),
    ]


def test_substitutes_command_arguments():
    content = "$1 ${@:2} $ARGUMENTS"
    template = PromptTemplate(name="one", description="", content=content)

    assert format_prompt_template_invocation(template, ["hello world", "test"]) == ("hello world test hello world test")
