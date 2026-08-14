"""Tests for `pi_agent.harness.skills`.

Ported from `packages/agent/test/harness/skills.test.ts`. The TypeScript
suite drives loading through `NodeExecutionEnv`; this port reads directly
from `pathlib.Path` (see `skills.py`'s module docstring), so fixtures write
files directly under `tmp_path` instead of going through an
execution-environment abstraction.
"""

from __future__ import annotations

from pathlib import Path

from pi_agent.harness.skills import load_skills, load_sourced_skills
from pi_agent.harness.types import Skill


async def test_loads_skill_md_files_from_directories(tmp_path: Path):
    skill_dir = tmp_path / ".agents" / "skills" / "example"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: example\ndescription: Example skill\ndisable-model-invocation: true\n---\nUse this skill.\n"
    )

    skills, diagnostics = await load_skills(tmp_path / ".agents" / "skills")

    assert diagnostics == []
    assert skills == [
        Skill(
            name="example",
            description="Example skill",
            content="Use this skill.",
            file_path=str(skill_dir / "SKILL.md"),
            disable_model_invocation=True,
        )
    ]


async def test_loads_skills_through_symlinked_directories(tmp_path: Path):
    actual = tmp_path / "actual" / "example"
    actual.mkdir(parents=True)
    (actual / "SKILL.md").write_text("---\nname: example\ndescription: Example skill\n---\nUse this skill.")
    link = tmp_path / "skills-link"
    link.symlink_to(tmp_path / "actual")

    skills, _diagnostics = await load_skills(link)

    assert [skill.name for skill in skills] == ["example"]
    assert skills[0].file_path == str(link / "example" / "SKILL.md")


async def test_preserves_source_info_for_sourced_skills(tmp_path: Path):
    skill_dir = tmp_path / "user" / "example"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: example\ndescription: Example skill\n---\nUse this skill.")

    skills, diagnostics = await load_sourced_skills([(tmp_path / "user", {"type": "user"})])

    assert diagnostics == []
    assert len(skills) == 1
    assert skills[0].skill == Skill(
        name="example",
        description="Example skill",
        content="Use this skill.",
        file_path=str(skill_dir / "SKILL.md"),
        disable_model_invocation=False,
    )
    assert skills[0].source == {"type": "user"}


async def test_attaches_source_info_to_diagnostics(tmp_path: Path):
    broken_dir = tmp_path / "user" / "broken"
    broken_dir.mkdir(parents=True)
    (broken_dir / "SKILL.md").write_text("---\nname: broken\n---\nMissing description.")

    skills, diagnostics = await load_sourced_skills([(tmp_path / "user", {"type": "user"})])

    assert skills == []
    assert len(diagnostics) == 1
    assert diagnostics[0].diagnostic.type == "warning"
    assert diagnostics[0].diagnostic.code == "invalid_metadata"
    assert diagnostics[0].diagnostic.message == "description is required"
    assert diagnostics[0].diagnostic.path == str(broken_dir / "SKILL.md")
    assert diagnostics[0].source == {"type": "user"}


async def test_loads_direct_markdown_children_only_from_root_directory(tmp_path: Path):
    skills_dir = tmp_path / "skills"
    (skills_dir / "nested").mkdir(parents=True)
    (skills_dir / "root.md").write_text("---\ndescription: Root skill\n---\nRoot content")
    (skills_dir / "nested" / "ignored.md").write_text("---\ndescription: Ignored\n---\nIgnored content")

    skills, _diagnostics = await load_skills(skills_dir)

    assert [skill.name for skill in skills] == ["skills"]
    assert skills[0].content == "Root content"
