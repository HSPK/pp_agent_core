"""Tests for `pi_agent.harness.system_prompt`.

Ported from `packages/agent/test/harness/system-prompt.test.ts`.
"""

from __future__ import annotations

from pi_agent.harness.system_prompt import format_skills_for_system_prompt
from pi_agent.harness.types import Skill

visible_skill = Skill(
    name="visible",
    description="Use <this> & that",
    content="visible content",
    file_path="/skills/visible/SKILL.md",
)

second_skill = Skill(
    name="second",
    description="Second skill",
    content="second content",
    file_path="/skills/second/SKILL.md",
)

disabled_skill = Skill(
    name="hidden",
    description="Hidden",
    content="hidden content",
    file_path="/skills/hidden/SKILL.md",
    disable_model_invocation=True,
)


def test_formats_visible_skills_in_order_and_skips_model_disabled_skills():
    assert format_skills_for_system_prompt([visible_skill, disabled_skill, second_skill]) == (
        "The following skills provide specialized instructions for specific tasks.\n"
        "Read the full skill file when the task matches its description.\n"
        "When a skill file references a relative path, resolve it against the skill directory "
        "(parent of SKILL.md / dirname of the path) and use that absolute path in tool commands.\n"
        "\n"
        "<available_skills>\n"
        "  <skill>\n"
        "    <name>visible</name>\n"
        "    <description>Use &lt;this&gt; &amp; that</description>\n"
        "    <location>/skills/visible/SKILL.md</location>\n"
        "  </skill>\n"
        "  <skill>\n"
        "    <name>second</name>\n"
        "    <description>Second skill</description>\n"
        "    <location>/skills/second/SKILL.md</location>\n"
        "  </skill>\n"
        "</available_skills>"
    )


def test_returns_empty_string_when_no_skills_are_model_visible():
    assert format_skills_for_system_prompt([disabled_skill]) == ""


def test_escapes_xml_in_all_model_visible_skill_fields():
    skill = Skill(
        name="a&b",
        description="Quote \"double\" and 'single'",
        content="content",
        file_path='/skills/<bad>&"quote"/SKILL.md',
    )
    result = format_skills_for_system_prompt([skill])

    assert (
        "<name>a&amp;b</name>\n"
        "    <description>Quote &quot;double&quot; and &apos;single&apos;</description>\n"
        "    <location>/skills/&lt;bad&gt;&amp;&quot;quote&quot;/SKILL.md</location>"
    ) in result
