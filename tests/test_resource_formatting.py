"""Python port of `packages/agent/test/harness/resource-formatting.test.ts`."""

from __future__ import annotations

from pi_agent.harness.prompt_templates import format_prompt_template_invocation
from pi_agent.harness.skills import format_skill_invocation
from pi_agent.harness.types import PromptTemplate, Skill


def test_formats_skill_invocations_with_additional_instructions() -> None:
    skill = Skill(
        name="inspect",
        description="Inspect things",
        content="Use inspection tools.",
        file_path="/project/.pi/skills/inspect/SKILL.md",
    )

    assert format_skill_invocation(skill, "Check errors.") == (
        '<skill name="inspect" location="/project/.pi/skills/inspect/SKILL.md">\n'
        "References are relative to /project/.pi/skills/inspect.\n\n"
        "Use inspection tools.\n</skill>\n\nCheck errors."
    )


def test_formats_prompt_template_invocations_with_positional_arguments() -> None:
    assert (
        format_prompt_template_invocation(
            PromptTemplate(name="review", content="Review $1 with $ARGUMENTS"), ["a.ts", "care"]
        )
        == "Review a.ts with a.ts care"
    )
