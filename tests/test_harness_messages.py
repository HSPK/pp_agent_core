"""Tests for `pi_agent.harness.messages`.

No dedicated TS test file exists for `harness/messages.ts` (it is only
exercised indirectly through `agent-loop.test.ts` via injected
`convertToLlm` callbacks). This suite covers the module's public behavior
directly: `bash_execution_to_text`, `create_custom_message`, and
`convert_to_llm` for every `HarnessMessage` role.
"""

from __future__ import annotations

from pi_agent.harness.messages import (
    BRANCH_SUMMARY_PREFIX,
    BRANCH_SUMMARY_SUFFIX,
    COMPACTION_SUMMARY_PREFIX,
    COMPACTION_SUMMARY_SUFFIX,
    BashExecutionMessage,
    CustomMessage,
    bash_execution_to_text,
    convert_to_llm,
    create_branch_summary_message,
    create_compaction_summary_message,
    create_custom_message,
)
from pi_ai.types import TextContent, ToolResultMessage, UserMessage


def _bash_message(**overrides: object) -> BashExecutionMessage:
    defaults = dict(
        command="echo hi",
        output="hi",
        exit_code=0,
        cancelled=False,
        truncated=False,
        timestamp=1,
    )
    defaults.update(overrides)
    return BashExecutionMessage(**defaults)


def test_bash_execution_to_text_includes_output_block():
    text = bash_execution_to_text(_bash_message())
    assert text == "Ran `echo hi`\n```\nhi\n```"


def test_bash_execution_to_text_reports_no_output():
    text = bash_execution_to_text(_bash_message(output=""))
    assert text == "Ran `echo hi`\n(no output)"


def test_bash_execution_to_text_reports_cancellation():
    text = bash_execution_to_text(_bash_message(cancelled=True))
    assert text.endswith("\n\n(command cancelled)")


def test_bash_execution_to_text_reports_nonzero_exit_code():
    text = bash_execution_to_text(_bash_message(exit_code=1))
    assert text.endswith("\n\nCommand exited with code 1")


def test_bash_execution_to_text_ignores_zero_exit_code():
    text = bash_execution_to_text(_bash_message(exit_code=0))
    assert "Command exited with code" not in text


def test_bash_execution_to_text_appends_truncation_note_with_full_output_path():
    text = bash_execution_to_text(_bash_message(truncated=True, full_output_path="/tmp/full.txt"))
    assert text.endswith("\n\n[Output truncated. Full output: /tmp/full.txt]")


def test_bash_execution_to_text_omits_truncation_note_without_full_output_path():
    text = bash_execution_to_text(_bash_message(truncated=True, full_output_path=None))
    assert "[Output truncated" not in text


def test_create_custom_message_builds_dataclass():
    message = create_custom_message("notification", "hello", True, {"a": 1}, 42)
    assert message == CustomMessage(
        custom_type="notification", content="hello", display=True, details={"a": 1}, timestamp=42
    )


def test_convert_to_llm_converts_bash_execution_message():
    message = _bash_message()
    result = convert_to_llm([message])
    assert result == [UserMessage(content=[TextContent(text=bash_execution_to_text(message))], timestamp=1)]


def test_convert_to_llm_excludes_bash_execution_message_from_context():
    message = _bash_message(exclude_from_context=True)
    assert convert_to_llm([message]) == []


def test_convert_to_llm_converts_string_custom_message():
    message = create_custom_message("notification", "hello", True, None, 5)
    result = convert_to_llm([message])
    assert result == [UserMessage(content=[TextContent(text="hello")], timestamp=5)]


def test_convert_to_llm_converts_content_list_custom_message():
    content = [TextContent(text="hello")]
    message = create_custom_message("notification", content, True, None, 5)
    result = convert_to_llm([message])
    assert result == [UserMessage(content=content, timestamp=5)]


def test_convert_to_llm_converts_branch_summary_message():
    message = create_branch_summary_message("summary text", "entry-1", 7)
    result = convert_to_llm([message])
    expected_text = BRANCH_SUMMARY_PREFIX + "summary text" + BRANCH_SUMMARY_SUFFIX
    assert result == [UserMessage(content=[TextContent(text=expected_text)], timestamp=7)]


def test_convert_to_llm_converts_compaction_summary_message():
    message = create_compaction_summary_message("summary text", 100, 9)
    result = convert_to_llm([message])
    expected_text = COMPACTION_SUMMARY_PREFIX + "summary text" + COMPACTION_SUMMARY_SUFFIX
    assert result == [UserMessage(content=[TextContent(text=expected_text)], timestamp=9)]


def test_convert_to_llm_passes_through_user_assistant_and_tool_result_messages():
    user = UserMessage(content="hi", timestamp=1)
    tool_result = ToolResultMessage(tool_call_id="c1", tool_name="bash", content=[], timestamp=2)

    assert convert_to_llm([user, tool_result]) == [user, tool_result]
