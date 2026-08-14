"""Python port of `packages/agent/test/harness/telemetry.test.ts`.

Covers `pi_agent.harness.telemetry` and the ported documentation generator
(`packages/pi-agent/scripts/generate_telemetry_docs.py`).

Three of the four upstream cases are largely `expectTypeOf` / `@ts-expect-error`
assertions: they check that `startAiSpan` and `startHarnessSpan` reject an
unknown, missing, or wrongly-valued attribute *at compile time*. Python has no
equivalent -- there is no type-level assertion to run and nothing raises at
runtime, because upstream states outright that "schema values are used only for
type inference; no runtime schema validation is performed" and this port keeps
that. Each such assertion is therefore ported against the same schema data the
TypeScript types are derived from: `AiSpanStartAttributes<"pi.ai.request">`
requiring exactly those five keys becomes an assertion on
`AI_TELEMETRY_SCHEMA["spans"]["pi.ai.request"]["startAttributes"]`, and
`expectTypeOf<End["pi.ai.response.stop_reason"]>` becomes an assertion on that
attribute's declared `values`. The runtime halves of those cases (the spans do
start, and end attributes can be set) are ported as-is, and recorded through
`InMemoryTelemetryContext` so the span tree is actually asserted rather than
merely not throwing.

`docs/telemetry-schema.md` is checked in on the Python side too, and this
suite pins it to the generator's output exactly as upstream does. That file is
byte-identical to `packages/agent/docs/telemetry-schema.md`, which is what makes
these schemas verified against the original rather than against themselves.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from pi_telemetry import NOOP_TELEMETRY_CONTEXT, InMemoryTelemetryContext, TelemetrySpan

from pi_agent.harness.telemetry import (
    AGENT_TELEMETRY_SCHEMAS,
    AI_TELEMETRY_SCHEMA,
    HARNESS_TELEMETRY_SCHEMA,
    create_agent_span_starter,
    span_names,
    start_ai_span,
    start_harness_span,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from generate_telemetry_docs import (
    DOCS_PATH,
    render_agent_telemetry_schema_markdown,
)


def test_serializes_both_schemas_and_generates_the_checked_in_reference() -> None:
    json.dumps(AI_TELEMETRY_SCHEMA)
    json.dumps(HARNESS_TELEMETRY_SCHEMA)
    assert AGENT_TELEMETRY_SCHEMAS == [AI_TELEMETRY_SCHEMA, HARNESS_TELEMETRY_SCHEMA]
    assert span_names(HARNESS_TELEMETRY_SCHEMA) == [
        "pi.harness.run",
        "pi.harness.compaction",
        "pi.harness.navigation",
        "pi.harness.checkpoint",
        "pi.harness.turn",
        "pi.harness.step",
        "pi.harness.tool",
        "pi.harness.hook",
        "pi.harness.sleep",
        "pi.harness.event_handler",
        "pi.session.write",
    ]

    assert DOCS_PATH.read_text(encoding="utf-8") == render_agent_telemetry_schema_markdown()


async def test_starts_ai_request_and_harness_spans_through_one_composed_typed_starter() -> None:
    telemetry = InMemoryTelemetryContext()
    start_span = create_agent_span_starter(telemetry)

    async def step(step_span: TelemetrySpan, start_child_span) -> None:
        step_span.set_attributes({"pi.step.outcome": "succeeded"})

        def request(request_span: TelemetrySpan, _start_grandchild) -> None:
            request_span.set_attributes({"pi.ai.response.stop_reason": "stop"})

        await start_child_span(
            "pi.ai.request",
            {
                "pi.ai.operation": "stream",
                "pi.ai.provider": "provider",
                "pi.ai.model": "model",
                "pi.ai.api": "api",
                "pi.ai.streaming": True,
            },
            request,
        )

    await start_span(
        "pi.harness.step",
        {
            "pi.lane.name": "main",
            "pi.operation.id": "operation",
            "pi.step.kind": "assistant",
            "pi.step.attempt": 1,
        },
        step,
    )

    recorded = telemetry.get_spans()
    [recorded_step, recorded_request] = recorded
    assert recorded_step.name == "pi.harness.step"
    assert recorded_step.attributes["pi.lane.name"] == "main"
    assert recorded_step.attributes["pi.step.outcome"] == "succeeded"
    assert recorded_request.parent_id == recorded_step.id
    assert recorded_request.name == "pi.ai.request"
    assert recorded_request.attributes["pi.ai.streaming"] is True
    assert recorded_request.attributes["pi.ai.response.stop_reason"] == "stop"


async def test_infers_exact_ai_start_and_optional_end_attributes() -> None:
    request_span_schema = AI_TELEMETRY_SCHEMA["spans"]["pi.ai.request"]
    start_attributes = request_span_schema["startAttributes"]

    # `AiSpanStartAttributes<"pi.ai.request">` in TypeScript. Required keys are
    # required, `pi.ai.deferred` is the one optional one.
    assert {name for name, definition in start_attributes.items() if definition["required"]} == {
        "pi.ai.operation",
        "pi.ai.provider",
        "pi.ai.model",
        "pi.ai.api",
        "pi.ai.streaming",
    }
    assert start_attributes["pi.ai.deferred"]["required"] is False
    assert start_attributes["pi.ai.operation"]["values"] == [
        "stream",
        "fetch_deferred",
        "cancel_deferred",
        "generate_images",
    ]
    assert start_attributes["pi.ai.provider"]["type"] == "string"
    assert start_attributes["pi.ai.model"]["type"] == "string"
    assert start_attributes["pi.ai.api"]["type"] == "string"
    assert start_attributes["pi.ai.streaming"]["type"] == "boolean"
    assert start_attributes["pi.ai.deferred"]["type"] == "boolean"

    # `AiSpanEndAttributes<"pi.ai.request">["pi.ai.response.stop_reason"]`.
    end_attributes = request_span_schema["endAttributes"]
    assert end_attributes["pi.ai.response.stop_reason"]["values"] == [
        "stop",
        "length",
        "tool_use",
        "error",
        "aborted",
        "deferred",
    ]
    assert all("required" not in definition for definition in end_attributes.values())

    # `pi.ai.request` declares no span events, so `span.addEvent(...)` is a
    # compile-time error upstream.
    assert "events" not in request_span_schema

    telemetry = InMemoryTelemetryContext()
    await start_ai_span(
        telemetry,
        "pi.ai.request",
        {
            "pi.ai.operation": "stream",
            "pi.ai.provider": "provider",
            "pi.ai.model": "model",
            "pi.ai.api": "api",
            "pi.ai.streaming": True,
        },
        lambda span: span.set_attributes({"pi.ai.response.stop_reason": "tool_use"}),
    )

    [recorded] = telemetry.get_spans()
    assert recorded.name == "pi.ai.request"
    assert recorded.attributes["pi.ai.response.stop_reason"] == "tool_use"


async def test_infers_per_span_harness_literals_and_optional_completion_enrichment() -> None:
    run_span_schema = HARNESS_TELEMETRY_SCHEMA["spans"]["pi.harness.run"]

    # `HarnessSpanStartAttributes<"pi.harness.run">["pi.operation.kind"]` is the
    # literal "run": a run span accepts no other operation kind.
    assert run_span_schema["startAttributes"]["pi.operation.kind"]["values"] == ["run"]
    assert {name for name, definition in run_span_schema["startAttributes"].items() if definition["required"]} == {
        "pi.session.id",
        "pi.lane.name",
        "pi.operation.id",
        "pi.operation.recovery",
        "pi.operation.kind",
    }

    # `HarnessSpanEndAttributes<"pi.harness.run">["pi.operation.outcome"]`.
    assert run_span_schema["endAttributes"]["pi.operation.outcome"]["values"] == [
        "completed",
        "aborted",
        "failed",
        "suspended",
    ]

    # Empty end schemas reject every attribute upstream.
    assert HARNESS_TELEMETRY_SCHEMA["spans"]["pi.harness.checkpoint"]["endAttributes"] == {}
    assert "events" not in run_span_schema

    telemetry = InMemoryTelemetryContext()

    def run(span: TelemetrySpan) -> None:
        span.set_attributes({"pi.operation.outcome": "completed"})
        span.set_attributes({})

    await start_harness_span(
        telemetry,
        "pi.harness.run",
        {
            "pi.session.id": "session",
            "pi.lane.name": "main",
            "pi.operation.id": "operation",
            "pi.operation.kind": "run",
            "pi.operation.recovery": False,
        },
        run,
    )

    [recorded] = telemetry.get_spans()
    assert recorded.name == "pi.harness.run"
    assert recorded.attributes["pi.operation.outcome"] == "completed"
    assert recorded.attributes["pi.operation.recovery"] is False


async def test_the_noop_context_accepts_every_agent_span() -> None:
    """Upstream drives the same starters through `NOOP_TELEMETRY_CONTEXT`."""
    start_span = create_agent_span_starter(NOOP_TELEMETRY_CONTEXT)

    result = await start_span(
        "pi.harness.checkpoint",
        {"pi.lane.name": "main", "pi.operation.id": "operation", "pi.checkpoint.kind": "normal"},
        lambda span, _start_child: "done",
    )

    assert result == "done"


@pytest.mark.parametrize("schema", AGENT_TELEMETRY_SCHEMAS)
def test_every_span_declares_a_version_parents_and_status(schema) -> None:
    assert schema["version"] == 1
    for span in schema["spans"].values():
        assert span["parents"]["kind"] in ("any", "root_or_external", "spans")
        if span["parents"]["kind"] == "spans":
            assert span["parents"]["spans"]
        assert span["status"]["default"] == "ok"
        assert span["status"]["errorWhen"]
