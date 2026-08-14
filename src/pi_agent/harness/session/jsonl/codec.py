"""JSONL v4 wire codec: header + mutation lines, and their nested payloads.

Python port of `packages/agent/src/harness/session/jsonl/codec.py`'s
TypeScript counterpart, `jsonl/codec.ts`. In TypeScript the in-memory entry
and record shapes are already the wire shapes (both are just plain JS
objects with camelCase keys), so `codec.ts` only has to validate a parsed
JSON value and cast it. Python's in-memory shapes are `snake_case`
dataclasses (`Entry`, `LaneRecord`, `pi_ai.types.Message`, ...), so this port
also has to convert field names and rebuild nested dataclasses (`AgentMessage`,
`Usage`, `DeferredHandle`, ...) that TypeScript leaves untouched. Wire field
names keep their exact TypeScript spelling (`parentId`, `toolCallId`,
`stopReason`, `cacheWrite1h`, ...) so sessions written by the TypeScript `pi`
stay readable by this port and vice versa.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

from pi_ai.types import (
    AssistantMessage,
    AssistantMessageDiagnostic,
    Content,
    Cost,
    DeferredHandle,
    ImageContent,
    Message,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)

from ....types import AgentMessage
from ..state import EntryMutation, LabelFactMutation, LaneMutation, NameFactMutation, RecordMutation, SessionMutation
from ..types import (
    ENTRY_TYPES,
    OPERATION_KINDS,
    RECORD_TYPES,
    AbortRequestedRecord,
    ActiveToolsEntry,
    BranchSummaryEntry,
    CompactionEntry,
    CompactionIntent,
    CustomEntry,
    Entry,
    JsonValue,
    LaneRecord,
    MessageEntry,
    ModelChangeEntry,
    NavigationIntent,
    OperationFinishedError,
    OperationFinishedRecord,
    OperationIntent,
    OperationStartedRecord,
    QueueCancelledRecord,
    QueueEnqueuedRecord,
    RunIntent,
    StepAttemptRecord,
    ThinkingLevelEntry,
    ToolStartedRecord,
    UsageRecord,
    WriteDeferredRecord,
)
from .errors import JsonlDecodeError
from .types import JsonlSessionMetadata, JsonlV4Header

# --------------------------------------------------------------------------
# Low-level validation helpers (mirror codec.ts's require* functions)
# --------------------------------------------------------------------------


def _is_object(value: Any) -> bool:
    return isinstance(value, dict)


def _parse_object(line: str) -> dict[str, Any]:
    try:
        value = json.loads(line)
    except ValueError as error:
        raise JsonlDecodeError("syntax", "is not valid JSON", error) from error
    if not _is_object(value):
        raise JsonlDecodeError("schema", "is not a JSON object")
    return value


def _require_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise JsonlDecodeError("schema", f"has invalid {field_name}")
    return value


def _require_sequence(value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise JsonlDecodeError("schema", "has invalid seq")
    return value


def _require_timestamp(value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise JsonlDecodeError("schema", "has invalid timestamp")
    return value


def _require_nullable_id(value: Any, field_name: str) -> str | None:
    if value is not None and not isinstance(value, str):
        raise JsonlDecodeError("schema", f"has invalid {field_name}")
    return value


# --------------------------------------------------------------------------
# Generic JSON-value passthrough (used for opaque `Any`/`JsonValue` payloads:
# `details`, `data`, `effective_args`, `resume_data`, ...). `json.loads`
# already produced plain `dict`/`list`/primitive values for these, so no
# further conversion is needed; this only guards against embedding a
# non-JSON-native Python value when encoding.
# --------------------------------------------------------------------------


def _encode_value(value: Any) -> JsonValue:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, list):
        return [_encode_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _encode_value(item) for key, item in value.items()}
    raise TypeError(f"Cannot encode {type(value)!r} as JSON")


# --------------------------------------------------------------------------
# Content
# --------------------------------------------------------------------------


def _content_to_wire(content: Content) -> dict[str, JsonValue]:
    if isinstance(content, TextContent):
        payload: dict[str, JsonValue] = {"type": "text", "text": content.text}
        if content.text_signature is not None:
            payload["textSignature"] = content.text_signature
        return payload
    if isinstance(content, ThinkingContent):
        payload = {"type": "thinking", "thinking": content.thinking}
        if content.thinking_signature is not None:
            payload["thinkingSignature"] = content.thinking_signature
        if content.redacted is not None:
            payload["redacted"] = content.redacted
        return payload
    if isinstance(content, ImageContent):
        return {"type": "image", "data": content.data, "mimeType": content.mime_type}
    if isinstance(content, ToolCall):
        payload = {
            "type": "toolCall",
            "id": content.id,
            "name": content.name,
            "arguments": _encode_value(content.arguments),
        }
        if content.thought_signature is not None:
            payload["thoughtSignature"] = content.thought_signature
        if content.namespace is not None:
            payload["namespace"] = content.namespace
        return payload
    raise TypeError(f"Cannot encode content {content!r}")


def _wire_to_content(value: dict[str, Any]) -> Content:
    content_type = value.get("type")
    if content_type == "text":
        return TextContent(
            text=_require_string(value.get("text", ""), "text"), text_signature=value.get("textSignature")
        )
    if content_type == "thinking":
        return ThinkingContent(
            thinking=_require_string(value.get("thinking", ""), "thinking"),
            thinking_signature=value.get("thinkingSignature"),
            redacted=value.get("redacted"),
        )
    if content_type == "image":
        return ImageContent(
            data=_require_string(value.get("data", ""), "data"),
            mime_type=_require_string(value.get("mimeType", ""), "mimeType"),
        )
    if content_type == "toolCall":
        return ToolCall(
            id=_require_string(value.get("id", ""), "id"),
            name=_require_string(value.get("name", ""), "name"),
            arguments=value.get("arguments") or {},
            thought_signature=value.get("thoughtSignature"),
            namespace=value.get("namespace"),
        )
    raise JsonlDecodeError("schema", f"has unknown content type {content_type!r}")


# --------------------------------------------------------------------------
# Usage / cost
# --------------------------------------------------------------------------


def _cost_to_wire(cost: Cost) -> dict[str, JsonValue]:
    return {
        "input": cost.input,
        "output": cost.output,
        "cacheRead": cost.cache_read,
        "cacheWrite": cost.cache_write,
        "total": cost.total,
    }


def _wire_to_cost(value: dict[str, Any]) -> Cost:
    return Cost(
        input=value.get("input", 0.0),
        output=value.get("output", 0.0),
        cache_read=value.get("cacheRead", 0.0),
        cache_write=value.get("cacheWrite", 0.0),
        total=value.get("total", 0.0),
    )


def _usage_to_wire(usage: Usage) -> dict[str, JsonValue]:
    payload: dict[str, JsonValue] = {
        "input": usage.input,
        "output": usage.output,
        "cacheRead": usage.cache_read,
        "cacheWrite": usage.cache_write,
        "totalTokens": usage.total_tokens,
        "cost": _cost_to_wire(usage.cost),
    }
    if usage.cache_write_1h is not None:
        payload["cacheWrite1h"] = usage.cache_write_1h
    if usage.reasoning is not None:
        payload["reasoning"] = usage.reasoning
    return payload


def _wire_to_usage(value: dict[str, Any]) -> Usage:
    return Usage(
        input=value.get("input", 0),
        output=value.get("output", 0),
        cache_read=value.get("cacheRead", 0),
        cache_write=value.get("cacheWrite", 0),
        cache_write_1h=value.get("cacheWrite1h"),
        reasoning=value.get("reasoning"),
        total_tokens=value.get("totalTokens", 0),
        cost=_wire_to_cost(value["cost"]) if "cost" in value else Cost(),
    )


def _deferred_to_wire(handle: DeferredHandle) -> dict[str, JsonValue]:
    payload: dict[str, JsonValue] = {
        "provider": handle.provider,
        "modelId": handle.model_id,
        "api": handle.api,
        "id": handle.id,
    }
    if handle.expires_at is not None:
        payload["expiresAt"] = handle.expires_at
    if handle.poll_after_ms is not None:
        payload["pollAfterMs"] = handle.poll_after_ms
    if handle.data is not None:
        payload["data"] = _encode_value(handle.data)
    return payload


def _wire_to_deferred(value: dict[str, Any]) -> DeferredHandle:
    return DeferredHandle(
        provider=value.get("provider", ""),
        model_id=value.get("modelId", ""),
        api=value.get("api", ""),
        id=value.get("id", ""),
        expires_at=value.get("expiresAt"),
        poll_after_ms=value.get("pollAfterMs"),
        data=value.get("data"),
    )


def _diagnostic_to_wire(diagnostic: AssistantMessageDiagnostic) -> dict[str, JsonValue]:
    payload: dict[str, JsonValue] = {
        "kind": diagnostic.kind,
        "message": diagnostic.message,
        "timestamp": diagnostic.timestamp,
    }
    if diagnostic.detail is not None:
        payload["detail"] = _encode_value(diagnostic.detail)
    return payload


def _wire_to_diagnostic(value: dict[str, Any]) -> AssistantMessageDiagnostic:
    return AssistantMessageDiagnostic(
        kind=value.get("kind", ""),
        message=value.get("message", ""),
        detail=value.get("detail"),
        timestamp=value.get("timestamp", 0),
    )


# --------------------------------------------------------------------------
# Messages
# --------------------------------------------------------------------------


def _message_to_wire(message: Message) -> dict[str, JsonValue]:
    if isinstance(message, UserMessage):
        content = (
            message.content if isinstance(message.content, str) else [_content_to_wire(c) for c in message.content]
        )
        return {"role": "user", "content": content, "timestamp": message.timestamp}
    if isinstance(message, AssistantMessage):
        payload: dict[str, JsonValue] = {
            "role": "assistant",
            "api": message.api,
            "provider": message.provider,
            "model": message.model,
            "content": [_content_to_wire(c) for c in message.content],
            "usage": _usage_to_wire(message.usage),
            "stopReason": message.stop_reason,
            "timestamp": message.timestamp,
        }
        if message.response_model is not None:
            payload["responseModel"] = message.response_model
        if message.response_id is not None:
            payload["responseId"] = message.response_id
        if message.diagnostics:
            payload["diagnostics"] = [_diagnostic_to_wire(d) for d in message.diagnostics]
        if message.deferred is not None:
            payload["deferred"] = _deferred_to_wire(message.deferred)
        if message.error_message is not None:
            payload["errorMessage"] = message.error_message
        if message.raw_stop_reason is not None:
            payload["rawStopReason"] = message.raw_stop_reason
        if message.end_turn is not None:
            payload["endTurn"] = message.end_turn
        return payload
    if isinstance(message, ToolResultMessage):
        payload = {
            "role": "toolResult",
            "toolCallId": message.tool_call_id,
            "toolName": message.tool_name,
            "content": [_content_to_wire(c) for c in message.content],
            "isError": message.is_error,
            "timestamp": message.timestamp,
        }
        if message.details is not None:
            payload["details"] = _encode_value(message.details)
        if message.usage is not None:
            payload["usage"] = _usage_to_wire(message.usage)
        if message.added_tool_names is not None:
            payload["addedToolNames"] = list(message.added_tool_names)
        return payload
    raise TypeError(f"Cannot encode message {message!r}")


def _wire_to_message(value: dict[str, Any]) -> AgentMessage:
    role = value.get("role")
    if role == "user":
        content = value.get("content", "")
        return UserMessage(
            content=content if isinstance(content, str) else [_wire_to_content(c) for c in content],
            timestamp=value.get("timestamp", 0),
        )
    if role == "assistant":
        return AssistantMessage(
            api=value.get("api", ""),
            provider=value.get("provider", ""),
            model=value.get("model", ""),
            content=[_wire_to_content(c) for c in value.get("content", [])],
            usage=_wire_to_usage(value["usage"]) if "usage" in value else Usage(),
            stop_reason=value.get("stopReason", "pending"),
            response_model=value.get("responseModel"),
            response_id=value.get("responseId"),
            diagnostics=[_wire_to_diagnostic(d) for d in value.get("diagnostics", [])],
            deferred=_wire_to_deferred(value["deferred"]) if value.get("deferred") is not None else None,
            error_message=value.get("errorMessage"),
            raw_stop_reason=value.get("rawStopReason"),
            end_turn=value.get("endTurn"),
            timestamp=value.get("timestamp", 0),
        )
    if role == "toolResult":
        return ToolResultMessage(
            tool_call_id=value.get("toolCallId", ""),
            tool_name=value.get("toolName", ""),
            content=[_wire_to_content(c) for c in value.get("content", [])],
            details=value.get("details"),
            usage=_wire_to_usage(value["usage"]) if "usage" in value else None,
            added_tool_names=list(value["addedToolNames"]) if "addedToolNames" in value else None,
            is_error=bool(value.get("isError", False)),
            timestamp=value.get("timestamp", 0),
        )
    raise JsonlDecodeError("schema", f"has unknown message role {role!r}")


# --------------------------------------------------------------------------
# Entries
# --------------------------------------------------------------------------


def _entry_payload_to_wire(entry: Entry) -> dict[str, JsonValue]:
    """Fields shared by every entry variant, excluding `parent_id`/`seq`/`timestamp`.

    Those three are storage-assigned: always present (even `parent_id: null`)
    on a full/stored entry, always absent on a provisioned entry. Callers
    choose which by using `entry_to_wire` or `provisioned_entry_to_wire`.
    """
    payload: dict[str, JsonValue] = {"type": entry.type, "id": entry.id}
    if isinstance(entry, MessageEntry):
        payload["message"] = _message_to_wire(entry.message)
        if entry.terminate is not None:
            payload["terminate"] = entry.terminate
    elif isinstance(entry, ModelChangeEntry):
        payload["provider"] = entry.provider
        payload["modelId"] = entry.model_id
    elif isinstance(entry, ThinkingLevelEntry):
        payload["thinkingLevel"] = entry.thinking_level
    elif isinstance(entry, ActiveToolsEntry):
        payload["activeToolNames"] = list(entry.active_tool_names)
    elif isinstance(entry, CompactionEntry):
        payload["summary"] = entry.summary
        payload["retainedTail"] = [_message_to_wire(m) for m in entry.retained_tail]
        payload["tokensBefore"] = entry.tokens_before
        if entry.details is not None:
            payload["details"] = _encode_value(entry.details)
        if entry.usage is not None:
            payload["usage"] = _usage_to_wire(entry.usage)
    elif isinstance(entry, BranchSummaryEntry):
        payload["fromId"] = entry.from_id
        payload["summary"] = entry.summary
        if entry.details is not None:
            payload["details"] = _encode_value(entry.details)
        if entry.usage is not None:
            payload["usage"] = _usage_to_wire(entry.usage)
    elif isinstance(entry, CustomEntry):
        payload["customType"] = entry.custom_type
        if entry.data is not None:
            payload["data"] = _encode_value(entry.data)
    else:
        raise TypeError(f"Cannot encode entry {entry!r}")
    return payload


def entry_to_wire(entry: Entry) -> dict[str, JsonValue]:
    """Encode a full (storage-assigned) entry, including `parentId`/`seq`/`timestamp`."""
    payload = _entry_payload_to_wire(entry)
    payload["parentId"] = entry.parent_id
    payload["seq"] = entry.seq
    payload["timestamp"] = entry.timestamp
    return payload


def provisioned_entry_to_wire(entry: Entry) -> dict[str, JsonValue]:
    """Encode a provisioned entry, omitting the storage-assigned fields entirely."""
    return _entry_payload_to_wire(entry)


def _decode_entry_payload(value: dict[str, Any]) -> Entry:
    """Decode the fields shared by every entry variant. Leaves `parent_id`/`seq`/
    `timestamp` at their dataclass defaults; callers fill them in as needed."""
    entry_type = _require_string(value.get("type"), "entry type")
    if entry_type not in ENTRY_TYPES:
        raise JsonlDecodeError("schema", f"has unknown entry type {entry_type}")
    id_ = _require_string(value.get("id"), "id")
    if entry_type == "message":
        terminate = value.get("terminate")
        return MessageEntry(id=id_, message=_wire_to_message(value.get("message") or {}), terminate=terminate)
    if entry_type == "model_change":
        return ModelChangeEntry(
            id=id_,
            provider=_require_string(value.get("provider"), "provider"),
            model_id=_require_string(value.get("modelId"), "modelId"),
        )
    if entry_type == "thinking_level_change":
        return ThinkingLevelEntry(id=id_, thinking_level=_require_string(value.get("thinkingLevel"), "thinkingLevel"))
    if entry_type == "active_tools_change":
        return ActiveToolsEntry(id=id_, active_tool_names=list(value.get("activeToolNames") or []))
    if entry_type == "compaction":
        return CompactionEntry(
            id=id_,
            summary=_require_string(value.get("summary"), "summary"),
            retained_tail=[_wire_to_message(m) for m in value.get("retainedTail") or []],
            tokens_before=value.get("tokensBefore", 0),
            details=value.get("details"),
            usage=_wire_to_usage(value["usage"]) if "usage" in value else None,
        )
    if entry_type == "branch_summary":
        return BranchSummaryEntry(
            id=id_,
            from_id=_require_string(value.get("fromId"), "fromId"),
            summary=_require_string(value.get("summary"), "summary"),
            details=value.get("details"),
            usage=_wire_to_usage(value["usage"]) if "usage" in value else None,
        )
    if entry_type == "custom":
        return CustomEntry(
            id=id_, custom_type=_require_string(value.get("customType"), "customType"), data=value.get("data")
        )
    raise JsonlDecodeError("schema", f"has unknown entry type {entry_type}")


def decode_entry(value: dict[str, Any], seq: int) -> Entry:
    """Decode a full (storage-assigned) entry: validates `parentId`/`timestamp`."""
    entry = _decode_entry_payload(value)
    parent_id = _require_nullable_id(value.get("parentId"), "parentId")
    timestamp = _require_timestamp(value.get("timestamp"))
    return replace(entry, parent_id=parent_id, seq=seq, timestamp=timestamp)


def decode_provisioned_entry(value: dict[str, Any]) -> Entry:
    """Decode a provisioned entry: `parent_id`/`seq`/`timestamp` stay at their defaults."""
    return _decode_entry_payload(value)


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------


def _intent_to_wire(intent: OperationIntent) -> dict[str, JsonValue]:
    if isinstance(intent, RunIntent):
        payload: dict[str, JsonValue] = {
            "kind": "run",
            "originalPrompt": [_message_to_wire(m) for m in intent.original_prompt],
            "initialMessages": [provisioned_entry_to_wire(e) for e in intent.initial_messages],
        }
        if intent.system_prompt_override is not None:
            payload["systemPromptOverride"] = intent.system_prompt_override
        if intent.resume_data is not None:
            payload["resumeData"] = _encode_value(intent.resume_data)
        return payload
    if isinstance(intent, CompactionIntent):
        payload = {"kind": "compaction", "resultEntryId": intent.result_entry_id}
        if intent.custom_instructions is not None:
            payload["customInstructions"] = intent.custom_instructions
        return payload
    if isinstance(intent, NavigationIntent):
        payload = {"kind": "navigation", "targetId": intent.target_id, "summarize": intent.summarize}
        if intent.custom_instructions is not None:
            payload["customInstructions"] = intent.custom_instructions
        if intent.label is not None:
            payload["label"] = intent.label
        if intent.summary_entry_id is not None:
            payload["summaryEntryId"] = intent.summary_entry_id
        return payload
    raise TypeError(f"Cannot encode intent {intent!r}")


def _wire_to_intent(value: dict[str, Any]) -> OperationIntent:
    kind = value.get("kind")
    if kind not in OPERATION_KINDS:
        raise JsonlDecodeError("schema", f"has unknown operation kind {kind}")
    if kind == "run":
        return RunIntent(
            original_prompt=[_wire_to_message(m) for m in value.get("originalPrompt") or []],
            initial_messages=[decode_provisioned_entry(e) for e in value.get("initialMessages") or []],
            system_prompt_override=value.get("systemPromptOverride"),
            resume_data=value.get("resumeData"),
        )
    if kind == "compaction":
        return CompactionIntent(
            result_entry_id=_require_string(value.get("resultEntryId"), "resultEntryId"),
            custom_instructions=value.get("customInstructions"),
        )
    return NavigationIntent(
        target_id=_require_nullable_id(value.get("targetId"), "targetId"),
        summarize=bool(value.get("summarize")),
        custom_instructions=value.get("customInstructions"),
        label=value.get("label"),
        summary_entry_id=value.get("summaryEntryId"),
    )


def _record_payload_to_wire(record: LaneRecord) -> dict[str, JsonValue]:
    """Fields shared by every record variant, excluding `seq`/`timestamp` (see
    `_entry_payload_to_wire` for the entry-side equivalent)."""
    payload: dict[str, JsonValue] = {"type": record.type, "id": record.id, "lane": record.lane}
    if isinstance(record, OperationStartedRecord):
        payload["sourceLeafId"] = record.source_leaf_id
        payload["intent"] = _intent_to_wire(record.intent)
    elif isinstance(record, AbortRequestedRecord):
        payload["runId"] = record.run_id
    elif isinstance(record, OperationFinishedRecord):
        payload["runId"] = record.run_id
        payload["outcome"] = record.outcome
        if record.error is not None:
            payload["error"] = {"code": record.error.code, "message": record.error.message}
    elif isinstance(record, StepAttemptRecord):
        payload["runId"] = record.run_id
        payload["step"] = record.step
        payload["attempt"] = record.attempt
        payload["resultEntryId"] = record.result_entry_id
        if record.compaction_reason is not None:
            payload["compactionReason"] = record.compaction_reason
    elif isinstance(record, ToolStartedRecord):
        payload["runId"] = record.run_id
        payload["assistantEntryId"] = record.assistant_entry_id
        payload["toolIndex"] = record.tool_index
        payload["toolCallId"] = record.tool_call_id
        payload["toolName"] = record.tool_name
        payload["effectiveArgs"] = _encode_value(record.effective_args)
        payload["resultEntryId"] = record.result_entry_id
        payload["replay"] = record.replay
    elif isinstance(record, QueueEnqueuedRecord):
        payload["queue"] = record.queue
        payload["target"] = provisioned_entry_to_wire(record.target)
        if record.run_id is not None:
            payload["runId"] = record.run_id
    elif isinstance(record, QueueCancelledRecord):
        payload["entryId"] = record.entry_id
        if record.run_id is not None:
            payload["runId"] = record.run_id
    elif isinstance(record, WriteDeferredRecord):
        payload["runId"] = record.run_id
        payload["target"] = provisioned_entry_to_wire(record.target)
    elif isinstance(record, UsageRecord):
        payload["cause"] = record.cause
        payload["usage"] = _usage_to_wire(record.usage)
        if record.run_id is not None:
            payload["runId"] = record.run_id
        if record.entry_id is not None:
            payload["entryId"] = record.entry_id
        if record.attempt is not None:
            payload["attempt"] = record.attempt
        if record.stop_reason is not None:
            payload["stopReason"] = record.stop_reason
        if record.tool_call_id is not None:
            payload["toolCallId"] = record.tool_call_id
        if record.details is not None:
            payload["details"] = _encode_value(record.details)
    else:
        raise TypeError(f"Cannot encode record {record!r}")
    return payload


def record_to_wire(record: LaneRecord) -> dict[str, JsonValue]:
    payload = _record_payload_to_wire(record)
    payload["seq"] = record.seq
    payload["timestamp"] = record.timestamp
    return payload


def _decode_record_payload(value: dict[str, Any]) -> LaneRecord:
    record_type = _require_string(value.get("type"), "record type")
    if record_type not in RECORD_TYPES:
        raise JsonlDecodeError("schema", f"has unknown record type {record_type}")
    id_ = _require_string(value.get("id"), "id")
    lane = _require_string(value.get("lane"), "lane")
    if record_type == "operation_started":
        intent_value = value.get("intent")
        if not _is_object(intent_value):
            raise JsonlDecodeError("schema", "has invalid intent")
        return OperationStartedRecord(
            id=id_,
            lane=lane,
            source_leaf_id=_require_nullable_id(value.get("sourceLeafId"), "sourceLeafId"),
            intent=_wire_to_intent(intent_value),
        )
    if record_type == "abort_requested":
        return AbortRequestedRecord(id=id_, lane=lane, run_id=_require_string(value.get("runId"), "runId"))
    if record_type == "operation_finished":
        run_id = _require_string(value.get("runId"), "runId")
        error_value = value.get("error")
        error = (
            None
            if error_value is None
            else OperationFinishedError(code=error_value.get("code", ""), message=error_value.get("message", ""))
        )
        return OperationFinishedRecord(id=id_, lane=lane, run_id=run_id, outcome=value.get("outcome"), error=error)
    if record_type == "step_attempt":
        return StepAttemptRecord(
            id=id_,
            lane=lane,
            run_id=_require_string(value.get("runId"), "runId"),
            step=value.get("step"),
            attempt=value.get("attempt", 0),
            result_entry_id=_require_string(value.get("resultEntryId"), "resultEntryId"),
            compaction_reason=value.get("compactionReason"),
        )
    if record_type == "tool_started":
        return ToolStartedRecord(
            id=id_,
            lane=lane,
            run_id=_require_string(value.get("runId"), "runId"),
            assistant_entry_id=_require_string(value.get("assistantEntryId"), "assistantEntryId"),
            tool_index=value.get("toolIndex", 0),
            tool_call_id=_require_string(value.get("toolCallId"), "toolCallId"),
            tool_name=_require_string(value.get("toolName"), "toolName"),
            effective_args=value.get("effectiveArgs") or {},
            result_entry_id=_require_string(value.get("resultEntryId"), "resultEntryId"),
            replay=value.get("replay"),
        )
    if record_type == "queue_enqueued":
        return QueueEnqueuedRecord(
            id=id_,
            lane=lane,
            queue=value.get("queue"),
            target=decode_provisioned_entry(value.get("target") or {}),
            run_id=value.get("runId"),
        )
    if record_type == "queue_cancelled":
        return QueueCancelledRecord(
            id=id_,
            lane=lane,
            entry_id=_require_string(value.get("entryId"), "entryId"),
            run_id=value.get("runId"),
        )
    if record_type == "write_deferred":
        return WriteDeferredRecord(
            id=id_,
            lane=lane,
            run_id=_require_string(value.get("runId"), "runId"),
            target=decode_provisioned_entry(value.get("target") or {}),
        )
    if record_type == "usage":
        return UsageRecord(
            id=id_,
            lane=lane,
            usage=_wire_to_usage(value["usage"]) if "usage" in value else Usage(),
            cause=value.get("cause"),
            run_id=value.get("runId"),
            entry_id=value.get("entryId"),
            attempt=value.get("attempt"),
            stop_reason=value.get("stopReason"),
            tool_call_id=value.get("toolCallId"),
            details=value.get("details"),
        )
    raise JsonlDecodeError("schema", f"has unknown record type {record_type}")


def decode_record(value: dict[str, Any], seq: int) -> LaneRecord:
    record = _decode_record_payload(value)
    timestamp = _require_timestamp(value.get("timestamp"))
    return replace(record, seq=seq, timestamp=timestamp)


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------


def decode_header(line: str) -> JsonlV4Header:
    value = _parse_object(line)
    if value.get("kind") != "header":
        raise JsonlDecodeError("schema", "is not a header")
    if value.get("version") != 4:
        raise JsonlDecodeError("schema", "has unsupported session version")
    parent_session_id = value.get("parentSessionId")
    if parent_session_id is not None and not isinstance(parent_session_id, str):
        raise JsonlDecodeError("schema", "has invalid parentSessionId")
    legacy_parent_session_path = value.get("legacyParentSessionPath")
    if legacy_parent_session_path is not None and not isinstance(legacy_parent_session_path, str):
        raise JsonlDecodeError("schema", "has invalid legacyParentSessionPath")
    if parent_session_id is not None and legacy_parent_session_path is not None:
        raise JsonlDecodeError("schema", "has both parentSessionId and legacyParentSessionPath")
    metadata_value = value.get("metadata")
    if metadata_value is not None and not _is_object(metadata_value):
        raise JsonlDecodeError("schema", "has invalid metadata")
    return JsonlV4Header(
        id=_require_string(value.get("id"), "id"),
        created_at=_require_timestamp(value.get("createdAt")),
        cwd=_require_string(value.get("cwd"), "cwd"),
        parent_session_id=parent_session_id,
        legacy_parent_session_path=legacy_parent_session_path,
        metadata=metadata_value,
    )


def parse_header(line: str) -> JsonlV4Header:
    return decode_header(line)


def encode_header(header: JsonlV4Header) -> str:
    payload: dict[str, JsonValue] = {
        "kind": "header",
        "version": 4,
        "id": header.id,
        "createdAt": header.created_at,
        "cwd": header.cwd,
    }
    if header.parent_session_id is not None:
        payload["parentSessionId"] = header.parent_session_id
    if header.legacy_parent_session_path is not None:
        payload["legacyParentSessionPath"] = header.legacy_parent_session_path
    if header.metadata is not None:
        payload["metadata"] = header.metadata
    return f"{json.dumps(payload)}\n"


def metadata_from_header(header: JsonlV4Header, path: str, modified_at: int) -> JsonlSessionMetadata:
    return JsonlSessionMetadata(
        id=header.id,
        created_at=header.created_at,
        cwd=header.cwd,
        path=path,
        modified_at=modified_at,
        source_format=4,
        parent_session_id=header.parent_session_id,
        legacy_parent_session_path=header.legacy_parent_session_path,
        metadata=header.metadata,
    )


# --------------------------------------------------------------------------
# Mutations
# --------------------------------------------------------------------------


def decode_mutation(line: str) -> SessionMutation:
    value = _parse_object(line)
    seq = _require_sequence(value.get("seq"))
    kind = value.get("kind")
    if kind == "entry":
        lane = value.get("lane")
        entry = decode_entry(value, seq)
        return EntryMutation(entry=entry, lane=None if lane is None else _require_string(lane, "lane"))
    if kind == "record":
        return RecordMutation(record=decode_record(value, seq))
    if kind == "lane":
        return LaneMutation(
            seq=seq,
            lane=_require_string(value.get("lane"), "lane"),
            leaf_id=_require_nullable_id(value.get("leafId"), "leafId"),
        )
    if kind == "fact":
        fact = value.get("fact")
        if fact == "name":
            # TS: `value.name !== undefined && typeof value.name !== "string"`
            # throws. An absent key (`undefined`) is allowed and means "clear
            # the name"; an explicit JSON `null` is present-but-not-a-string
            # and must be rejected the same as any other non-string value.
            if "name" in value and not isinstance(value["name"], str):
                raise JsonlDecodeError("schema", "has invalid name")
            return NameFactMutation(seq=seq, name=value.get("name"))
        if fact == "label":
            if "label" in value and not isinstance(value["label"], str):
                raise JsonlDecodeError("schema", "has invalid label")
            return LabelFactMutation(
                seq=seq, target_id=_require_string(value.get("targetId"), "targetId"), label=value.get("label")
            )
        raise JsonlDecodeError("schema", "has unknown fact type")
    raise JsonlDecodeError("schema", "has unknown mutation kind")


def parse_mutation(line: str) -> SessionMutation:
    return decode_mutation(line)


def encode_mutation(mutation: SessionMutation) -> str:
    if isinstance(mutation, EntryMutation):
        payload: dict[str, JsonValue] = {"kind": "entry"}
        if mutation.lane is not None:
            payload["lane"] = mutation.lane
        payload.update(entry_to_wire(mutation.entry))
        return f"{json.dumps(payload)}\n"
    if isinstance(mutation, RecordMutation):
        payload = {"kind": "record", **record_to_wire(mutation.record)}
        return f"{json.dumps(payload)}\n"
    if isinstance(mutation, LaneMutation):
        payload = {"kind": "lane", "seq": mutation.seq, "lane": mutation.lane, "leafId": mutation.leaf_id}
        return f"{json.dumps(payload)}\n"
    if isinstance(mutation, NameFactMutation):
        payload = {"kind": "fact", "seq": mutation.seq, "fact": "name"}
        if mutation.name is not None:
            payload["name"] = mutation.name
        return f"{json.dumps(payload)}\n"
    if isinstance(mutation, LabelFactMutation):
        payload = {"kind": "fact", "seq": mutation.seq, "fact": "label", "targetId": mutation.target_id}
        if mutation.label is not None:
            payload["label"] = mutation.label
        return f"{json.dumps(payload)}\n"
    raise TypeError(f"Cannot encode mutation {mutation!r}")


__all__ = [
    "decode_entry",
    "decode_header",
    "decode_mutation",
    "decode_provisioned_entry",
    "decode_record",
    "encode_header",
    "encode_mutation",
    "entry_to_wire",
    "metadata_from_header",
    "parse_header",
    "parse_mutation",
    "provisioned_entry_to_wire",
    "record_to_wire",
]
