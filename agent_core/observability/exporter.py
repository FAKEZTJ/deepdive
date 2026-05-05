from __future__ import annotations

import hashlib
import json
from typing import Any

from agent_core.observability.pricing import estimate_cost
from agent_core.persistence.session_store import EventRecord, SessionStore
from agent_core.runtime.events import (
    ContextCompressed,
    LLMCallCompleted,
    LLMCallStarted,
    StepCompleted,
    StepStarted,
    ToolCallCompleted,
    ToolCallStarted,
)


async def export_session_as_otel_json(
    store: SessionStore,
    session_id: str,
) -> dict[str, Any]:
    event_records = await store.get_event_records(session_id)
    record = await store.get_session(session_id)
    if record is None:
        raise ValueError(f"Session {session_id} not found")

    trace_id = _resolve_trace_id(session_id, event_records)
    run_span_id = _resolve_run_span_id(session_id, event_records)
    spans: list[dict[str, Any]] = [
        {
            "trace_id": trace_id,
            "span_id": run_span_id,
            "parent_span_id": None,
            "name": "agent_run",
            "start_time": record.created_at,
            "end_time": record.updated_at,
            "attributes": {
                "agent.session_id": session_id,
                "agent.total_steps": record.total_steps,
                "agent.total_input_tokens": record.total_usage.input_tokens,
                "agent.total_output_tokens": record.total_usage.output_tokens,
                "agent.total_cost_usd": record.total_cost_usd,
                "agent.stop_reason": record.stop_reason,
                "agent.status": record.status,
            },
        }
    ]

    step_spans: dict[int, dict[str, Any]] = {}
    llm_pending: dict[int, float] = {}
    tool_pending: dict[str, dict[str, Any]] = {}
    tool_dispatch_pending: dict[int, dict[str, Any]] = {}

    for event_record in event_records:
        event = event_record.event
        event_time = event_record.created_at

        match event:
            case StepStarted(step=step):
                step_spans[step] = {
                    "trace_id": trace_id,
                    "span_id": event_record.span_id or _make_span_id(session_id, f"step:{step}"),
                    "parent_span_id": event_record.parent_span_id or run_span_id,
                    "name": "step",
                    "start_time": event_time,
                    "attributes": {"step.number": step},
                }
            case StepCompleted(step=step):
                if step in step_spans:
                    span = step_spans.pop(step)
                    span["end_time"] = event_time
                    spans.append(span)

            case LLMCallStarted(step=step):
                llm_pending[step] = event_time
            case LLMCallCompleted(step=step, usage=usage):
                start_time = llm_pending.pop(step, event_time)
                step_parent_id = _lookup_step_parent_span_id(step_spans, step, session_id)
                provider = event.provider or record.metadata.get("provider")
                model = event.model or record.metadata.get("model")
                llm_cost = event.cost_usd
                if llm_cost is None and provider and model:
                    llm_cost = estimate_cost(
                        provider=provider,
                        model=model,
                        input_tokens=usage.input_tokens,
                        output_tokens=usage.output_tokens,
                    )
                attributes: dict[str, Any] = {
                    "step.number": step,
                    "llm.usage.input_tokens": usage.input_tokens,
                    "llm.usage.output_tokens": usage.output_tokens,
                }
                if provider:
                    attributes["llm.provider"] = provider
                if model:
                    attributes["llm.model"] = model
                if llm_cost is not None:
                    attributes["llm.cost_usd"] = llm_cost
                spans.append(
                    {
                        "trace_id": trace_id,
                        "span_id": _make_span_id(session_id, f"llm:{step}:{start_time}"),
                        "parent_span_id": step_parent_id,
                        "name": "llm_call",
                        "start_time": start_time,
                        "end_time": event_time,
                        "attributes": attributes,
                    }
                )

            case ToolCallStarted(step=step, tool_call_id=tool_call_id, tool_name=tool_name, input=tool_input):
                tool_dispatch = tool_dispatch_pending.setdefault(
                    step,
                    {
                        "trace_id": trace_id,
                        "span_id": _make_span_id(session_id, f"tool_dispatch:{step}"),
                        "parent_span_id": _lookup_step_parent_span_id(step_spans, step, session_id),
                        "name": "tool_dispatch",
                        "start_time": event_time,
                        "attributes": {"step.number": step, "tools.count": 0},
                    },
                )
                tool_dispatch["start_time"] = min(tool_dispatch["start_time"], event_time)
                tool_dispatch["attributes"]["tools.count"] += 1
                tool_pending[tool_call_id] = {
                    "trace_id": trace_id,
                    "span_id": _make_span_id(session_id, f"tool:{tool_call_id}"),
                    "parent_span_id": tool_dispatch["span_id"],
                    "name": "tool_call",
                    "start_time": event_time,
                    "attributes": {
                        "step.number": step,
                        "tool.id": tool_call_id,
                        "tool.name": tool_name,
                        "tool.input": json.dumps(tool_input, ensure_ascii=False)[:500],
                    },
                }
            case ToolCallCompleted(tool_call_id=tool_call_id, duration_ms=duration_ms, is_error=is_error, output=output):
                span = tool_pending.pop(tool_call_id, None)
                if span is None:
                    continue
                span["end_time"] = event_time
                span["attributes"]["tool.duration_ms"] = duration_ms
                span["attributes"]["tool.is_error"] = is_error
                span["attributes"]["tool.output_preview"] = output[:500]
                spans.append(span)
                step = int(span["attributes"]["step.number"])
                if step in tool_dispatch_pending:
                    tool_dispatch_pending[step]["end_time"] = event_time

            case ContextCompressed(step=step, new_message_count=new_message_count):
                spans.append(
                    {
                        "trace_id": trace_id,
                        "span_id": _make_span_id(session_id, f"context:{step}:{event_time}"),
                        "parent_span_id": _lookup_step_parent_span_id(step_spans, step, session_id),
                        "name": "context_compression",
                        "start_time": event_time,
                        "end_time": event_time,
                        "attributes": {
                            "step.number": step,
                            "context.new_message_count": new_message_count,
                        },
                    }
                )

    for dispatch_span in tool_dispatch_pending.values():
        dispatch_span.setdefault("end_time", dispatch_span["start_time"])
        spans.append(dispatch_span)
    for step_span in step_spans.values():
        step_span.setdefault("end_time", record.updated_at)
        spans.append(step_span)

    spans.sort(key=lambda span: (span["start_time"], span["name"], span["span_id"]))
    return {
        "session_id": session_id,
        "trace_id": trace_id,
        "spans": spans,
    }


def _resolve_trace_id(session_id: str, event_records: list[EventRecord]) -> str:
    for record in event_records:
        if record.trace_id is not None:
            return record.trace_id
    digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()
    return digest[:32]


def _resolve_run_span_id(session_id: str, event_records: list[EventRecord]) -> str:
    for record in event_records:
        if record.parent_span_id is not None:
            return record.parent_span_id
    return _make_span_id(session_id, "run")


def _lookup_step_parent_span_id(
    step_spans: dict[int, dict[str, Any]],
    step: int,
    session_id: str,
) -> str:
    step_span = step_spans.get(step)
    if step_span is not None:
        return str(step_span["span_id"])
    return _make_span_id(session_id, f"step:{step}")


def _make_span_id(session_id: str, suffix: str) -> str:
    digest = hashlib.sha256(f"{session_id}:{suffix}".encode("utf-8")).hexdigest()
    return digest[:16]
