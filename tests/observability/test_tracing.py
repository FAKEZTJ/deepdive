from __future__ import annotations

from typing import Any, Sequence

import pytest
from opentelemetry.sdk.trace.export import SpanExportResult, SpanExporter

from agent_core.observability import configure_tracing, shutdown_tracing
from agent_core.runtime.loop import AgentLoop
from agent_core.testing import FakeProvider
from agent_core.tools.registry import ToolRegistry
from agent_core.types import CompletionResponse, Message, ToolUseContent, Usage


class _CollectingExporter(SpanExporter):
    def __init__(self) -> None:
        self.spans: list[Any] = []

    def export(self, spans: Sequence[Any]) -> SpanExportResult:
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        return None


def _tool_use_response(*, tool_name: str = "noop") -> CompletionResponse:
    return CompletionResponse(
        message=Message(
            role="assistant",
            content=[ToolUseContent(id="call_1", name=tool_name, input={"value": "ok"})],
        ),
        finish_reason="tool_use",
        usage=Usage(input_tokens=2, output_tokens=3),
    )


def _stop_response(text: str = "done") -> CompletionResponse:
    return CompletionResponse(
        message=Message.assistant_text(text),
        finish_reason="stop",
        usage=Usage(input_tokens=1, output_tokens=1),
    )


@pytest.mark.anyio
async def test_tracing_creates_nested_spans_for_agent_loop():
    from tests.runtime.test_loop import _NoopTool

    exporter = _CollectingExporter()
    configure_tracing(exporter=exporter, use_batch_processor=False)
    try:
        provider = FakeProvider(
            scripted_responses=[
                _tool_use_response(),
                _stop_response(),
            ]
        )
        loop = AgentLoop(
            provider=provider,
            tools=ToolRegistry([_NoopTool()]),
        )

        result = await loop.run("trace this run")

        assert result.stop_reason == "finished"
    finally:
        shutdown_tracing()

    llm_call_spans = [span for span in exporter.spans if span.name == "llm_call"]
    step_spans = [span for span in exporter.spans if span.name == "step"]

    assert len(step_spans) == 2
    assert len(llm_call_spans) == 2

    agent_run = next(span for span in exporter.spans if span.name == "agent_run")
    step_one = next(span for span in step_spans if span.attributes["step.number"] == 1)
    tool_dispatch = next(span for span in exporter.spans if span.name == "tool_dispatch")
    tool_call = next(span for span in exporter.spans if span.name == "tool_call")
    first_llm_call = llm_call_spans[0]

    assert step_one.parent is not None
    assert step_one.parent.span_id == agent_run.context.span_id
    assert first_llm_call.parent is not None
    assert first_llm_call.parent.span_id == step_one.context.span_id
    assert tool_dispatch.parent is not None
    assert tool_dispatch.parent.span_id == step_one.context.span_id
    assert tool_call.parent is not None
    assert tool_call.parent.span_id == tool_dispatch.context.span_id

    assert first_llm_call.attributes["llm.provider"] == "fake"
    assert first_llm_call.attributes["llm.model"] == "fake"
    assert first_llm_call.attributes["llm.usage.input_tokens"] == 2
    assert first_llm_call.attributes["llm.usage.output_tokens"] == 3
    assert tool_dispatch.attributes["tools.count"] == 1
    assert tool_call.attributes["tool.name"] == "noop"


@pytest.mark.anyio
async def test_rejected_tool_call_marks_span_as_error():
    exporter = _CollectingExporter()
    configure_tracing(exporter=exporter, use_batch_processor=False)
    try:
        provider = FakeProvider(
            scripted_responses=[
                _tool_use_response(tool_name="missing"),
                _stop_response(),
            ]
        )
        loop = AgentLoop(
            provider=provider,
            tools=ToolRegistry(),
        )

        result = await loop.run("reject missing tool")

        assert result.stop_reason == "finished"
    finally:
        shutdown_tracing()

    tool_call = next(span for span in exporter.spans if span.name == "tool_call")
    assert tool_call.status.status_code.name == "ERROR"
