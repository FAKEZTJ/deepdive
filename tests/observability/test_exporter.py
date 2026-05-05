from __future__ import annotations

import pytest

from agent_core.observability.exporter import export_session_as_otel_json
from agent_core.observability import SpanScope, configure_tracing, shutdown_tracing
from agent_core.runtime.events import LLMCallCompleted, LLMCallStarted, StepCompleted, StepStarted
from agent_core.types import Message, Usage


@pytest.mark.anyio
async def test_export_session_as_otel_json_builds_run_step_and_llm_spans(tmp_path):
    from agent_core.persistence.session_store import SessionStore

    configure_tracing(use_batch_processor=False)
    store = SessionStore(str(tmp_path / "agent.db"))
    await store.initialize()
    try:
        session_id = await store.create_session(metadata={"provider": "openai", "model": "gpt-4o-mini"})
        async with SpanScope("agent_run"):
            async with SpanScope("step"):
                await store.append_event(session_id, StepStarted(step=1))
                await store.append_event(session_id, LLMCallStarted(step=1))
                await store.append_event(
                    session_id,
                    LLMCallCompleted(
                        step=1,
                        message=Message.assistant_text("hi"),
                        usage=Usage(input_tokens=10, output_tokens=20),
                        provider="openai",
                        model="gpt-4o-mini",
                        cost_usd=(10 * 0.15 + 20 * 0.60) / 1_000_000,
                    ),
                )
                await store.append_event(session_id, StepCompleted(step=1))
        await store.update_session_state(
            session_id,
            total_steps=1,
            usage_delta=Usage(input_tokens=10, output_tokens=20),
            cost_delta_usd=(10 * 0.15 + 20 * 0.60) / 1_000_000,
            status="completed",
            stop_reason="finished",
        )

        payload = await export_session_as_otel_json(store, session_id)
    finally:
        await store.close()
        shutdown_tracing()

    assert payload["session_id"] == session_id
    spans = payload["spans"]
    names = [span["name"] for span in spans]
    assert "agent_run" in names
    assert "step" in names
    assert "llm_call" in names

    llm_span = next(span for span in spans if span["name"] == "llm_call")
    assert llm_span["attributes"]["llm.usage.input_tokens"] == 10
    assert llm_span["attributes"]["llm.usage.output_tokens"] == 20
    assert llm_span["attributes"]["llm.provider"] == "openai"
    assert llm_span["attributes"]["llm.model"] == "gpt-4o-mini"
