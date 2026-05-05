from __future__ import annotations

import pytest

from agent_core.observability import SpanScope, configure_tracing, shutdown_tracing
from agent_core.runtime.events import LLMCallCompleted, StepCompleted, StepStarted
from agent_core.types import Message, Usage


@pytest.mark.anyio
async def test_create_session_and_round_trip_metadata(tmp_path):
    from agent_core.persistence.session_store import SessionStore

    store = SessionStore(str(tmp_path / "agent.db"))
    await store.initialize()
    try:
        session_id = await store.create_session(
            system_prompt="You are helpful.",
            metadata={"source": "test"},
        )

        record = await store.get_session(session_id)

        assert record is not None
        assert record.id == session_id
        assert record.status == "running"
        assert record.system_prompt == "You are helpful."
        assert record.metadata == {"source": "test"}
        assert record.total_steps == 0
        assert record.total_usage == Usage()
        assert record.total_cost_usd == 0.0
        assert record.stop_reason is None
        assert record.error_message is None
    finally:
        await store.close()


@pytest.mark.anyio
async def test_append_and_replace_messages(tmp_path):
    from agent_core.persistence.session_store import SessionStore

    store = SessionStore(str(tmp_path / "agent.db"))
    await store.initialize()
    try:
        session_id = await store.create_session()
        await store.append_message(session_id, Message.user("first"))
        await store.append_message(session_id, Message.assistant_text("second"))

        original = await store.get_messages(session_id)
        assert original == [
            Message.user("first"),
            Message.assistant_text("second"),
        ]

        compressed = [
            Message.user("first"),
            Message.assistant_text("summary"),
        ]
        await store.replace_messages(session_id, compressed)

        assert await store.get_messages(session_id) == compressed
    finally:
        await store.close()


@pytest.mark.anyio
async def test_append_and_get_events_round_trip(tmp_path):
    from agent_core.persistence.session_store import SessionStore

    store = SessionStore(str(tmp_path / "agent.db"))
    await store.initialize()
    try:
        session_id = await store.create_session()
        events = [
            StepStarted(step=1),
            LLMCallCompleted(
                step=1,
                message=Message.assistant_text("hi"),
                usage=Usage(input_tokens=3, output_tokens=5),
            ),
            StepCompleted(step=1),
        ]

        for event in events:
            await store.append_event(session_id, event)

        assert await store.get_events(session_id) == events
        assert await store.get_events(session_id, limit=2) == events[-2:]

        records = await store.get_event_records(session_id)
        assert [record.event for record in records] == events
        assert all(record.trace_id is None for record in records)
    finally:
        await store.close()


@pytest.mark.anyio
async def test_append_event_persists_trace_envelope_columns(tmp_path):
    from agent_core.persistence.session_store import SessionStore

    configure_tracing(enabled=True, use_batch_processor=False)
    store = SessionStore(str(tmp_path / "agent.db"))
    await store.initialize()
    try:
        session_id = await store.create_session()
        async with SpanScope("run"):
            await store.append_event(session_id, StepStarted(step=1))
            async with SpanScope("child"):
                await store.append_event(session_id, StepCompleted(step=1))

        records = await store.get_event_records(session_id)

        assert len(records) == 2
        assert records[0].trace_id is not None
        assert records[0].span_id is not None
        assert records[0].parent_span_id is None
        assert records[1].trace_id == records[0].trace_id
        assert records[1].span_id is not None
        assert records[1].parent_span_id == records[0].span_id
    finally:
        await store.close()
        shutdown_tracing()


@pytest.mark.anyio
async def test_update_session_state_accumulates_usage_and_status(tmp_path):
    from agent_core.persistence.session_store import SessionStore

    store = SessionStore(str(tmp_path / "agent.db"))
    await store.initialize()
    try:
        session_id = await store.create_session()

        await store.update_session_state(
            session_id,
            status="paused",
            total_steps=2,
            usage_delta=Usage(input_tokens=11, output_tokens=7),
            cost_delta_usd=0.123,
            stop_reason="max_steps",
        )
        await store.update_session_state(
            session_id,
            status="completed",
            usage_delta=Usage(input_tokens=2, output_tokens=3),
            cost_delta_usd=0.456,
        )

        record = await store.get_session(session_id)

        assert record is not None
        assert record.status == "completed"
        assert record.total_steps == 2
        assert record.total_usage == Usage(input_tokens=13, output_tokens=10)
        assert record.total_cost_usd == pytest.approx(0.579)
        assert record.stop_reason == "max_steps"
    finally:
        await store.close()


@pytest.mark.anyio
async def test_checkpoint_step_appends_messages_and_updates_totals_atomically(tmp_path):
    from agent_core.persistence.session_store import SessionStore

    store = SessionStore(str(tmp_path / "agent.db"))
    await store.initialize()
    try:
        session_id = await store.create_session()
        await store.append_message(session_id, Message.user("task"))

        new_messages = [
            Message.assistant_text("calling tool"),
            Message.tool_result("call_1", "tool output"),
        ]
        await store.checkpoint_step(
            session_id,
            new_messages=new_messages,
            completed_step=1,
            usage_delta=Usage(input_tokens=5, output_tokens=8),
        )

        assert await store.get_messages(session_id) == [
            Message.user("task"),
            *new_messages,
        ]

        record = await store.get_session(session_id)
        assert record is not None
        assert record.total_steps == 1
        assert record.total_usage == Usage(input_tokens=5, output_tokens=8)
    finally:
        await store.close()


@pytest.mark.anyio
async def test_list_sessions_filters_by_status_and_orders_recent_first(tmp_path):
    from agent_core.persistence.session_store import SessionStore

    store = SessionStore(str(tmp_path / "agent.db"))
    await store.initialize()
    try:
        first = await store.create_session(metadata={"name": "first"})
        second = await store.create_session(metadata={"name": "second"})
        await store.update_session_state(first, status="completed")

        completed = await store.list_sessions(status="completed")
        all_sessions = await store.list_sessions()

        assert [record.id for record in completed] == [first]
        assert [record.id for record in all_sessions] == [first, second]
    finally:
        await store.close()
