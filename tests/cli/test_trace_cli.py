from __future__ import annotations

import asyncio
import json

from click.testing import CliRunner

from agent_core.cli.trace import cli
from agent_core.runtime.events import LLMCallCompleted, LLMCallStarted, StepCompleted, StepStarted
from agent_core.types import Message, Usage


async def _seed_trace_db(db_path: str) -> str:
    from agent_core.persistence.session_store import SessionStore

    store = SessionStore(db_path)
    await store.initialize()
    try:
        session_id = await store.create_session()
        await store.append_event(session_id, StepStarted(step=1))
        await store.append_event(session_id, LLMCallStarted(step=1))
        await store.append_event(
            session_id,
            LLMCallCompleted(
                step=1,
                message=Message.assistant_text("done"),
                usage=Usage(input_tokens=12, output_tokens=8),
                provider="openai",
                model="gpt-4o-mini",
                cost_usd=0.0023,
            ),
        )
        await store.append_event(session_id, StepCompleted(step=1))
        await store.update_session_state(
            session_id,
            total_steps=1,
            usage_delta=Usage(input_tokens=12, output_tokens=8),
            cost_delta_usd=0.0023,
            status="completed",
            stop_reason="finished",
        )
        return session_id
    finally:
        await store.close()


def test_sessions_command_lists_sessions(tmp_path):
    db_path = str(tmp_path / "agent.db")
    session_id = asyncio.run(_seed_trace_db(db_path))

    runner = CliRunner()
    result = runner.invoke(cli, ["sessions", "--db", db_path])

    assert result.exit_code == 0
    assert "Sessions" in result.output
    assert session_id[:8] in result.output
    assert "completed" in result.output


def test_trace_command_prints_tree(tmp_path):
    db_path = str(tmp_path / "agent.db")
    session_id = asyncio.run(_seed_trace_db(db_path))

    runner = CliRunner()
    result = runner.invoke(cli, ["trace", session_id, "--db", db_path])

    assert result.exit_code == 0
    assert "agent_run" in result.output
    assert "step #1" in result.output
    assert "llm_call" in result.output
    assert "openai:gpt-4o-mini" in result.output


def test_trace_command_json_output_exports_spans(tmp_path):
    db_path = str(tmp_path / "agent.db")
    session_id = asyncio.run(_seed_trace_db(db_path))

    runner = CliRunner()
    result = runner.invoke(cli, ["trace", session_id, "--db", db_path, "--json-output"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["session_id"] == session_id
    assert any(span["name"] == "agent_run" for span in payload["spans"])
