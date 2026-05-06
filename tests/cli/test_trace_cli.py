from __future__ import annotations

import asyncio
import json

from click.testing import CliRunner

from agent_core.apps.deepdive.report import DeepdiveReport
from agent_core.apps.deepdive.tools import SourceTracker
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


def test_run_command_invokes_deepdive_and_prints_report(tmp_path, monkeypatch):
    async def fake_deepdive_research(*args, **kwargs):
        return DeepdiveReport(
            topic="demo topic",
            markdown="# Demo\n\nHello world.",
            sources=SourceTracker(),
            total_steps=2,
            total_tokens=123,
            total_cost_usd=0.0,
            duration_seconds=1.5,
            session_id="session-123",
        )

    monkeypatch.setattr("agent_core.cli.trace.build_provider", lambda *args, **kwargs: object())
    monkeypatch.setattr("agent_core.cli.trace.deepdive_research", fake_deepdive_research)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "run",
            "demo topic",
            "--no-stream",
            "--provider",
            "openai",
            "--api-key",
            "test-key",
            "--db",
            str(tmp_path / "agent.db"),
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )

    assert result.exit_code == 0
    assert "Hello world." in result.output
    assert "Report saved to" in result.output
    assert "Session: session-123" in result.output


def test_deepdive_command_invokes_deepdive_and_prints_report(tmp_path, monkeypatch):
    async def fake_deepdive_research(*args, **kwargs):
        return DeepdiveReport(
            topic="demo topic",
            markdown="# Deepdive\n\nResearch body.",
            sources=SourceTracker(),
            total_steps=3,
            total_tokens=456,
            total_cost_usd=0.0012,
            duration_seconds=2.5,
            session_id="session-456",
        )

    monkeypatch.setattr("agent_core.cli.trace.build_provider", lambda *args, **kwargs: object())
    monkeypatch.setattr("agent_core.cli.trace.deepdive_research", fake_deepdive_research)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "deepdive",
            "demo",
            "topic",
            "--no-stream",
            "--provider",
            "anthropic",
            "--api-key",
            "test-key",
            "--db",
            str(tmp_path / "agent.db"),
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )

    assert result.exit_code == 0
    assert "Research body." in result.output
    assert "Report saved to" in result.output
    assert "Session: session-456" in result.output
