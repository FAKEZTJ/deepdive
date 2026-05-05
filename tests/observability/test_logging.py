from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from agent_core.observability.logging import LoggingContext, configure_logging, get_logger
from agent_core.runtime.loop import AgentLoop
from agent_core.testing import FakeProvider
from agent_core.tools.base import Tool, ToolResult
from agent_core.tools.registry import ToolRegistry
from agent_core.types import CompletionResponse, Message, ToolUseContent, Usage


class _NoopParams(BaseModel):
    value: str = "ok"


class _LoggingTool(Tool[_NoopParams]):
    name = "logging_tool"
    description = "Emit a log from inside tool execution."
    params_model = _NoopParams

    async def execute(self, params: _NoopParams) -> ToolResult:
        del params
        get_logger(__name__).info("tool.execute.inside")
        return ToolResult(content="done")


def _tool_use_response() -> CompletionResponse:
    return CompletionResponse(
        message=Message(
            role="assistant",
            content=[ToolUseContent(id="call_1", name="logging_tool", input={"value": "ok"})],
        ),
        finish_reason="tool_use",
        usage=Usage(input_tokens=1, output_tokens=1),
    )


def _stop_response() -> CompletionResponse:
    return CompletionResponse(
        message=Message.assistant_text("done"),
        finish_reason="stop",
        usage=Usage(input_tokens=1, output_tokens=1),
    )


def test_logging_context_injects_bound_fields_and_resets(capsys: pytest.CaptureFixture[str]):
    configure_logging(json_output=True)
    logger = get_logger("test")

    with LoggingContext(session_id="session-1", step=2, llm_call_id="llm-1", tool_call_id="tool-1"):
        logger.info("log.inside", answer=42)
    logger.info("log.outside")

    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    inside = json.loads(lines[0])
    outside = json.loads(lines[1])

    assert inside["event"] == "log.inside"
    assert inside["session_id"] == "session-1"
    assert inside["step"] == 2
    assert inside["llm_call_id"] == "llm-1"
    assert inside["tool_call_id"] == "tool-1"
    assert inside["answer"] == 42

    assert outside["event"] == "log.outside"
    assert "session_id" not in outside
    assert "step" not in outside
    assert "llm_call_id" not in outside
    assert "tool_call_id" not in outside


@pytest.mark.anyio
async def test_tool_execution_logs_inherit_tool_call_context(
    tmp_path,
    capsys: pytest.CaptureFixture[str],
):
    from agent_core.persistence.session_store import SessionStore

    configure_logging(json_output=True)
    store = SessionStore(str(tmp_path / "agent.db"))
    await store.initialize()
    try:
        provider = FakeProvider(
            scripted_responses=[
                _tool_use_response(),
                _stop_response(),
            ]
        )
        loop = AgentLoop(
            provider=provider,
            tools=ToolRegistry([_LoggingTool()]),
            session_store=store,
        )

        await loop.run("run logging tool")
    finally:
        await store.close()

    payloads = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if line.strip()
    ]
    tool_log = next(payload for payload in payloads if payload["event"] == "tool.execute.inside")

    assert tool_log["session_id"]
    assert tool_log["step"] == 1
    assert tool_log["tool_call_id"] == "call_1"
