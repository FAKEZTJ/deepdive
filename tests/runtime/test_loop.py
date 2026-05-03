from __future__ import annotations

import sys
from typing import Literal

import pytest
from pydantic import BaseModel

from agent_core.runtime.events import (
    RunCompleted,
    StepStarted,
    ToolCallCompleted,
    ToolCallStarted,
)
from agent_core.runtime.loop import AgentLoop, Budget
from agent_core.testing import FakeProvider
from agent_core.tools.base import Tool, ToolResult
from agent_core.tools.builtins.shell_exec import ShellExecTool
from agent_core.tools.registry import ToolRegistry
from agent_core.types import (
    CompletionResponse,
    Message,
    ToolResultContent,
    ToolUseContent,
    Usage,
)


class _NoopParams(BaseModel):
    value: str = "ok"


class _NoopTool(Tool[_NoopParams]):
    name = "noop"
    description = "Return a fixed string."
    params_model = _NoopParams

    async def execute(self, params: _NoopParams) -> ToolResult:
        return ToolResult(content=f"tool:{params.value}")


class _CrashParams(BaseModel):
    value: str


class _CrashTool(Tool[_CrashParams]):
    name = "crash"
    description = "Always crash."
    params_model = _CrashParams

    async def execute(self, params: _CrashParams) -> ToolResult:
        raise RuntimeError("boom")


class _StrictParams(BaseModel):
    mode: Literal["ok"]


class _StrictTool(Tool[_StrictParams]):
    name = "strict"
    description = "Validate params strictly."
    params_model = _StrictParams

    async def execute(self, params: _StrictParams) -> ToolResult:
        return ToolResult(content="strict:ok")


def _tool_use_response(*, input_tokens: int, output_tokens: int) -> CompletionResponse:
    return CompletionResponse(
        message=Message(
            role="assistant",
            content=[ToolUseContent(id="call_1", name="noop", input={"value": "ok"})],
        ),
        finish_reason="tool_use",
        usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def _parallel_tool_use_response() -> CompletionResponse:
    return CompletionResponse(
        message=Message(
            role="assistant",
            content=[
                ToolUseContent(id="call_1", name="noop", input={"value": "left"}),
                ToolUseContent(id="call_2", name="noop", input={"value": "right"}),
            ],
        ),
        finish_reason="tool_use",
        usage=Usage(input_tokens=10, output_tokens=5),
    )


def _single_tool_response(
    *,
    tool_name: str,
    tool_input: dict[str, object],
    tool_id: str = "call_1",
    input_tokens: int = 1,
    output_tokens: int = 1,
) -> CompletionResponse:
    return CompletionResponse(
        message=Message(
            role="assistant",
            content=[ToolUseContent(id=tool_id, name=tool_name, input=tool_input)],
        ),
        finish_reason="tool_use",
        usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def _stop_response(text: str = "Done.") -> CompletionResponse:
    return CompletionResponse(
        message=Message.assistant_text(text),
        finish_reason="stop",
        usage=Usage(input_tokens=1, output_tokens=1),
    )


@pytest.mark.anyio
async def test_max_steps_stops_after_completed_first_step():
    provider = FakeProvider(
        scripted_responses=[_tool_use_response(input_tokens=1, output_tokens=1)]
    )
    loop = AgentLoop(
        provider=provider,
        tools=ToolRegistry([_NoopTool()]),
        budget=Budget(max_steps=1),
    )

    events = [event async for event in loop.run_stream("do one step")]

    step_starts = [event.step for event in events if isinstance(event, StepStarted)]
    final = next(event for event in events if isinstance(event, RunCompleted))

    assert step_starts == [1]
    assert final.stop_reason == "max_steps"
    assert final.total_steps == 1
    assert len(provider.calls) == 1


@pytest.mark.anyio
async def test_non_step_budget_does_not_start_a_new_step_after_first_step_completed():
    provider = FakeProvider(
        scripted_responses=[_tool_use_response(input_tokens=2, output_tokens=3)]
    )
    loop = AgentLoop(
        provider=provider,
        tools=ToolRegistry([_NoopTool()]),
        budget=Budget(max_steps=5, max_tokens=5),
    )

    events = [event async for event in loop.run_stream("stop on token budget")]

    step_starts = [event.step for event in events if isinstance(event, StepStarted)]
    final = next(event for event in events if isinstance(event, RunCompleted))

    assert step_starts == [1]
    assert final.stop_reason == "max_tokens"
    assert final.total_steps == 1
    assert len(provider.calls) == 1


@pytest.mark.anyio
async def test_parallel_tool_calls_message_shape():
    provider = FakeProvider(
        scripted_responses=[
            _parallel_tool_use_response(),
            _stop_response(),
        ]
    )
    loop = AgentLoop(
        provider=provider,
        tools=ToolRegistry([_NoopTool()]),
    )

    result = await loop.run("run two tools")

    assert result.stop_reason == "finished"
    assert len(provider.calls) == 2

    second_call_messages = provider.calls[1]["messages"]
    tool_message = second_call_messages[-1]

    assert tool_message.role == "tool"
    assert len(tool_message.content) == 2
    assert all(isinstance(block, ToolResultContent) for block in tool_message.content)
    assert [block.tool_use_id for block in tool_message.content] == ["call_1", "call_2"]
    assert [block.content for block in tool_message.content] == ["tool:left", "tool:right"]


@pytest.mark.anyio
async def test_tool_call_events_emit_started_and_completed_pairs():
    provider = FakeProvider(
        scripted_responses=[
            _parallel_tool_use_response(),
            _stop_response(),
        ]
    )
    loop = AgentLoop(
        provider=provider,
        tools=ToolRegistry([_NoopTool()]),
    )

    events = [event async for event in loop.run_stream("run two tools")]

    started = [event for event in events if isinstance(event, ToolCallStarted)]
    completed = [event for event in events if isinstance(event, ToolCallCompleted)]

    assert [(event.step, event.tool_call_id, event.tool_name) for event in started] == [
        (1, "call_1", "noop"),
        (1, "call_2", "noop"),
    ]
    assert [(event.step, event.tool_call_id, event.tool_name) for event in completed] == [
        (1, "call_1", "noop"),
        (1, "call_2", "noop"),
    ]


@pytest.mark.anyio
async def test_llm_error_still_emits_run_completed():
    provider = FakeProvider(scripted_responses=[])
    loop = AgentLoop(provider=provider, tools=ToolRegistry())

    events = [event async for event in loop.run_stream("fail immediately")]

    final = next(event for event in events if isinstance(event, RunCompleted))

    assert final.stop_reason == "error"
    assert "FakeProvider exhausted scripted responses" in final.final_message.content[0].text


@pytest.mark.anyio
async def test_missing_tool_returns_error_tool_result_and_completes_run():
    provider = FakeProvider(
        scripted_responses=[
            _single_tool_response(tool_name="missing", tool_input={}),
            _stop_response(),
        ]
    )
    loop = AgentLoop(provider=provider, tools=ToolRegistry())

    result = await loop.run("missing tool")

    tool_message = provider.calls[1]["messages"][-1]

    assert result.stop_reason == "finished"
    assert tool_message.role == "tool"
    assert tool_message.content[0].is_error is True
    assert "Tool 'missing' not found" in tool_message.content[0].content


@pytest.mark.anyio
async def test_invalid_tool_params_return_error_tool_result_and_completes_run():
    provider = FakeProvider(
        scripted_responses=[
            _single_tool_response(tool_name="strict", tool_input={"mode": "bad"}),
            _stop_response(),
        ]
    )
    loop = AgentLoop(provider=provider, tools=ToolRegistry([_StrictTool()]))

    result = await loop.run("bad params")

    tool_message = provider.calls[1]["messages"][-1]

    assert result.stop_reason == "finished"
    assert tool_message.role == "tool"
    assert tool_message.content[0].is_error is True
    assert "Invalid params:" in tool_message.content[0].content


@pytest.mark.anyio
async def test_tool_crash_returns_error_tool_result_and_completes_run():
    provider = FakeProvider(
        scripted_responses=[
            _single_tool_response(tool_name="crash", tool_input={"value": "x"}),
            _stop_response(),
        ]
    )
    loop = AgentLoop(provider=provider, tools=ToolRegistry([_CrashTool()]))

    result = await loop.run("crash tool")

    tool_message = provider.calls[1]["messages"][-1]

    assert result.stop_reason == "finished"
    assert tool_message.role == "tool"
    assert tool_message.content[0].is_error is True
    assert "Tool crashed: boom" in tool_message.content[0].content


@pytest.mark.anyio
async def test_day2_acceptance_can_count_python_files_with_shell_exec(tmp_path):
    (tmp_path / "a.py").write_text("print('a')\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("print('b')\n", encoding="utf-8")
    (tmp_path / "note.txt").write_text("ignore\n", encoding="utf-8")
    command = (
        f'"{sys.executable}" -c '
        f'"from pathlib import Path; print(sum(1 for p in Path(r\'{tmp_path}\').iterdir() if p.suffix == \'.py\'))"'
    )
    provider = FakeProvider(
        scripted_responses=[
            _single_tool_response(
                tool_name="shell_exec",
                tool_input={"command": command, "timeout_seconds": 5.0},
            ),
            _stop_response("Found 2 Python files."),
        ]
    )
    loop = AgentLoop(provider=provider, tools=ToolRegistry([ShellExecTool()]))

    result = await loop.run("Count Python files in the directory")

    tool_message = provider.calls[1]["messages"][-1]

    assert result.stop_reason == "finished"
    assert result.final_message.content[0].text == "Found 2 Python files."
    assert tool_message.role == "tool"
    assert "STDOUT:\n2" in tool_message.content[0].content
