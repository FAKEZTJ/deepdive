from __future__ import annotations

import asyncio
import time
from typing import Literal

import pytest
from pydantic import BaseModel

from agent_core.runtime.dispatcher import ToolDispatcher
from agent_core.runtime.events import ToolCallCompleted, ToolCallStarted
from agent_core.tools.base import Tool, ToolResult
from agent_core.tools.registry import ToolRegistry
from agent_core.types import ToolResultContent, ToolUseContent


class _NoopParams(BaseModel):
    value: str = "ok"


class _NoopTool(Tool[_NoopParams]):
    name = "noop"
    description = "Return a fixed string."
    params_model = _NoopParams

    async def execute(self, params: _NoopParams) -> ToolResult:
        return ToolResult(content=f"tool:{params.value}")


class _StrictParams(BaseModel):
    mode: Literal["ok"]


class _StrictTool(Tool[_StrictParams]):
    name = "strict"
    description = "Validate params strictly."
    params_model = _StrictParams

    async def execute(self, params: _StrictParams) -> ToolResult:
        return ToolResult(content="strict:ok")


class _DangerousTool(Tool[_NoopParams]):
    name = "dangerous_noop"
    description = "A dangerous tool for permission tests."
    params_model = _NoopParams
    permission = "dangerous"

    async def execute(self, params: _NoopParams) -> ToolResult:
        return ToolResult(content=f"danger:{params.value}")


class _SleepParams(BaseModel):
    label: str
    delay: float = 0.05


class _SleepReadTool(Tool[_SleepParams]):
    name = "sleep_read"
    description = "Sleep for a short time."
    params_model = _SleepParams
    permission = "read_only"

    async def execute(self, params: _SleepParams) -> ToolResult:
        await asyncio.sleep(params.delay)
        return ToolResult(content=f"read:{params.label}")


class _SleepWriteTool(Tool[_SleepParams]):
    name = "sleep_write"
    description = "Sleep for a short time."
    params_model = _SleepParams
    permission = "write"

    async def execute(self, params: _SleepParams) -> ToolResult:
        await asyncio.sleep(params.delay)
        return ToolResult(content=f"write:{params.label}")


async def _collect_dispatch(dispatcher: ToolDispatcher, tool_uses: list[ToolUseContent], step: int = 1):
    started: list[ToolCallStarted] = []
    completed: list[tuple[ToolCallCompleted, ToolResultContent]] = []
    async for item in dispatcher.dispatch(tool_uses, step):
        if isinstance(item, ToolCallStarted):
            started.append(item)
        else:
            completed.append(item)
    return started, completed


@pytest.mark.anyio
async def test_dispatcher_emits_started_then_completed_in_input_order():
    dispatcher = ToolDispatcher(
        ToolRegistry([_NoopTool(), _DangerousTool()]),
        allowed_permissions={"read_only", "dangerous"},
    )
    tool_uses = [
        ToolUseContent(id="call_1", name="dangerous_noop", input={"value": "x"}),
        ToolUseContent(id="call_2", name="noop", input={"value": "y"}),
    ]

    started, completed = await _collect_dispatch(dispatcher, tool_uses)

    assert [event.tool_call_id for event in started] == ["call_1", "call_2"]
    assert [event.tool_call_id for event, _ in completed] == ["call_1", "call_2"]
    assert [block.content for _, block in completed] == ["danger:x", "tool:y"]


@pytest.mark.anyio
async def test_dispatcher_returns_permission_and_missing_tool_errors():
    dispatcher = ToolDispatcher(
        ToolRegistry([_DangerousTool()]),
        allowed_permissions={"read_only", "write"},
    )
    tool_uses = [
        ToolUseContent(id="call_1", name="missing", input={}),
        ToolUseContent(id="call_2", name="dangerous_noop", input={"value": "x"}),
    ]

    _, completed = await _collect_dispatch(dispatcher, tool_uses)

    assert completed[0][1].is_error is True
    assert completed[0][1].content == "Tool 'missing' not registered"
    assert completed[1][1].is_error is True
    assert "requires permission 'dangerous'" in completed[1][1].content


@pytest.mark.anyio
async def test_dispatcher_formats_validation_errors_for_llm():
    dispatcher = ToolDispatcher(
        ToolRegistry([_StrictTool()]),
        allowed_permissions={"read_only", "write"},
    )
    tool_uses = [
        ToolUseContent(id="call_1", name="strict", input={"mode": "bad"}),
    ]

    _, completed = await _collect_dispatch(dispatcher, tool_uses)

    assert completed[0][1].is_error is True
    assert "Invalid arguments for tool 'strict':" in completed[0][1].content
    assert "mode" in completed[0][1].content


@pytest.mark.anyio
async def test_dispatcher_runs_read_only_tools_concurrently():
    dispatcher = ToolDispatcher(
        ToolRegistry([_SleepReadTool()]),
        allowed_permissions={"read_only", "write"},
        max_concurrent=10,
    )
    tool_uses = [
        ToolUseContent(id="call_1", name="sleep_read", input={"label": "a", "delay": 0.05}),
        ToolUseContent(id="call_2", name="sleep_read", input={"label": "b", "delay": 0.05}),
    ]

    t0 = time.monotonic()
    await _collect_dispatch(dispatcher, tool_uses)
    elapsed = time.monotonic() - t0

    assert elapsed < 0.09


@pytest.mark.anyio
async def test_dispatcher_runs_write_tools_serially():
    dispatcher = ToolDispatcher(
        ToolRegistry([_SleepWriteTool()]),
        allowed_permissions={"read_only", "write"},
        max_concurrent=10,
    )
    tool_uses = [
        ToolUseContent(id="call_1", name="sleep_write", input={"label": "a", "delay": 0.05}),
        ToolUseContent(id="call_2", name="sleep_write", input={"label": "b", "delay": 0.05}),
    ]

    t0 = time.monotonic()
    await _collect_dispatch(dispatcher, tool_uses)
    elapsed = time.monotonic() - t0

    assert elapsed >= 0.09


@pytest.mark.anyio
async def test_dispatcher_runs_parallel_and_serial_groups_together():
    dispatcher = ToolDispatcher(
        ToolRegistry([_SleepReadTool(), _SleepWriteTool()]),
        allowed_permissions={"read_only", "write"},
        max_concurrent=10,
    )
    tool_uses = [
        ToolUseContent(id="call_1", name="sleep_read", input={"label": "a", "delay": 0.05}),
        ToolUseContent(id="call_2", name="sleep_write", input={"label": "b", "delay": 0.05}),
    ]

    t0 = time.monotonic()
    await _collect_dispatch(dispatcher, tool_uses)
    elapsed = time.monotonic() - t0

    assert elapsed < 0.09
