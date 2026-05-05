from __future__ import annotations

import sys
from typing import Literal

import pytest
from pydantic import BaseModel

from agent_core.runtime.events import (
    ContextCompressed,
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


class _DangerousTool(Tool[_NoopParams]):
    name = "dangerous_noop"
    description = "A dangerous tool for permission tests."
    params_model = _NoopParams
    permission = "dangerous"

    async def execute(self, params: _NoopParams) -> ToolResult:
        return ToolResult(content=f"danger:{params.value}")


class _ThresholdEstimator:
    def estimate(
        self,
        messages: list[Message],
        *,
        system_prompt: str | None = None,
    ) -> int:
        return 999 if len(messages) >= 5 else 0


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
    assert "Tool 'missing' not registered" in tool_message.content[0].content


@pytest.mark.anyio
async def test_default_permissions_hide_dangerous_tools_from_provider_schema():
    provider = FakeProvider(scripted_responses=[_stop_response()])
    loop = AgentLoop(
        provider=provider,
        tools=ToolRegistry([_NoopTool(), _DangerousTool()]),
    )

    await loop.run("list tools")

    exported_tools = provider.calls[0]["tools"]

    assert exported_tools is not None
    assert [tool.name for tool in exported_tools] == ["noop"]


@pytest.mark.anyio
async def test_explicit_permissions_can_expose_dangerous_tools_to_provider_schema():
    provider = FakeProvider(scripted_responses=[_stop_response()])
    loop = AgentLoop(
        provider=provider,
        tools=ToolRegistry([_NoopTool(), _DangerousTool()]),
        allowed_permissions={"read_only", "dangerous"},
    )

    await loop.run("list tools")

    exported_tools = provider.calls[0]["tools"]

    assert exported_tools is not None
    assert [tool.name for tool in exported_tools] == ["noop", "dangerous_noop"]


@pytest.mark.anyio
async def test_disallowed_tool_call_returns_error_tool_result():
    provider = FakeProvider(
        scripted_responses=[
            _single_tool_response(tool_name="dangerous_noop", tool_input={"value": "x"}),
            _stop_response(),
        ]
    )
    loop = AgentLoop(
        provider=provider,
        tools=ToolRegistry([_DangerousTool()]),
    )

    result = await loop.run("try dangerous tool")

    tool_message = provider.calls[1]["messages"][-1]

    assert result.stop_reason == "finished"
    assert tool_message.role == "tool"
    assert tool_message.content[0].is_error is True
    assert "requires permission 'dangerous'" in tool_message.content[0].content


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
    assert "Invalid arguments for tool 'strict':" in tool_message.content[0].content


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
    loop = AgentLoop(
        provider=provider,
        tools=ToolRegistry([ShellExecTool()]),
        allowed_permissions={"read_only", "write", "dangerous"},
    )

    result = await loop.run("Count Python files in the directory")

    tool_message = provider.calls[1]["messages"][-1]

    assert result.stop_reason == "finished"
    assert result.final_message.content[0].text == "Found 2 Python files."
    assert tool_message.role == "tool"
    assert "STDOUT:\n2" in tool_message.content[0].content


@pytest.mark.anyio
async def test_context_manager_compresses_before_llm_call_and_emits_event():
    from agent_core.context.manager import ContextManager

    summary_provider = FakeProvider(scripted_responses=[_stop_response("compressed summary")])
    main_provider = FakeProvider(
        scripted_responses=[
            _single_tool_response(tool_name="noop", tool_input={"value": "one"}, input_tokens=1, output_tokens=1),
            _single_tool_response(tool_name="noop", tool_input={"value": "two"}, input_tokens=1, output_tokens=1),
            _stop_response("Done."),
        ]
    )
    loop = AgentLoop(
        provider=main_provider,
        tools=ToolRegistry([_NoopTool()]),
        context_manager=ContextManager(
            provider=summary_provider,
            threshold_tokens=100,
            keep_recent_pairs=1,
            token_estimator=_ThresholdEstimator(),
        ),
    )

    events = [event async for event in loop.run_stream("compress later")]

    compressed = [event for event in events if isinstance(event, ContextCompressed)]
    assert [(event.step, event.new_message_count) for event in compressed] == [(3, 3)]

    third_call_messages = main_provider.calls[2]["messages"]
    assert third_call_messages[0] == Message.assistant_text(
        "<previous_conversation_summary>\n"
        "compressed summary\n"
        "</previous_conversation_summary>"
    )
    assert third_call_messages[1].role == "assistant"
    assert third_call_messages[2].role == "tool"


@pytest.mark.anyio
async def test_session_store_persists_only_completed_step_messages(tmp_path):
    from agent_core.persistence.session_store import SessionStore

    store = SessionStore(str(tmp_path / "agent.db"))
    await store.initialize()
    try:
        provider = FakeProvider(
            scripted_responses=[
                _single_tool_response(
                    tool_name="noop",
                    tool_input={"value": "ok"},
                    input_tokens=3,
                    output_tokens=2,
                ),
                CompletionResponse(
                    message=Message.assistant_text("unfinished second step"),
                    finish_reason="max_tokens",
                    usage=Usage(input_tokens=4, output_tokens=1),
                ),
            ]
        )
        loop = AgentLoop(
            provider=provider,
            tools=ToolRegistry([_NoopTool()]),
            session_store=store,
        )

        result = await loop.run("persist checkpoints only")

        assert result.stop_reason == "max_tokens"
        assert loop.session_id is not None

        persisted_messages = await store.get_messages(loop.session_id)
        assert persisted_messages == [
            Message.user("persist checkpoints only"),
            Message(
                role="assistant",
                content=[ToolUseContent(id="call_1", name="noop", input={"value": "ok"})],
            ),
            Message.tool_result("call_1", "tool:ok"),
        ]

        record = await store.get_session(loop.session_id)
        assert record is not None
        assert record.status == "paused"
        assert record.total_steps == 1
        assert record.total_usage == Usage(input_tokens=3, output_tokens=2)
        assert record.stop_reason == "max_tokens"
    finally:
        await store.close()


@pytest.mark.anyio
async def test_resume_stream_continues_from_checkpointed_step_and_updates_session(tmp_path):
    from agent_core.persistence.session_store import SessionStore

    store = SessionStore(str(tmp_path / "agent.db"))
    await store.initialize()
    try:
        first_provider = FakeProvider(
            scripted_responses=[
                _single_tool_response(
                    tool_name="noop",
                    tool_input={"value": "one"},
                    input_tokens=2,
                    output_tokens=3,
                ),
            ]
        )
        first_loop = AgentLoop(
            provider=first_provider,
            tools=ToolRegistry([_NoopTool()]),
            budget=Budget(max_steps=1),
            session_store=store,
        )

        first_events = [event async for event in first_loop.run_stream("resume me")]
        first_final = next(event for event in first_events if isinstance(event, RunCompleted))
        assert first_final.stop_reason == "max_steps"
        assert first_loop.session_id is not None

        second_provider = FakeProvider(
            scripted_responses=[_stop_response("resumed done")]
        )
        second_loop = AgentLoop(
            provider=second_provider,
            tools=ToolRegistry([_NoopTool()]),
            session_store=store,
        )

        resumed_events = [
            event
            async for event in second_loop.resume_stream(
                first_loop.session_id,
                additional_input="continue please",
            )
        ]

        step_starts = [event.step for event in resumed_events if isinstance(event, StepStarted)]
        final = next(event for event in resumed_events if isinstance(event, RunCompleted))

        assert step_starts == [2]
        assert final.stop_reason == "finished"
        assert final.total_steps == 2
        assert final.total_usage == Usage(input_tokens=3, output_tokens=4)

        resumed_call_messages = second_provider.calls[0]["messages"]
        assert resumed_call_messages == [
            Message.user("resume me"),
            Message(
                role="assistant",
                content=[ToolUseContent(id="call_1", name="noop", input={"value": "one"})],
            ),
            Message.tool_result("call_1", "tool:one"),
            Message.user("continue please"),
        ]

        record = await store.get_session(first_loop.session_id)
        assert record is not None
        assert record.status == "completed"
        assert record.total_steps == 2
        assert record.total_usage == Usage(input_tokens=3, output_tokens=4)
        assert record.stop_reason == "finished"

        persisted_messages = await store.get_messages(first_loop.session_id)
        assert persisted_messages == [
            Message.user("resume me"),
            Message(
                role="assistant",
                content=[ToolUseContent(id="call_1", name="noop", input={"value": "one"})],
            ),
            Message.tool_result("call_1", "tool:one"),
            Message.user("continue please"),
            Message.assistant_text("resumed done"),
        ]
    finally:
        await store.close()


@pytest.mark.anyio
async def test_session_store_tracks_cost_without_double_counting_usage(tmp_path):
    from agent_core.persistence.session_store import SessionStore
    from agent_core.providers.base import ProviderConfig

    store = SessionStore(str(tmp_path / "agent.db"))
    await store.initialize()
    try:
        provider = FakeProvider(
            scripted_responses=[
                CompletionResponse(
                    message=Message.assistant_text("priced reply"),
                    finish_reason="stop",
                    usage=Usage(input_tokens=10, output_tokens=20),
                ),
            ],
            config=ProviderConfig(model="gpt-4o-mini", api_key="fake"),
        )
        provider.name = "openai"
        loop = AgentLoop(
            provider=provider,
            tools=ToolRegistry(),
            session_store=store,
        )

        result = await loop.run("price this")

        assert result.stop_reason == "finished"
        assert loop.session_id is not None

        record = await store.get_session(loop.session_id)
        assert record is not None
        assert record.metadata == {"provider": "openai", "model": "gpt-4o-mini"}
        assert record.total_usage == Usage(input_tokens=10, output_tokens=20)
        assert record.total_cost_usd == pytest.approx((10 * 0.15 + 20 * 0.60) / 1_000_000)
    finally:
        await store.close()


@pytest.mark.anyio
async def test_llm_call_completed_event_carries_provider_model_and_cost():
    from agent_core.persistence.session_store import SessionStore
    from agent_core.providers.base import ProviderConfig

    store = SessionStore(":memory:")
    await store.initialize()
    try:
        provider = FakeProvider(
            scripted_responses=[
                CompletionResponse(
                    message=Message.assistant_text("priced reply"),
                    finish_reason="stop",
                    usage=Usage(input_tokens=10, output_tokens=20),
                ),
            ],
            config=ProviderConfig(model="gpt-4o-mini", api_key="fake"),
        )
        provider.name = "openai"
        loop = AgentLoop(
            provider=provider,
            tools=ToolRegistry(),
            session_store=store,
        )

        events = [event async for event in loop.run_stream("price this")]

        llm_completed = next(event for event in events if event.type == "llm_call_completed")
        assert llm_completed.provider == "openai"
        assert llm_completed.model == "gpt-4o-mini"
        assert llm_completed.cost_usd == pytest.approx((10 * 0.15 + 20 * 0.60) / 1_000_000)
    finally:
        await store.close()
