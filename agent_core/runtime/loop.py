# agent_core/runtime/loop.py
from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator

from pydantic import BaseModel, ValidationError

from agent_core.providers.base import LLMProvider
from agent_core.runtime.events import (
    LLMCallCompleted,
    LLMCallStarted,
    RunCompleted,
    RunEvent,
    StepCompleted,
    StepStarted,
    ToolCallCompleted,
    ToolCallStarted,
)
from agent_core.tools.registry import ToolRegistry
from agent_core.types import Message, ToolResultContent, ToolUseContent, Usage


class Budget(BaseModel):
    max_steps: int = 20
    max_tokens: int | None = None
    timeout_seconds: float | None = None
    max_concurrent_tools: int = 10 

class AgentLoop:
    """ReAct-style agent loop."""

    def __init__(
        self,
        *,
        provider: LLMProvider,
        tools: ToolRegistry,
        system_prompt: str | None = None,
        budget: Budget | None = None,
    ):
        self.provider = provider
        self.tools = tools
        self.system_prompt = system_prompt
        self.budget = budget or Budget()

    async def run(self, task: str) -> RunCompleted:
        final: RunCompleted | None = None
        async for event in self.run_stream(task):
            if isinstance(event, RunCompleted):
                final = event
        assert final is not None
        return final

    async def run_stream(self, task: str) -> AsyncIterator[RunEvent]:
        messages: list[Message] = [Message.user(task)]
        total_usage = Usage()
        start_time = time.monotonic()
        step = 1

        while True:
            stop_reason = self._check_budget(step, total_usage, start_time)
            if stop_reason:
                yield RunCompleted(
                    final_message=messages[-1] if messages else Message.assistant_text(""),
                    total_steps=step - 1,
                    total_usage=total_usage,
                    stop_reason=stop_reason,
                )
                return

            yield StepStarted(step=step)
            yield LLMCallStarted(step=step)

            try:
                response = await self.provider.chat(
                    messages=messages,
                    tools=self.tools.schemas() if len(self.tools) > 0 else None,
                    system=self.system_prompt,
                )
            except Exception as exc:
                yield RunCompleted(
                    final_message=Message.assistant_text(f"LLM error: {exc}"),
                    total_steps=step,
                    total_usage=total_usage,
                    stop_reason="error",
                )
                return

            total_usage = Usage(
                input_tokens=total_usage.input_tokens + response.usage.input_tokens,
                output_tokens=total_usage.output_tokens + response.usage.output_tokens,
            )

            yield LLMCallCompleted(
                step=step,
                message=response.message,
                usage=response.usage,
            )

            messages.append(response.message)

            if response.finish_reason == "stop":
                yield StepCompleted(step=step)
                yield RunCompleted(
                    final_message=response.message,
                    total_steps=step,
                    total_usage=total_usage,
                    stop_reason="finished",
                )
                return

            if response.finish_reason == "max_tokens":
                yield RunCompleted(
                    final_message=response.message,
                    total_steps=step,
                    total_usage=total_usage,
                    stop_reason="max_tokens",
                )
                return

            tool_uses = [
                block for block in response.message.content if isinstance(block, ToolUseContent)
            ]
            if not tool_uses:
                yield RunCompleted(
                    final_message=response.message,
                    total_steps=step,
                    total_usage=total_usage,
                    stop_reason="finished",
                )
                return

            for tool_use in tool_uses:
                yield ToolCallStarted(
                    step=step,
                    tool_call_id=tool_use.id,
                    tool_name=tool_use.name,
                    input=tool_use.input,
                )

            tool_results = await self._execute_tools(tool_uses, step)
            for event, _ in tool_results:
                yield event

            result_blocks = [result_block for _, result_block in tool_results]
            messages.append(Message(role="tool", content=result_blocks))

            yield StepCompleted(step=step)
            step += 1

    def _check_budget(
        self,
        next_step: int,
        usage: Usage,
        start_time: float,
    ) -> str | None:
        if next_step > self.budget.max_steps:
            return "max_steps"
        if self.budget.max_tokens is not None and usage.total_tokens >= self.budget.max_tokens:
            return "max_tokens"
        if self.budget.timeout_seconds is not None:
            if time.monotonic() - start_time >= self.budget.timeout_seconds:
                return "timeout"
        return None

    async def _execute_tools(
        self,
        tool_uses: list[ToolUseContent],
        step: int,
    ) -> list[tuple[ToolCallCompleted, ToolResultContent]]:
        async def one(tu: ToolUseContent) -> tuple[ToolCallCompleted, ToolResultContent]:
            t0 = time.monotonic()

            try:
                tool = self.tools.get(tu.name)
            except KeyError:
                return self._make_failed_result(tu, step, t0, f"Tool '{tu.name}' not found")

            try:
                params = tool.parse_input(tu.input)
            except ValidationError as exc:
                return self._make_failed_result(tu, step, t0, f"Invalid params: {exc}")

            try:
                result = await tool.execute(params)
            except Exception as exc:
                return self._make_failed_result(tu, step, t0, f"Tool crashed: {exc}")

            duration_ms = (time.monotonic() - t0) * 1000
            event = ToolCallCompleted(
                step=step,
                tool_call_id=tu.id,
                tool_name=tu.name,
                output=result.content,
                is_error=result.is_error,
                duration_ms=duration_ms,
            )
            block = ToolResultContent(
                tool_use_id=tu.id,
                content=result.content,
                is_error=result.is_error,
            )
            return event, block

        return await asyncio.gather(*[one(tu) for tu in tool_uses])

    def _make_failed_result(
        self,
        tu: ToolUseContent,
        step: int,
        t0: float,
        message: str,
    ) -> tuple[ToolCallCompleted, ToolResultContent]:
        duration_ms = (time.monotonic() - t0) * 1000
        return (
            ToolCallCompleted(
                step=step,
                tool_call_id=tu.id,
                tool_name=tu.name,
                output=message,
                is_error=True,
                duration_ms=duration_ms,
            ),
            ToolResultContent(tool_use_id=tu.id, content=message, is_error=True),
        )
