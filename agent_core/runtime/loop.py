from __future__ import annotations

import time
from typing import AsyncIterator

from pydantic import BaseModel

from agent_core.providers.base import LLMProvider
from agent_core.runtime.dispatcher import ToolDispatcher
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
from agent_core.tools.base import ToolPermission
from agent_core.tools.registry import ToolRegistry
from agent_core.types import Message, ToolUseContent, Usage


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
        allowed_permissions: set[ToolPermission] | None = None,
    ):
        self.provider = provider
        self.tools = tools
        self.system_prompt = system_prompt
        self.budget = budget or Budget()
        self._allowed_permissions = allowed_permissions or {"read_only", "write"}
        self._dispatcher = ToolDispatcher(
            registry=tools,
            allowed_permissions=self._allowed_permissions,
            max_concurrent=self.budget.max_concurrent_tools,
        )

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
                    tools=self.tools.schemas(self._allowed_permissions) if len(self.tools) > 0 else None,
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

            result_blocks = []
            async for item in self._dispatcher.dispatch(tool_uses, step):
                if isinstance(item, ToolCallStarted):
                    yield item
                    continue

                completed_event, result_block = item
                yield completed_event
                result_blocks.append(result_block)
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
