from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator

from pydantic import ValidationError

from agent_core.runtime.events import ToolCallCompleted, ToolCallStarted
from agent_core.tools.base import Tool, ToolPermission
from agent_core.tools.registry import ToolRegistry
from agent_core.types import ToolResultContent, ToolUseContent


class ToolDispatcher:
    """Schedule and execute a batch of tool calls."""

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        allowed_permissions: set[ToolPermission],
        max_concurrent: int = 10,
    ):
        self._registry = registry
        self._allowed = allowed_permissions
        self._sem = asyncio.Semaphore(max_concurrent)

    async def dispatch(
        self,
        tool_uses: list[ToolUseContent],
        step: int,
    ) -> AsyncIterator[ToolCallStarted | tuple[ToolCallCompleted, ToolResultContent]]:
        parallel_uses: list[ToolUseContent] = []
        serial_uses: list[ToolUseContent] = []
        unknown_uses: list[ToolUseContent] = []

        for tu in tool_uses:
            if not self._registry.has(tu.name):
                unknown_uses.append(tu)
                continue

            tool = self._registry.get(tu.name)
            if tool.permission not in self._allowed:
                unknown_uses.append(tu)
                continue

            if tool.permission == "read_only":
                parallel_uses.append(tu)
            else:
                serial_uses.append(tu)

        for tu in tool_uses:
            yield ToolCallStarted(
                step=step,
                tool_call_id=tu.id,
                tool_name=tu.name,
                input=tu.input,
            )

        async def run_parallel() -> dict[str, tuple[ToolCallCompleted, ToolResultContent]]:
            results = await asyncio.gather(*[self._exec_one(tu, step) for tu in parallel_uses])
            return {tu.id: result for tu, result in zip(parallel_uses, results)}

        async def run_serial() -> dict[str, tuple[ToolCallCompleted, ToolResultContent]]:
            results: dict[str, tuple[ToolCallCompleted, ToolResultContent]] = {}
            for tu in serial_uses:
                results[tu.id] = await self._exec_one(tu, step)
            return results

        async def run_unknown() -> dict[str, tuple[ToolCallCompleted, ToolResultContent]]:
            return {
                tu.id: self._fail(tu, step, self._unknown_reason(tu))
                for tu in unknown_uses
            }

        parallel_task = asyncio.create_task(run_parallel())
        serial_task = asyncio.create_task(run_serial())
        unknown_task = asyncio.create_task(run_unknown())

        await asyncio.gather(parallel_task, serial_task, unknown_task)
        all_results = {
            **parallel_task.result(),
            **serial_task.result(),
            **unknown_task.result(),
        }

        for tu in tool_uses:
            yield all_results[tu.id]

    async def _exec_one(
        self,
        tu: ToolUseContent,
        step: int,
    ) -> tuple[ToolCallCompleted, ToolResultContent]:
        async with self._sem:
            t0 = time.monotonic()
            tool = self._registry.get(tu.name)

            try:
                params = tool.parse_input(tu.input)
            except ValidationError as exc:
                return self._fail(tu, step, _format_validation_error(exc, tool), t0=t0)

            try:
                result = await tool.execute(params)
            except Exception as exc:
                return self._fail(tu, step, f"Tool crashed: {exc}", t0=t0)

            duration_ms = (time.monotonic() - t0) * 1000
            return (
                ToolCallCompleted(
                    step=step,
                    tool_call_id=tu.id,
                    tool_name=tu.name,
                    output=result.content,
                    is_error=result.is_error,
                    duration_ms=duration_ms,
                ),
                ToolResultContent(
                    tool_use_id=tu.id,
                    content=result.content,
                    is_error=result.is_error,
                ),
            )

    def _fail(
        self,
        tu: ToolUseContent,
        step: int,
        reason: str,
        *,
        t0: float | None = None,
    ) -> tuple[ToolCallCompleted, ToolResultContent]:
        duration_ms = (time.monotonic() - t0) * 1000 if t0 is not None else 0.0
        return (
            ToolCallCompleted(
                step=step,
                tool_call_id=tu.id,
                tool_name=tu.name,
                output=reason,
                is_error=True,
                duration_ms=duration_ms,
            ),
            ToolResultContent(tool_use_id=tu.id, content=reason, is_error=True),
        )

    def _unknown_reason(self, tu: ToolUseContent) -> str:
        if not self._registry.has(tu.name):
            return f"Tool '{tu.name}' not registered"

        tool = self._registry.get(tu.name)
        return (
            f"Tool '{tu.name}' requires permission '{tool.permission}' "
            "which is not allowed in this session."
        )


def _format_validation_error(exc: ValidationError, tool: Tool) -> str:
    """Format Pydantic validation errors into LLM-friendly text."""

    errors: list[str] = []
    for err in exc.errors():
        loc = ".".join(str(item) for item in err["loc"])
        errors.append(f"  - {loc}: {err['msg']} (input={err.get('input')!r})")
    return f"Invalid arguments for tool '{tool.name}':\n" + "\n".join(errors)
