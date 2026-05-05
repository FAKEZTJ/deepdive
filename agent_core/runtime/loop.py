from __future__ import annotations

import time
import uuid
from typing import AsyncIterator

from pydantic import BaseModel

from agent_core.context.manager import ContextManager
from agent_core.observability.logging import (
    LoggingContext,
    clear_logging_context,
    get_logger,
)
from agent_core.observability.tracing import SpanScope
from agent_core.persistence.session_store import SessionStore
from agent_core.providers.base import LLMProvider
from agent_core.runtime.dispatcher import ToolDispatcher
from agent_core.runtime.events import (
    ContextCompressed,
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

logger = get_logger(__name__)


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
        context_manager: ContextManager | None = None,
        session_store: SessionStore | None = None,
    ):
        self.provider = provider
        self.tools = tools
        self.system_prompt = system_prompt
        self.budget = budget or Budget()
        self._allowed_permissions = allowed_permissions or {"read_only", "write"}
        self._context_manager = context_manager
        self._session_store = session_store
        self._session_id: str | None = None
        self._dispatcher = ToolDispatcher(
            registry=tools,
            allowed_permissions=self._allowed_permissions,
            max_concurrent=self.budget.max_concurrent_tools,
        )

    @property
    def session_id(self) -> str | None:
        return self._session_id

    async def run(self, task: str) -> RunCompleted:
        final: RunCompleted | None = None
        async for event in self.run_stream(task):
            if isinstance(event, RunCompleted):
                final = event
        assert final is not None
        return final

    async def resume(
        self,
        session_id: str,
        *,
        additional_input: str | None = None,
    ) -> RunCompleted:
        final: RunCompleted | None = None
        async for event in self.resume_stream(session_id, additional_input=additional_input):
            if isinstance(event, RunCompleted):
                final = event
        assert final is not None
        return final

    async def run_stream(self, task: str) -> AsyncIterator[RunEvent]:
        clear_logging_context()
        initial_messages = [Message.user(task)]
        if self._session_store is not None:
            self._session_id = await self._session_store.create_session(
                system_prompt=self.system_prompt,
            )
            await self._session_store.append_message(self._session_id, initial_messages[0])

        async for event in self._run_loop(
            initial_messages=initial_messages,
            initial_step=1,
            initial_usage=Usage(),
        ):
            yield event

    async def resume_stream(
        self,
        session_id: str,
        *,
        additional_input: str | None = None,
    ) -> AsyncIterator[RunEvent]:
        clear_logging_context()
        if self._session_store is None:
            raise RuntimeError("session_store is required for resume")

        record = await self._session_store.get_session(session_id)
        if record is None:
            raise ValueError(f"Session {session_id} not found")

        self._session_id = session_id
        loaded_messages = await self._session_store.get_messages(session_id)
        await self._session_store.update_session_state(
            session_id,
            status="running",
            stop_reason=None,
            error_message=None,
        )
        if additional_input is not None:
            additional_message = Message.user(additional_input)
            loaded_messages.append(additional_message)
            await self._session_store.append_message(session_id, additional_message)

        async for event in self._run_loop(
            initial_messages=loaded_messages,
            initial_step=record.total_steps + 1,
            initial_usage=record.total_usage,
        ):
            yield event

    async def _run_loop(
        self,
        *,
        initial_messages: list[Message],
        initial_step: int,
        initial_usage: Usage,
    ) -> AsyncIterator[RunEvent]:
        messages: list[Message] = list(initial_messages)
        total_usage = Usage(
            input_tokens=initial_usage.input_tokens,
            output_tokens=initial_usage.output_tokens,
        )
        start_time = time.monotonic()
        step = initial_step

        async with SpanScope(
            "agent_run",
            attributes={
                "agent.session_id": self._session_id,
                "agent.initial_step": initial_step,
                "agent.has_session_store": self._session_store is not None,
            },
        ):
            with LoggingContext(session_id=self._session_id):
                logger.info(
                    "agent.run.started",
                    initial_step=initial_step,
                    initial_message_count=len(messages),
                )
                while True:
                    stop_reason = self._check_budget(step, total_usage, start_time)
                    if stop_reason:
                        logger.info(
                            "agent.run.completed",
                            stop_reason=stop_reason,
                            total_steps=step - 1,
                            total_input_tokens=total_usage.input_tokens,
                            total_output_tokens=total_usage.output_tokens,
                        )
                        yield await self._persist_event(
                            RunCompleted(
                                final_message=messages[-1] if messages else Message.assistant_text(""),
                                total_steps=step - 1,
                                total_usage=total_usage,
                                stop_reason=stop_reason,
                            )
                        )
                        await self._finish_session(
                            status=self._status_for_stop_reason(stop_reason),
                            stop_reason=stop_reason,
                        )
                        return

                    async with SpanScope(
                        "step",
                        attributes={"step.number": step},
                    ):
                        with LoggingContext(step=step):
                            logger.info("step.started")
                            yield await self._persist_event(StepStarted(step=step))
                            if self._context_manager is not None:
                                messages, compressed = await self._context_manager.compress_if_needed(
                                    messages,
                                    system_prompt=self.system_prompt,
                                )
                                if compressed:
                                    logger.info(
                                        "context.compressed",
                                        new_message_count=len(messages),
                                    )
                                    if self._session_store is not None and self._session_id is not None:
                                        await self._session_store.replace_messages(self._session_id, messages)
                                    yield await self._persist_event(
                                        ContextCompressed(step=step, new_message_count=len(messages))
                                    )
                            yield await self._persist_event(LLMCallStarted(step=step))

                            llm_call_id = uuid.uuid4().hex
                            try:
                                with LoggingContext(llm_call_id=llm_call_id):
                                    async with SpanScope(
                                        "llm_call",
                                        attributes={
                                            "llm.provider": self.provider.name,
                                            "llm.model": self.provider.config.model,
                                            "llm.messages_count": len(messages),
                                            "llm.tools_count": len(self.tools),
                                        },
                                    ) as llm_span:
                                        logger.info(
                                            "llm.call.started",
                                            message_count=len(messages),
                                            tool_count=len(self.tools),
                                        )
                                        response = await self.provider.chat(
                                            messages=messages,
                                            tools=self.tools.schemas(self._allowed_permissions) if len(self.tools) > 0 else None,
                                            system=self.system_prompt,
                                        )
                                        llm_span.set_attribute("llm.usage.input_tokens", response.usage.input_tokens)
                                        llm_span.set_attribute("llm.usage.output_tokens", response.usage.output_tokens)
                                        llm_span.set_attribute("llm.finish_reason", response.finish_reason)
                                        logger.info(
                                            "llm.call.completed",
                                            finish_reason=response.finish_reason,
                                            input_tokens=response.usage.input_tokens,
                                            output_tokens=response.usage.output_tokens,
                                            content_blocks=len(response.message.content),
                                        )
                            except Exception as exc:
                                logger.exception("llm.call.failed")
                                error_event = RunCompleted(
                                    final_message=Message.assistant_text(f"LLM error: {exc}"),
                                    total_steps=step - 1,
                                    total_usage=total_usage,
                                    stop_reason="error",
                                )
                                yield await self._persist_event(error_event)
                                await self._finish_session(
                                    status="error",
                                    stop_reason="error",
                                    error_message=str(exc),
                                )
                                return

                            step_usage = response.usage
                            step_total_usage = Usage(
                                input_tokens=total_usage.input_tokens + step_usage.input_tokens,
                                output_tokens=total_usage.output_tokens + step_usage.output_tokens,
                            )

                            yield await self._persist_event(
                                LLMCallCompleted(
                                    step=step,
                                    message=response.message,
                                    usage=response.usage,
                                )
                            )

                            pending_checkpoint_messages: list[Message] = [response.message]
                            messages.append(response.message)

                            if response.finish_reason == "stop":
                                logger.info("step.completed", finish_reason="stop")
                                yield await self._persist_event(StepCompleted(step=step))
                                await self._checkpoint_step(
                                    step=step,
                                    step_usage=step_usage,
                                    new_messages=pending_checkpoint_messages,
                                )
                                total_usage = step_total_usage
                                logger.info(
                                    "agent.run.completed",
                                    stop_reason="finished",
                                    total_steps=step,
                                    total_input_tokens=total_usage.input_tokens,
                                    total_output_tokens=total_usage.output_tokens,
                                )
                                final_event = RunCompleted(
                                    final_message=messages[-1] if messages else Message.assistant_text(""),
                                    total_steps=step,
                                    total_usage=total_usage,
                                    stop_reason="finished",
                                )
                                yield await self._persist_event(final_event)
                                await self._finish_session(
                                    status="completed",
                                    stop_reason="finished",
                                )
                                return

                            if response.finish_reason == "max_tokens":
                                logger.info(
                                    "agent.run.completed",
                                    stop_reason="max_tokens",
                                    total_steps=step - 1,
                                    total_input_tokens=total_usage.input_tokens,
                                    total_output_tokens=total_usage.output_tokens,
                                )
                                final_event = RunCompleted(
                                    final_message=response.message,
                                    total_steps=step - 1,
                                    total_usage=total_usage,
                                    stop_reason="max_tokens",
                                )
                                yield await self._persist_event(final_event)
                                await self._finish_session(
                                    status="paused",
                                    stop_reason="max_tokens",
                                )
                                return

                            tool_uses = [
                                block for block in response.message.content if isinstance(block, ToolUseContent)
                            ]
                            if not tool_uses:
                                logger.info("step.completed", finish_reason=response.finish_reason)
                                yield await self._persist_event(StepCompleted(step=step))
                                await self._checkpoint_step(
                                    step=step,
                                    step_usage=step_usage,
                                    new_messages=pending_checkpoint_messages,
                                )
                                total_usage = step_total_usage
                                logger.info(
                                    "agent.run.completed",
                                    stop_reason="finished",
                                    total_steps=step,
                                    total_input_tokens=total_usage.input_tokens,
                                    total_output_tokens=total_usage.output_tokens,
                                )
                                final_event = RunCompleted(
                                    final_message=response.message,
                                    total_steps=step,
                                    total_usage=total_usage,
                                    stop_reason="finished",
                                )
                                yield await self._persist_event(final_event)
                                await self._finish_session(
                                    status="completed",
                                    stop_reason="finished",
                                )
                                return

                            logger.info("tool.dispatch.started", tool_call_count=len(tool_uses))
                            result_blocks = []
                            async with SpanScope(
                                "tool_dispatch",
                                attributes={"tools.count": len(tool_uses)},
                            ):
                                async for item in self._dispatcher.dispatch(tool_uses, step):
                                    if isinstance(item, ToolCallStarted):
                                        yield await self._persist_event(item)
                                        continue

                                    completed_event, result_block = item
                                    yield await self._persist_event(completed_event)
                                    result_blocks.append(result_block)

                            tool_message = Message(role="tool", content=result_blocks)
                            messages.append(tool_message)
                            pending_checkpoint_messages.append(tool_message)

                            logger.info("step.completed", finish_reason=response.finish_reason)
                            yield await self._persist_event(StepCompleted(step=step))
                            await self._checkpoint_step(
                                step=step,
                                step_usage=step_usage,
                                new_messages=pending_checkpoint_messages,
                            )
                            total_usage = step_total_usage
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

    async def _persist_event(self, event: RunEvent) -> RunEvent:
        if self._session_store is not None and self._session_id is not None:
            await self._session_store.append_event(self._session_id, event)
        return event

    async def _checkpoint_step(
        self,
        *,
        step: int,
        step_usage: Usage,
        new_messages: list[Message],
    ) -> None:
        if self._session_store is None or self._session_id is None:
            return
        await self._session_store.checkpoint_step(
            self._session_id,
            new_messages=new_messages,
            completed_step=step,
            usage_delta=step_usage,
        )

    async def _finish_session(
        self,
        *,
        status: str,
        stop_reason: str,
        error_message: str | None = None,
    ) -> None:
        if self._session_store is None or self._session_id is None:
            return
        await self._session_store.complete_run(
            self._session_id,
            status=status,
            stop_reason=stop_reason,
            error_message=error_message,
        )

    @staticmethod
    def _status_for_stop_reason(stop_reason: str) -> str:
        if stop_reason == "finished":
            return "completed"
        if stop_reason == "error":
            return "error"
        return "paused"
