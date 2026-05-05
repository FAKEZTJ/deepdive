# agent_core/runtime/events.py
from __future__ import annotations

from typing import Any, Literal, Union
from pydantic import BaseModel, Field
from typing import Annotated

from agent_core.types import Message, Usage


class StepStarted(BaseModel):
    type: Literal["step_started"] = "step_started"
    step: int


class LLMCallStarted(BaseModel):
    type: Literal["llm_call_started"] = "llm_call_started"
    step: int


class LLMCallCompleted(BaseModel):
    type: Literal["llm_call_completed"] = "llm_call_completed"
    step: int
    message: Message
    usage: Usage


class ContextCompressed(BaseModel):
    type: Literal["context_compressed"] = "context_compressed"
    step: int
    new_message_count: int


class ToolCallStarted(BaseModel):
    type: Literal["tool_call_started"] = "tool_call_started"
    step: int
    tool_call_id: str
    tool_name: str
    input: dict[str, Any]


class ToolCallCompleted(BaseModel):
    type: Literal["tool_call_completed"] = "tool_call_completed"
    step: int
    tool_call_id: str
    tool_name: str
    output: str
    is_error: bool
    duration_ms: float


class StepCompleted(BaseModel):
    type: Literal["step_completed"] = "step_completed"
    step: int


class RunCompleted(BaseModel):
    type: Literal["run_completed"] = "run_completed"
    final_message: Message      # 最后一条 assistant 消息（纯文本）
    total_steps: int
    total_usage: Usage
    stop_reason: Literal["finished", "max_steps", "max_tokens", "timeout", "error"]


RunEvent = Annotated[
    Union[
        StepStarted, ContextCompressed, LLMCallStarted, LLMCallCompleted,
        ToolCallStarted, ToolCallCompleted,
        StepCompleted, RunCompleted,
    ],
    Field(discriminator="type"),
]
