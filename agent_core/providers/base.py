"""Provider abstraction placeholders."""

# agent_core/providers/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator
from pydantic import BaseModel

from agent_core.types import (
    Message, ToolSchema, CompletionResponse, StreamEvent
)


class ProviderConfig(BaseModel):
    """所有 provider 共享的配置。子类可以扩展。"""
    model: str
    api_key: str
    base_url: str | None = None
    timeout_seconds: float = 60.0
    max_retries: int = 2


class LLMProvider(ABC):
    """LLM Provider 的统一接口。
    
    实现要点：
    - 输入永远是 List[Message] + List[ToolSchema]
    - 输出永远是 CompletionResponse 或 AsyncIterator[StreamEvent]
    - 内部把我们的统一格式转成各家 SDK 的格式（adapter 模式）
    - 各家 SDK 抛出的异常必须翻译成 ProviderError 子类
    """

    name: str  # 子类设置，如 "openai" / "anthropic"

    def __init__(self, config: ProviderConfig):
        self.config = config

    @abstractmethod
    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        system: str | None = None,        # 单独传，而不是塞 messages，便于 Anthropic
    ) -> CompletionResponse:
        """非流式调用。"""
        ...

    @abstractmethod
    def chat_stream(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        system: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """流式调用。
        
        注意：这个方法本身不是 async，返回的是 async iterator。
        实现时用 async generator (async def + yield)。
        """
        ...
