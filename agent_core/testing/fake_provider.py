from __future__ import annotations

import copy
from typing import Any, AsyncIterator

from agent_core.providers.base import LLMProvider, ProviderConfig
from agent_core.types import CompletionResponse, Message, StreamEvent, ToolSchema


class FakeProvider(LLMProvider):
    """Replay scripted responses for tests."""

    name = "fake"

    def __init__(
        self,
        scripted_responses: list[CompletionResponse],
        config: ProviderConfig | None = None,
    ):
        super().__init__(config or ProviderConfig(model="fake", api_key="fake"))
        self._responses = list(scripted_responses)
        self.calls: list[dict[str, Any]] = []

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        system: str | None = None,
    ) -> CompletionResponse:
        self.calls.append(
            {
                "messages": copy.deepcopy(messages),
                "tools": copy.deepcopy(tools),
                "system": system,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        if not self._responses:
            raise RuntimeError("FakeProvider exhausted scripted responses")
        return self._responses.pop(0)

    async def chat_stream(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        system: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        raise NotImplementedError("FakeProvider streaming not implemented for Day 2")
        yield  # type: ignore[misc]
