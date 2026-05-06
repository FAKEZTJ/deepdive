from __future__ import annotations

import copy
import json
from typing import Any, AsyncIterator

from agent_core.providers.base import LLMProvider, ProviderConfig
from agent_core.types import (
    CompletionResponse,
    Message,
    StreamEnd,
    StreamEvent,
    TextContent,
    TextDelta,
    TextEnd,
    TextStart,
    ToolSchema,
    ToolUseContent,
    ToolUseDelta,
    ToolUseEnd,
    ToolUseStart,
)


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

    def _record_call(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None,
        *,
        temperature: float,
        max_tokens: int | None,
        system: str | None,
    ) -> None:
        self.calls.append(
            {
                "messages": copy.deepcopy(messages),
                "tools": copy.deepcopy(tools),
                "system": system,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        system: str | None = None,
    ) -> CompletionResponse:
        self._record_call(
            messages,
            tools,
            temperature=temperature,
            max_tokens=max_tokens,
            system=system,
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
        self._record_call(
            messages,
            tools,
            temperature=temperature,
            max_tokens=max_tokens,
            system=system,
        )
        if not self._responses:
            raise RuntimeError("FakeProvider exhausted scripted responses")

        response = self._responses.pop(0)
        for index, block in enumerate(response.message.content):
            if isinstance(block, TextContent):
                yield TextStart(index=index)
                if block.text:
                    yield TextDelta(index=index, text=block.text)
                yield TextEnd(index=index)
                continue

            if isinstance(block, ToolUseContent):
                yield ToolUseStart(index=index, id=block.id, name=block.name)
                yield ToolUseDelta(
                    index=index,
                    partial_json=json.dumps(block.input, ensure_ascii=False, separators=(",", ":")),
                )
                yield ToolUseEnd(index=index)
                continue

            raise RuntimeError(f"Unsupported block type for fake streaming: {type(block).__name__}")

        yield StreamEnd(
            finish_reason=response.finish_reason,
            usage=response.usage,
        )
