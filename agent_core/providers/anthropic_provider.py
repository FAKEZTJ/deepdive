from __future__ import annotations

from dataclasses import dataclass
from typing import Any, AsyncIterator

from anthropic import (
    APITimeoutError as AnthropicAPITimeoutError,
    AsyncAnthropic,
    AuthenticationError as AnthropicAuthenticationError,
    BadRequestError as AnthropicBadRequestError,
    RateLimitError as AnthropicRateLimitError,
)

from agent_core.providers.base import LLMProvider, ProviderConfig
from agent_core.types import (
    AuthError,
    CompletionResponse,
    ContextLengthError,
    Message,
    ProviderError,
    ProviderTimeoutError,
    RateLimitError,
    StreamEnd,
    TextContent,
    TextEnd,
    TextDelta,
    TextStart,
    ToolResultContent,
    ToolSchema,
    ToolUseContent,
    ToolUseDelta,
    ToolUseEnd,
    ToolUseStart,
    Usage,
)


@dataclass
class _AnthropicBlockState:
    type: str
    tool_use_id: str | None = None


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, config: ProviderConfig, client: Any | None = None):
        super().__init__(config)
        self._client = client or AsyncAnthropic(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout_seconds,
            max_retries=config.max_retries,
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
        try:
            response = await self._client.messages.create(
                **self._build_request(
                    messages=messages,
                    tools=tools,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    system=system,
                    stream=False,
                )
            )
        except Exception as exc:
            raise self._map_error(exc) from exc

        return CompletionResponse(
            message=self._assistant_message_from_anthropic(response),
            finish_reason=self._map_finish_reason(getattr(response, "stop_reason", None)),
            usage=self._usage_from_anthropic(getattr(response, "usage", None)),
            raw=response.model_dump() if hasattr(response, "model_dump") else None,
        )

    async def chat_stream(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        system: str | None = None,
    ) -> AsyncIterator[object]:
        try:
            stream = await self._client.messages.create(
                **self._build_request(
                    messages=messages,
                    tools=tools,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    system=system,
                    stream=True,
                )
            )
        except Exception as exc:
            raise self._map_error(exc) from exc

        block_states: dict[int, _AnthropicBlockState] = {}
        finish_reason = "stop"
        usage = Usage()
        sent_end = False

        try:
            async for event in stream:
                event_type = getattr(event, "type", None)

                if event_type == "content_block_start":
                    block = getattr(event, "content_block", None)
                    block_type = getattr(block, "type", None)
                    if block_type is None:
                        continue

                    if block_type == "tool_use":
                        block_states[event.index] = _AnthropicBlockState(
                            type="tool_use",
                            tool_use_id=getattr(block, "id", None),
                        )
                        yield ToolUseStart(index=event.index, id=block.id, name=block.name)
                    else:
                        block_states[event.index] = _AnthropicBlockState(type=block_type)
                        if block_type == "text":
                            yield TextStart(index=event.index)

                elif event_type == "content_block_delta":
                    delta = getattr(event, "delta", None)
                    delta_type = getattr(delta, "type", None)
                    if delta_type == "text_delta":
                        yield TextDelta(index=event.index, text=delta.text)
                    elif delta_type == "input_json_delta":
                        block_state = block_states.get(event.index)
                        if block_state and block_state.type == "tool_use" and block_state.tool_use_id:
                            yield ToolUseDelta(
                                index=event.index,
                                partial_json=delta.partial_json,
                            )

                elif event_type == "content_block_stop":
                    block_state = block_states.pop(event.index, None)
                    if block_state and block_state.type == "tool_use" and block_state.tool_use_id:
                        yield ToolUseEnd(index=event.index)
                    elif block_state and block_state.type == "text":
                        yield TextEnd(index=event.index)

                elif event_type == "message_delta":
                    delta = getattr(event, "delta", None)
                    stop_reason = getattr(delta, "stop_reason", None)
                    if stop_reason is not None:
                        finish_reason = self._map_finish_reason(stop_reason)

                    if getattr(event, "usage", None) is not None:
                        usage = self._usage_from_anthropic(event.usage)

                elif event_type == "message_stop":
                    for index in sorted(block_states):
                        block_state = block_states[index]
                        if block_state.type == "text":
                            yield TextEnd(index=index)
                        elif block_state.type == "tool_use" and block_state.tool_use_id:
                            yield ToolUseEnd(index=index)
                    block_states.clear()
                    sent_end = True
                    yield StreamEnd(finish_reason=finish_reason, usage=usage)
        except Exception as exc:
            raise self._map_error(exc) from exc

        if not sent_end:
            yield StreamEnd(finish_reason=finish_reason, usage=usage)

    def _build_request(
        self,
        *,
        messages: list[Message],
        tools: list[ToolSchema] | None,
        temperature: float,
        max_tokens: int | None,
        system: str | None,
        stream: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [self._message_to_anthropic(message) for message in messages],
            "max_tokens": max_tokens or 1024,
            "temperature": temperature,
            "stream": stream,
        }

        if system is not None:
            payload["system"] = system

        if tools:
            payload["tools"] = [self._tool_to_anthropic(tool) for tool in tools]

        return payload

    def _message_to_anthropic(self, message: Message) -> dict[str, Any]:
        if message.role == "user":
            return {"role": "user", "content": self._user_content_to_anthropic(message.content)}

        if message.role == "assistant":
            return {"role": "assistant", "content": self._assistant_content_to_anthropic(message.content)}

        if message.role == "tool":
            return {"role": "user", "content": self._tool_result_content_to_anthropic(message.content)}

        if message.role == "system":
            return {"role": "user", "content": self._user_content_to_anthropic(message.content)}

        raise ProviderError(f"Unsupported message role: {message.role}", provider=self.name)

    def _user_content_to_anthropic(self, content: list[object]) -> str:
        text_parts: list[str] = []
        for block in content:
            if not isinstance(block, TextContent):
                raise ProviderError(
                    f"Unsupported user content block: {type(block).__name__}",
                    provider=self.name,
                )
            text_parts.append(block.text)
        return "".join(text_parts)

    def _assistant_content_to_anthropic(self, content: list[object]) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        for block in content:
            if isinstance(block, TextContent):
                blocks.append({"type": "text", "text": block.text})
            elif isinstance(block, ToolUseContent):
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    }
                )
            else:
                raise ProviderError(
                    f"Unsupported assistant content block: {type(block).__name__}",
                    provider=self.name,
                )
        return blocks

    def _tool_result_content_to_anthropic(self, content: list[object]) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        for block in content:
            if not isinstance(block, ToolResultContent):
                raise ProviderError(
                    f"Unsupported tool content block: {type(block).__name__}",
                    provider=self.name,
                )
            blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.tool_use_id,
                    "content": block.content,
                    "is_error": block.is_error,
                }
            )
        return blocks

    def _tool_to_anthropic(self, tool: ToolSchema) -> dict[str, Any]:
        return {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.parameters,
        }

    def _assistant_message_from_anthropic(self, response: Any) -> Message:
        content: list[object] = []

        for block in getattr(response, "content", []) or []:
            if block.type == "text":
                content.append(TextContent(text=block.text))
            elif block.type == "tool_use":
                content.append(
                    ToolUseContent(
                        id=block.id,
                        name=block.name,
                        input=block.input,
                    )
                )
            else:
                raise ProviderError(
                    f"Unsupported Anthropic response block: {block.type}",
                    provider=self.name,
                )

        return Message(role="assistant", content=content)

    def _usage_from_anthropic(self, usage: Any) -> Usage:
        if usage is None:
            return Usage()
        return Usage(
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
        )

    def _map_finish_reason(self, finish_reason: str | None) -> str:
        if finish_reason in {"end_turn", "stop_sequence", "stop"}:
            return "stop"
        if finish_reason == "tool_use":
            return "tool_use"
        if finish_reason == "max_tokens":
            return "max_tokens"
        return "error"

    def _map_error(self, exc: Exception) -> ProviderError:
        message = str(exc)

        if isinstance(exc, AnthropicRateLimitError):
            return RateLimitError(message, provider=self.name)

        if isinstance(exc, AnthropicAuthenticationError):
            return AuthError(message, provider=self.name)

        if isinstance(exc, AnthropicAPITimeoutError):
            return ProviderTimeoutError(message, provider=self.name)

        if isinstance(exc, AnthropicBadRequestError):
            if "context_length_exceeded" in message:
                return ContextLengthError(message, provider=self.name)
            return ProviderError(message, provider=self.name)

        if isinstance(exc, ProviderError):
            return exc

        return ProviderError(message, provider=self.name)
