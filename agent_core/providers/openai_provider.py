from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from openai import (
    APITimeoutError as OpenAIAPITimeoutError,
    AsyncOpenAI,
    AuthenticationError as OpenAIAuthenticationError,
    BadRequestError as OpenAIBadRequestError,
    RateLimitError as OpenAIRateLimitError,
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
class _ToolStreamState:
    id: str | None = None
    name: str | None = None
    started: bool = False
    pending_arguments: list[str] = field(default_factory=list)


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self, config: ProviderConfig, client: Any | None = None):
        super().__init__(config)
        self._client = client or AsyncOpenAI(
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
            response = await self._client.chat.completions.create(
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

        choice = response.choices[0]
        return CompletionResponse(
            message=self._assistant_message_from_openai(choice.message),
            finish_reason=self._map_finish_reason(choice.finish_reason),
            usage=self._usage_from_openai(getattr(response, "usage", None)),
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
            stream = await self._client.chat.completions.create(
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

        tool_states: dict[int, _ToolStreamState] = {}
        tool_block_indices: dict[int, int] = {}
        finish_reason = "stop"
        usage = Usage()
        text_started = False
        text_block_index: int | None = None
        next_block_index = 0

        try:
            async for chunk in stream:
                if getattr(chunk, "usage", None) is not None:
                    usage = self._usage_from_openai(chunk.usage)

                for choice in getattr(chunk, "choices", []):
                    finish = getattr(choice, "finish_reason", None)
                    if finish is not None:
                        finish_reason = self._map_finish_reason(finish)

                    delta = getattr(choice, "delta", None)
                    if delta is None:
                        continue

                    if getattr(delta, "content", None):
                        if text_block_index is None:
                            text_block_index = next_block_index
                            next_block_index += 1
                        if not text_started:
                            text_started = True
                            yield TextStart(index=text_block_index)
                        yield TextDelta(index=text_block_index, text=delta.content)

                    for partial_tool_call in getattr(delta, "tool_calls", []) or []:
                        tool_state = tool_states.setdefault(partial_tool_call.index, _ToolStreamState())
                        block_index = tool_block_indices.get(partial_tool_call.index)
                        if block_index is None:
                            block_index = next_block_index
                            tool_block_indices[partial_tool_call.index] = block_index
                            next_block_index += 1

                        if getattr(partial_tool_call, "id", None):
                            tool_state.id = partial_tool_call.id

                        function = getattr(partial_tool_call, "function", None)
                        if function is not None and getattr(function, "name", None):
                            tool_state.name = function.name

                        if not tool_state.started and tool_state.id and tool_state.name:
                            tool_state.started = True
                            yield ToolUseStart(index=block_index, id=tool_state.id, name=tool_state.name)
                            for pending in tool_state.pending_arguments:
                                yield ToolUseDelta(index=block_index, partial_json=pending)
                            tool_state.pending_arguments.clear()

                        arguments_delta = None
                        if function is not None:
                            arguments_delta = getattr(function, "arguments", None)

                        if arguments_delta:
                            if tool_state.started:
                                yield ToolUseDelta(index=block_index, partial_json=arguments_delta)
                            else:
                                tool_state.pending_arguments.append(arguments_delta)
        except Exception as exc:
            raise self._map_error(exc) from exc

        if text_started and text_block_index is not None:
            yield TextEnd(index=text_block_index)

        for tool_index, tool_state in sorted(
            tool_states.items(),
            key=lambda item: tool_block_indices[item[0]],
        ):
            if tool_state.started:
                yield ToolUseEnd(index=tool_block_indices[tool_index])

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
            "messages": self._messages_to_openai(messages, system=system),
            "temperature": temperature,
            "stream": stream,
        }

        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        if tools:
            payload["tools"] = [self._tool_to_openai(tool) for tool in tools]

        if stream:
            payload["stream_options"] = {"include_usage": True}

        return payload

    def _messages_to_openai(self, messages: list[Message], *, system: str | None) -> list[dict[str, Any]]:
        openai_messages: list[dict[str, Any]] = []
        if system:
            openai_messages.append({"role": "system", "content": system})

        for message in messages:
            openai_messages.extend(self._message_to_openai(message))

        return openai_messages

    def _message_to_openai(self, message: Message) -> list[dict[str, Any]]:
        if message.role in {"system", "user"}:
            return [
                {
                    "role": message.role,
                    "content": self._join_text_blocks(message.content),
                }
            ]

        if message.role == "assistant":
            text_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []

            for block in message.content:
                if isinstance(block, TextContent):
                    text_parts.append(block.text)
                elif isinstance(block, ToolUseContent):
                    tool_calls.append(
                        {
                            "id": block.id,
                            "type": "function",
                            "function": {
                                "name": block.name,
                                "arguments": json.dumps(block.input, ensure_ascii=False),
                            },
                        }
                    )
                else:
                    raise ProviderError(
                        f"Unsupported assistant content block: {type(block).__name__}",
                        provider=self.name,
                    )

            payload: dict[str, Any] = {
                "role": "assistant",
                "content": "".join(text_parts) if text_parts else None,
            }
            if tool_calls:
                payload["tool_calls"] = tool_calls
            return [payload]

        if message.role == "tool":
            tool_messages: list[dict[str, Any]] = []
            for block in message.content:
                if not isinstance(block, ToolResultContent):
                    raise ProviderError(
                        f"Unsupported tool content block: {type(block).__name__}",
                        provider=self.name,
                    )
                tool_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": block.tool_use_id,
                        "content": block.content,
                    }
                )
            return tool_messages

        raise ProviderError(f"Unsupported message role: {message.role}", provider=self.name)

    def _tool_to_openai(self, tool: ToolSchema) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }

    def _assistant_message_from_openai(self, message: Any) -> Message:
        content: list[object] = []

        if getattr(message, "content", None):
            content.append(TextContent(text=message.content))

        for tool_call in getattr(message, "tool_calls", []) or []:
            arguments = tool_call.function.arguments or "{}"
            try:
                parsed_arguments = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise ProviderError(
                    f"OpenAI returned invalid tool arguments JSON: {arguments}",
                    provider=self.name,
                ) from exc

            content.append(
                ToolUseContent(
                    id=tool_call.id,
                    name=tool_call.function.name,
                    input=parsed_arguments,
                )
            )

        return Message(role="assistant", content=content)

    def _join_text_blocks(self, content: list[object]) -> str:
        text_parts: list[str] = []
        for block in content:
            if not isinstance(block, TextContent):
                raise ProviderError(
                    f"Unsupported content block for text-only role: {type(block).__name__}",
                    provider=self.name,
                )
            text_parts.append(block.text)
        return "".join(text_parts)

    def _usage_from_openai(self, usage: Any) -> Usage:
        if usage is None:
            return Usage()
        return Usage(
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )

    def _map_finish_reason(self, finish_reason: str | None) -> str:
        if finish_reason == "stop":
            return "stop"
        if finish_reason in {"tool_calls", "function_call"}:
            return "tool_use"
        if finish_reason == "length":
            return "max_tokens"
        return "error"

    def _map_error(self, exc: Exception) -> ProviderError:
        message = str(exc)

        if isinstance(exc, OpenAIRateLimitError):
            return RateLimitError(message, provider=self.name)

        if isinstance(exc, OpenAIAuthenticationError):
            return AuthError(message, provider=self.name)

        if isinstance(exc, OpenAIAPITimeoutError):
            return ProviderTimeoutError(message, provider=self.name)

        if isinstance(exc, OpenAIBadRequestError):
            if "context_length_exceeded" in message:
                return ContextLengthError(message, provider=self.name)
            return ProviderError(message, provider=self.name)

        if isinstance(exc, ProviderError):
            return exc

        return ProviderError(message, provider=self.name)
