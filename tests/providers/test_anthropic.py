from __future__ import annotations

from types import SimpleNamespace

import pytest
import httpx
from anthropic import (
    APITimeoutError as AnthropicAPITimeoutError,
    AuthenticationError,
    BadRequestError,
    RateLimitError as AnthropicRateLimitError,
)

from agent_core.providers.base import ProviderConfig
from agent_core.providers.anthropic_provider import AnthropicProvider
from agent_core.types import (
    AuthError,
    ContextLengthError,
    Message,
    ProviderError,
    ProviderTimeoutError,
    RateLimitError,
    TextContent,
    ToolSchema,
    ToolUseContent,
)


def _provider_with_client(fake_client: object) -> AnthropicProvider:
    return AnthropicProvider(
        ProviderConfig(model="claude-sonnet-4-5", api_key="test-key"),
        client=fake_client,
    )


def _fake_error_response(status_code: int):
    return SimpleNamespace(
        request=SimpleNamespace(),
        status_code=status_code,
        headers={},
    )


@pytest.mark.anyio
async def test_chat_converts_messages_tools_and_response():
    captured: dict[str, object] = {}

    class FakeMessages:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                content=[
                    SimpleNamespace(type="text", text="Need tool output."),
                    SimpleNamespace(type="tool_use", id="call_1", name="weather_lookup", input={"city": "Shanghai"}),
                ],
                stop_reason="tool_use",
                usage=SimpleNamespace(input_tokens=12, output_tokens=7),
                model_dump=lambda: {"id": "resp_123"},
            )

    fake_client = SimpleNamespace(messages=FakeMessages())
    provider = _provider_with_client(fake_client)

    response = await provider.chat(
        messages=[
            Message.user("Hi"),
            Message(
                role="assistant",
                content=[
                    TextContent(text="Calling tool."),
                    ToolUseContent(id="call_0", name="lookup", input={"x": 1}),
                ],
            ),
            Message.tool_result("call_0", '{"ok": true}'),
        ],
        tools=[
            ToolSchema(
                name="weather_lookup",
                description="Fetch weather",
                parameters={"type": "object", "properties": {"city": {"type": "string"}}},
            )
        ],
        system="You are helpful.",
        temperature=0.2,
        max_tokens=128,
    )

    assert captured["model"] == "claude-sonnet-4-5"
    assert captured["system"] == "You are helpful."
    assert captured["temperature"] == 0.2
    assert captured["max_tokens"] == 128
    assert captured["tools"] == [
        {
            "name": "weather_lookup",
            "description": "Fetch weather",
            "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
        }
    ]
    assert captured["messages"] == [
        {"role": "user", "content": "Hi"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Calling tool."},
                {"type": "tool_use", "id": "call_0", "name": "lookup", "input": {"x": 1}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "call_0", "content": '{"ok": true}', "is_error": False}
            ],
        },
    ]

    assert response.finish_reason == "tool_use"
    assert response.usage.input_tokens == 12
    assert response.usage.output_tokens == 7
    assert response.raw == {"id": "resp_123"}
    assert response.message.role == "assistant"
    assert response.message.content[0].text == "Need tool output."
    assert response.message.content[1].id == "call_1"
    assert response.message.content[1].name == "weather_lookup"
    assert response.message.content[1].input == {"city": "Shanghai"}


@pytest.mark.anyio
async def test_chat_stream_emits_text_and_tool_events():
    class FakeStream:
        def __init__(self, events: list[object]):
            self._events = events

        def __aiter__(self):
            self._iter = iter(self._events)
            return self

        async def __anext__(self):
            try:
                return next(self._iter)
            except StopIteration as exc:
                raise StopAsyncIteration from exc

    class FakeMessages:
        async def create(self, **kwargs):
            return FakeStream(
                [
                    SimpleNamespace(
                        type="content_block_start",
                        index=0,
                        content_block=SimpleNamespace(type="text"),
                    ),
                    SimpleNamespace(
                        type="content_block_delta",
                        index=0,
                        delta=SimpleNamespace(type="text_delta", text="Hello "),
                    ),
                    SimpleNamespace(
                        type="content_block_start",
                        index=1,
                        content_block=SimpleNamespace(
                            type="tool_use",
                            id="call_1",
                            name="lookup",
                            input={},
                        ),
                    ),
                    SimpleNamespace(
                        type="content_block_delta",
                        index=1,
                        delta=SimpleNamespace(type="input_json_delta", partial_json='{"city":"Shang'),
                    ),
                    SimpleNamespace(
                        type="content_block_delta",
                        index=0,
                        delta=SimpleNamespace(type="text_delta", text="world"),
                    ),
                    SimpleNamespace(
                        type="content_block_delta",
                        index=1,
                        delta=SimpleNamespace(type="input_json_delta", partial_json='hai"}'),
                    ),
                    SimpleNamespace(type="content_block_stop", index=1),
                    SimpleNamespace(
                        type="message_delta",
                        delta=SimpleNamespace(stop_reason="tool_use"),
                        usage=SimpleNamespace(input_tokens=9, output_tokens=4),
                    ),
                    SimpleNamespace(type="message_stop"),
                ]
            )

    fake_client = SimpleNamespace(messages=FakeMessages())
    provider = _provider_with_client(fake_client)

    events = []
    async for event in provider.chat_stream(messages=[Message.user("Hi")], max_tokens=128):
        events.append(event)

    assert [event.type for event in events] == [
        "text_start",
        "text_delta",
        "tool_use_start",
        "tool_use_delta",
        "text_delta",
        "tool_use_delta",
        "tool_use_end",
        "text_end",
        "stream_end",
    ]
    assert events[0].index == 0
    assert events[1].index == 0
    assert events[1].text == "Hello "
    assert events[2].index == 1
    assert events[2].id == "call_1"
    assert events[2].name == "lookup"
    assert events[3].index == 1
    assert events[3].partial_json == '{"city":"Shang'
    assert events[4].index == 0
    assert events[4].text == "world"
    assert events[5].index == 1
    assert events[5].partial_json == 'hai"}'
    assert events[6].index == 1
    assert events[7].index == 0
    assert events[8].finish_reason == "tool_use"
    assert events[8].usage.input_tokens == 9
    assert events[8].usage.output_tokens == 4


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (
            BadRequestError(
                "prompt is too long: context_length_exceeded",
                response=_fake_error_response(400),
                body={},
            ),
            ContextLengthError,
        ),
        (
            AuthenticationError(
                "bad key",
                response=_fake_error_response(401),
                body={},
            ),
            AuthError,
        ),
        (
            AnthropicRateLimitError(
                "slow down",
                response=_fake_error_response(429),
                body={},
            ),
            RateLimitError,
        ),
        (
            AnthropicAPITimeoutError(
                request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
            ),
            ProviderTimeoutError,
        ),
        (
            BadRequestError(
                "bad request",
                response=_fake_error_response(400),
                body={},
            ),
            ProviderError,
        ),
    ],
)
async def test_chat_maps_anthropic_errors(error: Exception, expected: type[Exception]):
    class FakeMessages:
        async def create(self, **kwargs):
            raise error

    fake_client = SimpleNamespace(messages=FakeMessages())
    provider = _provider_with_client(fake_client)

    with pytest.raises(expected):
        await provider.chat(messages=[Message.user("Hi")], max_tokens=128)
