from __future__ import annotations

from types import SimpleNamespace

import pytest
import httpx
from openai import (
    APITimeoutError as OpenAIAPITimeoutError,
    AuthenticationError,
    BadRequestError,
    RateLimitError as OpenAIRateLimitError,
)

from agent_core.providers.base import ProviderConfig
from agent_core.providers.openai_provider import OpenAIProvider
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


def _fake_tool_call(
    tool_id: str,
    name: str,
    arguments: str,
):
    return SimpleNamespace(
        id=tool_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _fake_chunk_tool_call(
    *,
    index: int,
    tool_id: str | None = None,
    name: str | None = None,
    arguments: str | None = None,
):
    return SimpleNamespace(
        index=index,
        id=tool_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _provider_with_client(fake_client: object) -> OpenAIProvider:
    return OpenAIProvider(
        ProviderConfig(model="gpt-4o-mini", api_key="test-key"),
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

    class FakeCompletions:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason="tool_calls",
                        message=SimpleNamespace(
                            content="Need tool output.",
                            tool_calls=[
                                _fake_tool_call("call_1", "weather_lookup", '{"city":"Shanghai"}')
                            ],
                        ),
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=12, completion_tokens=7),
                model_dump=lambda: {"id": "resp_123"},
            )

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
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

    assert captured["model"] == "gpt-4o-mini"
    assert captured["temperature"] == 0.2
    assert captured["max_tokens"] == 128
    assert captured["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "weather_lookup",
                "description": "Fetch weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        }
    ]
    assert captured["messages"] == [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi"},
        {
            "role": "assistant",
            "content": "Calling tool.",
            "tool_calls": [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"x": 1}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_0", "content": '{"ok": true}'},
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
        def __init__(self, chunks: list[object]):
            self._chunks = chunks

        def __aiter__(self):
            self._iter = iter(self._chunks)
            return self

        async def __anext__(self):
            try:
                return next(self._iter)
            except StopIteration as exc:
                raise StopAsyncIteration from exc

    class FakeCompletions:
        async def create(self, **kwargs):
            return FakeStream(
                [
                    SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                delta=SimpleNamespace(
                                    content="Hello ",
                                    tool_calls=[
                                        _fake_chunk_tool_call(
                                            index=0,
                                            tool_id="call_1",
                                            name="lookup",
                                            arguments="",
                                        )
                                    ],
                                ),
                                finish_reason=None,
                            )
                        ],
                        usage=None,
                    ),
                    SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                delta=SimpleNamespace(
                                    content="world",
                                    tool_calls=[
                                        _fake_chunk_tool_call(
                                            index=0,
                                            arguments='{"city":"Shang',
                                        )
                                    ],
                                ),
                                finish_reason=None,
                            )
                        ],
                        usage=None,
                    ),
                    SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                delta=SimpleNamespace(
                                    content=None,
                                    tool_calls=[
                                        _fake_chunk_tool_call(
                                            index=0,
                                            arguments='hai"}',
                                        )
                                    ],
                                ),
                                finish_reason="tool_calls",
                            )
                        ],
                        usage=SimpleNamespace(prompt_tokens=9, completion_tokens=4),
                    ),
                ]
            )

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    provider = _provider_with_client(fake_client)

    events = []
    async for event in provider.chat_stream(messages=[Message.user("Hi")]):
        events.append(event)

    assert [event.type for event in events] == [
        "text_start",
        "text_delta",
        "tool_use_start",
        "text_delta",
        "tool_use_delta",
        "tool_use_delta",
        "text_end",
        "tool_use_end",
        "stream_end",
    ]
    assert events[0].index == 0
    assert events[1].index == 0
    assert events[1].text == "Hello "
    assert events[2].index == 1
    assert events[2].id == "call_1"
    assert events[2].name == "lookup"
    assert events[3].index == 0
    assert events[3].text == "world"
    assert events[4].index == 1
    assert events[4].partial_json == '{"city":"Shang'
    assert events[5].index == 1
    assert events[5].partial_json == 'hai"}'
    assert events[6].index == 0
    assert events[7].index == 1
    assert events[8].finish_reason == "tool_use"
    assert events[8].usage.input_tokens == 9
    assert events[8].usage.output_tokens == 4


@pytest.mark.anyio
async def test_chat_stream_tool_only_response_starts_tool_index_at_zero():
    class FakeStream:
        def __init__(self, chunks: list[object]):
            self._chunks = chunks

        def __aiter__(self):
            self._iter = iter(self._chunks)
            return self

        async def __anext__(self):
            try:
                return next(self._iter)
            except StopIteration as exc:
                raise StopAsyncIteration from exc

    class FakeCompletions:
        async def create(self, **kwargs):
            return FakeStream(
                [
                    SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                delta=SimpleNamespace(
                                    content=None,
                                    tool_calls=[
                                        _fake_chunk_tool_call(
                                            index=0,
                                            tool_id="call_1",
                                            name="lookup",
                                            arguments='{"city":"Shanghai"}',
                                        )
                                    ],
                                ),
                                finish_reason="tool_calls",
                            )
                        ],
                        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=3),
                    ),
                ]
            )

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    provider = _provider_with_client(fake_client)

    events = []
    async for event in provider.chat_stream(messages=[Message.user("Hi")]):
        events.append(event)

    assert [event.type for event in events] == [
        "tool_use_start",
        "tool_use_delta",
        "tool_use_end",
        "stream_end",
    ]
    assert events[0].index == 0
    assert events[0].id == "call_1"
    assert events[1].index == 0
    assert events[1].partial_json == '{"city":"Shanghai"}'
    assert events[2].index == 0
    assert events[3].finish_reason == "tool_use"


@pytest.mark.anyio
async def test_chat_stream_tool_end_order_follows_block_index_not_first_seen_openai_index():
    class FakeStream:
        def __init__(self, chunks: list[object]):
            self._chunks = chunks

        def __aiter__(self):
            self._iter = iter(self._chunks)
            return self

        async def __anext__(self):
            try:
                return next(self._iter)
            except StopIteration as exc:
                raise StopAsyncIteration from exc

    class FakeCompletions:
        async def create(self, **kwargs):
            return FakeStream(
                [
                    SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                delta=SimpleNamespace(
                                    content=None,
                                    tool_calls=[
                                        _fake_chunk_tool_call(
                                            index=1,
                                            tool_id="call_2",
                                            name="lookup_b",
                                            arguments='{"b":2}',
                                        )
                                    ],
                                ),
                                finish_reason=None,
                            )
                        ],
                        usage=None,
                    ),
                    SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                delta=SimpleNamespace(
                                    content=None,
                                    tool_calls=[
                                        _fake_chunk_tool_call(
                                            index=0,
                                            tool_id="call_1",
                                            name="lookup_a",
                                            arguments='{"a":1}',
                                        )
                                    ],
                                ),
                                finish_reason="tool_calls",
                            )
                        ],
                        usage=SimpleNamespace(prompt_tokens=8, completion_tokens=4),
                    ),
                ]
            )

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    provider = _provider_with_client(fake_client)

    events = []
    async for event in provider.chat_stream(messages=[Message.user("Hi")]):
        events.append(event)

    assert [event.type for event in events] == [
        "tool_use_start",
        "tool_use_delta",
        "tool_use_start",
        "tool_use_delta",
        "tool_use_end",
        "tool_use_end",
        "stream_end",
    ]
    assert events[0].index == 0
    assert events[2].index == 1
    assert [event.index for event in events if event.type == "tool_use_end"] == [0, 1]


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (
            BadRequestError(
                "context_length_exceeded: too long",
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
            OpenAIRateLimitError(
                "slow down",
                response=_fake_error_response(429),
                body={},
            ),
            RateLimitError,
        ),
        (
            OpenAIAPITimeoutError(
                request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
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
async def test_chat_maps_openai_errors(error: Exception, expected: type[Exception]):
    class FakeCompletions:
        async def create(self, **kwargs):
            raise error

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    provider = _provider_with_client(fake_client)

    with pytest.raises(expected):
        await provider.chat(messages=[Message.user("Hi")])
