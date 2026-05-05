from __future__ import annotations

import pytest

from agent_core.testing import FakeProvider
from agent_core.types import CompletionResponse, Message, ToolUseContent, Usage


class _StaticEstimator:
    def __init__(self, tokens: int):
        self._tokens = tokens

    def estimate(
        self,
        messages: list[Message],
        *,
        system_prompt: str | None = None,
    ) -> int:
        return self._tokens


def _summary_response(text: str) -> CompletionResponse:
    return CompletionResponse(
        message=Message.assistant_text(text),
        finish_reason="stop",
        usage=Usage(input_tokens=10, output_tokens=5),
    )


@pytest.mark.anyio
async def test_compress_if_needed_returns_original_messages_when_below_threshold():
    from agent_core.context.manager import ContextManager

    provider = FakeProvider(scripted_responses=[])
    manager = ContextManager(
        provider=provider,
        threshold_tokens=100,
        keep_recent_pairs=1,
        token_estimator=_StaticEstimator(tokens=20),
    )
    messages = [Message.user("task"), Message.assistant_text("done")]

    compressed, changed = await manager.compress_if_needed(
        messages,
        system_prompt="You are helpful.",
    )

    assert changed is False
    assert compressed == messages
    assert provider.calls == []


@pytest.mark.anyio
async def test_compress_if_needed_summarizes_middle_and_keeps_recent_group():
    from agent_core.context.manager import ContextManager

    provider = FakeProvider(scripted_responses=[_summary_response("Summary text.")])
    manager = ContextManager(
        provider=provider,
        threshold_tokens=100,
        keep_recent_pairs=1,
        token_estimator=_StaticEstimator(tokens=200),
    )
    messages = [
        Message.user("original task"),
        Message.assistant_text("first response"),
        Message.assistant_text("most recent response"),
    ]

    compressed, changed = await manager.compress_if_needed(messages, system_prompt="System prompt")

    assert changed is True
    assert compressed == [
        Message.assistant_text(
            "<previous_conversation_summary>\n"
            "Summary text.\n"
            "</previous_conversation_summary>"
        ),
        Message.assistant_text("most recent response"),
    ]
    assert provider.calls[0]["system"] is None
    assert provider.calls[0]["messages"][0].role == "system"
    assert provider.calls[0]["messages"][1].role == "user"


@pytest.mark.anyio
async def test_compress_if_needed_keeps_tool_use_and_tool_result_in_same_tail_group():
    from agent_core.context.manager import ContextManager

    provider = FakeProvider(scripted_responses=[_summary_response("Summary text.")])
    manager = ContextManager(
        provider=provider,
        threshold_tokens=100,
        keep_recent_pairs=1,
        token_estimator=_StaticEstimator(tokens=200),
    )
    messages = [
        Message.user("task"),
        Message.assistant_text("earlier"),
        Message(
            role="assistant",
            content=[ToolUseContent(id="call_1", name="lookup", input={"city": "Shanghai"})],
        ),
        Message.tool_result("call_1", '{"temp": 25}'),
    ]

    compressed, changed = await manager.compress_if_needed(messages)

    assert changed is True
    assert compressed[0] == Message.assistant_text(
        "<previous_conversation_summary>\n"
        "Summary text.\n"
        "</previous_conversation_summary>"
    )
    assert compressed[1:] == messages[2:]
