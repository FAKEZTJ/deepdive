from __future__ import annotations

from agent_core.types import (
    Message,
    StreamEnd,
    TextDelta,
    TextStart,
    ToolUseDelta,
    ToolUseEnd,
    ToolUseStart,
    Usage,
)


def test_reconstructor_rebuilds_mixed_text_and_tool_message():
    from agent_core.runtime.reconstruct import StreamReconstructor

    reconstructor = StreamReconstructor()
    events = [
        TextStart(index=0),
        TextDelta(index=0, text="Hello "),
        ToolUseStart(index=1, id="call_1", name="lookup"),
        TextDelta(index=0, text="world"),
        ToolUseDelta(index=1, partial_json='{"city":"Shang'),
        ToolUseDelta(index=1, partial_json='hai"}'),
        ToolUseEnd(index=1),
        StreamEnd(finish_reason="tool_use", usage=Usage(input_tokens=9, output_tokens=4)),
    ]

    for event in events:
        reconstructor.feed(event)

    assert reconstructor.build_message() == Message(
        role="assistant",
        content=[
            {"type": "text", "text": "Hello world"},
            {"type": "tool_use", "id": "call_1", "name": "lookup", "input": {"city": "Shanghai"}},
        ],
    )


def test_reconstructor_rebuilds_tool_only_message():
    from agent_core.runtime.reconstruct import StreamReconstructor

    reconstructor = StreamReconstructor()
    events = [
        ToolUseStart(index=0, id="call_1", name="lookup"),
        ToolUseDelta(index=0, partial_json='{"city":"Shanghai"}'),
        ToolUseEnd(index=0),
        StreamEnd(finish_reason="tool_use", usage=Usage(input_tokens=5, output_tokens=3)),
    ]

    for event in events:
        reconstructor.feed(event)

    assert reconstructor.build_message() == Message(
        role="assistant",
        content=[
            {"type": "tool_use", "id": "call_1", "name": "lookup", "input": {"city": "Shanghai"}},
        ],
    )


def test_reconstructor_falls_back_to_partial_json_when_stream_was_incomplete():
    from agent_core.runtime.reconstruct import StreamReconstructor

    reconstructor = StreamReconstructor()
    events = [
        ToolUseStart(index=0, id="call_1", name="lookup"),
        ToolUseDelta(index=0, partial_json='{"city":"Shang'),
        StreamEnd(finish_reason="error", usage=Usage(input_tokens=1, output_tokens=1)),
    ]

    for event in events:
        reconstructor.feed(event)

    assert reconstructor.build_message() == Message(
        role="assistant",
        content=[
            {
                "type": "tool_use",
                "id": "call_1",
                "name": "lookup",
                "input": {"_partial_json": '{"city":"Shang'},
            },
        ],
    )
