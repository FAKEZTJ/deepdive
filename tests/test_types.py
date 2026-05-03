from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent_core.types import Message, TextContent, ToolResultContent, ToolSchema, ToolUseContent


def test_assistant_message_rejects_tool_result_blocks():
    with pytest.raises(ValidationError):
        Message(
            role="assistant",
            content=[ToolResultContent(tool_use_id="call_1", content="result")],
        )


def test_tool_message_rejects_text_blocks():
    with pytest.raises(ValidationError):
        Message(
            role="tool",
            content=[TextContent(text="not allowed")],
        )


def test_user_message_rejects_tool_use_blocks():
    with pytest.raises(ValidationError):
        Message(
            role="user",
            content=[ToolUseContent(id="call_1", name="lookup", input={"x": 1})],
        )


def test_tool_schema_requires_object_json_schema():
    with pytest.raises(ValidationError):
        ToolSchema(
            name="lookup",
            description="Lookup data",
            parameters={"type": "string"},
        )
