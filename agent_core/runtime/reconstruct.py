from __future__ import annotations

import json
from dataclasses import dataclass, field

from agent_core.types import (
    ContentBlock,
    Message,
    StreamEvent,
    TextContent,
    ToolUseContent,
)


@dataclass
class _BlockBuilder:
    """Accumulate the intermediate state of one streamed content block."""

    kind: str
    text_parts: list[str] = field(default_factory=list)
    tool_id: str | None = None
    tool_name: str | None = None
    json_parts: list[str] = field(default_factory=list)


class StreamReconstructor:
    """Rebuild a final assistant Message from provider StreamEvents."""

    def __init__(self) -> None:
        self._builders: dict[int, _BlockBuilder] = {}
        self._block_order: list[int] = []

    def feed(self, event: StreamEvent) -> None:
        match event.type:
            case "text_start":
                self._builders[event.index] = _BlockBuilder(kind="text")
                self._remember_block_order(event.index)
            case "text_delta":
                self._builders[event.index].text_parts.append(event.text)
            case "text_end":
                return
            case "tool_use_start":
                self._builders[event.index] = _BlockBuilder(
                    kind="tool_use",
                    tool_id=event.id,
                    tool_name=event.name,
                )
                self._remember_block_order(event.index)
            case "tool_use_delta":
                self._builders[event.index].json_parts.append(event.partial_json)
            case "tool_use_end":
                return
            case "stream_end":
                return

    def build_message(self) -> Message:
        content: list[ContentBlock] = []
        for index in self._block_order:
            builder = self._builders[index]
            if builder.kind == "text":
                content.append(TextContent(text="".join(builder.text_parts)))
                continue

            raw_json = "".join(builder.json_parts) or "{}"
            try:
                parsed_input = json.loads(raw_json)
            except json.JSONDecodeError:
                parsed_input = {"_partial_json": raw_json}

            content.append(
                ToolUseContent(
                    id=builder.tool_id or "",
                    name=builder.tool_name or "",
                    input=parsed_input,
                )
            )
        return Message(role="assistant", content=content)

    def _remember_block_order(self, index: int) -> None:
        if index not in self._block_order:
            self._block_order.append(index)
