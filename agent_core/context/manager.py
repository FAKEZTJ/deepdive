from __future__ import annotations

import json
from typing import Protocol

from agent_core.providers.base import LLMProvider
from agent_core.types import Message, TextContent, ToolResultContent, ToolUseContent


class TokenEstimator(Protocol):
    def estimate(
        self,
        messages: list[Message],
        *,
        system_prompt: str | None = None,
    ) -> int: ...


class SimpleTokenEstimator:
    """Cheap token estimate based on serialized payload size."""

    def estimate(
        self,
        messages: list[Message],
        *,
        system_prompt: str | None = None,
    ) -> int:
        total = sum(
            len(json.dumps(message.model_dump(), ensure_ascii=False)) // 4
            for message in messages
        )
        if system_prompt:
            total += len(system_prompt.encode("utf-8")) // 4
        return total


class ContextManager:
    """Compress long-running context into summary + recent atomic groups."""

    def __init__(
        self,
        *,
        provider: LLMProvider,
        threshold_tokens: int = 8000,
        keep_recent_pairs: int = 4,
        token_estimator: TokenEstimator | None = None,
    ) -> None:
        self._provider = provider
        self._threshold = threshold_tokens
        self._keep_pairs = keep_recent_pairs
        self._estimator = token_estimator or SimpleTokenEstimator()

    async def compress_if_needed(
        self,
        messages: list[Message],
        *,
        system_prompt: str | None = None,
    ) -> tuple[list[Message], bool]:
        if self._estimator.estimate(messages, system_prompt=system_prompt) < self._threshold:
            return messages, False

        groups = self._group_messages(messages)
        if len(groups) <= self._keep_pairs:
            return messages, False

        split_idx = len(groups) - self._keep_pairs
        middle_groups = groups[:split_idx]
        tail_groups = groups[split_idx:]

        middle = [message for group in middle_groups for message in group]
        if not middle:
            return messages, False

        tail = [message for group in tail_groups for message in group]
        summary_text = await self._summarize(middle)
        summary_message = Message.assistant_text(
            "<previous_conversation_summary>\n"
            f"{summary_text}\n"
            "</previous_conversation_summary>"
        )
        return [summary_message, *tail], True

    async def _summarize(self, messages: list[Message]) -> str:
        transcript = self._render_transcript(messages)
        response = await self._provider.chat(
            messages=[
                Message.system(
                    "You are a conversation summarizer for an AI agent system. "
                    "Summarize the following exchange concisely, preserving:\n"
                    "- Key user requests and goals\n"
                    "- Tool calls and their important results\n"
                    "- Decisions and conclusions\n"
                    "- Any open questions or pending tasks\n"
                    "Output as plain text. Do not exceed 300 words."
                ),
                Message.user(transcript),
            ],
            temperature=0.0,
            max_tokens=600,
        )
        for block in response.message.content:
            if isinstance(block, TextContent):
                return block.text
        return "(empty summary)"

    @staticmethod
    def _group_messages(messages: list[Message]) -> list[list[Message]]:
        groups: list[list[Message]] = []
        idx = 0
        while idx < len(messages):
            current = messages[idx]
            if (
                current.role == "assistant"
                and any(isinstance(block, ToolUseContent) for block in current.content)
                and idx + 1 < len(messages)
                and messages[idx + 1].role == "tool"
            ):
                groups.append([current, messages[idx + 1]])
                idx += 2
                continue
            groups.append([current])
            idx += 1
        return groups

    @staticmethod
    def _render_transcript(messages: list[Message]) -> str:
        lines: list[str] = []
        for message in messages:
            for block in message.content:
                if isinstance(block, TextContent):
                    lines.append(f"[{message.role}] {block.text}")
                elif isinstance(block, ToolUseContent):
                    payload = json.dumps(block.input, ensure_ascii=False)
                    lines.append(f"[{message.role}] tool_call: {block.name}({payload})")
                elif isinstance(block, ToolResultContent):
                    error_suffix = " (ERROR)" if block.is_error else ""
                    lines.append(f"[{message.role}]{error_suffix} tool_result: {block.content[:500]}")
        return "\n".join(lines)
