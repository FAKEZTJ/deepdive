from __future__ import annotations

import inspect
import re
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

from agent_core.apps.deepdive.prompt import SYSTEM_PROMPT, build_task
from agent_core.apps.deepdive.report import DeepdiveReport
from agent_core.apps.deepdive.tools import SourceTracker
from agent_core.persistence.session_store import SessionStore
from agent_core.providers.base import LLMProvider
from agent_core.runtime.events import RunCompleted, RunEvent, ToolCallCompleted
from agent_core.runtime.loop import AgentLoop, Budget
from agent_core.tools.builtins.http_get import HttpGetTool
from agent_core.tools.builtins.web_search import WebSearchTool
from agent_core.tools.builtins.write_file import WriteFileTool
from agent_core.tools.registry import ToolRegistry
from agent_core.types import Message, StreamEvent, TextContent

EventCallback = Callable[[RunEvent], Awaitable[None] | None]
StreamCallback = Callable[[StreamEvent], Awaitable[None] | None]

_SEARCH_RESULT_PATTERN = re.compile(
    r"### (?P<title>.+?)\nURL: (?P<url>https?://[^\s]+)\n(?P<snippet>[^#]*?)(?=\n###|\Z)",
    re.DOTALL,
)


async def deepdive_research(
    topic: str,
    *,
    provider: LLMProvider,
    output_dir: str = "./",
    budget: Budget | None = None,
    on_event: EventCallback | None = None,
    on_stream_event: StreamCallback | None = None,
    session_store: SessionStore | None = None,
    stream_llm: bool = False,
) -> DeepdiveReport:
    output_dir_path = Path(output_dir).expanduser().resolve()
    output_dir_path.mkdir(parents=True, exist_ok=True)
    report_path = output_dir_path / "report.md"

    tracker = SourceTracker()
    start_time = time.monotonic()

    registry = ToolRegistry(
        [
            WebSearchTool(),
            HttpGetTool(),
            WriteFileTool(allowed_root=str(output_dir_path)),
        ]
    )
    loop = AgentLoop(
        provider=provider,
        tools=registry,
        system_prompt=SYSTEM_PROMPT,
        budget=budget or Budget(max_steps=15, max_tokens=100_000, timeout_seconds=300),
        allowed_permissions={"read_only", "write"},
        session_store=session_store,
        stream_llm=stream_llm,
        on_llm_stream_event=on_stream_event,
    )

    final_completed: RunCompleted | None = None
    async for event in loop.run_stream(build_task(topic, report_path=str(report_path))):
        _update_source_tracker(event, tracker)
        if isinstance(event, RunCompleted):
            final_completed = event
        await _maybe_invoke(on_event, event)

    assert final_completed is not None

    if report_path.exists():
        markdown = report_path.read_text(encoding="utf-8")
    else:
        markdown = _message_to_text(final_completed.final_message) or "(agent did not produce a report)"

    total_cost_usd = 0.0
    if session_store is not None and loop.session_id is not None:
        session = await session_store.get_session(loop.session_id)
        if session is not None:
            total_cost_usd = session.total_cost_usd

    duration_seconds = time.monotonic() - start_time
    return DeepdiveReport(
        topic=topic,
        markdown=markdown,
        sources=tracker,
        total_steps=final_completed.total_steps,
        total_tokens=final_completed.total_usage.total_tokens,
        total_cost_usd=total_cost_usd,
        duration_seconds=duration_seconds,
        session_id=loop.session_id,
    )


def _update_source_tracker(event: RunEvent, tracker: SourceTracker) -> None:
    if not isinstance(event, ToolCallCompleted) or event.is_error:
        return

    if event.tool_name == "web_search":
        results = event.metadata.get("results")
        if isinstance(results, list):
            for result in results:
                if not isinstance(result, dict):
                    continue
                url = str(result.get("url", "")).strip()
                if not url:
                    continue
                tracker.record_search_result(
                    url=url,
                    title=str(result.get("title", "")).strip(),
                    snippet=str(result.get("snippet", "")).strip()[:300],
                    step=event.step,
                )
            return

        for match in _SEARCH_RESULT_PATTERN.finditer(event.output):
            tracker.record_search_result(
                url=match.group("url").strip(),
                title=match.group("title").strip(),
                snippet=match.group("snippet").strip()[:300],
                step=event.step,
            )
        return

    if event.tool_name == "http_get":
        url = event.metadata.get("url")
        if isinstance(url, str) and url.strip():
            tracker.record_fetch(url=url.strip(), step=event.step)


def _message_to_text(message: Message) -> str:
    return "".join(
        block.text
        for block in message.content
        if isinstance(block, TextContent)
    )


async def _maybe_invoke(
    callback: Callable[[object], Awaitable[None] | None] | None,
    payload: object,
) -> None:
    if callback is None:
        return
    result = callback(payload)
    if inspect.isawaitable(result):
        await result
