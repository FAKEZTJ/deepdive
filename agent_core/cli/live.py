from __future__ import annotations

from rich.console import Group
from rich.live import Live
from rich.markup import escape
from rich.panel import Panel
from rich.text import Text
from rich.tree import Tree

from agent_core.runtime.events import (
    LLMCallCompleted,
    LLMCallStarted,
    RunCompleted,
    RunEvent,
    StepStarted,
    ToolCallCompleted,
    ToolCallStarted,
)
from agent_core.types import StreamEvent


class RunLiveRenderer:
    """Render runtime events and transient LLM stream deltas side by side."""

    def __init__(self, *, task: str, provider_label: str) -> None:
        self._root = Tree(
            f"[bold]agent-core run[/bold] {escape(f'[{provider_label}]')} {escape(task)}"
        )
        self._step_nodes: dict[int, Tree] = {}
        self._dispatch_nodes: dict[int, Tree] = {}
        self._tool_nodes: dict[str, Tree] = {}
        self._draft_text = ""
        self._pending_tools: list[str] = []
        self._live = Live(self._renderable(), refresh_per_second=12, transient=True)

    def start(self) -> None:
        self._live.start()
        self._live.refresh()

    def stop(self) -> None:
        self._live.stop()

    def on_event(self, event: RunEvent) -> None:
        if isinstance(event, StepStarted):
            self._step_nodes[event.step] = self._root.add(f"[cyan]step #{event.step}[/cyan]")
        elif isinstance(event, LLMCallStarted):
            self._draft_text = ""
            self._pending_tools = []
        elif isinstance(event, LLMCallCompleted):
            step_node = self._step_nodes.get(event.step)
            if step_node is not None:
                cost_label = (
                    f", ${event.cost_usd:.4f}" if event.cost_usd is not None else ""
                )
                step_node.add(
                    escape(
                        f"llm_call [{event.provider}:{event.model}, "
                        f"{event.usage.input_tokens} in / {event.usage.output_tokens} out"
                        f"{cost_label}]"
                    )
                )
        elif isinstance(event, ToolCallStarted):
            dispatch_node = self._dispatch_nodes.get(event.step)
            if dispatch_node is None:
                step_node = self._step_nodes.setdefault(
                    event.step,
                    self._root.add(f"[cyan]step #{event.step}[/cyan]"),
                )
                dispatch_node = step_node.add("[yellow]tool_dispatch[/yellow]")
                self._dispatch_nodes[event.step] = dispatch_node
            self._tool_nodes[event.tool_call_id] = dispatch_node.add(
                escape(f"{event.tool_name} [running]")
            )
        elif isinstance(event, ToolCallCompleted):
            tool_node = self._tool_nodes.get(event.tool_call_id)
            if tool_node is not None:
                status = "error" if event.is_error else "ok"
                tool_node.label = escape(
                    f"{event.tool_name} [{status}, {event.duration_ms:.0f}ms]"
                )
        elif isinstance(event, RunCompleted):
            self._root.label = (
                f"[bold]agent-core run[/bold] "
                f"{escape(f'[{event.stop_reason}, {event.total_steps} steps, {event.total_usage.total_tokens} tokens]')}"
            )
        self._live.update(self._renderable(), refresh=True)

    def on_stream_event(self, event: StreamEvent) -> None:
        if event.type == "text_delta":
            self._draft_text += event.text
        elif event.type == "tool_use_start":
            self._pending_tools.append(event.name)
        self._live.update(self._renderable(), refresh=True)

    def _renderable(self) -> Group:
        draft = self._draft_text[-4000:] if self._draft_text else "(waiting for model output)"
        if self._pending_tools:
            draft = draft + "\n\nPending tools:\n" + "\n".join(
                f"- {name}" for name in self._pending_tools[-6:]
            )
        return Group(
            self._root,
            Panel(Text(draft), title="Assistant Draft", border_style="green"),
        )
