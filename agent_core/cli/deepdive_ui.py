from __future__ import annotations

from collections import defaultdict
from time import monotonic

from rich.console import Group
from rich.panel import Panel
from rich.text import Text
from rich.tree import Tree

from agent_core.runtime.events import (
    LLMCallCompleted,
    RunCompleted,
    RunEvent,
    StepStarted,
    ToolCallCompleted,
    ToolCallStarted,
)
from agent_core.types import StreamEvent


class DeepdiveUI:
    """Deepdive-specific live UI for CLI demos."""

    def __init__(self, *, topic: str, provider_label: str) -> None:
        self.topic = topic
        self.provider_label = provider_label
        self.start_time: float | None = None
        self.current_step = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        self.completed = False
        self._draft_text = ""
        self._steps: dict[int, list[tuple[object, ...]]] = defaultdict(list)
        self._tool_status: dict[str, tuple[int, int]] = {}

    def on_event(self, event: RunEvent) -> None:
        if self.start_time is None:
            self.start_time = monotonic()

        if isinstance(event, StepStarted):
            self.current_step = event.step
            self._draft_text = ""
            return

        if isinstance(event, LLMCallCompleted):
            self._steps[event.step].append(("llm_call", event.usage, event.provider, event.model, event.cost_usd))
            self.total_tokens += event.usage.total_tokens
            if event.cost_usd is not None:
                self.total_cost += event.cost_usd
            self._draft_text = ""
            return

        if isinstance(event, ToolCallStarted):
            step_events = self._steps[event.step]
            event_index = len(step_events)
            step_events.append(("tool", event.tool_name, "running", None, False))
            self._tool_status[event.tool_call_id] = (event.step, event_index)
            return

        if isinstance(event, ToolCallCompleted):
            location = self._tool_status.get(event.tool_call_id)
            if location is None:
                self._steps[event.step].append(
                    ("tool", event.tool_name, "completed", event.duration_ms, event.is_error)
                )
                return
            step, index = location
            self._steps[step][index] = (
                "tool",
                event.tool_name,
                "completed",
                event.duration_ms,
                event.is_error,
            )
            return

        if isinstance(event, RunCompleted):
            self.completed = True

    def on_stream_event(self, event: StreamEvent) -> None:
        if event.type == "text_delta":
            self._draft_text += event.text

    def render(self) -> Group:
        elapsed = (monotonic() - self.start_time) if self.start_time is not None else 0.0
        cost_line = f" | ${self.total_cost:.4f}" if self.total_cost > 0 else ""
        status = "completed" if self.completed else "running"
        header = Panel(
            f"[bold]Topic:[/bold] {self.topic}\n"
            f"[bold]Provider:[/bold] {self.provider_label}\n"
            f"[dim]Step {self.current_step} | {elapsed:.1f}s | "
            f"{self.total_tokens:,} tokens{cost_line} | {status}[/dim]",
            title="deepdive",
            border_style="cyan",
        )

        tree = Tree("[bold]Execution[/bold]")
        if not self._steps:
            tree.add("[dim]Waiting for first step...[/dim]")
        for step_num in sorted(self._steps):
            step_node = tree.add(f"[cyan]Step {step_num}[/cyan]")
            for item in self._steps[step_num]:
                kind = item[0]
                if kind == "llm_call":
                    usage = item[1]
                    provider = item[2] or "unknown"
                    model = item[3] or "unknown"
                    cost = item[4]
                    cost_suffix = f", ${cost:.4f}" if cost is not None else ""
                    step_node.add(
                        f"[green]LLM[/green] {provider}:{model} "
                        f"[dim]({usage.input_tokens} in / {usage.output_tokens} out{cost_suffix})[/dim]"
                    )
                    continue

                if kind == "tool":
                    tool_name = str(item[1])
                    state = str(item[2])
                    if state == "running":
                        step_node.add(f"[yellow]{tool_name}[/yellow] [dim]starting...[/dim]")
                        continue
                    duration_ms = item[3]
                    is_error = bool(item[4])
                    label = "error" if is_error else "ok"
                    color = "red" if is_error else "green"
                    step_node.add(
                        f"[yellow]{tool_name}[/yellow] [{color}]{label}[/] "
                        f"[dim]{duration_ms:.0f}ms[/dim]"
                    )

        draft_panel = Panel(
            Text(self._draft_text[-800:] if self._draft_text else "(waiting for model output)"),
            title="Assistant Draft",
            border_style="green",
        )
        return Group(header, tree, draft_panel)
