from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.markup import escape
from rich.table import Table
from rich.tree import Tree

from agent_core.observability.exporter import export_session_as_otel_json
from agent_core.persistence.session_store import EventRecord, SessionRecord, SessionStore
from agent_core.runtime.events import LLMCallCompleted, StepCompleted, StepStarted, ToolCallCompleted, ToolCallStarted

console = Console()


@click.group()
def cli() -> None:
    """agent-core CLI."""


@cli.command("sessions")
@click.option("--db", default="./agent.db", help="SQLite DB path")
@click.option("--limit", default=50, type=int, show_default=True)
def sessions_command(db: str, limit: int) -> None:
    asyncio.run(_list_sessions(db, limit))


@cli.command("trace")
@click.argument("session_id")
@click.option("--db", default="./agent.db", help="SQLite DB path")
@click.option("--json-output", is_flag=True, help="Print exported trace JSON instead of a tree.")
def trace_command(session_id: str, db: str, json_output: bool) -> None:
    asyncio.run(_show_trace(db, session_id, json_output=json_output))


@cli.command("export-trace")
@click.argument("session_id")
@click.option("--db", default="./agent.db", help="SQLite DB path")
@click.option("--output", required=True, type=click.Path(dir_okay=False, path_type=Path))
def export_trace_command(session_id: str, db: str, output: Path) -> None:
    asyncio.run(_export_trace(db, session_id, output))


async def _list_sessions(db: str, limit: int) -> None:
    store = SessionStore(db)
    await store.initialize()
    try:
        records = await store.list_sessions(limit=limit)

        table = Table(title="Sessions")
        table.add_column("ID")
        table.add_column("Created")
        table.add_column("Steps", justify="right")
        table.add_column("Tokens", justify="right")
        table.add_column("Cost", justify="right")
        table.add_column("Status")

        for record in records:
            table.add_row(
                record.id[:8] + "...",
                datetime.fromtimestamp(record.created_at).strftime("%Y-%m-%d %H:%M"),
                str(record.total_steps),
                f"{record.total_usage.total_tokens:,}",
                f"${record.total_cost_usd:.4f}" if record.total_cost_usd else "-",
                record.status,
            )

        console.print(table)
    finally:
        await store.close()


async def _show_trace(db: str, session_id: str, *, json_output: bool = False) -> None:
    store = SessionStore(db)
    await store.initialize()
    try:
        record = await store.get_session(session_id)
        if record is None:
            console.print(f"[red]Session {session_id} not found[/red]")
            return

        event_records = await store.get_event_records(session_id)
        if json_output:
            payload = await export_session_as_otel_json(store, session_id)
            console.print_json(json.dumps(payload, ensure_ascii=False))
            return

        tree = _build_trace_tree(record, event_records)
        console.print(tree)
    finally:
        await store.close()


async def _export_trace(db: str, session_id: str, output: Path) -> None:
    store = SessionStore(db)
    await store.initialize()
    try:
        payload = await export_session_as_otel_json(store, session_id)
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        console.print(f"Wrote trace JSON to {output}")
    finally:
        await store.close()


def _build_trace_tree(record: SessionRecord, event_records: list[EventRecord]) -> Tree:
    duration_s = max(record.updated_at - record.created_at, 0.0)
    cost_str = f", ${record.total_cost_usd:.4f}" if record.total_cost_usd else ""
    root = Tree(
        f"[bold]agent_run[/bold] "
        f"{escape(f'[{record.total_steps} steps, {duration_s:.1f}s{cost_str}, {record.status}]')}"
    )

    by_step: dict[int, list[EventRecord]] = {}
    for event_record in event_records:
        step = getattr(event_record.event, "step", None)
        if step is not None:
            by_step.setdefault(step, []).append(event_record)

    for step_num in sorted(by_step):
        step_records = by_step[step_num]
        step_duration_ms = _step_duration_ms(step_records)
        step_label = f"[cyan]step #{step_num}[/cyan]"
        if step_duration_ms is not None:
            step_label += " " + escape(f"[{step_duration_ms:.0f}ms]")
        step_node = root.add(step_label)

        llm_completed = next(
            (
                record_item.event
                for record_item in step_records
                if isinstance(record_item.event, LLMCallCompleted)
            ),
            None,
        )
        if llm_completed is not None:
            model_label = _format_llm_model_label(llm_completed)
            cost_label = (
                f", ${llm_completed.cost_usd:.4f}"
                if llm_completed.cost_usd is not None
                else ""
            )
            step_node.add(
                f"[green]llm_call[/green] "
                f"{escape(f'[{model_label}, {llm_completed.usage.input_tokens} in / {llm_completed.usage.output_tokens} out{cost_label}]')}"
            )

        tool_starts = [
            record_item.event
            for record_item in step_records
            if isinstance(record_item.event, ToolCallStarted)
        ]
        tool_completes = {
            record_item.event.tool_call_id: record_item.event
            for record_item in step_records
            if isinstance(record_item.event, ToolCallCompleted)
        }
        if tool_starts:
            dispatch_node = step_node.add(
                f"[yellow]tool_dispatch[/yellow] {escape(f'[{len(tool_starts)} tools]')}"
            )
            for tool_start in tool_starts:
                tool_completed = tool_completes.get(tool_start.tool_call_id)
                if tool_completed is None:
                    dispatch_node.add(
                        f"tool_call: {tool_start.tool_name} {escape('[pending]')}"
                    )
                    continue
                status = "[red]error[/red]" if tool_completed.is_error else "[green]ok[/green]"
                dispatch_node.add(
                    f"tool_call: {tool_start.tool_name} "
                    f"{escape(f'[{tool_completed.duration_ms:.0f}ms, ')}{status}{escape(']')}"
                )

    return root


def _step_duration_ms(step_records: list[EventRecord]) -> float | None:
    started = next(
        (record.created_at for record in step_records if isinstance(record.event, StepStarted)),
        None,
    )
    completed = next(
        (record.created_at for record in step_records if isinstance(record.event, StepCompleted)),
        None,
    )
    if started is None or completed is None:
        return None
    return max((completed - started) * 1000, 0.0)


def _format_llm_model_label(event: LLMCallCompleted) -> str:
    provider = event.provider or "unknown"
    model = event.model or "unknown"
    return f"{provider}:{model}"


if __name__ == "__main__":
    cli()
