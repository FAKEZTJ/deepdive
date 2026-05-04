from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, cast

import aiosqlite
from pydantic import TypeAdapter

from agent_core.runtime.events import RunEvent
from agent_core.types import ContentBlock, Message, Usage


_UNSET = object()
_MESSAGE_CONTENT_ADAPTER = TypeAdapter(list[ContentBlock])
_RUN_EVENT_ADAPTER = TypeAdapter(RunEvent)


@dataclass
class SessionRecord:
    id: str
    created_at: float
    updated_at: float
    status: str
    system_prompt: str | None
    metadata: dict[str, Any]
    total_steps: int
    total_usage: Usage
    stop_reason: str | None
    error_message: str | None


class SessionStore:
    """SQLite-backed persistence for resumable agent sessions."""

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._conn: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        self._conn = await aiosqlite.connect(self._db_path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA foreign_keys=ON")
        await self._create_tables()
        await self._conn.commit()

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    async def _create_tables(self) -> None:
        conn = self._require_conn()
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                status TEXT NOT NULL CHECK (status IN ('running', 'completed', 'paused', 'error')),
                system_prompt TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                total_steps INTEGER NOT NULL DEFAULT 0,
                total_input_tokens INTEGER NOT NULL DEFAULT 0,
                total_output_tokens INTEGER NOT NULL DEFAULT 0,
                stop_reason TEXT,
                error_message TEXT
            )
            """
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sessions_updated_at ON sessions(updated_at DESC)"
        )
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                seq INTEGER NOT NULL,
                role TEXT NOT NULL,
                content_json TEXT NOT NULL,
                created_at REAL NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
                UNIQUE(session_id, seq)
            )
            """
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_session_seq ON messages(session_id, seq)"
        )
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                seq INTEGER NOT NULL,
                type TEXT NOT NULL,
                step INTEGER,
                data_json TEXT NOT NULL,
                created_at REAL NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
                UNIQUE(session_id, seq)
            )
            """
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_session_seq ON events(session_id, seq)"
        )

    async def create_session(
        self,
        *,
        system_prompt: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        conn = self._require_conn()
        session_id = str(uuid.uuid4())
        now = time.time()
        await conn.execute(
            """
            INSERT INTO sessions (
                id, created_at, updated_at, status, system_prompt, metadata_json
            ) VALUES (?, ?, ?, 'running', ?, ?)
            """,
            (session_id, now, now, system_prompt, json.dumps(metadata or {})),
        )
        await conn.commit()
        return session_id

    async def get_session(self, session_id: str) -> SessionRecord | None:
        conn = self._require_conn()
        async with conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)) as cursor:
            row = await cursor.fetchone()
        return self._row_to_session_record(row)

    async def list_sessions(
        self,
        *,
        status: str | None = None,
        limit: int = 50,
    ) -> list[SessionRecord]:
        conn = self._require_conn()
        query = "SELECT * FROM sessions"
        params: list[Any] = []
        if status is not None:
            query += " WHERE status = ?"
            params.append(status)
        query += " ORDER BY updated_at DESC, created_at DESC LIMIT ?"
        params.append(limit)

        async with conn.execute(query, params) as cursor:
            rows = await cursor.fetchall()
        return [record for row in rows if (record := self._row_to_session_record(row)) is not None]

    async def update_session_state(
        self,
        session_id: str,
        *,
        status: str | object = _UNSET,
        total_steps: int | object = _UNSET,
        usage_delta: Usage | None = None,
        stop_reason: str | None | object = _UNSET,
        error_message: str | None | object = _UNSET,
    ) -> None:
        conn = self._require_conn()
        await self._update_session_state_in_tx(
            conn,
            session_id,
            status=status,
            total_steps=total_steps,
            usage_delta=usage_delta,
            stop_reason=stop_reason,
            error_message=error_message,
        )
        await conn.commit()

    async def append_message(self, session_id: str, message: Message) -> int:
        conn = self._require_conn()
        seq = await self._append_message_in_tx(conn, session_id, message)
        await conn.commit()
        return seq

    async def replace_messages(self, session_id: str, messages: list[Message]) -> None:
        conn = self._require_conn()
        await conn.execute("BEGIN IMMEDIATE")
        try:
            await conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            for message in messages:
                await self._append_message_in_tx(conn, session_id, message)
        except Exception:
            await conn.rollback()
            raise
        else:
            await conn.commit()

    async def get_messages(self, session_id: str) -> list[Message]:
        conn = self._require_conn()
        async with conn.execute(
            """
            SELECT role, content_json
            FROM messages
            WHERE session_id = ?
            ORDER BY seq
            """,
            (session_id,),
        ) as cursor:
            rows = await cursor.fetchall()

        messages: list[Message] = []
        for row in rows:
            content = _MESSAGE_CONTENT_ADAPTER.validate_python(json.loads(row["content_json"]))
            messages.append(Message(role=row["role"], content=content))
        return messages

    async def append_event(self, session_id: str, event: RunEvent) -> int:
        conn = self._require_conn()
        seq = await self._next_seq(conn, "events", session_id)
        await conn.execute(
            """
            INSERT INTO events (session_id, seq, type, step, data_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                seq,
                event.type,
                getattr(event, "step", None),
                json.dumps(event.model_dump()),
                time.time(),
            ),
        )
        await conn.commit()
        return seq

    async def get_events(
        self,
        session_id: str,
        *,
        limit: int | None = None,
    ) -> list[RunEvent]:
        conn = self._require_conn()
        if limit is None:
            query = """
                SELECT data_json
                FROM events
                WHERE session_id = ?
                ORDER BY seq
            """
            params: list[Any] = [session_id]
            reverse = False
        else:
            query = """
                SELECT data_json
                FROM events
                WHERE session_id = ?
                ORDER BY seq DESC
                LIMIT ?
            """
            params = [session_id, limit]
            reverse = True

        async with conn.execute(query, params) as cursor:
            rows = await cursor.fetchall()

        data = [_RUN_EVENT_ADAPTER.validate_python(json.loads(row["data_json"])) for row in rows]
        if reverse:
            data.reverse()
        return cast(list[RunEvent], data)

    async def checkpoint_step(
        self,
        session_id: str,
        *,
        new_messages: list[Message],
        completed_step: int,
        usage_delta: Usage,
    ) -> None:
        conn = self._require_conn()
        await conn.execute("BEGIN IMMEDIATE")
        try:
            for message in new_messages:
                await self._append_message_in_tx(conn, session_id, message)
            await self._update_session_state_in_tx(
                conn,
                session_id,
                total_steps=completed_step,
                usage_delta=usage_delta,
            )
        except Exception:
            await conn.rollback()
            raise
        else:
            await conn.commit()

    async def complete_run(
        self,
        session_id: str,
        *,
        status: str,
        stop_reason: str | None = None,
        error_message: str | None = None,
    ) -> None:
        await self.update_session_state(
            session_id,
            status=status,
            stop_reason=stop_reason,
            error_message=error_message,
        )

    def _require_conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("SessionStore is not initialized")
        return self._conn

    async def _append_message_in_tx(
        self,
        conn: aiosqlite.Connection,
        session_id: str,
        message: Message,
    ) -> int:
        seq = await self._next_seq(conn, "messages", session_id)
        await conn.execute(
            """
            INSERT INTO messages (session_id, seq, role, content_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                session_id,
                seq,
                message.role,
                json.dumps([block.model_dump() for block in message.content]),
                time.time(),
            ),
        )
        return seq

    async def _next_seq(
        self,
        conn: aiosqlite.Connection,
        table: str,
        session_id: str,
    ) -> int:
        async with conn.execute(
            f"SELECT COALESCE(MAX(seq), 0) + 1 FROM {table} WHERE session_id = ?",
            (session_id,),
        ) as cursor:
            row = await cursor.fetchone()
        assert row is not None
        return int(row[0])

    async def _update_session_state_in_tx(
        self,
        conn: aiosqlite.Connection,
        session_id: str,
        *,
        status: str | object = _UNSET,
        total_steps: int | object = _UNSET,
        usage_delta: Usage | None = None,
        stop_reason: str | None | object = _UNSET,
        error_message: str | None | object = _UNSET,
    ) -> None:
        assignments = ["updated_at = ?"]
        params: list[Any] = [time.time()]

        if status is not _UNSET:
            assignments.append("status = ?")
            params.append(status)
        if total_steps is not _UNSET:
            assignments.append("total_steps = ?")
            params.append(total_steps)
        if usage_delta is not None:
            assignments.append("total_input_tokens = total_input_tokens + ?")
            assignments.append("total_output_tokens = total_output_tokens + ?")
            params.extend([usage_delta.input_tokens, usage_delta.output_tokens])
        if stop_reason is not _UNSET:
            assignments.append("stop_reason = ?")
            params.append(stop_reason)
        if error_message is not _UNSET:
            assignments.append("error_message = ?")
            params.append(error_message)

        params.append(session_id)
        await conn.execute(
            f"UPDATE sessions SET {', '.join(assignments)} WHERE id = ?",
            params,
        )

    def _row_to_session_record(self, row: aiosqlite.Row | None) -> SessionRecord | None:
        if row is None:
            return None
        return SessionRecord(
            id=row["id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            status=row["status"],
            system_prompt=row["system_prompt"],
            metadata=json.loads(row["metadata_json"]),
            total_steps=row["total_steps"],
            total_usage=Usage(
                input_tokens=row["total_input_tokens"],
                output_tokens=row["total_output_tokens"],
            ),
            stop_reason=row["stop_reason"],
            error_message=row["error_message"],
        )
