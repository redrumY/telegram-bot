"""Persistent turn queue for web/API chat requests."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any
from uuid import uuid4

from persistence.database import get_connection

TURN_PENDING = "pending"
TURN_PROCESSING = "processing"
TURN_DONE = "done"
TURN_FAILED = "failed"
TERMINAL_TURN_STATUSES = {TURN_DONE, TURN_FAILED}

_claim_lock = threading.Lock()


@dataclass(frozen=True)
class Turn:
    id: str
    user_id: int
    session_id: int
    content: str
    status: str
    answer: str | None
    error: str | None
    metadata: dict[str, Any]
    attempts: int
    created_at: str | None
    updated_at: str | None
    started_at: str | None
    finished_at: str | None


def _row_to_turn(row: Mapping[str, Any]) -> Turn:
    metadata_raw = row.get("metadata_json") or "{}"
    try:
        metadata = json.loads(metadata_raw or "{}")
    except json.JSONDecodeError:
        metadata = {}
    if not isinstance(metadata, dict):
        metadata = {}
    return Turn(
        id=str(row["id"]),
        user_id=int(row["user_id"]),
        session_id=int(row["session_id"]),
        content=str(row["content"]),
        status=str(row["status"]),
        answer=row["answer"],
        error=row["error"],
        metadata=metadata,
        attempts=int(row["attempts"] or 0),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
    )


class TurnStore:
    """SQLite-backed reliable queue for agent turns."""

    def create_turn(
        self,
        *,
        user_id: int,
        session_id: int,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> Turn:
        turn_id = str(uuid4())
        conn = get_connection()
        conn.execute(
            """
            INSERT INTO conversation_turns
                (id, user_id, session_id, content, status, metadata_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (
                turn_id,
                int(user_id),
                int(session_id),
                content,
                TURN_PENDING,
                json.dumps(metadata or {}, ensure_ascii=False),
            ),
        )
        conn.commit()
        turn = self.get_turn(turn_id)
        if turn is None:
            raise RuntimeError(f"Created turn not found: {turn_id}")
        return turn

    def get_turn(self, turn_id: str) -> Turn | None:
        conn = get_connection()
        row = _fetchone_dict(
            conn,
            """
            SELECT id, user_id, session_id, content, status, answer, error,
                   metadata_json, attempts, created_at, updated_at, started_at,
                   finished_at
            FROM conversation_turns
            WHERE id = ?
            """,
            (turn_id,),
        )
        return _row_to_turn(row) if row is not None else None

    def claim_next_pending(self) -> Turn | None:
        """Claim the oldest pending turn whose session is not already active."""
        with _claim_lock:
            conn = get_connection()
            row = _fetchone_dict(
                conn,
                """
                SELECT id
                FROM conversation_turns AS candidate
                WHERE candidate.status = ?
                  AND NOT EXISTS (
                      SELECT 1
                      FROM conversation_turns AS active
                      WHERE active.status = ?
                        AND active.user_id = candidate.user_id
                        AND active.session_id = candidate.session_id
                  )
                ORDER BY candidate.created_at ASC, candidate.rowid ASC
                LIMIT 1
                """,
                (TURN_PENDING, TURN_PROCESSING),
            )
            if row is None:
                return None
            turn_id = str(row["id"])
            conn.execute(
                """
                UPDATE conversation_turns
                SET status = ?,
                    attempts = attempts + 1,
                    started_at = COALESCE(started_at, CURRENT_TIMESTAMP),
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ? AND status = ?
                """,
                (TURN_PROCESSING, turn_id, TURN_PENDING),
            )
            conn.commit()
        return self.get_turn(turn_id)

    def mark_done(self, turn_id: str, answer: str) -> Turn:
        conn = get_connection()
        conn.execute(
            """
            UPDATE conversation_turns
            SET status = ?,
                answer = ?,
                error = NULL,
                finished_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (TURN_DONE, answer, turn_id),
        )
        conn.commit()
        turn = self.get_turn(turn_id)
        if turn is None:
            raise RuntimeError(f"Turn not found after mark_done: {turn_id}")
        return turn

    def mark_failed(self, turn_id: str, error: str) -> Turn:
        conn = get_connection()
        conn.execute(
            """
            UPDATE conversation_turns
            SET status = ?,
                error = ?,
                finished_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (TURN_FAILED, error, turn_id),
        )
        conn.commit()
        turn = self.get_turn(turn_id)
        if turn is None:
            raise RuntimeError(f"Turn not found after mark_failed: {turn_id}")
        return turn


def _fetchone_dict(
    conn: Any,
    sql: str,
    params: tuple[Any, ...] = (),
) -> dict[str, Any] | None:
    cursor = conn.execute(sql, params)
    row = cursor.fetchone()
    if row is None:
        return None
    return {column[0]: row[index] for index, column in enumerate(cursor.description)}


_turn_store: TurnStore | None = None


def get_turn_store() -> TurnStore:
    global _turn_store
    if _turn_store is None:
        from config.settings import settings

        if settings.TURN_STORE_BACKEND.lower() == "postgres":
            from persistence.postgres_turn_store import PostgresTurnStore

            _turn_store = PostgresTurnStore()  # type: ignore[assignment]
        else:
            _turn_store = TurnStore()
    return _turn_store
