"""PostgreSQL-backed reliable turn queue using SKIP LOCKED."""

from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

from persistence.postgres import get_postgres_pool, init_postgres
from persistence.turn_store import (
    TURN_DONE,
    TURN_FAILED,
    TURN_PENDING,
    TURN_PROCESSING,
    Turn,
)


class PostgresTurnStore:
    """Postgres queue store for multi-worker web chat turns."""

    def __init__(self) -> None:
        init_postgres()
        self.pool = get_postgres_pool()

    def create_turn(
        self,
        *,
        user_id: int,
        session_id: int,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> Turn:
        turn_id = str(uuid4())
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO conversation_turns
                        (id, user_id, session_id, content, status, metadata_json)
                    VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                    RETURNING id, user_id, session_id, content, status, answer, error,
                              metadata_json, attempts, created_at, updated_at,
                              started_at, finished_at
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
                row = cur.fetchone()
            conn.commit()
        if row is None:
            raise RuntimeError(f"Created turn not returned: {turn_id}")
        return _row_to_turn(row)

    def get_turn(self, turn_id: str) -> Turn | None:
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, user_id, session_id, content, status, answer, error,
                           metadata_json, attempts, created_at, updated_at,
                           started_at, finished_at
                    FROM conversation_turns
                    WHERE id = %s
                    """,
                    (turn_id,),
                )
                row = cur.fetchone()
        return _row_to_turn(row) if row is not None else None

    def claim_next_pending(self) -> Turn | None:
        """Atomically claim the oldest pending turn with session-level serialization."""
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    WITH candidate AS (
                        SELECT id
                        FROM conversation_turns AS pending
                        WHERE pending.status = %s
                          AND NOT EXISTS (
                              SELECT 1
                              FROM conversation_turns AS active
                              WHERE active.status = %s
                                AND active.user_id = pending.user_id
                                AND active.session_id = pending.session_id
                          )
                        ORDER BY pending.created_at ASC, pending.id ASC
                        FOR UPDATE SKIP LOCKED
                        LIMIT 1
                    )
                    UPDATE conversation_turns AS turn
                    SET status = %s,
                        attempts = turn.attempts + 1,
                        started_at = COALESCE(turn.started_at, now()),
                        updated_at = now()
                    FROM candidate
                    WHERE turn.id = candidate.id
                    RETURNING turn.id, turn.user_id, turn.session_id, turn.content,
                              turn.status, turn.answer, turn.error, turn.metadata_json,
                              turn.attempts, turn.created_at, turn.updated_at,
                              turn.started_at, turn.finished_at
                    """,
                    (TURN_PENDING, TURN_PROCESSING, TURN_PROCESSING),
                )
                row = cur.fetchone()
            conn.commit()
        return _row_to_turn(row) if row is not None else None

    def mark_done(self, turn_id: str, answer: str) -> Turn:
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE conversation_turns
                    SET status = %s,
                        answer = %s,
                        error = NULL,
                        finished_at = now(),
                        updated_at = now()
                    WHERE id = %s
                    RETURNING id, user_id, session_id, content, status, answer, error,
                              metadata_json, attempts, created_at, updated_at,
                              started_at, finished_at
                    """,
                    (TURN_DONE, answer, turn_id),
                )
                row = cur.fetchone()
            conn.commit()
        if row is None:
            raise RuntimeError(f"Turn not found after mark_done: {turn_id}")
        return _row_to_turn(row)

    def mark_failed(self, turn_id: str, error: str) -> Turn:
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE conversation_turns
                    SET status = %s,
                        error = %s,
                        finished_at = now(),
                        updated_at = now()
                    WHERE id = %s
                    RETURNING id, user_id, session_id, content, status, answer, error,
                              metadata_json, attempts, created_at, updated_at,
                              started_at, finished_at
                    """,
                    (TURN_FAILED, error, turn_id),
                )
                row = cur.fetchone()
            conn.commit()
        if row is None:
            raise RuntimeError(f"Turn not found after mark_failed: {turn_id}")
        return _row_to_turn(row)


def _row_to_turn(row: Any) -> Turn:
    metadata = row.get("metadata_json") or {}
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
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
        created_at=_iso(row["created_at"]),
        updated_at=_iso(row["updated_at"]),
        started_at=_iso(row["started_at"]),
        finished_at=_iso(row["finished_at"]),
    )


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)
