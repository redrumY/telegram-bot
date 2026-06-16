"""PostgreSQL + pgvector memory store."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from agent.core.types import MemoryItem
from memory.embedder import Embedder
from persistence.postgres import get_postgres_pool, init_postgres


class PostgresMemoryStore:
    def __init__(self, embedder: Embedder) -> None:
        init_postgres()
        self.embedder = embedder
        self.pool = get_postgres_pool()

    async def upsert_item(
        self,
        memory_type: str,
        summary: str,
        user_id: int,
        emotional_weight: int = 0,
        source_ref: str | None = None,
    ) -> MemoryItem:
        embedding = await self.embedder.embed(summary)
        item_id = uuid4()
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO memory_items
                        (id, user_id, memory_type, summary, embedding, status, source_ref)
                    VALUES (%s, %s, %s, %s, %s::vector, 'active', %s)
                    RETURNING id, user_id, memory_type, summary, embedding::text AS embedding,
                              status, source_ref, created_at, updated_at
                    """,
                    (
                        str(item_id),
                        int(user_id),
                        memory_type,
                        summary,
                        _vector_literal(embedding),
                        source_ref,
                    ),
                )
                row = cur.fetchone()
            conn.commit()
        return _row_to_memory(row)

    async def vector_search(
        self,
        query_vec: list[float],
        user_id: int,
        top_k: int = 5,
        memory_types: list[str] | None = None,
        include_superseded: bool = False,
    ) -> list[MemoryItem]:
        statuses = ["active", "superseded"] if include_superseded else ["active"]
        clauses = ["user_id = %s", "status = ANY(%s)", "embedding IS NOT NULL"]
        params: list[Any] = [int(user_id), statuses]
        if memory_types:
            clauses.append("memory_type = ANY(%s)")
            params.append(memory_types)
        query_literal = _vector_literal(query_vec)
        params.extend([query_literal, query_literal, max(1, int(top_k))])
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT id, user_id, memory_type, summary, embedding::text AS embedding,
                           status, source_ref, created_at, updated_at,
                           embedding <=> %s::vector AS distance
                    FROM memory_items
                    WHERE {' AND '.join(clauses)}
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    tuple(params),
                )
                rows = cur.fetchall()
        return [_row_to_memory(row) for row in rows]

    async def keyword_search(
        self,
        terms: str,
        user_id: int,
        limit: int = 3,
        memory_types: list[str] | None = None,
        include_superseded: bool = False,
    ) -> list[MemoryItem]:
        statuses = ["active", "superseded"] if include_superseded else ["active"]
        clauses = ["user_id = %s", "status = ANY(%s)", "summary ILIKE %s"]
        params: list[Any] = [int(user_id), statuses, f"%{terms}%"]
        if memory_types:
            clauses.append("memory_type = ANY(%s)")
            params.append(memory_types)
        params.append(max(1, int(limit)))
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT id, user_id, memory_type, summary, embedding::text AS embedding,
                           status, source_ref, created_at, updated_at
                    FROM memory_items
                    WHERE {' AND '.join(clauses)}
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    tuple(params),
                )
                rows = cur.fetchall()
        return [_row_to_memory(row) for row in rows]

    def list_memories(
        self,
        *,
        user_id: int,
        memory_types: list[str] | None = None,
        created_start: datetime | None = None,
        created_end: datetime | None = None,
        include_superseded: bool = False,
        limit: int = 50,
    ) -> list[MemoryItem]:
        statuses = ["active", "superseded"] if include_superseded else ["active"]
        clauses = ["user_id = %s", "status = ANY(%s)"]
        params: list[Any] = [int(user_id), statuses]
        if memory_types:
            clauses.append("memory_type = ANY(%s)")
            params.append(memory_types)
        if created_start is not None:
            clauses.append("created_at >= %s")
            params.append(created_start)
        if created_end is not None:
            clauses.append("created_at < %s")
            params.append(created_end)
        params.append(max(1, min(int(limit), 200)))
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT id, user_id, memory_type, summary, embedding::text AS embedding,
                           status, source_ref, created_at, updated_at
                    FROM memory_items
                    WHERE {' AND '.join(clauses)}
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    tuple(params),
                )
                rows = cur.fetchall()
        return [_row_to_memory(row) for row in rows]

    async def supersede(
        self, old_ids: list[UUID], new_id: UUID, relation_type: str = "supersede"
    ) -> None:
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                for old_id in old_ids:
                    cur.execute(
                        """
                        UPDATE memory_items
                        SET status = 'superseded', updated_at = now()
                        WHERE id = %s
                        """,
                        (str(old_id),),
                    )
                    cur.execute(
                        """
                        INSERT INTO memory_replacements (old_id, new_id)
                        VALUES (%s, %s)
                        ON CONFLICT DO NOTHING
                        """,
                        (str(old_id), str(new_id)),
                    )
            conn.commit()

    def mark_superseded_batch(
        self,
        ids: list[str | UUID],
        *,
        user_id: int | None = None,
    ) -> list[str]:
        clean_ids = []
        seen = set()
        for raw in ids or []:
            item_id = str(raw).strip()
            if item_id and item_id not in seen:
                seen.add(item_id)
                clean_ids.append(item_id)
        if not clean_ids:
            return []
        clauses = ["id = ANY(%s)", "status = 'active'"]
        params: list[Any] = [clean_ids]
        if user_id is not None:
            clauses.append("user_id = %s")
            params.append(int(user_id))
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    UPDATE memory_items
                    SET status = 'superseded', updated_at = now()
                    WHERE {' AND '.join(clauses)}
                    RETURNING id
                    """,
                    tuple(params),
                )
                rows = cur.fetchall()
            conn.commit()
        return [str(row["id"]) for row in rows]


def _row_to_memory(row: Any) -> MemoryItem:
    return MemoryItem(
        id=UUID(str(row["id"])),
        user_id=int(row["user_id"]),
        memory_type=str(row["memory_type"]),
        summary=str(row["summary"]),
        embedding=_parse_vector(row.get("embedding")),
        status=str(row["status"]),
        source_ref=row["source_ref"],
        created_at=_to_datetime(row.get("created_at")),
        updated_at=_to_datetime(row.get("updated_at")),
    )


def _vector_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{float(value):.8g}" for value in values) + "]"


def _parse_vector(raw: Any) -> list[float] | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    if not text:
        return []
    return [float(part) for part in text.split(",")]


def _to_datetime(raw: Any) -> datetime:
    if isinstance(raw, datetime):
        return raw
    if raw:
        return datetime.fromisoformat(str(raw))
    return datetime.utcnow()
