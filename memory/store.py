import sqlite3
import threading
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from agent.core.types import MemoryItem
from config.settings import settings
from memory.embedder import Embedder
from persistence.database import get_connection, init_db

LONG_TERM_MEMORY_TYPES = ["profile", "preference", "procedure", "event", "fact"]


class MemoryStore:
    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder

    async def upsert_item(
        self,
        memory_type: str,
        summary: str,
        user_id: int,
        emotional_weight: int = 0,
        source_ref: str | None = None,
    ) -> MemoryItem:
        """Insert or update a memory item with embedding."""
        # Generate embedding
        embedding = await self.embedder.embed(summary)
        item_id = uuid4()

        conn = get_connection()
        cursor = conn.cursor()

        # Insert into memory_items
        cursor.execute(
            """
            INSERT INTO memory_items (id, user_id, memory_type, summary, embedding, status, source_ref)
            VALUES (?, ?, ?, ?, ?, 'active', ?)
            """,
            (str(item_id), user_id, memory_type, summary, _encode_embedding(embedding), source_ref),
        )

        # Insert into vec_items for vector search
        cursor.execute(
            """
            INSERT INTO vec_items (embedding_id, embedding)
            VALUES (?, ?)
            """,
            (str(item_id), _encode_embedding(embedding)),
        )

        conn.commit()

        return MemoryItem(
            id=item_id,
            user_id=user_id,
            memory_type=memory_type,
            summary=summary,
            embedding=embedding,
            status="active",
            source_ref=source_ref,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

    async def vector_search(
        self,
        query_vec: list[float],
        user_id: int,
        top_k: int = 5,
        memory_types: list[str] | None = None,
        include_superseded: bool = False,
    ) -> list[MemoryItem]:
        """Search memories by vector similarity using sqlite-vec."""
        conn = get_connection()
        cursor = conn.cursor()

        query_bytes = _encode_embedding(query_vec)

        # Build type filter
        type_filter = ""
        params: list[Any] = []
        if memory_types:
            placeholders = ",".join(["?"] * len(memory_types))
            type_filter = f"AND mi.memory_type IN ({placeholders})"
            params.extend(memory_types)

        # sqlite-vec vector search using vec_distance_l2
        sql = f"""
            SELECT
                mi.id, mi.user_id, mi.memory_type, mi.summary,
                mi.embedding, mi.status, mi.source_ref, mi.created_at, mi.updated_at,
                vec_distance_l2(v.embedding, ?) as distance
            FROM vec_items v
            JOIN memory_items mi ON v.embedding_id = mi.id
            WHERE mi.user_id = ?
                AND mi.status IN ({','.join(['?'] * (2 if include_superseded else 1))})
                {type_filter}
            ORDER BY distance
            LIMIT ?
        """

        statuses = ["active", "superseded"] if include_superseded else ["active"]
        params = [query_bytes, user_id] + statuses + params + [top_k]
        cursor.execute(sql, params)

        results = []
        for row in cursor.fetchall():
            results.append(
                MemoryItem(
                    id=UUID(row[0]),
                    user_id=row[1],
                    memory_type=row[2],
                    summary=row[3],
                    embedding=_decode_embedding(row[4]),
                    status=row[5],
                    source_ref=row[6],
                    created_at=datetime.fromisoformat(row[7]) if row[7] else datetime.utcnow(),
                    updated_at=datetime.fromisoformat(row[8]) if row[8] else datetime.utcnow(),
                )
            )

        return results

    async def keyword_search(
        self,
        terms: str,
        user_id: int,
        limit: int = 3,
        memory_types: list[str] | None = None,
        include_superseded: bool = False,
    ) -> list[MemoryItem]:
        """Simple keyword search using LIKE."""
        conn = get_connection()
        cursor = conn.cursor()

        type_filter = ""
        params: list[Any] = [user_id]
        statuses = ["active", "superseded"] if include_superseded else ["active"]
        status_filter = ",".join(["?"] * len(statuses))
        params.extend(statuses)
        if memory_types:
            placeholders = ",".join(["?"] * len(memory_types))
            type_filter = f"AND memory_type IN ({placeholders})"
            params.extend(memory_types)
        params.extend([f"%{terms}%", limit])

        cursor.execute(
            f"""
            SELECT id, user_id, memory_type, summary, embedding, status, source_ref, created_at, updated_at
            FROM memory_items
            WHERE user_id = ? AND status IN ({status_filter}) {type_filter} AND summary LIKE ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            params,
        )

        results = []
        for row in cursor.fetchall():
            results.append(
                MemoryItem(
                    id=UUID(row[0]),
                    user_id=row[1],
                    memory_type=row[2],
                    summary=row[3],
                    embedding=_decode_embedding(row[4]),
                    status=row[5],
                    source_ref=row[6],
                    created_at=datetime.fromisoformat(row[7]) if row[7] else datetime.utcnow(),
                    updated_at=datetime.fromisoformat(row[8]) if row[8] else datetime.utcnow(),
                )
            )

        return results

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
        """List active memories with optional type and created_at filters."""
        conn = get_connection()
        cursor = conn.cursor()

        statuses = ["active", "superseded"] if include_superseded else ["active"]
        placeholders = ",".join(["?"] * len(statuses))
        clauses = ["user_id = ?", f"status IN ({placeholders})"]
        params: list[Any] = [user_id]
        params.extend(statuses)
        if memory_types:
            placeholders = ",".join(["?"] * len(memory_types))
            clauses.append(f"memory_type IN ({placeholders})")
            params.extend(memory_types)
        if created_start is not None:
            clauses.append("created_at >= ?")
            params.append(created_start.isoformat(sep=" "))
        if created_end is not None:
            clauses.append("created_at < ?")
            params.append(created_end.isoformat(sep=" "))

        params.append(max(1, min(int(limit), 200)))
        rows = cursor.execute(
            f"""
            SELECT id, user_id, memory_type, summary, embedding, status, source_ref, created_at, updated_at
            FROM memory_items
            WHERE {' AND '.join(clauses)}
            ORDER BY created_at DESC
            LIMIT ?
            """,
            params,
        ).fetchall()

        return [
            MemoryItem(
                id=UUID(row[0]),
                user_id=row[1],
                memory_type=row[2],
                summary=row[3],
                embedding=_decode_embedding(row[4]),
                status=row[5],
                source_ref=row[6],
                created_at=datetime.fromisoformat(row[7]) if row[7] else datetime.utcnow(),
                updated_at=datetime.fromisoformat(row[8]) if row[8] else datetime.utcnow(),
            )
            for row in rows
        ]

    async def supersede(
        self, old_ids: list[UUID], new_id: UUID, relation_type: str = "supersede"
    ) -> None:
        """Mark old memories as superseded and track the replacement."""
        conn = get_connection()
        cursor = conn.cursor()

        for old_id in old_ids:
            # Update status of old memory
            cursor.execute(
                """
                UPDATE memory_items
                SET status = 'superseded', updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (str(old_id),),
            )

            # Track replacement relationship
            cursor.execute(
                """
                INSERT INTO memory_replacements (old_id, new_id)
                VALUES (?, ?)
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
        """Mark active memories as superseded and return ids actually updated."""
        clean_ids = []
        seen = set()
        for raw in ids or []:
            item_id = str(raw).strip()
            if item_id and item_id not in seen:
                seen.add(item_id)
                clean_ids.append(item_id)
        if not clean_ids:
            return []

        conn = get_connection()
        cursor = conn.cursor()
        placeholders = ",".join(["?"] * len(clean_ids))
        params: list[Any] = clean_ids.copy()
        user_filter = ""
        if user_id is not None:
            user_filter = "AND user_id = ?"
            params.append(user_id)

        rows = cursor.execute(
            f"""
            SELECT id
            FROM memory_items
            WHERE id IN ({placeholders}) AND status = 'active' {user_filter}
            """,
            params,
        ).fetchall()
        updated_ids = [str(row[0]) for row in rows]
        if not updated_ids:
            return []

        update_placeholders = ",".join(["?"] * len(updated_ids))
        update_params: list[Any] = updated_ids.copy()
        if user_id is not None:
            update_params.append(user_id)

        cursor.execute(
            f"""
            UPDATE memory_items
            SET status = 'superseded', updated_at = CURRENT_TIMESTAMP
            WHERE id IN ({update_placeholders}) {user_filter}
            """,
            update_params,
        )
        conn.commit()
        return updated_ids


def _encode_embedding(vec: list[float]) -> bytes:
    """Encode float vector as bytes for sqlite-vec."""
    import struct

    return struct.pack(f"{len(vec)}f", *vec)


def _decode_embedding(data: bytes | None) -> list[float] | None:
    """Decode bytes to float vector."""
    if data is None:
        return None
    import struct

    return list(struct.unpack(f"{len(data) // 4}f", data))
