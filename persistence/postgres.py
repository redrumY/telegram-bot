"""PostgreSQL persistence bootstrap for the web-scale backend."""

from __future__ import annotations

import logging
import threading
from typing import Any

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from config.settings import settings

logger = logging.getLogger(__name__)

_pool: ConnectionPool[Any] | None = None
_pool_lock = threading.Lock()
_schema_ready = False

POSTGRES_SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS web_users (
    id BIGINT PRIMARY KEY,
    display_name TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS conversation_sessions (
    user_id BIGINT NOT NULL,
    chat_id BIGINT NOT NULL,
    messages_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    last_consolidated INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, chat_id)
);

CREATE TABLE IF NOT EXISTS conversation_messages (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL,
    session_id BIGINT NOT NULL,
    seq INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL DEFAULT '',
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (user_id, session_id, seq)
);

CREATE INDEX IF NOT EXISTS idx_conversation_messages_session_seq
    ON conversation_messages (user_id, session_id, seq);

CREATE TABLE IF NOT EXISTS conversation_turns (
    id UUID PRIMARY KEY,
    user_id BIGINT NOT NULL,
    session_id BIGINT NOT NULL,
    content TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    answer TEXT,
    error TEXT,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    attempts INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_conversation_turns_status_created
    ON conversation_turns (status, created_at, id);

CREATE INDEX IF NOT EXISTS idx_conversation_turns_session_status
    ON conversation_turns (user_id, session_id, status, created_at);

CREATE TABLE IF NOT EXISTS memory_items (
    id UUID PRIMARY KEY,
    user_id BIGINT NOT NULL,
    memory_type TEXT NOT NULL,
    summary TEXT NOT NULL,
    embedding vector(1024),
    status TEXT NOT NULL DEFAULT 'active',
    source_ref TEXT,
    extra_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_memory_items_user_status
    ON memory_items (user_id, status, memory_type, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_memory_items_embedding_hnsw
    ON memory_items USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS memory_replacements (
    old_id UUID NOT NULL,
    new_id UUID NOT NULL,
    replaced_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (old_id, new_id)
);
"""


def get_postgres_pool() -> ConnectionPool[Any]:
    """Return a process-wide psycopg pool."""
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = ConnectionPool(
                    settings.POSTGRES_DSN,
                    min_size=1,
                    max_size=max(4, int(settings.WORKER_CONCURRENCY) + 2),
                    kwargs={"row_factory": dict_row},
                    open=True,
                )
    return _pool


def init_postgres() -> None:
    """Create Postgres/pgvector schema for the target web architecture."""
    global _schema_ready
    if _schema_ready:
        return
    pool = get_postgres_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(POSTGRES_SCHEMA)
        conn.commit()
    _schema_ready = True
    logger.info("PostgreSQL schema ready")


def close_postgres_pool() -> None:
    global _pool, _schema_ready
    if _pool is not None:
        _pool.close()
    _pool = None
    _schema_ready = False
