import sqlite3
import threading
from pathlib import Path
from typing import Final

import sqlite_vec

from config.settings import settings

# Ensure each thread has its own connection
_local: threading.local = None
_lock: threading.Lock = threading.Lock()

TABLE_SCHEMA: Final = """
-- Core memory storage
CREATE TABLE IF NOT EXISTS memory_items (
    id TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL,
    memory_type TEXT NOT NULL,
    summary TEXT NOT NULL,
    embedding BLOB,
    status TEXT NOT NULL DEFAULT 'active',
    source_ref TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Vector index for similarity search (sqlite-vec virtual table)
CREATE VIRTUAL TABLE IF NOT EXISTS vec_items USING vec0(
    embedding_id TEXT PRIMARY KEY,
    embedding FLOAT[1024]
);

-- Memory replacement tracking
CREATE TABLE IF NOT EXISTS memory_replacements (
    old_id TEXT NOT NULL,
    new_id TEXT NOT NULL,
    replaced_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (old_id, new_id)
);

-- Session persistence (对齐 akashic sessions.db)
CREATE TABLE IF NOT EXISTS conversation_sessions (
    user_id INTEGER NOT NULL,
    chat_id INTEGER NOT NULL,
    messages_json TEXT NOT NULL DEFAULT '[]',
    last_consolidated INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, chat_id)
);

-- Web/API turn queue. Each row is one user message waiting for agent execution.
CREATE TABLE IF NOT EXISTS conversation_turns (
    id TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL,
    session_id INTEGER NOT NULL,
    content TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    answer TEXT,
    error TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    attempts INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    finished_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_conversation_turns_status_created
    ON conversation_turns (status, created_at);

CREATE INDEX IF NOT EXISTS idx_conversation_turns_session_status
    ON conversation_turns (user_id, session_id, status, created_at);
"""


def _ensure_conversation_session_columns(conn: sqlite3.Connection) -> None:
    """Apply lightweight migrations for existing conversation_sessions tables."""
    rows = conn.execute("PRAGMA table_info(conversation_sessions)").fetchall()
    existing = {str(row[1]) for row in rows}
    if "last_consolidated" not in existing:
        conn.execute(
            "ALTER TABLE conversation_sessions "
            "ADD COLUMN last_consolidated INTEGER NOT NULL DEFAULT 0"
        )


def _ensure_conversation_turn_columns(conn: sqlite3.Connection) -> None:
    """Apply lightweight migrations for existing conversation_turns tables."""
    rows = conn.execute("PRAGMA table_info(conversation_turns)").fetchall()
    existing = {str(row[1]) for row in rows}
    if not existing:
        return
    if "metadata_json" not in existing:
        conn.execute(
            "ALTER TABLE conversation_turns "
            "ADD COLUMN metadata_json TEXT NOT NULL DEFAULT '{}'"
        )
    if "attempts" not in existing:
        conn.execute(
            "ALTER TABLE conversation_turns "
            "ADD COLUMN attempts INTEGER NOT NULL DEFAULT 0"
        )


def init_db() -> None:
    """Initialize database with schema."""
    db_path = Path(settings.DATABASE_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.executescript(TABLE_SCHEMA)
    _ensure_conversation_session_columns(conn)
    _ensure_conversation_turn_columns(conn)
    conn.commit()
    conn.close()


def get_connection() -> sqlite3.Connection:
    """Get thread-local database connection with vec extension loaded."""
    global _local
    if _local is None:
        _local = threading.local()

    conn = getattr(_local, "conn", None)
    if conn is None:
        with _lock:
            conn = sqlite3.connect(settings.DATABASE_PATH)
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
            _local.conn = conn
    return conn
