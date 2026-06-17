import os

os.environ["TG_BOT_TOKEN"] = "test_token"
os.environ["DEEPSEEK_API_KEY"] = "test_deepseek_key"
os.environ["ALIYUN_DASHSCOPE_API_KEY"] = "test_aliyun_key"

import pytest

from memory.postgres_store import PostgresMemoryStore


class FakeCursor:
    def __init__(self) -> None:
        self.sql = ""
        self.params = ()

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def execute(self, sql: str, params: tuple) -> None:
        self.sql = sql
        self.params = params

    def fetchall(self) -> list:
        return []


class FakeConnection:
    def __init__(self, cursor: FakeCursor) -> None:
        self._cursor = cursor

    def __enter__(self) -> "FakeConnection":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def cursor(self) -> FakeCursor:
        return self._cursor


class FakePool:
    def __init__(self, cursor: FakeCursor) -> None:
        self._cursor = cursor

    def connection(self) -> FakeConnection:
        return FakeConnection(self._cursor)


@pytest.mark.asyncio
async def test_vector_search_binds_query_vector_before_where_params() -> None:
    cursor = FakeCursor()
    store = object.__new__(PostgresMemoryStore)
    store.pool = FakePool(cursor)

    results = await store.vector_search(
        query_vec=[0.1, 0.2],
        user_id=42,
        top_k=3,
        memory_types=["profile"],
    )

    assert results == []
    assert "embedding <=> %s::vector AS distance" in cursor.sql
    assert cursor.params == (
        "[0.1,0.2]",
        42,
        ["active"],
        ["profile"],
        "[0.1,0.2]",
        3,
    )
