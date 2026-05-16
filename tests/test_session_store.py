import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ["DATABASE_PATH"] = tempfile.mktemp(suffix=".db")

from persistence.database import init_db
from persistence.session_store import SessionStore


def test_search_messages_returns_source_refs() -> None:
    init_db()
    store = SessionStore()
    store.save(
        42,
        100,
        [
            {"role": "user", "content": "我是个程序员，最常用 Python"},
            {"role": "assistant", "content": "好的，我记住了。"},
            {"role": "user", "content": "我最近在学 Rust"},
        ],
    )

    messages, total = store.search_messages(
        "Python 程序员",
        user_id=42,
        role="user",
        limit=10,
    )

    assert total == 1
    assert len(messages) == 1
    assert messages[0]["seq"] == 0
    assert messages[0]["source_ref"] == "session:42:100#msg:0"
    print("test_search_messages_returns_source_refs: PASS")


def test_fetch_messages_by_message_ref_with_context() -> None:
    init_db()
    store = SessionStore()
    store.save(
        42,
        101,
        [
            {"role": "user", "content": "第一条"},
            {"role": "assistant", "content": "第二条"},
            {"role": "user", "content": "第三条 Python"},
        ],
    )

    messages, matched = store.fetch_messages(42, 101, seq=2, context=1)

    assert matched == 1
    assert [m["seq"] for m in messages] == [1, 2]
    assert messages[-1]["in_source_ref"] is True
    assert messages[-1]["source_ref"] == "session:42:101#msg:2"
    print("test_fetch_messages_by_message_ref_with_context: PASS")


def test_fetch_messages_by_message_range_ref() -> None:
    init_db()
    store = SessionStore()
    store.save(
        42,
        102,
        [
            {"role": "user", "content": "第一条"},
            {"role": "assistant", "content": "第二条"},
            {"role": "user", "content": "第三条 Python"},
            {"role": "assistant", "content": "第四条"},
        ],
    )

    messages, matched = store.fetch_messages(42, 102, seq=1, seq_end=2)

    assert matched == 2
    assert [m["seq"] for m in messages] == [1, 2]
    assert [m["in_source_ref"] for m in messages] == [True, True]
    print("test_fetch_messages_by_message_range_ref: PASS")


if __name__ == "__main__":
    test_search_messages_returns_source_refs()
    test_fetch_messages_by_message_ref_with_context()
    test_fetch_messages_by_message_range_ref()
    print("\nAll session store tests passed!")
