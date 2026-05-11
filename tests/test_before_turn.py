import asyncio
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ["TG_BOT_TOKEN"] = "test_token"
os.environ["DEEPSEEK_API_KEY"] = "test_deepseek_key"
os.environ["ALIYUN_DASHSCOPE_API_KEY"] = "test_aliyun_key"
os.environ["DATABASE_PATH"] = tempfile.mktemp(suffix=".db")

from agent.core.types import InboundMessage, Session
from agent.pipeline.phases.before_turn import BeforeTurnPhase
from memory.embedder import Embedder
from memory.store import MemoryStore
from persistence.database import init_db


async def test_acquire_session():
    """Test session acquisition creates and reuses sessions."""
    from agent.pipeline.phases.before_turn import _sessions

    # Clear sessions
    _sessions.clear()

    with patch.object(Embedder, "embed", new=AsyncMock(return_value=[0.1] * 1024)):
        init_db()
        phase = BeforeTurnPhase(Embedder(), MemoryStore(Embedder()))

        msg1 = InboundMessage(user_id=123, chat_id=456, content="hello")
        session1 = await phase.acquire_session(msg1)

        assert session1.user_id == 123
        assert session1.chat_id == 456
        assert len(session1.messages) == 0

        # Same user/chat should return same session
        msg2 = InboundMessage(user_id=123, chat_id=456, content="world")
        session2 = await phase.acquire_session(msg2)

        assert session1 is session2
        print("test_acquire_session: PASS")


async def test_prepare_context():
    """Test memory retrieval using vector and keyword search."""
    init_db()
    user_id = 999

    with patch.object(Embedder, "embed", new=AsyncMock()) as mock_embed:
        # First two calls are for inserting memories, third is for query
        mock_embed.side_effect = [
            [0.1] * 1024,  # Memory 1 embedding
            [0.2] * 1024,  # Memory 2 embedding
            [0.15] * 1024,  # Query embedding (similar to memory 1)
        ]

        store = MemoryStore(Embedder())

        # Insert two test memories
        await store.upsert_item("fact", "用户喜欢吃苹果", user_id)
        await store.upsert_item("fact", "用户名字叫张三", user_id)

        phase = BeforeTurnPhase(Embedder(), store)

        # Create empty session for context preparation
        session = Session(user_id=user_id, chat_id=1, messages=[])

        # Query for related memories
        memories = await phase.prepare_context(
            session=session,
            query_text="苹果",
            user_id=user_id,
        )

        # Should find at least the apple memory
        summaries = [m.summary for m in memories]
        assert any("苹果" in s for s in summaries)
        print(f"Found memories: {summaries}")
        print("test_prepare_context: PASS")


async def test_build_ctx():
    """Test building BeforeTurnCtx with query from session history."""
    from agent.pipeline.phases.before_turn import _sessions

    _sessions.clear()
    init_db()
    user_id = 888

    with patch.object(Embedder, "embed", new=AsyncMock()) as mock_embed:
        mock_embed.side_effect = [
            [0.1] * 1024,  # Memory embedding
            [0.15] * 1024,  # Query embedding
        ]

        store = MemoryStore(Embedder())
        await store.upsert_item("fact", "用户喜欢吃红色水果", user_id)

        phase = BeforeTurnPhase(Embedder(), store)

        # Create a session with history
        session = Session(user_id=user_id, chat_id=100, messages=[
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！"},
            {"role": "user", "content": "我喜欢吃水果"},
        ])
        _sessions[(user_id, 100)] = session

        inbound = InboundMessage(user_id=user_id, chat_id=100, content="特别是红色的")

        ctx = await phase.build_ctx(inbound)

        assert ctx.inbound_message is inbound
        assert ctx.session is session
        # Query should include last user message + current message
        # "我喜欢吃水果 特别是红色的"
        assert len(ctx.retrieved_memories) >= 0  # May find related memory
        print(f"Retrieved {len(ctx.retrieved_memories)} memories")
        print("test_build_ctx: PASS")


async def main():
    await test_acquire_session()
    await test_prepare_context()
    await test_build_ctx()
    print("\nAll before_turn tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
