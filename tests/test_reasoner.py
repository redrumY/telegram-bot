import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ["TG_BOT_TOKEN"] = "test_token"
os.environ["DEEPSEEK_API_KEY"] = "test_deepseek_key"
os.environ["ALIYUN_DASHSCOPE_API_KEY"] = "test_aliyun_key"
os.environ["DATABASE_PATH"] = tempfile.mktemp(suffix=".db")
os.environ["DEEPSEEK_BASE_URL"] = "https://api.test.com"
os.environ["LLM_MODEL"] = "test-model"

from agent.core.types import BeforeReasoningCtx, MemoryItem, Session
from agent.pipeline.reasoner import Reasoner


def _mock_choice(content: str, tool_calls=None, finish_reason="stop"):
    """Helper to create a mock choice."""
    choice = MagicMock()
    choice.message.content = content
    choice.message.tool_calls = tool_calls or []
    choice.finish_reason = finish_reason
    return choice


def _mock_tool_call(id: str, name: str, arguments: str):
    """Helper to create a mock tool call."""
    tc = MagicMock()
    tc.id = id
    tc.type = "function"
    tc.function.name = name
    tc.function.arguments = arguments
    return tc


def _make_ctx(user_id=1, chat_id=1, messages=None, memories=None, tools=None):
    """Helper to create a BeforeReasoningCtx."""
    if messages is None:
        messages = [{"role": "user", "content": "你好"}]
    return BeforeReasoningCtx(
        session=Session(user_id=user_id, chat_id=chat_id, messages=[]),
        memories=memories or [],
        messages=messages,
        tools=tools or [],
    )


async def test_simple_response():
    """Test simple response without tool calls."""
    with patch("agent.pipeline.reasoner.AsyncOpenAI") as MockClient:
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [_mock_choice("你好！我是你的助手。")]
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        MockClient.return_value = mock_client

        reasoner = Reasoner()
        ctx = _make_ctx(messages=[{"role": "user", "content": "你好"}])

        result = await reasoner.run_turn(ctx)

        assert result.content == "你好！我是你的助手。"
        assert result.tool_calls == []
        assert result.finish_reason == "stop"
        print("test_simple_response: PASS")


async def test_recall_memory_tool():
    """Test recall_memory tool call. 需要 store + embedder mock。"""
    mock_store = AsyncMock()
    mock_embedder = AsyncMock()
    mock_embedder.embed = AsyncMock(return_value=[0.1] * 256)

    from uuid import uuid4
    mock_item = MemoryItem(
        id=uuid4(),
        user_id=1,
        memory_type="profile",
        summary="用户是程序员",
        embedding=[0.1] * 256,
        status="active",
        source_ref="session:1:1",
    )
    mock_store.vector_search = AsyncMock(return_value=[mock_item])
    mock_store.keyword_search = AsyncMock(return_value=[])

    with patch("agent.pipeline.reasoner.AsyncOpenAI") as MockClient:
        mock_client = AsyncMock()

        # First response: recall_memory tool call
        response1 = MagicMock()
        response1.choices = [_mock_choice(
            "",
            tool_calls=[_mock_tool_call(
                "call_1", "recall_memory",
                '{"query":"用户的职业是什么","memory_type":"profile","limit":5}'
            )],
            finish_reason="tool_calls"
        )]

        # Second response: answer based on recalled memory
        response2 = MagicMock()
        response2.choices = [_mock_choice("根据记忆，用户是程序员。")]

        mock_client.chat.completions.create = AsyncMock(
            side_effect=[response1, response2]
        )
        MockClient.return_value = mock_client

        reasoner = Reasoner(store=mock_store, embedder=mock_embedder)
        ctx = _make_ctx(messages=[{"role": "user", "content": "我是什么职业"}])

        result = await reasoner.run_turn(ctx)

        assert "程序员" in result.content
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["function"]["name"] == "recall_memory"
        mock_store.vector_search.assert_called_once()
        assert mock_store.vector_search.call_args.kwargs["user_id"] == 1
        print("test_recall_memory_tool: PASS")


async def test_recall_then_fetch_messages_tool_chain():
    """Test Akashic-style multi-step memory tools across LLM iterations."""
    mock_store = AsyncMock()
    mock_embedder = AsyncMock()
    mock_embedder.embed = AsyncMock(return_value=[0.1] * 256)
    mock_session_store = MagicMock()
    mock_session_store.load.return_value = [
        {"role": "user", "content": "我是个程序员，最常用 Python"},
        {"role": "assistant", "content": "好的，我记住了。"},
    ]
    mock_session_store.fetch_messages.return_value = (
        [
            {
                "role": "user",
                "content": "我是个程序员，最常用 Python",
                "seq": 0,
                "source_ref": "session:1:1#msg:0",
                "in_source_ref": True,
            }
        ],
        1,
    )

    from uuid import uuid4
    mock_item = MemoryItem(
        id=uuid4(),
        user_id=1,
        memory_type="profile",
        summary="用户是程序员，常用 Python。",
        embedding=[0.1] * 256,
        status="active",
        source_ref="session:1:1",
    )
    mock_store.vector_search = AsyncMock(return_value=[mock_item])
    mock_store.keyword_search = AsyncMock(return_value=[])

    with patch("agent.pipeline.reasoner.AsyncOpenAI") as MockClient:
        mock_client = AsyncMock()

        response1 = MagicMock()
        response1.choices = [_mock_choice(
            "",
            tool_calls=[_mock_tool_call(
                "call_1", "recall_memory",
                '{"query":"用户职业和技术栈","memory_type":"profile","limit":5}'
            )],
            finish_reason="tool_calls",
        )]

        response2 = MagicMock()
        response2.choices = [_mock_choice(
            "",
            tool_calls=[_mock_tool_call(
                "call_2", "fetch_messages",
                '{"source_ref":"session:1:1","limit":10}'
            )],
            finish_reason="tool_calls",
        )]

        response3 = MagicMock()
        response3.choices = [_mock_choice("你是程序员，主要用 Python。")]

        mock_client.chat.completions.create = AsyncMock(
            side_effect=[response1, response2, response3]
        )
        MockClient.return_value = mock_client

        reasoner = Reasoner(
            store=mock_store,
            embedder=mock_embedder,
            session_store=mock_session_store,
        )
        ctx = _make_ctx(
            messages=[{"role": "user", "content": "我是什么职业，用什么语言？"}],
            tools=[
                {"type": "function", "function": {"name": "recall_memory"}},
                {"type": "function", "function": {"name": "fetch_messages"}},
            ],
        )

        result = await reasoner.run_turn(ctx)

        assert result.content == "你是程序员，主要用 Python。"
        assert [c["function"]["name"] for c in result.tool_calls] == [
            "recall_memory",
            "fetch_messages",
        ]
        assert mock_client.chat.completions.create.call_count == 3
        for call in mock_client.chat.completions.create.call_args_list:
            assert call.kwargs["tools"] == ctx.tools
        mock_session_store.fetch_messages.assert_called_once()
        print("test_recall_then_fetch_messages_tool_chain: PASS")


async def test_fetch_messages_accepts_source_refs_array():
    """Test Akashic-style source_refs argument works for fetch_messages."""
    mock_session_store = MagicMock()
    mock_session_store.fetch_messages.side_effect = [
        (
            [
                {
                    "role": "user",
                    "content": "我喜欢拿铁。",
                    "seq": 0,
                    "source_ref": "session:1:1#msg:0",
                    "in_source_ref": True,
                }
            ],
            1,
        ),
        (
            [
                {
                    "role": "user",
                    "content": "后来我改喝茶。",
                    "seq": 2,
                    "source_ref": "session:1:1#msg:2",
                    "in_source_ref": True,
                }
            ],
            1,
        ),
    ]

    with patch("agent.pipeline.reasoner.AsyncOpenAI"):
        reasoner = Reasoner(session_store=mock_session_store)

    raw = await reasoner._fetch_messages(
        {
            "source_refs": ["session:1:1#msg:0", "session:1:1#msg:2"],
            "context": 1,
            "limit": 10,
        }
    )
    payload = json.loads(raw)

    assert payload["matched_count"] == 2
    assert payload["source_refs"] == ["session:1:1#msg:0", "session:1:1#msg:2"]
    assert [m["content"] for m in payload["messages"]] == [
        "我喜欢拿铁。",
        "后来我改喝茶。",
    ]
    assert mock_session_store.fetch_messages.call_count == 2
    print("test_fetch_messages_accepts_source_refs_array: PASS")


async def test_search_messages_tool():
    """Test search_messages uses current session user and returns source refs."""
    mock_session_store = MagicMock()
    mock_session_store.search_messages.return_value = (
        [
            {
                "role": "user",
                "content": "我是个程序员，最常用 Python",
                "seq": 0,
                "source_ref": "session:1:1#msg:0",
            }
        ],
        1,
    )

    with patch("agent.pipeline.reasoner.AsyncOpenAI") as MockClient:
        mock_client = AsyncMock()

        response1 = MagicMock()
        response1.choices = [_mock_choice(
            "",
            tool_calls=[_mock_tool_call(
                "call_1", "search_messages",
                '{"query":"Python 程序员","role":"user","limit":5}'
            )],
            finish_reason="tool_calls",
        )]

        response2 = MagicMock()
        response2.choices = [_mock_choice("你说过自己是程序员，常用 Python。")]

        mock_client.chat.completions.create = AsyncMock(
            side_effect=[response1, response2]
        )
        MockClient.return_value = mock_client

        reasoner = Reasoner(session_store=mock_session_store)
        ctx = _make_ctx(messages=[{"role": "user", "content": "我用什么语言？"}])

        result = await reasoner.run_turn(ctx)

        assert "Python" in result.content
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["function"]["name"] == "search_messages"
        mock_session_store.search_messages.assert_called_once()
        assert mock_session_store.search_messages.call_args.kwargs["user_id"] == 1
        assert mock_session_store.search_messages.call_args.kwargs["role"] == "user"
        print("test_search_messages_tool: PASS")


async def test_recall_memory_grep_mode_with_time_filter():
    """Test recall_memory grep mode lists time-filtered memories."""
    mock_store = MagicMock()
    mock_embedder = AsyncMock()
    from uuid import uuid4
    mock_store.list_memories.return_value = [
        MemoryItem(
            id=uuid4(),
            user_id=1,
            memory_type="event",
            summary="用户今天聊了 Rust meetup。",
            embedding=None,
            status="active",
            source_ref="session:1:1#msg:0",
        )
    ]

    with patch("agent.pipeline.reasoner.AsyncOpenAI") as MockClient:
        mock_client = AsyncMock()
        response1 = MagicMock()
        response1.choices = [_mock_choice(
            "",
            tool_calls=[_mock_tool_call(
                "call_1", "recall_memory",
                '{"query":"用户今天做了什么","search_mode":"grep","time_filter":"today","limit":20}'
            )],
            finish_reason="tool_calls",
        )]
        response2 = MagicMock()
        response2.choices = [_mock_choice("你今天聊了 Rust meetup。")]
        mock_client.chat.completions.create = AsyncMock(side_effect=[response1, response2])
        MockClient.return_value = mock_client

        reasoner = Reasoner(store=mock_store, embedder=mock_embedder)
        ctx = _make_ctx(messages=[{"role": "user", "content": "今天聊了什么？"}])

        result = await reasoner.run_turn(ctx)

        assert "Rust" in result.content
        mock_store.list_memories.assert_called_once()
        assert mock_store.list_memories.call_args.kwargs["memory_types"] == ["event"]
        assert mock_store.list_memories.call_args.kwargs["user_id"] == 1
        print("test_recall_memory_grep_mode_with_time_filter: PASS")


async def test_recall_memory_infers_profile_type():
    """When memory_type is omitted, profile-shaped queries should not search all types."""
    mock_store = AsyncMock()
    mock_embedder = AsyncMock()
    mock_embedder.embed = AsyncMock(return_value=[0.1] * 256)
    mock_store.vector_search = AsyncMock(return_value=[])
    mock_store.keyword_search = AsyncMock(return_value=[])

    with patch("agent.pipeline.reasoner.AsyncOpenAI"):
        reasoner = Reasoner(store=mock_store, embedder=mock_embedder)

    raw = await reasoner._recall_memory(
        {"query": "用户的职业和常用编程语言技术栈"},
        _make_ctx(),
    )
    payload = json.loads(raw)

    assert payload["applied_memory_types"] == ["profile"]
    assert mock_store.vector_search.call_args.kwargs["memory_types"] == ["profile"]
    assert mock_store.keyword_search.call_args.kwargs["memory_types"] == ["profile"]
    print("test_recall_memory_infers_profile_type: PASS")


async def test_single_tool_call():
    """Test single memorize tool call. 需要 store + embedder mock。"""
    mock_store = AsyncMock()
    mock_embedder = AsyncMock()
    mock_embedder.embed = AsyncMock(return_value=[0.1] * 256)

    from uuid import uuid4
    mock_item = MemoryItem(
        id=uuid4(),
        user_id=1,
        memory_type="preference",
        summary="用户喜欢吃苹果",
        embedding=[0.1] * 256,
        status="active",
        source_ref="session:1:1",
    )
    mock_store.upsert_item = AsyncMock(return_value=mock_item)

    with patch("agent.pipeline.reasoner.AsyncOpenAI") as MockClient:
        mock_client = AsyncMock()

        # First response: memorize tool call
        response1 = MagicMock()
        response1.choices = [_mock_choice(
            "",
            tool_calls=[_mock_tool_call(
                "call_1", "memorize",
                '{"summary":"用户喜欢吃苹果","memory_type":"preference"}'
            )],
            finish_reason="tool_calls"
        )]

        # Second response: final answer
        response2 = MagicMock()
        response2.choices = [_mock_choice("好的，我已经记住了你喜欢苹果。")]

        mock_client.chat.completions.create = AsyncMock(
            side_effect=[response1, response2]
        )
        MockClient.return_value = mock_client

        reasoner = Reasoner(store=mock_store, embedder=mock_embedder)
        ctx = _make_ctx(messages=[{"role": "user", "content": "记住我喜欢吃苹果"}])

        result = await reasoner.run_turn(ctx)

        assert result.content == "好的，我已经记住了你喜欢苹果。"
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["function"]["name"] == "memorize"
        assert result.finish_reason == "stop"

        # Verify upsert_item was actually called
        mock_store.upsert_item.assert_called_once()
        call_kwargs = mock_store.upsert_item.call_args.kwargs
        assert call_kwargs["memory_type"] == "preference"
        assert call_kwargs["summary"] == "用户喜欢吃苹果"
        assert call_kwargs["user_id"] == 1
        assert call_kwargs["source_ref"] == "session:1:1"

        print("test_single_tool_call: PASS")


async def test_multiple_tool_calls():
    """Test multiple tool calls in one response."""
    mock_store = AsyncMock()
    mock_embedder = AsyncMock()
    mock_embedder.embed = AsyncMock(return_value=[0.1] * 256)

    from uuid import uuid4
    mock_store.upsert_item = AsyncMock(side_effect=[
        MemoryItem(
            id=uuid4(), user_id=1, memory_type="preference",
            summary="用户喜欢吃苹果", embedding=[0.1]*256,
            status="active", source_ref="session:1:1",
        ),
        MemoryItem(
            id=uuid4(), user_id=1, memory_type="profile",
            summary="用户名字叫张三", embedding=[0.1]*256,
            status="active", source_ref="session:1:1",
        ),
    ])

    with patch("agent.pipeline.reasoner.AsyncOpenAI") as MockClient:
        mock_client = AsyncMock()

        response1 = MagicMock()
        response1.choices = [_mock_choice(
            "",
            tool_calls=[
                _mock_tool_call(
                    "call_1", "memorize",
                    '{"summary":"用户喜欢吃苹果","memory_type":"preference"}'
                ),
                _mock_tool_call(
                    "call_2", "memorize",
                    '{"summary":"用户名字叫张三","memory_type":"profile"}'
                ),
            ],
            finish_reason="tool_calls"
        )]

        response2 = MagicMock()
        response2.choices = [_mock_choice("我已经记住了你的偏好和名字。")]

        mock_client.chat.completions.create = AsyncMock(
            side_effect=[response1, response2]
        )
        MockClient.return_value = mock_client

        reasoner = Reasoner(store=mock_store, embedder=mock_embedder)
        ctx = _make_ctx(messages=[{"role": "user", "content": "记住我的信息"}])

        result = await reasoner.run_turn(ctx)

        assert "记住了" in result.content
        assert len(result.tool_calls) == 2
        assert mock_store.upsert_item.call_count == 2
        print("test_multiple_tool_calls: PASS")


async def test_max_iterations():
    """Test max iterations limit."""
    mock_store = AsyncMock()
    mock_embedder = AsyncMock()
    mock_embedder.embed = AsyncMock(return_value=[0.1] * 256)

    from uuid import uuid4
    mock_store.upsert_item = AsyncMock(return_value=MemoryItem(
        id=uuid4(), user_id=1, memory_type="profile",
        summary="test", embedding=[0.1]*256,
        status="active", source_ref="session:1:1",
    ))

    with patch("agent.pipeline.reasoner.AsyncOpenAI") as MockClient:
        mock_client = AsyncMock()

        # Keep returning tool calls to exceed max iterations
        tool_response = MagicMock()
        tool_response.choices = [_mock_choice(
            "",
            tool_calls=[_mock_tool_call(
                "call_x", "memorize",
                '{"summary":"test","memory_type":"profile"}'
            )],
            finish_reason="tool_calls"
        )]

        mock_client.chat.completions.create = AsyncMock(return_value=tool_response)
        MockClient.return_value = mock_client

        reasoner = Reasoner(store=mock_store, embedder=mock_embedder)
        ctx = _make_ctx(messages=[{"role": "user", "content": "test"}])

        result = await reasoner.run_turn(ctx)

        assert result.finish_reason == "max_iterations"
        assert "处理请求时遇到问题" in result.content
        assert mock_client.chat.completions.create.call_count == 3
        print("test_max_iterations: PASS")


async def test_api_retry():
    """Test API error retry."""
    with patch("agent.pipeline.reasoner.AsyncOpenAI") as MockClient:
        mock_client = AsyncMock()

        # First call fails, second succeeds
        response = MagicMock()
        response.choices = [_mock_choice("重试成功")]

        mock_client.chat.completions.create = AsyncMock(
            side_effect=[Exception("API error"), response]
        )
        MockClient.return_value = mock_client

        reasoner = Reasoner()
        ctx = _make_ctx(messages=[{"role": "user", "content": "test"}])

        result = await reasoner.run_turn(ctx)

        assert result.content == "重试成功"
        assert mock_client.chat.completions.create.call_count == 2
        print("test_api_retry: PASS")


async def main():
    await test_simple_response()
    await test_recall_memory_tool()
    await test_recall_then_fetch_messages_tool_chain()
    await test_fetch_messages_accepts_source_refs_array()
    await test_search_messages_tool()
    await test_recall_memory_grep_mode_with_time_filter()
    await test_recall_memory_infers_profile_type()
    await test_single_tool_call()
    await test_multiple_tool_calls()
    await test_max_iterations()
    await test_api_retry()
    print("\nAll reasoner tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
