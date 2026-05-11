import asyncio
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

from agent.core.types import BeforeReasoningCtx, Session
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


async def test_simple_response():
    """Test simple response without tool calls."""
    with patch("agent.pipeline.reasoner.AsyncOpenAI") as MockClient:
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [_mock_choice("你好！我是你的助手。")]
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        MockClient.return_value = mock_client

        reasoner = Reasoner()
        ctx = BeforeReasoningCtx(
            session=Session(user_id=1, chat_id=1, messages=[]),
            memories=[],
            messages=[{"role": "user", "content": "你好"}],
            tools=[],
        )

        result = await reasoner.run_turn(ctx)

        assert result.content == "你好！我是你的助手。"
        assert result.tool_calls == []
        assert result.finish_reason == "stop"
        print("test_simple_response: PASS")


async def test_single_tool_call():
    """Test single tool call and final response."""
    with patch("agent.pipeline.reasoner.AsyncOpenAI") as MockClient:
        mock_client = AsyncMock()

        # First response: tool call
        response1 = MagicMock()
        response1.choices = [_mock_choice(
            "",
            tool_calls=[_mock_tool_call("call_1", "memorize", '{"content":"用户喜欢吃苹果","memory_type":"preference"}')],
            finish_reason="tool_calls"
        )]

        # Second response: final answer after tool execution
        response2 = MagicMock()
        response2.choices = [_mock_choice("好的，我已经记住了你喜欢苹果。")]

        mock_client.chat.completions.create = AsyncMock(side_effect=[response1, response2])
        MockClient.return_value = mock_client

        reasoner = Reasoner()
        ctx = BeforeReasoningCtx(
            session=Session(user_id=1, chat_id=1, messages=[]),
            memories=[],
            messages=[{"role": "user", "content": "记住我喜欢吃苹果"}],
            tools=[],
        )

        result = await reasoner.run_turn(ctx)

        assert result.content == "好的，我已经记住了你喜欢苹果。"
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["function"]["name"] == "memorize"
        assert result.finish_reason == "stop"

        # Verify create was called twice (initial + after tool execution)
        assert mock_client.chat.completions.create.call_count == 2
        print("test_single_tool_call: PASS")


async def test_multiple_tool_calls():
    """Test multiple tool calls in one response."""
    with patch("agent.pipeline.reasoner.AsyncOpenAI") as MockClient:
        mock_client = AsyncMock()

        # First response: two tool calls
        response1 = MagicMock()
        response1.choices = [_mock_choice(
            "",
            tool_calls=[
                _mock_tool_call("call_1", "memorize", '{"content":"用户喜欢吃苹果","memory_type":"preference"}'),
                _mock_tool_call("call_2", "memorize", '{"content":"用户名字叫张三","memory_type":"fact"}'),
            ],
            finish_reason="tool_calls"
        )]

        # Second response: final answer
        response2 = MagicMock()
        response2.choices = [_mock_choice("我已经记住了你的偏好和名字。")]

        mock_client.chat.completions.create = AsyncMock(side_effect=[response1, response2])
        MockClient.return_value = mock_client

        reasoner = Reasoner()
        ctx = BeforeReasoningCtx(
            session=Session(user_id=1, chat_id=1, messages=[]),
            memories=[],
            messages=[{"role": "user", "content": "记住我的信息"}],
            tools=[],
        )

        result = await reasoner.run_turn(ctx)

        assert "记住了" in result.content
        assert len(result.tool_calls) == 2
        print("test_multiple_tool_calls: PASS")


async def test_max_iterations():
    """Test max iterations limit."""
    with patch("agent.pipeline.reasoner.AsyncOpenAI") as MockClient:
        mock_client = AsyncMock()

        # Keep returning tool calls to exceed max iterations
        tool_response = MagicMock()
        tool_response.choices = [_mock_choice(
            "",
            tool_calls=[_mock_tool_call("call_x", "memorize", '{"content":"test"}')],
            finish_reason="tool_calls"
        )]

        mock_client.chat.completions.create = AsyncMock(return_value=tool_response)
        MockClient.return_value = mock_client

        reasoner = Reasoner()
        ctx = BeforeReasoningCtx(
            session=Session(user_id=1, chat_id=1, messages=[]),
            memories=[],
            messages=[{"role": "user", "content": "test"}],
            tools=[],
        )

        result = await reasoner.run_turn(ctx)

        assert result.finish_reason == "max_iterations"
        assert "处理请求时遇到问题" in result.content
        # Should have attempted 3 iterations
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
        ctx = BeforeReasoningCtx(
            session=Session(user_id=1, chat_id=1, messages=[]),
            memories=[],
            messages=[{"role": "user", "content": "test"}],
            tools=[],
        )

        result = await reasoner.run_turn(ctx)

        assert result.content == "重试成功"
        assert mock_client.chat.completions.create.call_count == 2
        print("test_api_retry: PASS")


async def main():
    await test_simple_response()
    await test_single_tool_call()
    await test_multiple_tool_calls()
    await test_max_iterations()
    await test_api_retry()
    print("\nAll reasoner tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
