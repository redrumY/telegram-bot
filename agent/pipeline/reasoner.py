import asyncio

from openai import AsyncOpenAI

from agent.core.types import BeforeReasoningCtx, ReasonerResult
from config.settings import settings


class Reasoner:
    def __init__(self) -> None:
        self.client = AsyncOpenAI(
            api_key=settings.DEEPSEEK_API_KEY,
            base_url=settings.DEEPSEEK_BASE_URL,
        )
        self.model = settings.LLM_MODEL

    async def _execute_tool(self, tool_name: str, arguments: dict) -> str:
        """Execute a tool call."""
        if tool_name == "memorize":
            # Stub implementation for now
            content = arguments.get("content", "")
            memory_type = arguments.get("memory_type", "fact")
            return f"已保存{memory_type}记忆: {content[:20]}..."
        return f"Unknown tool: {tool_name}"

    async def run_turn(self, ctx: BeforeReasoningCtx) -> ReasonerResult:
        """Run a reasoning turn with potential tool calls."""
        messages = ctx.messages.copy()
        tool_calls: list[dict] = []

        for iteration in range(3):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=ctx.tools if iteration == 0 else [],
                )
            except Exception as e:
                if iteration == 0:
                    await asyncio.sleep(0.5)
                    continue
                raise

            choice = response.choices[0]
            message = choice.message

            # Check for tool calls
            if message.tool_calls:
                tool_calls.extend([
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in message.tool_calls
                ])

                # Add assistant message with tool calls
                messages.append({
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in message.tool_calls
                    ],
                })

                # Execute tools and add responses
                for tc in message.tool_calls:
                    import json
                    args = json.loads(tc.function.arguments)
                    result = await self._execute_tool(tc.function.name, args)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })

                # Continue loop for LLM to respond with final answer
                continue
            else:
                # No tool calls, this is the final response
                return ReasonerResult(
                    content=message.content or "",
                    tool_calls=tool_calls,
                    finish_reason=choice.finish_reason or "stop",
                )

        # Max iterations reached
        return ReasonerResult(
            content="抱歉，处理请求时遇到问题。",
            tool_calls=tool_calls,
            finish_reason="max_iterations",
        )
