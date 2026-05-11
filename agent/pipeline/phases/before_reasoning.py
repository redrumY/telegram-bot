from agent.core.types import (
    BeforeReasoningCtx,
    BeforeTurnCtx,
)


# Tool definition schema (OpenAI format)
_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "memorize",
            "description": "保存重要信息到长期记忆中",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "需要保存的内容",
                    },
                    "memory_type": {
                        "type": "string",
                        "enum": ["fact", "preference", "event"],
                        "description": "记忆类型",
                    },
                },
                "required": ["content", "memory_type"],
            },
        },
    }
]


class BeforeReasoningPhase:
    """Prepare context for LLM reasoning."""

    async def preheat(self) -> None:
        """Preheat resources (no-op for now)."""
        pass

    def _build_system_prompt(self, memories: list) -> str:
        """Build system prompt with memory context."""
        if not memories:
            return "你是一个友好的 AI 助手。"

        memory_summaries = "\n".join(
            f"- {m.summary}" for m in memories[:5]  # Limit to top 5
        )
        return f"""你是一个友好的 AI 助手。

以下是与用户相关的记忆：
{memory_summaries}

请根据这些记忆提供个性化的回答。"""

    async def build_ctx(self, turn_ctx: BeforeTurnCtx) -> BeforeReasoningCtx:
        """Build BeforeReasoningCtx for LLM reasoning."""
        # 1. Build system prompt with memories
        system_prompt = self._build_system_prompt(turn_ctx.retrieved_memories)

        # 2. Prepare messages in OpenAI format
        messages = [
            {"role": "system", "content": system_prompt},
        ]

        # 3. Add session history
        for msg in turn_ctx.session.messages:
            messages.append({
                "role": msg["role"],
                "content": msg["content"],
            })

        # 4. Add current user message
        messages.append({
            "role": "user",
            "content": turn_ctx.inbound_message.content,
        })

        return BeforeReasoningCtx(
            session=turn_ctx.session,
            memories=turn_ctx.retrieved_memories,
            messages=messages,
            tools=_TOOLS,
        )
