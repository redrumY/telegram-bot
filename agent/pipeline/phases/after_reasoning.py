from agent.core.types import (
    AfterReasoningCtx,
    MemoryItem,
    OutboundMessage,
    ReasonerResult,
    Session,
)
from agent.pipeline.phases.before_turn import _sessions
from memory.store import MemoryStore
from uuid import uuid4
from datetime import datetime


class AfterReasoningPhase:
    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    async def build_ctx(
        self,
        result: ReasonerResult,
        session: Session,
        chat_id: int,
        user_id: int,
    ) -> AfterReasoningCtx:
        """Build AfterReasoningCtx from ReasonerResult."""
        # Extract content and create OutboundMessage
        content = result.content
        outbound_msg = OutboundMessage(
            chat_id=chat_id,
            content=content,
            format="text",
        )

        # Persist user message and assistant message as memories
        # For now, we'll store them as "event" type memories
        # The actual message content would be passed in, but for now we use what's in session

        return AfterReasoningCtx(
            reasoner_result=result,
            outbound_message=outbound_msg,
        )

    async def persist_messages(
        self,
        session: Session,
        user_message: str,
        assistant_message: str,
        user_id: int,
    ) -> list[MemoryItem]:
        """Persist user and assistant messages to memory."""
        memory_ids = []

        # Persist user message
        user_memory = await self.store.upsert_item(
            memory_type="user_message",
            summary=user_message[:500],  # Truncate if too long
            user_id=user_id,
            source_ref="chat",
        )
        memory_ids.append(user_memory)

        # Persist assistant message
        assistant_memory = await self.store.upsert_item(
            memory_type="assistant_message",
            summary=assistant_message[:500],
            user_id=user_id,
            source_ref="chat",
        )
        memory_ids.append(assistant_memory)

        return memory_ids
