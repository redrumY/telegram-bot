import time
from typing import Any

from agent.core.types import (
    AfterReasoningCtx,
    MemoryItem,
    OutboundMessage,
    TurnCommittedEvent,
)
from agent.core.event_bus import EventBus
from agent.pipeline.phases.after_reasoning import AfterReasoningPhase
from channels.telegram.adapter import TelegramAdapter
from uuid import uuid4, UUID


class AfterTurnPhase:
    def __init__(
        self,
        event_bus: EventBus,
        telegram_adapter: TelegramAdapter,
    ) -> None:
        self.event_bus = event_bus
        self.telegram_adapter = telegram_adapter

    async def execute(
        self,
        ctx: AfterReasoningCtx,
        user_id: int,
        new_memory_ids: list[UUID],
        inbound_content: str = "",
    ) -> None:
        """Execute post-turn operations."""
        # Generate turn ID
        turn_id = str(uuid4())

        # Create and emit TurnCommittedEvent
        event = TurnCommittedEvent(
            turn_id=turn_id,
            user_id=user_id,
            inbound_content=inbound_content,
            outbound_message=ctx.outbound_message,
            new_memory_ids=new_memory_ids,
        )

        await self.event_bus.emit("turn_committed", event=event)

        # Send message via Telegram (skip if no adapter, e.g. during testing)
        if self.telegram_adapter is not None:
            await self.telegram_adapter.send(ctx.outbound_message)
