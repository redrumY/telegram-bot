"""Application service facade for chat-oriented entry points."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agent.core.types import InboundMessage

if TYPE_CHECKING:
    from agent.pipeline.passive_turn import PassiveTurnPipeline


@dataclass(frozen=True)
class ChatResult:
    user_id: int
    session_id: int
    answer: str
    metadata: dict[str, Any] = field(default_factory=dict)


class SessionLockManager:
    """In-process per-session locks for local and single-worker deployments."""

    def __init__(self) -> None:
        self._locks: dict[tuple[int, int], asyncio.Lock] = {}
        self._guard = asyncio.Lock()

    async def get(self, user_id: int, session_id: int) -> asyncio.Lock:
        key = (user_id, session_id)
        async with self._guard:
            lock = self._locks.get(key)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[key] = lock
            return lock


class AgentService:
    """Small API-facing facade around the existing PassiveTurnPipeline."""

    def __init__(
        self,
        pipeline: "PassiveTurnPipeline",
        *,
        locks: SessionLockManager | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.locks = locks or SessionLockManager()

    async def chat(
        self,
        *,
        user_id: int,
        session_id: int,
        content: str,
        channel: str = "web",
        metadata: dict[str, Any] | None = None,
    ) -> ChatResult:
        """Run one user message through the agent, serializing per session."""
        lock = await self.locks.get(user_id, session_id)
        async with lock:
            inbound = InboundMessage(
                user_id=user_id,
                chat_id=session_id,
                content=content,
                metadata={
                    "channel": channel,
                    **(metadata or {}),
                },
            )
            outbound = await self.pipeline.execute(inbound)
            return ChatResult(
                user_id=user_id,
                session_id=session_id,
                answer=outbound.content,
                metadata={"format": outbound.format},
            )
