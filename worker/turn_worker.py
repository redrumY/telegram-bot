"""Worker loop for consuming pending web chat turns."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from agent.service import AgentService
from persistence.turn_store import TurnStore, get_turn_store

logger = logging.getLogger(__name__)


@dataclass
class WorkerStats:
    claimed: int = 0
    completed: int = 0
    failed: int = 0


class TurnWorker:
    def __init__(
        self,
        *,
        service: AgentService,
        store: TurnStore | None = None,
        poll_interval: float = 0.5,
    ) -> None:
        self.service = service
        self.store = store or get_turn_store()
        self.poll_interval = poll_interval
        self.stats = WorkerStats()

    async def run_once(self) -> bool:
        turn = self.store.claim_next_pending()
        if turn is None:
            return False

        self.stats.claimed += 1
        logger.info(
            "Processing turn id=%s user=%d session=%d",
            turn.id,
            turn.user_id,
            turn.session_id,
        )
        try:
            result = await self.service.chat(
                user_id=turn.user_id,
                session_id=turn.session_id,
                content=turn.content,
                channel=str(turn.metadata.get("channel") or "web"),
                metadata={"turn_id": turn.id, **turn.metadata},
            )
        except Exception as exc:
            self.stats.failed += 1
            logger.exception("Turn failed id=%s", turn.id)
            self.store.mark_failed(turn.id, str(exc))
            return True

        self.store.mark_done(turn.id, result.answer)
        self.stats.completed += 1
        return True

    async def run_forever(self) -> None:
        while True:
            worked = await self.run_once()
            if not worked:
                await asyncio.sleep(self.poll_interval)
