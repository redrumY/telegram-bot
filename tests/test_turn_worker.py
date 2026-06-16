import os
from dataclasses import replace

os.environ["TG_BOT_TOKEN"] = "test_token"
os.environ["DEEPSEEK_API_KEY"] = "test_deepseek_key"
os.environ["ALIYUN_DASHSCOPE_API_KEY"] = "test_aliyun_key"

import pytest

from agent.service import ChatResult
from persistence.turn_store import TURN_DONE, TURN_FAILED, TURN_PENDING, TURN_PROCESSING, Turn
from worker.turn_worker import TurnWorker


def _turn(
    turn_id: str,
    *,
    user_id: int = 1,
    session_id: int = 1,
    content: str = "hello",
    status: str = TURN_PENDING,
) -> Turn:
    return Turn(
        id=turn_id,
        user_id=user_id,
        session_id=session_id,
        content=content,
        status=status,
        answer=None,
        error=None,
        metadata={"channel": "web"},
        attempts=0,
        created_at=None,
        updated_at=None,
        started_at=None,
        finished_at=None,
    )


class FakeTurnStore:
    def __init__(self, turns: list[Turn]) -> None:
        self.turns = list(turns)

    def claim_next_pending(self) -> Turn | None:
        processing = {
            (turn.user_id, turn.session_id)
            for turn in self.turns
            if turn.status == TURN_PROCESSING
        }
        for index, turn in enumerate(self.turns):
            if turn.status != TURN_PENDING:
                continue
            if (turn.user_id, turn.session_id) in processing:
                continue
            claimed = replace(turn, status=TURN_PROCESSING, attempts=turn.attempts + 1)
            self.turns[index] = claimed
            return claimed
        return None

    def mark_done(self, turn_id: str, answer: str) -> Turn:
        return self._replace(turn_id, status=TURN_DONE, answer=answer, error=None)

    def mark_failed(self, turn_id: str, error: str) -> Turn:
        return self._replace(turn_id, status=TURN_FAILED, error=error)

    def _replace(self, turn_id: str, **changes) -> Turn:
        for index, turn in enumerate(self.turns):
            if turn.id == turn_id:
                updated = replace(turn, **changes)
                self.turns[index] = updated
                return updated
        raise AssertionError(f"turn not found: {turn_id}")


class FakeService:
    async def chat(self, *, user_id, session_id, content, channel="web", metadata=None):
        return ChatResult(
            user_id=user_id,
            session_id=session_id,
            answer=f"reply:{content}",
            metadata={},
        )


@pytest.mark.asyncio
async def test_turn_worker_completes_claimed_turn() -> None:
    store = FakeTurnStore([_turn("t1")])
    worker = TurnWorker(service=FakeService(), store=store)  # type: ignore[arg-type]

    assert await worker.run_once() is True

    assert store.turns[0].status == TURN_DONE
    assert store.turns[0].answer == "reply:hello"


@pytest.mark.asyncio
async def test_turn_worker_respects_processing_session() -> None:
    store = FakeTurnStore([
        _turn("active", status=TURN_PROCESSING),
        _turn("same-session", session_id=1),
        _turn("other-session", session_id=2, content="parallel"),
    ])
    worker = TurnWorker(service=FakeService(), store=store)  # type: ignore[arg-type]

    assert await worker.run_once() is True

    assert store.turns[1].status == TURN_PENDING
    assert store.turns[2].status == TURN_DONE
    assert store.turns[2].answer == "reply:parallel"
