import asyncio

import pytest

from agent.core.types import InboundMessage, OutboundMessage
from agent.service import AgentService


class FakePipeline:
    def __init__(self) -> None:
        self.active_by_session: dict[tuple[int, int], int] = {}
        self.max_active_by_session: dict[tuple[int, int], int] = {}
        self.seen: list[InboundMessage] = []

    async def execute(self, inbound: InboundMessage) -> OutboundMessage:
        key = (inbound.user_id, inbound.chat_id)
        self.active_by_session[key] = self.active_by_session.get(key, 0) + 1
        self.max_active_by_session[key] = max(
            self.max_active_by_session.get(key, 0),
            self.active_by_session[key],
        )
        self.seen.append(inbound)
        await asyncio.sleep(0.01)
        self.active_by_session[key] -= 1
        return OutboundMessage(
            chat_id=inbound.chat_id,
            content=f"echo: {inbound.content}",
        )


@pytest.mark.asyncio
async def test_agent_service_invokes_pipeline_with_web_metadata() -> None:
    pipeline = FakePipeline()
    service = AgentService(pipeline)  # type: ignore[arg-type]

    result = await service.chat(
        user_id=1,
        session_id=2,
        content="hello",
        metadata={"request_id": "req-1"},
    )

    assert result.answer == "echo: hello"
    assert result.user_id == 1
    assert result.session_id == 2
    assert pipeline.seen[0].metadata["channel"] == "web"
    assert pipeline.seen[0].metadata["request_id"] == "req-1"


@pytest.mark.asyncio
async def test_agent_service_serializes_same_session() -> None:
    pipeline = FakePipeline()
    service = AgentService(pipeline)  # type: ignore[arg-type]

    await asyncio.gather(
        service.chat(user_id=1, session_id=2, content="one"),
        service.chat(user_id=1, session_id=2, content="two"),
    )

    assert pipeline.max_active_by_session[(1, 2)] == 1
