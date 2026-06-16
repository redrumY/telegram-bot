"""FastAPI routes for the web chat adapter."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agent.service import AgentService

if TYPE_CHECKING:
    from agent.runtime import AgentRuntime
    from agent.service import ChatResult
    from persistence.turn_store import Turn


class HealthResponse(BaseModel):
    status: str = "ok"


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    user_id: int = 1
    session_id: int | None = None
    metadata: dict[str, Any] | None = None


class ChatResponse(BaseModel):
    user_id: int
    session_id: int
    turn_id: str
    status: str
    answer: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MessageListResponse(BaseModel):
    user_id: int
    session_id: int
    messages: list[dict[str, Any]]


def _chat_response(result: "ChatResult") -> ChatResponse:
    return ChatResponse(
        user_id=result.user_id,
        session_id=result.session_id,
        turn_id="inline",
        status="done",
        answer=result.answer,
        metadata=result.metadata,
    )


SessionReader = Callable[[int, int], list[dict[str, Any]]]
TurnCreator = Callable[[int, int, str, dict[str, Any] | None], "Turn"]
TurnReader = Callable[[str], "Turn | None"]


def _turn_response(turn: Any) -> ChatResponse:
    return ChatResponse(
        user_id=turn.user_id,
        session_id=turn.session_id,
        turn_id=turn.id,
        status=turn.status,
        answer=turn.answer,
        error=turn.error,
        metadata=turn.metadata,
    )


def create_app(
    service: AgentService | None = None,
    *,
    session_reader: SessionReader | None = None,
    turn_creator: TurnCreator | None = None,
    turn_reader: TurnReader | None = None,
) -> FastAPI:
    """Create the Web API.

    A service can be injected by tests or alternate launchers. When omitted,
    the app owns an AgentRuntime for the process lifetime.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        runtime: "AgentRuntime | None" = None
        if service is None:
            from agent.runtime import AgentRuntime

            runtime = await AgentRuntime.create()
            app.state.agent_runtime = runtime
            app.state.agent_service = AgentService(runtime.pipeline)
        else:
            app.state.agent_service = service
        try:
            yield
        finally:
            if runtime is not None:
                await runtime.shutdown()

    app = FastAPI(title="Telegram Bot MVP Web API", lifespan=lifespan)
    static_dir = Path(__file__).with_name("static")
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    def read_turn_by_id(turn_id: str) -> Any:
        if turn_reader is None:
            from persistence.turn_store import get_turn_store

            return get_turn_store().get_turn(turn_id)
        return turn_reader(turn_id)

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        index_path = static_dir / "index.html"
        if not index_path.exists():
            raise HTTPException(status_code=404, detail="Web app is not built")
        return FileResponse(index_path)

    @app.get("/api/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse()

    @app.post("/api/chat", response_model=ChatResponse)
    async def chat(payload: ChatRequest) -> ChatResponse:
        session_id = payload.session_id if payload.session_id is not None else payload.user_id
        metadata = {"channel": "web", **(payload.metadata or {})}
        if turn_creator is None:
            from persistence.turn_store import get_turn_store

            turn = get_turn_store().create_turn(
                user_id=payload.user_id,
                session_id=session_id,
                content=payload.message,
                metadata=metadata,
            )
        else:
            turn = turn_creator(payload.user_id, session_id, payload.message, metadata)
        return _turn_response(turn)

    @app.get("/api/turns/{turn_id}", response_model=ChatResponse)
    async def get_turn(turn_id: str) -> ChatResponse:
        turn = read_turn_by_id(turn_id)
        if turn is None:
            raise HTTPException(status_code=404, detail="Turn not found")
        return _turn_response(turn)

    @app.get("/api/turns/{turn_id}/events")
    async def turn_events(turn_id: str) -> StreamingResponse:
        if read_turn_by_id(turn_id) is None:
            raise HTTPException(status_code=404, detail="Turn not found")

        async def event_stream() -> AsyncIterator[str]:
            last_payload = ""
            terminal_statuses = {"done", "failed"}
            while True:
                turn = read_turn_by_id(turn_id)
                if turn is None:
                    payload = {"turn_id": turn_id, "status": "failed", "error": "Turn not found"}
                    yield _sse("failed", payload)
                    return
                response = _turn_response(turn)
                payload = response.model_dump()
                encoded = json.dumps(payload, ensure_ascii=False)
                if encoded != last_payload:
                    yield _sse(str(turn.status), payload)
                    last_payload = encoded
                if str(turn.status) in terminal_statuses:
                    return
                await asyncio.sleep(0.75)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    def _sse(event: str, payload: dict[str, Any]) -> str:
        data = json.dumps(payload, ensure_ascii=False)
        return f"event: {event}\ndata: {data}\n\n"

    @app.get("/api/sessions/{session_id}/messages", response_model=MessageListResponse)
    async def session_messages(
        session_id: int,
        user_id: int = Query(default=1),
    ) -> MessageListResponse:
        if session_reader is None:
            from persistence.session_store import get_session_store

            messages = get_session_store().load(user_id, session_id) or []
        else:
            messages = session_reader(user_id, session_id)
        return MessageListResponse(
            user_id=user_id,
            session_id=session_id,
            messages=messages,
        )

    return app
