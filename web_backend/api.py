"""FastAPI routes for the web chat adapter."""

from __future__ import annotations

from collections.abc import AsyncIterator
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agent.service import AgentService, ChatResult

if TYPE_CHECKING:
    from agent.runtime import AgentRuntime


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
    answer: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class MessageListResponse(BaseModel):
    user_id: int
    session_id: int
    messages: list[dict[str, Any]]


def _chat_response(result: ChatResult) -> ChatResponse:
    return ChatResponse(
        user_id=result.user_id,
        session_id=result.session_id,
        answer=result.answer,
        metadata=result.metadata,
    )


SessionReader = Callable[[int, int], list[dict[str, Any]]]


def create_app(
    service: AgentService | None = None,
    *,
    session_reader: SessionReader | None = None,
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
        agent_service: AgentService | None = getattr(app.state, "agent_service", None)
        if agent_service is None:
            raise HTTPException(status_code=503, detail="Agent service is not ready")
        session_id = payload.session_id if payload.session_id is not None else payload.user_id
        result = await agent_service.chat(
            user_id=payload.user_id,
            session_id=session_id,
            content=payload.message,
            channel="web",
            metadata=payload.metadata,
        )
        return _chat_response(result)

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
