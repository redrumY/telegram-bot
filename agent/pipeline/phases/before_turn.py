from collections.abc import Sequence

from agent.core.event_bus import EventBus
from agent.core.types import BeforeTurnCtx, InboundMessage, MemoryItem, Session
from agent.lifecycle.phase import (
    PhaseFrame,
    PhaseModuleRunner,
    append_string_exports,
    collect_prefixed_slots,
)
from memory.engine import MemoryRetrieveRequest, MemoryScope
from memory.store import LONG_TERM_MEMORY_TYPES
from persistence.session_store import get_session_store

# 内存缓存（对应 akashic sm._cache），SessionStore 负责持久化
_sessions: dict[tuple[int, int], Session] = {}

# RRF 融合参数
_SESSION_SLOT = "session:session"
_CTX_SLOT = "session:ctx"
_EXTRA_HINT_PREFIX = "session:extra_hint:"
_ABORT_REPLY_SLOT = "session:abort_reply"


class BeforeTurnPhase:
    """检索阶段：加载会话 + RRF 融合检索 + 构建上下文"""

    def __init__(
        self,
        *,
        memory_engine: object,
        event_bus: EventBus | None = None,
        plugin_modules: Sequence[object] | None = None,
    ) -> None:
        self.event_bus = event_bus or EventBus.get_instance()
        self.plugin_modules = list(plugin_modules or [])
        self.memory_engine = memory_engine
        self.last_retrieved: list[MemoryItem] = []
        self.last_query_text = ""
        self.last_retrieved_memory_block = ""
        self.last_retrieval_trace: dict = {}

    async def acquire_session(self, message: InboundMessage) -> Session:
        key = (message.user_id, message.chat_id)
        session = _sessions.get(key)
        if session is not None:
            return session
        session_store = get_session_store()
        session_state = session_store.load_state(message.user_id, message.chat_id)
        if session_state is None:
            saved_messages = []
            last_consolidated = 0
        else:
            saved_messages, last_consolidated = session_state
        session = Session(
            user_id=message.user_id,
            chat_id=message.chat_id,
            messages=saved_messages,
            last_consolidated=last_consolidated,
        )
        _sessions[key] = session
        return session

    async def prepare_context(
        self, session: Session, query_text: str, user_id: int
    ) -> list[MemoryItem]:
        """Retrieve passive memory context through the shared MemoryEngine."""
        result = await self.memory_engine.retrieve(  # type: ignore[attr-defined]
            MemoryRetrieveRequest(
                query=query_text,
                scope=MemoryScope(
                    user_id=user_id,
                    chat_id=session.chat_id,
                    session_key=f"{session.user_id}:{session.chat_id}",
                ),
                top_k=8,
                memory_types=LONG_TERM_MEMORY_TYPES,
            )
        )
        self.last_query_text = query_text
        self.last_retrieved = list(result.items)
        self.last_retrieved_memory_block = result.text_block
        self.last_retrieval_trace = dict(result.trace or {})
        self._last_hyde_used = bool(result.trace.get("hyde_used"))
        self._last_hypothesis = str(result.trace.get("hypothesis") or "")
        return list(result.items)

    async def build_ctx(self, inbound_message: InboundMessage) -> BeforeTurnCtx:
        session = await self.acquire_session(inbound_message)
        plugin_runner = PhaseModuleRunner(
            self.plugin_modules,
            phase_name="before_turn",
        )
        frame = PhaseFrame(
            input=inbound_message,
            slots={
                _SESSION_SLOT: session,
                "before_turn.acquire_session": True,
            },
        )
        frame = await plugin_runner.run_ready(frame)
        early_ctx = frame.slots.get(_CTX_SLOT)
        if isinstance(early_ctx, BeforeTurnCtx):
            return early_ctx
        early_abort = frame.slots.get(_ABORT_REPLY_SLOT)
        if isinstance(early_abort, str) and early_abort:
            return BeforeTurnCtx(
                inbound_message=inbound_message,
                session=session,
                retrieved_memories=[],
                session_key=f"{inbound_message.user_id}:{inbound_message.chat_id}",
                channel=str(inbound_message.metadata.get("channel") or "telegram"),
                chat_id=str(inbound_message.chat_id),
                content=inbound_message.content,
                history_messages=tuple(session.messages),
                abort=True,
                abort_reply=early_abort,
            )

        user_messages = [
            msg["content"]
            for msg in session.messages[-3:]
            if msg.get("role") == "user"
        ]
        user_messages.append(inbound_message.content)
        query_text = " ".join(user_messages) if user_messages else inbound_message.content
        retrieved_memories = await self.prepare_context(
            session=session, query_text=query_text, user_id=inbound_message.user_id,
        )
        retrieved_memory_block = self.last_retrieved_memory_block
        frame.slots["session:retrieved_memories"] = retrieved_memories
        frame.slots["session:retrieved_memory_block"] = retrieved_memory_block
        frame.slots["session:retrieval_trace_raw"] = dict(self.last_retrieval_trace)
        frame.slots["before_turn.prepare_context"] = True
        frame = await plugin_runner.run_ready(frame)
        ctx = BeforeTurnCtx(
            inbound_message=inbound_message,
            session=session,
            retrieved_memories=retrieved_memories,
            session_key=f"{inbound_message.user_id}:{inbound_message.chat_id}",
            channel=str(inbound_message.metadata.get("channel") or "telegram"),
            chat_id=str(inbound_message.chat_id),
            content=inbound_message.content,
            retrieved_memory_block=retrieved_memory_block,
            retrieval_trace_raw=dict(self.last_retrieval_trace),
            history_messages=tuple(session.messages),
        )
        frame.slots[_CTX_SLOT] = ctx
        frame.slots["before_turn.build_ctx"] = True
        frame = await plugin_runner.run_ready(frame)
        ctx = frame.slots.get(_CTX_SLOT, ctx)
        emitted = await self.event_bus.emit(ctx)
        if emitted is None:
            ctx.abort = True
            if not ctx.abort_reply:
                ctx.abort_reply = "请求已被生命周期处理器阻断。"
            return ctx
        ctx = emitted
        frame.slots[_CTX_SLOT] = ctx
        frame.slots["before_turn.emit"] = True
        frame = await plugin_runner.run_ready(frame)
        ctx = frame.slots.get(_CTX_SLOT, ctx)
        append_string_exports(
            ctx.extra_hints,
            collect_prefixed_slots(frame.slots, _EXTRA_HINT_PREFIX),
        )
        frame.slots["before_turn.collect_exports"] = True
        abort_reply = frame.slots.get(_ABORT_REPLY_SLOT)
        if isinstance(abort_reply, str) and abort_reply:
            ctx.abort = True
            ctx.abort_reply = abort_reply
        frame.slots["before_turn.return"] = True
        plugin_runner.warn_unresolved()
        return ctx
