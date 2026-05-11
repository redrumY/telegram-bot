from collections import OrderedDict

from agent.core.types import BeforeTurnCtx, InboundMessage, MemoryItem, Session
from memory.embedder import Embedder
from memory.store import MemoryStore


# In-memory session storage (temporary, will be replaced with persistence)
_sessions: dict[tuple[int, int], Session] = {}


class BeforeTurnPhase:
    def __init__(self, embedder: Embedder, store: MemoryStore) -> None:
        self.embedder = embedder
        self.store = store

    async def acquire_session(self, message: InboundMessage) -> Session:
        """Get or create a session for the given user and chat."""
        key = (message.user_id, message.chat_id)

        session = _sessions.get(key)
        if session is None:
            session = Session(
                user_id=message.user_id,
                chat_id=message.chat_id,
                messages=[],
            )
            _sessions[key] = session

        return session

    async def prepare_context(
        self, session: Session, query_text: str, user_id: int
    ) -> list[MemoryItem]:
        """Retrieve relevant memories using vector and keyword search."""
        # 1. Get query embedding
        query_vec = await self.embedder.embed(query_text)

        # 2. Vector search for top 5
        vec_results = await self.store.vector_search(
            query_vec=query_vec,
            user_id=user_id,
            top_k=5,
        )

        # 3. Keyword search for top 3
        kw_results = await self.store.keyword_search(
            terms=query_text,
            user_id=user_id,
            limit=3,
        )

        # 4. Deduplicate by memory ID (keeping vector results order)
        seen_ids = set()
        combined: list[MemoryItem] = []

        for mem in vec_results:
            if mem.id not in seen_ids:
                seen_ids.add(mem.id)
                combined.append(mem)

        for mem in kw_results:
            if mem.id not in seen_ids:
                seen_ids.add(mem.id)
                combined.append(mem)

        return combined

    async def build_ctx(self, inbound_message: InboundMessage) -> BeforeTurnCtx:
        """Build the BeforeTurnCtx for pipeline execution."""
        # 1. Get session
        session = await self.acquire_session(inbound_message)

        # 2. Build query text from recent user messages (last 3)
        user_messages = [
            msg["content"]
            for msg in session.messages[-3:]
            if msg.get("role") == "user"
        ]
        user_messages.append(inbound_message.content)
        query_text = " ".join(user_messages) if user_messages else inbound_message.content

        # 3. Retrieve relevant memories
        retrieved_memories = await self.prepare_context(
            session=session,
            query_text=query_text,
            user_id=inbound_message.user_id,
        )

        return BeforeTurnCtx(
            inbound_message=inbound_message,
            session=session,
            retrieved_memories=retrieved_memories,
        )
