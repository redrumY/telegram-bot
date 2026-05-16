from collections import OrderedDict

from agent.core.types import BeforeTurnCtx, InboundMessage, MemoryItem, Session
from memory.embedder import Embedder
from memory.store import LONG_TERM_MEMORY_TYPES, MemoryStore
from persistence.session_store import get_session_store

# 内存缓存（对应 akashic sm._cache），SessionStore 负责持久化
_sessions: dict[tuple[int, int], Session] = {}

# RRF 融合参数
_RRF_K = 60  # Reciprocal Rank Fusion 的 k 值


class BeforeTurnPhase:
    """检索阶段：加载会话 + RRF 融合检索 + 构建上下文"""

    def __init__(self, embedder: Embedder, store: MemoryStore) -> None:
        self.embedder = embedder
        self.store = store
        self.last_retrieved: list[MemoryItem] = []

    async def acquire_session(self, message: InboundMessage) -> Session:
        key = (message.user_id, message.chat_id)
        session = _sessions.get(key)
        if session is not None:
            return session
        session_store = get_session_store()
        saved_messages = session_store.load(message.user_id, message.chat_id)
        session = Session(
            user_id=message.user_id,
            chat_id=message.chat_id,
            messages=saved_messages if saved_messages is not None else [],
        )
        _sessions[key] = session
        return session

    async def prepare_context(
        self, session: Session, query_text: str, user_id: int
    ) -> list[MemoryItem]:
        """HyDE + RRF 融合检索（对齐 akashic injection_planner）

        akashic 做法：
          hyde_enhancer.augment(raw_query, retrieve_fn, top_k)
          → raw 检索 + LLM 生成假想记忆 → 第二次检索 → union dedup

        我们：
          1. HyDE 增强向量检索 → vec_results (raw ∪ hyde)
          2. 关键词检索 → kw_results
          3. RRF 融合：vec_results + kw_results
        """
        from memory.hyde_enhancer import HyDEEnhancer

        # 1. HyDE 增强（对齐 akashic hyde_enhancer.augment 调用模式）
        enhancer = HyDEEnhancer()

        async def _retrieve(query: str):
            query_vec = await self.embedder.embed(query)
            return await self.store.vector_search(
                query_vec=query_vec,
                user_id=user_id,
                top_k=8,
                memory_types=LONG_TERM_MEMORY_TYPES,
            )

        hyde_result = await enhancer.augment(
            raw_query=query_text,
            retrieve_fn=_retrieve,
            top_k=8,
        )
        vec_results = hyde_result.items
        self._last_hyde_used = hyde_result.used_hyde
        self._last_hypothesis = hyde_result.hypothesis

        # 2. 关键词检索
        kw_results = await self.store.keyword_search(
            terms=query_text,
            user_id=user_id,
            limit=8,
            memory_types=LONG_TERM_MEMORY_TYPES,
        )

        # 3. RRF 融合（替代原来的简单拼接）
        fused: dict[str, dict] = {}

        for rank, mem in enumerate(vec_results, 1):
            mid = str(mem.id)
            if mid not in fused:
                fused[mid] = {"mem": mem, "rrf_score": 0.0}
            fused[mid]["rrf_score"] += 1.0 / (_RRF_K + rank)

        for rank, mem in enumerate(kw_results, 1):
            mid = str(mem.id)
            if mid not in fused:
                fused[mid] = {"mem": mem, "rrf_score": 0.0}
            fused[mid]["rrf_score"] += 1.0 / (_RRF_K + rank)

        # 按 RRF 分数降序排列
        sorted_items = sorted(
            fused.values(), key=lambda x: x["rrf_score"], reverse=True
        )

        combined = [item["mem"] for item in sorted_items]
        self.last_retrieved = combined
        return combined

    async def build_ctx(self, inbound_message: InboundMessage) -> BeforeTurnCtx:
        session = await self.acquire_session(inbound_message)
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
        return BeforeTurnCtx(
            inbound_message=inbound_message,
            session=session,
            retrieved_memories=retrieved_memories,
        )
