"""Pipeline for processing a single turn of conversation."""

import asyncio

from agent.core.types import InboundMessage, OutboundMessage
from agent.pipeline.consolidation_worker import ConsolidationWorker
from agent.pipeline.invalidation_worker import InvalidationWorker
from agent.pipeline.phases.after_reasoning import AfterReasoningPhase
from agent.pipeline.phases.after_turn import AfterTurnPhase
from agent.pipeline.phases.before_reasoning import BeforeReasoningPhase
from agent.pipeline.phases.before_turn import BeforeTurnPhase
from agent.pipeline.reasoner import Reasoner
from memory.store import MemoryStore
from uuid import UUID, uuid4


class PassiveTurnPipeline:
    """Pipeline for processing a single turn of conversation."""

    def __init__(
        self,
        before_turn: BeforeTurnPhase,
        before_reasoning: BeforeReasoningPhase,
        reasoner: Reasoner,
        after_reasoning: AfterReasoningPhase,
        after_turn: AfterTurnPhase,
        store: MemoryStore | None = None,
        consolidation_worker: ConsolidationWorker | None = None,
        invalidation_worker: InvalidationWorker | None = None,
    ) -> None:
        self.before_turn = before_turn
        self.before_reasoning = before_reasoning
        self.reasoner = reasoner
        self.after_reasoning = after_reasoning
        self.after_turn = after_turn
        self._store = store
        self._consolidation = consolidation_worker
        self._invalidation = invalidation_worker
        self._consolidation_inflight: set[tuple[int, int]] = set()

    async def execute(self, inbound_message: InboundMessage) -> OutboundMessage:
        """Execute the full pipeline for a single turn."""
        # Phase 1: BeforeTurn - acquire session and retrieve memories
        turn_ctx = await self.before_turn.build_ctx(inbound_message)

        # Phase 2: BeforeReasoning - prepare messages and tools for LLM
        reasoning_ctx = await self.before_reasoning.build_ctx(turn_ctx)

        # Phase 3: Reasoner - call LLM and handle tool calls
        result = await self.reasoner.run_turn(reasoning_ctx)
        self.last_reasoner_result = result

        # Phase 4: AfterReasoning - create outbound message and persist
        after_ctx = await self.after_reasoning.build_ctx(
            result=result,
            session=turn_ctx.session,
            chat_id=inbound_message.chat_id,
            user_id=inbound_message.user_id,
        )

        # Persist messages（对应 akashic PostResponseWorker：异步，不阻塞回复）
        asyncio.create_task(
            self.after_reasoning.persist_messages(
                session=turn_ctx.session,
                user_message=inbound_message.content,
                assistant_message=result.content,
                user_id=inbound_message.user_id,
                chat_id=inbound_message.chat_id,
            )
        )

        # Phase 5: AfterTurn - emit event and send message
        new_memory_ids = []  # persist 异步，此处不再等待
        await self.after_turn.execute(
            ctx=after_ctx,
            user_id=inbound_message.user_id,
            new_memory_ids=new_memory_ids,
            inbound_content=inbound_message.content,
        )

        # Update session with new messages
        turn_ctx.session.messages.append({
            "role": "user",
            "content": inbound_message.content,
        })
        turn_ctx.session.messages.append({
            "role": "assistant",
            "content": result.content,
        })

        # Persist session（对应 akashic sm.save(session)）
        from persistence.session_store import get_session_store
        get_session_store().save(
            inbound_message.user_id,
            inbound_message.chat_id,
            turn_ctx.session.messages,
        )

        # ── 窗口期 consolidation（对应 akashic on_turn_committed → _enqueue_maintenance）──
        self._maybe_consolidate(turn_ctx.session, inbound_message)
        self._maybe_invalidate(inbound_message, result)

        return after_ctx.outbound_message

    def _maybe_invalidate(self, inbound_message: InboundMessage, result) -> None:
        """Run akashic-style post-response invalidation asynchronously."""
        if self._invalidation is None:
            return

        async def _run():
            try:
                await self._invalidation.run(
                    user_msg=inbound_message.content,
                    agent_response=result.content,
                    tool_calls=result.tool_calls,
                    user_id=inbound_message.user_id,
                    chat_id=inbound_message.chat_id,
                    source_ref=f"session:{inbound_message.user_id}:{inbound_message.chat_id}",
                )
            except Exception:
                import logging
                logging.getLogger(__name__).exception(
                    "Invalidation failed user=%d chat=%d",
                    inbound_message.user_id,
                    inbound_message.chat_id,
                )

        asyncio.create_task(_run())

    def _maybe_consolidate(
        self,
        session,
        inbound_message: InboundMessage,
    ) -> None:
        """
        对齐 akashic on_turn_committed → _enqueue_maintenance：
          每轮对话后异步检查是否攒够新消息，触发 LLM 提取长期记忆。

        fire-and-forget，不阻塞用户回复。
        """
        if self._consolidation is None or self._store is None:
            return

        if not self._consolidation.should_consolidate(session):
            return

        user_id = inbound_message.user_id
        chat_id = inbound_message.chat_id
        session_key = (user_id, chat_id)
        if session_key in self._consolidation_inflight:
            return
        self._consolidation_inflight.add(session_key)

        async def _run():
            try:
                await self._consolidation.consolidate(
                    session=session,
                    store=self._store,
                    user_id=user_id,
                    chat_id=chat_id,
                )
            except Exception:
                import logging
                logging.getLogger(__name__).exception(
                    "Consolidation failed user=%d chat=%d", user_id, chat_id,
                )
            finally:
                self._consolidation_inflight.discard(session_key)

        asyncio.create_task(_run())
