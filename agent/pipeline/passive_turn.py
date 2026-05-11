from agent.core.types import InboundMessage, OutboundMessage
from agent.pipeline.phases.after_reasoning import AfterReasoningPhase
from agent.pipeline.phases.after_turn import AfterTurnPhase
from agent.pipeline.phases.before_reasoning import BeforeReasoningPhase
from agent.pipeline.phases.before_turn import BeforeTurnPhase
from agent.pipeline.reasoner import Reasoner
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
    ) -> None:
        self.before_turn = before_turn
        self.before_reasoning = before_reasoning
        self.reasoner = reasoner
        self.after_reasoning = after_reasoning
        self.after_turn = after_turn

    async def execute(self, inbound_message: InboundMessage) -> OutboundMessage:
        """Execute the full pipeline for a single turn."""
        # Phase 1: BeforeTurn - acquire session and retrieve memories
        turn_ctx = await self.before_turn.build_ctx(inbound_message)

        # Phase 2: BeforeReasoning - prepare messages and tools for LLM
        reasoning_ctx = await self.before_reasoning.build_ctx(turn_ctx)

        # Phase 3: Reasoner - call LLM and handle tool calls
        result = await self.reasoner.run_turn(reasoning_ctx)

        # Phase 4: AfterReasoning - create outbound message and persist
        after_ctx = await self.after_reasoning.build_ctx(
            result=result,
            session=turn_ctx.session,
            chat_id=inbound_message.chat_id,
            user_id=inbound_message.user_id,
        )

        # Persist messages
        new_memories = await self.after_reasoning.persist_messages(
            session=turn_ctx.session,
            user_message=inbound_message.content,
            assistant_message=result.content,
            user_id=inbound_message.user_id,
        )

        # Phase 5: AfterTurn - emit event and send message
        new_memory_ids = [m.id for m in new_memories]
        await self.after_turn.execute(
            ctx=after_ctx,
            user_id=inbound_message.user_id,
            new_memory_ids=new_memory_ids,
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

        return after_ctx.outbound_message
