import asyncio
import logging
import os

from agent.core.event_bus import EventBus
from agent.pipeline.passive_turn import PassiveTurnPipeline
from agent.pipeline.phases.after_reasoning import AfterReasoningPhase
from agent.pipeline.phases.after_turn import AfterTurnPhase
from agent.pipeline.phases.before_reasoning import BeforeReasoningPhase
from agent.pipeline.phases.before_turn import BeforeTurnPhase
from agent.pipeline.reasoner import Reasoner
from channels.telegram.adapter import TelegramAdapter
from config.settings import settings
from evaluation.conversation_logger import ConversationLogger
from memory.embedder import Embedder
from memory.store import MemoryStore
from persistence.database import init_db
from persistence.session_store import get_session_store

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def main() -> None:
    """Main entry point."""
    logger.info("Bot starting...")

    # 1. Initialize database
    logger.info("Initializing database...")
    init_db()

    # 2. Initialize core components
    embedder = Embedder()
    memory_store = MemoryStore(embedder)
    event_bus = EventBus.get_instance()

    # 3. Initialize conversation logger (for evaluation)
    conversation_logger = ConversationLogger()
    await conversation_logger.start()
    logger.info("Conversation logger started")

    # 3. Initialize pipeline phases
    before_turn = BeforeTurnPhase(embedder, memory_store)
    before_reasoning = BeforeReasoningPhase()
    await before_reasoning.preheat()

    reasoner = Reasoner(store=memory_store, embedder=embedder, session_store=get_session_store())
    after_reasoning = AfterReasoningPhase(memory_store)
    after_turn = AfterTurnPhase(event_bus, None)  # adapter set later

    # 4. Consolidation worker（窗口期 LLM 提取长期记忆）
    from agent.pipeline.consolidation_worker import ConsolidationWorker
    from agent.pipeline.invalidation_worker import InvalidationWorker
    consolidation = ConsolidationWorker(keep_count=10, min_new_messages=6)
    invalidation = InvalidationWorker(memory_store, embedder)

    # 5. Create pipeline
    pipeline = PassiveTurnPipeline(
        before_turn=before_turn,
        before_reasoning=before_reasoning,
        reasoner=reasoner,
        after_reasoning=after_reasoning,
        after_turn=after_turn,
        store=memory_store,
        consolidation_worker=consolidation,
        invalidation_worker=invalidation,
    )

    # 5. Create Telegram adapter
    adapter = TelegramAdapter(
        token=settings.TG_BOT_TOKEN,
        pipeline=pipeline,
        proxy=settings.HTTP_PROXY,
    )
    after_turn.telegram_adapter = adapter  # Inject adapter

    # 6. Start bot
    logger.info("Starting Telegram bot...")
    await adapter.start()

    # Get bot info after starting
    me = await adapter.application.bot.get_me()
    logger.info(f"Bot started as @{me.username}")

    # Keep running
    try:
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        pass
    finally:
        # Stop conversation logger
        await conversation_logger.stop()
        logger.info("Conversation logger stopped")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
