import asyncio
import logging
from pathlib import Path

from agent.core.event_bus import EventBus
from agent.pipeline.passive_turn import PassiveTurnPipeline
from agent.pipeline.phases.after_reasoning import AfterReasoningPhase
from agent.pipeline.phases.after_turn import AfterTurnPhase
from agent.pipeline.phases.before_reasoning import BeforeReasoningPhase
from agent.pipeline.phases.before_turn import BeforeTurnPhase
from agent.pipeline.reasoner import Reasoner
from agent.plugins import PluginManager
from agent.tool_hooks import ToolExecutor
from agent.tools import ToolRegistry
from agent.tools.memory import register_memory_tools
from channels.telegram.adapter import TelegramAdapter
from config.settings import settings
from evaluation.conversation_logger import ConversationLogger
from memory.embedder import Embedder
from memory.bootstrap import build_memory_runtime
from memory.store import MemoryStore
from persistence.database import init_db
from persistence.session_store import get_session_store

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
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
    session_store = get_session_store()
    memory_runtime = build_memory_runtime(
        embedder=embedder,
        memory_store=memory_store,
        session_store=session_store,
    )
    event_bus = EventBus.get_instance()
    tool_registry = ToolRegistry()
    tool_executor = ToolExecutor()

    # 3. Initialize conversation logger (for evaluation)
    conversation_logger = ConversationLogger()
    await conversation_logger.start()
    logger.info("Conversation logger started")

    # 4. Initialize reasoner + built-in tools before plugin discovery.
    reasoner = Reasoner(
        tool_registry=tool_registry,
        tool_executor=tool_executor,
        event_bus=event_bus,
    )
    register_memory_tools(tool_registry, memory_runtime.engine)

    # 5. Initialize plugin runtime after built-ins, so plugin tools can override by name.
    plugin_manager = PluginManager(
        [Path.cwd() / "plugins"],
        event_bus=event_bus,
        tool_registry=tool_registry,
        workspace=Path.cwd(),
        memory_engine=memory_runtime.engine,
    )
    await plugin_manager.load_all()
    tool_executor.add_hooks(plugin_manager.tool_hooks)
    reasoner.set_step_modules(
        before_step=plugin_manager.before_step_modules,
        after_step=plugin_manager.after_step_modules,
    )

    # 6. Initialize pipeline phases
    before_turn = BeforeTurnPhase(
        event_bus=event_bus,
        plugin_modules=plugin_manager.before_turn_modules,
        memory_engine=memory_runtime.engine,
    )
    before_reasoning = BeforeReasoningPhase(
        tool_registry=tool_registry,
        event_bus=event_bus,
        plugin_modules=plugin_manager.before_reasoning_modules,
        prompt_render_modules=plugin_manager.prompt_render_modules,
        self_model_reader=memory_runtime.markdown.store.read_self,
        long_term_memory_reader=memory_runtime.markdown.store.read_long_term,
        recent_context_reader=memory_runtime.markdown.store.read_recent_context,
    )
    await before_reasoning.preheat()
    after_reasoning = AfterReasoningPhase(
        memory_store,
        event_bus=event_bus,
        plugin_modules=plugin_manager.after_reasoning_modules,
    )
    after_turn = AfterTurnPhase(
        event_bus,
        None,
        plugin_modules=plugin_manager.after_turn_modules,
    )  # adapter set later

    # 7. Consolidation worker（窗口期 LLM 提取长期记忆）
    from agent.pipeline.consolidation_worker import ConsolidationWorker
    from agent.pipeline.invalidation_worker import InvalidationWorker
    consolidation = ConsolidationWorker(
        keep_count=10,
        min_new_messages=6,
        markdown_store=memory_runtime.markdown.store,
    )
    invalidation = InvalidationWorker(memory_store, embedder)

    # 8. Create pipeline
    pipeline = PassiveTurnPipeline(
        before_turn=before_turn,
        before_reasoning=before_reasoning,
        reasoner=reasoner,
        after_reasoning=after_reasoning,
        after_turn=after_turn,
        store=memory_store,
        consolidation_worker=consolidation,
        invalidation_worker=invalidation,
        memory_runtime=memory_runtime,
    )

    # 9. Create Telegram adapter
    adapter = TelegramAdapter(
        token=settings.TG_BOT_TOKEN,
        pipeline=pipeline,
        proxy=settings.HTTP_PROXY,
    )
    after_turn.telegram_adapter = adapter  # Inject adapter

    # 10. Start bot
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
        await plugin_manager.terminate_all()
        await conversation_logger.stop()
        logger.info("Conversation logger stopped")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
