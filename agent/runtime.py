"""Runtime wiring for the reusable agent backend."""

from __future__ import annotations

from pathlib import Path

from agent.core.event_bus import EventBus
from agent.pipeline.consolidation_worker import ConsolidationWorker
from agent.pipeline.invalidation_worker import InvalidationWorker
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
from evaluation.conversation_logger import ConversationLogger
from memory.bootstrap import build_memory_runtime
from memory.embedder import Embedder
from memory.store import MemoryStore
from persistence.database import init_db
from persistence.session_store import get_session_store


class AgentRuntime:
    """Owns the long-lived objects shared by adapters and API routes."""

    def __init__(
        self,
        *,
        pipeline: PassiveTurnPipeline,
        plugin_manager: PluginManager,
        conversation_logger: ConversationLogger | None,
        after_turn: AfterTurnPhase,
    ) -> None:
        self.pipeline = pipeline
        self.plugin_manager = plugin_manager
        self.conversation_logger = conversation_logger
        self.after_turn = after_turn
        self._closed = False

    @classmethod
    async def create(
        cls,
        *,
        workspace: Path | None = None,
        start_conversation_logger: bool = True,
    ) -> "AgentRuntime":
        """Build the agent graph once so any channel can reuse it."""
        workspace = workspace or Path.cwd()
        init_db()

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

        conversation_logger: ConversationLogger | None = None
        if start_conversation_logger:
            conversation_logger = ConversationLogger()
            await conversation_logger.start()

        reasoner = Reasoner(
            tool_registry=tool_registry,
            tool_executor=tool_executor,
            event_bus=event_bus,
        )
        register_memory_tools(tool_registry, memory_runtime.engine)

        plugin_manager = PluginManager(
            [workspace / "plugins"],
            event_bus=event_bus,
            tool_registry=tool_registry,
            workspace=workspace,
            memory_engine=memory_runtime.engine,
        )
        await plugin_manager.load_all()
        tool_executor.add_hooks(plugin_manager.tool_hooks)
        reasoner.set_step_modules(
            before_step=plugin_manager.before_step_modules,
            after_step=plugin_manager.after_step_modules,
        )

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
        )

        consolidation = ConsolidationWorker(
            keep_count=10,
            min_new_messages=6,
            markdown_store=memory_runtime.markdown.store,
        )
        invalidation = InvalidationWorker(memory_store, embedder)

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

        return cls(
            pipeline=pipeline,
            plugin_manager=plugin_manager,
            conversation_logger=conversation_logger,
            after_turn=after_turn,
        )

    def set_telegram_adapter(self, adapter: TelegramAdapter) -> None:
        """Attach Telegram dispatch after the adapter is constructed."""
        self.after_turn.telegram_adapter = adapter

    async def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self.plugin_manager.terminate_all()
        if self.conversation_logger is not None:
            await self.conversation_logger.stop()
