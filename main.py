import asyncio
import logging

from agent.runtime import AgentRuntime
from channels.telegram.adapter import TelegramAdapter
from config.settings import settings

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

    runtime = await AgentRuntime.create()
    logger.info("Conversation logger started")

    adapter = TelegramAdapter(
        token=settings.TG_BOT_TOKEN,
        pipeline=runtime.pipeline,
        proxy=settings.HTTP_PROXY,
    )
    runtime.set_telegram_adapter(adapter)

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
        await runtime.shutdown()
        logger.info("Conversation logger stopped")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
