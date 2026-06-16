"""Command entry point for the web chat turn worker."""

from __future__ import annotations

import asyncio
import logging

from agent.runtime import AgentRuntime
from agent.service import AgentService
from worker.turn_worker import TurnWorker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


async def main() -> None:
    runtime = await AgentRuntime.create()
    worker = TurnWorker(service=AgentService(runtime.pipeline))
    try:
        await worker.run_forever()
    finally:
        await runtime.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
