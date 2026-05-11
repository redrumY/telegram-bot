import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.core.event_bus import EventBus


async def test_subscribe_and_emit():
    bus = EventBus()
    results = []

    def sync_handler(value: int) -> None:
        results.append(("sync", value))

    async def async_handler(value: int) -> None:
        await asyncio.sleep(0)
        results.append(("async", value))

    bus.subscribe("test_event", sync_handler)
    bus.subscribe("test_event", async_handler)

    await bus.emit("test_event", value=42)

    assert results == [("sync", 42), ("async", 42)]
    print("test_subscribe_and_emit: PASS")


async def test_handler_exception():
    bus = EventBus()
    results = []

    def failing_handler() -> None:
        raise RuntimeError("handler error")

    def working_handler() -> None:
        results.append("ok")

    bus.subscribe("error_event", failing_handler)
    bus.subscribe("error_event", working_handler)

    await bus.emit("error_event")

    assert results == ["ok"]
    print("test_handler_exception: PASS")


async def test_no_subscribers():
    bus = EventBus()
    # Should not raise
    await bus.emit("nonexistent_event")
    print("test_no_subscribers: PASS")


async def main():
    await test_subscribe_and_emit()
    await test_handler_exception()
    await test_no_subscribers()
    print("\nAll tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
