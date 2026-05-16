import asyncio

from openai import AsyncOpenAI

from config.settings import settings


class Embedder:
    def __init__(self) -> None:
        self.client = AsyncOpenAI(
            api_key=settings.ALIYUN_DASHSCOPE_API_KEY,
            base_url=settings.EMBEDDING_BASE_URL,
            timeout=30.0,
        )
        self.model = settings.EMBEDDING_MODEL

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for the given text with retry."""
        for attempt in range(2):
            try:
                response = await self.client.embeddings.create(
                    model=self.model,
                    input=text,
                )
                return response.data[0].embedding
            except Exception as e:
                if attempt == 0:
                    await asyncio.sleep(0.5)
                    continue
                raise
        return []  # unreachable but satisfies type checker
