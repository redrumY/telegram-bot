import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ["TG_BOT_TOKEN"] = "test_token"
os.environ["DEEPSEEK_API_KEY"] = "test_deepseek_key"
os.environ["ALIYUN_DASHSCOPE_API_KEY"] = "sk-8a0ec2bce95b407eab421e3cae336e0d"
os.environ["DATABASE_PATH"] = "/tmp/test.db"

from memory.embedder import Embedder


async def main():
    embedder = Embedder()
    result = await embedder.embed("你好，世界")
    print(f"Embedding dimension: {len(result)}")
    print(f"First 5 values: {result[:5]}")
    print("Success!")


if __name__ == "__main__":
    asyncio.run(main())
