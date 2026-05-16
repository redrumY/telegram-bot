import asyncio
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ["TG_BOT_TOKEN"] = "test_token"
os.environ["DEEPSEEK_API_KEY"] = "test_deepseek_key"
os.environ["ALIYUN_DASHSCOPE_API_KEY"] = "test_aliyun_key"
os.environ["DATABASE_PATH"] = tempfile.mktemp(suffix=".db")
os.environ["DEEPSEEK_BASE_URL"] = "https://api.test.com"
os.environ["LLM_MODEL"] = "test-model"

from eval.replay_runner import (
    _build_parser,
    _clear_eval_state,
    _create_pipeline,
    _load_replay_conversations,
    _tool_policy_for_case,
)
from evaluation.dataset_builder import EvalCase, QuestionType, DistanceType
from agent.pipeline.reasoner import Reasoner
from memory.embedder import Embedder
from memory.store import MemoryStore
from persistence.database import init_db


async def test_replay_pipeline_uses_real_reasoner() -> None:
    """Replay eval should wire the real agent loop, not a mock reasoner."""
    init_db()
    _clear_eval_state()
    embedder = Embedder()
    store = MemoryStore(embedder)
    pipeline = await _create_pipeline(store=store, embedder=embedder)

    assert isinstance(pipeline.reasoner, Reasoner)
    assert pipeline.before_reasoning.benchmark_mode is True
    print("test_replay_pipeline_uses_real_reasoner: PASS")


def test_replay_parser_has_no_mock_mode() -> None:
    parser = _build_parser()

    args = parser.parse_args(["--set", "green", "--fresh", "--limit", "1"])
    assert args.set == "green"
    assert args.fresh is True
    assert not hasattr(args, "mock")
    print("test_replay_parser_has_no_mock_mode: PASS")


def test_replay_conversations_loader_keeps_curated_dataset() -> None:
    conversations = _load_replay_conversations()

    assert conversations
    assert "messages" in conversations[0]
    print("test_replay_conversations_loader_keeps_curated_dataset: PASS")


def test_tool_policy_requires_recall_search_fetch_for_update() -> None:
    case = EvalCase(
        case_id="test-policy",
        question="我现在喝茶还是咖啡？",
        gold_answer="你现在喝茶",
        question_type=QuestionType.KNOWLEDGE_UPDATE,
        context_sessions=[],
        distance_type=DistanceType.SEMANTIC_SIMILARITY,
    )

    policy = _tool_policy_for_case(
        case,
        [
            {"function": {"name": "recall_memory"}},
            {"function": {"name": "search_messages"}},
        ],
    )

    assert policy["required"] == ["recall_memory", "fetch_messages", "search_messages"]
    assert policy["missing"] == ["fetch_messages"]
    assert policy["satisfied"] is False
    print("test_tool_policy_requires_recall_search_fetch_for_update: PASS")


async def main() -> None:
    test_replay_parser_has_no_mock_mode()
    test_replay_conversations_loader_keeps_curated_dataset()
    test_tool_policy_requires_recall_search_fetch_for_update()
    await test_replay_pipeline_uses_real_reasoner()
    print("\nAll replay runner tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
