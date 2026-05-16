"""
Consolidation 窗口期处理：从对话窗口提取长期记忆

复刻 akashic-agent 的 _MarkdownConsolidationWorker + _on_consolidation_committed 逻辑：
  每轮对话后异步检查 → 攒够 N 条新消息 → LLM 提取 profile/preference/event
  → 写入向量库，附带 source_ref 可追溯

akashic 模式：
  TurnCommitted → _enqueue_maintenance → _should_consolidate_session
  → _consolidate_unlocked → ConsolidationCommitted event
  → _on_consolidation_committed → _extract_implicit_long_term → _save_implicit_long_term
"""

from __future__ import annotations

import json
import logging
from typing import Any

from openai import AsyncOpenAI

from config.settings import settings

logger = logging.getLogger(__name__)

# ── 提取 prompt（对齐 akashic _build_long_term_prompt 的核心规则）─────────────

_EXTRACTION_PROMPT = """你是长期记忆提取专家。从对话窗口中提取值得长期记住的信息，返回 JSON。

默认答案是所有数组为空。提取门槛要高，宁可不提取，也不要把临时信息写进长期记忆。

【核心判断标准】
把这条信息放进 6 个月后的一次全新对话，它还有用吗？
→ 是 → 可能是长期记忆，继续检查
→ 否 → 不是长期记忆，留空

【三类记忆的语义】
profile — 关于用户本人的客观事实：身份/职业/技能/持有物/状态/重要决定
preference — 用户明确表达的偏好/喜好/厌恶/倾向
event — 用户提到的重要事件、行为、状态变化

【提取规则】
1. 只提取 USER 明确表达的信息，禁止推测
2. ASSISTANT 的回复/建议/解释一律不得作为提取来源
3. "你还记得吗""我之前是不是"这类提问句不提取——这不是事实披露
4. 涉及"今天""这次""当前"的临时信息不提取
5. 用户说"记住……"时优先信任 memorize 工具，不在此重复提取
6. 保留具体实体、名称、地点、型号、数量、人物名，不要写成"某个/一些/相关"
7. 每条 summary 只表达一个原子事实；不要把多个事实揉成一条
8. 更新类事实必须写清"旧状态 -> 新状态"，例如"用户以前用 iPhone，现在改用 Android 手机"
9. 用户只是短期计划、即时情绪、临时进度、一次性请求时不提取，除非它揭示了长期稳定事实
10. 外部 transcript、转贴聊天、示例、假设里的第一人称不等于当前用户事实；除非用户明确说明，否则不提取为用户事实
11. summary 写成完整陈述句，必须包含可独立检索的关键词，脱离对话也能理解

【对话】
{conversation}

只返回合法 JSON，不要 markdown 代码块：
{{
  "profile": [
    {{"summary": "...", "category": "personal_fact"}}
  ],
  "preference": [
    {{"summary": "..."}}
  ],
  "event": [
    {{"summary": "..."}}
  ]
}}"""


# ── ConsolidationWorker ──────────────────────────────────────────────────────

class ConsolidationWorker:
    """
    窗口期 consolidation：异步检查 session 是否有足够新消息，
    调 LLM 提取长期记忆，写入 MemoryStore。

    对齐 akashic：
      _should_consolidate_session → _select_consolidation_window
      consolidate → _extract_implicit_long_term + _save_implicit_long_term
    """

    def __init__(
        self,
        *,
        keep_count: int = 10,
        min_new_messages: int = 6,
    ) -> None:
        self._keep_count = keep_count
        self._min_new_messages = min_new_messages

    def should_consolidate(self, session: Any) -> bool:
        """
        判断是否需要 consolidation。

        对齐 akashic _should_consolidate_session → _select_consolidation_window：
          total_messages > keep_count AND 有未 consolidate 的新消息 AND 窗口非空
        """
        total = len(session.messages)
        new_count = total - session.last_consolidated

        if new_count <= 0:
            return False
        if total <= self._keep_count:
            return False
        if new_count < self._min_new_messages:
            return False

        # 检查窗口是否非空（对齐 akashic _select_consolidation_window 返回 None）
        consolidate_up_to = total - self._keep_count
        if session.last_consolidated >= consolidate_up_to:
            return False

        return True

    def get_consolidation_window(self, session: Any) -> list[dict[str, str]]:
        """
        取待 consolidate 的对话窗口。

        对齐 akashic _select_consolidation_window：
          old_messages = session.messages[last_consolidated : total - keep_count]
        """
        total = len(session.messages)
        consolidate_up_to = total - self._keep_count
        return session.messages[session.last_consolidated : consolidate_up_to]

    async def consolidate(
        self,
        session: Any,
        store: Any,
        user_id: int,
        chat_id: int,
    ) -> int:
        """
        执行一次 consolidation：
          1. 取对话窗口
          2. LLM 提取
          3. 写入 MemoryStore
          4. 更新 session.last_consolidated

        返回写入的记忆条目数。
        """
        total_at_start = len(session.messages)
        consolidate_up_to = total_at_start - self._keep_count
        window_start = session.last_consolidated
        window = session.messages[window_start:consolidate_up_to]
        if not window:
            logger.debug(
                "Consolidation skipped: empty window "
                "user=%d chat=%d last_consolidated=%d total=%d",
                user_id, chat_id, session.last_consolidated, total_at_start,
            )
            return 0

        # 构建对话文本
        conversation = "\n".join(
            f"[{m['role']}]: {m['content']}" for m in window
        )

        logger.info(
            "Consolidation: user=%d chat=%d window=%d messages total=%d",
            user_id, chat_id, len(window), len(session.messages),
        )

        # LLM 提取
        summaries = await self._llm_extract(conversation)
        if not summaries:
            logger.info("Consolidation: LLM extracted nothing user=%d chat=%d", user_id, chat_id)
            # 仍然推进 last_consolidated，避免下次重复提取同窗口
            session.last_consolidated = consolidate_up_to
            return 0

        # 写入 MemoryStore
        source_ref = _build_window_source_ref(
            user_id=user_id,
            chat_id=chat_id,
            start=window_start,
            end=consolidate_up_to - 1,
        )
        written = 0
        for s in summaries:
            try:
                await store.upsert_item(
                    memory_type=s["memory_type"],
                    summary=s["summary"],
                    user_id=user_id,
                    source_ref=source_ref,
                )
                written += 1
                logger.info(
                    "Consolidation saved: [%s] %s",
                    s["memory_type"], s["summary"][:80],
                )
            except Exception as e:
                logger.error("Consolidation upsert failed: %s", e)

        # 推进指针（对齐 akashic session.last_consolidated = consolidate_up_to）
        session.last_consolidated = consolidate_up_to

        logger.info(
            "Consolidation done: user=%d chat=%d written=%d last_consolidated=%d",
            user_id, chat_id, written, session.last_consolidated,
        )
        return written

    async def _llm_extract(self, conversation: str) -> list[dict[str, str]]:
        """调 LLM 提取结构化摘要"""
        if not conversation.strip():
            return []

        prompt = _EXTRACTION_PROMPT.format(conversation=conversation[:4000])

        try:
            client = AsyncOpenAI(
                api_key=settings.DEEPSEEK_API_KEY,
                base_url=settings.DEEPSEEK_BASE_URL,
                timeout=60.0,
            )
            resp = await client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.1,
            )
            text = (resp.choices[0].message.content or "").strip()
        except Exception as e:
            logger.warning("Consolidation LLM extraction failed: %s", e)
            return []

        # 解析 LLM 返回的 JSON
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Consolidation LLM returned invalid JSON: %r", text[:200])
            return []

        if not isinstance(result, dict):
            return []

        summaries: list[dict[str, str]] = []

        # profile → memory_type="profile"
        for item in result.get("profile") or []:
            if isinstance(item, dict) and item.get("summary"):
                summaries.append({
                    "summary": str(item["summary"]).strip(),
                    "memory_type": "profile",
                })

        # preference → memory_type="preference"
        for item in result.get("preference") or []:
            if isinstance(item, dict) and item.get("summary"):
                summaries.append({
                    "summary": str(item["summary"]).strip(),
                    "memory_type": "preference",
                })

        # event → memory_type="event"
        for item in result.get("event") or []:
            if isinstance(item, dict) and item.get("summary"):
                summaries.append({
                    "summary": str(item["summary"]).strip(),
                    "memory_type": "event",
                })

        return summaries


def _build_window_source_ref(*, user_id: int, chat_id: int, start: int, end: int) -> str:
    """Build a fetchable source_ref for the exact consolidated message window."""
    if start < 0 or end < start:
        return f"session:{user_id}:{chat_id}"
    return f"session:{user_id}:{chat_id}#msg:{start}-{end}"
