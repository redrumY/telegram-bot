import asyncio
import json
from datetime import datetime, timedelta

from openai import AsyncOpenAI

from agent.core.types import BeforeReasoningCtx, ReasonerResult
from memory.store import LONG_TERM_MEMORY_TYPES
from config.settings import settings


class Reasoner:
    """LLM 调用器，管理 DeepSeek API 调用和 tool call 循环"""

    def __init__(
        self,
        store: "MemoryStore | None" = None,
        embedder: "Embedder | None" = None,
        session_store: "SessionStore | None" = None,
    ) -> None:
        self.client = AsyncOpenAI(
            api_key=settings.DEEPSEEK_API_KEY,
            base_url=settings.DEEPSEEK_BASE_URL,
            timeout=60.0,
        )
        self.model = settings.LLM_MODEL
        self._store = store
        self._embedder = embedder
        self._session_store = session_store

    async def _execute_tool(self, tool_name: str, arguments: dict, ctx: BeforeReasoningCtx) -> str:
        """Execute a tool call. 工具结果以 JSON 字符串返回给 LLM。"""
        if tool_name == "memorize":
            return await self._memorize(arguments, ctx)

        if tool_name == "recall_memory":
            return await self._recall_memory(arguments, ctx)

        if tool_name == "fetch_messages":
            return await self._fetch_messages(arguments)

        if tool_name == "search_messages":
            return await self._search_messages(arguments, ctx)

        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    async def _memorize(self, args: dict, ctx: BeforeReasoningCtx) -> str:
        """
        memorize 工具：将用户明确要求记住的内容写入 MemoryStore。

        对应 akashic-agent MemorizeTool → engine.remember() → Memorizer.save_item_with_supersede()。

        参数对齐 akashic：summary + memory_type (procedure/preference/event/profile)。
        """
        if not self._store or not self._embedder:
            return json.dumps(
                {"status": "failed", "error": "memory_not_available"},
                ensure_ascii=False,
            )

        summary = str(args.get("summary", "")).strip()
        memory_type = str(args.get("memory_type", "")).strip()

        if not summary or not memory_type:
            return json.dumps(
                {"status": "failed", "error": "summary and memory_type required"},
                ensure_ascii=False,
            )

        if memory_type not in ("procedure", "preference", "event", "profile"):
            return json.dumps(
                {"status": "failed", "error": f"invalid memory_type: {memory_type}"},
                ensure_ascii=False,
            )

        try:
            user_id = ctx.session.user_id
            chat_id = ctx.session.chat_id
            source_ref = f"session:{user_id}:{chat_id}"

            item = await self._store.upsert_item(
                memory_type=memory_type,
                summary=summary,
                user_id=user_id,
                source_ref=source_ref,
            )

            return json.dumps(
                {
                    "status": "saved",
                    "item_id": str(item.id),
                    "summary": summary,
                    "memory_type": memory_type,
                },
                ensure_ascii=False,
            )
        except Exception as e:
            return json.dumps(
                {"status": "failed", "error": str(e)},
                ensure_ascii=False,
            )

    async def _recall_memory(self, args: dict, ctx: BeforeReasoningCtx) -> str:
        """
        recall_memory 工具：语义+关键词混合检索长期记忆。

        对应 akashic-agent RecallMemoryTool → retrieve_explicit()。

        返回格式：
        {
          "count": N,
          "items": [
            {"id": "...", "memory_type": "...", "summary": "...",
             "source_ref": "...", "score": 0.95},
            ...
          ]
        }
        """
        if not self._store or not self._embedder:
            return json.dumps({"count": 0, "items": [], "error": "memory_not_available"})

        query = str(args.get("query", "")).strip()
        if not query:
            return json.dumps({"count": 0, "items": []}, ensure_ascii=False)

        memory_type = str(args.get("memory_type", "")).strip() or None
        include_superseded = bool(args.get("include_superseded", False))
        search_mode = str(args.get("search_mode", "semantic")).strip() or "semantic"
        time_filter = str(args.get("time_filter", "")).strip()
        memory_types = _infer_memory_types(
            query=query,
            explicit_memory_type=memory_type,
            search_mode=search_mode,
            time_filter=time_filter,
        )
        time_window = _parse_time_filter(time_filter)
        if time_filter and time_window is None:
            return json.dumps(
                {"count": 0, "items": [], "error": "invalid_time_filter"},
                ensure_ascii=False,
            )
        if search_mode not in {"semantic", "grep"}:
            search_mode = "semantic"
        max_limit = 50 if search_mode == "grep" else 10
        limit = max(1, min(int(args.get("limit", 5)), max_limit))
        user_id = int(args.get("user_id") or ctx.session.user_id)

        try:
            if search_mode == "grep":
                if time_window is None:
                    return json.dumps(
                        {"count": 0, "items": [], "error": "time_filter_required"},
                        ensure_ascii=False,
                    )
                start, end = time_window
                grep_results = self._store.list_memories(
                    user_id=user_id,
                    memory_types=[memory_type] if memory_type else ["event"],
                    include_superseded=include_superseded,
                    created_start=start,
                    created_end=end,
                    limit=limit,
                )
                return json.dumps(
                    {
                        "count": len(grep_results),
                        "applied_memory_types": [memory_type] if memory_type else ["event"],
                        "items": [
                            {
                                "id": str(mem.id),
                                "memory_type": mem.memory_type,
                                "summary": mem.summary,
                                "source_ref": mem.source_ref,
                                "status": mem.status,
                                "score": 1.0,
                            }
                            for mem in grep_results
                        ],
                    },
                    ensure_ascii=False,
                )

            # 1. 向量检索
            query_vec = await self._embedder.embed(query)
            vec_results = await self._store.vector_search(
                query_vec=query_vec,
                user_id=user_id,
                top_k=limit,
                memory_types=memory_types,
                include_superseded=include_superseded,
            )

            # 2. 关键词检索（补充）
            kw_results = await self._store.keyword_search(
                terms=query,
                user_id=user_id,
                limit=max(1, limit // 2),
                memory_types=memory_types,
                include_superseded=include_superseded,
            )

            # 3. 去重合并（向量结果优先）
            seen_ids: set[str] = set()
            items: list[dict] = []

            for mem in vec_results:
                if str(mem.id) not in seen_ids:
                    seen_ids.add(str(mem.id))
                    items.append({
                        "id": str(mem.id),
                        "memory_type": mem.memory_type,
                        "summary": mem.summary,
                        "source_ref": mem.source_ref,
                        "status": mem.status,
                        "score": 1.0,  # 向量结果无显式分数，简化
                    })

            for mem in kw_results:
                if str(mem.id) not in seen_ids:
                    seen_ids.add(str(mem.id))
                    items.append({
                        "id": str(mem.id),
                        "memory_type": mem.memory_type,
                        "summary": mem.summary,
                        "source_ref": mem.source_ref,
                        "status": mem.status,
                        "score": 0.5,
                    })

            if time_window is not None:
                start, end = time_window
                items = [
                    item for item in items
                    if _memory_created_in_window(item["id"], [*vec_results, *kw_results], start, end)
                ]

            items = items[:limit]
            return json.dumps(
                {
                    "count": len(items),
                    "applied_memory_types": memory_types,
                    "items": items,
                },
                ensure_ascii=False,
            )

        except Exception as e:
            return json.dumps(
                {"count": 0, "items": [], "error": str(e)},
                ensure_ascii=False,
            )

    async def _fetch_messages(self, args: dict) -> str:
        """
        fetch_messages 工具：按 source_ref 取原始对话消息。

        对应 akashic-agent FetchMessagesTool。

        返回格式：
        {
          "matched_count": N,
          "messages": [
            {"role": "user", "content": "...", "seq": 0},
            ...
          ]
        }
        """
        if not self._session_store:
            return json.dumps(
                {"matched_count": 0, "messages": [], "error": "session_store_not_available"},
                ensure_ascii=False,
            )

        source_refs = _coerce_source_refs(args)
        if not source_refs:
            return json.dumps(
                {"matched_count": 0, "messages": [], "error": "source_ref_required"},
                ensure_ascii=False,
            )

        limit = max(1, min(int(args.get("limit", 20)), 50))
        context = max(0, min(int(args.get("context", 0)), 10))

        all_fetched: list[dict] = []
        total_matched = 0
        invalid_refs: list[str] = []
        seen: set[tuple[str, int | None]] = set()
        for source_ref in source_refs:
            # 解析 source_ref: "session:user_id:chat_id" or "...#msg:<seq[-end]>"
            user_id, chat_id, seq, seq_end = _parse_session_ref(source_ref)
            if user_id is None or chat_id is None:
                invalid_refs.append(source_ref)
                continue

            fetched, matched = self._session_store.fetch_messages(
                user_id,
                chat_id,
                seq=seq,
                seq_end=seq_end,
                context=context,
                limit=limit,
            )
            total_matched += matched
            for message in fetched:
                key = (str(message.get("source_ref", "")), message.get("seq"))
                if key in seen:
                    continue
                seen.add(key)
                all_fetched.append(message)

        if not all_fetched:
            payload = {"matched_count": 0, "messages": []}
            if invalid_refs:
                payload["error"] = f"invalid_source_ref: {', '.join(invalid_refs)}"
            return json.dumps(payload, ensure_ascii=False)

        if len(all_fetched) > limit:
            all_fetched = all_fetched[:limit]

        if not all_fetched:
            return json.dumps({"matched_count": 0, "messages": []}, ensure_ascii=False)

        result_messages = [
            {
                "role": m.get("role", "?"),
                "content": str(m.get("content", ""))[:500],
                "seq": m.get("seq"),
                "source_ref": m.get("source_ref", ""),
                "in_source_ref": bool(m.get("in_source_ref")),
            }
            for m in all_fetched
        ]

        payload = {
            "matched_count": total_matched,
            "source_refs": source_refs,
            "messages": result_messages,
        }
        if invalid_refs:
            payload["invalid_source_refs"] = invalid_refs
        if len(source_refs) == 1:
            payload["source_ref"] = source_refs[0]
        return json.dumps(payload, ensure_ascii=False)

    async def _search_messages(self, args: dict, ctx: BeforeReasoningCtx) -> str:
        """
        search_messages 工具：对持久化原始会话做关键词搜索。

        对应 akashic-agent SearchMessagesTool。返回 source_ref，供
        fetch_messages 继续回源取证。
        """
        if not self._session_store:
            return json.dumps(
                {
                    "count": 0,
                    "matched_count": 0,
                    "messages": [],
                    "error": "session_store_not_available",
                },
                ensure_ascii=False,
            )

        query = str(args.get("query", "")).strip()
        if not query:
            return json.dumps(
                {
                    "count": 0,
                    "matched_count": 0,
                    "messages": [],
                    "has_more": False,
                    "next_offset": None,
                },
                ensure_ascii=False,
            )

        limit = max(1, min(int(args.get("limit", 10)), 50))
        offset = max(0, int(args.get("offset", 0)))
        role = str(args.get("role", "")).strip() or None
        user_id = int(args.get("user_id") or ctx.session.user_id)

        try:
            messages, total = self._session_store.search_messages(
                query,
                user_id=user_id,
                role=role,
                limit=limit,
                offset=offset,
            )
        except Exception as e:
            return json.dumps(
                {"count": 0, "matched_count": 0, "messages": [], "error": str(e)},
                ensure_ascii=False,
            )

        public_messages = [
            {
                "role": m.get("role", ""),
                "content": str(m.get("content", ""))[:300],
                "seq": m.get("seq"),
                "source_ref": m.get("source_ref", ""),
            }
            for m in messages
        ]
        next_offset = offset + len(public_messages)
        has_more = next_offset < total
        return json.dumps(
            {
                "count": len(public_messages),
                "matched_count": total,
                "limit": limit,
                "offset": offset,
                "has_more": has_more,
                "next_offset": next_offset if has_more else None,
                "messages": public_messages,
            },
            ensure_ascii=False,
        )

    async def run_turn(self, ctx: BeforeReasoningCtx) -> ReasonerResult:
        """Run a reasoning turn with potential tool calls."""
        messages = ctx.messages.copy()
        tool_calls: list[dict] = []

        for iteration in range(3):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=ctx.tools,
                )
            except Exception as e:
                if iteration == 0:
                    await asyncio.sleep(0.5)
                    continue
                raise

            choice = response.choices[0]
            message = choice.message

            # Check for tool calls
            if message.tool_calls:
                tool_calls.extend([
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in message.tool_calls
                ])

                # Add assistant message with tool calls
                messages.append({
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in message.tool_calls
                    ],
                })

                # Execute tools and add responses
                for idx, tc in enumerate(message.tool_calls):
                    args = json.loads(tc.function.arguments)
                    result = await self._execute_tool(tc.function.name, args, ctx)
                    tool_calls[len(tool_calls) - len(message.tool_calls) + idx]["result"] = result
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })

                # Continue loop for LLM to respond with final answer
                continue
            else:
                # No tool calls, this is the final response
                return ReasonerResult(
                    content=message.content or "",
                    tool_calls=tool_calls,
                    finish_reason=choice.finish_reason or "stop",
                )

        # Max iterations reached
        return ReasonerResult(
            content="抱歉，处理请求时遇到问题。",
            tool_calls=tool_calls,
            finish_reason="max_iterations",
        )


def _parse_session_ref(source_ref: str) -> tuple[int | None, int | None, int | None, int | None]:
    """解析 source_ref 格式 'session:user_id:chat_id[#msg:seq[-end]]'"""
    base, _, suffix = source_ref.partition("#")
    parts = base.split(":")
    if len(parts) >= 3 and parts[0] == "session":
        try:
            seq = None
            seq_end = None
            if suffix.startswith("msg:"):
                raw_seq = suffix.split(":", 1)[1]
                if "-" in raw_seq:
                    left, right = raw_seq.split("-", 1)
                    seq = int(left)
                    seq_end = int(right)
                else:
                    seq = int(raw_seq)
            return int(parts[1]), int(parts[2]), seq, seq_end
        except (ValueError, IndexError):
            pass
    return None, None, None, None


def _infer_memory_types(
    *,
    query: str,
    explicit_memory_type: str | None,
    search_mode: str,
    time_filter: str,
) -> list[str]:
    """Infer a conservative memory_type filter when the model omits it."""
    if explicit_memory_type:
        return [explicit_memory_type]
    if search_mode == "grep" or time_filter:
        return ["event"]

    text = query.lower()
    if _contains_any(
        text,
        (
            "以后",
            "下次",
            "你要怎么做",
            "怎么做",
            "流程",
            "规则",
            "操作规范",
            "必须",
            "应该",
            "工具",
        ),
    ):
        return ["procedure"]

    if _contains_any(
        text,
        (
            "今天聊",
            "今天做",
            "昨天聊",
            "昨天做",
            "最近聊",
            "最近做",
            "聊过什么",
            "做过什么",
            "发生过",
            "历史事件",
        ),
    ):
        return ["event"]

    if _contains_any(
        text,
        (
            "职业",
            "工作",
            "公司",
            "城市",
            "居住",
            "住在",
            "生日",
            "年龄",
            "编程语言",
            "技术栈",
            "手机",
            "设备",
            "iphone",
            "android",
        ),
    ):
        return ["profile"]

    if _contains_any(
        text,
        (
            "喜欢",
            "偏好",
            "推荐",
            "喝",
            "咖啡",
            "茶",
            "饮品",
            "饮料",
            "音乐",
            "音乐人",
            "食物",
            "川菜",
            "摇滚",
            "爵士",
            "不喜欢",
            "讨厌",
        ),
    ):
        # Preference updates may be extracted as profile status changes, so keep
        # profile in this lane while still excluding procedure/event noise.
        return ["preference", "profile"]

    return LONG_TERM_MEMORY_TYPES


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def _coerce_source_refs(args: dict) -> list[str]:
    refs: list[str] = []
    seen: set[str] = set()

    def _add(value) -> None:
        ref = str(value or "").strip()
        if ref and ref not in seen:
            seen.add(ref)
            refs.append(ref)

    _add(args.get("source_ref"))
    raw_refs = args.get("source_refs")
    if isinstance(raw_refs, list):
        for ref in raw_refs:
            _add(ref)
    elif raw_refs:
        _add(raw_refs)
    return refs


def _parse_time_filter(value: str) -> tuple[datetime, datetime] | None:
    text = (value or "").strip()
    if not text:
        return None
    now = datetime.utcnow()
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    if text == "today":
        return today, today + timedelta(days=1)
    if text == "yesterday":
        return today - timedelta(days=1), today
    presets = {"recent_3d": 3, "recent_7d": 7, "recent_30d": 30}
    if text in presets:
        return now - timedelta(days=presets[text]), now
    if "~" in text:
        left, right = [part.strip() for part in text.split("~", 1)]
        try:
            start = datetime.strptime(left, "%Y-%m-%d")
            end = datetime.strptime(right, "%Y-%m-%d") + timedelta(days=1)
            return start, end
        except ValueError:
            return None
    try:
        start = datetime.strptime(text, "%Y-%m-%d")
        return start, start + timedelta(days=1)
    except ValueError:
        return None


def _memory_created_in_window(
    item_id: str,
    memories,
    start: datetime,
    end: datetime,
) -> bool:
    for mem in memories:
        if str(mem.id) != item_id:
            continue
        created_at = mem.created_at
        if created_at is None:
            return False
        if created_at.tzinfo is not None:
            created_at = created_at.replace(tzinfo=None)
        return start <= created_at < end
    return False
