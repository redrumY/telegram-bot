# 项目进展 (2026-05-15)

更完整的待办、进度和 Roadmap 见 `docs/roadmap.md`。

## 架构

### 被动回复链 (对齐 akashic-agent)
```
Telegram消息 → pipeline.execute(inbound)
  ├─ BeforeTurnPhase  (HyDE增强 + RRF融合检索 + SessionStore加载)
  ├─ BeforeReasoning  (分组注入prompt + 工具定义)
  ├─ Reasoner         (DeepSeek + tool_call循环: recall_memory/fetch_messages/memorize)
  ├─ AfterReasoning   (异步persist_messages, source_ref=session:user:chat)
  └─ AfterTurn        (emit事件 + 发送回复 + SessionStore.save)
```

### RAG 存取层
- **MemoryStore**: sqlite-vec 向量检索 (L2距离) + 关键词 LIKE 搜索
- **SessionStore**: SQLite 持久化会话 (conversation_sessions表)
- **Consolidation**: LLM提取摘要 (evaluation/consolidation.py)
- **HyDE增强**: LLM生成假想记忆条目 → 二次向量检索 (memory/hyde_enhancer.py)
- **source_ref**: 格式 "session:user_id:chat_id"，链接记忆到原始会话

### Eval 框架 (对齐 akashic eval/longmemeval)
```
eval/replay_runner.py     主评估入口：会话回放 + consolidation/invalidation + 真实被动回复链 QA
eval/metrics.py           token_f1 + exact_match + judge + score_results
eval/dataset_builder.py   EvalCase数据结构 + green_set/red_set存取
eval/runner.py            旧的 seeded regression 入口，不作为真实链路指标
eval/seed_memory.py       legacy 播种工具，仅用于旧 eval/runner.py
eval/consolidation.py     legacy eval 摘要提炼；replay eval 使用 ConsolidationWorker
eval/conversation_logger.py 生产环境对话日志
eval/rag_layer_smoke.py   consolidation窗口期 + invalidation生命周期smoke
```

## 已完成

1. source_ref 追溯 (session:user_id:chat_id)
2. recall_memory 主动检索工具
3. fetch_messages 原始消息取证工具
4. RRF 融合排序 (替代简单拼接, BeforeTurnPhase)
5. HyDE 增强检索 (LLM生成假想记忆 → 二次检索, memory/hyde_enhancer.py)
6. LLM Consolidation (对话摘要提炼, evaluation/consolidation.py)
7. 异步 persist_messages (asyncio.create_task, 不阻塞回复)
8. Session 持久化 (persistence/session_store.py, 重启不丢失)
9. Replay Eval CLI: --fresh --trace --set --output
10. Invalidation: 回复后异步纠错废弃旧记忆 (参考 akashic PostResponseMemoryWorker)
11. CI smoke: `eval/rag_layer_smoke.py` + GitHub Actions 验证 RAG 生命周期

## 当前 Eval 数据
- 本地 curated conversation corpus: 15条历史会话（旧文件名可能仍含 mock）
- green_set.json: 13个绿集用例
- red_set.json: 5个红灯用例
- legacy seeded baseline.json: F1≈0.128, Judge Acc=100%（只作历史对照）
- replay live baseline: F1=0.6338, Judge Acc=94.44%, Errors=0
- replay live prompt experiment: F1=0.5655, Judge Acc=88.89%, Errors=0；局部修复 green-003，但引入回归，未保留

## 下一步
- 将当前 curated conversation cases 转成 LongMemEval 风格 JSON
- 增加显式invalidation端到端benchmark case
- 增加tool trace断言: reasoner是否真的调用recall_memory/fetch_messages
- 扩展 eval case 覆盖新主题 (音乐/居住地/Rust学习)
- 给 invalidation 增加端到端 eval case: 旧偏好 → 用户纠正 → 旧记忆 superseded
- 给 replay eval 增加代码级 tool policy guard，减少 prompt-only 随机性
- 调 consolidation 提取 prompt
- 调 HyDE prompt 生成更精准的假想记忆
- 增加更多 curated 历史会话作为检索噪音

## 技术栈
- LLM: DeepSeek chat
- Embedding: DashScope text-embedding-v3 (1024维)
- 向量库: SQLite + sqlite-vec
- Bot: python-telegram-bot 21.x
- 配置: pydantic-settings (.env)
