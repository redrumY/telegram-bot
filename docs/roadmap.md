# Roadmap

Last updated: 2026-05-15

## Current Status

This project is following akashic-agent's passive reply chain shape while keeping
the implementation scoped to this bot.

Completed:

1. Passive turn pipeline with the five phases preserved.
2. `source_ref` format aligned to `session:{user_id}:{chat_id}`.
3. `recall_memory` and `fetch_messages` tools available to the reasoner.
4. Session persistence through `SessionStore`.
5. RRF and HyDE retrieval enhancements.
6. Akashic-style consolidation mounted after turns.
7. Akashic-style invalidation mounted after replies.
8. Lifecycle smoke eval for consolidation window and invalidation status changes.
9. CI workflow that runs smoke eval and script-based tests.
10. Replay eval runner that replays curated haystack conversations before QA.
11. Replay eval no longer has a mock execution layer: it uses the real
    `Embedder`, `ConsolidationWorker`, `InvalidationWorker`, and `Reasoner`.
12. Benchmark eval decisions now default to `eval/replay_runner.py`; seeded
    `eval/runner.py` remains only as a legacy regression shortcut.

## Latest Eval Results

Legacy eval:

- Command: `python eval/runner.py --fresh --compare data/evaluation/baseline.json --output data/evaluation/results/current_eval.json`
- Baseline F1: `0.1278`
- Current F1: `0.1290`
- Judge accuracy: `100%`
- Errors: `0`
- Result: no regressions, no significant improvements.
- Status: legacy seeded path only; do not use this as evidence for the real
  passive reply chain.

Replay eval, live full run:

- Command: `HTTP_PROXY=http://127.0.0.1:7897 HTTPS_PROXY=http://127.0.0.1:7897 python eval/replay_runner.py --set both --fresh --trace --output data/evaluation/results/learn_type_baseline.json`
- Cases: `18`
- Overall F1: `0.6338`
- Judge accuracy: `94.44%`
- Errors: `0`
- Trace note: the full replay path goes through real haystack replay,
  consolidation, invalidation, `PassiveTurnPipeline`, `Reasoner.run_turn`, and
  tool calls. There is no `--mock` mode in replay eval.

## How To Judge Whether The New RAG Layers Work

Consolidation is effective when:

1. A replayed conversation with enough messages writes structured long-term
   memories.
2. `last_consolidated` advances by the expected window size.
3. QA can answer from memories produced by replay, not from pre-seeded memory.

Invalidation is effective when:

1. A correction or updated preference causes stale structured memory to become
   `superseded`.
2. The newly written memory remains `active`.
3. QA uses the new fact instead of the stale fact.

Replay eval is effective when:

1. Each case uses `selection_mode=context_sessions`.
2. `consolidated_memories` is greater than zero for cases with haystack history.
3. `invalidated_memories` is greater than zero only for explicit update cases.
4. `tool_calls` show that the reasoner can use `recall_memory` or
   `fetch_messages`.

## Roadmap

### P0: Make The Current Eval Trustworthy

1. Add a code-level benchmark tool policy guard in the real agent loop:
   if benchmark mode is active and the model does not call `recall_memory`,
   force another tool-use round or inject a deterministic recall tool result.
2. Add a code-level source verification guard:
   if the case requires `fetch_messages` and `recall_memory` returned a
   `source_ref`, fetch the source messages before final answer generation.
3. Add a code-level historical query guard:
   if the question asks about `以前/之前/曾经/全部/变化过程/旧值到新值`, ensure
   `recall_memory(include_superseded=true)` is used even when the model omits it.
4. Move query formulation from prompt-only guidance toward a small deterministic
   requery helper for benchmark mode. The prompt already contains the mapping,
   but a helper would make regression results less model-random.
5. Add trace assertions for `recall_memory`, `fetch_messages`, and
   `include_superseded` usage.
6. Add explicit invalidation end-to-end benchmark cases:
   old preference -> correction -> old memory superseded -> QA answers with the
   new preference.
7. Save replay eval outputs consistently under `data/evaluation/results/`.

### P0 Notes For Next Window

Mock-layer behavior that has already been removed or migrated from replay eval:

1. `_MockEmbedder` was removed from `eval/replay_runner.py`; replay eval now
   uses the real embedding API through `Embedder`.
2. `_MockReasoner` was removed from `eval/replay_runner.py`; QA now goes through
   the real `Reasoner.run_turn`.
3. Mock extraction and mock invalidation hooks were removed from replay eval;
   ingest now uses real consolidation and invalidation workers.
4. The old `_mock_recall_query` idea was migrated into the real benchmark prompt
   and `recall_memory` tool description.
5. Direct memory seeding is not the benchmark path; replay eval builds memories
   by replaying conversations through the real consolidation/invalidation flow.

Soft prompt behavior that still needs code-level guards:

1. The real reasoner currently relies on prompt instructions to call
   `recall_memory`, so this is only a soft constraint.
2. Query rewrite used to be deterministic Python logic; the real loop now
   relies on prompt-based query formulation.
3. Replay traces can still miss required `fetch_messages`; this needs code-level
   enforcement, not more prompt text.
4. Historical questions need deterministic `include_superseded=true`
   enforcement in benchmark mode.

### P1: Align More Closely With Akashic LongMemEval

1. Convert current curated conversation cases into LongMemEval-style JSON fields:
   `question_id`, `question`, `answer`, `haystack_sessions`,
   `answer_session_ids`, `question_date`.
2. Add a LongMemEval-format loader while keeping green/red sets for quick local
   regression checks.
3. Add a real LLM judge path for live eval.
4. Compare current replay runner behavior against akashic-agent's
   `eval/longmemeval` flow.

### P2: Improve Dataset Coverage

1. Add multi-user isolation cases.
2. Add long distractor histories.
3. Add temporal reasoning cases with dates.
4. Add abstention cases where memory is missing or insufficient.
5. Add noisy contradiction cases where assistant messages conflict with user
   facts.
6. Add more multi-turn personalization cases.

## Regular Verification Commands

Fast local checks:

```bash
python eval/rag_layer_smoke.py
python tests/test_invalidation_worker.py
python tests/test_consolidation_worker.py
python tests/test_consolidation_integration.py
python tests/test_pipeline_integration.py
python tests/test_memory_store.py
python tests/test_reasoner.py
```

Full replay on curated local data, using the real agent loop:

```bash
HTTP_PROXY=http://127.0.0.1:7897 HTTPS_PROXY=http://127.0.0.1:7897 python eval/replay_runner.py --set both --fresh --trace --output data/evaluation/results/replay_live_current.json
```

Live smoke with proxy:

```bash
HTTP_PROXY=http://127.0.0.1:7897 HTTPS_PROXY=http://127.0.0.1:7897 python eval/replay_runner.py --limit 3 --trace --output data/evaluation/results/replay_live_smoke.json
```

Legacy seeded eval:

```bash
python eval/runner.py --fresh --compare data/evaluation/baseline.json --output data/evaluation/results/current_eval.json
```

Use legacy seeded eval only for old regression continuity. Use replay eval for
any claim about the real passive reply chain.
