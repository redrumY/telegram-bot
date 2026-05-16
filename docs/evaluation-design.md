# RAG Eval Design

## Current Goal

This project should evaluate the memory layer as an agent capability, not as a
single-hop retrieval benchmark.

The target flow follows akashic-agent's LongMemEval path:

```text
haystack conversations
-> replay into session
-> consolidation writes long-term memory
-> invalidation retires stale memory
-> QA through the normal passive reply chain
-> agent may use recall_memory / fetch_messages
-> score answer and inspect trace
```

Replay eval must not use a mock execution layer. Each eval case should replay
curated historical conversations into `SessionStore`, run real consolidation
and invalidation, then ask the question through the real `PassiveTurnPipeline`.
Some filenames or old helpers may still contain `mock` for historical reasons,
but that means "local curated eval corpus", not a mock reasoner, mock embedder,
or seeded shortcut.

## Why Not Only Top-K Retrieval

Top-k retrieval only answers whether a memory can be found. A conversation bot
also needs to decide when to call tools, whether to fetch source evidence, how
to resolve old/new conflicts, and how to turn recalled snippets into a useful
answer. Those are agent-loop abilities, so the eval must include the loop.

## Mock Layer Removal Status

Removed from replay eval:

- mock embedder
- mock reasoner
- mock consolidation extraction
- mock invalidation topic extraction and supersede checker
- `--mock` CLI switch
- direct memory seeding as the primary replay path

Migrated into the real agent-loop prompt/tool layer:

- `recall_memory` must be used first in benchmark mode.
- historical questions should use `include_superseded=true`.
- recall queries should be rewritten from raw questions into user-memory
  retrieval statements such as `用户的饮品偏好以及后来的更新`.
- source-backed answers should use `fetch_messages(source_ref=...)`.

Still missing as code-level guarantees:

- If the model skips `recall_memory`, benchmark mode should force a recall step
  or force a second tool-use round.
- If `recall_memory` returns source refs and the case requires evidence,
  benchmark mode should ensure `fetch_messages` happens before final answer.
- If the question is historical, benchmark mode should enforce
  `include_superseded=true`, not only ask the model to do it.
- Query rewrite is currently prompt-only. A deterministic benchmark requery
  helper would make eval results easier to reproduce.

## Case Distribution

The current local curated suite has 13 green cases and 5 red cases.

Green set:

| Type | Count | What It Tests |
| --- | ---: | --- |
| single_session_fact | 3 | Can the bot recall a fact or exact detail from one session? |
| cross_session_preference | 2 | Can the bot apply a prior preference in a later request? |
| knowledge_update | 4 | Can the bot use updated information instead of stale memory? |
| user_identity | 3 | Can the bot remember stable user identity facts? |
| multi_turn_context | 1 | Can the bot use remembered user context to personalize an answer? |

Red set:

| Type | Count | Why It Exists |
| --- | ---: | --- |
| user_identity | 1 | Historically weak semantic match: "工作" vs "程序员". |
| multi_turn_context | 1 | Checks whether recalled context is actually used in generation. |
| knowledge_update | 2 | Stress tests old/new conflict and synonym wording. |
| cross_session_preference | 1 | Tests broad wording such as "喝的" instead of exact keywords. |

## Why These Cases

- `single_session_fact` is the minimum viable memory check.
- `cross_session_preference` tests whether memory affects later recommendations.
- `knowledge_update` is the most important failure mode for long-term memory:
  old facts and new facts can coexist, and the agent must choose correctly.
- `user_identity` tests stable profile memory.
- `multi_turn_context` tests whether the final answer uses recalled facts,
  not just whether retrieval succeeded.

## What These Cases Do Not Cover Yet

- multi-user isolation
- long distractor histories
- time-sensitive answers with dates
- source verification quality via `fetch_messages`
- explicit invalidation end-to-end in the QA score
- noisy contradictory assistant messages

## How To Read Results

Use three layers:

```text
smoke eval       -> did the lifecycle break?
legacy eval      -> historical seeded regression check only
replay eval      -> can the full agent loop solve cases after haystack replay?
```

Commands:

```bash
python eval/rag_layer_smoke.py
HTTP_PROXY=http://127.0.0.1:7897 HTTPS_PROXY=http://127.0.0.1:7897 python eval/replay_runner.py --fresh --trace --set both
```

Use replay eval for benchmark decisions. `eval/runner.py` is a legacy seeded
shortcut and should not be treated as evidence that the real passive reply
chain works.

For replay eval, inspect each result's `ingest` block:

- `selection_mode=context_sessions` means the case used explicit haystack ids.
- `selection_mode=all_conversations_fallback` means the case metadata did not
  match the curated corpus and the runner had to replay everything. This should
  be avoided for benchmark cases.
- `consolidated_memories` should be greater than zero for cases with enough
  haystack messages or finalized tail history.
- `invalidated_memories` should be greater than zero only when the haystack has
  explicit correction / forgetting signals.

Then inspect `tool_calls`:

- `recall_memory` should appear for every benchmark question.
- `fetch_messages` should appear for exact facts, identity facts, and update
  cases that need source evidence.
- `include_superseded=true` should appear for questions asking about old values,
  all values, or a change history.

Recent replay runs show that query rewrite reaches the real agent loop:
`recall_memory` can be called with queries such as `用户的饮品偏好以及后来的更新`.
The next missing piece is making tool policy deterministic: prompt guidance can
still be skipped by the model, so `recall_memory`, `fetch_messages`, and
`include_superseded` need code-level benchmark guards.
