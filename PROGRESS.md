# Project Progress

Last updated: 2026-06-17

## 文档边界

- `AGENTS.md`: 给 Codex/agent 看的协作规约、代码规范、关键命令。
- `README.md`: 给人看的项目介绍、快速启动、项目结构。
- `DOCKER.md`: Docker 启动、停止、日志、部署相关操作。
- `PROGRESS.md`: 当前阶段进展、验收结果、未完成事项。

插件系统、记忆系统、pipeline 规范仍以 `README.md` 和 `AGENTS.md` 为主；本文只记录这次 Web 后端拆分、PostgreSQL、队列和 worker 架构升级的进展，避免和 agent/plugin/memory 线混在一起。

## 当前目标

把原先 Telegram bot / 被动回复链条升级为可面试展示的 Web 后端架构：

```text
Frontend
  |
FastAPI API 层
  |
PostgreSQL + pgvector
  |
conversation_turns 队列表
  |
Python Worker Pool
  |
AgentService / PassiveTurnPipeline
  |
LLM / Tools / Memory
```

核心叙事：

- FastAPI 负责 HTTP API、SSE、前端静态页分发。
- PostgreSQL 负责会话、消息、turn 状态、长期记忆持久化。
- pgvector 负责向量记忆检索。
- `conversation_turns` 表承担轻量任务队列职责。
- worker 独立消费 pending turn，避免 API 被 LLM 长耗时阻塞。
- `AgentService` 封装原有 `PassiveTurnPipeline`，让 Telegram/Web 都能复用同一条 agent 链路。

## 已完成

### 1. Agent runtime / service 拆分

- 新增 `agent/runtime.py`，把原 `main.py` 中构建 pipeline、memory、tools 的逻辑封装为 `AgentRuntime`。
- 新增 `agent/service.py`，通过 `AgentService.chat(...)` 复用 `PassiveTurnPipeline`。
- `main.py` 改为使用 `AgentRuntime`，保留 Telegram bot 入口。
- 为同一 `user_id + session_id` 增加会话级锁，避免同一会话多条消息并发写乱上下文。

### 2. FastAPI Web API

- 新增 `web_backend/api.py` 和 `web_backend/main.py`。
- 已提供接口：
  - `GET /`
  - `GET /api/health`
  - `POST /api/chat`
  - `GET /api/turns/{turn_id}`
  - `GET /api/turns/{turn_id}/events`
  - `GET /api/sessions/{session_id}/messages`
- `POST /api/chat` 现在创建 pending turn，不直接阻塞等待 LLM 完成。

### 3. 最小 Web 前端

- 新增 `web_backend/static/index.html`
- 新增 `web_backend/static/styles.css`
- 新增 `web_backend/static/app.js`
- 支持发送消息、创建 session、显示 pending/processing/done/failed 状态。
- 支持 SSE，SSE 不可用时可退化为轮询。

### 4. turn queue / worker

- 新增 `persistence/turn_store.py`，定义 `TurnStore`、`Turn`、状态常量。
- 支持状态：
  - `pending`
  - `processing`
  - `done`
  - `failed`
- 新增 `worker/turn_worker.py`，负责 claim pending turn、调用 `AgentService`、写回结果。
- 新增 `worker/main.py`，支持 `WORKER_CONCURRENCY` 并发 worker。

### 5. PostgreSQL + pgvector

- 新增 `persistence/postgres.py`，初始化 PostgreSQL schema。
- 已建表：
  - `web_users`
  - `conversation_sessions`
  - `conversation_messages`
  - `conversation_turns`
  - `memory_items`
  - `memory_replacements`
- `memory_items.embedding` 使用 `vector(1024)`。
- 新增 `persistence/postgres_turn_store.py`，使用 `FOR UPDATE SKIP LOCKED` claim pending turn。
- 新增 `memory/postgres_store.py`，支持 pgvector 语义检索和关键词检索。
- 新增 `PostgresSessionStore`，把会话历史同步写入 `conversation_sessions` / `conversation_messages`。

### 6. Docker Compose

- `docker-compose.yml` 已包含：
  - `postgres`: `pgvector/pgvector:pg16`
  - `api`: FastAPI / Uvicorn
  - `worker`: Python worker pool
  - `bot`: Telegram bot
- `postgres` 使用 volume `postgres-data` 持久化数据。
- `api` 暴露 `8000:8000`。
- `postgres` 暴露 `5432:5432`。
- `api` / `worker` / `bot` 都通过 `depends_on: service_healthy` 等待 Postgres ready。

### 7. 已修复的集成 bug

- Docker 真实链路中发现 `pgvector` 查询失败：

```text
cannot cast type smallint to vector
```

- 原因：SQL 里 `embedding <=> %s::vector` 的参数出现在 WHERE 前，但参数列表先放了 `user_id`，导致 Postgres 把 `1001` 当成 vector 转换。
- 已修复 `memory/postgres_store.py` 的参数顺序。
- 新增 `tests/test_postgres_memory_store.py` 防止回归。

## 已验收

### 本地/单元验收

- `memory/postgres_store.py` py_compile 通过。
- `tests/test_postgres_memory_store.py` 通过。
- `tests/test_turn_worker.py` 通过。
- Docker 容器内测试结果：

```text
3 passed
```

### Docker 链路验收

Docker Desktop 启动后，已执行：

```bash
docker compose up -d postgres api worker
```

验收结果：

- `postgres` 状态 healthy。
- `api` 状态 Up，`GET /api/health` 返回 `{"status":"ok"}`。
- `worker` 状态 Up，并输出 `PostgreSQL schema ready`。
- PostgreSQL 已安装 `vector` extension。
- 数据库表已创建。
- 真实调用 `POST /api/chat` 后生成 pending turn。
- worker 成功消费 turn 并写回 `done + answer`。
- SSE `GET /api/turns/{turn_id}/events` 能返回 `done` 事件。
- `GET /api/sessions/{session_id}/messages?user_id=...` 能读取 user/assistant 消息历史。

## 当前可运行命令

启动 Web/API/Worker/Postgres：

```bash
PATH=/Applications/Docker.app/Contents/Resources/bin:$PATH docker compose up -d postgres api worker
```

查看状态：

```bash
PATH=/Applications/Docker.app/Contents/Resources/bin:$PATH docker compose ps
curl http://127.0.0.1:8000/api/health
```

打开 Web：

```text
http://127.0.0.1:8000
```

查看日志：

```bash
PATH=/Applications/Docker.app/Contents/Resources/bin:$PATH docker compose logs -f api worker postgres
```

停止：

```bash
PATH=/Applications/Docker.app/Contents/Resources/bin:$PATH docker compose stop api worker postgres
```

## 还未完成

- 登录系统未做；后续可做邮箱密码登录和 user/session 绑定。
- 前端只展示最终回答；思考过程/trace 展示位置已预留但未完整实现。
- worker 已支持并发数配置，但还未做 100-200 并发压测。
- 目前用 PostgreSQL 表做轻量队列；后续更大规模可引入 Redis Queue / Celery / Kafka。
- retry/backoff/timeout/cancel 还需要完善。
- 数据库 migration 还未接 Alembic，目前 schema 由 `init_postgres()` 初始化。
- API 鉴权、限流、CORS 白名单、HTTPS/domain 未做。
- 生产部署未做；当前已完成本地 Docker Compose 跑通。
- Telegram 与 Web 的用户身份映射还未统一设计。
- 观测能力还可以增强：结构化 trace、request_id、turn 执行耗时、错误面板。
- `DOCKER.md` 仍偏旧，后续可以补充 `api/worker/postgres` 的详细说明。

## 已推送提交

当前分支：

```text
codex/web-agent-architecture
```

最近关键提交：

```text
ca8cba0 Fix pgvector query parameter order
438fc82 Add Postgres pgvector backend for web workers
317a6ff Add SSE turn status stream
2652115 Add queued web turns and worker
a836f3a Add minimal web chat frontend
7c09442 Add FastAPI web chat API
ee4e9c2 Refactor agent runtime for reusable chat service
```

## 面试表达草稿

可以这样概括当前进展：

> 我把原来的 Telegram 被动回复链路抽成了可复用的 AgentService，并用 FastAPI 暴露 Web API。用户请求进入后不再直接阻塞等待 LLM，而是写入 PostgreSQL 的 conversation_turns 表，状态为 pending。独立的 Python worker pool 使用 FOR UPDATE SKIP LOCKED 消费 pending turn，调用 PassiveTurnPipeline，完成后写回 done/failed 和 answer/error。会话历史、消息、turn 状态和 pgvector 长期记忆都持久化在 PostgreSQL 中，前端通过 SSE 或轮询拿状态。Docker Compose 负责在本地同时拉起 Postgres、API 和 worker，保证开发和演示环境一致。
