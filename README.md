# Telegram AI Bot

一个基于记忆驱动的 Telegram AI 机器人，使用 DeepSeek LLM 和向量记忆。

## 功能特性

- 💾 **持久化记忆**：使用向量数据库存储和检索对话
- 🧠 **智能检索**：基于语义相似度和关键词混合检索
- 🔄 **对话管理**：支持多轮对话，记忆会话历史
- 🤖 **工具调用**：通过 ToolRegistry / ToolExecutor 统一调度内置记忆工具和插件工具
- 🧩 **插件生命周期**：支持 Akashic 风格 PhaseModule、slot export、prompt_render 插入点

## 技术栈

- **语言**：Python 3.11+
- **LLM**：DeepSeek API
- **向量**：阿里云 DashScope
- **数据库**：SQLite + sqlite-vec
- **Bot 框架**：python-telegram-bot 20.7+
- **依赖管理**：Poetry

## 快速开始

### 本地运行

1. 安装依赖：
   ```bash
   poetry install
   ```

2. 配置环境变量（复制 `.env.example` 为 `.env` 并填入密钥）

3. 启动：
   ```bash
   python main.py
   ```

### Docker 部署

```bash
docker-compose up -d
```

详细指南见 [DOCKER.md](DOCKER.md)

## 项目结构

```
telegram-bot-mvp/
├── agent/               # Agent 核心逻辑
│   ├── core/            # 类型定义、EventBus、PromptBlock
│   ├── lifecycle/       # PhaseFrame / PhaseModule / slot export
│   ├── pipeline/        # 被动回复流水线与各阶段
│   ├── plugins/         # 插件管理器、上下文、装饰器
│   ├── prompting/       # Prompt 渲染与 section 组装
│   ├── tool_hooks/      # 工具调用前置 hook 链
│   └── tools/           # ToolRegistry、ToolExecutor、内置工具注册
├── channels/           # 消息通道
│   └── telegram/        # Telegram 集成
├── memory/             # 记忆管理
│   ├── embedder.py      # 向量生成
│   ├── hyde_enhancer.py # HyDE 检索增强
│   └── store.py         # 记忆存储和检索
├── persistence/         # 数据持久化
│   ├── database.py      # SQLite + sqlite-vec
│   └── session_store.py # 原始消息与会话游标
├── proactive_v2/        # 主动推送链路 scaffold
├── config/             # 配置管理
│   └── settings.py      # Pydantic 设置
├── tests/              # 单元测试
├── main.py             # 入口文件
├── Dockerfile          # Docker 镜像构建
└── docker-compose.yml  # Docker 编排
```

## 获取 API 密钥

| 服务 | 地址 |
|------|------|
| DeepSeek | https://platform.deepseek.com/ |
| 阿里云 DashScope | https://dashscope.aliyuncs.com/ |

## 许可证

MIT License
