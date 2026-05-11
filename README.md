# Telegram AI Bot

一个基于记忆驱动的 Telegram AI 机器人，使用 DeepSeek LLM 和向量记忆。

## 功能特性

- 💾 **持久化记忆**：使用向量数据库存储和检索对话
- 🧠 **智能检索**：基于语义相似度和关键词混合检索
- 🔄 **对话管理**：支持多轮对话，记忆会话历史
- 🤖 **工具调用**：支持 memorize 工具保存重要信息

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
│   ├── core/          # 类型定义、EventBus
│   ├── pipeline/       # 流水线各阶段
│   └── reasoner.py     # LLM 调用器
├── channels/           # 消息通道
│   └── telegram/     # Telegram 集成
├── memory/             # 记忆管理
│   ├── embedder.py     # 向量生成
│   └── store.py        # 记忆存储和检索
├── persistence/         # 数据持久化
│   └── database.py     # SQLite + sqlite-vec
├── config/             # 配置管理
│   └── settings.py     # Pydantic 设置
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
