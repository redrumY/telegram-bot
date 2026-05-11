# Docker 部署指南

## 快速开始

1. 配置 `.env` 文件
2. 启动服务：
   ```bash
   docker-compose up -d
   ```

## 停止服务

```bash
docker-compose down
```

## 查看日志

```bash
docker-compose logs -f
```

## 部署到云平台

### Railway

1. 推送项目到 GitHub
2. 在 Railway 创建新项目，选择 GitHub 仓库
3. 添加环境变量（在 Dashboard → Variables）
4. 部署自动开始

### Render

1. 推送到 GitHub
2. 在 Render 创建 Web Service
3. 选择 Python 类型
4. 添加环境变量
5. 部署

## 注意事项

- `data/` 目录用于持久化数据库
- 容器重启后数据库不会丢失
- 健康检查每 30 秒一次
