# HaoZhong Multi-Agent System

HaoZhong Multi-Agent System 是一个面向复杂问答场景的多Agent后端服务, 基于 FastAPI + LangGraph 构建, 采用分层编排架构, 将检索、生成、评估解耦为独立Agent, 并通过证据图驱动可解释输出。

## 项目定位

- 目标: 在 RAG + KG + Web 检索基础上, 提供可追踪、可回滚、可评估的问答流水线。
- 形态: HTTP API 服务, 支持同步响应与流式 SSE 输出。
- 重点: 证据优先, 评估闭环, HITL 人审可插拔。

## 核心能力

- Retrieval Agent: QDMR/子查询分解, 向量检索, Brave Web 检索, KG 检索与融合。
- Generation Agent: 基于 `plan_dag` 的槽位填充式生成, 从证据图提取并组织答案。
- Evaluation Agent: 结合打分与匹配机制进行输出质量评估, 支持回滚重试。
- Orchestration Gateway: 统一状态机路由, 明确控制流边界, 避免 worker 节点随意跳转。
- Observability: 集成 Langfuse tracing + Prometheus 指标 + Grafana 看板。

## 架构概览

### 分层架构

```text
Presentation Layer
  -> Orchestration Layer
    -> Worker Agents (Retrieval / Generation / Evaluation)
      -> Infra Layer (LLM / Postgres+pgvector / Neo4j / Brave / Langfuse)
```

### Pipeline 状态机

```text
explore -> freeze -> plan -> execute -> validate
                     ^                    |
                     |                    v
                 rollback <--------- quality_fail

validate_pass -> human_review(可选) -> done
```

说明:
- 只有 `gateway` 节点负责路由决策。
- Worker 节点只返回状态更新, 不直接控制下一个节点。

## 目录结构

```text
HaoZhong_project/
├── app/
│   ├── agents/                 # Retrieval / Generation / Evaluation agents
│   ├── orchestration/          # LangGraph graph, gateway, helper nodes
│   ├── core/                   # config, logging, metrics, limiter, shared types
│   ├── infra/                  # db, llm, neo4j, vector store, external services
│   ├── presentation/           # FastAPI routers, request/response schemas, static UI
│   └── prompts/                # system prompts
├── docs/                       # 需求/设计/接口/专项文档
├── scripts/                    # 启停脚本, 数据灌库脚本
├── tests/                      # unit + smoke tests
├── docker-compose.yml
├── Dockerfile
├── Makefile
└── pyproject.toml
```

## 快速开始

### 1) 环境要求

- Python >= 3.13
- `uv` 包管理工具
- PostgreSQL 16+ (建议带 pgvector)
- Neo4j 5.x

### 2) 安装依赖

```bash
uv sync
```

### 3) 配置环境变量

复制模板并按环境修改:

```bash
cp .env.example .env.development
cp .env.example .env.staging
cp .env.example .env.production
```

若使用 Docker Compose, 额外准备 `.env`:

```bash
cp .env.example .env
```

最低必配建议:
- `OPENAI_API_KEY` / `OPENAI_API_BASE`
- `POSTGRES_HOST` `POSTGRES_PORT` `POSTGRES_DB` `POSTGRES_USER` `POSTGRES_PASSWORD`
- `NEO4J_URI` `NEO4J_USER` `NEO4J_PASSWORD`
- `LANGFUSE_PUBLIC_KEY` `LANGFUSE_SECRET_KEY` (如需观测)

### 4) 本地启动

开发环境:

```bash
make dev
```

其他环境:

```bash
make staging
make prod
```

默认访问:
- API 文档: `http://localhost:8000/docs`
- 健康检查: `http://localhost:8000/health`
- Chat UI: `http://localhost:8000/chat`
- 指标: `http://localhost:8000/metrics`

## Docker 启动

仅 API + DB + Neo4j:

```bash
make docker-run-env ENV=development
```

完整栈(API + Postgres + Neo4j + Prometheus + Grafana):

```bash
make docker-compose-up ENV=development
```

常用命令:

```bash
make docker-compose-logs ENV=development
make docker-compose-down ENV=development
```

## API 说明

基础前缀: `/api/v1/agent`

| Method | Path | 说明 |
| --- | --- | --- |
| POST | `/chat` | 同步执行完整流水线并返回最终结果 |
| POST | `/chat/stream` | SSE 流式返回事件与 token |
| POST | `/review` | 人审动作(approve/reject/edit)继续流程 |
| GET | `/status/{session_id}` | 查询当前会话流水线状态 |

### 示例: 同步问答

```bash
curl -X POST "http://localhost:8000/api/v1/agent/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "知识图谱在RAG中的作用是什么?",
    "user_id": "demo-user",
    "max_iterations": 10
  }'
```

### 示例: 流式问答(SSE)

```bash
curl -N -X POST "http://localhost:8000/api/v1/agent/chat/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "请比较RAG和Graph RAG的差异",
    "user_id": "demo-user"
  }'
```

## 数据准备

项目提供示例灌库脚本, 可把 demo 文档写入 pgvector 与 Neo4j:

```bash
uv run python scripts/seed_data.py
```

建议先确认数据库连通性与 `.env` 中的连接配置。

## 测试与质量

```bash
make lint
uv run pytest tests/unit -q
uv run pytest tests/smoke -q
```

## 观测与运维

- Langfuse: 追踪 LLM 调用链路与会话元数据。
- Prometheus: 指标采集端点 `/metrics`。
- Grafana: 默认 `http://localhost:3000` (admin/admin)。

## 文档索引

- 总体需求: `docs/2026-02-06-v1.0-需求文档.md`
- 系统设计: `docs/2026-02-06-v1.0-系统设计文档.md`
- 详细设计: `docs/2026-02-06-v1.0-详细设计文档.md`
- 接口文档: `docs/2026-02-06-v1.0-接口文档.md`
- 检索链路专项: `docs/agents/retrieval-agent/2026-02-10-v1.1-Brave结果进入RAG与KG链路详解.md`

## 安全说明

- 不要提交任何真实密钥到仓库。
- 生产环境请配置独立的 `JWT_SECRET_KEY` 和数据库凭据。
- 公开部署前请检查 `.env*` 是否被正确忽略。

## License

见 `LICENSE`。
