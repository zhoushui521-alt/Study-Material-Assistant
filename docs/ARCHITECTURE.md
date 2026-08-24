# Study Material Assistant V2 · Architecture

本文以 `study-material-v2-stage5-part4` 为 Stage 5.4 起点，并同步当前进行中的真实实现。
它仍是具备容器部署定义的单实例原型，不代表生产环境已经部署。

## 1. Frontend

当前前端是由 FastAPI 同源提供的原生 Web 页面：

- `web/index.html`
- `web/static/styles.css`
- `web/static/app.js`

没有 React、Next.js、前端构建系统或独立 Node 服务。FastAPI 把 `/static` 挂载到静态资源目录，`GET /` 返回主页面。页面负责注册登录、资料管理、问答、Tutor、Citation 展示和 Document Job 轮询；RAG、Prompt 与权限判断都留在后端。

## 2. Backend

后端入口是 `app/api.py`，框架为 FastAPI。当前主要职责包括：

- 认证：注册、登录、退出、当前用户；
- 学习数据：Session 与历史；
- 资料摄取：本地文件与公开网页预览、暂存、确认索引、删除；
- AI 能力：`/api/ask`、受限 Agent、`/api/tutor/chat`、Study Workflow；
- 异步任务：创建 Document Job、查询任务状态；
- 运行边界：输入校验、单进程并发/频率/费用保护、错误映射与安全日志。

认证后的业务入口通过 `current_user` 取得用户身份。资料、向量库、Job、Tutor Session 和 Study Workflow 都按该身份限定 owner；客户端提交的资源 ID 不能代替后端所有权检查。

## 3. RAG Pipeline

当前正式问答链路为：

```text
用户 Workspace 中的 Document
  ↓
Parser（TXT / Markdown / DOCX / 带文字层 PDF / 已确认网页 Markdown）
  ↓
Parser output normalization / metadata normalization
  ↓
fixed-character-180-v1 Chunker
  ↓
Embedding
  ↓
用户独立 Chroma Vector Store + Index Manifest
  ↓
Dense Vector Top 10
  ↓
relevance >= 0.25
  ↓
80% vector relevance + 20% keyword coverage rerank
  ↓
Top 3 Seed + 同源 ±2 Adjacent Expansion（最多 8 块）
  ↓
EvidenceScoreContextSelector
  ↓
请求内 Evidence Map（S1、S2……）
  ↓
LCEL Prompt → ChatModel
  ↓
服务端验证回答中的 Evidence ID
  ↓
回填权威 Citation metadata → Answer
```

三个概念不能混用：

- **Retrieval Candidate**：检索或扩展得到的候选 Chunk。
- **Evidence**：真正进入本次模型上下文、带稳定资料定位信息的结构化证据。
- **Citation**：模型回答引用的 Evidence ID 经服务端确认存在后，由服务端回填的关联结果。

当前 Citation 校验能拒绝不存在于本次 Evidence Map 的 ID，但不能仅凭 ID 存在证明 Evidence 支持某个具体 Claim。Citation Support / Accuracy 仍需要独立标注、人工核验或经授权的 Judge。

### 正式链路与实验能力

- 正式 Retrieval：Stage 2 Dense + keyword coverage baseline。
- 正式 Context Construction：Stage 3.4 `EvidenceScoreContextSelector`。
- 未正式接入：BM25 + Dense + RRF、Cross-Encoder Reranker、Structure-aware Chunking。

## 4. Tutor Agent

`app/tutor_workflow.py` 实现一个受控的单 Tutor LangGraph Workflow：

```text
Message
  → deterministic intent classification
  → StateGraph router
     ├─ QA / Explanation → KnowledgeRetrievalTool → RAGService.ask()
     ├─ Quiz → KnowledgeRetrievalTool → QuizGeneratorTool
     ├─ Summary → Session Summary 或 Retrieval → LearningSummaryTool
     └─ Learning Plan → Retrieval → deterministic learning loop
  → Tutor Response
```

意图分类是确定性规则，不会为了路由额外调用模型。`KnowledgeRetrievalTool` 复用正式 RAG，不复制检索或 Citation 逻辑。Quiz 与 Summary 的结构化生成按路径调用 ChatModel。

Tutor 的业务历史、Session topic 与学习记录存放在 SQLite；LangGraph 的短期执行 checkpoint 使用 `AsyncSqliteSaver`。这属于持久化学习状态，不等同于无限对话长期记忆或用户画像。

## 5. Storage

### SQLite

- `data/learning/learning.sqlite3`：用户、认证 Session、学习 Session、Conversation、Learning Record，以及 Tutor 的 LangGraph checkpoint。
- `data/jobs/document_jobs.sqlite3`：Document Job 状态、进度、所有权、安全错误摘要与完成结果；不保存文件正文、Chunk、Prompt、模型输出或密钥。
- `data/study_workflows/checkpoints.sqlite3`：历史 Study Workflow 的 LangGraph checkpoint。

学习数据库与 Job 数据库使用 `aiosqlite`、WAL 和 `busy_timeout=5000`。当前方案适合本地单实例，不是高并发、多实例或生产灾备方案。

### User Workspace

每个认证用户使用独立目录：

```text
data/user_workspaces/<user_uuid>/
  ├─ documents/
  ├─ pending_uploads/
  ├─ pending_deletions/
  └─ vector_store/
```

资料文件与 Chroma 在物理目录上按用户隔离。当前设计避免只依赖容易遗漏的 metadata filter；代价是用户增多时会增加 Chroma 客户端和本地目录数量。

### Vector Store

Vector Store 是本地持久化 Chroma。每用户 `vector_store/` 中的 Index Manifest 记录 Embedding、集合、距离度量、Parser/Chunker 与 Metadata Schema 的兼容身份。没有 Manifest 的旧全局索引只允许读取；写入、删除或同步必须先显式迁移，不能自动归属给某个认证用户。

## 6. Async Processing

异步化从用户确认真实索引费用后开始；文件校验、解析预览和预计批次数仍在请求内完成。

```text
Stage / preview
  → confirm_api_cost=true
  → SQLite 创建 pending Job
  → HTTP 202 Accepted + job_id
  → 单进程 asyncio Worker 串行领取
  → processing
  → asyncio.to_thread() 复用 MaterialManager commit 路径
  → Parser / Chunker / Embedding / Chroma sync
  → completed 或 failed
  → GET /api/jobs/{job_id} 查询
```

Worker 复用现有费用保护、Manifest 检查、增量同步、回滚与 RAG 缓存失效机制。服务重启后继续处理 `pending`；中断时已经是 `processing` 的任务会标记为 `failed`，不会盲目自动重试并可能重复付费。

当前没有 Redis、Celery、RabbitMQ、Kafka、分布式队列、多 Worker 抢占、自动重试、取消、优先级或精确剩余时间。

## 7. Service Runtime 与 Docker Compose

Stage 5.3 的部署候选保持单进程、单实例：

```text
Browser
  → app container / FastAPI :8000
       ├─ 同源 web/
       ├─ RAG / Tutor / Material / Study Workflow Service
       └─ in-process Document Worker
  → /app/data named volume
       ├─ learning SQLite
       ├─ document job SQLite
       ├─ workflow checkpoint SQLite
       ├─ user_workspaces/<uuid>/{documents,pending_uploads,pending_deletions,vector_store}
       ├─ request logs
       └─ Crawl4AI runtime
```

`app.config.Settings` 统一解析 `APP_HOST`、`APP_PORT`、`APP_DATA_DIR` 和百炼模型配置。
`app.server` 是本地与容器共用启动入口。Compose 在容器内使用 `/app/data`，宿主端口由
`ZHIXING_PORT` 控制；`.env` 与本地 `data/` 不进入镜像。

没有独立 frontend：当前前端是 FastAPI 同源静态资源。没有独立 worker：当前 Job 领取、
OperationGuard、SQLite 与 Chroma 都是单进程一致性边界。没有 database 服务：当前仍使用
SQLite。Compose network 只为部署定义清楚边界，不代表存在微服务通信。

`GET /health` 作为容器 liveness check，只检查 API 可响应，不初始化 RAG、Chroma 或模型。
当前机器已完成本地 startup、Health、首页和静态资源 HTTP 200 验证；由于没有 Docker CLI，
镜像构建、Compose 启动、volume 重启恢复和容器内 HTTP 尚未验证。

## 8. Observability Layer（Stage 5.4 In Progress）

Stage 5.4.1 已建立标准库结构化日志基础：

```text
HTTP Request
  → FastAPI Middleware
  → app.observability
       ├─ JSON Console Event
       └─ RequestHistoryWriter（仅完成事件的白名单 HTTP 元数据）
```

每条控制台事件固定包含 `time`、`level`、`service`、`request_id`、`user_id`、
`event` 和 `duration_ms`。HTTP Middleware 已记录开始与结束、状态码和耗时；服务启动、
关闭及清理错误也使用同一 Logger。

`RequestHistoryWriter` 不是完整 Trace Store：它继续轮转保存最小 HTTP 完成记录，且拒绝
问题、回答、文件正文、Prompt、密钥、任意未知路径与 URL 查询参数。当前控制台 Logger 同样
只接纳白名单详情字段。

尚未实现：Request Context 在 RAG/LLM/Document Job 内部的贯穿、阶段耗时事件、进程内
Metrics 聚合与 Observability HTTP 接口。这些属于 5.4.2～5.4.4，不能由 5.4.1 外推。


## 9. 当前部署边界

当前运行拓扑是本地 FastAPI 单实例 + 本地文件系统 + 本地 SQLite + 本地 Chroma。已有自动化和历史本地运行证据不能证明：

- 多实例 Session / Job / Chroma 一致性；
- 并发与负载能力；
- 数据库加密、备份恢复和灾难恢复；
- 公开网络下的 HTTPS、反向代理、登录防护和渗透安全；
- 生产可观测性与 SLA。
