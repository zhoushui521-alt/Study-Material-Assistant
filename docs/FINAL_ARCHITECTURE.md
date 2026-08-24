# 知行（Study Material Assistant）· Final Architecture

> Stage 6.1 冻结展示快照。基线：`9ac680a`，代码主链 checkpoint：`d54a90b`。
> 本文用于作品展示与面试讲解；持续变化的状态仍以 [PROJECT_STATUS](PROJECT_STATUS.md) 为唯一入口，
> 详细实现边界以 [ARCHITECTURE](ARCHITECTURE.md) 和代码为准。

## 1. 架构结论

知行当前是一个本地单实例、单 FastAPI 进程的 AI 学习应用原型。它把 RAG、证据与引用、
Tutor 学习工作流、用户隔离、异步资料任务、模型路由和基础可观测性组织在同一个可测试系统中。

“Production Readiness”在本项目中表示：已有能力具备明确契约、持久化、错误边界、测试证据、
部署定义和运维说明；它不表示已经通过公开部署、并发负载、容灾、渗透、真实 Provider
兼容矩阵或生产 SLA 验收。

## 2. 系统整体架构

```text
Browser
  └─ Native HTML / CSS / JavaScript
       │
       ▼
FastAPI Application (single process)
  ├─ Auth & User Scope
  ├─ API / Validation / Error Mapping / OperationGuard
  │
  ├─ Application Services
  │    ├─ MaterialManager
  │    ├─ DocumentJobService + one asyncio worker
  │    ├─ RAGService + LCEL chain
  │    ├─ AgentService (bounded tools)
  │    ├─ TutorWorkflowService (LangGraph)
  │    ├─ StudyWorkflowService (LangGraph + approval interrupt)
  │    └─ ModelGateway
  │
  └─ Local Infrastructure
       ├─ SQLite: user/session/learning/Tutor checkpoint
       ├─ SQLite: document jobs
       ├─ SQLite: Study Workflow checkpoint
       ├─ SQLite: encrypted BYOK metadata
       ├─ per-user filesystem workspaces
       ├─ per-user Chroma vector stores + Index Manifest
       └─ JSONL logs + in-process runtime metrics

External boundaries
  ├─ Embedding Provider: index/query embedding
  ├─ Chat Provider: selected by ModelGateway
  └─ public web page: preview only after URL safety checks
```

这不是微服务架构：Frontend、API、Worker、RAG、Workflow 和 Gateway 都在一个应用进程中；
SQLite、Chroma 与文件系统都在本地数据目录。选择这一边界是为了匹配当前本地单实例需求，
避免在没有多实例和吞吐证据时引入 Redis、Celery、外部数据库或独立网关。

## 3. 分层与职责

| 层 | 当前实现 | 主要职责 | 当前边界 |
| --- | --- | --- | --- |
| Frontend | `web/index.html`、`web/static/` | 登录、资料管理、问答、Citation、Tutor、任务状态与学习记录交互 | 原生静态页面，由 FastAPI 托管 |
| API | `app/api.py` | Pydantic 校验、认证依赖、用户作用域、HTTP 状态码、服务装配与缓存失效 | 单进程 FastAPI |
| Material | `app/material_ingestion.py`、`app/web_materials.py` | 上传/网页预览、解析、暂存、费用估算、提交、删除与回滚 | 暂存预解析同步；正式索引由 Job 执行 |
| Job | `app/document_jobs.py` | 持久化 Job、串行消费、启动恢复、结果/安全错误落库 | 单 Worker、无自动重试和多实例抢占 |
| RAG | `app/rag_service.py`、`app/hybrid_search.py`、`app/langchain_rag.py` | 检索、Context Construction、Evidence Map、模型调用和 Citation ID 校验 | 正式链路不是 BM25/Reranker |
| Learning | `app/agent_service.py`、`app/tutor_workflow.py`、`app/study_workflow.py` | 受限工具编排、Tutor 路由、学习计划、审批与进度恢复 | 单 Tutor；没有 Multi-Agent |
| Model | `app/model_gateway/` | system/BYOK 路由、Provider Adapter、凭据密文存储和安全错误 | 进程内网关；无自动故障切换 |
| Persistence | `app/learning_data.py`、SQLite、Chroma、filesystem | 用户、Session、学习记录、Checkpoint、Job、凭据、资料与向量 | 本地单实例，无生产备份/HA |
| Observability | `app/observability.py`、`app/metrics.py`、`app/request_history.py` | Request ID、结构化事件、运行聚合指标和安全请求历史 | 进程指标重启清零，无外部 Collector |

## 4. 资料上传与索引数据流

```text
Authenticated User
  → POST stage / stage-batch / web preview
  → user_workspace_paths(user_id)
  → MaterialManager.stage_upload()
       ├─ filename/content-type/signature/resource-limit checks
       ├─ Parser → normalized document units
       ├─ fixed-character-180-v1 chunks
       ├─ local Embedding batch estimate
       └─ pending_uploads/<upload_id> + metadata
  → user explicitly confirms API cost
  → POST index → HTTP 202 + job_id
  → DocumentJobStore(status=pending)
  → single asyncio worker claims job
  → OperationGuard acquires index lease and reserves actual units
  → MaterialManager.commit_staged[_batch]()
       ├─ revalidate staged state
       ├─ synchronize documents
       ├─ Embedding
       ├─ Index Manifest compatibility/write protection
       └─ per-user Chroma update
  → job completed/failed persisted
  → invalidate only that user's cached RAG/Agent/Tutor services
```

关键理解：

- 暂存阶段解析并估算，不创建 Embedding Client，也不打开 Chroma，因此可作为零外部调用的预检；
- 正式提交复用既有 `MaterialManager`、Manifest、费用保护和回滚路径，Job 不复制索引逻辑；
- Job 数据库只保存状态、进度、所有权、安全错误摘要和结果摘要，不保存正文、Chunk、Prompt 或模型输出；
- 进程重启时，遗留 `processing` Job 会转为明确失败，`pending` Job 继续等待单 Worker；
- 资料删除采用“先隔离文件、再定向删除 Chroma、失败则恢复文件”的补偿路径，但无法提供分布式事务保证。

## 5. 正式 RAG 问答数据流

```text
POST /api/ask
  → authenticate current_user
  → validate bounded question
  → OperationGuard(ask)
  → get/create user-scoped RAGService
  → Query Embedding
  → Chroma Dense Top 10 candidates
  → relevance score >= 0.25
  → 0.8 × vector score + 0.2 × keyword coverage
  → Top 3 seed chunks
  → same-source adjacent expansion (window ±2, context limit 8)
  → EvidenceScoreContextSelector
       ├─ keep every seed
       └─ keep at most one selected adjacent chunk per seed
  → request-local Evidence Map (S1, S2, ...)
  → LCEL Prompt
  → ModelGateway → Provider Adapter → BaseChatModel
  → parse answer
  → validate every returned Evidence ID
  → resolve valid IDs to Citation metadata
  → Answer + retrieval candidate sources + validated citations
```

这里必须区分三个概念：

- Retrieval Candidate：Retriever 返回的候选 Chunk；
- Evidence：进入当前请求 Evidence Map、具有稳定资料与 Chunk 定位的结构化对象；
- Citation：回答中出现的 Evidence ID 经服务端验证后解析出的引用记录。

Citation ID 有效只证明“引用指向本次 Evidence Map 中存在的证据”，不自动证明每个回答主张都被
该证据支持。当前没有稳定的 Claim-level Citation Accuracy / Faithfulness 生产验收。

### 正式与实验能力

| 能力 | 状态 | 原因 |
| --- | --- | --- |
| Dense + keyword coverage rerank | 正式 | Stage 2 baseline |
| `EvidenceScoreContextSelector` | 正式 | 历史 Trace 回放中 Context Precision 提升且 Final Context Recall 保持 |
| BM25 + Dense + RRF | 实验保留 | Recall@5 改善，但 Recall@3 不变、MRR 回退 |
| Cross-Encoder Reranker | 实验保留 | 排序指标改善，但 Context Precision 回退且 CPU 延迟显著增加 |
| Structure-aware Chunking | 实验保留 | 部分前排指标改善，但 Recall@3、Final Context Recall 和 Context Precision 回退 |

## 6. Model Gateway 数据流

```text
RAG / Agent / Tutor
  → user-scoped RAGService
  → ModelGateway.route(user_id)
       ├─ BYOK exists
       │    → read encrypted credential → Fernet decrypt
       │    → no silent system-key fallback on decrypt/auth failure
       └─ no BYOK
            → server default provider/model/key
  → ProviderAdapterRegistry
  → OpenAICompatibleProviderAdapter
  → BaseChatModel
  → provider call
```

当前已注册 `qwen`、`deepseek`、`openai`、`openai_compatible`，但“已注册并通过
Fake/Mock 构造测试”不等于真实账号、模型名、响应 metadata 或兼容性已验证。
Claude/Anthropic 原生 Adapter 未实现。

用户凭据 API 只允许 Provider、模型和 Key；不允许用户提交任意 `base_url`。Key 以 Fernet
密文存入独立 SQLite，主密钥只来自环境。没有 KMS/HSM、在线轮换、团队凭据、价格表、预算
阻断或 Provider 自动故障切换。

## 7. Agent 与 LangGraph Workflow

### 7.1 Bounded AgentService

`AgentService` 使用 LangChain Agent，但只暴露三个受限工具：

```text
answer_from_materials  → reuse RAGService.ask()
list_available_materials → list metadata only
preview_web_material   → separate authorization; preview never indexes
```

单次运行受到输入长度、工具次数、总时限、费用确认、网页预览独立授权和
`OperationGuard` 约束。它不是 Multi-Agent，也不能任意读文件、写索引、删除资料或访问任意网络。

### 7.2 Tutor LangGraph

持久化状态包括 `user_id`、`session_id`、`intent`、`topic`、有界 conversation、
Evidence/Citation、Quiz/Summary、工具记录、route trace 与安全错误。

```text
START
  → classify_intent (deterministic, zero model call)
  ├─ existing session summary → generate_summary
  └─ retrieve_knowledge → RAGService
       ├─ no evidence → build_response(insufficient)
       ├─ knowledge_qa / explanation → build_response
       ├─ quiz → generate_quiz → build_response
       ├─ summary → generate_summary → build_response
       └─ study_plan → deterministic plan → build_response
  → END
```

工具：

- `KnowledgeRetrievalTool`：复用正式 RAG 与 Citation；
- `QuizGeneratorTool`：只依据检索上下文输出受 Pydantic 约束的 Quiz；
- `LearningSummaryTool`：只总结有界会话内容；
- Study Plan：确定性生成，不额外调用模型。

### 7.3 Study Workflow LangGraph

这是另一个有状态、可暂停恢复的学习计划流程，不应与 Tutor Graph 混称为一个图。

```text
START → route_request
  ├─ start → validate_goal → gather_evidence(AgentService)
  │          → draft 3-task plan
  │          → interrupt for human approval
  │             ├─ approve → activate plan → END
  │             └─ reject → rejected → END
  ├─ retry → gather_evidence → ...
  └─ progress → record progress
               ├─ continue → END
               └─ all completed → finalize_review → END
```

状态含目标、Evidence 摘要、来源、三项任务、当前任务、进度历史、审批、重试次数、最终 Review
和 route trace。Checkpoint 按 `user_id + workflow_id` 校验所有权；重试需要新的费用确认。

## 8. 数据所有权与持久化

```text
APP_DATA_DIR/
  ├─ learning/learning.sqlite3
  │    ├─ users + revocable auth sessions
  │    ├─ learning sessions + conversation + learning records
  │    └─ Tutor LangGraph checkpoints
  ├─ jobs/document_jobs.sqlite3
  ├─ study_workflows/checkpoints.sqlite3
  ├─ model_gateway/model_credentials.sqlite3
  ├─ request_logs/http_requests.jsonl
  └─ user_workspaces/<canonical-user-uuid>/
       ├─ documents/
       ├─ pending_uploads/
       ├─ pending_deletions/
       └─ vector_store/ + Index Manifest
```

用户隔离依赖两层：

1. API 只信任服务端认证得到的 `current_user.user_id`；
2. 资料目录、Chroma、Job、Session、Workflow 与凭据 Store 都按 canonical UUID 校验所有权。

这能满足当前单实例原型，但还不是公开多租户安全验收。SQLite 文件、Fernet 主密钥、资料目录
与 Chroma 需要协调备份；当前没有正式备份恢复演练和静态数据全盘加密。

## 9. Observability 与失败边界

每个 HTTP 请求生成/回传 `X-Request-ID`，并通过 ContextVar 关联用户、RAG、模型和 Job
enqueue 事件。后台 Worker 使用独立 Context，依靠 `job_id` 与持久化状态关联生命周期。

当前信号：

- JSON 结构化日志：固定字段白名单，不记录问题、Prompt、回答、资料正文、API Key 或 Provider 错误正文；
- Runtime Metrics：HTTP、Retrieval、LLM、Document Job 的进程内累计；
- Model 维度：Provider、模型、`system/byok` 来源、耗时、成功/失败和可获得时的 token usage；
- Request History：保存受限元数据，不保存内容；
- 稳定错误映射：输入、权限、限流、RAG、Agent/Tutor、Provider 认证和 Job 失败分开。

Runtime Metrics 重启清零，没有时间窗口、持久化 Dashboard、SLO、告警、OpenTelemetry Exporter
或 LangSmith。Stage 5.4 的选择是先建立可验证的本地信号，再由真实多实例/运维需求触发外部平台。

## 10. 部署拓扑与运行边界

```text
Host / Docker Compose
  → one non-root Python container
  → app.server / Uvicorn
  → FastAPI + static web + in-process worker
  → named volume mounted at /app/data
  → GET /health healthcheck
```

仓库具有 Dockerfile、单服务 Compose、统一 Settings、healthcheck、非 root 用户和 named volume
契约；本机历史验证覆盖 `app.server`、Health、首页和静态资源。当前环境没有 Docker CLI，
因此镜像 build/up、容器重启恢复和容器内页面未验证。也没有 HTTPS、反向代理、公开部署、
并发/负载、长时间运行、生产备份、灾难恢复或 SLA 证据。

## 11. 安全与成本控制

已实现：

- Pydantic `extra="forbid"` 与长度/数量/资源上限；
- 文件名、签名、解析页数/字符、路径和用户目录校验；
- URL scheme、DNS、每跳重定向与非公网地址检查；
- 显式 API 费用确认、单次/滚动窗口预算与单进程并发保护；
- Index Manifest 不兼容与 legacy read-only 写保护；
- 用户/Session/Job/Workflow/Chroma/BYOK 所有权检查；
- 凭据密文、Secret repr 隐藏、错误脱敏和日志字段白名单；
- Prompt 中把资料、网页、会话和目标视为不可信数据。

未完成的生产安全能力：

- 公开网络防护、管理员 RBAC、审计留存策略、WAF/反向代理与 HTTPS；
- KMS/HSM、密钥版本化轮换、数据库/资料全盘加密和正式备份恢复；
- 多实例共享配额、分布式锁/队列、租户级预算和生产渗透测试；
- 真实 Provider、依赖供应链、并发负载和长期稳定性验收。

## 12. 架构判断

这个项目的技术亮点不是组件数量，而是四条可解释的工程主线：

1. Evidence First：从 Chunk 候选到 Evidence Map，再到服务端校验的 Citation；
2. Evaluation Driven：实验能力只有通过固定 Dataset、基线、逐案例回归和延迟/成本门槛才进入正式链路；
3. Deterministic before Agentic：检索与学习流程优先使用可预测节点，只有资料整理和受限工具选择交给 Agent；
4. Smallest durable backend：在本地单实例范围内使用 SQLite、单 Worker、单服务 Compose 和进程内 Gateway，并明确何时才值得升级。

因此当前不应新增 Multi-Agent、MCP、GraphRAG、微服务或更多数据库。下一次架构升级必须由
可复现的检索/回答缺陷、真实并发/多实例需求或明确运维目标触发。
