# 知行（Study Material Assistant）· Project Context

> 面向开发、Review 与交接的最小上下文。当前状态以 [PROJECT_STATUS](PROJECT_STATUS.md) 为准，
> 架构以 [ARCHITECTURE](ARCHITECTURE.md) 为持续文档，
> [FINAL_ARCHITECTURE](FINAL_ARCHITECTURE.md) 是 Stage 6 的冻结展示快照。

## 1. 项目是什么

知行是一个面向个人学习资料的 AI Learning Companion。核心问题不是“让模型聊天”，而是：

- 把 TXT、Markdown、DOCX、带文字层 PDF 和受控网页内容变成可检索资料；
- 用 RAG 基于用户自己的资料回答，并返回可定位的 Evidence/Citation；
- 用 Tutor 与学习计划 Workflow 支持解释、练习、总结、计划和进度恢复；
- 用评测、用户隔离、任务持久化、部署定义与可观测性约束 AI 应用的不确定性。

当前是本地单实例工程原型，不是公开生产系统。

## 2. 当前 Git 与 Stage 边界

- Stage 5.5 代码 checkpoint：`d54a90b`；
- Stage 5.5 文档收口：`9ac680a`；
- Stage 6.1 Final Architecture：`c3c39e0`；
- 当前 Stage 6：Production Readiness & Portfolio，进行中；
- 当前分支：`main`；本地提交尚未 push；
- Stage 6 不新增核心业务功能，不引入 Multi-Agent、MCP、GraphRAG、新数据库或微服务。

提交、Tag 与工作区可能继续变化，准确边界必须用当前 `git status`、`git log` 和
`git tag` 重新确认，不能只依赖本文。

## 3. 事实证据层级

描述能力时按以下层级分别报告：

1. 代码存在并接入真实入口；
2. 自动化测试覆盖；
3. 本次或历史自动化运行通过；
4. 本地进程、HTTP 或浏览器实际验证；
5. 真实 Embedding / ChatModel / Reranker / Provider 验证；
6. Docker、公开部署、并发、生产安全或 SLA 验证；
7. 计划、实验或尚未验证。

Fake、Mock、Stub 和临时 SQLite 只能证明对应代码契约。Completion Report 是证据索引，
不能代替代码、Git、运行结果或生产验收。

## 4. 当前正式主链路

```text
Authenticated User
  → per-user workspace / Chroma
  → Dense Top 10
  → score threshold 0.25
  → 0.8 vector + 0.2 keyword coverage
  → Top 3 seeds + adjacent expansion
  → EvidenceScoreContextSelector
  → Evidence Map
  → LCEL Prompt
  → ModelGateway (BYOK or system)
  → ChatModel
  → server-side Citation ID validation
  → Answer + sources + citations
```

BM25 + RRF、Cross-Encoder Reranker 与 Structure-aware Chunking 是隔离实验，不是正式
`RAGService` 主链路。

## 5. 关键模块入口

| 需求 | 从哪里开始读 | 向下追踪 |
| --- | --- | --- |
| HTTP/API | `app/api.py` | dependency、service cache、错误映射、response model |
| RAG 问答 | `get_rag_service()` | `rag_service.py` → `hybrid_search.py` → `langchain_rag.py` |
| Evidence/Citation | `app/evidence.py` | Evidence Map 构造、Citation ID 解析与 metadata |
| 资料上传/索引 | `stage_material_*` API | `material_ingestion.py` → `document_jobs.py` → Chroma/Manifest |
| Tutor | `/api/tutor/chat` | `tutor_workflow.py` StateGraph、Tools、checkpoint |
| 学习计划 | `/api/study-workflows` | `study_workflow.py` StateGraph、interrupt、owner check |
| 用户与学习数据 | `get_current_user()` | `auth.py`、`learning_data.py`、`user_workspace.py` |
| Model Gateway/BYOK | credential API | `app/model_gateway/`、`rag_service.py` |
| Observability | middleware / model boundary | `observability.py`、`metrics.py`、`request_history.py` |
| Deployment | `app/server.py` | `app/config.py`、Dockerfile、Compose、`.env.example` |
| Evaluation | CLI modules | [EVALUATION](EVALUATION.md) 与 Stage 3 Completion Reports |

不要根据文件名推断“已生效”。从 `app/api.py`、服务创建函数、真实调用链和测试向下确认。

## 6. 数据与安全边界

- 不读取、输出或提交 `.env`、API Key、真实用户资料或本地索引内容；
- 不在未确认费用时运行真实 Embedding、ChatModel 或 Reranker；
- 上传、网页、Query、Prompt 上下文、会话和模型输出都按不可信输入处理；
- 资料、Chroma、Job、Session、Workflow 与 BYOK 以服务端认证得到的 UUID 作为所有者；
- `legacy_read_only` 只允许 Chroma 读取，必须在任何 Embedding 或写入前拒绝迁移外操作；
- 本地 SQLite/Chroma 不是多实例共享存储，不应并发挂载到多个应用副本写入。

## 7. 修改前决策

任何新增技术或架构变化先回答：

1. 当前可观察问题是什么？
2. 问题属于 Parser、Chunk、Retrieval、Context、Prompt、Model、Workflow 还是工程系统？
3. 当前结构能否用更小改动解决？
4. 输入、输出、Evidence、索引或 API 契约是否变化？
5. 如何固定 Baseline、数据集和验收门槛？
6. 会增加哪些外部调用、费用、数据发送、部署和维护成本？
7. 失败时如何回退，什么时候删除该方案？

没有真实瓶颈和可复现证据时，不增加 Agent、模型、数据库、队列或平台。

## 8. 标准交付流程

```text
Git / rules / current evidence
  → trace real call chain
  → define smallest scoped change
  → implement
  → targeted verification
  → full relevant regression
  → independent diff / security / contract review
  → update status and completion evidence
```

代码修改后必须检查真实 diff、入口是否接通、异常和用户隔离、成本边界、测试证据是否被夸大。
未经明确授权不 commit、tag、push、deploy 或迁移数据。

## 9. 当前主要未验证项

- Docker build/up、volume 重启恢复与容器内页面；
- 真实 Provider compatibility matrix 与系统/BYOK 全链路；
- 当前认证版本的真实付费 RAG/Tutor/Study Workflow；
- 并发、负载、长时间运行、多实例一致性与共享限流；
- 正式备份恢复、主密钥轮换、公开安全、渗透、SLO 与告警；
- Claim-level Faithfulness、Citation Support/Accuracy 的稳定基准。

这些不是“已经失败”，而是当前证据不足，不能写成已完成。
