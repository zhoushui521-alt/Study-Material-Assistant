# Study Material Assistant V2 · Technical Decisions

本文只记录已有代码、实验数据和 Completion Report 能支持的关键取舍。指标均绑定当时的 10 案例 Dataset、对应 commit 与索引/回放条件，不能外推为普遍或生产结论。

## 1. BM25 + Dense + RRF 不替换正式 Retrieval

### 背景

Stage 2 正式基线只有 Dense 召回；关键词覆盖率只重排 Dense Top 10，无法找回候选池之外的精确术语 Chunk。Stage 3.1 因此验证独立 BM25 召回与 RRF 融合。

### 方案

保持 Embedding、Chunk、阈值、TopK、相邻扩展、Context、Evidence、Citation 与 LLM 不变，只比较：

- Baseline：Dense Top 10 → threshold → keyword coverage rerank；
- Experiment：Dense Top 10 + BM25 Top 10 → RRF(`k=60`)。

### 结果

- Recall@5：`0.8333 → 0.9444`
- MRR：`0.6111 → 0.5778`
- nDCG@5：`0.6422 → 0.6561`
- Recall@3：保持 `0.8333`

`wishart_definition` 改善，但 `cauchy_properties` 从 Gold Rank 2 退化到 5。正式问答只取 Top 3 Seed，因此 Recall@5 的收益不能抵消前排排序退化。

### 最终选择

保留 BM25、RRF 和实验 Runner，不接入正式 `RAGService`。正式 Retrieval 继续使用 Stage 2 baseline，直到更大的冻结 Dataset 与明确验收门槛证明收益。

证据：[Stage 3.1 Completion Report](stage3-1-completion-report.md)

## 2. Cross-Encoder Reranker 保留实验能力，不进入正式 Pipeline

### 背景

Stage 3.1 表明 Sparse 召回能补充精确术语，但 RRF 可能破坏 Dense 的前排顺序。Stage 3.2 验证 Cross-Encoder 能否只重排已有 RRF Candidate Pool。

### 方案

使用固定版本的 `BAAI/bge-reranker-base`，在 CPU/FP32 上对 RRF 最多 20 个候选执行 Pair Scoring；不重新召回，不接入 `RAGService`。

### 结果

- MRR：`0.5778 → 0.9259`
- nDCG@5：`0.6561 → 0.9015`
- Recall@3：`0.8333 → 0.9444`
- Context Precision：`0.1577 → 0.1458`
- 平均本地 Retrieval 延迟：`211.5 ms → 3073.9 ms`，增加约 `2862.4 ms`
- 模型目录约 `1.27 GB`

排序收益没有转化为 Context Precision 收益；唯一的无答案处理失败也没有解决。

### 最终选择

保留隔离、可复现的 Reranker Adapter 与评测能力，不接入正式问答 Pipeline。原因是当前交互延迟代价过高、Dataset 只有 10 个案例，并且缺少真实 Answer、Faithfulness、Citation Accuracy、并发和长期验证。

证据：[Stage 3.2 Completion Report](stage3-2-completion-report.md)

## 3. Structure-aware Chunking 不替换固定字符 Chunker

### 背景

固定 180 字符 Chunk 可能切断完整语义。Stage 3.3 在保持 Embedding、Retrieval、阈值、Top-K、Context 和 Dataset 不变时，验证更大的结构感知块是否更好。

### 方案

对比：

- A：`fixed-character-180-v1`；
- B：`structure-aware-block-600-overlap-0-v1`，按标题、段落、列表、代码块、句子与字符逐级回退。

A/B 都用相同运行时 Embedding 配置建立临时 Chroma，不修改或迁移正式索引。

### 结果

- Chunk 数：`147 → 55`
- Recall@1：`0.3333 → 0.5556`
- MRR：`0.6111 → 0.6426`
- nDCG@5：`0.6422 → 0.7019`
- Recall@3：`0.8333 → 0.6667`
- Final Context Recall：`1.0000 → 0.7778`
- Context Precision：`0.1500 → 0.1202`

更完整的大块改善了部分跨边界案例，但粒度变粗并制造新的 Recall / Ranking Failure。

### 最终选择

继续使用正式 `fixed-character-180-v1`。Structure-aware 实现只作为实验能力保留，不接入 `build_chunks()`、资料摄取或正式 RAG，也不重建生产索引。后续若重做实验，应保持单变量，先测试更小的结构块上限。

证据：[Stage 3.3 Completion Report](stage3-3-completion-report.md)

## 4. EvidenceScoreContextSelector 接入正式 Pipeline

### 背景

Stage 2 历史 Retrieval 基线的 Final Context Recall 为 `1.0`，但 Context Precision 只有 `0.15`，平均 Context 为 `7.3` 块。问题位于相邻扩展后的 Context Construction，而不是 Candidate Generation。

### 方案

只处理 Retriever 已返回的 Evidence：

- 保留所有 Top 3 Seed；
- 每个 Seed 最多保留一个关键词覆盖率最高的已有相邻 Evidence；
- 按距离与原顺序稳定打破并列；
- 不重新召回、不调用模型、不改变 Retrieval 排名。

### 结果

- Context Precision：`0.1500 → 0.2000`
- Final Context Recall：保持 `1.0000`
- 平均 Context Chunk：`7.3 → 5.2`
- 平均 Context 字符：`1222.2 → 848.7`
- Query Embedding / ChatModel / Reranker：`0 / 0 / 0`

### 最终选择

接入正式 `RAGService`，同时保留 `BaselineContextSelector` 作为可回退、可对比实现。

该结论来自 `local_historical_retrieval_trace_replay`，不是当前索引的新 Retrieval 或真实回答验收；Dataset 扩大后若 Final Context Recall 回归，应先回退或单独调整 Selector。

证据：[Stage 3.4 Completion Report](stage3-4-completion-report.md)

## 5. SQLite 用于当前本地单实例持久化

### 背景

Stage 4 的 Tutor 使用 `InMemorySaver`，进程退出后 Session 状态丢失；文档确认索引也长期占用同步 HTTP 请求。项目需要可恢复的学习数据与任务状态，但没有多实例、高吞吐或托管数据库需求证据。

### 方案

- 使用显式 SQL + `aiosqlite` 保存用户、认证 Session、学习 Session、Conversation、Learning Record 和 Document Job；
- 使用 `AsyncSqliteSaver` 保存 Tutor 与历史 Study Workflow checkpoint；
- 开启 WAL、外键与 5 秒 busy timeout；
- Document Job 配合单进程串行 `asyncio` Worker，而不是 Redis/Celery。

### 结果

- Stage 5.1 历史验证覆盖数据库关闭重开后的 Tutor 续学；
- Stage 5.2 历史验证覆盖 Job 去重、`pending` 恢复、处理中断转 `failed` 和 `202 + job_id`；
- User Identity & Data Isolation 阶段的历史验证覆盖认证、复合外键、A/B 数据隔离与 owner 校验。

这些是 Fake/Mock 与临时真实 SQLite 证据，不是并发负载或生产数据库验收。

### 最终选择

当前继续使用 SQLite，因为它能以最小运维成本满足本地单实例原型。只有出现公开多用户、多 API 实例、共享事务吞吐、独立 Worker、高可用或正式备份恢复需求后，才考虑 PostgreSQL 与共享 Session/Job 基础设施。

“未来 PostgreSQL”是条件式迁移方向，不是已经开始或已承诺的 Stage。迁移时应保持 `current_user → owner-scoped repository` 与服务层契约，而不是让数据库替换扩散到 RAG Pipeline。

证据：[Stage 5.1 Persistence](stage5-1-completion-report.md)、[Stage 5.2 Async Jobs](stage5-2-completion-report.md)、[User Identity & Data Isolation](stage5-1-user-identity-completion-report.md)

## 6. 每用户独立 Chroma，而不是只依赖 metadata filter

### 背景

只有业务表按 `user_id` 过滤并不能阻止 RAG 从其他用户的 Chunk 生成答案、文件名、摘录、`material_id` 或 Citation。单个全局 Chroma 若遗漏 owner filter，会直接形成数据泄露。

### 方案

通过 `current_user.id` 解析 `data/user_workspaces/<uuid>/vector_store`，每个用户打开独立 Chroma；资料目录、暂存与删除隔离使用同一 Workspace 根目录。

### 结果

[User Identity & Data Isolation Completion Report](stage5-1-user-identity-completion-report.md) 记录的 `392/392` Fake/Mock/临时 SQLite 回归覆盖两个独立 Chroma 的 Retrieval 与来源 metadata 不串库，并覆盖 Material、Job、Tutor 与 Workflow 所有权。

### 最终选择

在当前本地规模下保留物理目录隔离。它复用现有 MaterialManager、Manifest、回滚和 Retrieval 调用链，并降低漏写 metadata filter 的风险。代价是用户增多时 Chroma 客户端与目录数量增加；未来外部向量库仍需服务端强制 owner filter。

证据：[User Identity & Data Isolation Completion Report](stage5-1-user-identity-completion-report.md)

## 7. 单服务 Compose，而不是提前拆分 frontend、worker 和 database

Decision: 以一个 FastAPI 应用容器承载同源前端、API、Service Layer 和进程内 Document Worker。

Status: Adopted for `study-material-v2-stage5-part4`。

Background: 当前前端没有独立构建系统，Worker 依赖进程内 OperationGuard、SQLite 和本地
Chroma；仓库没有跨进程锁、共享 Session Store、分布式任务领取或外部向量库。

Options: 单服务 Compose；拆分 frontend/backend/worker；引入 PostgreSQL/Redis 后再拆分。

Choice: 当前只定义 `app` 服务、named volume 和 bridge network。

Reason: 单服务忠实表达真实调用链，并用最小运维成本实现可重复启动和持久化。

Trade-off: 不能水平扩展，应用更新会同时影响 API 与 Worker，named volume 只适合单写实例。

Evidence: `Dockerfile`、`docker-compose.yml`、Deployment Contract tests、Stage 5.3 本地启动验证。

Revisit Trigger: 需要独立扩缩 Worker、多 API 实例、共享吞吐、跨实例故障恢复或零停机部署。

## 8. 保持 SQLite，PostgreSQL 迁移由多实例需求触发

Decision: Stage 5.3 继续使用 SQLite，不安装 PostgreSQL 驱动或执行数据迁移。

Status: Adopted；PostgreSQL 为 Future。

Background: 当前业务数据、Auth Session、Job 和 LangGraph checkpoint 都已用 SQLite 实现并
通过单实例自动化；没有多实例、负载或生产灾备证据证明迁移收益大于复杂度。

Options: 立即迁移 PostgreSQL；保留 SQLite 并记录迁移契约；增加 Redis 作为旁路。

Choice: 保留 SQLite，并让统一 Settings 显式固定数据库和数据目录边界。

Reason: Stage 5.3 的问题是可启动与可维护，不是共享事务或分布式协调。Redis 不能替代关系
约束、Migration 或 Chroma/文件共享问题。

Trade-off: 仍然不能安全运行多个写实例，也没有生产备份、故障切换和连接池能力。

Evidence: Stage 5.1/5.2/part3 Completion Reports、400/400 回归、Stage 5.3 Compose 单服务设计。

Revisit Trigger: 多 API 实例、独立 Worker、共享事务吞吐、正式灾备或 SQLite 锁竞争成为
可观察瓶颈。届时需同时处理 SQL 方言、Migration、PostgreSQL checkpointer、事务 Job 领取、
共享文件/向量存储与回滚，不能只更换数据库连接字符串。

## 9. Stage 5.4 先采用本地基础可观测性，不接入 OpenTelemetry / LangSmith

Decision: 保留标准库 JSON Logging、`ContextVar` Request Trace 与单进程 `RuntimeMetrics`，
Stage 5.4 不安装 OpenTelemetry SDK/Collector，也不连接 LangSmith。

Status: Adopted for Stage 5.4。

Background: 当前运行拓扑是一个 FastAPI 进程、进程内 Document Worker、本地 SQLite 与
本地 Chroma。现阶段需要回答请求经历、阶段耗时、空召回、模型失败和进程健康，但没有跨服务
Trace、多个副本聚合、长期指标留存、告警或外部 Trace 平台的已验证需求。

Options: 标准库本地基础能力；立即接入 OpenTelemetry；立即接入 LangSmith；同时接入两者。

Choice: 采用本地基础能力。结构化日志只接收白名单非内容字段；Request Trace 用
`request_id` / `job_id` 关联；Metrics 只保留当前进程匿名累计值；认证接口只返回聚合快照。

Reason: 该方案已经覆盖当前单实例的问题定位闭环，不增加 Collector、Exporter、外部账号、
网络故障、数据发送、费用和部署维护边界。LangSmith 更适合需要持久化模型/链路追踪和质量分析的
场景，但当前把 Prompt、学习资料或模型输出发送到外部平台会引入额外隐私与授权问题。

Trade-off: 日志和 Metrics 没有统一 Trace Store；进程重启后 Metrics 清零；没有百分位、
时间窗口、跨实例聚合、Dashboard、告警、管理员 RBAC，也不能把运行指标自动关联到 Evaluation
Dataset。当前能力是基础可观测性，不是生产级监控平台。

Evidence: `app/observability.py`、`app/metrics.py`、认证 Metrics API、417/417 全量回归和
Stage 5.4 独立临时实例手动验证。

Revisit Trigger: 出现多个服务或实例、独立 Worker、跨进程 Trace、长期留存、SLO/告警、
生产 Dashboard，或经批准需要把运行 Trace 与模型质量评测关联。届时应先定义数据分级、采样、
脱敏、保留期、费用上限和退出方案，再选择 OpenTelemetry Exporter、Collector 或 LangSmith。
## 10. Stage 5.5 采用进程内 Model Gateway，而不是只在业务接口散落 BYOK

Decision: 在正式 FastAPI / RAG / Agent / Tutor 模型调用前加入进程内 `ModelGateway`、
确定性 Router、Provider Adapter Registry 与用户凭据存储；不拆微服务，不接外部 LLM
Gateway 平台。

Status: Accepted and implemented in `d54a90b`；未单独创建 Stage 5.5 Tag。

Background: Stage 5.4 结束时，正式 `RAGService` 仍从固定百炼/OpenAI-compatible 配置创建
ChatModel。直接在各业务接口增加用户 Key，会把 Provider、凭据选择、参数转换、错误处理和
观测逻辑复制到 RAG、Agent 与 Tutor，并增加明文泄露和用户隔离遗漏风险。

Options: 在每个 API 中直接处理 BYOK；建立单体内 Model Gateway；立即接入外部模型平台或
拆分 Gateway 微服务。

Choice: 复用 LangChain `BaseChatModel` 作为业务接口，由 `ModelGateway` 选择“当前用户
BYOK 或系统默认”，再由 `ProviderAdapterRegistry` 创建具体模型。`qwen`、`deepseek`、
`openai` 与 `openai_compatible` 当前复用 OpenAI-compatible Adapter；用户不能通过 API
提交任意 Base URL。

Reason: 该方案在不改变 Retrieval、Context、Prompt、Evidence/Citation、Agent 决策或 Tutor
Workflow 的前提下，集中 Provider 选择、timeout、max tokens、temperature、认证错误和
Stage 5.4 观测字段。它保留现有 LCEL/Agent/Tutor 调用契约，不需要重复发明另一套
`generate/stream` 接口，也不增加网络 hop、独立部署和外部数据发送。

Security Choice: 每位用户当前只保存一条活跃凭据，主键是服务端认证得到的 UUID；
API Key 使用 Fernet 认证加密后写入独立 SQLite，主密钥只来自
`MODEL_CREDENTIAL_ENCRYPTION_KEY`。API、日志和模型元数据响应都不返回明文 Key。
已保存 BYOK 无法解密或认证失败时不静默回退系统 Key；这能避免掩盖密钥问题和产生未授权的
系统费用。保存/删除只失效该用户模型相关缓存。

Trade-off: 这不是生产密钥管理系统。Fernet 主密钥还没有 KMS/HSM 托管、版本化轮换或灾备
流程；SQLite 仍是单实例边界。没有实时价格表、货币成本计算、自动故障转移、按任务复杂度
路由、负载均衡、熔断或流式产品 API。Claude/Anthropic 需要独立原生 Adapter，当前不能宣称
支持。仓库早期教学 `app/chat_client.py` 仍是非正式 V2 兼容脚本，不在本次正式调用链迁移
结论内。

Evidence: `app/model_gateway/`、`app/rag_service.py`、认证凭据 API、Provider/模型/凭据来源
日志与聚合指标，以及最终 `434/434` Fake/Mock/临时 SQLite 自动化回归。没有调用真实
Embedding、ChatModel、Reranker 或外部网关，因此这些证据不证明真实 Provider 兼容性、回答
质量、货币成本或生产安全。

Revisit Trigger: 需要 Claude/Anthropic、多个凭据/组织级凭据、真实流式 API、价格版本与预算
阻断、多 Provider 容灾、多个应用实例、正式主密钥轮换/托管，或观测到当前 Adapter/SQLite
成为瓶颈时，再分别评估原生 Adapter、定价注册表、共享凭据存储或独立 Gateway。

## 11. Stage 6 以生产就绪证据和作品化收口，不新增核心技术

Decision: Stage 6 只整理、验证和解释 Stage 0～5.5 已有能力，输出最终架构、工程上下文、
评测汇总、Demo、面试材料和最终质量报告；不新增核心业务功能或基础设施。

Status: Accepted and completed in `study-material-v2-stage6-final`。

Background: 当前系统已经包含正式 RAG、Evidence/Citation、Retrieval Evaluation、
Tutor/Study Workflow、身份与用户隔离、异步 Job、部署定义、Observability 和 Model Gateway。
继续加入 Multi-Agent、MCP、GraphRAG、新数据库或更多模型，会扩大测试、部署、安全和解释成本，
但当前没有对应故障、规模或用户价值证据。

Options: 继续扩展技术清单；只做展示文档；以代码事实、测试、Demo 和限制清单完成工程收口。

Choice: 选择第三项。Stage 6 文档必须回指真实入口、Git checkpoint、阶段报告和本次验证；
发现代码/文档不一致时降低表述或记录未验证，不用宣传性语言掩盖边界。

Reason: 对 AI 应用工程作品，能说明“为什么这样设计、如何验证、何时拒绝技术、还有什么未验收”
比新增框架更能证明工程判断。该选择也保持当前单实例原型的最小可维护边界。

Trade-off: Stage 6 的业务代码量可能接近零，也不会把项目自动变成公开生产系统。Docker、
真实 Provider、并发负载、备份恢复、公开安全和 Claim-level Faithfulness/Citation Accuracy
若没有新证据，仍必须列为未验证。

Evidence: [Stage 6 Completion Report](stage6-completion-report.md)、Final Architecture、Stage 3～5
Completion Reports、正式调用链与 Stage 6 最终全量 `434/434` 自动化回归。

Revisit Trigger: 出现可复现的回答质量缺陷、真实多实例/吞吐需求、明确外部集成场景或运维 SLO，
并且现有结构无法用更小改动满足时，再为具体问题立项。
