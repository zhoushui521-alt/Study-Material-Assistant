# Study Material Assistant V2 · Project Status

> 当前状态唯一入口。本文只汇总已经能由 Git checkpoint、阶段报告、当前调用链或历史验证记录核对的事实；详细设计与指标仍以链接的原始证据为准。

## 同步基线

- 同步日期：2026-08-24
- 分支：`main`
- Stage 5.3 起点：`29e7c7e4e11781e26d8cf0dc3a523b83efe6acbd`
- Stage 5.4 起点：`6f0e770`（`study-material-v2-stage5-part4`）
- Stage 5.5 起点：`e141e95`（Stage 5.4 收口提交）
- Stage 5.5 代码 checkpoint：`d54a90b`；本阶段未单独创建 Tag
- Stage 5.5 文档收口：`9ac680a`
- Stage 6.1 Final Architecture checkpoint：`c3c39e0`
- Stage 6.2 Engineering Documentation checkpoint：`826f75a`
- Stage 6.3 RAG Evaluation checkpoint：`6f63f6a`
- Stage 6.4 Demo checkpoint：`8554491`
- Stage 6.5 Interview checkpoint：`c773d05`
- Stage 6 Final Tag：`study-material-v2-stage6-final`，指向 Stage 6 Completion Report 提交
- 远程边界：本地提交未 push；公开部署状态不能由本地 Git 推断
- 证据层级：代码实现、自动化测试、历史本地运行、真实模型实验、生产验证分别记录，不能互相替代

状态含义：

- **Completed**：存在可解析的 Git checkpoint，或有明确的审计记录与当前实现证据。
- **In Progress**：已经开始，但尚未形成完成 checkpoint。
- **Planned**：仅有计划，不代表代码已实现。

## 1. 项目定位

项目名称为 **AI Learning Companion「知行」**，仓库名为 Study Material Assistant。

它解决的核心问题是：学习资料分散、长文档难以检索，以及模型回答缺少可核对的依据。当前系统把本地资料摄取、RAG、Evidence/Citation、学习型 Tutor、身份与数据隔离、持久化学习数据和异步文档处理组合成一个同源 Web 应用。

当前定位是**本地单实例、具备较完整工程边界的 AI 学习应用原型**，不是生产级分布式系统。当前没有 PostgreSQL、Redis、Celery、分布式 Worker、多实例共享 Session、公开部署或生产负载验收。

## 2. 阶段状态

### Stage 0：Architecture Audit

- 状态：**Completed（证据边界较弱）**
- 结果：形成 V2 的真实架构、能力边界、RAG 分层、阶段路线与证据分级规则。
- 证据边界：仓库存在 `study-material-v2-baseline`（`fd4746d`），当前 `AGENTS.md` 保存审计后的项目规则；但没有独立的 `study-material-v2-stage0` Tag 或 Stage 0 Completion Report，因此不能把它描述为与 Stage 1～5 相同强度的独立 Git checkpoint。

### Stage 1：Evidence / Citation Foundation

- 状态：**Completed**
- Checkpoint：`study-material-v2-stage1` → `3a50576`
- 已实现：稳定 Material/Chunk 身份、结构化 Evidence、请求内 Evidence ID、服务端 Citation ID 校验与权威 metadata 回填、Index Manifest，以及 legacy index 的只读兼容边界。
- 证据边界：仓库没有独立 Stage 1 Completion Report；结论来自 Tag、阶段末代码、测试与 README。Citation ID 有效只证明它指向本次 Evidence Map，不能自动证明 Evidence 支持回答中的具体 Claim。

### Stage 2：Retrieval Evaluation & Observability

- 状态：**Completed**
- Checkpoint：`study-material-v2-stage2` → `fda1ac0`
- 已实现：版本化 Retrieval Dataset、Stable Gold Meaning 与 Current Chunk Mapping、完整 Retrieval Trace、Recall@1/3/5/10、MRR、nDCG@5、Final Context Recall、Context Precision 和失败分类。
- Dataset：`study-material-retrieval-v1`，10 个案例，其中 9 个可回答。
- 历史 `local_real_retrieval` 基线：Ranked Recall@1/3/5/10 为 `0.3333 / 0.8333 / 0.8333 / 0.9444`，MRR 为 `0.6111`，nDCG@5 为 `0.6422`，Final Context Recall 为 `1.0`，Context Precision 为 `0.15`。
- 失败：1 个 `ranking_failure`，1 个 `unanswerable_handling_failure`。
- 证据边界：仓库没有独立 Stage 2 Completion Report；上述运行数据由 [Stage 3.1 Baseline](stage3-1-baseline.md) 保存。它绑定当时的 commit、Dataset 与索引快照，不证明当前索引、回答正确性、Faithfulness、Citation Support 或生产稳定性。

### Stage 3：RAG Optimization

#### Stage 3.1：BM25 + Dense + RRF

- 状态：**Completed as experiment**
- Checkpoint：`study-material-v2-stage3-part1` → `2d526c5`
- 结果：Recall@5 `0.8333 → 0.9444`，nDCG@5 `0.6422 → 0.6561`，但 MRR `0.6111 → 0.5778`，Recall@3 不变。
- 决策：保留独立 BM25/RRF 实验能力；正式 `RAGService` 继续使用 Stage 2 Dense + keyword coverage baseline。
- 证据：[Stage 3.1 Completion Report](stage3-1-completion-report.md)

#### Stage 3.2：Cross-Encoder Reranker Experiment

- 状态：**Completed as experiment**
- Checkpoint：`study-material-v2-stage3-part2` → `cc6e344`
- 结果：相对 Hybrid，MRR `0.5778 → 0.9259`，nDCG@5 `0.6561 → 0.9015`；平均本地 CPU Retrieval 延迟增加约 `2862.4 ms`，Context Precision `0.1577 → 0.1458`。
- 决策：保留隔离的 `BAAI/bge-reranker-base` 实验能力，不接入正式 Pipeline。
- 证据：[Stage 3.2 Completion Report](stage3-2-completion-report.md)

#### Stage 3.3：Structure-aware Chunking Experiment

- 状态：**Completed as experiment**
- Checkpoint：`study-material-v2-stage3-part3` → `b4757b2`
- 结果：候选策略 `structure-aware-block-600-overlap-0-v1` 提升 Recall@1、MRR 与 nDCG@5，但 Recall@3 `0.8333 → 0.6667`、Final Context Recall `1.0000 → 0.7778`、Context Precision `0.1500 → 0.1202`。
- 决策：不替换正式 `fixed-character-180-v1` Chunker，不迁移生产索引。
- 证据：[Stage 3.3 Completion Report](stage3-3-completion-report.md)

#### Stage 3.4：Context Selector

- 状态：**Completed and production-wired**
- Checkpoint：`study-material-v2-stage3-part4` → `ff7423f`
- 结果：历史 Retrieval Trace 回放中，Context Precision `0.1500 → 0.2000`，平均 Context Chunk `7.3 → 5.2`，Final Context Recall 保持 `1.0`。
- 决策：`EvidenceScoreContextSelector` 接入正式 `RAGService`；保留全部 Seed，每个 Seed 最多选择一个已有相邻 Evidence。
- 证据边界：本轮为 `local_historical_retrieval_trace_replay`，外部调用为 `0 / 0 / 0`，不是当前索引的新 Retrieval、真实回答或生产验收。
- 证据：[Stage 3.4 Completion Report](stage3-4-completion-report.md)

### Stage 4：Tutor Agent

- 状态：**Completed**
- Checkpoint：`study-material-v2-stage4` → `4d0fbcd`
- 已实现：单 Tutor `StateGraph`、确定性意图分类与路由、Knowledge Retrieval Tool、Quiz Generator Tool、Learning Summary Tool，以及 QA、Explanation、Quiz、Summary、Learning Plan 五类路径。
- RAG 复用：Knowledge Retrieval Tool 调用现有 `RAGService.ask()`，没有复制 Retrieval、Prompt、Evidence 或 Citation 实现。
- 历史验证：Stage 4 专项 `17/17`、全量自动化 `360/360`；属于 Fake/Mock 自动化证据。
- 未验证：该阶段没有真实 Embedding/ChatModel，也没有本地 Uvicorn 与真实浏览器/API 验收。
- 证据：[Stage 4 Completion Report](stage4-completion-report.md)

### Stage 5：Backend Engineering

#### Stage 5.1 / part1：Persistent Learning Data Foundation

- 状态：**Completed**
- Checkpoint：`study-material-v2-stage5-part1` → `69e9a9f`
- 已实现：UUID 用户、学习 Session、稳定排序的 Tutor 对话、学习行为记录、SQLite Schema 迁移、`AsyncSqliteSaver` checkpoint 与关闭重开恢复。
- 边界：这是数据分区基础，不是认证授权；客户端当时仍可自行提交 `user_id`。
- 证据：[Stage 5.1 Persistence Completion Report](stage5-1-completion-report.md)

#### Stage 5.2 / part2：Async Document Processing

- 状态：**Completed**
- Checkpoint：`study-material-v2-stage5-part2` → `1b9c0c2`
- 已实现：持久化 Document Job、`pending / processing / completed / failed` 状态、单进程 `asyncio` Worker、启动恢复、`202 Accepted + job_id` 与 `GET /api/jobs/{job_id}`。
- 历史验证：Stage 5.2 + API 专项 `45/45`、全量 Fake/Mock/临时 SQLite `376/376`，并完成零费用本地页面检查。
- 边界：预解析仍同步；没有分布式队列、多实例抢占、自动重试、负载测试或生产验收。
- 证据：[Stage 5.2 Completion Report](stage5-2-completion-report.md)

#### Stage 5 / part3：User Identity & Data Isolation

- 状态：**Completed**
- Checkpoint：`study-material-v2-stage5-part3` → `ea8459b`
- 已实现：邮箱注册登录、scrypt 密码哈希、可撤销服务端 Session、`current_user`、复合外键、每用户独立资料目录与 Chroma、Document Job 与 Study Workflow 所有权校验。
- 历史验证：全量 Fake/Mock/临时 SQLite `392/392`，覆盖认证、A/B 数据隔离、Tutor、Material、Vector Retrieval、Citation 来源、Job 与 Workflow IDOR。
- 未验证：本阶段没有真实模型、并发、负载、公开部署或渗透测试验收；浏览器连接受本地沙箱辅助程序错误影响，未完成登录后的真实视觉回归。
- 证据：[User Identity & Data Isolation Completion Report](stage5-1-user-identity-completion-report.md)

#### Stage 5.3 / part4：Serviceization & Deployment Readiness

- 状态：**Completed with environment-limited Docker validation**
- Checkpoint：`study-material-v2-stage5-part4`（以 Tag 解析到的提交为准）
- 已实现：统一 Settings、`APP_DATA_DIR` 持久化根目录、本地/容器共用启动入口、非 root
  Dockerfile、单服务 Compose、named volume、bridge network、容器 Healthcheck 与部署文档。
- 自动验证：新增配置/部署契约测试，全量 Fake/Mock/临时 SQLite 回归 `400/400`；
  `compileall`、内存编译、`pip check`、前端语法和 Compose YAML 结构检查通过。
- 本地运行：使用独立临时数据目录完成 `app.server` startup，`/health`、首页和静态脚本均
  返回 HTTP 200，未调用外部模型或修改真实索引。
- 边界：当前机器没有 Docker CLI，因此镜像 build/up、容器重启恢复与容器内页面未验证；
  没有公开部署、HTTPS、负载、渗透、生产备份或多实例证据。
- 证据：[Stage 5.3 Completion Report](stage5-3-completion-report.md)

#### Stage 5.4：Observability

- 状态：**Completed（本地提交与报告；未创建 Stage 5.4 Tag）**
- Stage 5.4.1 已实现：`app.observability` 统一 JSON Logger、固定
  `time / level / service / request_id / user_id / event / duration_ms` 基线字段、
  HTTP 请求开始/结束事件，以及既有 API 生命周期安全日志迁移。
- 保留边界：轮转 `RequestHistoryWriter` 继续只持久化白名单 HTTP 元数据；控制台结构化日志
  也拒绝未列入白名单的任意内容字段，不记录问题、回答、Prompt、文件正文、密钥或 URL 查询参数。
- Stage 5.4.2 已实现：`ContextVar` 贯穿 HTTP、同步线程池、RAG Service、Retriever、
  RAG ChatModel 与 Tutor 结构化模型调用；LLM usage 仅在返回可用时记录。Document Job
  入队事件关联原 `request_id`，长期 Worker 使用空 Context 并以 `job_id` 追踪处理状态。
- Stage 5.4.3 已实现：线程安全的单进程 `RuntimeMetrics` 聚合 HTTP、Retrieval、LLM 与
  Document Job 数值指标；`GET /api/observability/metrics` 只向已认证用户返回匿名全局
  快照，不返回 `request_id`、`user_id` 或内容数据，也不改变 Retrieval Evaluation。
- 当前验证：改动前全量 `400/400`；5.4.1 专项 `45/45`；5.4.2 Logging / API / RAG /
  Tutor / Document Job 专项 `97/97`；5.4.3 Metrics 集成专项 `102/102`；最终 42 个
  测试模块分两批回归 `219 + 198 = 417/417`，97 个 Python 文件内存编译和 `pip check` 通过。
- 本地手动验证：独立临时实例完成 Health、注册、认证 Metrics、422 错误计数与 request_id
  日志关联；Retrieval / LLM 调用为 0，验证后临时数据已删除。
- 证据：[Stage 5.4 Completion Report](stage5-4-completion-report.md)

#### Stage 5.5：BYOK + Model Gateway

- 状态：**Completed（代码、自动化、Review 与本地 Git checkpoint 已完成）**
- 起点：`e141e95`；代码 checkpoint：`d54a90b`；未单独创建 Stage 5.5 Tag。
- 已实现：进程内 `ModelGateway`、确定性 BYOK-first Router、Provider Adapter Registry、
  OpenAI-compatible Adapter、统一 Model Settings，以及认证后的凭据元数据查询/密文保存/删除。
- 正式接入：`RAGService` 按 `current_user` 从网关取得 `BaseChatModel`；Agent 与 Tutor
  继续复用该 RAG 模型。Retrieval、Context、Prompt、Evidence/Citation、Agent 决策、Tutor
  Workflow 与 Evaluation 指标未改变。
- 安全：每用户一条活跃凭据；Fernet 密文写入独立 SQLite；主密钥只从环境读取；API 与日志
  不返回 Key；用户不能提交任意 Base URL；无法解密或认证失败时不静默使用系统 Key。
- Observability：模型事件和进程指标增加 Provider；日志增加受限的
  `credential_source=system/byok`，继续记录模型、耗时、成功/失败和返回可用时的 token usage。
- 当前 Provider：`qwen`、`deepseek`、`openai`、`openai_compatible`；均通过当前
  OpenAI-compatible Adapter。Claude/Anthropic 尚未实现。
- 自动验证：Stage 5.5 核心/安全/配置/观测专项 `29/29`；最终 44 个 `test_*.py` 模块
  全量 Fake/Mock/临时 SQLite 回归 `434/434`，无失败、错误或跳过。
- 证据边界：没有真实 Provider/ChatModel/Embedding/Reranker 调用，没有 Docker、浏览器、
  并发、负载、主密钥轮换、货币成本核算或生产安全验收。早期教学 `app/chat_client.py`
  不属于正式 V2 API 调用链，仍保留历史直连实现。
- 证据：[Stage 5.5 Completion Report](stage5-5-completion-report.md)

### Stage 6：Production Readiness & Portfolio

- 状态：**Completed**；最终边界为 `study-material-v2-stage6-final`
- 目标：把现有工程整理为可展示、可复现、可维护、可面试解释的作品；不新增核心业务功能。
- Stage 6.1：**Completed**；`c3c39e0` 新增 [Final Architecture](FINAL_ARCHITECTURE.md)，
  冻结系统图、上传/RAG/Gateway 数据流、Agent/Workflow、持久化、观测、安全与部署边界。
- Stage 6.2：**Completed**；`826f75a` 新增
  [Project Context](PROJECT_CONTEXT.md)、[Evaluation Guide](EVALUATION.md) 与
  [Development Guide](DEVELOPMENT_GUIDE.md)，并同步 README、Architecture 与 Decisions。
- Stage 6.3：**Completed**；`6f63f6a` 新增
  [RAG Evaluation Report](RAG_EVALUATION_REPORT.md)，汇总历史受控实验、失败案例与接入决策。
- Stage 6.4：**Completed**；`8554491` 新增
  [5 分钟 Demo Guide](DEMO_GUIDE.md)，覆盖演示主线、费用授权、故障恢复与 Gateway 展示边界。
- Stage 6.5：**Completed**；`c773d05` 新增
  [Interview Guide](INTERVIEW_GUIDE.md)，提供事实有界的项目介绍、追问答案和技术难点表达。
- Stage 6.6：**Completed in the final report commit**；新增 [Stage 6 Completion Report](stage6-completion-report.md)，
  最终全量自动化 `434/434`，并完成代码、文档、部署定义和敏感路径 Review。
- 证据边界：除文档外只修正 `.gitignore` 与 Compose 镜像标识；未验证 Docker、真实 Provider 或生产环境。

## 3. 当前正式主链路

当前正式 RAG 仍是 Stage 2 Retrieval baseline，加上 Stage 3.4 Context Selector：

```text
用户资料
  → Parser
  → fixed-character-180-v1 Chunker
  → Embedding
  → 用户独立 Chroma
  → Dense Top 10
  → relevance >= 0.25
  → 80% vector score + 20% keyword coverage
  → Top 3 Seed + 同源相邻扩展
  → EvidenceScoreContextSelector
  → Evidence Map
  → LCEL Prompt
  → Model Gateway（当前用户 BYOK，否则系统默认）
  → Provider Adapter / ChatModel
  → 服务端校验 Citation ID 并回填 metadata
  → Answer
```

BM25 + RRF、Cross-Encoder 和 Structure-aware Chunking 都是已完成、可复现但未接入正式主链路的实验能力。

## 4. 未开始与未验证

- **未实现：** Claude/Anthropic 原生 Adapter、多凭据、复杂任务 Router、自动 Provider
  故障切换、价格表/货币成本计算、流式产品 API、KMS/HSM 与主密钥轮换流程。
- **未验证：** Docker build/up、容器重启恢复、真实 Provider 与 Tutor 全路径质量、当前
  认证版本的真实付费 RAG 闭环、并发/负载、长时间运行、多实例一致性、凭据备份恢复和公开
  安全验收。

## 5. 持续同步规则

每次形成新的 Stage checkpoint 后，按以下顺序同步：

1. 用 `git status`、最近提交、全部 `study-material-v2-*` Tag 固定真实边界。
2. Completion Report 记录阶段目标、调用链、验证层级、Review 与未验证项；报告不能替代代码和 Git。
3. 更新本文的 commit、Tag、状态和证据链接；只有形成可信边界才标记 Completed。
4. 仅当真实调用链、存储或部署拓扑变化时更新 [ARCHITECTURE.md](ARCHITECTURE.md)。
5. 仅当出现有数据支撑、会影响后续方案的取舍时更新 [DECISIONS.md](DECISIONS.md)。
6. README 只保留项目介绍、运行方式与本文入口，不在多个位置复制完整状态表。
7. 完成后执行 `git diff --check` 与 `git status`，确认只修改文档。

若文档与代码不一致，先在本文记录差异与证据；不得为了让文档“看起来一致”而自行修改业务代码。

## 6. 当前发现的同步差异

- 用户提供的同步任务假设仓库停在 Stage 5.2，并要求“不要开始 Stage 5.3”；真实 `HEAD` 已是 `study-material-v2-stage5-part3`，用户身份与端到端数据隔离已经完成并形成 checkpoint。
- 仓库历史同时存在两个“Stage 5.1”语义：`stage5-part1` 是持久化学习数据基础；`stage5-part3` 的报告标题沿用了“Stage 5.1 User Identity”。本文按 Git part 编号同时保留两个历史事实，避免覆盖或改名历史报告。
- Stage 0、Stage 1、Stage 2 没有独立 Completion Report；Stage 0 也没有独立 Tag。它们的证据强度低于 Stage 3～5，本文已显式标注。
- README 同时保留早期“阶段 1～7”和 V2 Stage 1～5 两套历史编号，容易混淆。README 本轮只增加状态入口，不重写历史；V2 当前状态以本文为准。
