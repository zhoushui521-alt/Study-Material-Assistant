# Study Material Assistant V2 · Stage 5.4 Observability Completion Report

## 1. 本阶段目标

Stage 5.4 将 AI Learning Companion「知行」从“能够运行”推进到“能够观察、分析和定位问题”的
本地单实例 AI 应用原型，建立以下最小闭环：

```text
Request → Trace → Logs → Metrics → Evaluation（保持独立）→ Problem Diagnosis
```

- `STAGE5_4_START_COMMIT`：`6f0e770`
- 起点 Tag：`study-material-v2-stage5-part4`
- 5.4.1 commit：`e7b410a`
- 5.4.2 commit：`1101453`
- 5.4.3 commit：`616dbd4`
- 5.4.4：本报告、ADR、最终验证与状态同步所在提交
- Stage 5.4 Tag：未创建
- 远端：未 push

本阶段没有修改 RAG Retrieval、Chunk、Embedding、Reranker、Prompt、Tutor 决策、
Evaluation Dataset 或正式指标计算逻辑。

## 2. 修改内容

### 5.4.1 Structured Logging Foundation

新增 `app.observability`，使用 Python 标准库 Logging 输出单行 JSON。每条事件固定包含：

```json
{
  "time": "",
  "level": "",
  "service": "zhixing",
  "request_id": null,
  "user_id": null,
  "event": "",
  "duration_ms": null
}
```

HTTP Middleware、应用启动/关闭、RAG 清理等事件迁移到统一 Logger。详情字段采用白名单，
不记录问题、回答、Prompt、文件正文、密钥、异常详情或 URL 查询参数。既有
`RequestHistoryWriter` 继续轮转保存最小 HTTP 完成记录，不被包装成完整 Trace Store。

### 5.4.2 Request Trace

使用 `ContextVar` 绑定 `request_id` 和认证后的可信 `user_id`，并覆盖：

- FastAPI Middleware 与同步线程池；
- `RAGService`、Retriever、RAG ChatModel；
- Tutor Quiz / Summary 结构化模型调用；
- Document Job enqueue、processing、completed、failed。

`asyncio.to_thread` 自动复制请求上下文。长期 Document Worker 创建时使用空
`Context`，防止首个请求的 `request_id` 泄漏到后续 Job；异步阶段使用稳定 `job_id`
关联。

LLM token usage 只读取 LangChain 返回的 metadata；不可获得时记录
`token_usage_available=false`，不估算。

### 5.4.3 AI Metrics

新增线程安全的单进程 `RuntimeMetrics`，聚合：

- System：请求总数、成功/失败数、成功率、错误率、平均响应时间；
- Retrieval：调用/失败/空召回次数、召回 Chunk 总数、平均召回数、平均耗时；
- LLM：调用/失败次数、模型名、平均耗时、可获得 usage 的调用数与 token 总数；
- Document Processing：完成/失败数量与平均处理时间。

新增认证接口 `GET /api/observability/metrics`。它只返回匿名全局聚合，不返回
`request_id`、`user_id`、问题、回答或资料内容。Metrics 查询本身在响应完成后计数，
所以快照不包含正在读取快照的当前请求。

### 5.4.4 Observability Integration

评估 OpenTelemetry 与 LangSmith 后，本阶段不安装、不连接：

- 当前只有一个 FastAPI 进程和进程内 Worker，没有跨服务 Trace 的真实问题；
- 本地 Logging + Context Trace + Runtime Metrics 已覆盖当前问题定位目标；
- 外部 Collector/平台会新增部署、网络、采样、保留期、数据授权、费用和故障边界；
- Prompt、学习资料和模型输出进入外部平台前必须另行定义数据分级与用户授权。

决策已记录到 `docs/DECISIONS.md`。这不是永久拒绝，而是由多实例、跨进程 Trace、长期
留存、SLO/告警或经批准的模型质量追踪需求触发重新评估。

## 3. 新增能力

系统现在可以回答：

1. 一个 HTTP 请求的 `request_id`、认证用户、开始/结束、状态码和耗时；
2. 同一同步请求是否进入 RAG、Retriever、LLM 或 Tutor 模型边界；
3. Retrieval 返回多少 Chunk、是否空召回、是否失败、耗时多少；
4. LLM 使用哪个模型、是否失败、耗时多少，以及返回中是否包含 token usage；
5. Document Job 从入队到成功/失败的状态变化和处理耗时；
6. 当前进程的请求、Retrieval、LLM 与文档任务整体聚合状态；
7. 422 等错误请求属于什么稳定错误类别，并通过 `request_id` 定位对应日志。

这些能力用于运行问题定位；它们不替代 Retrieval/Answer Evaluation，也不证明回答质量、
并发能力或生产健康。

## 4. 技术选型原因

选择标准库 Logging、`ContextVar`、线程锁和 FastAPI 认证接口，原因是它们直接复用当前
单服务调用链，无新依赖、无外部数据发送、无付费调用，也不要求部署 Collector 或数据库。

没有把每个日志事件写入 SQLite：日志和 Metrics 的当前目标是进程内诊断，不是构建另一套
Trace 数据库。没有直接接 OpenTelemetry/LangSmith：当前不存在足以抵消接入、运行、安全和
维护复杂度的跨进程或长期分析需求。

## 5. 架构变化

```text
HTTP Request
  → Middleware：request_id / elapsed / status
  → ContextVar：request_id + trusted user_id
       ├─ RAGService
       │    ├─ Retriever events + metrics
       │    └─ LLM events + available usage metrics
       ├─ Tutor model events + metrics
       └─ Document Job enqueue
             → empty-context Worker
             → job_id status events + metrics

Structured JSON Logs                 RuntimeMetrics
  → event-level diagnosis              → anonymous process snapshot
  → request_id / job_id correlation    → authenticated GET endpoint
```

业务链路的输入、输出和正式 RAG 参数没有改变。Evaluation 继续使用独立 Dataset、版本、
口径和报告，不与运行 Metrics 混为同一证据。

## 6. 测试结果

### 自动化

- Stage 5.4 开始前全量回归：`400/400`；
- 5.4.1 Logging / Request History / API 专项：`45/45`；
- 5.4.2 Logging / API / RAG / Tutor / Document Job 专项：`97/97`；
- 5.4.3 Metrics 集成专项：`102/102`；
- 最终 42 个 `test_*.py` 模块全量回归：
  - 批次 1：`219/219`；
  - 批次 2：`198/198`；
  - 合计：`417/417`；
- `app/` 与 `tests/` 共 97 个 Python 文件内存编译：通过；
- `pip check`：`No broken requirements found.`；
- `git diff --check`：通过。

全量测试采用 Fake/Mock 与临时 SQLite/文件目录，没有调用真实 Embedding、ChatModel、
Reranker 或其他付费外部 API。

### 手动本地验证

使用独立临时数据目录与端口 `18014` 启动真实 FastAPI 进程：

- `GET /health`：HTTP 200，响应包含 `X-Request-ID`；
- 临时用户注册：HTTP 201；
- 认证 `GET /api/observability/metrics`：HTTP 200；
- 提交空白问题：HTTP 422，不进入 Retrieval/LLM；
- 下一次 Metrics 快照：4 个已完成请求，3 成功、1 失败，Retrieval/LLM/Document 均为 0；
- 422 日志由同一 `request_id` 关联 started/completed，`error_category=request_validation`；
- 日志不包含问题字段或临时测试密码；
- 验证后停止进程并删除临时数据目录。

本轮没有打开正式用户 Chroma、提交资料、重建索引或调用外部模型。

## 7. 当前限制

- Metrics 只存在于当前 Python 进程，重启清零，多实例不聚合；
- 只有累计平均值，没有 P50/P95/P99、时间窗口、直方图、Dashboard 或告警；
- 没有持久化 Trace Store，跨请求分析依赖控制台/容器日志采集；
- 已认证用户可读取匿名全局聚合，但项目没有管理员角色或 RBAC，不适合直接公开该接口；
- 没有负载、长时间运行、日志吞吐、采样策略或高基数压力验证；
- token usage 取决于模型响应 metadata，不可获得时不会估算；
- 运行 Metrics 与 Evaluation Dataset/Case 没有自动关联；
- 没有接入 OpenTelemetry、LangSmith、Prometheus 或生产日志平台；
- Docker build/up、公开部署、HTTPS、生产备份、SLO 和告警均未验证。

## 8. 下一阶段建议

不要立即叠加监控组件。先在真实本地使用中收集具体故障模式，并按触发条件演进：

1. 出现慢请求定位需求时，先定义窗口和 P95/P99 口径，再决定直方图或 Metrics 后端；
2. 出现多实例、独立 Worker 或跨服务调用时，再引入 OpenTelemetry Trace/Metric Exporter；
3. 需要把模型链路与质量评测关联时，先建立独立 Dataset、数据分级、脱敏和费用上限，再评估
   LangSmith 或自建 Trace Store；
4. 对外部署前增加管理员权限、接口暴露策略、采样、保留期、告警和容量验证；
5. 保持 Evaluation 与运行监控分层：运行指标用于发现“哪里异常”，评测用于判断“质量是否达标”。

## 9. Review

Review 覆盖需求边界、真实调用链、RAG 参数、异步 Context 隔离、日志敏感字段、指标口径、
认证接口、异常路径、测试有效性、文档证据层级和 Git 差异。

Review 中发现并修复：

- 早期补丁曾把 `_safe_text` 返回与字段集合括号放错位置，内存编译后立即修复；
- 文档标题在补丁中一度遗漏，检查实际 diff 后恢复；
- Windows 补丁重计数最初只写入 `tests/test_metrics.py` 首行，Review 发现后补全，并把
  聚焦验证从无效的 99 项更正为真实包含 3 个 Metrics 单测的 102 项；
- 第一次全量 discovery 未显式指定 `tests/`，发现 0 项后废弃该结果，改为覆盖全部
  42 个测试模块的稳定双批次执行；
- 临时 SQLite `-shm` 文件在服务停止瞬间短暂占用，确认端口释放后对同一已校验路径重试，
  最终完成临时目录清理。

没有发现对 Retrieval、Chunk、Embedding、Reranker、Prompt、Tutor 决策、Evaluation 或
正式索引/用户数据的越界改动。

## 10. 完成状态

| 状态 | 内容 |
| --- | --- |
| **Completed** | Structured Logging、Request Trace、Runtime Metrics、认证快照接口、ADR、自动化与本地手动验证。 |
| **Not Implemented** | OpenTelemetry、LangSmith、Prometheus、Trace Store、Dashboard、告警、管理员 RBAC。 |
| **Not Verified** | 真实模型全链路、负载/长稳、多实例、容器、公开部署、生产 SLO 与告警。 |
| **No External Cost** | 本阶段真实 Embedding / ChatModel / Reranker / 外部 Observability 调用均为 0。 |

Stage 5.4 的结论是“具备适合当前本地单实例原型的基础可观测性闭环”，不是“已经完成生产级
监控平台”或“AI 回答质量已经通过生产验收”。
