# Study Material Assistant V2 · Stage 5.2 Completion Report

## 1. 阶段目标与事实边界

Stage 5.2 将“用户确认费用后的知识库构建”从同步 HTTP 请求中移出，改为持久化 Job 与
后台 Worker。起点为 `69e9a9f3fe7ca6ae5194c5540baf35d5a3ba97ca`
（`study-material-v2-stage5-part1`）。

本阶段解决的是请求与耗时索引工作的解耦，不等于已经具备大规模生产吞吐。当前实现仍是
本地单实例、单 Worker 原型；没有负载测试、多机队列或公开部署证据。

另一个必须明确的边界是：`/api/materials/stage` 与 `stage-batch` 仍会在请求内完成零费用
的本地文件校验和预解析，以便在用户确认真实 Embedding 费用前展示 Chunk 与批次数。
异步化从“确认写入索引”开始；确认接口不再等待后续重解析、Embedding 和 Chroma 同步。
因此，本阶段没有把上传字节落盘本身或费用预览改成完全异步。

## 2. 原同步流程问题

Stage 5.1 基线中，用户确认费用后，`POST /api/materials/{upload_id}/index` 或批量入口会在
同一个请求内依次执行：

```text
费用确认 → 重新解析资料 → Chunk → 计算实际 Embedding 批次
        → Embedding → Chroma 增量同步 → RAG 缓存失效 → 返回结果
```

请求要一直等待外部 Embedding 和本地索引完成。资料增多或外部服务变慢时，这会增加超时、
断线后状态不明和客户端重复提交的风险；处理中状态也没有独立、可查询的持久化记录。

## 3. 新异步架构

当前流程为：

```text
本地暂存与费用预览
        ↓
用户显式 confirm_api_cost=true
        ↓
SQLite 创建 pending Job
        ↓
HTTP 202 + job_id 立即返回
        ↓
单进程 asyncio Worker 串行领取 Job
        ↓
processing → 复用既有费用保护与 MaterialManager
        ↓
Parser / Chunker / Embedding / Chroma 增量同步
        ↓
completed + 结果摘要，或 failed + 安全错误信息
```

Worker 使用 `asyncio.to_thread()` 承载现有同步摄取链，避免阻塞 FastAPI 事件循环。它没有
创建第二套 Parser、Chunker 或索引代码，也没有修改 Retrieval、Context Selector、
Evidence/Citation、Reranker 或 Prompt。

## 4. Job 数据模型

默认数据库为 `data/jobs/document_jobs.sqlite3`，目录已加入 `.gitignore`。Schema 版本 1
包含 `schema_migrations` 与 `document_jobs`。Job 保存：

- `job_id`；
- 一次单文件或批量请求的 `upload_ids` 与 `filenames`；
- `pending / processing / completed / failed` 状态；
- `progress`；
- 安全、限长的 `error_message`；
- 完成后的 added/deleted/unchanged/cleanup_pending 摘要；
- created/started/finished/updated 时间字段。

数据库不保存文件正文、Chunk、Prompt、模型输出、密钥或异常堆栈。同一组 upload ID 在
`pending` 或 `processing` 期间由 SQLite 部分唯一索引去重；顺序不同也视为同一组任务。

当前资料库仍是全局本地资料库，不是 Stage 5.1 用户私有 Workspace。因此 Job 没有伪造
`user_id` 归属字段；只增加 Job 用户字段而不隔离文件和 Chroma 数据，会制造错误的安全
印象。用户级资料隔离应在 Workspace/权限模型建立后整体实现。

## 5. Worker、失败与恢复

Worker 每次只处理一个 Job，并在真正执行前继续取得现有 `OperationGuard` 的 `index`
租约。实际待新增 Embedding 批次数仍由 `estimate_index_batches*()` 计算并预留，随后调用
`commit_staged*()`。因此：

- Legacy Index / Manifest 不兼容仍在 Embedding 前失败；
- 单次和进程级费用上限继续生效；
- 单文件和批量文件移动、索引同步及补偿回滚继续复用原实现；
- 成功后才失效现有 RAG/Agent/Tutor 缓存。

普通失败会落为 `failed`，并保存不含路径、密钥、正文和堆栈的错误信息。回滚结果不确定时，
错误提示要求检查资料与索引状态，同时主动失效 RAG 缓存。

服务重启时，持久化 `pending` Job 会继续消费。上次进程已经标为 `processing` 的 Job 不会
自动重试，而是转为 `failed` 并要求重新提交。原因是进程可能恰好在外部 Embedding 或
文件/索引切换期间退出，盲目自动重试既可能重复付费，也可能掩盖需要人工核对的一致性
问题。当前没有自动 Retry 或指数退避。

## 6. API 与页面变化

- `POST /api/materials/{upload_id}/index`：保留 `confirm_api_cost: true`，改为返回
  `202 Accepted` 与 Job 状态；
- `POST /api/materials/batch/index`：一次批量确认创建一个原子批量 Job；
- `GET /api/jobs/{job_id}`：返回状态、粗粒度进度、时间字段、失败原因和完成摘要；
- Web 页面在创建 Job 后轮询状态，完成后刷新资料列表，失败时展示 Worker 保存的安全
  原因。

进度当前只表示阶段：`pending=0`、`processing=10`、`completed=100`。它不是解析页数、
Embedding 批次或真实剩余时间百分比，避免展示无法证明的精确进度。

## 7. 技术选择

选择 SQLite + asyncio Worker，而不是 Redis Queue 或 Celery，原因是当前事实仍是本地
单实例产品原型：

- 已有 `aiosqlite` 依赖，不增加外部服务、Broker、部署与运维面；
- SQLite 足以保存任务状态、去重和重启后的 pending 恢复；
- 单 Worker 与 Chroma 本地写入、现有进程级 `OperationGuard` 的边界一致；
- 后续若出现多实例、水平扩展、独立 Worker 部署、优先级、重试调度或高吞吐证据，再将
  Job Store/领取接口替换为 PostgreSQL + 专用队列，而不是现在提前建设。

## 8. 验证结果

本轮没有读取 `.env`，没有调用真实 Embedding、ChatModel、Reranker 或其他付费 API，
也没有写入或重建真实 Chroma Index。

- Stage 5.2 + API 专项：45/45 通过；
- 全量 Fake/Mock 自动化回归：376/376 通过；
- 临时真实 SQLite：Schema 初始化、任务创建/读取、活动任务去重、时间字段、处理中断恢复
  和关闭重开后的 pending 继续执行通过；
- Worker：单文件与批量 `pending → processing → completed`、Manifest 只读失败转
  `failed`、失败不调用 commit、成功保存索引摘要并失效 RAG 缓存通过；
- API：费用确认校验、`202 + job_id`、批量 Job、活动任务冲突和完成状态查询通过。
- Python 内存编译：`app/` 与 `tests/` 共 84 个 Python 文件通过；
- `pip check`、`node --check web/static/app.js` 与 `git diff --check` 通过；
- 本地 Uvicorn + 应用内浏览器零费用检查：页面有实际内容、无错误遮罩、浏览器控制台无
  warning/error，新任务按钮正确渲染，问答表单与任务按钮均只有一个实例。

本地 Uvicorn 检查期间，服务端观察到既有 `data/request_logs` 目录无法写入并按原设计输出
`request_history_write_failed` warning；HTTP 页面与 API 未受影响。本阶段没有修改请求日志
模块或该忽略目录的权限，因为它不属于异步文档任务范围。自动化日志测试在临时可写目录
中通过，但当前目录权限仍是一个独立的本地环境待处理项。

自动化测试证明当前代码契约和临时 SQLite 恢复行为，不证明真实模型调用、真实大文件
耗时、浏览器长时间轮询、并发吞吐、进程强杀时的文件/索引一致性或生产可靠性。

## 9. Review

Review 覆盖需求范围、实际 diff、Worker 生命周期、SQLite 事务、活动任务去重、启动恢复、
异常路径、错误脱敏、费用保护、Manifest/回滚调用链、RAG 缓存失效、API 契约、页面轮询、
请求日志与 RAG Pipeline 隔离。

Review 中发现并修正：同一批 upload ID 交换顺序后可重复建活动任务、Worker 异常退出时
数据库连接可能跳过关闭、页面状态查询错误被统一覆盖为“无法连接”。修正后重新执行专项
与全量回归；没有剩余自动化阻断项。

## 10. 未实现内容

本阶段明确没有实现：

- Redis、Celery、RabbitMQ、Kafka 或分布式任务系统；
- Kubernetes、自动扩缩容、多实例 Worker 或跨实例锁；
- 自动重试、取消、优先级、延迟任务、任务保留/清理 API；
- 用户级 Workspace、资料归属、认证、RBAC 或 Job 访问控制；
- 精确的解析/Embedding 百分比和预计完成时间；
- 从上传字节落盘开始的完全异步预解析；
- 真实模型、真实大文件、并发、负载、故障注入、部署或生产验收。

本阶段在此停止，不继续 Docker、Deployment、Observability、Multi-Agent、MCP 或 A2A。
