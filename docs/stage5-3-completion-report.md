# Study Material Assistant V2 · Stage 5.3 Technical Completion Report

## 1. 阶段目标

Stage 5.3 将本地可运行的 AI Learning Companion「知行」整理为可由统一配置启动、
具备单实例容器运行定义、持久化边界和基础健康检查的软件系统。

- `STAGE5_3_START_COMMIT`：`29e7c7e4e11781e26d8cf0dc3a523b83efe6acbd`
- 起点历史 checkpoint：`study-material-v2-stage5-part3`
- 本阶段 checkpoint：`study-material-v2-stage5-part4`

本阶段没有新增 AI 能力，也没有修改 RAG Pipeline、Retrieval、Embedding 算法、Chunk、
Evaluation Dataset、Tutor 路由、LangGraph 业务状态或 Prompt。

## 2. 当前问题

- 模型配置由客户端分别读取，监听地址和持久化根目录没有统一 Settings；
- SQLite、Job、Workflow、用户工作区与请求日志路径没有由同一运行配置派生；
- 没有本地与容器共用的启动模块；
- 没有 Dockerfile、Compose、build context 排除规则和持久卷定义；
- README 缺少新环境最短启动路径、环境变量、持久化边界和部署 FAQ；
- 已有 `/health` 尚未接入容器健康检查。

## 3. 服务化改造

### Completed

新增 `app.config.Settings`，统一管理 `APP_HOST`、`APP_PORT`、`APP_DATA_DIR`、百炼模型配置，
并从数据根目录派生学习数据库、Document Job 数据库、Workflow checkpoint、用户工作区、
请求日志和 Crawl4AI 运行目录。

新增 `app.server`，本地与容器统一通过 `python -m app.server` 启动 Uvicorn，并关闭可能
记录原始 URL 的默认 access log。

Settings 已进入真实调用链：

- `LearningDataStore.open(settings.learning_database_path)`；
- `DocumentJobService.open(database_path=settings.document_job_database_path)`；
- `open_sqlite_study_workflow_service(settings.study_workflow_database_path)`；
- 用户 Material/Chroma、RequestHistoryWriter 与 Crawl4AI 使用统一数据根目录；
- Embedding 与 Chat 客户端从同一 Settings 读取模型配置。

缺少 Key/Base URL 时，应用与 Health 仍可启动，只有真实模型请求会明确失败。

### 保持不变

API 继续负责 HTTP、Pydantic 校验、认证依赖和响应映射；业务操作继续委托现有 RAG、Tutor、
Document Job、Material、Learning Data 与 Study Workflow Service。当前结构已经存在真实
Service Layer，因此没有为了展示分层而大规模拆分 `app/api.py` 或制造 Repository 接口。

## 4. Docker 设计

### Implemented

- `Dockerfile` 使用 Python 3.12 slim，只复制运行文件，并以 UID/GID 10001 非 root 运行；
- 容器统一使用 `/app/data`，通过 `python -m app.server` 启动；
- Dockerfile 与 Compose 都使用标准库请求 `/health`，不依赖 curl；
- Compose 只定义一个 `app` 服务、`zhixing-data` named volume 和 bridge network；
- `.dockerignore` 排除 `.env`、`data/`、Git、虚拟环境、缓存、临时目录和本地评测结果。

没有拆 frontend、worker 或 database 容器：原生前端由 FastAPI 同源托管，Document Worker
依赖同进程锁、SQLite 与本地 Chroma；强拆会超出当前一致性边界。

### Not Verified

当前 Windows 环境没有安装 `docker` 命令，因此没有成功执行 `docker compose build` 或
`docker compose up`。已完成 Compose YAML 解析、静态契约测试和本地进程验证，不能声称
容器镜像已经构建或运行通过。

## 5. 配置管理

`.env.example` 包含 APP 监听、统一数据目录、Compose 宿主端口以及全部百炼配置。进程环境
优先于 `.env`，`.env` 继续被 Git 与 Docker build context 排除。本阶段没有输出、提交或
向外部服务发送真实密钥。

## 6. Health Check

复用已有 `GET /health`：

```json
{"status":"ok"}
```

它只证明 API 进程可响应，不打开 Chroma、不初始化 RAG、不检查真实模型凭证，也不调用
Embedding、ChatModel 或 Reranker。它是 liveness check，不是完整 readiness 证明。

## 7. 部署验证

### Completed

用独立临时数据目录和端口 8013 完成本机真实启动：

- `python -m app.server`：startup 完成；
- `GET /health`：HTTP 200；
- `GET /`：HTTP 200，包含“知行”和认证区域；
- `GET /static/app.js`：HTTP 200；
- Uvicorn 正常执行应用 shutdown；
- 临时目录只产生 631 字节安全请求日志，验证后已删除。

本轮没有提交资料、创建用户、打开真实用户 Chroma、重建索引或调用外部模型。

### Implemented but Not Runtime-Verified

Compose YAML 可解析，且结构确认只含 `app` 服务、named volume 与 bridge network；
Docker build、容器启动、volume 重启恢复和容器内页面访问未执行。

## 8. 测试结果

- Settings 专项：4/4；
- Deployment Contract 专项：4/4；
- FastAPI `test_api.py`：38/38；
- 全量 Fake/Mock/临时 SQLite 回归：400/400，27.477 秒；
- `python -m compileall app`：通过；
- `app/` 与 `tests/` 共 93 个 Python 文件内存编译：通过；
- `pip check`：`No broken requirements found.`；
- `node --check web/static/app.js`：通过；
- Compose YAML 结构解析与 `git diff --check`：通过。

第一次全量回归有一条旧测试仍断言 Workflow 数据库打开函数无参数调用。实现已改为显式
传入 Settings 路径；更新该契约测试后，400/400 重新通过。

## 9. SQLite → PostgreSQL 未来迁移分析

### 当前限制

SQLite、本地 Chroma、进程内 Worker 与 OperationGuard 只适合单应用实例。同一个 named
volume 不应被多个应用副本同时写入，当前也没有生产备份、负载或故障切换验收。

### Future，未实施

出现多实例、独立 Worker、共享事务吞吐或正式灾备需求后，需要：

1. 适配 `LearningDataStore`、`DocumentJobStore` 的 PostgreSQL SQL、连接池和事务并发；
2. 建立版本化 Migration、数据校验、停机切换与回滚；
3. 保留 Auth Session、复合外键、Job owner 与幂等约束；
4. 将 `AsyncSqliteSaver` 替换为兼容的 PostgreSQL checkpointer；
5. 为独立 Worker 设计事务领取、租约、崩溃恢复与费用幂等；
6. 另外设计共享向量/文件存储，不能只迁移业务数据库就宣布多实例安全。

本阶段没有安装 PostgreSQL 驱动、创建 Schema、运行 Migration 或修改现有数据。

## 10. Review

Review 覆盖需求边界、Settings 真实调用链、Service 委托、Compose 拓扑、持久卷、秘密排除、
Health 零模型调用、文档证据层级及是否越界修改 AI/RAG。

Review 中发现并修正：

- 多行补丁传输曾遗漏 `api.py` 的导入与路径参数，在首次编译前修复，未进入提交；
- Workflow API 旧测试未反映新的显式数据库路径契约；
- FastAPI 兼容 MaterialManager 最初仍使用默认目录，后改为统一 `APP_DATA_DIR`；
- 历史文档把 `stage5-part3` 写成“Stage 5.3”，已改为“Stage 5 / part3”。

没有发现 RAG、Retrieval、Embedding 算法、Chunk、Evaluation、Tutor、Prompt 或数据
Migration 的越界改动。

## 11. 完成状态与边界

| 状态 | 内容 |
| --- | --- |
| **Completed** | 统一 Settings、共用启动入口、本机 Health/页面验证、README、文档与全量回归。 |
| **Implemented** | 非 root Dockerfile、单服务 Compose、持久卷、网络、环境传入与容器 Healthcheck。 |
| **Not Implemented** | PostgreSQL、Redis、Kubernetes、CI/CD、云部署、微服务、分布式 Worker、复杂 readiness/observability。 |
| **Not Verified** | Docker build/up、容器重启恢复、公开 URL、HTTPS、负载、渗透与生产备份恢复。 |
| **Future** | 由多实例/独立 Worker 等真实需求触发数据库、共享存储与队列迁移。 |

Stage 5.3 的结论是“具备明确、可重复的本地启动和容器部署定义”，不是“生产级部署完成”
或“云环境已经上线”。
