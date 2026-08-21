# Study Material Assistant V2 · Stage 5.1 Completion Report

## 1. 修改目标

Stage 5.1 将 Stage 4 的单进程、`InMemorySaver` Tutor Session 升级为本地可持久化的
用户学习数据基础。它回答三个工程问题：用户由什么 ID 标识、学习数据保存在哪里、服务
关闭重开后如何继续同一学习 Session。

本阶段起点为 `4d0fbcd039f8a9ba66c52a1cf6ce8aef79d22fa5`
（`study-material-v2-stage4`）。实现没有修改 Retriever、Embedding、Chunker、Context
Selector、RAG Prompt、旧 Agent 工具或 Study Workflow 路由。

## 2. 原系统问题

Stage 4 的 `app/tutor_workflow.py` 使用 `InMemorySaver`。同一进程内可以按
`session_id` 复用最近 20 条、合计 12000 字符的 conversation 和 topic，但存在以下
边界：

- 服务退出后 checkpoint 和 Tutor 对话全部丢失；
- `session_id` 没有关联用户，不能验证 Session 归属；
- 没有独立的完整对话历史和学习行为记录；
- LangGraph State 同时承担当前推理输入和唯一 Session 数据来源，不适合作为长期业务
  数据模型。

仓库此前没有用户业务数据库，但已经锁定 `aiosqlite==0.22.1` 与
`langgraph-checkpoint-sqlite==3.1.0`，并在另一条 Study Workflow 中验证过本地 SQLite
checkpoint 的关闭重开恢复。

## 3. 新增数据模型

默认数据库为 `data/learning/learning.sqlite3`。该目录已加入 `.gitignore`。业务 SQL
与 LangGraph checkpoint 使用同一数据库文件、两个独立 `aiosqlite` 连接；连接启用
WAL 和 5 秒 busy timeout，业务连接额外启用外键约束。

| 表 | 关键字段 | 关系与用途 |
| --- | --- | --- |
| `schema_migrations` | `version`, `applied_at` | 记录已应用的显式 Schema 版本。 |
| `users` | `user_id`, `created_at` | 服务端生成 UUID；不保存密码、邮箱或权限。 |
| `learning_sessions` | `session_id`, `user_id`, `topic`, `created_at`, `updated_at` | Session 通过外键属于一个用户。 |
| `conversation_messages` | `message_id`, `session_id`, `message_order`, `role`, `content`, `intent`, `created_at` | 保存完整 user/tutor 消息；Session 内顺序唯一且稳定。 |
| `learning_records` | `record_id`, `user_id`, `session_id`, `topic`, `activity_type`, `metadata_json`, `created_at` | 保存一次成功 Tutor 行为；metadata 当前只记录工具名，不复制消息正文。 |
| LangGraph checkpoint 表 | 由 `AsyncSqliteSaver.setup()` 管理 | 保存线程级短期推理状态和恢复点。 |

首次打开数据库时，程序在事务中应用版本 1 迁移并记录版本。重复打开是幂等的；数据库
版本高于当前程序支持版本时会拒绝打开，不自动降级或猜测迁移。当前没有引入 ORM、
Alembic 或独立迁移 CLI；后续 Schema 变化应增加顺序迁移，而不是改写已发布版本。

## 4. 数据流变化

当前 Tutor 数据流为：

```text
user_id + session_id + message
  ↓
按 user_id + session_id 校验 Session 归属
  ↓
从业务表加载最近有限对话和持久化 topic
  ↓
Tutor LangGraph（当前推理短期 State）
  ↓
AsyncSqliteSaver 同步写入线程 checkpoint
  ↓
Tutor 成功结果
  ↓
一个业务事务写入 user/tutor 消息、learning record，并更新 Session topic
```

长期历史不是通过无限 Prompt 拼接实现。每次调用都以业务表作为历史来源，只选最近
20 条并继续受 12000 字符上限约束；LangGraph State 保存当前执行所需的有限副本，
下次调用会由数据库历史覆盖。服务关闭重开后，新 graph 使用相同 `thread_id` 的 SQLite
checkpoint，同时重新读取独立业务历史。

业务消息事务与 LangGraph checkpoint 不是跨连接原子事务：如果 graph 已成功但业务
事务失败，API 返回 Tutor 处理失败，不会把未持久化结果报告为成功；下一次调用仍以业务
历史为准。要获得跨实例、强原子语义，需要后续引入幂等请求或 outbox，而不是在本阶段
加入分布式事务。

## 5. API 变化

新增：

- `POST /api/users`：创建用户，返回 `user_id`、`created_at`；
- `GET /api/users/{user_id}`：查询用户；
- `POST /api/users/{user_id}/sessions`：用 `topic` 创建 Session；
- `GET /api/users/{user_id}/sessions`：返回最近最多 100 个 Session；
- `GET /api/users/{user_id}/history`：返回最近最多 100 条消息和 100 条学习记录；可用
  `session_id` 查询参数限定 Session。

调整：

```json
POST /api/tutor/chat
{
  "user_id": "<UUID>",
  "session_id": "<UUID>",
  "message": "帮我出题练习 Embedding",
  "confirm_api_cost": true
}
```

Tutor 只接受已存在且属于该用户的 Session。归属不匹配与不存在统一返回 `404`，避免
通过响应区分“存在但属于别人”。数据库内部异常统一映射为不含路径和原始异常的 `503`。
请求历史日志继续排除 `user_id`、`session_id`、消息、回答和 Evidence。

`/api/ask`、`/api/agent`、`/api/study-workflows` 以及已有资料接口未改契约。Stage 4 的
`/api/tutor/chat` 请求新增必填 `user_id`，这是接入用户归属校验所需的显式 Stage 5.1
契约变化。

## 6. 为什么这样设计

本阶段选择 SQLite，而不是立即引入推荐的 PostgreSQL，原因是当前事实仍是本地单实例
学习产品原型：没有生产并发、横向扩容或托管数据库需求证据；仓库已经具备 SQLite 的
异步依赖、序列化安全设置和恢复测试。SQLite + 显式 SQL 可以在不增加数据库服务、
连接池、ORM 和部署配置的情况下完成当前验收目标。

当需求变为公开多用户、多个 API 实例、共享事务吞吐或生产备份恢复时，应迁移到
PostgreSQL，并使用对应的 `AsyncPostgresSaver` 和正式迁移工具。当前业务表与 Tutor
服务的边界已把这项替换限制在数据层和初始化层，不需要改 RAG Pipeline。

## 7. 验证结果

本轮没有读取 `.env`，没有调用真实 Embedding、ChatModel、Reranker 或其他付费 API。

- Stage 5.1 专项：26/26 通过；
- 全量自动化回归：369/369 通过；
- 临时真实 SQLite：用户创建/查询、Session 创建/列表、归属隔离、消息与学习记录查询、
  Schema 幂等打开通过；
- Tutor 重启恢复：关闭第一个数据库连接，重新打开数据库并创建新 graph 后，能够基于
  原 Session topic 和历史执行 Session Summary；
- Python 内存编译：`app/` 与 `tests/` 共 82 个 Python 文件通过；
- `pip check`：通过；
- `node --check web/static/app.js`：通过；
- `git diff --check`：通过。

自动化测试使用 Fake/Mock 工具和临时本地 SQLite，证明代码契约与恢复行为，不证明真实
模型质量、公开网络安全、并发负载或生产数据库可靠性。本阶段没有启动 Uvicorn 做人工
HTTP 验收，也没有执行真实模型 Tutor 对话。

强制 Review 覆盖需求范围、实际 diff、迁移/事务、连接生命周期、Session 归属、消息
顺序、错误脱敏、请求日志、LangGraph 与长期数据边界、旧 API 回归和 RAG 隔离。Review
中发现并修复：同毫秒消息顺序不稳定、Session Summary 未继承持久化 topic、数据库目录
未忽略、重复消息索引和连接关闭缺少 `finally`。修正后重新执行专项与全量回归，无剩余
阻断项。

## 8. 未实现内容

本阶段明确没有实现：

- Long-term Memory Agent、用户画像或推荐系统；
- Multi-Agent、MCP、A2A、Autonomous Agent；
- OAuth、SSO、密码登录、RBAC、企业账号或用户级费用配额；
- PostgreSQL、外部数据库服务、跨实例锁、分布式事务或异步任务；
- 数据库加密、密钥轮换、备份恢复、数据保留/删除 API；
- Docker、部署、公开服务或生产并发验收。

当前 UUID + 外键查询只是数据分区基础，不是认证授权。知道其他用户 UUID 的调用者仍
可能读取其 API 数据；本地 SQLite 也未加密。因此当前实现只适合受信任本机原型，不应
保存密码、令牌等敏感凭据，也不能直接作为公网多用户系统。
