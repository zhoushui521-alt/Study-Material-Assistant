# Study Material Assistant V2 · Stage 5.1 Technical Completion Report

## 1. 阶段范围与起点

本阶段把已有“UUID 用户 + 持久化学习数据”的数据分区基础，升级为后端可验证的身份与
端到端所有权边界。起点提交为
`b461166450848e36238f42ff83337ee75612736e`。

仓库历史上已经有一个名为 Stage 5.1 的持久化 checkpoint 及
`docs/stage5-1-completion-report.md`；它只实现 UUID、Session 和学习记录，没有认证。
为保留历史事实，本报告使用新文件名，当前 Git checkpoint 按用户指定命名为
`study-material-v2-stage5-part3`。

本阶段没有加入 PostgreSQL、Docker、Redis、OAuth、RBAC、Observability、BYOK、Agent
升级或 RAG 效果优化。

## 2. 原系统为什么不能算多用户

原系统虽然已经有 `users`、`learning_sessions`、`conversation_messages` 和
`learning_records`，但客户端自行保存并提交 `user_id`。后端无法证明请求者就是该 UUID
对应的人，因此知道或猜到其他 UUID 后仍可能读取对方 Session 与历史。

资料、暂存上传、Chroma 和后台 Job 还是全局目录与全局服务实例。即使业务表按 UUID
过滤，RAG 仍可能从另一人的 Chunk 生成回答、文件名、摘录、`material_id` 与 Citation。
这说明“有 User 表”不等于“有用户系统”，登录页面也不等于数据隔离。

## 3. 用户与学习数据关系

当前关系为：

```text
User
  ├─ Auth Session（可撤销登录状态）
  ├─ User Workspace
  │    ├─ Documents / Pending Uploads / Pending Deletions
  │    └─ Chroma Vector Store → Evidence → Citation
  ├─ Learning Session
  │    ├─ Conversation Message
  │    ├─ Learning Record
  │    └─ Tutor Memory / SQLite Checkpoint
  ├─ Document Job
  └─ Study Workflow Checkpoint（状态中保存 user_id 并校验所有权）
```

数据归属规则：

| 资源 | 当前归属实现 | 关键约束 |
| --- | --- | --- |
| User | `users.user_id` | 邮箱规范化后唯一；密码只保存哈希。 |
| Auth Session | `auth_sessions.user_id` | 数据库只保存 token SHA-256；删除即撤销。 |
| Learning Session | `learning_sessions.user_id` | Session 只能由当前用户创建和查询。 |
| Conversation | `conversation_messages.user_id` | 同时以 `(session_id, user_id)` 复合外键约束。 |
| Learning Record | `learning_records.user_id` | 同时以 `(session_id, user_id)` 复合外键约束。 |
| Material | 后端解析出的用户工作区 | API 不接受 owner 参数；目录由 current_user UUID 决定。 |
| Vector Data | 用户工作区内独立 Chroma | RAG 只打开当前用户 vector store。 |
| Document Job | `document_jobs.user_id` | 创建、状态查询和 Worker manager 都绑定 owner。 |
| Study Workflow | LangGraph state 的 `user_id` | get/confirm/progress/retry/delete 均校验 owner。 |

## 4. 认证方案

本阶段选择可撤销的服务端 Session，而不是 JWT。

注册流程：

```text
email + password + display_name
  → 后端规范化和长度校验
  → scrypt（随机 salt）
  → users.password_hash
  → 随机不透明 token
  → auth_sessions 只保存 SHA-256(token)
  → HttpOnly Cookie
```

登录时，无论邮箱不存在还是密码错误，都会执行一次 scrypt 验证并统一返回“邮箱或密码
错误”，减少账号枚举和明显的时序差异。密码长度为 10～128 字符，未新增第三方认证依赖。

浏览器 Cookie 使用 `HttpOnly`、`SameSite=Strict`、`Path=/` 和 7 天有效期；HTTPS 请求会
增加 `Secure`。API 也接受 Bearer 形式的不透明 token，但当 Header 与 Cookie 同时存在且
不一致时直接拒绝。数据库泄露时不会直接暴露可用 Session token。

选择服务端 Session 的原因：当前是同源原生前端、单实例 FastAPI 和 SQLite，退出需要
立即撤销；使用 JWT 仍需处理密钥轮换、撤销名单和客户端安全存储，复杂度没有带来当前
收益。代价是未来多实例部署必须共享 Session Store；迁移时可保留 `current_user` 业务
边界，只替换凭证验证层。

## 5. 数据库与兼容迁移

`app/learning_data.py` 的 Schema 从 v1 升级到 v2：

- `users` 增加 `email`、`password_hash`、`display_name`、`updated_at`；
- 新增规范化邮箱唯一索引；
- 新增 `auth_sessions` 与用户、过期时间索引；
- `conversation_messages` 增加直接 `user_id`，并重建为复合 Session/User 外键；
- `learning_records` 重建为复合 Session/User 外键；
- 为 `learning_sessions(session_id, user_id)` 增加父键唯一约束。

迁移在 SQLite 事务中执行。旧消息通过所属 Session 反填 `user_id`，旧学习记录原样复制，
不会删除已有学习数据。旧用户没有 email/password_hash，因此不能直接通过认证系统登录，
也不会被新注册用户冒领。

`app/document_jobs.py` 的 Schema 从 v1 升级到 v2并增加 `user_id`。旧 `pending` 或
`processing` Job 因没有可靠 owner 会明确标记为 failed；已完成历史仍保留。Worker 只领取
非空 owner 的任务，API 查询必须同时匹配当前用户。

旧的 `data/documents/` 和 `data/vector_store/` 保持原位，不删除、不移动、不自动归属。
新认证用户使用 `data/user_workspaces/<uuid>/...`。本阶段不创建“默认用户迁移”，因为把
旧全局资料自动分给第一个注册者会形成更严重的数据泄露。若未来需要认领，应另做停服、
显式指定 owner、可校验和可回滚的迁移。

## 6. RAG、Evidence 与 Citation 隔离

本阶段选择每用户独立持久化目录，而不是在同一个 Chroma collection 上只依赖 metadata
过滤：

```text
current_user.id
  → data/user_workspaces/<uuid>/vector_store
  → create_rag_service(user_vector_store)
  → Retrieval / Context / Evidence / Citation
```

当前 `MaterialManager` 和 Index Manifest 的设计假设一个实例拥有完整语料；独立目录能复用
现有增量索引、回滚、Manifest 与检索调用链，并在 Retriever 忘记 metadata filter 时仍有
物理隔离。对当前本地规模，目录和少量用户级 RAG 缓存的复杂度低于重写所有 Chroma
filter 调用。

代价是用户多时会增加 Chroma 客户端与小目录数量。未来迁移 PostgreSQL 或外部向量库时，
可改为显式 owner 字段和服务端 filter，但 `current_user → owner-scoped repository` 契约应
保持不变。

资料新增、替换或删除后只失效该用户的 RAG、Agent 与 Tutor 服务，不影响其他用户。
因此用户 B 的检索候选中不会出现用户 A 的 Chunk，后续 Evidence/Citation 也没有机会泄露
A 的文件名、excerpt、locator 或 `material_id`。

## 7. API 变化

新增：

- `POST /api/auth/register`；
- `POST /api/auth/login`；
- `POST /api/auth/logout`；
- `GET /api/auth/me`；
- `POST /api/sessions`、`GET /api/sessions`；
- `GET /api/history`，可用属于当前用户的 `session_id` 进一步过滤。

删除旧的匿名身份路径：

- `POST /api/users`；
- `GET /api/users/{user_id}`；
- `/api/users/{user_id}/sessions`；
- `/api/users/{user_id}/history`。

`/api/tutor/chat` 不再接受 `user_id`。资料列表/上传/网页预览/索引/删除、Job 查询、
`/api/ask`、`/api/agent`、Tutor 与 Study Workflow 都需要后端验证身份。资源 ID 只用于定位，
查询仍必须同时带 current_user owner 条件；归属不匹配统一按不存在处理，降低 IDOR 信息
泄露。

## 8. 前端变化

原生 HTML/CSS/JavaScript 页面增加登录、注册、错误提示、当前用户问候和退出按钮。未登录
时工作区隐藏；页面先调用 `/api/auth/me`，验证成功后才加载个人资料、Session 和历史。

前端删除 `zhixing.user_id` 和匿名用户创建流程，不再在 Tutor 请求中发送 `user_id`。
Session ID 与 topic 只作为当前浏览器的续学提示，后端仍会验证 Session owner；401 会清理
个人工作区状态并返回登录入口。资料问答、上传、异步 Job、Tutor、Citation 和三栏布局继续
复用 refinement 后的页面结构。

## 9. 测试与验证

本阶段没有读取 `.env`，没有调用真实 Embedding、ChatModel、Reranker 或付费服务，也没有
重建真实用户 Chroma。

- 全量 Fake/Mock/临时 SQLite 回归：392/392 通过；
- 注册成功、重复邮箱、密码哈希、登录成功、错误密码、退出、失效 Bearer：通过；
- 用户 A/B 的 Session、Conversation、Learning Record 查询隔离：通过；
- Material 目录列表隔离：通过；
- 两个独立 Chroma 中 Vector Retrieval 与来源 metadata 不串库：通过；
- Tutor 在模型调用前拒绝另一用户 Session：通过；
- Document Job 和 Study Workflow 同 ID 越权查询/操作：通过；
- JavaScript `node --check`：通过；
- lxml 解析确认 `workspace-shell` 包含 materials/learning/context 三栏：通过；
- `git diff --check`：通过。

本地 Uvicorn 成功启动并完成应用 startup；应用内浏览器连接因当前 Windows 沙箱辅助程序
错误失败，所以本轮没有把真实视觉或登录后页面声称为已验收。

## 10. Review

Review 覆盖需求边界、完整 diff、认证输入与失败行为、Cookie/token 存储、迁移事务、复合
外键、API 身份依赖、IDOR、per-user Material/Chroma、Citation 来源、Job Worker、Workflow
checkpoint、前端登录态、Stage 1～4 回归和付费调用边界。

Review 中发现并修正：

- 用户 MaterialManager 与服务缓存共用锁导致的潜在死锁；
- 工作区认证区插入后漏掉三栏容器开标签；
- 旧 RAG test 仍断言全局默认路径；
- Tutor 测试 Fake 没有同步直接 `user_id` 消息契约；
- Legacy Job dataclass 没有诚实表达可空 owner。

修正后重新运行相关专项与全量回归，没有剩余自动化阻断项。

## 11. 安全边界与未完成事项

已经建立的是本地产品基础，不是生产身份平台。当前仍未实现：

- 邮箱验证、密码重置、修改密码、多因素认证；
- 登录专用限流、账号锁定、异常登录审计；
- OAuth、SSO、RBAC、组织/公共知识库与资料分享；
- 账户注销、数据导出、细粒度保留期和加密密钥管理；
- SQLite 数据库与 LangGraph checkpoint 静态加密；
- 多实例共享 Session、分布式缓存/队列、PostgreSQL、生产备份恢复；
- Docker、Observability、BYOK；
- 真实模型、并发、负载、公开部署和渗透测试验收。

`SameSite=Strict` 和同源前端降低当前 CSRF 面，但未来若增加跨站前端或第三方登录，应重新
设计 CSRF/CORS。公开部署必须使用 HTTPS，不能把本地 HTTP Cookie 配置直接当作生产安全
结论。数据库仍包含未加密的邮箱、学习历史和资料，不能同步或公开 `data/`。

Stage 5.1 在此停止，不自动进入 Docker、PostgreSQL、Observability 或 BYOK。
