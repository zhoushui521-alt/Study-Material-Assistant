# 知行 Release 1.0 Deployment Guide

> 本文是部署策略与验收清单，不是部署成功记录。Release 1.0 不执行云部署、不调用真实
> Provider，也不修改 RAG、Agent、Prompt、Model Gateway 或前端 API 契约。

## 1. 当前部署结论

知行当前最合适的部署单元是：

```text
Browser
  -> HTTPS / reverse proxy or platform ingress
  -> one FastAPI application instance
       -> same-origin static frontend
       -> Auth / API / RAG / Tutor / Study Workflow
       -> one in-process Document Worker
       -> Model Gateway
  -> /app/data persistent volume
       -> SQLite
       -> per-user files
       -> per-user Chroma
       -> Job / Workflow checkpoints
       -> request history
  -> Embedding and Chat Providers
```

这是单实例、有状态应用，不是无状态 Web Service，也不是已经拆分的前后端项目。

第一版部署建议：

- 保留单个应用实例；
- 保留 FastAPI 同源托管前端；
- 把整个 `APP_DATA_DIR` 挂载到持久卷；
- 使用 HTTPS，并验证反向代理后的 Secure Cookie；
- 使用平台 Secret 保存 Provider Key 与 BYOK 加密主密钥；
- 先做受访问控制的演示环境，不开放匿名注册与无限公共使用；
- 不启用多副本、不拆独立 Worker、不让多个进程共享写入同一 SQLite / Chroma。

## 2. 已具备与尚未验证

### 已具备

- `app.server` 统一启动入口；
- `APP_HOST`、`APP_PORT`、`APP_DATA_DIR` 等统一 Settings；
- 非 root Dockerfile；
- 单服务 Docker Compose；
- `/health` 容器 Healthcheck；
- `zhixing-data` named volume；
- SQLite、用户资料、Chroma、Job、Workflow 与 BYOK 的统一持久化根目录；
- 请求 ID、结构化日志与单进程运行指标；
- 环境变量示例与 Git / Docker ignore 规则。

### 当前未验证

- Docker image build 与 `docker compose up`；
- 容器内首页、登录、静态资源和业务 API；
- volume 重启恢复与备份恢复；
- 云平台部署、域名、HTTPS 与代理头；
- 反向代理后的 Session Cookie Secure 属性；
- 当前认证版本的真实 Embedding / Chat Provider 全链路；
- 并发、负载、长时间运行与资源上限；
- 多实例一致性、共享限流和跨实例 Session；
- 公开安全、渗透测试、SLO 与告警。

因此，仓库中的 Dockerfile 和 Compose 代表“部署定义已实现”，不代表“生产部署已验收”。

## 3. 为什么当前不直接拆成 Vercel Frontend

当前前端位于 `web/`，由 FastAPI 同源提供，并通过相对路径调用 `/api/*`。登录依赖
HttpOnly、SameSite=Strict 的服务端 Session Cookie。直接把静态文件放到 Vercel、把 API
放到另一个域名，会引入以下新契约：

- 前端 API Base URL；
- CORS allowlist；
- 跨站 Cookie、`credentials` 与 Secure / SameSite 策略；
- CSRF 防护；
- 前后端独立环境配置与错误处理；
- Preview 域名和生产域名的认证边界。

这些都需要代码修改和新的浏览器安全验收，超出 Release 1.0“只做发布准备”的范围。

Vercel 可以托管静态资源和前端 Deployment，但在本项目中应视为未来拆分方案，而不是当前
可直接执行的部署方式。参考 [Vercel Deployments](https://vercel.com/docs/deployments/overview)。

## 4. 部署方式比较

| 方式 | 与当前架构匹配度 | 关键要求 | 当前结论 |
| --- | --- | --- | --- |
| Docker Host / Cloud VM | 高 | 单实例、持久卷、反向代理、TLS、备份、主机运维 | 第一版最可控方案 |
| Render Docker Web Service | 较高 | 持久磁盘挂载到 `/app/data`、单实例、平台 Secret | 可作为托管候选，未验证 |
| Railway Docker Service | 条件匹配 | Volume 挂载到 `/app/data`、非 root 写权限验证、单实例 | 需先解决 volume UID / 权限证据 |
| Vercel Frontend + 独立 Backend | 当前不匹配 | 前后端契约、CORS、Cookie、CSRF 与独立部署改造 | 未来方案，不在 Release 1.0 执行 |
| Serverless Backend | 低 | 无状态存储、外部数据库/对象存储/队列 | 当前 SQLite / Chroma / Worker 架构不适合 |

Render 官方文档说明 Docker 服务可从仓库 Dockerfile 构建，持久磁盘只保留挂载路径下的
数据，且挂载磁盘的服务只能使用单个实例；这与当前单实例边界基本一致，但仍需实际验证。
参考 [Render Docker](https://render.com/docs/docker) 与
[Render Persistent Disks](https://render.com/docs/disks)。

Railway Volume 可以挂载到 `/app/data`，但官方文档也提示 Volume 以 root 用户挂载，
非 root 镜像需要额外处理运行 UID。当前 Dockerfile 使用 UID 10001，因此不能在未验证写权限
前宣称 Railway 可直接运行。参考 [Railway Volumes](https://docs.railway.com/volumes)。

## 5. 推荐方案 A：单机 Docker Compose

适合第一轮受控 Demo、作品展示和单实例验证。

### 拓扑

```text
Internet
  -> DNS
  -> HTTPS reverse proxy
  -> 127.0.0.1:<host-port>
  -> Docker Compose app
  -> zhixing-data:/app/data
  -> external model providers
```

### 必需条件

- 只有一个 `app` 容器；
- 宿主机磁盘容量和 inode 有监控；
- `zhixing-data` 有独立备份；
- 防火墙只开放 HTTPS 与必要管理入口；
- 应用端口不直接暴露给公网；
- Secret 不写入镜像、仓库、Compose 文件或命令历史；
- `STUDY_MATERIAL_ALLOW_PROXY_FAKE_IP` 不得在公开环境启用；
- 网页预览只对受信用户开放，并继续视目标网页为不可信输入。

### 建议环境变量

| 变量 | 部署值原则 |
| --- | --- |
| `APP_HOST` | 容器内使用 `0.0.0.0` |
| `APP_PORT` | 与容器 Healthcheck 和平台端口一致 |
| `APP_DATA_DIR` | `/app/data` |
| `BAILIAN_API_KEY` | Secret；仅供 Embedding / Qwen fallback |
| `BAILIAN_BASE_URL` | 服务端固定 Provider URL |
| `MODEL_API_KEY` | Secret；系统 ChatModel Key |
| `MODEL_BASE_URL` | 服务端固定，不允许用户覆盖 |
| `MODEL_CREDENTIAL_ENCRYPTION_KEY` | 稳定 Fernet Key；跨重启、备份和恢复保持一致 |

`MODEL_CREDENTIAL_ENCRYPTION_KEY` 丢失后，已有 BYOK 密文不能解密。它不能与
`/app/data` 放在同一个无保护备份中，也不能在每次部署时重新生成。

## 6. 推荐方案 B：托管 Docker 平台

### Render 候选

部署前至少确认：

1. Runtime 选择 Docker，并使用仓库根目录 Dockerfile；
2. Persistent Disk 挂载到 `/app/data`；
3. 只运行一个实例；
4. 环境变量通过平台 Secret 注入；
5. 平台入口端口与 `APP_PORT` 一致；
6. Healthcheck 使用 `GET /health`；
7. 实际请求的 scheme / proxy headers 能让登录 Cookie 带 Secure；
8. 记录磁盘快照、恢复和停机窗口的真实行为。

Render Persistent Disk 只对挂载路径持久化，并且挂盘服务不支持多实例。这不是缺点包装，
而是与当前 SQLite / Chroma 单写者边界一致。

### Railway 候选

部署前至少确认：

1. Volume 挂载到 `/app/data`；
2. UID 10001 对 Volume 有真实读写权限；
3. 不为了“先跑起来”静默改为 root，而不记录安全取舍；
4. 只运行一个 Service 实例；
5. 备份、恢复和 Volume 容量策略可用；
6. Health、登录、资料、Job 和重启恢复完成实际验收。

若必须设置 `RAILWAY_RUN_UID=0` 才能写 Volume，应把它视为新的安全决策，先评估替代方案，
而不是把非 root Dockerfile 宣称为仍然生效。

## 7. 上线前验证顺序

### Gate 1：本地镜像

```powershell
docker compose build
docker compose up
```

验收：

- 镜像构建成功；
- 容器以预期用户运行；
- `/health`、首页和静态资源返回 200；
- 注册、登录、退出和 Session 恢复正常；
- 日志不出现 Key、Cookie、问题、回答或资料正文。

### Gate 2：持久化与恢复

1. 使用非敏感演示账号和资料；
2. 创建 Session、Job 与必要测试记录；
3. 停止并重启容器；
4. 确认用户、Session、资料、Chroma、Job 与 Workflow 状态仍存在；
5. 在停写窗口备份整个 `APP_DATA_DIR`；
6. 恢复到空环境并重复检查。

SQLite、Chroma、文件和多个 checkpoint 共同组成一致性边界。只复制单个 SQLite 文件不能
代表完整备份；优先在停止应用写入后备份整个持久卷。

### Gate 3：HTTPS 与认证

- HTTP 强制跳转 HTTPS；
- 登录响应 Cookie 含 HttpOnly、SameSite 和 Secure；
- 反向代理不信任任意来源的 Forwarded Headers；
- 未认证业务请求被拒绝；
- 跨用户资料、Job、Session、Workflow 和 BYOK 仍隔离；
- 公开环境关闭调试页面和不必要的管理入口。

当前 `/docs` 默认可访问。是否对公网保留 Swagger UI 是部署策略，不应未经评估直接开放。

### Gate 4：受控真实 Provider

在明确 Key、模型、发送数据、预计调用次数和费用后，只执行最小矩阵：

- 一次 Query Embedding；
- 一次资料内 RAG；
- 一次证据不足问题；
- 一次 BYOK metadata / route 验证；
- 如需展示 Tutor，再执行一次已彩排路径。

实际结果应分别记录 Provider、模型、时间、调用数、错误和 Citation 人工检查。不能用
Fake/Mock 自动化替代。

### Gate 5：有限并发与资源

至少验证：

- 同时 Ask 时的 429 / 503 行为；
- 上传解析上限；
- Document Job 排队与失败状态；
- Provider timeout；
- 进程内配额重启后的边界；
- 磁盘增长、内存峰值和日志轮转。

在这些数据出现前，不设多副本，也不声称并发容量或 SLA。

## 8. 公开 Demo 的安全边界

当前虽然有用户认证和数据隔离，但不建议直接开放为无限注册的公共服务：

- 注册接口可被公开访问；
- 系统 Key 路径可能承担未知用户费用；
- OperationGuard 和 Metrics 是单进程全局状态，不是持久化用户配额；
- 没有邮箱验证、密码重置、管理员 RBAC、WAF、公开渗透和滥用处置证据；
- 网页预览会访问外部内容，仍存在资源消耗与 Prompt Injection 残余风险。

Release 1.1 第一版建议使用平台访问控制、VPN、IP allowlist 或受控演示账号限制访问。
是否开放注册应在费用账本、用户级配额、滥用防护和公开安全验收之后决定。

## 9. Health、Observability 与运维

`GET /health` 当前是零模型调用的进程级 liveness，不检查：

- SQLite / Chroma 是否可写；
- 持久卷剩余空间；
- Provider Key 和网络；
- Worker 是否能继续消费；
- 备份是否可恢复。

部署平台可以先用它判断进程存活，但不能把它写成完整 readiness。

运维至少需要：

- 平台容器日志；
- HTTP 5xx、429、503 与延迟；
- Job pending / failed；
- LLM failure 与可用 token usage；
- 持久盘容量；
- Provider 账单与配额告警；
- 备份成功和定期恢复演练。

当前 Runtime Metrics 重启清零，没有外部告警；这是下一阶段运维验收项。

## 10. 回滚策略

代码回滚与数据回滚必须分开：

- 代码：回到已验证镜像或 Git Tag；
- 数据：从部署前一致性备份恢复整个持久卷；
- Secret：恢复与 BYOK 密文匹配的加密主密钥；
- Schema：高版本数据库可能拒绝旧程序打开，不能假设代码回滚自动完成数据库降级；
- Provider：路由失败时不要静默切换系统 Key，避免跨用户费用与数据边界变化。

每次部署前记录镜像、Git SHA、数据备份时间、Schema 版本和 Secret 版本。没有恢复演练时，
“有备份”不能称为“可恢复”。

## 11. Release 1.1 建议验收清单

- [ ] Docker image build 成功；
- [ ] Compose 单实例启动成功；
- [ ] 容器内 `/health`、首页、静态资源正常；
- [ ] 登录 Cookie 在 HTTPS 后符合预期；
- [ ] `/app/data` 持久化重启恢复通过；
- [ ] 完整卷备份与恢复演练通过；
- [ ] Secret 未进入 Git、镜像、日志和响应；
- [ ] 受控真实 Provider 最小矩阵通过；
- [ ] 单实例并发和资源基线已记录；
- [ ] 公网入口具备访问控制、TLS 和最小暴露面；
- [ ] 当前限制、失败案例与回滚方式写入部署报告。

只有以上实际执行并形成证据后，才可以把“云端 Demo 已部署”或“Docker production build
已验证”写入项目状态。
