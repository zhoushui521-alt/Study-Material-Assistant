# Study Material Assistant V2 · Stage 5.5 BYOK + Model Gateway Completion Report

## 1. 阶段目标

Stage 5.5 把正式 AI 调用从“`RAGService` 直接按固定百炼配置创建 ChatModel”调整为
“业务层依赖统一模型契约，Model Gateway 负责路由、凭据与 Provider Adapter”。

本阶段起点为 `e141e95`（Stage 5.4 收口提交）。代码、配置与测试 checkpoint 为
`d54a90b`；本文随文档收口提交进入 Git。本阶段没有创建 Tag、push、部署或真实付费模型调用。

本阶段没有修改 Retrieval、Chunk、Embedding、Chroma、Index Manifest、Context Selector、
Prompt、Evidence/Citation、Agent 决策、Tutor Workflow 或 Evaluation Dataset/指标。

## 2. 架构变化

正式调用链由：

```text
Application / RAGService
  → 固定 OpenAI-compatible ChatModel
```

调整为：

```text
RAG / Agent / Tutor
  → 用户级 RAGService
  → ModelGateway
       ├─ BYOK route：当前认证用户有凭据
       └─ system route：当前认证用户无凭据
  → ProviderAdapterRegistry
  → OpenAICompatibleProviderAdapter
  → LLM Provider
```

业务层使用 LangChain `BaseChatModel`，不感知具体 SDK。没有另建重复的
`LLMClient.generate()/stream()` 抽象，因为 `BaseChatModel` 已提供当前 LCEL、Agent 和 Tutor
需要的 `invoke` / `stream` 标准契约。当前 FastAPI 产品接口仍为非流式；底层存在 `stream`
契约不等于已经实现或验收流式产品能力。

`RAGService` 仍持有同一个 ChatModel；Agent 与 Tutor 继续复用该模型。因此改动发生在 RAG
Pipeline 的 LLM 调用边界，不改变上游检索和上下文，也不复制业务链路。

## 3. 新增与修改模块

### 新增

- `app/model_gateway/gateway.py`：Provider/模型/Key 输入约束、BYOK-first 确定性 Router、
  系统配置回退和 ChatModel 构造门面。
- `app/model_gateway/providers.py`：`ModelRoute`、Adapter Protocol、Provider Registry 与
  OpenAI-compatible Adapter。
- `app/model_gateway/credentials.py`：Fernet 认证加密、用户级凭据元数据和独立 SQLite
  短连接存储。
- `app/model_gateway/errors.py`：配置、存储、解密、Provider 与认证的稳定错误类型。
- `tests/test_model_gateway.py`、`tests/test_model_gateway_api.py`：网关、BYOK、安全与认证 API
  专项覆盖。

### 正式接入

- `app/rag_service.py`：通过 `ModelGateway.create_chat_model(user_id)` 获取模型。
- `app/api.py`：创建进程内网关；按 `current_user` 注入 RAG；新增凭据管理 API；保存/删除后
  只刷新该用户的 RAG/Agent/Tutor 缓存。
- `app/langchain_rag.py`：正式模型已经由网关注入；历史 Evaluation 构造入口也委托给
  Provider Adapter，不再直接构造 `ChatOpenAI`。
- `app/config.py`、`.env.example`、`docker-compose.yml`：统一 Model Gateway 配置。
- `app/observability.py`、`app/metrics.py`：增加 Provider 与受限凭据来源观测。
- `requirements.txt`：固定 `cryptography==49.0.0`。

## 4. BYOK 实现方式

认证接口：

- `GET /api/model-gateway/credential`：返回当前选择与 BYOK 元数据；
- `PUT /api/model-gateway/credential`：保存或替换当前用户唯一一条活跃凭据；
- `DELETE /api/model-gateway/credential`：删除当前用户凭据并恢复系统默认选择。

请求允许 `provider`、`model_name`、`api_key`，拒绝额外字段。响应只返回是否已配置、
Provider、模型、凭据来源、支持列表和时间戳，不返回 Key。

凭据数据库为：

```text
data/model_gateway/model_credentials.sqlite3
```

表以服务端认证得到的 UUID `user_id` 为主键，保存 `provider`、`model_name`、
`encrypted_api_key`、`created_at` 和 `updated_at`。这表达“每位用户一条活跃凭据”的当前
最小需求，不提前实现多凭据、团队凭据或复杂策略。

配置项：

- `DEFAULT_PROVIDER` / `DEFAULT_MODEL`；
- `MODEL_API_KEY` / `MODEL_BASE_URL`；
- `MODEL_TIMEOUT` / `MODEL_MAX_TOKENS` / `MODEL_TEMPERATURE`；
- `MODEL_CREDENTIAL_ENCRYPTION_KEY`。

只有 `DEFAULT_PROVIDER=qwen` 时，`MODEL_API_KEY`、`DEFAULT_MODEL` 和 `MODEL_BASE_URL` 才兼容回退既有百炼配置；Embedding
配置保持原样，现有 Chroma Index Manifest 不受影响。

## 5. 安全设计

- API Key 使用 Fernet 认证加密后写入 SQLite；自动化直接检查数据库文件中不存在明文 Key。
- 加密主密钥只从 `MODEL_CREDENTIAL_ENCRYPTION_KEY` 读取，`Settings` 的 `repr` 隐藏系统
  Key 与主密钥。
- 主密钥缺失时允许系统默认模型继续工作，但拒绝保存新 BYOK；已有密文无法用当前密钥解密时
  请求失败，不回退系统 Key。
- Provider 名称、模型名、Key 长度与空白都有边界；用户 API 不接受 `base_url`，避免任意
  内网/外网代理和 SSRF。
- 所有凭据读写都使用 `current_user.user_id`；API 用户隔离与 Store 用户隔离均有测试。
- API 响应、结构化日志和 Metrics 不保存 Key、Prompt、回答或 Provider 错误正文。
- 上游 401/403/稳定认证错误被归一化为 `Model authentication failed.`，不向客户端传播
  Provider 内部详情。
- 保存或删除凭据后只失效当前用户模型缓存，避免旧 Key 继续驻留在后续调用模型实例中。

Review 中发现并修复：SQLite 已建立连接但 schema 初始化失败时，旧实现可能遗漏关闭连接；
最终实现会在包装为安全 Store Error 前显式关闭句柄，并有回归测试。

## 6. Provider 支持情况

| Provider | 当前 Adapter | Base URL 来源 | 状态 |
| --- | --- | --- | --- |
| `qwen` | OpenAI-compatible | 服务端覆盖或固定 DashScope compatible endpoint | 代码接入，未真实调用 |
| `deepseek` | OpenAI-compatible | 服务端覆盖或固定 DeepSeek endpoint | 代码接入，未真实调用 |
| `openai` | OpenAI-compatible | `ChatOpenAI` 默认官方地址或服务端覆盖 | 代码接入，未真实调用 |
| `openai_compatible` | OpenAI-compatible | 必须由服务端 `MODEL_BASE_URL` 配置 | 代码接入，未真实调用 |
| Claude / Anthropic | 无 | 无 | 未实现 |

“代码接入”只表示路由和 Adapter 构造可由自动化验证，不证明账号权限、具体模型名、响应
metadata、token usage 或 Provider 兼容性已经通过真实验收。模型名由受限格式的配置/用户输入
决定，没有把某个时效性模型版本写死成产品事实。

## 7. 验证结果

### 自动化

- Stage 5.5 核心、安全、配置与观测专项：`29/29`；
- Agent/Tutor/Workflow 与 Model Gateway 跨链专项：`41/41`；
- 最终 44 个 `test_*.py` 模块全量回归：`434/434`；
- 最终结果：`failures=0`、`errors=0`、`skipped=0`。

覆盖包括：

- 系统配置路由、BYOK 优先与删除后系统回退；
- Provider/模型参数转换和不支持 Provider；
- 密文落库、解密、主密钥缺失/轮换失败；
- 凭据 API 元数据响应、用户隔离、任意 Base URL 拒绝；
- 保存/删除后的用户级模型缓存失效；
- Provider、模型、耗时、token usage 和成功/失败指标；
- API Key、Prompt 与 Provider 错误详情不进入响应或日志；
- 认证失败稳定错误；
- RAG/LCEL/API/Agent/Tutor/Job/认证/部署契约等既有回归。

### 静态与依赖检查

- `app/` 与 `tests/` 共 104 个 Python 文件内存编译通过，`app.api` 导入通过；
- `pip check`：`No broken requirements found.`；
- `node --check web/static/app.js`：通过；
- Compose YAML 解析通过：1 个服务、8 个 Model Gateway 环境变量、named volume 契约正确；
- `git diff --check`：通过；25 个本轮文件的尾随空白、冲突标记和最终换行检查通过；
- 当前环境没有 Docker CLI，因此没有执行镜像 build/up。

### 外部调用

本阶段真实 Query Embedding、ChatModel、Reranker、外部网关和外部观测调用均为 `0`。
没有读取 `.env` 或真实 Key，没有打开/修改正式 Chroma，也没有产生模型费用。

## 8. 当前限制

- 当前是一进程内 Gateway，不是独立 LLM Gateway 服务；这符合单体本地原型边界。
- 每用户只能保存一条活跃凭据；没有团队/组织凭据、作用域、额度分配或凭据共享。
- 主密钥没有 KMS/HSM、版本、在线轮换、托管备份或灾难恢复；主密钥与数据库必须协调备份。
- 没有真实 Provider compatibility matrix；Claude/Anthropic 未实现。
- 没有流式 FastAPI/UI、取消、熔断、健康探测、自动故障切换、负载均衡或复杂任务 Router。
- `MODEL_MAX_TOKENS` 提供单次生成上限，Runtime Metrics 聚合 token usage；但没有价格版本、
  货币成本计算、用户预算、日/月额度或硬阻断，因此不能宣称“成本控制平台已完成”。
- Metrics 仍是单进程累计值，重启清零；没有 Provider 维度持久报表或时间窗口。
- 早期教学 `app/chat_client.py` / `app/ask_documents.py` 仍保留历史直连实现；当前结论只覆盖
  正式 V2 FastAPI → RAG → Agent/Tutor 调用链。
- 没有 Docker build/up、浏览器 UI、并发/负载、公开部署、渗透或生产安全验收。
- 本阶段没有新增 BYOK 前端页面；功能当前通过认证 API 管理。

## 9. 下一阶段建议

Stage 5.5 已完成本地 Git checkpoint。下一阶段应以 `d54a90b` 及其文档收口提交为基线，
继续区分代码、自动化、本地运行、真实 Provider 与生产验证证据。

后续建议按证据推进，而不是一次补齐模型平台清单：

1. 经用户明确批准后，用隔离测试账号和严格调用上限分别验证一个系统 Provider 与一个 BYOK
   Provider，记录请求数、预计费用、模型、token metadata 和失败分类。
2. 若真实需求是“可见成本”，先建立版本化价格表、计费口径与 token usage 缺失策略，再实现
   预算告警/阻断；不要把 token 计数直接等同于货币成本。
3. 若需要 Claude/Anthropic，新增原生 Adapter 和契约测试，不把它硬塞进 OpenAI-compatible。
4. 只有出现多 Provider 容灾、多个应用实例或网关独立扩缩的真实需求，才评估共享凭据存储、
   熔断/故障切换或拆分独立 Gateway。

## 10. Review 与完成状态

Review 已覆盖需求、差异、正式调用链、异常路径、缓存、Provider 配置、用户隔离、密钥泄露、
SSRF、Observability、旧 CLI 边界和既有回归。发现并修复了三项问题：SQLite schema 初始化
失败后的连接关闭；Agent/Tutor/Workflow 包装认证错误的统一响应；旧百炼 Key 仅允许在 Qwen
默认 Provider 下回退，避免跨 Provider 发送。三项均已重测；当前未发现阻断 Stage 5.5
工作区交付的问题。

| 状态 | 内容 |
| --- | --- |
| **Implemented** | Model Gateway、Router、Adapter、统一配置、BYOK 密文存储/API、用户隔离、缓存失效、观测集成。 |
| **Automated Verified** | 专项 `29/29`，全量 `434/434`。 |
| **Not Verified** | 真实模型、流式产品、Docker、浏览器、并发/负载、多实例、货币成本与生产安全。 |
| **No External Cost** | 外部模型/Embedding/Reranker/网关调用 `0`。 |
| **Git Checkpoint** | 代码 checkpoint 为 `d54a90b`；本文随文档收口提交进入 Git。未创建 Tag、未 push。 |
