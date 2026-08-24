# 知行（Study Material Assistant）· Development Guide

## 1. 开发环境

当前容器基于 Python 3.12。Windows 本地优先使用仓库 `.venv`，避免全局依赖漂移。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python -m app.server
```

默认页面：`http://127.0.0.1:8000/`；Health：`http://127.0.0.1:8000/health`。
启动和 Health 不调用外部模型。不要在聊天、日志、commit 或截图中暴露 `.env` 与真实 Key。

可选 Cross-Encoder 实验依赖在 `requirements-reranker.txt`，不属于正式运行必需依赖。

## 2. 配置

配置入口为 `app.config.Settings`，示例值见 `.env.example`。

| 类别 | 变量 |
| --- | --- |
| Runtime | `APP_HOST`、`APP_PORT`、`APP_DATA_DIR`、`ZHIXING_PORT` |
| Embedding | `BAILIAN_API_KEY`、`BAILIAN_BASE_URL`、`BAILIAN_EMBEDDING_MODEL`、`BAILIAN_EMBEDDING_DIMENSIONS` |
| Model Gateway | `DEFAULT_PROVIDER`、`DEFAULT_MODEL`、`MODEL_API_KEY`、`MODEL_BASE_URL`、`MODEL_TIMEOUT`、`MODEL_MAX_TOKENS`、`MODEL_TEMPERATURE` |
| BYOK encryption | `MODEL_CREDENTIAL_ENCRYPTION_KEY` |

只有默认 Provider 为 Qwen 时，Model Gateway 才兼容回退旧百炼 Chat 配置。Embedding 配置和
Index Manifest 不因 Chat Provider 切换而改变。

`MODEL_CREDENTIAL_ENCRYPTION_KEY` 必须跨重启稳定；丢失或错误轮换会使已有 BYOK 密文无法解密。
本地开发可以按 `.env.example` 的命令生成 Fernet Key，但不要提交真实值。

## 3. 零费用验证

常规代码改动优先执行：

```powershell
$env:PYTHONIOENCODING = "utf-8"
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p "test_*.py" -v
.\.venv\Scripts\python.exe -m pip check
node --check web\static\app.js
git diff --check
```

Windows 上为避免权限型 `__pycache__` 失败，可用内存编译：

```powershell
.\.venv\Scripts\python.exe -B -c "from pathlib import Path; files=sorted((*Path('app').rglob('*.py'),*Path('tests').rglob('*.py'))); [compile(p.read_text(encoding='utf-8'),str(p),'exec') for p in files]; print(len(files))"
```

专项测试示例：

```powershell
.\.venv\Scripts\python.exe -B -m unittest tests.test_model_gateway tests.test_model_gateway_api -v
.\.venv\Scripts\python.exe -B -m unittest tests.test_rag_service tests.test_langchain_rag -v
.\.venv\Scripts\python.exe -B -m unittest tests.test_tutor_workflow tests.test_tutor_api -v
```

测试存在不等于本次已运行；最终汇报要写明实际命令、数量、失败/错误/跳过和证据边界。

## 4. 本地与 Docker

本地：

```powershell
.\.venv\Scripts\python.exe -B -m app.server
```

Docker（只有环境安装 Docker CLI 时）：

```powershell
Copy-Item .env.example .env
docker compose config
docker compose up --build
```

Compose 是单服务、单应用副本、一个 named volume。不要让多个副本同时写同一 SQLite/Chroma
volume，也不要在未备份时运行删除 volume 的命令。

## 5. 数据目录

`APP_DATA_DIR` 包含用户资料、SQLite、Chroma、Job、Workflow、BYOK 密文和请求日志。
开发或测试必须优先使用临时目录和 Fake/Mock，不读取或修改真实用户工作区。

```text
data/
  ├─ learning/
  ├─ jobs/
  ├─ study_workflows/
  ├─ model_gateway/
  ├─ request_logs/
  ├─ crawl4ai_runtime/
  └─ user_workspaces/<uuid>/
```

涉及 Chroma 的操作先检查 Index Manifest。Legacy index 只能读取；sync、rebuild、delete 和
migration 必须在创建 Embedding Client 或进入 commit path 前被拒绝，除非正在执行明确授权的
隔离迁移流程。

## 6. 按层修改

### RAG

先判断问题属于 Parser、Chunk、Metadata、Embedding、Retrieval、Context、Prompt、Model
还是 Citation。优化必须固定 Dataset 与 Baseline，并用 [EVALUATION](EVALUATION.md) 的门槛验证。

不要把实验 `HybridRRFRetriever`、Cross-Encoder 或 Structure-aware Chunker 直接接入
`RAGService`。

### API / Backend

检查 Pydantic `extra="forbid"`、长度/数量限制、状态码、错误脱敏、认证依赖、所有权、幂等/
冲突行为、服务缓存失效和资源关闭。异步索引继续复用 `MaterialManager`，不要在 Worker
复制第二套索引逻辑。

### Agent / Workflow

优先确定性函数和显式 StateGraph。工具必须有固定权限、输入校验、调用次数、超时、失败回退
和 Trace。新增节点时验证 checkpoint 序列化、`user_id` 所有权与人工审批恢复。

### Model Gateway

业务层依赖 LangChain `BaseChatModel`。Provider 参数转换放在 Adapter，路由和凭据选择放在
Gateway。不要让用户提交任意 Base URL；不要在响应、repr、日志、Metrics 或异常正文暴露 Key。

### Frontend

保留同源 API 契约；运行 JavaScript 语法检查、相关测试，并在可用浏览器中验证交互、错误、
空状态、console 和 390px 移动视口。只改 CSS/源码不能当作视觉验收。

## 7. 费用与外部调用

以下操作可能调用外部服务：

- 索引新增/变化 Chunk：Embedding；
- 问答与 Tutor/Agent 部分路径：Query Embedding + ChatModel；
- Retrieval 实验：Query Embedding；
- Chunking 实验：临时索引 Embedding + Query Embedding；
- Reranker 首次运行：本地模型下载与 CPU 推理；
- 公开网页预览：访问外部 URL，但不产生模型费用。

运行前必须说明调用类型、预计次数/上限、发送的数据和结果落点，并获得明确授权。
评测命令与费用参数见 [EVALUATION](EVALUATION.md)。

## 8. Review 清单

代码修改后至少检查：

- 需求是否完整，是否跨 Stage 或加入无真实需求的技术；
- `git diff` 是否只有预期文件，无调试残留、生成物、密钥和个人数据；
- 正常、边界、失败、恢复和关闭路径；
- API、Evidence、Index、SQLite、Workflow 与 Provider 契约；
- 用户隔离、SSRF、路径、Prompt Injection、秘密和日志；
- 测试是否真的执行，结论是否超过 Fake/Mock/本地证据；
- 文档、README、Status、Architecture 与 Decisions 是否需要同步。

发现阻断项先修复并重新验证；测试通过不能代替 Review。

## 9. Git 与阶段交付

开始前：

```powershell
git status --short --branch
git log --oneline --decorate --max-count=20
git tag --list --sort=version:refname
```

Windows 若出现 dubious ownership，只对单条命令使用：

```powershell
git -c safe.directory=F:/AI-Project-Test/study-material-assistant status
```

不要修改全局 Git 配置。保留用户已有改动，不使用 `git reset --hard` 或回退无关文件。
Commit、Tag、push、deploy、迁移和真实数据操作必须有当前明确授权。

阶段收口顺序：

```text
implementation / docs
  → targeted checks
  → relevant full regression
  → diff and security review
  → Completion Report
  → PROJECT_STATUS / ARCHITECTURE / DECISIONS / README sync
  → authorized commit and tag
```

## 10. 常见故障

- `/health` 正常但问答失败：Health 不初始化 RAG；检查当前用户资料、Manifest、Embedding 和 Chat 配置；
- 端口占用：先检查现有服务，确认不是正确实例后再调整 `APP_PORT` / `ZHIXING_PORT`；
- Python 中文乱码：设置 `PYTHONIOENCODING=utf-8` 或使用 `-X utf8`；
- `compileall` 因 `__pycache__` 权限失败：使用 `-B` + 内存 `compile()`；
- Docker 命令不存在：只能报告 Docker 定义/静态契约，不能声称 build/up 已通过；
- 真实评测未加确认参数：这是费用门槛正常拒绝，不是功能缺陷。
