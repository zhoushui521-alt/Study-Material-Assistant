# 知行（Study Material Assistant）· 5 分钟 Demo Guide

> Stage 6.4 演示脚本。目标不是在 5 分钟内展示所有 API，而是让观众看懂：资料如何进入系统、
> 回答如何回到证据、Tutor 如何复用 RAG，以及系统怎样显式管理费用与失败。

## 1. 一句话定位

知行不是通用聊天壳，而是一个从个人资料出发、通过可定位 Evidence 和 Citation 帮助用户
理解与复习内容的本地 AI 学习系统。

演示只证明本次实际观察到的本地行为，不证明公开部署、并发容量或生产可用性。

## 2. 选择演示模式

### A. 安全演示（默认）

- 使用已经索引且不敏感的演示资料；
- 现场上传只执行“本地校验与解析”，展示费用估算后停止；
- 现场只进行一次资料问答；是否调用 Tutor 由演示前的费用授权决定；
- 不新增 BYOK、不展示 `.env`、不在终端粘贴密钥；
- 新增索引调用为 0，问答仍可能产生 Query Embedding 与 ChatModel 费用。

### B. 完整任务演示（明确授权后）

- 使用小而确定的非敏感文件；
- 在费用估算后点击“确认费用并创建任务”；
- 观察 `pending/processing/completed` 或失败状态；
- 再执行 Ask/Tutor；
- 用演示前后聚合指标和日志核对调用与延迟。

不要为了展示 Job 临时索引大 PDF。解析时间、Embedding 批次数、Provider 波动和网络延迟都会
让 5 分钟演示失控。

## 3. 演示前准备

### 3.1 数据

准备一份你有权使用、没有隐私和密钥的资料，并提前写下：

1. 一个资料中明确可回答的问题；
2. 预期命中的章节或页码；
3. 一个资料外问题，用于解释证据不足时为什么应拒答；
4. 一个适合 Tutor 的追问，例如“请换一种方式解释这个概念”。

不要把临时下载的陌生网页或个人简历当演示资料。前者有 Prompt Injection 和内容变动风险，
后者有隐私风险。

### 3.2 账号与系统

- 使用专门的本地演示账号，不使用真实个人邮箱和常用密码；
- 预先完成需要展示的资料索引，并确认它属于该演示账号；
- 确认当前 Provider、Model、余额和额度；
- 关闭包含 `.env`、Token、真实日志内容和私人目录的窗口；
- 保留一个终端显示服务日志，但不要滚动到历史敏感输入；
- 不在演示前更新依赖、重建索引或切换分支。

### 3.3 启动与零费用检查

在项目根目录使用已经安装好的 `.venv`：

```powershell
.\.venv\Scripts\Activate.ps1
python -m app.server
```

另开终端检查：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

再打开 `http://127.0.0.1:8000/`。启动和 `/health` 不调用 Embedding、ChatModel 或
Reranker。页面应显示本地 API 可用。

### 3.4 两分钟彩排门槛

正式演示前至少确认：

- 登录成功，资料库只显示演示账号资料；
- 已索引资料能回答准备好的问题；
- Citation 卡片包含资料名、摘录和定位信息；
- Request ID 可见；
- 如果演示 Tutor，会话创建、费用确认和一次回复已在相同环境验证；
- 如果演示 Job，文件大小和费用估算在可接受范围，且有失败后的退出方案；
- 浏览器缩放、投屏分辨率和中文字体可读。

任何一项失败，都应切换到安全演示，不现场改代码、改 Prompt 或重建索引。

## 4. 五分钟主脚本

### 00:00～00:30 · 问题与定位

操作：展示首页和“可信回答”流程。

讲法：

> 很多资料助手只给出一段看起来合理的回答。知行更关心这段回答能否回到我的资料、具体
> 片段和定位信息，所以核心不是聊天，而是 RAG、Evidence、Citation 与可评测的学习流程。

不要在开场罗列技术栈。先让观众知道系统解决什么问题。

### 00:30～01:10 · 登录与用户边界

操作：登录演示账号，展开“我的知识库”。

讲法：

> 登录不仅是页面门槛。资料、Chroma、Job、Session、历史记录和 BYOK 都按 `user_id`
> 隔离；服务端不接受客户端传来的 `user_id` 来决定数据范围。

现场证据：页面显示当前用户和该用户的资料列表。不要把一个账号看不到另一个账号资料的
自动化测试说成现场多用户压力验证。

### 01:10～01:50 · 上传、估算与任务边界

操作：选择演示文件，点击“本地校验与解析”。展示解析结果、Chunk/费用估算和
“确认费用并创建任务”按钮。

讲法：

> 上传分两步。第一步只暂存、校验、解析和估算；第二步必须显式确认费用，才返回
> `202 + job_id`，由持久化的单 Worker 串行写入索引。这样网络请求不会一直占着，失败也能
> 落到 Job 状态中。

安全演示到这里停止，不点击费用确认。

完整任务演示可以点击确认，并观察页面轮询 `GET /api/jobs/{job_id}` 的进度。若 20～30 秒
仍未完成，不等待、不重复点击，直接进入已预索引资料的问答。

### 01:50～03:05 · Ask、Evidence 与 Citation

操作：输入准备好的资料内问题，点击“向资料提问”。

讲法：

> 查询先进入当前用户的 Dense Top 10，再做阈值和关键词覆盖率重排，保留 Top 3 Seed，
> 扩展同资料相邻块，最后由 EvidenceScoreContextSelector 压缩上下文。模型只看到结构化
> Evidence Map；服务端还会校验回答中声明的 Citation ID，不能凭空引用不存在的证据。

回答出现后，只讲三件事：

1. 回答是否直接解决问题；
2. Citation 的资料名、摘录和 locator 能否回到原内容；
3. Request ID 如何把这次交互与日志关联。

如果返回证据不足拒答，不要立即换一个“更容易成功”的问题。先检查这是否是预期的安全
失败；只有确认演示资料确实包含答案时，才切换到彩排过的备用问题。

### 03:05～04:15 · Tutor 复用同一证据链

操作：切换到“AI Tutor”，创建一个学习主题，输入准备好的追问，勾选费用确认后发送。

讲法：

> Tutor 不是第二套 RAG。LangGraph 负责有限状态和路由，资料问答仍复用
> `RAGService.ask()`，因此 Evidence、Citation、拒答和用户隔离契约保持一致。回复之外，
> 系统还持久化 Session、消息顺序和下一步学习行动，重启后可继续。

现场指出 Tutor 回复里的 Citation 或学习行动。如果本次路由没有 Citation，应按实际路由
解释，不要把所有 Tutor 回复都说成经过资料检索。

### 04:15～05:00 · Gateway、观测与工程边界

操作：可选地打开 `/docs`，展示以下两个受鉴权接口的 Schema：

- `GET /api/model-gateway/credential`
- `GET /api/observability/metrics`

讲法：

> Model Gateway 让 RAG、Agent 和 Tutor 通过同一个入口选择系统配置或用户 BYOK。读取接口
> 只返回 Provider、Model 和凭证来源，不返回 API Key。观测接口只返回当前单实例的匿名聚合
> 指标，Request ID 和 JSONL 日志用于排障。

> 当前仍是本地单实例工程原型：SQLite、进程内指标、单 Worker，没有公开部署、并发压测或
> 多实例一致性证明。Stage 6 的目标是把边界讲清楚，不是给它贴“生产级”标签。

## 5. Model Gateway 演示边界

当前原生前端没有 Model Gateway 配置控件。不要指着页面声称可以切换 Provider。

推荐只展示已配置选择的非敏感元数据：

```text
byok_configured
provider
model_name
credential_source
supported_providers
created_at / updated_at
```

`api_key` 不在响应模型中。`PUT /api/model-gateway/credential` 和
`DELETE /api/model-gateway/credential` 会让该用户已缓存的 RAG Service 失效，下一次业务请求
按新 Route 重建。

只有同时满足以下条件才现场切换：

- 使用可撤销、限额、专门为 Demo 创建的 Key；
- Key 通过隐藏输入提供，不出现在命令历史、Swagger 请求示例、投屏或日志；
- 已获得本次外部调用和费用授权；
- 已验证目标 Provider/Model 兼容；
- 演示后立即撤销 Key，并确认本地密文记录是否需要删除。

五分钟面试 Demo 不满足这些条件时，只展示 GET 元数据和架构图更专业。

## 6. 费用与外部调用台账

| 动作 | 可能的外部调用 | 默认演示决策 |
| --- | --- | --- |
| 启动、`/health`、登录、列资料 | 无模型调用 | 现场执行 |
| 上传暂存、解析、估算 | 无 Embedding/Chat | 现场执行 |
| 确认索引任务 | 按 Chunk/Batch 调用 Embedding | 默认不执行 |
| Ask | Query Embedding + ChatModel | 只执行一次 |
| Tutor | 取决于路由；可能复用 RAG 并调用模型 | 授权后最多一次 |
| Gateway GET、Metrics GET | 无 Provider 模型调用 | 可选执行 |
| Gateway PUT/DELETE | 不验证模型效果；改变后续 Route | 默认不执行 |

实际次数和费用应以暂存估算、运行日志、聚合指标与 Provider 账单为准，不能从这张表推断固定金额。

## 7. 故障恢复卡

| 现象 | 现场处理 | 不要做 |
| --- | --- | --- |
| API 状态不可用 | 检查服务终端和 `/health`；若进程已存在，不重复启动 | 临时改端口或配置 |
| 登录失败 | 使用已彩排账号；说明鉴权失败后停止 | 现场重置真实密码 |
| 资料列表为空 | 确认登录账号；切换到已验证账号 | 临时索引大文件 |
| Job 长时间 processing | 记录 `job_id`，继续预索引问答 | 重复提交同一任务 |
| Job failed | 展示脱敏错误与 Request ID，使用预索引资料 | 删除数据库或索引 |
| Ask 返回 429/503 | 说明单进程预算/保护边界，等待当前请求结束 | 连续重试制造更多请求 |
| Ask 返回 502 | 说明 Provider 失败与安全错误映射，停止付费流程 | 展示密钥或 Provider 原始正文 |
| 没有 Citation | 判断是否为证据不足拒答或对应路由无检索 | 口头编造来源 |
| Tutor 失败 | 保留 Session/Request ID，回到已成功的 Ask 结果 | 临时修改 Prompt |

故障本身不是演示失败。能用状态、Request ID、日志和已知边界解释失败，比隐藏异常更能体现
工程能力。

## 8. 收尾话术

> 这个项目最重要的不是用了多少 AI 技术，而是每一层都有明确契约：资料先经过安全和费用
> 门槛，检索结果先变成 Evidence，回答里的 Citation 再由服务端校验；优化则通过固定
> Dataset 和失败案例决定是否接入。当前边界也很明确，它是本地单实例原型，下一步只有在
> 真实使用目标出现后，才值得补并发、部署和更完整的 Answer/Citation 评测。

## 9. 演示完成清单

- [ ] 定位在 30 秒内说清；
- [ ] 未展示密钥、`.env`、私人资料或敏感日志；
- [ ] 上传暂存与付费提交边界说清；
- [ ] 至少一条回答和 Citation 可现场核对；
- [ ] Tutor 是否走 RAG 按实际结果说明；
- [ ] Job/Request ID/聚合指标至少展示一项；
- [ ] Model Gateway 没有被描述成现有前端功能；
- [ ] 没有把自动测试说成真实模型或生产验收；
- [ ] 5 分钟到点停止，不临场扩展 Agent、Workflow 或实验模块。

## 10. 相关文档

- [Final Architecture](FINAL_ARCHITECTURE.md)
- [RAG Evaluation Report](RAG_EVALUATION_REPORT.md)
- [Development Guide](DEVELOPMENT_GUIDE.md)
- [Current Project Status](PROJECT_STATUS.md)
