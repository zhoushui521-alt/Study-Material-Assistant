# 知行（Study Material Assistant）· Interview Guide

> Stage 6.5 面试材料。以下表达只使用当前代码、自动化测试、历史本地运行和受控实验能够支持的事实。
> 第一人称回答是可复述模板，不代表新增的真实用户、业务效果或生产验证。

## 1. 使用方法

不要逐字背诵整份文档。面试时按三层回答：

1. 先用一句话给结论；
2. 再用真实实现或实验解释为什么；
3. 最后主动说清边界和下一步。

遇到追问时，从 `Document → Retrieval → Context → LLM → Citation` 的链路定位问题，
不要把所有效果问题都归因于 Prompt 或模型。

## 2. 30 秒项目介绍

> 我做的是知行，一个基于个人学习资料的本地 AI 学习系统。它不是简单的 PDF Chatbot，
> 核心是让回答能回到可定位的 Evidence 和 Citation。系统用 FastAPI 承载上传、异步索引、
> RAG 问答和 Tutor；用 LangChain/LCEL 组织正式 RAG，用 LangGraph 管理 Tutor 和学习工作流状态；
> 用 Chroma 保存向量索引，用 SQLite 保存用户、会话、Job 和 BYOK 元数据。我还建立了固定
> Dataset 的 Retrieval Evaluation，对 BM25+Dense、Reranker、Chunking 和 Context
> Construction 做单变量实验，最终只接入了通过验收门槛的 Context Selector。当前是本地
> 单实例工程原型，不把自动测试或历史实验包装成生产验证。

## 3. 两分钟项目介绍

> 我做这个项目是因为普通资料问答经常有两个问题：第一，答案看起来合理，但用户无法确认它
> 到底来自哪段资料；第二，团队很容易为了“效果”不断加 BM25、Reranker 或 Agent，却没有
> 可复现的评测证明这些复杂度值得。
>
> 所以我先把系统拆成清晰的 RAG Pipeline。资料上传后先暂存、解析和估算费用，用户确认后
> 才创建持久化 Job，由单进程 Worker 串行完成 Embedding 和 Chroma 写入。问答时，在当前
> 用户空间做 Dense Top 10、阈值过滤和关键词覆盖率重排，选 Top 3 Seed、扩展相邻块，再由
> EvidenceScoreContextSelector 压缩上下文。最终传给模型的是带 ID 的 Evidence Map，服务端
> 会解析并校验模型声明的 Citation ID，再补齐文件名、页码、摘录和 locator。
>
> 我没有把所有能力都做成 Agent。普通问答走确定性的 LCEL；Tutor 用显式 LangGraph 路由
> QA、解释、总结和测验，但资料问答仍复用同一 `RAGService.ask()`；开放式 Agent 只有三个
> 受限工具，没有索引、删除、任意文件和任意网络权限。这样既能展示工具编排，也不会牺牲
> 核心证据链的可测试性。
>
> 优化上，我固定 10-case Dataset 和 Baseline，分别测试 Hybrid RRF、Cross-Encoder、
> Structure-aware Chunking 和 Context Selector。Reranker 的 MRR 明显提升，但 CPU 延迟增加约
> 2.86 秒，Context Precision 还下降，所以没有接入；新 Chunker 也因为 Recall@3 和 Final
> Context Recall 回退被拒绝。只有 Context Selector 在历史 Trace 回放中把 Context Precision
> 从 0.15 提到 0.20、保持 Final Context Recall 1.0，且零外部调用，才进入正式链路。
>
> 工程上我补了用户隔离、Session/历史持久化、异步 Job、Request ID、结构化日志、匿名聚合
> 指标、Model Gateway 和加密 BYOK。当前限制也很明确：Dataset 小、Context 实验是历史
> Trace、没有公开部署和并发压测，Citation 也还缺 claim-level 准确率评测。

## 4. 架构总览怎么讲

```text
Browser
  → FastAPI / Auth / Validation / OperationGuard
  → Material / RAG / Agent / Tutor / Study Workflow
  → ModelGateway → Provider Adapter
  → SQLite + per-user files + Chroma + JSONL logs
```

我的解释顺序：

- Browser 负责交互，不保存服务端信任边界；
- FastAPI 从 Session 解析当前用户，做输入校验、费用确认、错误映射和 Request ID；
- 业务层按任务选择确定性 RAG、Tutor Graph 或受限 Agent；
- Model Gateway 统一 Provider/Model/凭证选择；
- 本地存储按职责拆分，但仍属于单实例边界。

## 5. RAG 高频问题

### Q1：为什么需要 RAG，而不是把整份资料直接发给模型？

> 整份资料可能超过上下文窗口，也会把大量无关内容和潜在注入一起送给模型。RAG 先把问题
> 定位到少量候选证据，能控制 token、延迟和费用，还能给 Citation 提供定位基础。它并不会
> 自动保证正确，检索、上下文和引用仍要分别评测。

### Q2：你的正式 RAG Pipeline 是什么？

> 当前正式链路是：Query Embedding → 当前用户 Chroma Dense Top 10 → 0.25 阈值 →
> 0.8 向量分 + 0.2 关键词覆盖率 → Top 3 Seed → 同资料相邻扩展 →
> EvidenceScoreContextSelector → Evidence Map → LCEL ChatModel → 服务端 Citation ID 校验。

### Q3：你说 Hybrid Retrieval，真的是 BM25 + Dense 吗？

> 正式链路不是完整 BM25 Hybrid。它先做 Dense Top 10，只在这些候选内结合关键词覆盖率重排。
> BM25 独立召回加 RRF 是 Stage 3.1 的隔离实验，Recall@5 提升，但 MRR 回退且 Top 3 没提升，
> 所以没有接入。我会明确区分正式实现和实验实现。

### Q4：为什么先做 Retrieval Evaluation？

> 如果没有固定 Dataset、Gold 和 Baseline，我无法判断一个“更智能”的回答来自检索变化、
> Context 变化还是模型随机性。先做 Retrieval Evaluation，可以把候选召回、排序和上下文问题
> 分开，之后每次只改变一层，并记录回归、延迟、调用数和索引指纹。

### Q5：Recall、MRR、nDCG 分别看什么？

> Recall@K 看前 K 个候选有没有找到 Gold；MRR 更关注第一个 Gold 出现得多早；nDCG 能处理
> 多个相关项及不同位置折损。只看 Recall 可能掩盖 Gold 排名过低，只看 MRR 又可能忽略多个
> 证据是否都被召回，所以要结合 case-level failure。

### Q6：为什么 Reranker 指标好却不接入？

> Cross-Encoder 相对 Hybrid 把 MRR 从 0.5778 提到 0.9259、Recall@1 从 0.3333 提到
> 0.8333，但 CPU 平均延迟从约 211.5 ms 增到 3073.9 ms，Context Precision 还从 0.1577
> 降到 0.1458。它证明了排序假设，却没有通过交互延迟和下游 Context 门槛，所以保留实验。

### Q7：为什么新 Chunker 也没上线？

> Structure-aware 600/0 的 Recall@1、MRR 和 nDCG@5 上升，但 Recall@3 从 0.8333 降到
> 0.6667，Final Context Recall 从 1.0 降到 0.7778，Context Precision 也下降。对当前正式
> Top 3 Seed 机制来说，这是关键回归，因此没有为了“结构感知”这个标签迁移正式索引。

### Q8：Context Selector 解决什么问题？

> Baseline 的 Final Context Recall 已是 1.0，但平均约 7.3 个 Chunk、Precision 只有 0.15，
> 问题在 Context Construction，不是召回。Selector 强制保留所有 Seed，每个 Seed 最多选择
> 一个已有相邻块。历史 Trace 回放中平均 Chunk 降到 5.2、Precision 到 0.20、Gold 删除为 0，
> 增量约 0.1042 ms，所以它通过了这一层的验收门槛。

### Q9：Stage 3.4 的证据有什么限制？

> 它复用了 Stage 2 的历史 Retrieval Trace，没有重新跑当前索引 Retrieval，也没有调用真实
> ChatModel。它能证明 Selector 对同一组历史候选的行为，不能证明当前 Retrieval、Answer、
> Faithfulness 或真实 token cost。我会把它叫可信 replay，不叫端到端验收。

### Q10：Evidence 和 Citation 有什么区别？

> Retrieval Candidate 只是候选 Chunk。Evidence 是经过结构化、带资料和片段定位的数据；
> Citation 是回答中的具体声明对某个 Evidence ID 的引用。当前实现能保证模型不能引用本次
> Evidence Map 之外的 ID，并由服务端补全元数据，但这还不等于每个自然语言 Claim 都被引用
> 内容充分支持，后者需要 claim-level Citation Accuracy/Support 评测。

### Q11：如何处理资料外问题？

> 目前有阈值和基础拒答，但 `mysql_out_of_scope` 历史案例仍返回低相关 Context，说明
> Unanswerable Handling 还没有解决。排序或 Selector 不负责这个问题。下一步应增加近似相关
> 负例，单独校准可回答判断和空上下文行为，而不是继续堆 Reranker。

### Q12：Prompt 在你的系统里重要吗？

> 重要，但不是默认根因。召回不到 Gold 时改 Prompt 没用；候选有 Gold 但排序靠后，应看
> Ranking；上下文噪声多，应看 Context Construction；Citation 不支持 Claim，才进一步看
> Evidence 契约和生成约束。我会先定位层级，再做最小实验。

## 6. Agent 与 LangGraph 高频问题

### Q1：为什么不让所有问答都走 Agent？

> 普通资料问答路径已知，用确定性的 LCEL 更容易测试、限费和保留证据。Agent 只有在步骤
> 无法提前确定、需要动态选工具时才有价值。把所有请求交给 Agent 会增加模型决策、延迟、
> 费用和不可预测路径，却不必然改善学习效果。

### Q2：项目里有哪些“Agent/Workflow”？

> 有三类：普通 RAG 是确定性 LCEL；开放式 Agent 使用 LangChain `create_agent`，只有回答
> 资料、列资料和安全预览网页三个工具；Tutor 和 Study Workflow 使用显式 LangGraph
> `StateGraph` 管理路由、状态、确认和恢复。它们职责不同，不能统称为一个 Multi-Agent 系统。

### Q3：Tutor 为什么用 LangGraph？

> Tutor 需要在 QA、解释、总结、测验等学习动作间路由，并保存 topic、消息、Citation 和下一步
> 行动。Graph 让节点、状态和失败路径显式可测。它仍然是单 Tutor，不是多个角色互相对话。

### Q4：Study Workflow 为什么不是普通函数？

> 它有生成计划、等待用户确认、推进任务、失败后再次确认重试等跨请求状态。普通函数可以实现
> 单次调用，但 Graph 配合 SQLite checkpointer 更适合表达暂停、恢复和状态约束。业务表仍由
> SQLite 管理，LangGraph 不替代数据库。

### Q5：Agent 工具怎么做安全边界？

> 工具集是白名单：没有任意文件读取、索引、删除和任意网络访问。网页预览还要单独授权，
> 并经过 URL 安全校验；工具和总流程有超时、调用预算和费用确认；资料回答复用服务端原始
> RAG 答案与 Citation，避免 Agent 二次改写证据。网页内容仍按不可信输入处理。

### Q6：为什么不用 Multi-Agent？

> 当前没有多个自治角色必须并行协作的真实需求。多 Agent 会复制上下文、增加通信、费用和
> 失败面，还让证据归属更难追踪。单 Tutor 加显式 Workflow 已能覆盖当前学习场景，所以再拆
> 角色属于过度设计。

### Q7：你做了 Human-in-the-loop 吗？

> 做的是明确的人类确认点，而不是泛泛的“人在回路中”。索引前确认费用，Agent/Tutor 调用
> 模型前确认费用，学习计划创建后等待用户确认，失败重试也要求再次确认。当前没有复杂人工
> 审批平台。

### Q8：Agent 的真实效果验证到什么程度？

> 工具选择、权限、错误、超时和契约主要由 Fake/Mock 自动化覆盖。项目没有把这些测试说成
> 真实工具调用模型的稳定性验收，也没有生产级 Agent Benchmark。面试时我会把实现事实、
> 自动化证据和真实模型证据分开。

## 7. 工程化高频问题

### Q1：为什么选 FastAPI？

> 这个项目需要明确的请求模型、状态码、依赖注入、Session 鉴权和异步生命周期。FastAPI 能
> 让 Pydantic 校验、OpenAPI 和服务编排保持清晰，同时不强迫我引入更重的分布式框架。

### Q2：为什么文档处理做异步 Job？

> 解析、Embedding 和索引提交耗时不稳定，不应让 HTTP 请求一直等待。接口先返回
> `202 + job_id`，Worker 串行消费；Job 状态和结果落 SQLite。启动时，旧 processing 被标记
> failed，pending 可继续。这解决了本地单实例的中断恢复，不等于分布式任务队列。

### Q3：为什么不用 Celery、Redis 或 Kafka？

> 当前目标是本地单用户/小规模使用，只有一个应用进程和一个 Worker。SQLite Job 加串行
> Worker 已能满足持久化、状态查询和恢复。引入 Redis/Celery 会增加部署、监控和一致性成本，
> 但当前没有吞吐或多实例需求证明它值得。

### Q4：SQLite 的边界是什么？

> SQLite 适合当前单实例、小规模、事务清晰的本地系统。项目用它保存用户、Session、历史、
> 学习记录、Job 和 BYOK 元数据，并用外键和 user_id 做约束。它没有经过高并发写、多实例
> 共享存储和线上备份恢复验收，部署扩大前需要重新选型或验证。

### Q5：用户隔离怎么实现？

> Session Token 在服务端解析为当前用户；文件和 Chroma 位于
> `data/user_workspaces/<user_uuid>/...`；业务查询、Job 和 BYOK 都按 user_id 过滤或约束。
> 客户端不能通过提交另一个 user_id 越权。自动化覆盖了跨用户拒绝，但还不是公开多租户渗透
> 测试。

### Q6：密码和 BYOK 怎么保护？

> 密码使用带随机盐的 scrypt hash，不保存明文。BYOK 用 Fernet 提供密文和完整性校验，
> 主密钥来自运行环境，数据库只保存密文；API 响应不返回 Key。Provider 是白名单，用户不能
> 提交任意 base_url，避免把 Gateway 变成 SSRF 出口。主密钥轮换、备份和托管仍是部署责任。

### Q7：为什么做 Model Gateway？

> RAG、Agent 和 Tutor 原来如果分别直接构造模型，会形成多套 Provider 逻辑和密钥边界。
> Gateway 统一按 user_id 选择系统配置或 BYOK，再通过 Provider Adapter 构造模型。凭据更新
> 后会失效该用户缓存的 RAG Service。它不是负载均衡、自动 Failover 或成本路由平台。

### Q8：Observability 做了什么？

> 每个 HTTP 响应有 Request ID，关键事件写结构化 JSONL，错误做脱敏分类；进程内指标聚合
> HTTP、LLM、Retrieval、Job、Agent/Tutor/Workflow 等计数和延迟。异步 Job enqueue 时保留
> Request ID，后台生命周期通过 job_id 关联。指标重启清零，没有外部 Trace Collector。

### Q9：OperationGuard 是什么？

> 它是单进程资源和费用保护层，对 Ask、Agent、Tutor、索引等操作限制并发、队列和实际单位。
> 它能做 429/503 等快速失败和预算保护，但不是跨实例分布式锁，也不能替代 Provider 侧配额。

### Q10：Docker 做到什么程度？

> 仓库有 Dockerfile、单服务 Compose、非 root 用户、healthcheck 和 named volume。单服务是有意
> 保持 SQLite、Chroma、Session 和进程 Worker 的一致性边界。历史本地 Compose 已验证过，
> 但这次 Stage 6 文档整理没有把它说成云端或生产部署验收。

### Q11：测试通过能证明什么？

> 自动化可以证明在受控输入和测试替身下，API 契约、权限、状态迁移、回退和很多边界行为符合
> 预期。它不能证明真实 Provider 质量、网络波动、并发容量、公开安全或用户学习效果。所以我
> 会分别汇报自动化、离线评测、本地运行、真实模型和生产证据。

### Q12：成本怎么控制？

> 索引先暂存和估算，再显式确认；Ask/Agent/Tutor 的付费路径有请求契约；实验记录 Embedding、
> Chat、Reranker 调用数和索引指纹；OperationGuard 控制单进程预算；Metrics 记录调用与延迟。
> 实际金额仍以 Provider 账单为准，项目没有虚构固定成本。

## 8. 四个最值得讲的技术难点

### 难点一：从“来源列表”升级为可检查证据

**情境**：早期 RAG 可以返回文件名和候选片段，但“显示来源”不等于回答真的被来源支持。

**任务**：建立稳定的 Evidence/Citation 契约，同时避免模型引用不存在的片段。

**行动**：

- 为 Chunk 建立稳定 Evidence ID、资料 ID、页码/Chunk index 和 locator；
- 构造只包含本次 Context 的 Evidence Map；
- 要求模型声明 Citation ID；
- 服务端只解析当前请求允许的 ID，并补全可信元数据；
- 对证据不足返回空 Citation，而不是编造来源。

**结果**：代码和自动化能保证引用 ID 属于本次 Evidence Map，并能定位资料；但 claim-level
支持度仍待独立评测。这个边界要主动说。

### 难点二：用实验拒绝“看起来更高级”的方案

**情境**：BM25、Reranker 和结构化 Chunking 都可能让个别问题更好，容易产生技术堆叠冲动。

**任务**：判断它们是否值得进入正式 Pipeline。

**行动**：冻结 10-case Dataset、Gold 和 Baseline；按 Candidate、Ranking、Chunk、Context
分层，每次只改变一个变量；同时看聚合指标、case regression、延迟、调用数和索引安全。

**结果**：Hybrid RRF 保留实验，Reranker 因延迟和 Context 回归保留实验，新 Chunker 因
Top 3 与 Final Context Recall 回归被拒绝；只接入满足门槛的 Context Selector。难点不是把
技术跑起来，而是有证据地说“不接”。

### 难点三：让耗时索引从请求变成可恢复任务

**情境**：Embedding 和 Chroma 写入时间不稳定，同步请求超时后用户不知道是否已经写入。

**任务**：在不引入分布式队列的前提下，提供任务状态、恢复和费用边界。

**行动**：先暂存与估算；确认后创建 SQLite Job 并返回 202；单 Worker 串行 claim；复用原
MaterialManager 的提交、Manifest、回滚和 OperationGuard；启动时处理 stale processing 与
pending；API 按用户校验 Job 所有权。

**结果**：获得持久状态、失败落库和本地重启恢复，同时明确只有单实例、无自动重试和多 Worker
抢占。这是“最小完整方案”，不是简化版分布式系统。

### 难点四：在 BYOK 灵活性与安全边界之间取舍

**情境**：不同用户可能需要不同 Provider/Model，但把 Key 直接放进 Ask 请求会泄露并形成多套
模型构造逻辑。

**任务**：统一路由、持久化凭据，并避免跨用户、明文、任意 base_url 和错误泄密。

**行动**：Model Gateway 统一 Route；按 user_id 只保留一条活跃 BYOK；Fernet 加密，主密钥
只来自环境；Provider 白名单；错误映射和日志脱敏；凭据变更后失效用户级服务缓存。

**结果**：代码与自动化覆盖用户隔离、密文、Key 不出响应和异常路径；真实 Provider 兼容范围、
密钥轮换和托管仍是部署边界。

## 9. 压力追问与诚实回答

### “你这个能叫生产级吗？”

> 不能。它是工程能力较完整的本地单实例原型。生产还缺公开部署、并发和容量测试、备份恢复、
> 外部监控、密钥托管/轮换、更多安全测试和明确 SLO。Stage 6 做的是 readiness 与作品收口，
> 不是生产认证。

### “10 个评测案例太少了吧？”

> 是，所以我只用它做受控的方向判断和逐案例分析，不做普遍化结论。下一步应扩大独立 Dataset，
> 加入更多术语、语义、Multi-Gold、长短 Query、跨段和不可回答负例，再重跑当前正式链路。

### “Citation 有了就不会幻觉吗？”

> 不会。当前服务端能阻止引用不存在的 Evidence ID，但模型仍可能写出没有被摘录充分支持的
> Claim。还需要 Answer Correctness、Faithfulness 和 claim-level Citation Support/Accuracy。

### “为什么不用云向量库？”

> 当前本地用户空间和 Chroma 已满足规模，没有容量、跨区域或多实例需求。云向量库会增加费用、
> 数据外发、鉴权和运维边界。只有真实规模和部署目标出现后才值得评估。

### “为什么不做 GraphRAG？”

> 当前问题主要是候选排序、上下文噪声和不可回答，不是需要实体关系多跳推理。没有数据和指标
> 证明图构建成本值得，所以现在加入会偏离问题。

### “你的 Agent 有多少自主性？”

> 自主性刻意受限。它能在三个白名单工具间选择，不能任意访问文件、索引或网络；Tutor 和学习
> Workflow 更多是显式状态机。我的目标是可控地解决学习任务，不是追求最大自主性。

### “如果让你下一步继续做，你先做什么？”

> 先扩展并冻结 Retrieval/Answer/Citation 基准集，特别补不可回答和 claim-level Citation；再在
> 当前正式索引上重跑 Context Selector 和真实 Answer 验收。只有新证据显示吞吐、部署或模型
> 路由成为瓶颈，才考虑多实例、外部队列或更复杂 Provider 策略。

## 10. Claim 与证据边界速查

| 可以说 | 不应说 |
| --- | --- |
| 实现了 FastAPI、本地 Web、RAG、Tutor、受限 Agent、持久化 Job | 已经是生产级 AI 平台 |
| 自动化覆盖权限、契约、状态和异常路径 | 大量真实用户验证稳定 |
| 历史受控实验比较了四类优化 | 当前所有模型和索引都重新 Benchmark 通过 |
| Context Selector 在历史 Trace 中 Precision 提升且 Recall 保持 | 回答准确率提升了固定百分比 |
| Citation ID 受当前 Evidence Map 约束 | 每个回答 Claim 都被准确引用 |
| Docker/Compose 有本地历史验证 | 已完成云部署与高可用 |
| BYOK 密文存储且按用户隔离 | 密钥管理已达到企业合规 |
| 单 Worker Job 可持久化和恢复 | 支持分布式高并发任务 |

## 11. 面试结束前可反问

- 这个岗位当前最需要解决的是模型效果、数据/评测、产品闭环，还是部署与可靠性？
- 团队如何区分离线评测、线上指标和人工验收？
- RAG 的失败案例会怎样进入 Dataset 和回归流程？
- Agent 的工具权限、费用和可观测性目前由哪一层负责？
- 对初级 AI 应用开发者，团队更看重哪类可独立交付的工程闭环？

## 12. 相关证据

- [Final Architecture](FINAL_ARCHITECTURE.md)
- [RAG Evaluation Report](RAG_EVALUATION_REPORT.md)
- [5 分钟 Demo Guide](DEMO_GUIDE.md)
- [Evaluation Guide](EVALUATION.md)
- [Current Project Status](PROJECT_STATUS.md)
