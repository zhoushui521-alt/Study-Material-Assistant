# 学习资料助手

这是一个面向学习资料的 RAG 项目：读入资料、检索相关文本、基于证据回答问题，
并展示可以核对的来源。

## 项目文档入口

- [PROJECT_STATUS](docs/PROJECT_STATUS.md)：当前项目状态的唯一入口，记录 V2 checkpoint、证据层级、已完成能力与未验证项；
- [ARCHITECTURE](docs/ARCHITECTURE.md)：当前真实前端、后端、RAG、Tutor、存储与异步处理架构；
- [DECISIONS](docs/DECISIONS.md)：由实验数据和 Completion Report 支撑的关键技术取舍。

README 保留项目介绍与运行方式；阶段状态发生变化时，以 `PROJECT_STATUS.md` 及其链接的 Git Tag、Completion Report 和代码证据为准。

## 当前状态

项目已保留手写 RAG 全链路，并完成 LangChain + Chroma 持久化增量索引、
混合检索、完整问答、无证据拒答、固定评测集和结构化评测报告。当前支持
TXT、Markdown、DOCX 和带文字层的 PDF。DOCX 会提取正文段落、1～3 级标题和
表格文字；不会伪造页码。模型输出残留 `\phi`、`\mu` 曾导致一次真实评测
只通过 9/10；补充格式归一化和自动回归测试后，最新一次真实固定评测通过 10/10。
该结果只代表当前版本通过这组案例，不代表所有问题或后续运行都能稳定通过。

当前还提供最小 FastAPI 服务和同源 Web 问答页，包括健康检查、结构化问答接口、
自动接口文档和基础请求日志；有效 `/api/ask` 已在本地 Uvicorn 中验证返回 `200`。
Web 页面只复用现有 API，不复制检索、Prompt 或模型调用逻辑。

阶段 1 到阶段 7 的代码与自动验证现已完成：Web 可以一次选择并安全暂存多个
TXT、Markdown、DOCX 或 PDF，逐文件展示解析成功或失败原因，并在用户二次确认后复用现有
Chroma 增量同步；同名文件默认拒绝，只有显式选择替换才会更新，删除资料时只定向
删除对应来源的索引记录。请求元数据会持久化为有限轮转的 JSONL，已有两份评测
报告可以用零费用 CLI 比较新增失败、恢复案例、案例集合和耗时变化。当前还增加了
单进程并发拒绝、滚动窗口限流、进程级费用预算、上传解析上限、暂存过期清理、统一
错误页，以及受限的公开网页抓取、Crawl4AI Markdown 预览和二次确认入库。固定问答流程现已组合为可复用
的 LCEL Runnable：检索后先判断证据，没有资料时直接拒答并跳过 ChatModel；有资料时
继续执行 Prompt、ChatModel、字符串解析、格式归一化和来源返回。受限 LangChain
Agent 通过三个固定工具动态选择资料问答、资料列表或公开网页预览，不具备索引、删除、
任意文件访问或任意网络访问工具。
阶段 7 进一步用显式 `StateGraph` 管理学习目标、资料证据、三步任务、人工确认、
进度和复盘；阶段 6 Agent 只作为受控的资料证据节点，不拥有整个流程控制权。
V2 Stage 4 在不改动上述旧接口的前提下新增单 Tutor workflow：确定性识别问答、解释、
练习、总结和学习规划，通过三个受控 Tool 复用现有 RAG、生成结构化练习或总结，并只在
当前推理中保留有限短期状态。后续持久化阶段增加 SQLite 用户、学习 Session、完整 Tutor
对话、学习行为与异步文档任务；当前 Stage 5.1 再用邮箱密码、可撤销服务端 Session、
`current_user` 和每用户独立资料/Chroma 工作区建立可信身份与数据隔离。前端不再提交
可伪造的 `user_id`。

上传、网页预览、替换、删除、回滚、日志轮转、报告对比和阶段 3 保护已经通过 Fake/Mock 自动
测试。用户随后在本地页面完成了一次真实 PDF 上传、费用确认、增量索引和问答：索引
结果为新增或更新 6、删除 0、未变化 141，问答返回了 PDF 页码来源。这证明当前样例
闭环可用，但不等同于并发压测、长期稳定性或对所有资料类型的普遍验证。

手写版本继续保留，用于解释 RAG 底层原理和对照框架行为。阶段 5 已完成本地
Crawl4AI 转换、Fake/Mock 安全验证，以及用户在本地页面对
`https://www.qiuzhi2046.com/` 的真实公开网页 Markdown 预览验收；该网页的付费确认
入库和问答闭环仍未验收。阶段 6 Agent 已通过 Fake/Mock 工具选择和失败路径验证，
但尚未调用真实工具调用模型。阶段 7 LangGraph 工作流已通过 Fake/Mock 和本地 SQLite
关闭后重开恢复测试，但同样尚未进行真实模型工作流验收。

## 运行

在项目目录执行：

```powershell
python -m app.chunk_documents
```

程序会读取 `data/documents/` 中的 `.txt`、`.md`、`.docx` 和带文字层的 `.pdf` 文件，
并打印每个文本块的来源、编号和内容预览。PDF 使用 `pdfplumber` 提取文字层，
再按页变成独立资料单元，
因此来源会保留文件名和页码；纯扫描图片或加密 PDF 暂不支持，程序会给出
错误提示，扫描件需要先进行 OCR。复杂数学公式可能出现 `(cid:xx)`、符号丢失
或阅读顺序变化，引用公式时需要回看原始 PDF。

DOCX 使用 `python-docx` 按文档顺序提取普通段落和表格，并把 Heading 1/2/3
保留为可检索的 section。它没有 PDF 那样可靠的固定分页，所以来源只记录标题层级、
段落序号或表格序号，不生成 page。当前不处理旧版 `.doc`、图片 OCR、SmartArt、
批注、修订、页眉页脚、宏、复杂公式或 Word 的视觉渲染；空白、损坏及伪装 DOCX
会在暂存阶段被拒绝。

## 关键词检索

```powershell
python -m app.search_documents
```

按提示输入关键词或一个简短问题，例如 `RAG`、`LangChain` 或 `RAG 为什么要切分文本？`。程序会显示命中的文本块和来源。

## 验证 Embedding API

1. 将 `.env.example` 复制为 `.env`。
2. 在 `.env` 中填写 `BAILIAN_API_KEY`，以及百炼控制台 API 文档提供的 OpenAI 兼容模式 `BAILIAN_BASE_URL`。
3. 打开 `app/embed_text.py`，点击 VS Code 右上角 ▶，输入一句文字。

成功后会打印向量维度和前 8 个数字。请勿提交 `.env`，它已被 `.gitignore` 排除。

## 语义检索与 RAG 问答

打开以下文件并点击 VS Code 右上角 ▶：

- `app/vector_search.py`：输入“资料太长怎么办？”，观察相似度排序；
- `app/ask_documents.py`：输入“RAG 为什么要切分文本？”，观察回答与来源。

`ask_documents.py` 会额外使用 `.env` 中的 `BAILIAN_CHAT_MODEL`（默认 `qwen-plus`）。如果控制台提示该模型无权限，请在百炼控制台选择已开通的 Qwen 对话模型，并将其模型名填入 `.env`。

## LangChain + Chroma

首次导入资料或资料发生变化时，运行 `app/index_langchain.py`。它会比较当前
资料 ID 与 Chroma 中已有 ID，只向量化新增或变化的文本块、删除已经失效的
旧记录，并保留未变化的向量。向量库持久化在 `data/vector_store/`，同目录的
`index_manifest.json` 记录 Embedding、集合、距离度量、Parser/Chunker 和 Metadata
Schema 版本。已有记录但没有 Manifest 的旧索引只允许问答读取；同步和删除会明确
失败，不会自动清空或迁移。
需要显式迁移时，必须按 [Legacy Index Migration Runbook](docs/legacy-index-migration-runbook.md)
先生成只读计划和本地候选索引，再在停服后提升；不要直接修改正式 Chroma 文件。
该命令会调用真实 Embedding API，可能产生费用；运行前应先确认解析结果。

索引成功后，运行 `app/search_langchain.py`。当前正式 Retrieval 仍采用 Stage 2 Baseline：
Chroma Dense Vector 先从完整语料召回 Top 10，再按 80% 向量相关度 + 20% 关键词
覆盖率重排并选择 Top 3 Seed。后续提问只需要向量化问题，不会重新向量化全部资料。
Stage 3.1 已实现 BM25 + Dense + RRF 双路实验能力，但受控 A/B 没有证明它在当前
Top 3 Seed 主链路上稳定优于 Baseline，因此没有替换正式问答 Retriever。
Stage 3.4 在 Retriever 完成同源 ±2 相邻扩展后增加 Context Selector：保留全部 Seed，
并为每个 Seed 只选择一个关键词覆盖率最高的已有相邻 Evidence，再构建请求内 Evidence
ID。该步骤不重新召回，也不修改 Dense、阈值、排序、Top-K 或 Chunking。

完整 LangChain RAG 入口为 `app/ask_langchain.py`：

```powershell
python -m app.ask_langchain "资料太长应该怎么办？"
```

它通过包装 Chroma 的混合 Retriever 检索资料，使用 `ChatPromptTemplate`
组织上下文，为每条上下文证据分配 `S1`、`S2` 等请求内 ID，再调用百炼 OpenAI
兼容 ChatModel 生成回答。服务端只接受当前上下文真实存在的 ID，并回填文件名、页码、
摘录和定位信息；`sources` 继续表示兼容用检索候选，`citations` 才表示验证后的实际引用。
资料没有直接提供答案时，程序会返回统一的证据不足提示。
该命令会调用真实 Embedding 和 Chat API，可能产生费用。

### LCEL 固定问答管道

`app/langchain_rag.py` 中的 `create_rag_chain()` 使用 `RunnableLambda`、
`RunnableParallel`、`RunnableBranch` 和 `StrOutputParser` 组合以下固定流程：

```text
问题 → 当前检索基线 → Context Selector → Evidence Map → 有无证据分支
     → Prompt → ChatModel → 文本解析 → 归一化与来源
```

CLI 与 FastAPI 问答都通过 `RAGService` 执行；每个服务实例在初始化时只构造一次
Runnable 并复用。兼容入口 `answer_with_retriever()` 也调用同一个管道工厂。无检索结果时走拒答
分支，不调用 ChatModel；模型返回证据不足标记时仍会清空来源并使用统一拒答文本。
这仍是可预测、可测试的两步 RAG，不是由模型动态选择工具的 Agent，也不负责
LangGraph 的状态、路由或恢复。LCEL 层没有叠加额外重试，外部客户端继续使用现有
的有限超时与重试配置。

## 固定 RAG 评测集

`evaluation/rag_cases.json` 保存 10 个固定案例，覆盖 Markdown 回归、PDF 概念、
跨段检索、相近概念和无证据拒答。每个非拒答案例都声明预期来源和答案必含要点，
评测程序还会检查引用标签及终端公式格式。

直接运行不会产生 API 请求，只会显示费用提示：

```powershell
python -m app.evaluate_rag
```

确认批量评测费用后才运行：

```powershell
python -m app.evaluate_rag --confirm-api-cost
```

该命令会对每个案例调用一次真实 Embedding，并在检索到资料时调用 Chat API；
同时复用正式问答入口的检索参数和 RAG 链路。评测完成后，终端输出仍会保留，
并在 `evaluation/results/` 生成一份唯一命名的 UTF-8 JSON 报告。报告记录运行时间、
评测集版本和 SHA-256、检索参数、模型名称、汇总及全部案例明细，但不记录 API Key
或 Base URL。失败案例也会写入报告；报告无法写入时程序会返回清晰错误。

如需把报告写到其他目录，可增加：

```powershell
python -m app.evaluate_rag --confirm-api-cost --results-dir <目录>
```

`evaluation/results/` 是本地运行历史，默认被 `.gitignore` 忽略，避免持续增长的生成
文件和完整回答被意外提交。最近一次真实固定评测基线为 10/10；此前柯西分布回答
残留 LaTeX 命令的问题已在针对性格式修复后的真实复测中通过。固定案例只能证明
当前版本通过这 10 个问题，不能替代更大评测集、人工检查或未来版本复测。

## Stage 2：独立 Retrieval Evaluation 与 Trace

`evaluation/retrieval_cases.json` 是与上述端到端问答 heuristic 分离的 Retrieval
Dataset。每个可回答案例分别保存不依赖当前切块编号的 `Stable Gold Meaning`，以及
可随 Chunker 更新的 `Current Chunk Mapping`（`chunk_id`、`material_id` 和
`content_hash`）；`legacy_chunk_index` 只用于人工定位，不参与 Gold 身份判定。

`app/retrieval_evaluation.py` 复用当前生产检索公式，并以 Evaluation 专用结构化 Trace
记录 Raw Vector Candidates、0.25 阈值过滤、80/20 Hybrid Ranking、Top 3 Seed、同源
±2 Adjacent Expansion 和最多 8 块 Final Context。Trace 只记录身份、分数、排名、原因、
大小和本地耗时，不写入候选正文；没有可靠 tokenizer 时 `token_size` 明确为 `null`。
报告提供 Raw/Ranked Recall@1/3/5/10、MRR、nDCG@5、Final Context Recall 和 Context
Precision，并把 Recall、Filtering、Ranking、Context Construction 与 Unanswerable
Handling Failure 分开。单 Gold 案例上的 nDCG 与 MRR 信息高度重合，主要在 Multi-Gold
案例中提供额外排序信息。

以下命令默认只显示费用边界，不打开索引、不调用 API：

```powershell
python -m app.evaluate_retrieval
```

显式确认后，Runner 先完成 Git commit、Dataset、Gold Mapping、Retrieval Config 和报告
输出位置 preflight，再把当前 Chroma 复制为 disposable Evaluation Snapshot。每个案例调用
一次真实 Query Embedding，并在案例完成后把结果原子持久化到唯一 JSON run state；最终聚合
可从已持久化的 per-case results 重建。Evaluation 只查询 snapshot，退出后以目录和文件内容
SHA-256 指纹验证原始 Index 未变化；不调用 ChatModel，不迁移 legacy index，也不自动补
Manifest：

```powershell
python -m app.evaluate_retrieval --confirm-query-embedding-cost
```

Stage 2 可以确定性验证 Citation ID 是否存在于本次 Evidence Map；Citation Coverage
仍需要 Claim 标注，Citation Support 仍需要人工或经授权的 Judge。有效 Citation ID
不等于 Evidence 真正支持对应 Claim。

修复 crash-safe Runner 后，已在 commit `0d9f2fe` 上完成一次 10 案例、10 次真实
Query Embedding、0 次 ChatModel 的 `local_real_retrieval` Baseline。报告记录的 Ranked
Recall@1/3/5/10 为 `0.3333 / 0.8333 / 0.8333 / 0.9444`，MRR 为 `0.6111`，
nDCG@5 为 `0.6422`；存在 1 个 Ranking Failure 和 1 个 Unanswerable Handling Failure。
该报告绑定当时的 commit、Dataset 和 Index 指纹，不代表资料或索引变化后的当前指标。

## Stage 3.1：BM25 + Dense Vector + RRF

Stage 3.1 只改变 Candidate Generation 与 Ranking。Dense 保持 Top 10 和 0.25 阈值，
BM25 从完整 Corpus 独立取 Top 10，RRF 使用 `k=60`；后续 Top 3 Seed、同源 ±2
Adjacent Expansion、最多 8 个 Context Chunk、Evidence、Citation、Prompt 和 LLM
流程保持不变。旧 Dense + Keyword Coverage 公式仍作为同次实验中的 Baseline。

以下命令默认只显示费用边界，不打开索引、不创建 Embedding Client：

```powershell
python -m app.evaluate_hybrid_retrieval
```

显式确认后，每个 Stage 2 Retrieval Case 只调用一次真实 Query Embedding，同一次
Dense 结果同时供 Baseline 与 Hybrid 使用；BM25、RRF 和报告聚合均为本地计算，
Chat/LLM 调用为 0：

```powershell
python -m app.evaluate_hybrid_retrieval --confirm-query-embedding-cost
```

报告分别保存 Baseline/Hybrid 的 Recall@1/3/5/10、MRR、nDCG@5、指标差值、逐案例
Gold 排名变化和两条 Pipeline 的本地延迟。代码与 Fixture/Mock 测试不能代替这次真实
受控实验。

2026-08-20 已在同一 disposable snapshot 上完成 10 个案例、10 次真实
`text-embedding-v4` Query Embedding、0 次 ChatModel 的 `local_real_retrieval` A/B：

- Recall@1：`0.3333 → 0.3333`
- Recall@3：`0.8333 → 0.8333`
- Recall@5：`0.8333 → 0.9444`
- Recall@10：`0.9444 → 0.9444`
- MRR：`0.6111 → 0.5778`
- nDCG@5：`0.6422 → 0.6561`
- 平均本地 Retrieval 延迟：`149.0 ms → 149.4 ms`

Hybrid 把 `wishart_definition` 的最佳 Gold 从第 6 名提升到第 3 名，但把
`cauchy_properties` 从第 2 名降到第 5 名，并把 `dirichlet_definition` 从第 2 名降到
第 3 名。由于正式链路只选择 Top 3 Seed，Recall@3 没有提升且 MRR 下降，当前结论是
“证明了 Sparse 独立召回的补充价值，但尚未证明 RRF 配置整体优于 Baseline”。因此
正式问答继续使用 Stage 2 Baseline；Stage 3.2 是否引入 Reranker，应先扩大评测集并
定位 RRF 的排序退化，而不是直接继续叠加组件。

## Stage 3.2：Cross-Encoder Reranker 受控实验

Stage 3.2 在不修改 Dense、BM25、RRF、Chunk、Evaluation Dataset 和 Context
Construction 的前提下，只对 Stage 3.1 的 RRF Candidate Pool 执行本地 Cross-Encoder
重排。正式 `RAGService` 仍使用 Stage 2 Baseline，实验不会自动进入问答主链路。

费用与模型下载都需要显式确认：

```powershell
python -m app.evaluate_reranker `
  --confirm-query-embedding-cost `
  --confirm-model-download-and-local-inference
```

2026-08-20 使用固定 revision 的 `BAAI/bge-reranker-base`、CPU、最多 20 个候选，完成
10 个案例、10 次成功运行 Query Embedding、130 对本地 Pair Scoring、0 次 ChatModel
的 `local_real_cross_encoder` A/B/C。C 相对 B 的 Recall@1/3 分别提高 `0.5000 / 0.1111`，
MRR 提高 `0.3481`，nDCG@5 提高 `0.2454`；但 Context Precision 降低 `0.0119`，平均
Retrieval 延迟增加约 `2862.4 ms`。因此当前只保留隔离实验能力，不接入正式 Pipeline。
完整证据和边界见 `docs/stage3-2-completion-report.md`。

## Stage 3.3：Structure-aware Chunking 受控实验

Stage 3.3 只改变 Chunking Strategy。A 使用正式 Pipeline 当前的固定 180 字符策略；
B 按标题、段落、连续列表和代码块组织内容，超限时按句子、词和字符回退，最大 600
字符、0 overlap。Embedding、Dense Top 10、0.25 阈值、80/20 排序、Top 3 Seed、同源
±2 Adjacent Expansion 和 Context Limit 8 全部冻结。正式 `build_chunks()`、资料摄取与
RAG Pipeline 保持不变。

默认运行只解析资料、重建 Gold Mapping 并显示费用边界，不创建 Embedding Client：

```powershell
python -m app.evaluate_chunking
```

真实 A/B 会使用同一 Embedding 配置在临时目录分别重建索引，并需要同时确认索引与查询
费用；生产索引只做只读核验与前后指纹检查：

```powershell
python -m app.evaluate_chunking `
  --confirm-controlled-index-embedding-cost `
  --confirm-query-embedding-cost
```

当前 10 案例真实实验中，Chunk 数 `147 → 55`，Recall@1、MRR、nDCG@5 分别提升
`0.2222 / 0.0315 / 0.0598`；但 Recall@3、Context Precision、Final Context Recall
分别下降 `0.1667 / 0.0298 / 0.2222`，并新增 Recall / Ranking Failure。因此只保留
隔离实验能力，不把 `structure-aware-block-600-overlap-0-v1` 接入生产 Pipeline。
完整证据和边界见 `docs/stage3-3-completion-report.md`。

## Stage 3.4：Context Optimization 受控实验

Stage 3.4 冻结 Stage 2 Retrieval、固定 180 字符 Chunk、Embedding、阈值、Top 3 Seed、
同源 ±2 Adjacent Expansion 和最多 8 块的 A 组 Context。B 组只在已扩展 Evidence 内
选择：保留全部 Seed，再为每个 Seed 选择一个 query keyword coverage 最高的相邻块；
分数并列时按更近距离和原 Context 顺序稳定处理。没有重新召回，也没有模型选择器。

以下命令复用已完成的 Stage 2 Retrieval Trace，并在 disposable index snapshot 中核对
对应 Chunk；不会调用 Query Embedding、ChatModel 或 Reranker：

```powershell
python -m app.evaluate_context
```

当前 10 案例 `local_historical_retrieval_trace_replay` 结果：Context Precision
`0.1500 → 0.2000`，Final Context Recall 保持 `1.0000`，平均 Context Chunk
`7.3 → 5.2`，平均字符 `1222.2 → 848.7`，Selector 平均增量约 `0.1042 ms`。
10 个案例均未移除 Gold；`mysql_out_of_scope` 仍返回 6 块 Context，因此无答案处理仍是
未解决边界。当前没有与 ChatModel 匹配的可靠 tokenizer，Token 数明确未测量。

由于 Precision 提升、Recall 保持且本地增量延迟很小，B 组已进入正式 `RAGService`
的 LCEL Pipeline。结论只绑定当前 10 案例与历史 Retrieval Trace，不代表真实回答质量、
Citation Support、并发或生产稳定性。完整证据见 `docs/stage3-4-completion-report.md`。

## 最小 FastAPI 服务

启动本地服务：

```powershell
python -m uvicorn app.api:app --host 127.0.0.1 --port 8000 --no-access-log
```

启动后可访问：

- `GET http://127.0.0.1:8000/`：打开最小 Web 问答页；页面加载只调用零费用的
  `/health`，不会自动发起问答；
- `GET http://127.0.0.1:8000/health`：只检查 API 进程是否响应，不读取模型配置、
  不打开 Chroma，也不调用 Embedding 或 Chat API；
- `POST /api/auth/register`：提交 `email`、至少 10 字符的 `password` 与
  `display_name`；密码只以 scrypt 哈希保存，成功后设置 HttpOnly 登录 Cookie；
- `POST /api/auth/login`、`POST /api/auth/logout`、`GET /api/auth/me`：登录、撤销当前
  服务端 Session 与读取当前身份；
- 下述业务接口都要求同源登录 Cookie 或有效 `Authorization: Bearer <token>`；用户归属
  只取后端验证后的 `current_user`，不接受客户端 `user_id` 作为身份来源；

- `POST http://127.0.0.1:8000/api/ask`：请求体为
  `{"question": "RAG 为什么需要切分资料？"}`，返回 `answer` 和 `sources`；
- `POST http://127.0.0.1:8000/api/agent`：请求体必须包含
  `{"message": "列出当前资料", "confirm_api_cost": true}`；可选
  `"allow_web_preview": true` 只授权本次请求预览用户明确提供的公开 URL；返回
  `answer`、`sources` 和 `tools_used`；
- `POST http://127.0.0.1:8000/api/tutor/chat`：请求体为
  `{"message": "帮我出题练习 Embedding", "session_id": "<UUID>",
  "confirm_api_cost": true}`；Session 必须属于当前登录用户；返回 Tutor 意图、学习动作、
  Evidence/Citation、可选练习或总结，并持久化成功的对话与学习行为；
- `POST /api/sessions`：请求体为 `{"topic": "Embedding"}`，为当前用户创建 Session；
  `GET /api/sessions` 最多返回当前用户最近 100 个 Session；
- `GET /api/history`：最多返回当前用户最近 100 条 Tutor 消息和 100 条学习记录；
  可用 `session_id=<UUID>` 查询参数限定到属于该用户的 Session；
- `POST http://127.0.0.1:8000/api/study-workflows`：请求体为
  `{"goal": "理解 RAG 的证据约束", "confirm_api_cost": true}`；调用一次受限 Agent
  整理资料证据、生成三步计划，并在返回前暂停等待人工确认；
- `POST /api/study-workflows/{workflow_id}/confirm`：用
  `{"decision": "approve"}` 或 `{"decision": "reject"}` 恢复同一检查点；
- `POST /api/study-workflows/{workflow_id}/progress`：使用 `note` 和
  `complete_current_task` 零模型调用地更新进度；失败工作流的 `/retry` 需要再次确认
  API 费用且最多一次；`GET` 读取最新状态，`DELETE` 经确认后删除全部本地检查点；
- `GET http://127.0.0.1:8000/api/materials`：零费用列出当前资料文件；
- `POST http://127.0.0.1:8000/api/materials/stage`：以 `multipart/form-data`
  暂存并本地解析单个资料，不读取模型配置、不打开 Chroma；
- `POST http://127.0.0.1:8000/api/web-materials/preview`：请求体为
  `{"url": "https://example.com/", "operation": "add"}`，安全抓取单个公开网页、
  使用 Crawl4AI 生成 Markdown 并暂存；不打开 Chroma，也不调用 Embedding 或 Chat；
- `POST http://127.0.0.1:8000/api/materials/{upload_id}/index`：请求体必须为
  `{"confirm_api_cost": true}`；明确确认后创建持久化后台任务并立即返回 `202`、
  `job_id` 和初始状态，不在该请求内执行重解析、Embedding 或 Chroma 写入；批量入口
  `/api/materials/batch/index` 使用同一任务契约；
- `GET http://127.0.0.1:8000/api/jobs/{job_id}`：查询文档任务的
  `pending / processing / completed / failed` 状态、粗粒度进度、失败原因和完成后的索引
  摘要；
- `DELETE http://127.0.0.1:8000/api/materials/{filename}`：请求体必须为
  `{"confirm_delete": true}`，删除资料文件和对应来源的 Chroma 记录，不调用 Embedding；
- `GET http://127.0.0.1:8000/docs`：FastAPI 自动生成的 Swagger UI。

每个 HTTP 响应都会带一个由服务端生成的 `X-Request-ID`。Uvicorn 控制台还会输出
对应的单行 JSON，例如：

```text
{"timestamp":"2026-08-08T01:00:00Z","event":"http_request_completed","request_id":"...","method":"POST","path":"/api/ask","status_code":200,"elapsed_ms":1260,"error_category":null}
```

日志只记录请求 ID、方法、已匹配路由、状态码、耗时和错误类别；不记录查询参数、
请求体、问题、回答、来源、异常原文、API Key 或 Base URL。未知路由统一记为
`unmatched`。启动命令使用 `--no-access-log` 关闭可能包含原始 URL 和查询参数的
Uvicorn 默认访问日志，由上述结构化日志替代。相同的安全字段还会追加到
`data/request_logs/http_requests.jsonl`；单文件达到 1 MiB 前会轮转，最多保留
3 份备份。持久化失败不会影响 HTTP 响应，也不会把原始异常写入日志。

当前错误类别包括 `request_validation`、`rate_limited`、`rag_processing`、`rag_unavailable`、
`agent_processing`、`agent_timeout`、`agent_protected`、`tutor_processing`、
`tutor_timeout`、`tutor_protected`、`workflow`、
`workflow_protected`、`web_preview`、
`web_preview_protected`、`client_error`、`server_error` 和
`unhandled_exception`。未预期异常会返回不含内部
异常信息的通用 `500`，并保留可与日志关联的 `X-Request-ID`；为避免敏感信息进入
控制台，当前不会记录异常原文或完整 traceback。

`/api/ask` 会在第一次问答时按需构造并复用现有 `HybridRetriever`、ChatModel 和
LCEL Runnable，没有复制另一套检索或 Prompt。每次有效问答会产生一次真实问题 Embedding；检索到
资料时还会产生 Chat API 调用，因此可能产生费用。问题为空、仅含空白、超过 2000
个字符或包含未声明字段时返回 `422`；RAG 初始化失败返回不含内部配置的 `503`，
检索或 ChatModel 失败返回不含堆栈及敏感详情的 `502`。

`/api/agent` 每次都要求 `confirm_api_cost=true`，因为即使最终只列出本地资料，Agent
也需要模型判断工具。它最多调用 Agent 路由模型 3 次、总工具 2 次，其中资料问答和
网页预览各最多 1 次；资料问答工具还会按现有 RAG 链调用问题 Embedding 和 ChatModel。
Agent 执行最多等待 90 秒；已经进入同步 RAG 的外部调用仍受各自客户端超时约束。
`answer_from_materials` 复用上述 LCEL RAG，并由
服务端保留原始回答和来源，避免 Agent 二次改写证据；`list_available_materials` 只列
文件名与大小；`preview_web_material` 还要求本次请求显式授权，只复用安全预览且永不
写入索引。工具异常会转换为不含 URL、密钥、路径和原始异常的安全结果。

## V2 Stage 4：Tutor Agent + LangGraph Learning Workflow

`app/tutor_workflow.py` 使用一个显式 `StateGraph` 组织单 Tutor 学习流程。Router 采用
可复现的确定性分类，不额外调用模型；Knowledge Retrieval Tool 直接复用
`RAGService.ask()`，所以 Stage 3 的 Context Selector、Evidence 与 Citation 契约继续
生效。Quiz Generator Tool 和 Learning Summary Tool 使用当前 ChatModel 的结构化输出，
资料不足时不会继续生成练习或基于模型知识补全。

Stage 4 最初使用 `InMemorySaver`，最多保留最近 20 条、合计 12000 字符，并让“继续出
一道题”等续问继承上一轮 topic。该阶段当时不支持进程重启恢复；这一限制由后续
Stage 5.1 的持久化数据层替代。旧 `/api/ask`、`/api/agent` 和
`/api/study-workflows` 继续可用。QA、解释和
学习规划最多执行现有 RAG 的 Query Embedding 与 Chat 调用；Quiz 和
新主题 Summary 会在此基础上增加一次结构化 Chat 调用；Session Summary 只执行一次
结构化 Chat 调用。完整边界和验证证据见 `docs/stage4-completion-report.md`。

## 历史 checkpoint：Persistent Learning Data Foundation（stage5-part1）

`app/learning_data.py` 使用 SQLite、`aiosqlite` 和显式 SQL 管理 `users`、
`learning_sessions`、`conversation_messages`、`learning_records` 与
`schema_migrations`。数据库默认位于 `data/learning/learning.sqlite3`，并被
`.gitignore` 排除。首次打开会在事务中应用版本 1 迁移；高于程序支持版本的数据库会
拒绝打开，不会猜测降级。

Tutor 不把长期学习历史拼进 Prompt，也不把 LangGraph State 当作长期业务数据库：
每次调用先按 `user_id + session_id` 校验归属，从业务表加载最近有限对话作为当前推理
输入；LangGraph 使用同一数据库文件中的 `AsyncSqliteSaver` 保存线程级短期 checkpoint；
成功后再以一个业务事务写入 user/tutor 两条消息、学习动作和更新后的 Session topic。
因此服务关闭重开后仍能继续同一 Session，而完整历史和学习行为有独立、可查询的数据
模型。

本阶段选择 SQLite 是因为当前仍是本地单实例原型，并且仓库已经锁定
`aiosqlite` 与 `langgraph-checkpoint-sqlite`；没有真实并发或部署证据支持立即引入
PostgreSQL、连接池和新迁移框架。当前只建立基于 UUID 和外键查询的用户数据分区，
当时尚无认证授权；这份历史边界由当前 Stage 5.1 身份与隔离阶段补齐。原始设计证据保留在
`docs/stage5-1-completion-report.md`，没有被新报告覆盖。


## V2 Stage 5.2：Async Document Processing Pipeline

`app/document_jobs.py` 使用独立 SQLite 数据库
`data/jobs/document_jobs.sqlite3` 持久化文档 Job，并由单进程 `asyncio` Worker 串行消费。
现有资料暂存与费用预估仍是零模型调用的同步本地校验；用户确认费用后，索引 API 只校验
暂存文件、创建 `pending` Job 并返回，真正的资料重解析、Chunk 构建、Embedding、Chroma
增量同步和 RAG 缓存失效由 Worker 完成。Worker 没有复制摄取实现，而是继续调用
`MaterialManager.estimate_index_batches*()` 和 `commit_staged*()`，因此 Manifest 只读保护、
单次/进程费用上限、文件补偿回滚和现有 Parser/Chunker/Embedding 契约保持不变。

Job 支持 `pending`、`processing`、`completed`、`failed` 四种状态。进程重启时，尚未开始的
`pending` Job 会继续处理；中断时已经是 `processing` 的 Job 会被明确标记为 `failed`，
不会假装仍在执行，也不会在外部调用结果不确定时自动重复付费。当前进度是
`0 / 10 / 100` 的阶段信号，不是 Embedding 百分比。完整设计、验证证据和边界见
`docs/stage5-2-completion-report.md`。

## V2 Stage 5.1：User Identity & Data Isolation（stage5-part3）

当前身份系统使用 SQLite 保存用户与可撤销服务端 Session：密码通过标准库 scrypt 加盐
哈希，浏览器只持有 HttpOnly、SameSite=Strict Cookie，数据库只保存令牌 SHA-256。相比
JWT，这更适合当前同源原生前端与本地单实例 FastAPI：退出可立即撤销，不需要密钥轮换和
JWT 黑名单；代价是后续多实例部署需要共享 Session Store。

所有资料、RAG、Agent、Tutor、Session、历史、文档 Job 与 Study Workflow 请求都从
后端 `current_user` 取得身份。每个用户拥有
`data/user_workspaces/<user_uuid>/{documents,pending_uploads,pending_deletions,vector_store}`；
RAG、Citation 和 Agent 只打开该用户 Chroma，因此另一用户的文件名、摘录、
`material_id` 和 Citation 不会进入候选上下文。业务表通过 `user_id` 与复合外键约束
Session、Conversation、Learning Record 的归属，Job 与 Workflow 查询也校验所有权。

旧匿名用户、Session、对话和学习记录由 Schema v2 原样保留；旧消息从 Session 反填直接
`user_id`。旧全局资料/Chroma 与无归属 Job 不会自动分配给新账户，避免“第一个注册者认领
全部旧数据”；活动旧 Job 会明确失败，后续如需认领必须执行显式、可审计迁移。完整设计、
迁移、安全边界与验证证据见
`docs/stage5-1-user-identity-completion-report.md`。

## 阶段 7：LangGraph 学习规划工作流

`app/study_workflow.py` 使用 `StateGraph` 显式定义请求路由、目标校验、Agent 资料证据、
计划生成、人工确认、计划激活/拒绝、进度更新和最终复盘节点。创建工作流会调用阶段 6
Agent，因此必须确认费用；图在 `interrupt()` 处暂停，确认接口使用相同 `thread_id`
和 `Command(resume=...)` 恢复。批准后第一项任务进入 `in_progress`，每次进度调用只更新
本地状态，三项任务完成后生成确定性复盘；拒绝后不会写入进度或继续调用模型。

检查点保存在 `data/study_workflows/checkpoints.sqlite3`，由
`langgraph-checkpoint-sqlite` 持久化，并使用禁用 pickle fallback、显式空模块白名单的
严格序列化器。数据库被 `.gitignore` 忽略，保存目标、Agent 证据、来源、计划和进度，
不保存服务对象、密钥或原始异常；它是未加密的本地开发数据，不应同步或公开。每条
进度最多 100 项，失败不会自动重试付费 Agent；只有资料证据节点失败时，用户才能在
再次确认费用后手动重试一次。`DELETE` 接口可删除指定工作流的全部检查点。

Web 页面使用项目内原生 HTML、CSS 和 JavaScript，没有引入前端框架或外部资源。
页面会展示答案、候选来源和 `X-Request-ID`，并在请求期间禁用提交按钮以降低重复
点击造成重复计费的风险。只有点击“向资料提问”或按 `Ctrl + Enter` 才会调用
`/api/ask`；一次有效问题会产生真实 Embedding，并可能产生 Chat API 费用。

## 上传与增量索引

页面上传分为两步：

1. “本地校验与解析”会逐文件检查安全文件名、扩展名、MIME、文件大小和内容特征，
   复用 `app/chunk_documents.py` 解析和切块，不调用真实 API；
2. 页面分别展示成功文件和失败原因。一个文件失败不会掩盖其他完整暂存结果，也不会
   产生临时索引；
3. 页面汇总文本块数和预计 Embedding 批次数，只有再次点击“确认费用并写入索引”，
   才把成功文件作为独立 Material 一次提交，并复用一次
   `app.langchain_store.sync_vector_store()`。

单次最多接收 20 个文件，每个文件不超过 10 MiB。支持 UTF-8 TXT、Markdown、DOCX
和带文字层 PDF；PDF 最多 200 页，每个文件最多提取 200,000 个字符，整批切块后预计最多 60 个
Embedding 批次；阻止目录、盘符、路径分隔符、Windows 保留名、伪造 PDF、伪造 DOCX
和二进制文本。DOCX 即使被客户端声明为通用 ZIP MIME，也必须通过完整 OOXML 结构
校验后才能暂存；旧版 `.doc` 仍不支持。同名
新增返回冲突，不会静默覆盖；替换必须在页面明确选择。上传和替换失败会恢复原文件
并补偿同步旧索引；删除失败会把隔离文件移回正式资料目录，但无法证明定向删除完全
回滚时会返回 `503` 并提示人工检查索引。索引成功或回滚状态不确定后，进程内缓存的
RAG 服务会关闭并在下一次问答时重新初始化，避免继续持有旧资料状态。

## 阶段 5：公开网页预览与确认入库

先安装锁定依赖；当前实现使用 Crawl4AI 的本地 HTML 清理和 Markdown 生成能力，
不需要额外运行浏览器安装命令：

```powershell
python -m pip install -r requirements.txt
```

页面中的“导入公开网页”仍复用现有两步入库链路：

1. 服务端只接受默认端口的 HTTP/HTTPS URL，默认要求解析得到的全部地址均为公网 IP；
   实际连接固定到本轮已校验地址，并在最多 5 次重定向中的每一跳重新做 URL、DNS 和
   地址校验，以降低内网访问、云元数据访问、DNS 重绑定和重定向绕过风险；
2. 单页 HTML 最大 2 MiB，HTTP 连接与读取链路共享 20 秒预算，只接受 HTML 且不接受
   压缩响应；初次 DNS 解析仍受操作系统解析器时限约束。Crawl4AI 在已安全获取的原始
   HTML 上移除脚本、表单、iframe、图片等内容并生成最多 30,000 字的 Markdown；
   预览阶段只写入受限暂存目录，不读取模型配置、不打开 Chroma；
3. 暂存文件保存 canonical URL、标题、UTC 抓取时间和待索引正文 SHA-256，读取时会
   复核正文哈希；索引来源保留原始
   URL。只有用户再次点击“确认费用并写入索引”，才调用现有增量索引和 Embedding。

第一版只支持公开、免登录的单个静态 HTML 页面，不执行站点 JavaScript，不支持
Cookie、登录态、代理、Stealth、自定义 Hook/脚本、批量站点爬取或反爬绕过。网页正文
属于不可信资料；现有 RAG Prompt 明确将资料视为参考信息而非可执行指令，但仍需人工
检查预览、遵守目标网站条款，并把 Prompt Injection 视为残余风险。项目使用
[Crawl4AI](https://github.com/unclecode/crawl4ai) 完成网页 HTML 到 Markdown 的提取。

如果可信本机代理使用 Fake-IP DNS，并把正常域名解析到 `198.18.0.0/15`，可以只在
启动 Uvicorn 的当前 PowerShell 终端显式开启兼容模式：

```powershell
$env:STUDY_MATERIAL_ALLOW_PROXY_FAKE_IP="true"
.\.venv\Scripts\python.exe -m uvicorn app.api:app --host 127.0.0.1 --port 8011 --no-access-log
```

兼容模式只允许“域名 DNS 解析结果”使用 `198.18.0.0/15`，用户直接输入该网段 IP、
localhost、内网、链路本地和云元数据地址仍会被拒绝；初始 URL 与每次重定向使用同一
规则。该模式依赖可信本机代理接管 Fake-IP 连接，只适用于本地开发，不得在公开部署中
启用。关闭当前终端即可清除该进程环境变量，也可以执行：

```powershell
Remove-Item Env:STUDY_MATERIAL_ALLOW_PROXY_FAKE_IP
```

## 阶段 3：稳定性与安全保护

本地单进程使用一个非阻塞独占保护器协调问答、Agent、学习工作流、网页预览、暂存、索引和删除：已有受保护操作
执行时，新操作立即返回 `429`，不会无限排队。滚动窗口和费用保护默认值为：

- 问答每 60 秒最多 5 次、当前进程最多 50 次；
- Agent 每 60 秒最多 3 次、当前进程最多 20 次；
- 零模型的工作流确认、进度和删除合计每 60 秒最多 10 次；
- 索引每小时最多确认 5 次；调用 Embedding 前会按 Chroma 现有 ID 重新计算实际待新增
  文本块，单次最多 60 个逻辑批次、当前进程最多 200 批；
- 本地暂存和删除各每 60 秒最多 10 次；
- 公开网页预览每 60 秒最多 3 次；
- 超过 24 小时的、名称符合上传 ID 格式的暂存目录会被安全清理；
- Embedding 客户端超时为 30 秒，外部 API 最多重试 2 次。

频率限制或繁忙响应会尽量返回 `Retry-After`；进程预算耗尽后不会自动恢复，必须先
确认真实费用和运行状态，再由操作者决定是否重启。所有计数都保守地在操作开始前
记入，失败调用不会自动退还额度。索引回滚为恢复资料与索引一致性，必要时可能执行
一次额外的有界补偿同步。这些限制只存在于当前 Python 进程内，不支持多
进程共享，也不能替代鉴权、用户级配额、反向代理限流和平台侧账单告警。

`app/url_safety.py` 已接入 `app/web_materials.py`：只接受默认端口的 HTTP/HTTPS，禁止
凭据和片段，并要求域名解析得到的所有 A/AAAA 地址都是公网地址；实际连接固定到校验
结果，每次重定向重新解析和校验，以防止 DNS 重绑定和重定向绕过。策略依据
[OWASP SSRF Prevention Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Server_Side_Request_Forgery_Prevention_Cheat_Sheet.html)
和 Python [`ipaddress.is_global`](https://docs.python.org/3/library/ipaddress.html#ipaddress.IPv4Address.is_global)
语义制定。

## 请求历史与评测报告对比

持久化请求历史只包含固定的安全字段，不包含请求体、问题、回答、来源、查询参数、
密钥、Base URL 或异常原文。`data/request_logs/` 已被 `.gitignore` 忽略。

比较两份已有评测报告不会调用任何模型：

```powershell
python -m app.compare_rag_reports <基线报告.json> <当前报告.json>
```

如需独占创建一份对比 JSON：

```powershell
python -m app.compare_rag_reports <基线报告.json> <当前报告.json> --output <对比报告.json>
```

命令会比较评测集 SHA-256、通过数量、新增失败、恢复案例、新增/删除案例和耗时变化。
出现新增失败时退出码为 `1`；输入或写入错误为 `2`。若两份评测集 SHA-256 不同，
报告会明确警告总通过率不能直接比较。

当前 API 已覆盖本地问答、受限 Agent、可恢复学习工作流、上传与增量索引、资料删除、有限请求历史和单进程保护，但
仍未实现鉴权、跨进程/跨实例配额和公开服务所需的完整并发性能基线，不应直接作为
公网生产服务部署。

## 验证

不调用真实 API 的本地检查：

```powershell
python -m unittest discover -s tests -v
python -m compileall app tests
python -m pip check
node --check web\static\app.js
```

## 后续开发阶段清单

以下状态以当前工作区代码、自动测试、已有真实评测报告和用户提供的本地运行
截图为准。显式 LCEL 固定问答管道、受限 Crawl4AI 网页预览、单个受限 LangChain Agent
和自定义 LangGraph 学习规划工作流已经实现。LCEL 只重组固定问答流程；动态工具选择
属于阶段 6 Agent；状态、条件路由、持久化检查点和人工中断属于阶段 7 LangGraph。

状态标记：

- **当前已完成**：已有真实代码和相应验证证据；
- **下一阶段**：当前优先实施，但本清单不代表已经实现；
- **后续阶段**：依赖前置能力稳定后再进入；
- **可选增强**：不纳入第一版完成标准。

### 当前已完成

| 阶段 | 主要内容 | 完成标志 | 关键依赖 / 边界 |
| --- | --- | --- | --- |
| **当前已完成 1：资料解析与手写 RAG** | TXT、Markdown、PDF 解析；固定字符切分；关键词检索；手写 Embedding、余弦相似度和端到端 RAG。 | CLI 可以基于本地资料回答并展示来源。 | 手写版本用于解释底层原理，后续阶段不得删除或改写成框架版本。 |
| **当前已完成 2：LangChain + Chroma 增量索引** | Chroma 本地持久化；稳定文档 ID；新增、变化和失效内容同步。 | 未变化资料不会重复向量化，已有索引可复用。 | 真实索引会产生 Embedding 费用；自动测试只能使用 Fake/Mock。 |
| **当前已完成 3：混合检索与完整 RAG** | HybridRetriever、向量与关键词混合排序、邻接文本扩展、来源标签、无证据拒答和输出归一化。 | CLI 与 API 复用同一套 Retriever、Prompt 和回答处理逻辑。 | 当前参数只通过固定小型评测集校准，不代表对所有资料普遍最优。 |
| **当前已完成 4：PDF 中文正文解析** | 使用 pdfplumber 按页提取带文字层 PDF，并保留页码来源。 | 中文正文基本可读；扫描件、加密或损坏 PDF 有清晰错误。 | 复杂公式仍可能出现符号丢失、`(cid:xx)` 或阅读顺序问题。 |
| **当前已完成 5：固定评测与 JSON 报告** | 10 个固定案例；来源、引用、必含词、拒答、LaTeX 和耗时检查；唯一 JSON 报告。 | 最新一次真实固定评测通过当前 10/10 案例，失败案例也能保存。 | 10/10 只代表这 10 个案例；模型输出具有概率性，修改后仍需复测。 |
| **当前已完成 6：最小 FastAPI 服务** | `/health`、`/api/ask`、`/docs`；惰性 RAG 初始化；安全的 422、429、502、503 和 500 错误。 | 本地真实 `/api/ask` 已返回 `200`；健康检查不调用模型或产生费用。 | 当前主要调用为同步方式，阶段 3 保护只覆盖单进程。 |
| **当前已完成 7：基础结构化请求日志** | `X-Request-ID`；状态码、耗时和错误类别的单行 JSON 控制台日志。 | 页面请求 ID 能与 Uvicorn 日志关联，不记录问题、回答、来源和敏感配置。 | 已由阶段 2 增加有限 JSONL 持久化；启动时仍需使用 `--no-access-log` 关闭原始访问日志。 |
| **当前已完成 8：最小 Web 问答页** | 原生 HTML、CSS 和 JavaScript；API 状态、登录注册、个人资料管理、问题输入、答案、来源、错误和请求 ID 展示。 | 用户已在历史版本完成真实问答与上传闭环；当前认证页面通过静态结构与 API 自动化验证。 | 本轮应用内浏览器连接被 Windows 沙箱阻断，登录后的真实视觉回归尚未完成。 |
| **当前已完成 9：上传与增量索引** | 单文件安全暂存；本地解析；二次费用确认；新增、显式替换、删除、失败回滚和 RAG 服务刷新。 | Fake/Mock 覆盖上传、变化、删除、失败和安全边界；用户已真实验证 PDF 上传、增量索引和来源问答。 | 真实入库会产生 Embedding 费用；尚未进行并发、长时间和多类型资料矩阵验收。 |
| **当前已完成 10：运行历史与评测对比** | 隐私安全请求元数据持久化为 1 MiB、3 备份的轮转 JSONL；零费用比较两份结构化评测报告。 | 能标出新增失败、恢复案例、案例集合和耗时变化；评测集 SHA 不同时给出不可直接比较的警告。 | 请求历史仍只适合本地单进程；固定案例结果不能外推为普遍效果。 |
| **当前已完成 11：阶段 3 稳定性与安全保护** | 非阻塞单进程独占、滚动窗口限流、问答/索引进程预算、上传解析上限、暂存 TTL、外部调用超时重试、统一错误页和 URL 公网校验基础。 | Fake/Mock 覆盖繁忙、频率、预算、PDF/字符/批次限制、清理和私网/混合 DNS 拒绝；本地页面/API 以零费用方式验证。 | 当前已有最小身份认证，但仍无用户级费用配额、分布式保护和并发压测；网页预览保护仍只适用于本地单进程。 |
| **当前已完成 12：阶段 4 LCEL 管道化 RAG** | 将“问题 → 检索 → 证据判断 → Prompt → 模型 → 结果处理”组合为可复用 Runnable，并保留混合检索、拒答、归一化和来源契约。 | CLI 与 FastAPI 经 `RAGService` 复用同一管道；Fake/Mock 验证成功、拒答、检索失败、模型失败和无证据时跳过 ChatModel。 | 本阶段未调用付费 API；此前真实 10/10 评测早于本次 LCEL 改造，不能作为改造后的真实效果证据。 |
| **当前已完成 13：阶段 5 Crawl4AI 网页资料导入** | 公网 URL/DNS/每跳重定向校验；固定到已验证 IP 的受控单页抓取；Crawl4AI 本地 Markdown 预览；来源元数据；复用现有暂存与确认索引链路。 | Fake/Mock 覆盖 SSRF、重定向、超限、错误映射和隐私日志；本地原始 HTML 已实际通过 Crawl4AI 清理转换，API/UI 契约已自动验证；用户已在本地页面完成 `qiuzhi2046.com` 的真实公开网页 Markdown 预览。 | 真实网页付费入库与问答仍未验收；不支持 JavaScript 渲染、登录态或批量爬取。部分代理/Fake-IP DNS 环境会把域名映射到保留测试网段并被 SSRF 保护正确拒绝。 |
| **当前已完成 14：阶段 6 LangChain Agent 工具编排** | 使用 `create_agent` 编排 `answer_from_materials`、`list_available_materials` 和 `preview_web_material` 三个受限工具；提供 `/api/agent`、单次费用确认、网页预览独立授权、模型/工具/总时限和单进程预算。 | Fake/Mock 验证工具选择、LCEL 回答与来源逐字保留、网页预览不入库、未授权拒绝、工具异常脱敏、重复付费工具限制、超时和 API 错误契约。 | 尚未调用真实工具调用模型，不能把自动测试当作真实 Agent 效果验收；没有索引、删除、任意文件或任意网络工具，也没有对话记忆和自定义 LangGraph 状态流。 |
| **当前已完成 15：阶段 7 LangGraph 学习规划工作流** | 用显式 `StateGraph` 管理目标、受控 Agent 证据节点、三步计划、`interrupt` 确认、批准/拒绝分支、进度、复盘、一次手动重试和 SQLite 检查点删除。 | Fake/Mock 覆盖完整状态流、路由原因、费用确认、拒绝后停止、进度上限和隐私日志；真实 SQLite 文件关闭并重开后可以读取并恢复等待确认的线程。 | 尚未调用真实模型验收工作流效果；检查点是未加密的本地单实例数据，没有鉴权、跨实例锁、后台清理或任意对话长期记忆。 |
| **当前已完成 16：历史 stage5-part1 持久化学习数据** | 服务端 UUID 用户、学习 Session、Tutor 消息、学习行为、版本迁移和 SQLite LangGraph checkpoint。 | 临时真实 SQLite 覆盖创建、归属分区、历史查询和关闭重开后的 Tutor 续学。 | 当时只有 UUID 分区；当前 Stage 5.1 已在其上补齐可信身份与所有权检查。 |
| **当前已完成 17：V2 Stage 5.2 异步文档处理** | SQLite Job、单进程后台 Worker、任务状态查询、启动恢复和失败落库；确认索引接口改为返回 `202 + job_id`。 | Fake/Mock 与临时真实 SQLite 覆盖单文件/批量成功、失败、去重、重启继续 pending、处理中断转 failed 和 API 查询。 | 暂存预解析仍同步；当前 Stage 5.1 已为 Job 增加所有权，仍没有分布式队列、多实例抢占、自动重试或负载测试。 |
| **当前已完成 18：V2 Stage 5.1 身份与数据隔离** | 邮箱注册登录、scrypt 密码哈希、可撤销服务端 Session、`current_user`、复合外键、每用户资料/Chroma 与 Job/Workflow 所有权。 | 392 项 Fake/Mock/临时 SQLite 回归覆盖认证失败、A/B 数据隔离、Tutor、Material、Vector Retrieval、Citation 来源、Job 与 Workflow IDOR。 | 仍是本地单实例 SQLite；无 OAuth、RBAC、密码重置、登录限流、多实例 Session、数据库加密、生产备份或公开部署验收。 |

### 下一阶段与后续阶段

| 阶段 | 主要内容 | 完成标志 | 与前一阶段的依赖 / 核心风险与边界 |
| --- | --- | --- | --- |
| **阶段 8（下一阶段）：部署** | 管理环境变量和启动配置、持久化目录、健康检查，并选择 Docker 或合适平台；区分 FastAPI 服务、Chroma 数据、LangGraph SQLite 检查点和网页抓取运行资源。 | 获得可公开访问的演示地址；重启后资料索引和必要状态不丢失；健康检查不调用模型或产生费用。 | 当前原始 HTML 转换不启动浏览器；公开服务还需要鉴权、跨实例配额、数据库并发方案、检查点加密/备份和完整性能基线。 |
| **阶段 9（后续阶段）：演示和求职材料** | 补充 README 架构图和完整运行步骤，说明 RAG、LCEL、Crawl4AI、Agent 和 LangGraph 的职责边界；整理评测证据、典型问题、拒答案例、故障定位案例、演示视频和项目讲解。 | 项目可以写入简历；能在面试中讲清需求、架构、取舍、验证和风险；明确区分已实现、已自动化验证、已真实运行和后续规划。 | 依赖部署与证据归档；不得把固定评测案例、原型能力或未上线功能包装成普遍稳定的生产成果。 |

排序遵循“先稳定资料进入和可观测性，再重组固定 RAG 管道，随后增加网页采集和
动态工具选择，最后才引入有状态工作流”的依赖关系。四类能力的职责不能混用：

- **LCEL**：固定、可组合、可测试的 RAG 能力管道；
- **Crawl4AI**：网页资料采集与 Markdown 预览，不负责回答和流程控制；
- **LangChain Agent**：由模型在受限工具集合中动态选择下一步；
- **LangGraph**：显式管理状态、节点、路由、中断、恢复和人工确认。

### 可选增强

| 能力 | 采用条件 | 当前边界 |
| --- | --- | --- |
| 多 Agent 协作 | 单 Agent 和受控 LangGraph 节点已经无法清晰覆盖真实需求，并且有可验证的职责拆分。 | 不纳入第一版完成标准，不为了简历技术名堆叠而引入。 |
| 长期记忆 | 已明确需要跨会话保留哪些用户状态、保留期限、删除方式和隐私边界。 | 当前只规划工作流状态恢复，不默认保存任意对话或用户内容。 |
| 复杂自适应抓取 | 基础公开网页导入稳定，并有合法性、资源成本和反爬策略评估。 | 暂不支持登录态、Stealth、任意脚本、自动绕过限制或无边界深度抓取。 |

规划阶段的 API 和术语以当前官方文档为准：

- [LangChain Runnables / LCEL](https://reference.langchain.com/python/langchain-core/runnables)
- [LangChain Agents 与 `create_agent`](https://docs.langchain.com/oss/python/langchain/agents)
- [LangGraph Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api)
- [LangGraph 持久化与人工中断](https://docs.langchain.com/oss/python/langgraph/persistence)
- [Crawl4AI Simple Crawling](https://docs.crawl4ai.com/core/simple-crawling/)
- [Crawl4AI Local Files & Raw HTML](https://docs.crawl4ai.com/core/local-files/)
- [Crawl4AI Deep Crawling](https://docs.crawl4ai.com/core/deep-crawling/)
