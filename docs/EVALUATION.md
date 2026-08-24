# 知行（Study Material Assistant）· Evaluation Guide

## 1. 为什么需要独立评测

RAG 的回答问题可能来自不同层：Parser 丢内容、Chunk 切分错误、Retrieval 没召回、Ranking
把 Gold 压到后面、Context Construction 引入噪声、模型生成错误，或 Citation 没有支持主张。

如果只看一条最终回答，很容易把检索问题误判为 Prompt 问题。知行把 Retrieval、Context、
Answer 和 System 指标分开，并要求每次优化绑定 Dataset、Baseline、配置、Git 和逐案例结果。

## 2. 评测资产

| 资产 | 路径 | 作用 |
| --- | --- | --- |
| 端到端 RAG Dataset | `evaluation/rag_cases.json` | Answer/Citation 的固定案例 |
| Retrieval Dataset | `evaluation/retrieval_cases.json` | Stable Gold Meaning、Current Chunk Mapping、可回答/不可回答案例 |
| 报告目录 | `evaluation/results/` | 本地生成、唯一命名的 JSON 报告；当前不作为 Git 状态事实 |
| Baseline 说明 | `docs/stage3-1-baseline.md` | Stage 2 固定 Retrieval 基线与历史运行 |
| 实验报告 | `docs/stage3-*-completion-report.md` | Stage 3.1～3.4 的指标、逐案例变化、成本和决策 |

历史报告绑定当时的 commit、Dataset 和 Index/Trace。资料、索引或配置变化后，历史数字不能当作
当前重新验证结果。

## 3. 指标口径

### Retrieval

- `Recall@K`：前 K 个候选是否覆盖 Gold；Multi-Gold 按 Gold 覆盖率计算；
- `MRR`：第一个 Gold 的倒数排名，强调前排是否尽快命中；
- `nDCG@5`：前 5 名的排序质量；单 Gold 时与 MRR 信息可能高度重合；
- `Raw Recall`：Dense 原始候选；
- `Ranked Recall`：过滤和重排后的候选；
- `Final Context Recall`：最终送入 Context 的 Gold 覆盖；
- `Context Precision`：最终 Context 中 Gold 占比。

### Answer / Citation

当前端到端 Runner 有确定性检查和固定案例，但不能把“Citation ID 有效”外推为 Claim-level
Citation Support。需要分别报告：

- Answer Correctness：答案是否正确；
- Faithfulness：答案主张是否能从 Evidence 推出；
- Citation Validity：Citation ID 是否存在于当前 Evidence Map；
- Citation Support / Accuracy：引用是否支持对应主张；
- Refusal：资料不足时是否拒答。

当前稳定工程证据强于 Claim-level Answer/Citation 基准；后两项仍是明确未完成边界。

### System

- Retrieval / LLM / Job latency；
- 外部调用次数；
- token usage（仅在 Provider 返回时记录）；
- 本地 CPU Reranker 或索引构建成本；
- 失败、超时、空召回和不可回答处理。

Token usage 不等于货币成本。没有版本化价格表时不能报告金额或“节省百分比”。

## 4. 失败分类

```text
Parser Failure
Chunk / Gold Mapping Failure
Recall Failure
Filtering Failure
Ranking Failure
Context Construction Failure
Generation Failure
Citation Failure
Unanswerable Handling Failure
Infrastructure / Configuration Failure
```

报告必须保留 case-level 分类，不能只给平均数。一个平均指标上升但关键案例进入 Top 3 失败，
仍可能拒绝接入正式 Pipeline。

## 5. 运行前边界

1. 确认 Git commit、Dataset hash、Index Manifest 和原始 Index 指纹；
2. 保持 Baseline、Gold 与非实验变量固定；
3. 先运行不带费用确认参数的 preflight，读取案例数和预计调用；
4. 真实调用前由用户明确授权 Embedding、ChatModel、模型下载和本地推理；
5. Retrieval 评测只查询 disposable snapshot，不能修改原始 Chroma；
6. 记录外部调用数、模型、配置、报告路径和退出码；
7. 报告不能覆盖旧文件；
8. 实验后检查原始 Index 指纹不变。

## 6. 安全命令

以下命令在不带确认参数时只执行 preflight，并以非零退出提示未运行真实实验：

```powershell
.\.venv\Scripts\python.exe -B -m app.evaluate_rag
.\.venv\Scripts\python.exe -B -m app.evaluate_retrieval
.\.venv\Scripts\python.exe -B -m app.evaluate_hybrid_retrieval
.\.venv\Scripts\python.exe -B -m app.evaluate_reranker
.\.venv\Scripts\python.exe -B -m app.evaluate_chunking
```

Stage 3.4 Context 评测复用历史 Retrieval Trace，不调用 Embedding 或 ChatModel：

```powershell
.\.venv\Scripts\python.exe -B -m app.evaluate_context
```

真实运行的费用门槛：

| Runner | 必需确认参数 | 外部/本地成本 |
| --- | --- | --- |
| End-to-end RAG | `--confirm-api-cost` | 每 Case Query Embedding；有证据时 Chat |
| Retrieval | `--confirm-query-embedding-cost` | 每 Case 一次 Query Embedding |
| Hybrid | `--confirm-query-embedding-cost` | 每 Case 一次 Dense Query Embedding；BM25/RRF 本地 |
| Reranker | 上述参数 + `--confirm-model-download-and-local-inference` | Query Embedding + 模型下载/CPU 推理 |
| Chunking | `--confirm-controlled-index-embedding-cost` + `--confirm-query-embedding-cost` | 两个临时索引 + A/B Query Embedding |

本 Stage 6 文档工作不运行这些付费命令。

## 7. 对照实验模板

每次实验至少写清：

- Problem：当前失败案例和所在层；
- Baseline：commit、Dataset、Index、参数；
- Hypothesis：单一改变为何会影响哪个指标；
- Prediction：预期改善和可能回归；
- Keep Fixed：其余 Pipeline、Gold、模型和次数；
- Primary Metric：决定成败的首要指标；
- Guardrails：不能下降的指标、延迟和费用上限；
- Acceptance Gate：Adopt / Keep Experimental / Reject；
- Case Analysis：改善、持平、退化、未解决；
- External Calls：Embedding、Chat、Reranker 和数据发送；
- Rollback：正式链路如何保持不变或恢复。

## 8. 当前正式决策摘要

| 实验 | 主要结果 | 决策 |
| --- | --- | --- |
| BM25 + Dense + RRF | Recall@5 与 nDCG@5 上升；MRR 下降，Recall@3 不变 | 保留实验，不接正式链路 |
| Cross-Encoder | MRR、nDCG@5、Recall@3 上升；Context Precision 下降且 CPU latency 大增 | 保留实验 |
| Structure-aware Chunking | Recall@1/MRR/nDCG@5 上升；Recall@3、Final Context Recall、Context Precision 下降 | 拒绝替换正式 Chunker |
| EvidenceScoreContextSelector | Context Precision `0.15 → 0.20`；Final Context Recall 保持 `1.0` | 接入正式 `RAGService` |

完整数值、版本和限制见 Stage 3 Completion Reports 以及
[RAG Evaluation Report](RAG_EVALUATION_REPORT.md)；该汇总没有重新运行付费实验。

## 9. 自动化与真实评测的区别

```text
Unit / contract tests
  ≠ current real retrieval benchmark
  ≠ real answer quality
  ≠ browser acceptance
  ≠ production validation
```

自动化测试可证明 Dataset 校验、指标计算、报告唯一写入、费用门槛、Trace 和原始 Index 保护；
只有受控真实运行才能证明当时的 Retrieval/Model 行为。
