# 知行（Study Material Assistant）· RAG Evaluation Report

> Stage 6.3 作品化汇总。本文整理 Stage 2 Baseline 与 Stage 3.1～3.4 的已有受控实验，
> 没有重新执行 Embedding、ChatModel、Reranker 或索引写入，新增外部调用与模型费用为 0。

## 1. Executive Summary

知行的 RAG 优化不是按技术清单接入 BM25、Reranker 和新 Chunker，而是用固定 Dataset、
Baseline、逐案例失败、Context 指标、延迟与成本门槛做选择。

最终结果：

- 正式 Retrieval 保持 Stage 2 的 Dense candidate generation + keyword coverage rerank；
- 正式 Chunker 保持 `fixed-character-180-v1`；
- BM25 + Dense + RRF、Cross-Encoder 与 Structure-aware Chunking 保留为隔离实验；
- Stage 3.4 `EvidenceScoreContextSelector` 进入正式 `RAGService`；
- Context Selector 的采用依据是 Context Precision `0.1500 → 0.2000`、Final Context Recall
  保持 `1.0000`、逐案例 Gold 删除为 0、Selector 增量约 `0.1042 ms`、外部调用为 0；
- `mysql_out_of_scope` 的 Unanswerable Handling Failure 仍未解决，不能用排序或 Context
  优化掩盖。

这说明 Evaluation Driven Development 的核心不是“每轮指标都涨”，而是只有满足正式链路
验收门槛的单一改动才被采用。

## 2. 证据范围

| 项目 | 固定边界 |
| --- | --- |
| Retrieval Dataset | `study-material-retrieval-v1` / `evaluation/retrieval_cases.json` |
| 案例数 | 10；其中 9 个可回答、1 个不可回答 |
| Stage 2 Baseline commit | `0d9f2fe97a9192db3fe653ea80e37042c7426e30` |
| Stage 3.1 运行基准 | `43e54de209e3d46d86f274f7b405f5c50f4b9677` |
| Stage 3.2 Dataset hash | `47ed2974070385d06e6ab9c8d57660e9a1f3c93f9536ed09c1993524585b3e8f` |
| Stage 3.3 Dataset hash | 与 Stage 3.2 相同 |
| 报告位置 | `evaluation/results/`，本地运行历史，被 Git 忽略 |
| 本文来源 | Stage 3 Baseline 与四份 Completion Report |

历史数字只证明当时 commit、Dataset、Index/Trace 与当前机器上的结果。本文不是对当前索引的
重新 Benchmark，也不证明 Answer Correctness、Faithfulness、Citation Support 或生产稳定性。

## 3. Baseline

### 3.1 正式 Pipeline

```text
Question
  → one real Query Embedding
  → Chroma Dense Top 10
  → score >= 0.25
  → 0.8 vector + 0.2 keyword coverage
  → Top 3 seeds
  → same-source adjacent expansion (±2, context limit 8)
  → Final Context
```

这里的“Hybrid”只是对 Dense Top 10 做关键词覆盖率重排，不是 BM25 独立召回。

### 3.2 Stage 2 指标

| Metric | Baseline |
| --- | ---: |
| Raw Recall@1 / @3 / @5 / @10 | 0.3889 / 0.6111 / 0.8333 / 0.9444 |
| Ranked Recall@1 / @3 / @5 / @10 | 0.3333 / 0.8333 / 0.8333 / 0.9444 |
| MRR | 0.6111 |
| nDCG@5 | 0.6422 |
| Final Context Recall | 1.0000 |
| Context Precision | 0.1500 |
| Average local Retrieval latency | 119.5 ms |
| Query Embedding / Chat | 10 / 0 |

### 3.3 可观察问题

- `wishart_definition`：Gold 排名靠后，暴露精确术语召回/排序问题；
- `mysql_out_of_scope`：不可回答案例仍返回低相关 Context，属于 Unanswerable Handling；
- Final Context Recall 已为 `1.0`，但 Context Precision 只有 `0.15`，平均 Context
  为 `7.3` 块，说明相邻扩展后噪声较多；
- Dataset 只有 10 个案例，不能用单个平均数做普遍结论。

因此 Stage 3 分别隔离 Candidate Generation、Ranking、Chunking 和 Context Construction，
没有一次同时调整多层。

## 4. Stage 3.1 · BM25 + Dense + RRF

### 4.1 假设

Dense 对语义表达较强，但精确术语可能不足；从完整 Corpus 独立 BM25 召回，再用 RRF 融合，
可能把 Dense Top 10 之外或靠后的 Gold 提前。

### 4.2 控制

- 同一 10-case Dataset、Index 与 Gold；
- 每个 Query 的同一次 Dense 结果同时供 Baseline 与 Hybrid；
- 10 次真实 Query Embedding，BM25/RRF 本地，Chat 为 0；
- 原始 Index 运行前后 SHA-256 一致。

### 4.3 结果

| Metric | Baseline | BM25 + Dense + RRF | Delta |
| --- | ---: | ---: | ---: |
| Recall@1 | 0.3333 | 0.3333 | 0.0000 |
| Recall@3 | 0.8333 | 0.8333 | 0.0000 |
| Recall@5 | 0.8333 | 0.9444 | +0.1111 |
| Recall@10 | 0.9444 | 0.9444 | 0.0000 |
| MRR | 0.6111 | 0.5778 | -0.0333 |
| nDCG@5 | 0.6422 | 0.6561 | +0.0139 |
| Average local latency | 149.0 ms | 149.4 ms | +0.4 ms |

### 4.4 Case 分析

改善：

- `wishart_definition`：最佳 Gold Rank `6 → 3`，进入 Top 3 Seed。

退化：

- `cauchy_properties`：`2 → 5`，新增 Ranking Failure；
- `dirichlet_definition`：`2 → 3`，仍可用但 MRR/nDCG 下降。

未解决：

- `mysql_out_of_scope` 仍返回低相关 Dense 候选；
- 没有出现“Gold 只在 BM25 Top 10、完全不在 Dense Top 10”的新增召回。

### 4.5 决策

Keep Experimental。Recall@5 的提升不能抵消正式 Top 3 不变和 MRR 回退；无权重 RRF
不能保证保留 Dense 前排质量。正式 `RAGService` 不接入 `HybridRRFRetriever`。

## 5. Stage 3.2 · Cross-Encoder Reranker

### 5.1 假设

Stage 3.1 已经召回 Gold，但融合顺序不稳。Cross-Encoder 对 Query/Chunk pair 评分，可能改善
前排 Ranking；它不能新增 Candidate，也不负责无答案判断。

### 5.2 控制与成本

- A：Stage 2 Baseline；
- B：BM25 + Dense + RRF；
- C：B 的 Candidate Pool + `BAAI/bge-reranker-base`；
- 10 次真实 Query Embedding，130 对本地 Cross-Encoder scoring，Chat 为 0；
- 总 wall-clock `37.185 s`；模型加载约 `4.348 s`；模型目录约 `1.27 GB`；
- API 金额未测量。

### 5.3 Ranking

| Metric | A Baseline | B Hybrid | C Reranker | C - B |
| --- | ---: | ---: | ---: | ---: |
| Recall@1 | 0.3333 | 0.3333 | 0.8333 | +0.5000 |
| Recall@3 | 0.8333 | 0.8333 | 0.9444 | +0.1111 |
| Recall@5 | 0.8333 | 0.9444 | 0.9444 | 0.0000 |
| Recall@10 | 0.9444 | 0.9444 | 0.9444 | 0.0000 |
| MRR | 0.6111 | 0.5778 | 0.9259 | +0.3481 |
| nDCG@5 | 0.6422 | 0.6561 | 0.9015 | +0.2454 |

### 5.4 Context 与延迟

| Metric | A Baseline | B Hybrid | C Reranker | C - B |
| --- | ---: | ---: | ---: | ---: |
| Context Precision | 0.1500 | 0.1577 | 0.1458 | -0.0119 |
| Final Context Recall | 1.0000 | 0.8889 | 1.0000 | +0.1111 |
| Average Context Size | 7.3 | 6.9 | 7.3 | +0.4 |
| Average local latency | 212.8 ms | 211.5 ms | 3073.9 ms | +2862.4 ms |

C 把 B 的 Ranking Failure 从 1 个降为 0，多个 Gold 进入 Rank 1；但 Seed 改变触发不同的
Adjacent Expansion，排序收益没有转化为 Context Precision 收益。CPU 延迟约为 B 的
`14.5 倍`。

### 5.5 决策

Keep Experimental。Reranker 证明能解决“候选已有、顺序不稳”，但交互延迟过高、Context
Precision 下降、无答案失败仍在，且缺少真实 Answer/Citation、并发和大语料证据。

## 6. Stage 3.3 · Structure-aware Chunking

### 6.1 假设

更完整的结构块可能减少固定字符跨边界切断，让定义和跨段内容更容易前排命中。

### 6.2 控制与成本

- A：正式 `fixed-character-180-v1`；
- B：`structure-aware-block-600-overlap-0-v1`；
- Retrieval、Embedding、Top-K、阈值、Adjacent 与 Context Limit 固定；
- 两个独立临时索引：A 147 chunks、B 55 chunks；
- Index Embedding 202 条、逻辑 21 batches；A/B Query Embedding 共 20 次；Chat 为 0；
- wall-clock 约 `7.809 s`；原始生产 Index 指纹不变，临时索引运行后清理。

### 6.3 结果

| Metric | A Current | B Structure-aware | Delta |
| --- | ---: | ---: | ---: |
| Recall@1 | 0.3333 | 0.5556 | +0.2222 |
| Recall@3 | 0.8333 | 0.6667 | -0.1667 |
| Recall@5 | 0.8333 | 0.8889 | +0.0556 |
| Recall@10 | 0.9444 | 0.8889 | -0.0556 |
| MRR | 0.6111 | 0.6426 | +0.0315 |
| nDCG@5 | 0.6422 | 0.7019 | +0.0598 |
| Context Precision | 0.1500 | 0.1202 | -0.0298 |
| Final Context Recall | 1.0000 | 0.7778 | -0.2222 |
| Average Context Chunks | 7.3 | 5.8 | -1.5 |
| Average local latency | 123.9 ms | 93.3 ms | -30.6 ms |

主要提升：`multinomial_cross_chunk_moments`、`student_t_suitability`、
`wishart_definition`。

主要回归：`cauchy_properties`、`dirichlet_definition`、
`cauchy_relationships`、`inverse_wishart_definition`。

### 6.4 决策

Reject for Production，保留实验实现。更少 Chunk 与部分前排提升不能抵消正式 Top 3 Recall、
Final Context Recall 和 Context Precision 回归；不重建或迁移正式索引。

## 7. Stage 3.4 · Context Optimization

### 7.1 问题定位

Baseline 的 Final Context Recall 已为 `1.0`，但 Precision `0.15`、平均 `7.3` 块。
这次只改变 Context Construction，不重新检索、不改 Chunk、Ranking、Prompt 或 Model。

### 7.2 方案

`EvidenceScoreContextSelector`：

- 所有 Top 3 Seed 必须保留；
- 每个 Seed 最多选择一个已经存在的 Adjacent Evidence；
- 选择依据只使用已有检索分数、关键词覆盖和邻接关系；
- 不调用新模型，不新建网络/持久化状态；
- 保留 `BaselineContextSelector` 作为回退与测试对照。

### 7.3 证据层级

- `local_historical_retrieval_trace_replay`；
- 同一个历史 Stage 2 Trace；当前 Index 指纹与历史不同；
- 全部 Final Context Chunk ID 能在当前 disposable snapshot 解析，字符大小一致；
- Query Embedding / Chat / Reranker：`0 / 0 / 0`；
- 这是可信的 Selector replay，不是当前真实 Retrieval 或 Answer 验收。

### 7.4 结果

| Metric | A Baseline | B Selector | Delta |
| --- | ---: | ---: | ---: |
| Context Precision | 0.1500 | 0.2000 | +0.0500 |
| Final Context Recall | 1.0000 | 1.0000 | 0.0000 |
| Average Context Chunks | 7.3 | 5.2 | -2.1 |
| Average Context Characters | 1222.2 | 848.7 | -373.5 |
| Duplicate Ratio | 0.0000 | 0.0000 | 0.0000 |
| Unanswerable Empty Context Rate | 0.0000 | 0.0000 | 0.0000 |
| Selector-only latency delta | — | — | +0.1042 ms |

逐案例移除 21 个非 Gold Chunk，Gold 删除数为 0。平均 Chunk 数下降约 `28.8%`、字符数
下降约 `30.6%`；这些减少本身不是成功依据，成功依据是 Precision 上升且 Recall 保持。

### 7.5 仍未解决

- `wishart_definition` 的原 Ranking Failure 没有恢复；Selector 只保留所需 Adjacent Gold；
- `mysql_out_of_scope` 的 Context 仍非空；
- 没有真实 ChatModel，未验证 Answer Correctness、Faithfulness、Citation Support 或真实 token cost；
- 正式 Token Budget 未启用。

### 7.6 决策

Adopt。该单变量改动同时满足 Precision、Recall、逐案例 Gold、延迟和零新增模型成本门槛，
因此接入正式 `RAGService`。

## 8. 决策矩阵

| 层 | 候选 | Primary Gain | Guardrail / Cost | 结论 |
| --- | --- | --- | --- | --- |
| Candidate Generation | BM25 + Dense + RRF | Recall@5 +0.1111 | Recall@3 不变；MRR -0.0333 | Experimental |
| Ranking | Cross-Encoder | MRR +0.3481 vs Hybrid | Context Precision -0.0119；+2862.4 ms | Experimental |
| Chunking | Structure-aware 600/0 | Recall@1 +0.2222 | Recall@3 -0.1667；Final Context Recall -0.2222 | Reject production |
| Context | EvidenceScore selector | Context Precision +0.0500 | Final Context Recall 不降；+0.1042 ms；0 external calls | Adopt |

## 9. 为什么最终不是“最佳指标拼装”

不能把四轮中各自最好的数字拼成一个不存在的 Pipeline：

- Stage 3.2 的 Reranker 指标来自 Hybrid Candidate Pool；
- Stage 3.3 重建了两个临时索引，Chunk 身份与粒度不同；
- Stage 3.4 复用 Stage 2 历史 Trace，没有重新执行 Retrieval；
- 各轮延迟起点不同，只能在同一轮 A/B 内比较；
- Context 与 Answer 是 Retrieval 下游，Ranking 提升不保证最终上下文或回答提升。

最终正式选择必须是一条真实接线并通过该层的 Acceptance Gate，而不是“拿每列最大值”。

## 10. Evaluation Driven Development 的工程闭环

```text
Observed case failure
  → locate pipeline layer
  → freeze Dataset / Gold / Baseline
  → change one variable
  → record calls / latency / index fingerprint
  → aggregate metrics + case-level regression
  → Adopt / Keep Experimental / Reject
  → wire only accepted change
  → regression tests + completion report
```

这套流程带来的价值：

1. 避免把 Prompt、Retrieval、Context、Model 问题混在一起；
2. 避免自动测试、一次输出或平均指标替代真实受控实验；
3. 让“拒绝接入技术”成为有证据的工程决策；
4. 保留实验模块，正式主链仍可回退、维护和解释。

## 11. 当前限制与下一次重评条件

限制：

- Dataset 只有 10 个案例，主题分布有限；
- Stage 3.4 是历史 Trace replay，不是当前索引 Retrieval；
- 缺少稳定 Answer Correctness、Faithfulness、Claim-level Citation Support/Accuracy；
- 没有并发、大语料、长时间、公开部署和生产成本基线；
- 实际 API 金额未测量；
- 不可回答阈值问题仍存在。

重新评估 BM25、Reranker、Chunking 或 Selector 前，应先扩大并冻结 Dataset，补充：

- Dense 强、Sparse 强、两者冲突的术语与语义案例；
- 长短 Query、Multi-Gold、跨段与不同资料类型；
- 更多不可回答和近似相关负例；
- Answer/Faithfulness/Citation 的独立人工或模型辅助标注协议；
- 明确的交互 latency、外部调用和成本 Guardrail。

没有这些前置条件，不应为了作品展示把实验能力接入正式 Pipeline。

## 12. 可复核来源

- [Evaluation Guide](EVALUATION.md)
- [Stage 3.1 Baseline](stage3-1-baseline.md)
- [Stage 3.1 Completion Report](stage3-1-completion-report.md)
- [Stage 3.2 Completion Report](stage3-2-completion-report.md)
- [Stage 3.3 Completion Report](stage3-3-completion-report.md)
- [Stage 3.4 Completion Report](stage3-4-completion-report.md)
- [Technical Decisions](DECISIONS.md)
- [Current Project Status](PROJECT_STATUS.md)
