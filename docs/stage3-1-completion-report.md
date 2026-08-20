# Stage 3.1 Technical Completion Report

## 1. 本阶段目标

Stage 3.1 只验证一个问题：在保持 Embedding、Chunk、阈值、TopK、Seed、Adjacent
Expansion、Context、Evidence、Citation 和 LLM 链路不变时，真正的 BM25 + Dense
Vector + RRF 双路召回是否优于 Stage 2 Retrieval Baseline。

阶段起点为 `43e54de209e3d46d86f274f7b405f5c50f4b9677`，前置 Tag
`study-material-v2-stage2` 存在。本阶段未加入 Reranker、Query Rewrite、Chunk 调整、
Embedding 变更、Agent 或其他 Stage 3 能力。

## 2. 当前 Baseline

正式问答继续使用以下 Stage 2 Baseline：

```text
Query
  → Dense Vector Top 10
  → relevance >= 0.25
  → 80% vector relevance + 20% keyword coverage
  → Top 3 Seed
  → 同源 ±2 Adjacent Expansion
  → 最多 8 个 Context Chunk
```

关键词覆盖率只重排 Dense 已召回的候选，不能从 Dense Top 10 之外补充 Chunk。
Evidence、Citation、Prompt 和 Generation 位于检索之后，本阶段没有修改。

## 3. 原 Retrieval 问题

Baseline 的主要结构性限制不是“完全没有关键词”，而是关键词没有独立召回通道。
当包含精确术语、缩写或 API 名称的 Gold Chunk 没有进入 Dense Top 10 时，后续关键词
覆盖率无法恢复它。

Stage 2 的历史报告与 Stage 3.1 当前索引指纹不同，因此不能把历史 Dense 排名和当前
BM25 结果拼成正式 A/B。本次实验在同一个 disposable snapshot 上重新执行 Dense，并
让同一次 Dense 结果同时供 Baseline 与 Hybrid 使用。

## 4. BM25 设计

BM25 输入是 Query 与完整 Chroma Chunk Corpus，输出独立的 Chunk 排名、BM25 Score
和 Rank。实现参数为 `k1=1.5`、`b=0.75`，候选数为 Top 10。

分词策略保留英文、数字和技术符号组成的完整词项，并把连续中文切为重叠双字词。
这是一种适合当前 147 个 Chunk、本地单用户实验的确定性策略；没有增加第三方依赖，
也不是为大规模、实时增量语料设计的 Sparse Index。

BM25 直接读取完整语料建立只读内存索引，不调用 Embedding，不修改 Chroma。重复
Chunk 通过稳定 `chunk_id` 去重。

## 5. Dense Retrieval 设计

Dense 路径继续使用当前 Chroma、`text-embedding-v4`、1024 维向量、Top 10 和 0.25
相关度阈值。没有修改 Embedding Model、Dimension、Distance Metric 或 Index 数据。

正式 A/B 对每个案例只生成一次 Query Embedding；该 Dense 结果同时输入旧 Baseline
排序和新 RRF 实验，避免因重复模型调用或不同 Dense 结果引入额外变量。

## 6. RRF 设计

RRF 使用 `k=60`，只融合 Dense Rank 和 BM25 Rank，不直接相加两种不可比较的原始
分数。每条路线的贡献为 `1 / (k + rank)`。

融合使用稳定 `chunk_id` 识别同一 Chunk。两路同时命中的 Chunk 累加排名贡献；只被
一条路线召回的 Chunk 仍保留。最终按 RRF Score 降序排列，相同分数时按 `chunk_id`
稳定排序，因此结果 deterministic、可去重、可复现。

## 7. 数据流变化

```text
Baseline:
Query → Dense Top 10 → threshold → keyword coverage rerank → Top 3

Experiment:
                  ┌→ Dense Top 10 → threshold ┐
Query ────────────┤                            ├→ RRF → Top 3
                  └→ BM25 Top 10 ─────────────┘

共同后续：Top 3 Seed → ±2 Adjacent Expansion → Context Limit 8
```

实验只改变 Candidate Generation 与 Ranking。由于评测结果没有证明 Hybrid 在当前
Top 3 主链路上稳定胜出，正式 `RAGService`、CLI Search 和端到端 RAG Evaluation
仍使用 Baseline；BM25、RRF、双路 Retriever 和专用 Runner 作为可复现实验能力保留。

## 8. Evaluation 结果

正式报告：
`evaluation/results/hybrid-retrieval-20260820T121525454249Z.json`。
`evaluation/results/` 是被 Git 忽略的本地运行历史，报告不会作为源码自动提交。

- Dataset：Stage 2 `evaluation/retrieval_cases.json`，10 个案例，其中 9 个可回答。
- Validation：`local_real_retrieval`、真实 `text-embedding-v4` Query Embedding。
- 调用：10 次 Query Embedding，0 次 ChatModel。
- 费用：调用次数已记录，实际金额未测量。
- Index filesystem SHA-256：`1b0be80c0b6dd2db318d9c47a738f9e02fd6b686202b800ad0bd02391aab92fb`。
- 原始 Index：运行前后 filesystem SHA-256 一致。
- 报告运行基准 commit：`43e54de209e3d46d86f274f7b405f5c50f4b9677`。
- 报告如实记录 `worktree_clean=false`，并记录运行时核心实现文件 SHA-256。

报告生成后只调整了“生产默认仍使用 Baseline”的接线和说明，没有改动本次已执行的
BM25、Dense、RRF 算法或报告数据。由于授权上限 10 次已用完，没有把接线调整包装成
第二次真实评测。

## 9. Baseline vs Hybrid 比较

| Metric | Baseline | Hybrid | Delta |
| --- | ---: | ---: | ---: |
| Recall@1 | 0.3333 | 0.3333 | 0.0000 |
| Recall@3 | 0.8333 | 0.8333 | 0.0000 |
| Recall@5 | 0.8333 | 0.9444 | +0.1111 |
| Recall@10 | 0.9444 | 0.9444 | 0.0000 |
| MRR | 0.6111 | 0.5778 | -0.0333 |
| nDCG@5 | 0.6422 | 0.6561 | +0.0139 |

Baseline 平均 Retrieval 延迟为 149.0 ms，Hybrid 为 149.4 ms，本地差值 0.4 ms；
其中 BM25 平均约 0.4 ms。该数据只是当前机器、当前小语料的本地 wall-clock 结果，
不是并发或生产延迟基线。

总体结果是混合的：Recall@5 与 nDCG@5 上升，但 Recall@3 没有变化，MRR 下降。
由于正式链路只取 Top 3 Seed，当前证据不足以认定 Hybrid 整体优于 Baseline。

## 10. Case 分析

### Ranking 提升

- `wishart_definition`：最佳 Gold Rank 从 6 提升到 3，恢复了 Top 5 Recall，并进入
  正式 Seed 边界。这说明独立 Sparse 路线能够补强精确术语场景。

### Ranking 退化

- `cauchy_properties`：最佳 Gold Rank 从 2 降到 5，形成新的 Ranking Failure。
  BM25 对其他字面重叠 Chunk 的排名贡献抵消了 Dense 的语义优势。
- `dirichlet_definition`：最佳 Gold Rank 从 2 降到 3。仍在 Seed 范围内，但 MRR 和
  nDCG 下降，说明无权重 RRF 并不保证保留 Dense 的前排质量。

### 保持不变

- `rag_chunking_reason`、`chroma_incremental_embedding`、`cauchy_relationships`、
  `multinomial_cross_chunk_moments`、`student_t_suitability`、
  `inverse_wishart_definition` 的最佳 Gold Rank 没有变化。
- `mysql_out_of_scope` 在两种策略下都产生 Unanswerable Handling Failure。BM25 没有
  返回候选，但低相关 Dense 候选仍通过现有阈值；这不是 RRF 能单独解决的问题。

本次没有出现 Gold 只存在于 BM25 Top 10、完全不存在于 Dense Top 10 的新增召回；
主要变化来自两路排名融合后的顺序调整。

## 11. 哪些问题解决

- 建立了真正从完整 Corpus 独立召回的 Sparse 路线，而不是 Dense 后 BM25 重排。
- 建立了不混合原始分数、可稳定去重的 RRF Fusion。
- 建立了同一 Dense 结果驱动 Baseline/Hybrid 的正式 A/B Runner。
- Trace 可以区分 Dense Candidate、BM25 Candidate、Baseline Ranking、RRF Ranking、
  Final Context、Latency 和 Failure Category，并且不记录候选正文。
- 用真实 Query Embedding 证明 BM25 对 `wishart_definition` 有可观察补强价值。

## 12. 哪些问题没有解决

- Hybrid 没有提高 Recall@3，且降低 MRR；不能称为全面优于 Baseline。
- `cauchy_properties` 出现新的 Top 3 退化。
- `mysql_out_of_scope` 的无答案处理仍失败，说明阈值或拒答判定需要独立实验。
- 10 个案例过小，统计波动大，且技术术语类问题占比较高。
- 当前 BM25 是进程内全量构建，不适合直接外推到大语料、频繁更新或多实例部署。
- Retrieval 改善不等于 Answer Correctness、Faithfulness 或 Citation Accuracy 改善。

## 13. 是否值得进入 Stage 3.2 Reranker

现在不应直接进入 Reranker 实现。

RRF 暴露了一个真实问题：Sparse 可以提升某些精确术语案例，但也会把 Dense 原本靠前
的正确结果推后。Reranker 理论上可能处理这种“召回已有、前排顺序不稳”的问题，
但当前只有 10 个案例，直接加入 Cross Encoder 会同时增加外部调用、延迟、费用和维护
复杂度。

更合理的下一步是先扩大并冻结 Retrieval Dataset，增加更多“Dense 强、Sparse 强、
两者冲突、不可回答”案例，然后为 Stage 3.2 定义明确门槛，例如 Recall@3 不下降、
MRR 恢复且新增延迟/费用可接受。达到该前置条件后，再决定是否做受控 Reranker 实验。

## 14. 未验证项

- 没有执行真实 ChatModel、端到端回答、Faithfulness 或 Citation Accuracy 评测。
- 没有执行真实前端/API 人工问答验收。
- 没有执行并发、长时间、大语料或生产环境性能验证。
- 实际 API 金额未测量，只确认并记录 10 次 Query Embedding。
- 本地 JSON 报告未纳入 Git；它是可检查的运行证据，不是生产稳定性证明。
- Commit 与 `study-material-v2-stage3-part1` Tag 尚待显式授权；Tag 创建前不能运行
  `$stage-review`，也不能把阶段描述为已形成 Git checkpoint。
