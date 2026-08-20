# Stage 3.2 Technical Completion Report

## 1. 实验目标与边界

Stage 3.2 只验证一个问题：在保持 Stage 3.1 的 Dense、BM25、RRF、Chunk、Evaluation
Dataset 和后续 Context Construction 不变时，真实 Cross-Encoder 是否能改善 RRF
候选池的前排排序质量。

实验起点是 commit `2d526c5efdf7d9bb92fef4c53fdadc0856d4cbd9`，对应 Tag
`study-material-v2-stage3-part1`。本次没有修改正式 Retrieval Baseline，没有把 Reranker
接入 `RAGService`，没有开始 Query Rewrite 或 Stage 3.3，也没有 commit 或 tag。

## 2. 三条受控 Pipeline

```text
A  Stage 2 Baseline:
Query → Dense Top 10 → threshold 0.25 → keyword coverage rerank

B  Stage 3.1 Hybrid:
Query → Dense Top 10 ┐
                     ├→ RRF(k=60)
Query → BM25 Top 10 ┘

C  Stage 3.2 Experiment:
Query → 与 B 完全相同的 RRF Candidate Pool → Cross-Encoder rerank

共同后续：Top 3 Seed → 同源 ±2 Adjacent Expansion → Context Limit 8
```

每个案例只生成一次 Dense Query Embedding，同一结果同时供 A、B、C 使用。C 只重排
B 已产生的最多 20 个候选，不读取完整 Corpus 重新召回，因此它不能修复 Candidate
Pool 之外的漏召回。

## 3. Reranker 接口与实现

实验新增独立 `Reranker` 抽象：输入 `query + candidate chunks`，输出稳定的
`reranker_score`、`reranker_rank`、`original_rank` 和 `rank_change`。相同分数时保留原
RRF 顺序，避免把不稳定排序误认为模型收益。

实现包含两种 Adapter：

- `DeterministicMockReranker`：只用于接口、排序、Trace 和评测聚合自动化测试。
- `FlagEmbeddingCrossEncoderReranker`：加载固定本地模型并执行真实 Pair Scoring；禁止
  隐式联网，`trust_remote_code=false`。

模型推理或分数校验失败时会抛出明确的 `RerankerError`，不会静默回退或伪造结果。

## 4. 模型选择与可复现配置

- 模型：`BAAI/bge-reranker-base`
- 固定 revision：`2cfc18c9415c912f9d8155881c133215df768a70`
- `model.safetensors` SHA-256：
  `ced967c45fd1902eb92716c9ceeca7c95a936770ea9db611f5a841b926e33fbd`
- 本地目录：`data/models/bge-reranker-base`，已由 `.gitignore` 排除
- 推理设备：CPU，FP32；本机未检测到 CUDA
- `max_length=512`，`batch_size=16`
- 依赖：`FlagEmbedding==1.4.0`、`transformers==4.57.6`、
  `sentence-transformers==5.1.2`

首次解析到的 Transformers 5.x 与 FlagEmbedding 1.4.0 不兼容，真实首例在 Tokenizer
处失败；把依赖固定到兼容的 Transformers 4.x / Sentence Transformers 5.x 后，离线
单 Pair Smoke Test 和完整实验均成功。失败尝试已产生 1 次 Query Embedding；成功报告
另记录 10 次，因此本次实验会话合计执行了 11 次真实 Query Embedding。

## 5. Evaluation 证据

正式运行报告：
`evaluation/results/reranker-retrieval-20260820T151139615621Z-73804a6b.json`。
该目录被 Git 忽略，报告是本地运行证据，不会自动进入源码提交。

- Dataset：未修改的 `evaluation/retrieval_cases.json`，10 个案例，其中 9 个可回答
- Dataset SHA-256：
  `47ed2974070385d06e6ab9c8d57660e9a1f3c93f9536ed09c1993524585b3e8f`
- Validation：`local_real_cross_encoder`
- Query Embedding：成功运行 10 次，模型 `text-embedding-v4`
- 本地 Cross-Encoder Pair Scoring：130 对
- ChatModel：0 次
- 实际 API 金额：未测量
- 成功运行总 wall-clock：37.185 秒
- 报告记录了 Dataset、Index、起点 commit、实现文件哈希、模型 revision 与模型哈希

## 6. Ranking 指标

以下指标按 9 个可回答案例平均：

| Metric | A Baseline | B Hybrid + RRF | C Hybrid + Reranker | C - B |
| --- | ---: | ---: | ---: | ---: |
| Recall@1 | 0.3333 | 0.3333 | 0.8333 | +0.5000 |
| Recall@3 | 0.8333 | 0.8333 | 0.9444 | +0.1111 |
| Recall@5 | 0.8333 | 0.9444 | 0.9444 | 0.0000 |
| Recall@10 | 0.9444 | 0.9444 | 0.9444 | 0.0000 |
| MRR | 0.6111 | 0.5778 | 0.9259 | +0.3481 |
| nDCG@5 | 0.6422 | 0.6561 | 0.9015 | +0.2454 |

C 相对 A 的 MRR 为 `+0.3148`，nDCG@5 为 `+0.2593`。Recall@10 没有变化，符合
“只重排已有候选、不新增召回”的实验设计。Reranker 把 B 的 1 个 Ranking Failure
降为 0，但没有解决唯一的 Unanswerable Handling Failure。

## 7. Context 指标

| Metric | A Baseline | B Hybrid + RRF | C Hybrid + Reranker | C - B |
| --- | ---: | ---: | ---: | ---: |
| Context Precision | 0.1500 | 0.1577 | 0.1458 | -0.0119 |
| Final Context Recall | 1.0000 | 0.8889 | 1.0000 | +0.1111 |
| Context Size | 7.3 | 6.9 | 7.3 | +0.4 |
| Duplicate Ratio | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

排序收益没有转化为 Context Precision 收益。原因不是 Reranker 排名退化，而是 Top 3
Seed 变化后触发了不同的 Adjacent Expansion；例如 `chroma_incremental_embedding`
的 Gold 仍排第 1，但 C 扩展到 8 个 Context Chunk，B 只有 3 个，导致单例 Context
Precision 从 `0.3333` 降到 `0.1250`。因此不能把 Ranking 指标提升直接等价为最终
上下文更精确。

## 8. 逐案例变化

### 提升（5 个）

- `cauchy_properties`：B 最佳 Gold Rank `5 → 1`，并把 Final Context Recall
  `0 → 1`；修复 Stage 3.1 的排序退化。
- `dirichlet_definition`：`3 → 1`。
- `multinomial_cross_chunk_moments`：`2 → 1`；但另一个 Gold 仍不在 Candidate Pool，
  Recall@10 保持 `0.5`。
- `student_t_suitability`：`2 → 1`。
- `wishart_definition`：`3 → 1`，相对 A 则为 `6 → 1`。

### 持平（4 个）

- `rag_chunking_reason`、`chroma_incremental_embedding`、`cauchy_relationships`：均
  保持第 1。
- `inverse_wishart_definition`：保持第 3，是唯一没有进入 Top 1 的可回答案例。

### 未解决（1 个）

- `mysql_out_of_scope`：仍返回非空候选与 Context。Cross-Encoder 负责候选相对排序，
  不是无答案判定器；本阶段又明确不修改阈值和拒答逻辑，因此该失败符合边界。

最佳 Gold Rank 没有案例退化，但当前 Dataset 很小，不能外推为普遍稳定性结论。

## 9. 延迟、存储与成本取舍

| Pipeline | 平均本地 Retrieval 延迟 |
| --- | ---: |
| A Baseline | 212.8 ms |
| B Hybrid + RRF | 211.5 ms |
| C Hybrid + Reranker | 3073.9 ms |

C 比 B 平均增加 `2862.4 ms`，总延迟约为 B 的 `14.5 倍`。这是当前 Windows 机器、
CPU、10 个案例和 10–16 个候选的顺序 wall-clock 结果，不是并发或生产基线。成功运行
中的模型加载耗时约 4.348 秒；模型目录约 1.27 GB。真实 Query Embedding 的金额没有
测量，本地 Cross-Encoder 推理没有外部推理费用，但会占用本机 CPU、内存和磁盘。

## 10. 工程判断

本实验已经证明一件具体的事：在当前冻结 Dataset 和 Candidate Pool 上，Cross-Encoder
明显改善了 RRF 的前排排序，尤其修复了 Stage 3.1 暴露的“召回已有、顺序不稳”。

但当前证据不支持直接进入正式 Pipeline：

1. 平均增加约 2.86 秒 CPU 延迟，对交互式问答过重。
2. Context Precision 没有提升，说明后续 Seed / Adjacent Expansion 仍会稀释排序收益。
3. 无答案处理仍失败，Reranker 不能替代拒答判定。
4. Dataset 只有 10 个案例，缺少更广泛主题、长短 Query 和候选冲突覆盖。
5. 没有真实 LLM Answer、Faithfulness、Citation Accuracy、并发或长时间验证。

因此建议保留 Stage 3.2 为隔离、可复现的实验能力，正式 Retrieval Baseline 继续不变。
在水哥确认结果前，不接入生产主链路，不开始 Stage 3.3。

## 11. 验证与未验证项

已执行：Reranker 单元测试、三路评测聚合测试、BM25/RRF/Stage 2 Retrieval 回归、离线
真实模型 Smoke Test、完整 10 案例真实实验、依赖一致性检查、Python 编译检查和 Git
差异审查。

未验证：真实 ChatModel 与回答质量、Citation Accuracy、前端/API 人工验收、GPU、并发、
大语料、生产部署、实际 API 金额。本报告不把自动化测试或一次小型真实实验包装成
生产稳定性证明。
