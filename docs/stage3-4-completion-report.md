# Stage 3.4 Technical Completion Report

## 1. 实验目标、起点与边界

Stage 3.4 只验证一个问题：在 Stage 2 Retrieval 已经得到高 Recall、但最终 Context
Precision 只有 `0.15` 时，能否只从已有 Retrieval Evidence 中移除噪声，同时保持回答所需
Gold Evidence 完整。

实验起点是 commit `b4757b29264aae54a06994207ac4e0f7be476971`，对应 Tag
`study-material-v2-stage3-part3`。本阶段没有修改 Dense Candidate Generation、0.25
阈值、80/20 Ranking、Top 3 Seed、Embedding、BM25、RRF、Reranker、Chunking、Query
Rewrite 或 LLM；没有调用付费模型，也没有实现新的拒答系统。

## 2. 当前 Context Baseline

```text
Query
→ Dense Top 10
→ threshold 0.25
→ 80% vector score + 20% keyword coverage
→ Top 3 Seed
→ same-source adjacent window 2
→ Context Limit 8
→ Evidence Map → Prompt → LLM
```

A 组完整保留上述 Retriever 输出。Stage 2 历史真实 Retrieval Report 的 10 个案例中，
9 个可回答案例 Final Context Recall 为 `1.0`，但平均 Context Precision 为 `0.15`；平均
Context 有 `7.3` 块。问题位于 Context Construction，而不是本阶段要重新优化 Retrieval。

## 3. 优化方案与设计理由

B 组新增抽象 `ContextSelector.select(query, evidences)`。当前实现为
`EvidenceScoreContextSelector`：

1. 输入只能是 Retriever 已经返回的 Context Evidence，不访问 Vector Store、不重新召回。
2. 保留全部 Seed，避免直接丢掉 Top 3 Ranking 的主要结果。
3. 每个 Seed 最多选择一个同源相邻 Evidence，以 query keyword coverage 作为分数。
4. 分数相同时按更小邻接距离、原 Context 顺序稳定打破并列。
5. 以 Chunk ID 去重；显式 Seed/Adjacent provenance 处理不足 3 个 Seed 的边界。

该规则不是新的 Retrieval Ranker。它不改变候选集合或 Seed 排名，只限制 Adjacent
Expansion 在 LLM Context 中的份额。相比直接增加 LLM Selector、Reranker 或 Agent，
它是本地、确定性、零模型成本且可逐案例解释的最小方案。

接口支持在注入可靠 Token Counter 时启用 Token Budget，但 Stage 3.4 正式配置没有启用。
当前 ChatModel 没有匹配且已验证的本地 tokenizer，不能把字符估算包装成真实 Token 数。

## 4. 数据流变化

```text
A Baseline:
Retrieved Seeds → Adjacent Expansion → all Context → Evidence IDs → Prompt

B Optimized:
Retrieved Seeds → Adjacent Expansion → Context Selector
                → selected Context → Evidence IDs → Prompt
```

`HybridRetriever` 只在返回的 Document metadata 中增加请求内 `_context_role` 与 Seed Rank
provenance，不改变检索分数、排序、窗口或数量。`RAGService` 将 Selector 注入复用的 LCEL
Runnable；选择发生在 Evidence ID 分配之前，因此被移除块不会得到 `S#`，`sources` 与
Citation Map 也只包含真正发送给 LLM 的 Context。

## 5. Evaluation 方法与证据层级

最终报告：
`evaluation/results/context-optimization-20260821T054111390893Z.json`。该目录被 Git
忽略，报告是本地运行证据，不进入源码提交。

- Dataset：未修改的 `evaluation/retrieval_cases.json`，10 个案例，其中 9 个可回答。
- Baseline：Stage 2 已完成的 `local_real_retrieval` Report。
- 本轮 Validation：`local_historical_retrieval_trace_replay`。
- 本轮没有重新执行 Retrieval；Raw/Ranked Recall、MRR、nDCG 复用同一 Trace，按设计不变。
- 当前 Index filesystem 指纹与历史 Baseline 不同，因此不能声称重新验证了当前 Retrieval。
- 但报告中的全部 Final Context Chunk ID 均能在当前 disposable snapshot 中解析，逐案例
  `char_size` 也与历史 Trace 一致，证明本轮选择使用的是同一批 Context 内容。
- Query Embedding / ChatModel / Reranker 调用：`0 / 0 / 0`。
- Token：未测量；字符数为实际统计。

因此这是一轮可信的 Context Selector 回放实验，不是新的真实 Retrieval、真实答案或生产
验收。

## 6. Evaluation 对比

| Metric | A Baseline | B Optimized | B - A |
| --- | ---: | ---: | ---: |
| Context Precision | 0.1500 | 0.2000 | +0.0500 |
| Final Context Recall | 1.0000 | 1.0000 | 0.0000 |
| Average Context Chunk Count | 7.3 | 5.2 | -2.1 |
| Average Context Character Count | 1222.2 | 848.7 | -373.5 |
| Average Duplicate Ratio | 0.0000 | 0.0000 | 0.0000 |
| Unanswerable Empty Context Rate | 0.0000 | 0.0000 | 0.0000 |

Context Precision 相对提高约三分之一，平均 Chunk 减少约 `28.8%`，平均字符减少约
`30.6%`，同时 Final Context Recall 没有下降。减少文本本身不是成功依据；成功依据是
Precision 上升与 Recall 保持同时成立。

Retrieval Recall@1/3/5/10、MRR 和 nDCG 没有重新计算，也没有发生代码路径变化；报告将
这些指标标为复用 Baseline 且 delta 为 `0`，避免把“未重新 Retrieval”误写成新的真实
Retrieval 改善。

## 7. Latency 与成本

Selector-only 本地 wall-clock 每个案例重复 100 次后，B 相对 A 平均增加约
`0.1042 ms`。该数字只表示当前 Windows 机器、小 Context 和本地规则选择的微小增量，
不是端到端 API、并发或生产延迟基线。

本轮没有外部模型费用。没有增加依赖、模型文件、网络请求或持久化状态。

## 8. Failure Analysis

- `rag_chunking_reason`：原 Context 只有 3 个 Seed，B 不做无意义压缩，Precision 保持
  `0.3333`。
- `chroma_incremental_embedding`：`6 → 5` 块，Precision `0.1667 → 0.2000`。
- `cauchy_properties`、`dirichlet_definition`、`student_t_suitability`、
  `inverse_wishart_definition`：均从 8 块降到 5 块，Gold 全部保留，Precision
  `0.1250 → 0.2000`。
- `cauchy_relationships`：`8 → 6`，Precision `0.1250 → 0.1667`。
- `multinomial_cross_chunk_moments`：Multi-Gold 全部保留，`8 → 6`，Precision
  `0.2500 → 0.3333`。
- `wishart_definition`：Gold 是距离 2 的 Adjacent Evidence；B 仍通过 keyword coverage
  保留它，`8 → 6`。原有 `ranking_failure` 没有被伪装成恢复，因为 Gold 仍不在 Seed。
- `mysql_out_of_scope`：`8 → 6`，但 Context 仍非空，Unanswerable Handling Failure
  保持。Selector 不是新的拒答系统，本阶段不修改阈值或无答案判定。

逐案例共有 21 个非 Gold Context Chunk 被移除，Gold 删除数为 0。报告只保存 Chunk ID、
来源、位置、role 与 Gold 标记，不保存 Chunk 正文。

## 9. 是否进入正式 Pipeline

结论：进入正式 `RAGService` Pipeline。

依据是当前受控数据上同时满足：Context Precision 提升、Final Context Recall 保持、
逐案例没有移除 Gold、增量延迟很小、没有新模型成本。正式接入仍保留
`BaselineContextSelector`，使 A 组行为可测试、可替换，不删除历史 Baseline。

这不是对所有 Query 的永久最优结论。若扩大 Dataset 后出现 Recall 回归，应优先回退
Selector 或调整单一选择规则，而不是同时修改 Retrieval、Chunking 与 Prompt。

## 10. 验证与 Review

本阶段新增测试覆盖：空 Evidence、单 Evidence、多 Evidence、重复 Evidence、不足 3 个
Seed 的显式 provenance、可靠 Token Counter 下的超 Budget、无效 Token Counter、Context
Precision、Final Context Recall、Context Size、Retrieval 指标不变、历史 Trace 与当前
Chunk 不一致时失败关闭、报告唯一写入、LCEL 选择顺序、错误脱敏和正式服务接入。

完整自动化回归本次实际运行 `343/343` 通过；定向 Context/RAG/Retrieval 回归实际运行
`85/85` 通过。评测报告生成成功，生产 Index 只通过 disposable snapshot 读取，原始
filesystem 指纹保持不变。

Review 中修正了两项问题：

1. 最初模拟的“只保留距离 1 邻块”会删除 `wishart_definition` Gold，使 Final Context
   Recall 降至约 `0.8889`，因此在实现前淘汰。
2. 第一版评测归因曾把 `wishart_definition` 原有 Ranking Failure 写成恢复；由于
   Retrieval 未变，该结论不成立。报告已改为保留原 Failure Category，并增加回归断言。

## 11. 风险与未验证项

- Dataset 只有 10 个案例，结论不能外推到所有学习资料和问题。
- 本轮复用历史真实 Retrieval Trace，没有重新调用当前 Embedding 或验证当前 Retrieval
  指标；当前与历史 Index filesystem 指纹不同。
- 没有真实 ChatModel，因此未验证 Answer Correctness、Faithfulness、Citation Coverage、
  Citation Support 或真实 Token Cost。
- 没有 API/Web 人工验收、并发、长时间运行、大语料或生产部署验证。
- `mysql_out_of_scope` 仍失败；它需要单独的无答案判定实验，不能由 Context Selector
  顺带解决。
- Token Budget 只有接口与自动化边界，正式 Pipeline 未启用，也没有可靠 Token 统计。

本阶段完成后停止，不进入 Tutor Agent、LangGraph、Multi-Agent、MCP 或 Production
Engineering。
