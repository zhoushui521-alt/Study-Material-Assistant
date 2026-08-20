# Stage 3.1 Retrieval Baseline

## 边界

- Stage 3.1 起点：`43e54de209e3d46d86f274f7b405f5c50f4b9677`
- 前置标签：`study-material-v2-stage2` → `fda1ac0`
- 当前阶段只比较旧 Retrieval Baseline 与 `BM25 + Dense Vector + RRF`。
- 不修改 Chunk、Embedding、Metadata、Evidence、Citation、Context、LLM、阈值、Seed 数量、相邻扩展或 Context 上限。

仓库中不存在独立的 Stage 2 Completion Report、Stage 2 Learning Review 或
`docs/stage3-plan.md`。Stage 2 的当前事实来源是 Git 标签、README、实现、测试、
`evaluation/retrieval_cases.json` 和已生成的 Retrieval Report。

## 当前生产 Baseline

```text
Query
  ↓
Dense Vector Search（Top 10）
  ↓
vector_score >= 0.25
  ↓
0.8 × vector_score + 0.2 × keyword_coverage
  ↓
Top 3 Seed
  ↓
同源 ±2 Adjacent Expansion
  ↓
最多 8 个 Final Context Chunk
  ↓
Evidence Map → Prompt → LLM → Citation 校验
```

关键词覆盖率只重排 Dense 已召回的候选，不能找回 Dense Top 10 之外的 Chunk，
因此它不是独立的 Sparse Retrieval。

## Stage 2 真实 Baseline 证据

`evaluation/results/retrieval-evaluation-20260819T104043883459Z-13d190ed.json`
记录了一次完成的 `local_real_retrieval`：

- 代码 commit：`0d9f2fe97a9192db3fe653ea80e37042c7426e30`
- Dataset：`study-material-retrieval-v1`，10 个案例
- Query Embedding：真实 `text-embedding-v4`，10 次
- Chat/LLM：0 次
- Raw Recall@1/3/5/10：`0.3889 / 0.6111 / 0.8333 / 0.9444`
- Baseline Ranked Recall@1/3/5/10：`0.3333 / 0.8333 / 0.8333 / 0.9444`
- Baseline MRR：`0.6111`
- Baseline nDCG@5：`0.6422`
- Final Context Recall：`1.0`
- Context Precision：`0.15`
- 失败：1 个 `ranking_failure`、1 个 `unanswerable_handling_failure`
- 平均本地 Retrieval 延迟：`119.5 ms`

这些结果只能说明该 commit、该索引快照和该 Dataset 上的 Retrieval 表现；不证明
Answer Correctness、Faithfulness、Citation Support、真实 LLM 效果或生产稳定性。

## 文档与仓库事实差异

Stage 3.1 开始时，README 仍声明“没有可恢复、已完成的真实 Retrieval Baseline
Report”，该说法已被上述完成报告推翻；本阶段已同步修正文档。

旧报告记录的原始 Index filesystem 指纹是 `cb6bb49a...65094`，Stage 3.1 开始时
当前 Index 指纹是 `1b0be80c...92fb`。因此不能把旧 Dense 排名与当前 BM25 Corpus
拼接成正式 A/B 结论；正式实验必须在同一个 disposable snapshot 上重新执行 Dense
与 BM25，并让同一次 Dense 结果同时供 Baseline 和 Hybrid 使用。

## Stage 3.1 控制变量

```text
Baseline:
Dense Top 10 → 0.25 阈值 → 80/20 关键词覆盖率重排

Experiment:
Dense Top 10 → 0.25 阈值 ┐
                          ├→ RRF(k=60)
BM25 Top 10 ──────────────┘
```

两组都继续使用 Top 3 Seed、±2 Adjacent Expansion 和最多 8 个 Context Chunk。
Dense TopK 保持 10，而不是改成提示词示例中的 50，避免同时改变召回深度。

BM25 使用 `k1=1.5`、`b=0.75`。英文、数字和常见技术符号作为完整词项；连续中文
使用重叠双字词。该方案复用项目现有中文字面匹配思路，不引入第三方分词或 BM25
依赖。它适合当前 147 个 Chunk 的本地单用户实验，但不是大规模增量 Sparse Index。

## 零费用 Sparse 诊断（不是正式 A/B）

在当前 147 个 Chunk 的 disposable snapshot 上，仅运行 BM25，不调用 Embedding：

- 9 个可回答案例中，7 个案例的 Gold 位于 BM25 Top 1。
- `cauchy_properties` 的 Gold 位于 BM25 第 9。
- `multinomial_cross_chunk_moments` 的一个 Gold 位于 BM25 第 7，另一个未进入 Top 10。
- `mysql_out_of_scope` 返回 0 个 BM25 候选。

这只能证明 BM25 对当前语料中的精确术语有补充潜力，不能证明 RRF 后的总体 Recall、
MRR、nDCG 或延迟已经优于 Baseline。
