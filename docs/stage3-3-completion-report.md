# Stage 3.3 Technical Completion Report

## 1. 实验目标与边界

Stage 3.3 只验证一个问题：在 Embedding、Retrieval Algorithm、阈值、Top-K、Context
Construction 和 Evaluation Dataset 全部不变时，Structure-aware Chunking 是否优于当前
固定字符 Chunking。

实验基线是 commit `cc6e344363f36e542f1f5a52cec3ada7731f6969`，对应 Tag
`study-material-v2-stage3-part2`。本阶段没有修改正式 `build_chunks()`，没有重建或迁移
生产索引，没有修改 BM25、RRF、Reranker、Query Rewrite、Prompt 或 LLM，也没有开始
Stage 3.4。

## 2. 受控 A/B Pipeline

```text
A Current:
MaterialUnit → whitespace normalize → fixed 180 characters → temporary Chroma

B Structure-aware:
MaterialUnit → heading / paragraph / list / code structure
             → sentence / word / character fallback
             → max 600 characters, overlap 0
             → temporary Chroma

共同后续：
text-embedding-v4 → Dense Top 10 → threshold 0.25
→ 80% vector + 20% keyword coverage → Top 3 Seed
→ same-source adjacent window 2 → Context Limit 8
```

现有生产索引是 `legacy_read_only`，没有 Manifest，无法从索引自身证明其 Embedding 与
本轮候选索引一致。因此 A、B 都使用同一运行时 Embedding 配置在临时目录重建，生产索引
只用于只读兼容性、Current Mapping 和文件指纹核验。这样避免把旧索引身份不完整的问题
混入 Chunking 变量。

## 3. Structure-aware 策略

候选 Chunker 版本为 `structure-aware-block-600-overlap-0-v1`：

- Markdown / Web：识别 1–6 级标题，把标题路径写入 Chunk 正文与 `section`；连续列表和
  fenced code block 作为结构块处理。
- PDF / Text：空行优先形成段落；普通段落超限时按句子、词、字符逐级回退，避免把 PDF
  版面换行误认为语义边界。
- DOCX：复用 Parser 已提供的段落、表格和 section 边界，并保留 paragraph/table locator。
- 最大长度 600 个字符，overlap 固定为 0，避免重复证据影响 Context Precision 与引用定位。
- 继续生成标准 `DocumentChunk`，复用 Material ID、Content Hash、Chunk ID、Evidence 和
  Manifest 契约。

第一版曾把 PDF 排版行优先当成分割边界，导致 Stable Gold 无法映射。修正为普通段落
优先按句子回退后，10 个案例全部能确定性重建 Current Chunk Mapping。

## 4. Gold Mapping 与 Citation 定位

Stage 2 的 Stable Gold Meaning 有 8 条是语义标注而不是资料原文，不能只靠字符串包含
重新映射。Stage 3.3 使用以下严格契约：

1. 优先用 Stable Gold 原文、Material、Source、Page 定位唯一候选 Chunk。
2. 原文不是逐字摘录时，只允许把已人工确认的 Current Gold Chunk 完整包含进唯一候选
   Chunk。
3. 没有匹配或存在多个匹配时立即失败，不使用模糊相似度猜测 Gold。

候选策略把原先跨两个固定 Chunk 的 Multi-Gold 内容合并为一个结构 Chunk，因此可回答
案例的 Mapping 从 A 的 10 个变为 B 的 9 个。B 的 9 个 Mapping 全部可解析，Material、
Source、Page 匹配率均为 1.0，locator 全部唯一；重复构建得到相同 Chunk ID 与 Content
Hash。当前 PDF Parser 没有结构化 section 元数据，所以 A、B 的 PDF section 可用率均为
0，这不是本阶段回归，也没有在本阶段越界修改 Parser。

这些证据只验证定位契约与可复现性，不验证生成答案中的 Citation Support 或 Citation
Accuracy；本次没有调用 ChatModel。

## 5. 真实 Evaluation 证据

正式运行报告：
`evaluation/results/chunking-20260820T170139796369Z-f15ee14e.json`。该目录被 Git
忽略，报告是本地运行证据，不进入源码提交。

- Dataset：未修改的 `evaluation/retrieval_cases.json`，10 个案例，其中 9 个可回答
- Dataset SHA-256：
  `47ed2974070385d06e6ab9c8d57660e9a1f3c93f9536ed09c1993524585b3e8f`
- Validation：`local_real_structure_aware_chunking`
- Embedding：两条 Pipeline 均为 `text-embedding-v4`
- Index Embedding：A 147 条、B 55 条，共 202 条；逻辑批次数 15 + 6 = 21
- Query Embedding：A/B 共 20 次逻辑调用
- ChatModel：0 次
- 实际 API 金额：未测量
- 成功运行 wall-clock：约 7.809 秒
- 生产索引运行前后 filesystem SHA-256 指纹一致
- A/B 临时索引都有独立 Manifest 与文件指纹，运行后自动清理

报告绑定 Stage 起点 commit、Dataset Hash、实现文件 Hash、两条策略参数、Retrieval
Config 和三个索引指纹，不保存 API Key、Base URL、Chunk 正文或查询正文。

## 6. Retrieval 与 Context 指标

以下 Retrieval 指标按 9 个可回答案例平均：

| Metric | A Current | B Structure-aware | B - A |
| --- | ---: | ---: | ---: |
| Recall@1 | 0.3333 | 0.5556 | +0.2222 |
| Recall@3 | 0.8333 | 0.6667 | -0.1667 |
| Recall@5 | 0.8333 | 0.8889 | +0.0556 |
| Recall@10 | 0.9444 | 0.8889 | -0.0556 |
| MRR | 0.6111 | 0.6426 | +0.0315 |
| nDCG@5 | 0.6422 | 0.7019 | +0.0598 |

| Context Metric | A Current | B Structure-aware | B - A |
| --- | ---: | ---: | ---: |
| Context Precision | 0.1500 | 0.1202 | -0.0298 |
| Final Context Recall | 1.0000 | 0.7778 | -0.2222 |
| Average Context Chunk Count | 7.3 | 5.8 | -1.5 |

B 提升了 Recall@1、MRR 和 nDCG@5，说明更完整的结构块能让部分定义或跨固定边界内容
更早出现；但正式 Pipeline 只取 Top 3 Seed，B 的 Recall@3 与 Final Context Recall 明显
下降。Chunk 数从 147 降到 55、Median 从 180 增至 481 后，候选粒度变粗，也使 Context
Precision 下降。不能用前排平均排序收益掩盖 Top 3 召回与最终上下文回归。

## 7. 逐案例变化与失败

主要提升：

- `multinomial_cross_chunk_moments`：两个固定 Chunk 的 Gold 合并为一个结构 Chunk，
  Recall@1 `0 → 1`，MRR `0.5 → 1.0`。
- `student_t_suitability`：Recall@1 `0 → 1`，MRR `0.5 → 1.0`。
- `wishart_definition`：从 A 的 Ranking Failure 恢复，MRR `0.1667 → 1.0`。

主要回归：

- `cauchy_properties`：Recall@3 `1 → 0`，Final Context Recall `1 → 0`，新增
  Ranking Failure。
- `dirichlet_definition`：Recall@3 `1 → 0`、MRR `0.5 → 0`，新增 Recall Failure。
- `cauchy_relationships`：仍进入 Top 3，但 MRR `1.0 → 0.3333`。
- `inverse_wishart_definition`：Recall@3 `1 → 0`，新增 Ranking Failure；Gold 仍在
  Top 5，Final Context 由邻接扩展恢复。

`mysql_out_of_scope` 在 A、B 中都保持 Unanswerable Handling Failure。Chunking 不负责
无答案判定，本阶段也没有修改 Threshold 或拒答逻辑。

## 8. 延迟与工程取舍

| Pipeline | 平均本地 Retrieval 延迟 |
| --- | ---: |
| A Current | 123.9 ms |
| B Structure-aware | 93.3 ms |

B 平均少约 30.6 ms，与索引 Chunk 数从 147 降到 55 有关，但这里只是当前 Windows
机器、10 个顺序案例的一次 wall-clock 观察，不是并发或生产性能基线。更少 Chunk 带来的
延迟收益不能抵消 Recall@3、Final Context Recall 与 Context Precision 的回归。

## 9. 工程判断

Stage 3.3 的结论不是“Structure-aware 一定更好”，而是当前 `max_chars=600`、0 overlap
的候选策略呈现明显混合结果：

- 它改善了部分跨边界内容和前排排序，并减少索引记录与本地检索耗时。
- 它同时制造了新的 Recall / Ranking Failure，并让正式 Top 3 + Adjacent Context 的最终
 召回下降 0.2222。

因此本阶段完成的是可复现的 Structure-aware Chunking 实验能力，而不是生产 Chunker
升级。候选策略不接入 `build_chunks()`、资料摄取或正式 RAG Pipeline；生产 Baseline
继续使用 `fixed-character-180-v1`。

后续若继续研究，应作为新的单变量实验，优先验证更小的结构块上限或段落级上限策略，
不能同时调整 Retrieval、Top-K、Threshold 或 Context Construction。Stage 3.3 本身不
开始 Stage 3.4。

## 10. 验证与未验证项

已验证：Structure-aware 单元行为、最大长度与 0 overlap、稳定身份、DOCX/Markdown
结构保留、生产 Chunker 不变、Gold Mapping 失败关闭、Citation locator、报告无正文、
双临时索引同配置构建、费用双确认、328 项自动化测试、真实 10 案例 A/B、生产索引
指纹不变、Python 编译与 Git 差异检查。

未验证：真实 ChatModel 与答案质量、Citation Support / Accuracy、不同 Chunk 上限、扫描
PDF、复杂表格/代码、大语料、并发、生产部署和实际 API 金额。本报告不把 10 个案例的
混合结果外推为普遍质量结论。
