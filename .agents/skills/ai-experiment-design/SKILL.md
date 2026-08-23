---
name: ai-experiment-design
description: 为 RAG、Agent、Prompt、模型、Context Construction、检索排序、延迟或成本优化设计可复现的对照实验；当用户要求比较方案、建立 Baseline、验证优化效果或决定是否接入实验能力时使用。不用于没有对照变量的普通功能开发或仅需定位代码缺陷的 Debug。
---

# AI Experiment Design

## 目标与边界

用可复现证据决定 AI 系统是否应该改变：

```text
发现问题 → 定位层级 → 固定 Baseline → 提出假设 → 控制变量实验 → 分析收益与代价 → 决策
```

不要因“感觉不好”直接更换模型、Prompt、Embedding、Chunk、Reranker 或 Agent 架构。没有可观察问题、稳定数据和对照基线时，先补齐最小评测条件，不把调参过程包装成实验结论。

实验设计不自动授权修改正式系统、发送数据或调用付费模型。执行前应说明外部调用次数、数据去向和预估成本，并在需要真实付费或数据发送时取得用户明确授权。Fake、Mock、Stub 和历史回放只能证明相应测试或离线行为，不能冒充新鲜检索、真实模型或生产验收。

## 1. 定位真实问题

先从日志、失败样例、Trace、评测数据或用户任务中确认症状，再定位层级：

- **Data / Parser / Chunk**：资料缺失、解析错误、切分破坏语义、Gold 不可检索。
- **Retrieval**：Recall 不足、相关结果未召回、排序位置不理想。
- **Context**：证据筛选错误、噪声过多、Token 过大、关键信息被截断。
- **Generation / Citation**：回答错误、无依据、引用不支持主张或拒答失效。
- **Agent / Workflow**：路由、规划、工具选择、状态转换、终止或恢复错误。
- **System**：延迟、Token、费用、资源占用、稳定性或并发问题。

不要用一个层级的技术掩盖另一个层级的问题。例如，召回缺失时先检查语料、切分、查询和 Retrieval，不要直接靠 Prompt 或 Reranker 补救。

## 2. 固定 Baseline

每次实验记录可复核的 Baseline：

```text
Baseline ID:
System / Commit / Config:
Dataset and Version:
Model / Prompt / Index Version:
Metrics and Calculation:
Known Problem:
Evidence Level:
```

Baseline 必须和实验使用同一套数据、指标口径与运行条件。若无法冻结模型版本、随机性或远程服务状态，记录这一限制及其可能影响。

不要只保存平均分。保留 case-level 结果、失败分类、输入、期望、实际输出和必要的 Trace，以便判断提升来自哪里、哪些样例发生回归。

## 3. 写出实验契约

一次实验只改变一个主要变量。无法完全隔离时，列出同时变化的因素，降低因果结论的强度。

```text
Experiment ID:
Problem:
Hypothesis: 为什么这个变化会改善哪个指标
Prediction: 若假设成立，应观察到什么
Change: 唯一主要变量及所属层级
Keep Fixed: 数据集、Gold、Chunk、Embedding、索引、检索参数、模型、Prompt、随机种子等
Primary Metric:
Guardrails: 不得明显回退的质量、延迟、成本或安全指标
Acceptance Gate:
Execution Plan:
```

在看结果前确定主要指标、容许波动和采用门槛，避免结果出来后挑选有利口径。变体很多时，按顺序做小实验，不把多个组件一起加入后归因给其中一个。

## 4. 选择指标与样例

指标必须绑定数据集、计算方式、K 值、样本量和版本：

- Retrieval：Recall@K、MRR、nDCG、无结果率及 case-level 排名。
- Answer：Correctness、Faithfulness、Citation Accuracy、拒答正确性。
- Agent / Workflow：路由正确率、工具成功率、任务完成率、步骤数、恢复与终止行为。
- System：端到端延迟、分阶段延迟、Token、单次成本、错误率和资源使用。

选择能覆盖正常、边界、失败和代表性真实任务的样例。Gold 必须独立于待测系统，不能为了让新方案获胜而同步修改；确需修正 Gold 时，将其作为独立的数据集版本变更，重新运行 Baseline 与实验组。

LLM-as-a-Judge 只是测量工具，需记录 Judge 模型、Prompt、量表、一致性检查及偏差边界。少量样本或波动较大时报告不确定性，不把微小上涨写成稳定提升。

## 5. 执行与记录

- 优先使用零费用、离线、只读实验；保护真实资料、索引和持久数据。
- 为 Baseline 与实验组使用相同运行顺序、缓存策略、并发和重试规则。
- 记录失败、超时、跳过、重试和无效样例，不能只统计成功请求。
- 把运行产物与正式系统隔离，注明实验配置、时间、环境和可复现命令。
- 若实验需要改代码，限制在声明的层级，并检查输入输出和上下游契约是否保持不变。

## 6. 分析结果与取舍

同时报告：

- 主要指标的绝对值、绝对差和相对变化；
- Guardrail 是否通过；
- case-level 改善、回归与失败分类；
- 延迟、Token、费用、依赖和维护复杂度；
- 数据、执行或测量中的不确定性。

指标上涨不自动等于值得接入。例如 MRR 提升但延迟或成本越过门槛，结论可能仍是保留实验或拒绝。平均值改善但关键安全样例回归，也不能采用。

## 7. 做出决策

根据预先定义的 Acceptance Gate 选择：

- **Adopt**：主要指标与 Guardrail 均达标，证据足以接入正式路径，并有回归验证方案。
- **Keep Experimental**：有潜力但证据、稳定性、成本或兼容性不足；说明下一项最小实验。证据无效或不足时也归入此状态，不强行宣布胜负。
- **Reject**：未达到门槛，或收益不足以覆盖成本与复杂度；记录原因，避免重复试错。

“Adopt”是技术决策，不等于已经上线。必须另外说明是否仅有实验实现、是否接入正式调用链、是否运行自动化测试、是否经过真实模型和生产验证。

## 输出要求

按以下结构输出，并省略与当前任务无关的空项：

1. Experiment Goal
2. Current Baseline
3. Observed Problem and Layer
4. Hypothesis and Prediction
5. Change / Keep Fixed
6. Dataset, Metrics and Acceptance Gate
7. Execution and Results
8. Case-level Regressions
9. Benefit, Cost and Trade-off
10. Final Decision and Next Experiment

如果实验尚未执行，明确写成“实验方案”，不要生成虚构 Result 或 Final Decision。

## 完成前自检

- 问题有可观察证据，并已定位到正确层级。
- Baseline、数据集、Gold、指标口径和主要变量可复核。
- 采用门槛在结果分析前定义，且没有选择性报告。
- 同时检查平均指标、case-level 回归、延迟、成本和复杂度。
- 结论没有超过自动化测试、离线回放、真实模型或生产证据的层级。
- 未经授权没有付费调用、发送数据、修改真实索引或接入正式系统。
