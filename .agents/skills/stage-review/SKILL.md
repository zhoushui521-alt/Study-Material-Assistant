---
name: stage-review
description: 基于真实 Git 标签、提交差异、阶段末代码、测试评测和项目文档，为 Study Material Assistant 已完成的 V2 阶段生成中文工程学习复盘。仅在用户要求阶段复盘、阶段总结、学习复盘、设计取舍或阶段证据核验时使用；不用于日常代码审查、修复缺陷、实现新阶段、部署或未形成可信边界的进行中工作。
---

# Stage Review

## 目标与边界

把一个已经完成的 V2 Stage 还原为可学习、可核对的工程叙事：为什么改、真实数据流如何变化、设计解决了什么、付出什么代价、哪些结论已经验证、哪些仍未证明。

把 Stage Review 与 Code Review 分开：

- Code Review 寻找当前差异中的缺陷、回归和安全问题。
- Stage Review 在边界明确后解释阶段成果、证据、取舍、修正过程和可迁移经验。
- 可以引用已有 Review 发现及其修复，但不得重新包装成一次代码审查，也不得借复盘修改业务代码。

整个流程保持只读。不得修改业务代码、索引、标签、提交、依赖或配置；不得 push、部署、迁移、调用付费模型或执行破坏性 Git 操作。若发现缺陷，只记录证据、影响和建议，不直接修复。

## 需要的输入

优先使用项目阶段标签：

- 结束边界：`study-material-v2-stageX`
- Stage 1 起点：`study-material-v2-baseline`
- Stage X（X > 1）起点：先计算上一个阶段编号，再验证对应的 `study-material-v2-stageN` 标签

先列出和验证标签，再采用上述命名关系。若任一标签不存在、无法解析为 commit、起点不是终点祖先，或用户指向的阶段尚未完成，停止并说明缺少的证据；不要用日期、文件时间、当前 `HEAD` 或主观判断猜边界。用户明确提供两个 commit/tag 时可以使用，但要说明这不是标准阶段标签边界。

## 工作流

### 1. 读取项目规则与阶段说明

1. 读取仓库根目录及相关文件路径适用的全部 `AGENTS.md`。
2. 读取 `README.md`、用户提供的阶段任务、阶段计划或 Completion Report；仅打开实际存在且与目标 Stage 相关的文档。
3. 从规则中提取当前 Stage 目标、禁止跨阶段事项、安全边界、完成标准和证据分层。
4. 把文档视为“声明或验收记录”，而不是自动等同于运行事实；后续用 Git、代码和验证输出交叉核对。

### 2. 固定 Git 边界

先运行只读检查：

```powershell
git status --short --branch
git tag --list "study-material-v2-*" --sort=version:refname
git log --oneline --decorate --max-count=20
git rev-parse --verify "<start>^{commit}"
git rev-parse --verify "<end>^{commit}"
git merge-base --is-ancestor <start> <end>
```

然后运行随 Skill 提供的只读采集器：

```powershell
python .agents/skills/stage-review/scripts/collect_stage_evidence.py --start <start> --end <end>
```

采集器只向标准输出写 JSON，不写仓库。它提供提交、标签、文件变更、统计和当前工作区状态的可复核快照。

记录两个边界的完整 commit hash。若工作区不干净，明确指出未提交改动不属于该 Stage；不要把当前文件内容混入阶段末事实。

### 3. 还原真实改动与调用链

从阶段差异而不是文件名猜测实现：

```powershell
git diff --stat <start>..<end>
git diff --name-status --find-renames <start>..<end>
git diff --check <start>..<end>
git diff <start>..<end> -- <focused-paths>
git show <end>:<path>
git grep -n "<symbol>" <end> -- <paths>
```

按真实入口向下追踪：API/CLI → 服务层 → RAG/摄取编排 → 数据契约 → 存储/Manifest → 测试。只有在 `<end>` 与当前 `HEAD` 相同且工作区干净时，才可直接把工作区文件当作阶段末快照；否则使用 `git show <end>:<path>`。

对每个核心能力回答：

- 输入、输出及上下游契约变了什么？
- 正常、空数据、非法引用、旧数据和失败路径怎样处理？
- 新设计解决了哪个可观察问题？
- 兼容性、安全性、复杂度和维护成本是什么？
- 哪个测试或评测覆盖了它，覆盖边界到哪里？

不要逐文件罗列。把多个文件串成一条数据流，并用少量关键路径作为证据锚点。

### 4. 建立结论证据账本

写作前在内部为每个关键结论记录：结论、证据来源、证据层级、反例或限制。至少区分：

1. **代码已实现**：阶段末代码存在且位于真实调用链中。
2. **自动化测试已覆盖**：测试断言覆盖对应行为；只有本次实际运行或可信 Completion Report 记录成功，才能写“通过”。
3. **本地运行已验证**：存在明确命令、环境、输入和结果；不能由测试替代。
4. **真实模型已验证**：必须有真实 LLM/Embedding/Rerank 调用证据及范围。
5. **生产环境已验证**：必须有部署环境和运行证据。
6. **架构改善或合理推断**：可从契约和控制流推出，但尚无效果指标。
7. **尚未验证**：缺少数据集、运行记录、指标或真实环境证据。

README、Prompt、类型定义、示例、Fake/Mock/Stub、测试代码和测试通过分别是不同证据。不得把“有测试”写成“测试已通过”，不得把 Fake/Mock 写成真实模型验收。

### 5. 专项核对 RAG Stage

沿 `Document → Parser → Cleaner → Chunker → Metadata → Embedding → Vector Store → Retrieval → Context → LLM → Citation` 定位改动层级。

对于 Evidence & Index Foundation，必须分别核对：

- `sources` 是检索候选/上下文来源，还是实际被答案引用的证据；不得把二者混称 Citation。
- Evidence 是否可定位到 material、chunk、locator、版本或索引信息。
- Citation ID 是否由模型自由提供事实，还是由服务端基于请求内 Evidence Map 校验并回填权威 metadata。
- 稳定 ID 的输入字段、规范化方式、哈希算法和变化语义；只说代码能够保证的稳定范围。
- Index Manifest 记录和校验哪些兼容性维度，在哪些 mutation 之前执行。
- `legacy_read_only` 是否覆盖仓库实际存在的同步、重建、删除以及其他 mutation 路径；若没有迁移实现，明确写成边界，不得假设它存在。读取兼容不等于迁移完成。
- 是否真的证明 Recall、MRR、Correctness、Faithfulness、Citation Accuracy、延迟或成本改善。没有独立评测就明确写“未证明”。

### 6. 还原 Review 修正过程

只从可核实来源描述阶段中发现并修复的问题，例如多提交历史、Completion Report、Review 记录、Issue 或用户提供的上下文。单个最终 commit 通常只能证明最终状态，不能证明中间曾出现某个 bug。

每个修正用“原风险 → 为什么会发生 → 修正位置/机制 → 回归证据 → 仍存边界”说明。若找不到过程证据，直接写“仅凭最终 diff 无法还原中间 Review 修正”，不要编造故事。

### 7. 必要时运行验证

默认复用阶段 Completion Report 中的历史验证记录，并清楚标为“历史记录，本次未复跑”。只有用户要求，且命令确认零费用、不接触真实个人资料、不修改真实索引或持久数据时，才复跑测试/评测。

若复跑，分别记录命令、退出状态、通过数量、跳过项和失败原因。命令成功只证明该命令范围；不要外推到真实模型、并发或生产。

## 输出格式

标题固定为 `《Stage X 学习复盘》`。使用自然中文、结论先行、工程叙事为主；避免 API 文档腔、逐文件清单和大表格堆砌。建议结构：

1. **复盘范围与一句话结论**：起止 tag/hash、工作区状态、阶段目标、证据来源。
2. **为什么要做这一阶段**：原问题、前置约束、为何没有提前引入后续技术。
3. **真实数据流如何变化**：从入口到输出串起核心调用链。
4. **关键设计与取舍**：每项设计说明收益、代价、兼容与失败行为。
5. **验证到了哪一层**：分开写实现、自动测试、本地运行、真实模型、生产。
6. **Review 中发现并修正的问题**：只写有过程证据的内容。
7. **边界与未验证项**：指标、环境、数据和跨阶段能力的真实限制。
8. **3–5 条可迁移学习收获**：说明何时适用、为何成立，避免技术名堆砌。
9. **3–5 个理解检查题**：能检验数据流、取舍、失败路径和证据边界；不要只问定义。
10. **证据锚点**：列出少量关键 commit、路径、测试和报告，便于复核。

引用路径时尽量给出阶段末文件与行号；若行号来自当前工作区而非 `<end>` 快照，要显式说明。事实、推断、建议和未知项使用明确措辞分开。

## 完成前自检

- 起止边界均已验证，且没有把工作区后续改动混入目标 Stage。
- 每个关键事实至少有一项可复核证据，重要结论尽量交叉验证。
- 已区分 Retrieval Candidate、Evidence、Citation 以及服务端权威 metadata。
- 已区分代码实现、测试覆盖、测试通过、本地运行、真实模型和生产验证。
- 没有从最终 diff 编造中间 bug、Review 过程、性能提升或用户行为。
- 没有执行付费 API、真实索引 mutation、业务代码修改或 Git 状态变更。
- 学习收获和理解题来自本阶段真实设计，而不是通用 RAG 套话。

如果任一关键证据缺失，在复盘中公开缺口；不要用更自信的措辞掩盖它。
