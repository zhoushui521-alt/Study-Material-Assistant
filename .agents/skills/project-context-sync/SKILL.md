---
name: project-context-sync
description: 审计或同步长期项目的代码事实、Git 边界、阶段状态、架构文档、技术决策与 README；当用户要求刷新项目上下文、更新交接文档、同步阶段完成状态或检查文档与实现一致性时使用。不用于阶段学习复盘、日常代码实现、缺陷修复或未经请求的文档重写。
---

# Project Context Sync

## 目标与边界

让项目文档成为当前仓库状态的可验证入口，而不是宣传材料：

```text
真实代码与调用链
  + Git 边界
  + 测试、评测和运行证据
  + 已记录的技术决策
  ↓
项目状态、架构、决策与 README
```

区分两种模式：

- **Audit**：用户只要求检查、说明、审计或交接分析时，保持只读，报告不一致和建议。
- **Sync**：用户明确要求更新或同步文档时，只修改确认范围内的项目文档，并在修改后验证差异。

调用本 Skill 不自动授权修改业务代码、依赖、索引、数据库或配置，也不授权 commit、tag、push、merge、部署或迁移。阶段完成同样不自动授权 Git 变更。若用户要求的是阶段工程学习复盘，应使用项目的 `stage-review`，不要用本 Skill 代替。

## 1. 读取真实状态

先读取适用的 `AGENTS.md` 和项目规则，再执行只读检查：

```powershell
git status --short --branch
git log --oneline --decorate --max-count=20
git tag --list --sort=version:refname
```

如果 Windows 因仓库所有权拒绝 Git，只在确认仓库绝对路径后对单条命令使用 `git -c safe.directory=<repo> ...`，不要修改全局 Git 配置。

然后按实际存在情况读取：

- `README`、`docs/`、阶段计划和相关 Completion Report；
- 已有的项目状态、架构、决策或 ADR 文档；
- 真实程序入口、服务层调用链、数据模型、配置和测试；
- 与目标 Stage 对应的 tag、commit、diff 和验证记录。

不要根据文件名猜职责。沿入口到服务、存储、AI Pipeline 和测试追踪真实调用关系。Completion Report、README 和旧架构图是声明或历史证据，不自动等于当前运行事实。

工作区不干净时，先区分：

- 已提交基线；
- 用户现有未提交改动；
- 本次文档同步产生的改动。

不得覆盖、回退或吸收用户的无关改动。历史 tag 说明的是历史边界，当前工作区说明的是当前候选状态，两者不能混写。

## 2. 建立事实账本

在写文档前，为每项关键能力记录：

- 结论；
- 代码或配置证据；
- 是否位于真实调用链；
- 测试、评测或运行证据；
- 对应 commit/tag 或当前工作区边界；
- 限制、反例和未验证项。

严格区分以下层级：

1. 代码已实现；
2. 自动化测试已覆盖；
3. 本次或历史记录显示测试通过；
4. 本地运行已验证；
5. 真实模型或外部服务已验证；
6. 生产环境已验证；
7. 计划、推断或尚未验证。

Fake、Mock、Stub、示例、Prompt、类型定义和文档声明不能提升证据层级。测试存在也不等于本次测试已运行。

## 3. 选择文档目标

先查找已有的等价文档和项目命名约定，优先更新单一事实来源，不制造重复文件。如果仓库没有等价文档，且用户要求建立标准上下文文件，可使用：

```text
docs/PROJECT_STATUS.md
docs/ARCHITECTURE.md
docs/DECISIONS.md
```

不要仅因运行本 Skill 就自动创建全部三个文件。只创建当前请求真正需要、能够由证据支撑的文档。

### PROJECT_STATUS.md

记录：

- 项目定位与真实使用范围；
- 当前 Git/Stage 边界及工作区说明；
- 已完成能力及其最高验证层级；
- 正在进行但尚未完成的内容；
- 明确未实现或不在范围内的能力；
- 下一步及其前置条件。

不要根据路线图编号推断当前 Stage，也不要把“代码存在”写成“Stage 已验收”。`Completed / Current / Next` 必须附证据或明确标为计划。

### ARCHITECTURE.md

只记录当前真实架构，包括适用部分：

- Frontend 与用户入口；
- Backend、API 与服务调用链；
- Database、Index、Storage 与数据所有权；
- AI / RAG Pipeline；
- Agent 或 Workflow 状态流；
- 外部依赖、Deployment 和运行边界；
- 关键失败行为、安全与成本边界。

未来架构必须放入独立的 Proposed/Future 区域并显式标记，不能和当前组件画在同一条已实现数据流中。

### DECISIONS.md

只记录有证据的重要决策，不从最终代码反向编造当时讨论。对新决策使用：

```text
Decision:
Status:
Background:
Options:
Choice:
Reason:
Trade-off:
Evidence:
Revisit Trigger:
```

保留历史决策。若决策已被替代，标记 Superseded 并指向新决策，不静默重写过去。像“不使用 Reranker”这样的结论必须写明适用版本、评测证据和重新评估条件。

### README

README 只承担项目入口、运行方式和主要能力边界。仅在它与当前事实冲突或缺少必要入口时更新；详细阶段状态、架构和决策应链接到专门文档，避免复制后再次漂移。

## 4. 阶段完成后的同步顺序

仅当 Stage 的边界和验收证据真实存在时，按以下顺序核对：

```text
Completion Report / 验收记录
  ↓
PROJECT_STATUS 当前与已完成状态
  ↓
ARCHITECTURE 当前数据流与边界
  ↓
DECISIONS 新决策或被替代决策
  ↓
README 入口与能力描述
```

任一环节缺少证据时，保留“未验证”或“待验收”，不要为了文档整齐强行宣布阶段完成。不要自动生成 Completion Report；只有用户明确要求且有足够证据时才创建。

## 5. 一致性检查

逐项检查：

- 文档中的文件、接口、配置和组件是否真实存在；
- 代码是否从真实入口接入，而不只是孤立模块或测试辅助；
- 文档描述的状态是否与 Git tag、commit 和工作区一致；
- 测试、离线评测、本地运行、真实模型与生产证据是否被混淆；
- Current、Proposed、Experimental、Legacy 与 Deprecated 是否清楚；
- RAG 中 Retrieval Candidate、Evidence 与 Citation 是否混称；
- 成本、数据发送、权限、安全和部署边界是否被遗漏；
- 多份文档之间是否出现阶段、命名、数据流或技术决策冲突。

将发现分类为：

- **Confirmed Inconsistency**：代码或 Git 证据已证明文档错误；
- **Needs Verification**：当前证据不足，不能直接改成另一种确定说法；
- **Planned, Not Implemented**：路线图或设计存在，但未进入真实调用链。

文档高于实际能力时降低表述；代码高于文档时，只把已验证事实补入文档。

## 6. 修改、验证与 Review

在 Sync 模式下：

1. 只修改用户请求和证据覆盖的文档。
2. 保留项目已有结构、术语、链接和用户未提交内容。
3. 检查 Markdown 结构、相对链接、路径、命令和版本引用。
4. 查看实际 `git diff` 与 `git status`，区分本次改动和既有改动。
5. Review 是否漏写限制、夸大能力、复制事实来源或误改无关内容。
6. 发现问题后修正并重新检查。

文档同步一般不需要运行完整业务测试；若某项文档结论依赖运行行为，只能复用清晰标注的可信记录，或在用户授权且零费用、不修改真实数据时执行最小验证。没有运行就明确说明。

## 输出要求

Audit 模式输出：

```text
Repository Boundary:
Current Project State:
Confirmed Inconsistencies:
Needs Verification:
Recommended Sync Scope:
```

Sync 模式输出：

```text
Updated Files:
Changed Content:
Detected Inconsistencies and Resolution:
Current Project State:
Validation and Review:
Remaining Unknowns:
```

路径、commit、tag、测试结果和时间范围应尽量可复核。没有发现不一致时明确说明检查范围，不声称整个项目永远一致。

## 完成前自检

- 结论来自真实仓库、调用链和证据，而不是聊天记忆或文件名。
- 已区分历史 commit、当前 HEAD、用户未提交改动和本次修改。
- 文档只描述当前已实现能力，未来方案有清楚标记。
- 没有把 Mock、测试覆盖或 Completion Report 外推成更高层验证。
- 没有创建重复事实来源，也没有静默改写历史决策。
- 没有因同步文档擅自修改业务代码、Git 历史、外部系统或真实数据。
