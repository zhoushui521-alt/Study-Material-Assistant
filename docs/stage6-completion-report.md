# Study Material Assistant V2 · Stage 6 Completion Report

## 1. 结论

Stage 6：Production Readiness & Portfolio 已达到本阶段验收条件。

本阶段没有新增核心 AI 业务能力，而是基于 Stage 0～5.5 的真实代码、Git checkpoint、自动化测试、
历史本地运行和受控评测，完成最终架构、工程文档、RAG 评测汇总、五分钟 Demo、面试材料和
全项目质量 Review。

最终交付 Tag：`study-material-v2-stage6-final`。该 Tag 在本报告提交后创建并指向同一提交。
本地提交未 push，也没有执行部署、付费模型调用、真实索引变更或数据迁移。

Stage 6 的完成结论是“作品化与工程证据闭环完成”，不是“生产环境验收完成”。

## 2. Git 基线与 checkpoint

| 范围 | Commit / Tag | 说明 |
| --- | --- | --- |
| Stage 5.5 代码 | `d54a90b` | BYOK + Model Gateway 正式调用链 |
| Stage 5.5 文档收口 | `9ac680a` | Stage 6 起点 |
| Stage 6.1 | `c3c39e0` | Final Architecture |
| Stage 6.2 | `826f75a` | Engineering Documentation |
| Stage 6.3 | `6f63f6a` | RAG Evaluation Report |
| Stage 6.4 | `8554491` | 5 分钟 Demo Guide |
| Stage 6.5 | `c773d05` | Interview Guide |
| Stage 6.6 | Final report commit | 全量验证、Review 与质量修复 |
| Final Tag | `study-material-v2-stage6-final` | Stage 6 最终可复核边界 |

远程 `origin/main` 没有在本阶段更新。公开仓库、CI 或部署平台状态不能由本地提交推断。

## 3. Stage 6.1 · Final Architecture

新增 [Final Architecture](FINAL_ARCHITECTURE.md)，从正式入口和调用链冻结以下结构：

- Browser → FastAPI → 业务服务 → Provider / Persistence；
- 上传暂存、费用确认、`202 + job_id`、单 Worker 和索引提交；
- Dense Retrieval、关键词覆盖率重排、Adjacent Expansion、Context Selector、Evidence Map、
  Model Gateway 和 Citation 校验；
- Agent、Tutor Graph 和 Study Workflow 的职责与状态流；
- 用户/Session/Job/BYOK/Chroma 的持久化边界；
- Request ID、JSONL、运行指标、安全和单服务部署边界；
- 正式、实验与未来能力分区。

正式 RAG 保持 Stage 2 Retrieval baseline + Stage 3.4 Context Selector；BM25 + RRF、
Cross-Encoder 与 Structure-aware Chunking 没有被包装成正式链路。

## 4. Stage 6.2 · Engineering Documentation

新增或同步：

- [Project Context](PROJECT_CONTEXT.md)：开发、Review 和交接最小上下文；
- [Evaluation Guide](EVALUATION.md)：Dataset、指标、费用门槛、失败分类和 CLI；
- [Development Guide](DEVELOPMENT_GUIDE.md)：本地环境、测试、数据、安全和 Git 流程；
- [README](../README.md)：文档导航、服务拓扑、Model Gateway 配置和排障；
- [Architecture](ARCHITECTURE.md) 与 [Decisions](DECISIONS.md)：持续架构和 Stage 6 决策边界。

7 个评测 CLI Parser 已逐个生成 `--help`，不调用 Embedding、ChatModel 或 Reranker。

## 5. Stage 6.3 · RAG Evaluation Report

新增 [RAG Evaluation Report](RAG_EVALUATION_REPORT.md)，汇总 Stage 2 与 Stage 3.1～3.4
已有受控实验。没有重新执行付费实验。

核心决策：

| 层 | 实验 | 结果 | 正式决策 |
| --- | --- | --- | --- |
| Candidate | BM25 + Dense + RRF | Recall@5 上升，MRR 回退，Top 3 不变 | Keep Experimental |
| Ranking | Cross-Encoder | MRR/Recall@1 上升；Context Precision 下降，CPU +2862.4 ms | Keep Experimental |
| Chunking | Structure-aware 600/0 | 部分前排提升；Recall@3、Final Context Recall、Precision 回退 | Reject Production |
| Context | EvidenceScore selector | Precision `0.1500 → 0.2000`，Recall `1.0` 保持，0 external calls | Adopt |

报告明确保留 10-case Dataset、历史 Trace、不同 Index/commit 和未验证 Answer/Citation 指标等
限制，不能把各轮最佳数字拼成不存在的 Pipeline。

## 6. Stage 6.4 · Demo Experience

新增 [5 分钟 Demo Guide](DEMO_GUIDE.md)：

- 30 秒定位；
- 登录与用户空间；
- 上传暂存、费用估算和可选 Job 进度；
- Ask、Evidence、Citation 与 Request ID；
- Tutor 复用 `RAGService.ask()` 的边界；
- Model Gateway 与 Metrics 的受鉴权 API 元数据展示；
- 安全演示和完整任务演示两种模式；
- 外部调用台账、故障恢复卡和 9 项完成清单。

Review 发现初次新文件补丁的行数声明少了 10 行，导致末尾验收项和链接被截断；已补齐并重新
验证。当前前端没有 Model Gateway 配置控件，Guide 没有虚构页面切换按钮。

## 7. Stage 6.5 · Interview Guide

新增 [Interview Guide](INTERVIEW_GUIDE.md)：

- 30 秒和 2 分钟第一人称项目介绍；
- 12 个 RAG、8 个 Agent/LangGraph、12 个工程化高频问题；
- 4 个可按情境/任务/行动/结果讲述的技术难点；
- 生产级、Dataset、Citation、云向量库、GraphRAG 和 Agent 自主性的压力追问；
- Claim/证据边界速查表和反问清单。

10 个关键 Claim 已映射到真实代码或评测报告。Review 发现“3 个难点”标题与正文 4 项不一致，
已修正。材料不虚构用户、业务指标、上线结果或生产稳定性。

## 8. Stage 6.6 · Full Project Review

### 8.1 Requirements Review

- Stage 6.1～6.5 均形成独立本地 commit；
- Final Architecture、Project Context、Evaluation、Development、RAG Report、Demo 与 Interview
  均有独立入口；
- 没有引入 Multi-Agent、MCP、GraphRAG、新数据库或微服务；
- 正式 RAG、Agent、Tutor、API 和数据契约没有被文档工作改写；
- 本报告提交后创建唯一 final Tag。

### 8.2 Code Quality / Dead Code / Duplicate Implementation

沿 `app/api.py` 检查了正式 RAG、Agent、Tutor、Document Job、Auth 与 Model Gateway 构造链，
6 个入口检查通过。

分类结果：

- `app/chat_client.py` 与 `app/ask_documents.py` 是早期教学/兼容入口，不在正式 V2 API 调用链；
- `HybridRRFRetriever`、Cross-Encoder、Structure-aware Chunker 与 Baseline Selector 用于隔离
  实验、对照和回退，不是重复接线；
- `app/`、`tests/` 未发现 TODO/FIXME/HACK/XXX 残留；
- 本阶段没有依据删除这些可解释历史或实验资产，避免把“代码少”误当成质量。

### 8.3 Security Review

- Git 已跟踪 153 个文件；未发现 `.env`、数据库、用户资料、向量索引、私钥文件或运行日志；
- 高风险 Key 模式只命中 2 个 `tests/` Fake fixture 文件，非测试路径命中为 0；
- 15 类实际运行路径通过 `git check-ignore`；
- Review 发现默认 `data/model_gateway/model_credentials.sqlite3` 原先未被 `.gitignore` 覆盖，
  已新增 `data/model_gateway/`；
- `.dockerignore` 已覆盖 `.env`、`data`、Git 和本地缓存；
- Model Gateway/Deployment 专项 17/17 通过，覆盖密文、用户隔离、输入约束、错误脱敏与部署契约；
- 未读取 `.env`、真实 Key 或用户资料。

### 8.4 Documentation Review

Review 发现并修复：

1. `PROJECT_STATUS` 的“当前正式主链路”标题在一次补丁中丢失；
2. Demo Guide 末尾 10 行被补丁行数截断；
3. Interview Guide 难点数量标题不一致；
4. README 仍把已完成的 Stage 5.4 和作品材料写成未开始；
5. README 早期阶段编号容易与 V2 Stage 6 混淆；
6. Compose 镜像仍标记 `stage5-3`，已更新为 `study-material-assistant:stage6-final`。

Stage 6 文档本地相对链接、最终换行、标题和关键事实断言在提交前统一检查。

## 9. 最终验证

### 9.1 自动化测试

最终命令：

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p 'test_*.py' -q
```

结果：

```text
Ran 434 tests in 30.899s
OK
full_test_exit=0
```

即 434/434 通过，0 failure、0 error、0 skip。测试使用 Fake/Mock/临时 SQLite 和本地文件；
没有调用付费 Embedding、ChatModel 或 Reranker。

### 9.2 Targeted / Static

| 检查 | 结果 |
| --- | --- |
| Deployment + Model Gateway 专项 | 17/17，0.395s，OK |
| Python 内存 `compile()` | 104 个 `app/` + `tests/` 文件通过 |
| `pip check` | `No broken requirements found` |
| `node --check web/static/app.js` | 通过 |
| Compose YAML | 1 个服务：`app` |
| Compose image | `study-material-assistant:stage6-final` |
| 正式调用链断言 | 6/6 |
| 运行路径 Git ignore | 15/15 |
| Stage 6 文档链接/结构 | 通过提交前全量检查 |
| `git diff --check` | 通过提交前检查 |

运行环境：Python 3.14.5、Node v24.15.0。当前没有 Docker CLI，因此没有执行 Docker
build/up、容器健康检查或重启恢复。

## 10. 未执行与证据不足

本阶段未执行：

- 当前认证版本的浏览器端到端 Demo；
- 真实 Embedding、ChatModel、Reranker 或 Provider 调用；
- 当前索引的 Retrieval/Answer/Citation 新 Benchmark；
- Docker build/up、容器重启、云部署或 HTTPS；
- 并发、负载、长时间、多实例、备份恢复和灾难恢复；
- KMS/HSM、主密钥轮换、公开多租户渗透或生产安全验收；
- Claim-level Faithfulness / Citation Support / Citation Accuracy。

历史本地 HTTP、浏览器、Docker 定义和真实模型实验仍按各自 Completion Report 的时间、commit、
Dataset 和环境理解；本报告不把历史证据升级为当前验收。

## 11. 最终边界与后续触发

当前项目是可展示、可维护、可复现的本地单实例 AI 学习系统原型。Stage 6 之后不预排固定技术
Stage。下一步优先由真实失败或目标触发：

1. 扩大并冻结 Retrieval/Answer/Citation Dataset；
2. 在当前正式索引上重跑 Context Selector 与端到端 Answer/Citation；
3. 先定义用户范围、容量、SLO、数据合规和部署目标，再讨论 PostgreSQL、外部队列、
   OpenTelemetry/LangSmith 或多实例；
4. 只有出现确切多步动态工具需求，才重新评估 Agent/Multi-Agent。

Stage 6 的工程经验是：最终收口不是继续堆技术，而是让实现、评测、文档、Demo、面试表达、
安全边界和 Git checkpoint 指向同一组可复核事实。
