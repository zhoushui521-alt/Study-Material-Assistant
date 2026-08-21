# Study Material Assistant V2 · Stage 4 Technical Completion Report

## 1. Stage 4 目标

Stage 4 将已经稳定的证据式 RAG 组合成面向学习任务的单 Tutor 工作流。它解决的不是
“再增加一种 Retrieval”，而是把用户输入路由到问答、解释、练习、总结或学习规划，
并在同一 Session 内保留有限的学习上下文。

本阶段没有替换 Vector Store、Retriever、Context Selector、Evidence 或 Citation，
也没有引入 Multi-Agent、MCP、A2A、长期 Memory 或新的外部依赖。

## 2. Agent 架构

实现入口为 `app/tutor_workflow.py`。Tutor 是一个受控的单 Agent workflow：

```text
用户消息
  ↓
确定性 Intent Classification
  ↓
LangGraph Router
  ├─ Knowledge QA / Explanation ── Knowledge Retrieval Tool
  ├─ Quiz ───────────────────────── Knowledge Retrieval → Quiz Tool
  ├─ Summary ────────────────────── Session Summary Tool
  │                                 或 Knowledge Retrieval → Summary Tool
  └─ Study Planning ─────────────── Knowledge Retrieval → 确定性学习闭环
  ↓
Tutor Response
```

分类器不调用模型，因此路由结果可复现，也不会为“决定调用哪个工具”额外产生一次模型
费用。模型只用于现有 RAG 回答，以及需要生成结构化练习或总结的节点。

## 3. LangGraph Workflow

图使用 `StateGraph`、显式节点和条件边：

```text
START
  → classify_intent
  → [session summary | retrieve_knowledge]
  → [respond | generate_quiz | generate_summary | build_study_plan]
  → build_response
  → END
```

每个动态节点只使用一种条件路由机制，没有同时叠加静态边与条件边。工具失败会写入
通用错误状态并停止，不会把原始异常、密钥、URL 或路径交给后续模型或 API 响应。

## 4. State 设计

`TutorState` 只保存 JSON 可序列化字段：

- 输入与决策：`session_id`、`user_input`、`intent`、`topic`、`learning_goal`；
- RAG 结果：`retrieved_context`、`sources`、`citations`、`evidence`；
- 学习产物：`quiz`、`summary`、`answer`、`learning_action`；
- 过程记录：`conversation`、`tools_used`、`route_trace`、`error`。

服务对象和模型不进入 State，而是通过 LangGraph Runtime context 注入。Session 使用
`InMemorySaver`，同一 `session_id` 可以跨多轮复用当前 topic，以及最近 20 条、合计
12000 字符的消息；“继续出一道题”等续问会继承上一轮 topic。进程退出后状态消失。
本阶段没有写入 SQLite、向量记忆、用户画像或跨进程状态。

## 5. Tool 设计

### Knowledge Retrieval Tool

`KnowledgeRetrievalTool` 直接调用现有 `RAGService.ask()`。因此正式路径仍是 Stage 3
完成后的 Retriever、Context Selector、Evidence Map、Citation 校验和 LCEL RAG 链。
Tool 只把 `RAGAnswer` 转为 Tutor 可序列化的 answer、sources、citations 和 evidence，
没有复制 Retrieval 或 Prompt。

### Quiz Generator Tool

`QuizGeneratorTool` 接收 `topic` 和经过 RAG 约束的 `knowledge_context`，通过当前
ChatModel 的结构化输出生成一道选择题或简答题，结果包含 question、options、answer
和 explanation。资料不足时工作流在进入该 Tool 前停止。

### Learning Summary Tool

`LearningSummaryTool` 接收受限 Session conversation 和 topic，输出 summary、
key_points、next_steps。总结“刚才/本次学习内容”时不重复调用 RAG；总结一个新主题时
先调用 RAG，再把资料回答作为总结输入。

## 6. API 变化

新增：

```http
POST /api/tutor/chat
```

请求：

```json
{
  "message": "帮我出题练习 Embedding",
  "session_id": "11111111-1111-4111-8111-111111111111",
  "confirm_api_cost": true
}
```

响应包含 `answer`、`intent`、`topic`、`sources`、`citations`、`evidence`、
`learning_action`、可选 `quiz`、可选 `summary`、`tools_used` 和 `route_trace`。

`confirm_api_cost` 必须显式为 `true`。Tutor 与旧 Agent 共用现有 `agent` 频率、并发和
进程级预算保护。QA、解释和学习规划最多执行现有 RAG 的 Query Embedding 与 Chat
调用；Quiz 和新主题 Summary 会再增加一次结构化 Chat 调用；Session Summary 只执行
一次结构化 Chat 调用。`/api/ask`、`/api/agent` 和 `/api/study-workflows` 保持原契约。

## 7. Evaluation

Stage 4 的自动评测不只断言回答文本，还覆盖：

- Router：QA、Explanation、Quiz、Summary、Study Planning 五类代表性输入；
- Tool Calling：工具名称、调用顺序和 question/topic/context 参数；
- RAG Integration：确认调用现有 `RAGService.ask()`，并保留 Evidence/Citation；
- Workflow：Session Summary 跳过 RAG、主题 Summary 先检索、Quiz 两节点路径、
  Study Plan 确定性路径、无证据停止和工具失败路径；
- API：费用确认、UUID Session、额外字段拒绝、限流、超时、错误脱敏、隐私日志和
  RAG 索引失效后的 Tutor 失效。

这些结果属于 Fake/Mock 自动化证据，不是当前 ChatModel 的真实意图识别、练习质量或
总结质量验收。

## 8. 测试结果

- Stage 4 专项测试：17/17 通过；
- 全量自动化回归：360/360 通过；
- 内存编译：`app/tutor_workflow.py` 与 `app/api.py` 通过；
- 全项目内存编译：79 个 `app/` 与 `tests/` Python 文件通过；
- 依赖检查与 `web/static/app.js` 语法检查通过；
- 真实 Embedding / ChatModel：本阶段未调用；
- 本地 Uvicorn 与真实浏览器/API：本阶段尚未执行。

## 9. 未实现能力

- 没有专用 Tutor Web UI；当前交付边界是后端 API 与工作流；
- 没有长期 Memory、用户画像、跨进程 Session、鉴权或多用户隔离；
- 没有自由规划、Multi-Agent、Supervisor、MCP 或 A2A；
- 没有真实模型的 Router/Quiz/Summary 质量基准和人工验收；
- 没有生产并发、重启恢复、Session 删除 API 或持久化生命周期策略。

## 10. 下一阶段建议

Stage 4 后应先停止并做真实但有数量边界的 Tutor 验收：固定一小组 QA、解释、Quiz、
Summary 和 Study Plan 案例，分别检查路由、证据支持、练习可回答性、总结忠实度、延迟
和调用成本。只有 Session 跨重启或多用户需求成为真实问题时，才进入后端持久化和
身份隔离；当前没有证据支持 Multi-Agent、MCP 或更复杂 Memory。
