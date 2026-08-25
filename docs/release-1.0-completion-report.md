# 知行 Release 1.0 Completion Report

> 完成日期：2026-08-25。本文记录 GitHub 仓库整理与部署准备结果，不代表代码已推送、
> 云端已部署或生产环境已验收。

## 1. Release 目标

Release 1.0 的目标是把 Stage 6 完成的本地工程作品整理为：

- 访问仓库后能快速理解产品问题、架构和工程亮点；
- 开发者能根据公开配置模板启动本地服务；
- 发布前的 Secret、运行数据和 BYOK 风险有明确检查；
- Demo 有可在五分钟内执行的稳定脚本；
- 部署路径、前置门槛、回滚和未验证项有独立指南；
- Git checkpoint 能区分 Stage 6 核心工程基线与 Release 1.0 发布准备。

本阶段没有修改 app/、web/、tests/、requirements.txt、Dockerfile、docker-compose.yml 或
任何 RAG、Retrieval、Chunk、Embedding、Reranker、Prompt、Tutor Workflow、Agent 与
Model Gateway 核心逻辑。

基线为 study-material-v2-stage6-final，指向 7c96f0b。

## 2. README 改进

根 README 从按历史阶段累积的长篇说明，收敛为面向面试官、开发者和技术评审的项目入口。

当前结构包括：

- 产品定位与问题；
- Features 与每项能力的证据边界；
- 当前系统架构图和正式 RAG 数据流；
- Frontend、Backend、AI、Retrieval、Storage、Infrastructure 与 Quality 技术栈；
- 本地 Quick Start、环境变量、零费用启动边界；
- Docker Compose 启动方式和未验证声明；
- 项目目录与文档导航；
- 自动化测试能够和不能够证明的内容；
- Completed、Release 1.0 与 Future Roadmap；
- Public Release Boundary 与许可证提醒。

README 不再复制 Stage 2 到 Stage 6 的全部实验过程。详细指标、失败案例、架构和状态分别链接到
RAG_EVALUATION_REPORT、FINAL_ARCHITECTURE 与 PROJECT_STATUS，减少多个事实来源漂移。

README checkpoint：6cec4c7，提交信息为 docs(release): rewrite project README。

## 3. 安全检查结果

### 3.1 Git 与历史扫描

本轮执行了以下只读检查：

- 当前 Git 已跟踪路径中的环境文件、数据库、日志、凭证文件、上传目录、Chroma 和缓存模式；
- 全部 Git 历史文件名中的敏感运行路径；
- 全部 Git 历史补丁中的高置信私钥、AWS、GitHub、OpenAI 风格、Google 与 Slack Token 模式；
- BAILIAN_API_KEY、MODEL_API_KEY、MODEL_CREDENTIAL_ENCRYPTION_KEY 和 PASSWORD 的非空
  环境赋值行。

结果：

- 高置信 Secret 模式命中 0；
- 非空凭证环境赋值命中 0；
- 历史敏感文件名只出现无真实值的 .env.example；
- 当前已跟踪运行路径扫描只出现 .env.example；
- 未读取本地 .env、真实 Key、用户资料或本地索引内容。

这是基于路径和高置信规则的仓库审查，不是完整熵扫描、GitHub Secret Scanning 或第三方安全
审计，不能解释为对所有未知 Secret 格式的数学保证。

### 3.2 Ignore 规则

.gitignore 与 .dockerignore 已增加通用保护：

- .env.*，并保留 .env.example 可跟踪；
- pem、key、p12、pfx、secrets 与 credentials；
- db、sqlite、sqlite3 及 WAL / SHM 类后缀；
- data、uploads、logs、runtime、tmp、cache、models；
- vector_store、chroma、chroma_db；
- Python、测试、覆盖率、编辑器与系统缓存。

14 个应忽略的 Secret / runtime 示例全部命中，.env.example 保持未忽略。

安全 checkpoint：5dc2292，提交信息为
security(release): review public repository safety。

### 3.3 BYOK

真实代码检查确认：

- API 请求使用 SecretStr 接收 Key；
- Key 在持久化前使用 Fernet 加密；
- 主密钥只从运行环境读取；
- StoredModelCredential 与 ModelRoute 的 repr 不展示 Key；
- Credential 响应只返回 Provider、Model、来源和时间，不含 api_key；
- 日志白名单只记录 Provider、Model 与 system/byok，不接纳 Key 字段；
- 无法解密或认证失败时不静默回退到系统 Key。

Model Gateway、API、Observability 与 Deployment 专项 25/25 通过，使用 Fake/Mock 与临时数据，
没有真实 Provider 调用。

## 4. 部署方案

新增 docs/DEPLOYMENT_GUIDE.md，并在 README 建立入口。

当前推荐部署单元是：

Browser -> HTTPS -> one FastAPI instance -> /app/data persistent volume -> model providers

推荐第一步使用单机 Docker Compose 或单实例 Docker Host / Cloud VM。原因是当前前端同源托管，
Session 使用同源 Cookie，Document Worker 在应用进程内，SQLite、Chroma 和 OperationGuard
都以单实例为一致性边界。

平台判断：

- Render Docker + Persistent Disk 与单实例边界条件匹配，但尚未实际验证；
- Railway Volume 可挂载 /app/data，但 UID 10001 的写权限需要先验证，不能静默以 root 替代；
- Vercel 适合未来独立前端，但当前直接拆分会新增 API Base URL、CORS、跨站 Cookie、CSRF 和
  Preview 域名认证契约，不属于 Release 1.0；
- Serverless Backend 不适合当前 SQLite、Chroma、持久 Worker 和本地文件架构。

部署指南还覆盖本地镜像、持久化恢复、HTTPS / Cookie、真实 Provider、有限并发、公开 Demo
访问控制、Health / Observability、回滚和 Release 1.1 验收清单。

部署指南 checkpoint：bade307，提交信息为 docs(release): add deployment guide。

## 5. Demo 准备情况

docs/DEMO_GUIDE.md 保留安全演示与完整任务演示两种模式，并补充 Release 1.0 七步索引：

1. 项目介绍；
2. 上传学习资料；
3. 知识库构建；
4. RAG 回答；
5. Citation；
6. Tutor Agent；
7. Model Gateway。

上传暂存与知识库构建被明确分开：前者是零模型本地预检，后者才可能触发 Embedding。安全演示
默认展示估算、已有 Job 和预索引资料；没有费用授权时不现场创建索引任务。

Gateway 只展示不含 Key 的元数据与受鉴权 Metrics，不虚构当前前端已有 Provider 切换控件。

本轮没有执行浏览器 Demo、真实上传入库、真实问答或 Tutor Provider 调用。

## 6. 测试结果

### 6.1 本轮全量自动化

命令：

.venv/Scripts/python.exe -B -m unittest discover -s tests -p test_*.py -q

结果：

- Ran 434 tests in 29.440s；
- OK；
- exit code 0；
- 0 failure、0 error。

测试使用 Fake/Mock、临时 SQLite 和本地文件，不调用真实 Embedding、ChatModel 或 Reranker。

### 6.2 专项与静态检查

| 检查 | 结果 |
| --- | --- |
| Model Gateway / API / Observability / Deployment | 25/25，OK |
| pip check | No broken requirements found |
| node --check web/static/app.js | 通过 |
| README 相对链接 | 全部存在 |
| Deployment Guide / Demo / README Markdown 结构 | 围栏成对、UTF-8 无控制字符 |
| Git diff --check | 通过，exit code 0 |
| Docker build/up | 未执行，当前机器没有 Docker CLI |

## 7. 当前限制

Release 1.0 完成的是发布准备，不是生产上线。当前仍缺少：

- Docker image build、Compose up、容器内页面和持久化重启证据；
- 云平台、域名、HTTPS、反向代理和 Secure Cookie 验收；
- 当前认证版本的真实 Provider / RAG / Tutor 兼容矩阵；
- 并发、负载、长时间运行、多实例一致性和容量基线；
- 完整备份恢复、主密钥轮换、外部监控、SLO 和告警；
- 公开注册的用户级费用账本、滥用防护、WAF 与渗透测试；
- 稳定 Answer Correctness、Faithfulness 与 claim-level Citation Support / Accuracy；
- 明确的 LICENSE。

最后一项是公开开源的法律边界：没有 LICENSE 时，仓库公开可见不等于他人已获得开源使用、
修改和分发授权。许可证类型需要维护者明确选择，本阶段没有擅自添加。

## 8. 下一步部署计划

### Release 1.0 后的发布门槛

1. 维护者选择 LICENSE，并确认 GitHub 仓库可见性；
2. 再做一次 GitHub 侧 Secret Scanning 与演示资料版权检查；
3. push 前确认本地 4 个 Release 提交与 final Tag；
4. 不把本地 data、Key、日志、评测输出或个人资料加入公开仓库。

### Release 1.1 Backend Deployment

1. 在有 Docker 的环境完成 build 与 Compose up；
2. 验证容器用户、Health、首页、登录和静态资源；
3. 验证 /app/data 重启持久化；
4. 完成全卷备份与恢复；
5. 在受控访问下部署单实例 Backend；
6. 明确调用数量后执行最小真实 Provider 矩阵；
7. 记录失败、成本、资源和回滚结果。

### Release 1.2 Frontend Deployment

只有在确认需要独立前端后，再设计 Vercel 拆分所需的 API Base URL、CORS、Cookie、CSRF 和
Preview / Production 域名策略。当前不应直接复制 web/ 到 Vercel 并宣称完成。

### Release 1.3 Online Demo

在访问控制、费用限制、日志脱敏、数据版权与公开安全验收通过后，再优化线上演示账号、引导、
失败恢复和观测面板。

## 9. Final Review 与 Git 边界

最终 Review 覆盖：

- 需求范围：只有 README、ignore 与 docs 发生变化；
- 核心逻辑：未修改 RAG、Agent、Workflow、Prompt、Gateway 与 API 契约；
- 文档事实：正式、实验、历史与未来能力分开；
- 安全：Secret 历史、运行路径、BYOK 响应和日志边界；
- 部署：单实例、持久卷、HTTPS、备份和平台权限风险；
- Demo：七步流程、费用授权和故障边界；
- 验证：全量测试、专项、依赖、前端语法、链接和 Git diff。

Release checkpoints：

- 6cec4c7 - docs(release): rewrite project README；
- 5dc2292 - security(release): review public repository safety；
- bade307 - docs(release): add deployment guide；
- final report commit - docs(release): finalize release preparation；
- final tag - study-material-v2-release-1.0。

Tag 在最终报告提交后创建并指向同一提交。本阶段不执行 push、部署、真实模型调用、索引变更、
数据迁移或依赖升级。
