# Legacy Index Migration Runbook

本文用于把“已有 Chroma 记录但缺少 `index_manifest.json`”的 legacy index，显式迁移为与当前 Parser、Chunker、Metadata 和 Embedding 配置一致的新索引。

这不是自动迁移或普通启动步骤。两个候选构建阶段可能调用真实 Embedding API；只有 `promote` 会替换正式资料目录和正式索引。

## 适用范围

仅在以下条件同时满足时使用：

- `data/vector_store/chroma.sqlite3` 有记录，但没有 `index_manifest.json`；
- 正式资料仍位于 `data/documents/`；
- 至少有一份经现有上传接口暂存、但因 legacy 只读保护无法入库的资料；
- 继续使用当前 Parser、Chunker、Metadata Schema 和 Embedding 配置；
- 最终提升和回滚时能够停止本地服务。

如果当前索引已有 Manifest、正式资料不完整、Embedding 配置来源不明，或需要跨磁盘、多实例、在线迁移，不要使用本工具。

## 安全边界

- 不读取或输出 `.env`；付费阶段只通过现有配置入口加载 Embedding 配置。
- `plan` 只读取正式资料、暂存摘要和 Chroma SQLite；不创建迁移目录，不调用外部 API。
- `prepare` 只复制正式资料快照；不移动正式文件，不调用外部 API。
- `build-base` 与 `add-staged` 分别要求 `--confirm-api-cost` 和本阶段精确批次数。
- `promote` 与 `rollback` 不调用外部 API，但必须停服并填写完整 `migration_id`。
- CLI 只接受 `data/index_migrations/<migration_id>` 下由 `prepare` 创建的一层工作区。
- 确认稳定前不要手工删除迁移工作区、候选索引或 `promotion_backup/`。

## 状态流转

```text
plan（只读）
  ↓ prepare
prepared
  ↓ build-base（付费阶段一）
base_built
  ↓ snapshot-staged（零费用）
staged_snapshot_ready
  ↓ add-staged（付费阶段二）
candidate_ready
  ↓ validate（零费用）
  ↓ promote（停服）
promoted
  ↓ rollback（可选，停服且正式版本未变化）
rolled_back
```

任一步不符合当前状态都会拒绝继续。不要手工编辑 `plan.json`、`state.json`、候选索引或快照文件来跳过检查。

## 1. 开始前检查

在项目根目录使用项目 `.venv`：

```powershell
git status --short --branch
git rev-parse HEAD
```

记录当前 commit、明确的 `upload_id`、备份策略、可用磁盘空间和服务状态。除非已经逐个核对，不要用 `--all-pending` 代替显式 `--upload-id`。

## 2. 生成只读计划

固定一次 migration ID，并在 `plan` 与 `prepare` 中复用：

```powershell
$migrationId = [guid]::NewGuid().ToString("N")
python -B -m app.migrate_legacy_index plan `
  --upload-id "<UPLOAD_ID>" `
  --migration-id $migrationId `
  --embedding-model "text-embedding-v4" `
  --embedding-dimensions 1024
```

逐项核对 `legacy_record_count`、文件快照、Chunk 数、两个阶段的批次数和 `legacy_count_matches_base_chunks`。

`legacy_count_matches_base_chunks=false` 不会自动阻断，因为候选索引基于当前正式资料重建；但它说明旧记录数与当前切分结果不同，必须先解释原因。

## 3. 创建基础资料快照

```powershell
python -B -m app.migrate_legacy_index prepare `
  --upload-id "<UPLOAD_ID>" `
  --migration-id $migrationId `
  --embedding-model "text-embedding-v4" `
  --embedding-dimensions 1024
$workspace = "data/index_migrations/$migrationId"
```

后续命令必须使用这个精确工作区；不要复制、重命名或移动它。

## 4. 构建基础候选索引

只在接受第一阶段费用后执行：

```powershell
python -B -m app.migrate_legacy_index build-base $workspace `
  --confirm-api-cost `
  --confirm-batches <BASE_EMBEDDING_BATCHES>
```

批次数必须与计划完全一致。程序会重新解析快照并重新计算批次数，然后才加载付费配置。

## 5. 固化暂存资料并构建最终候选

```powershell
python -B -m app.migrate_legacy_index snapshot-staged $workspace
python -B -m app.migrate_legacy_index add-staged $workspace `
  --confirm-api-cost `
  --confirm-batches <INCREMENTAL_EMBEDDING_BATCHES>
```

第二条命令也必须单独确认费用。增量失败会恢复已验证的基础候选；清理冗余备份失败不会破坏已经验证的最终候选。

## 6. 零费用校验

```powershell
python -B -m app.migrate_legacy_index validate $workspace
```

该命令验证资料 SHA-256、候选 Chroma ID 集合和 Index Manifest 兼容性。它不证明 Retrieval、Citation 或回答质量。

## 7. 停服并提升

停止所有可能访问正式资料、Chroma 或后台 Document Job 的进程，并确认端口监听已经消失。

```powershell
python -B -m app.migrate_legacy_index promote $workspace `
  --confirm-service-stopped `
  --confirm-migration-id $migrationId
```

提升会再次验证正式资料和 legacy ID 指纹，把旧版本移入 `promotion_backup/`，再把候选版本移动到正式路径。

普通异常会尝试自动恢复。强制终止、断电或磁盘故障可能发生在目录移动之间；此时不要重跑命令或删除目录，先记录 `state.json`、正式路径和工作区各目录的实际状态，再按备份关系人工恢复。

## 8. 提升后验证

重新启动服务后先做零费用检查：

- `GET /health` 返回 200；
- 资料列表与预期文件集合一致；
- 本地索引兼容性检查通过。

真实问答或新增资料验收需要另行确认 Query Embedding、ChatModel 或新增 Embedding 费用。

## 9. 回滚

只有状态仍为 `promoted`、提升后的正式资料和索引未变化、`promotion_backup/` 完整且服务已停止时，才能自动回滚：

```powershell
python -B -m app.migrate_legacy_index rollback $workspace `
  --confirm-service-stopped `
  --confirm-migration-id $migrationId
```

回滚后，新候选保存在 `rolled_back_candidate/`，原 legacy 版本恢复到正式路径。提升后若已经发生资料写入，自动回滚会拒绝继续。

## 未覆盖能力

- 不支持没有暂存资料的纯 legacy 重建；
- 不支持跨磁盘原子提升；
- 不提供多进程锁、在线迁移或多实例协调；
- 不自动清理历史工作区和备份；
- 不验证 Retrieval、Citation 或回答质量；
- 不替代项目外部备份与断电恢复方案。
