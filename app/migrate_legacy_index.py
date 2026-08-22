"""显式迁移没有 Manifest 的 legacy Chroma 索引。

默认流程只做本地读取和文件快照。两个会调用 Embedding 的阶段必须分别提供
费用确认与精确批次数；正式资料和索引只会在独立的 promote 阶段被替换。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sqlite3
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from math import ceil
from pathlib import Path
from typing import Callable, Sequence
from uuid import uuid4

from langchain_core.embeddings import Embeddings

from app.chunk_documents import (
    CLEANER_VERSION,
    CHUNKER_VERSION,
    DOCUMENTS_DIR,
    METADATA_SCHEMA_VERSION,
    PARSER_VERSION,
    PROJECT_ROOT,
    SUPPORTED_DOCUMENT_SUFFIXES,
    DocumentChunk,
    build_chunks,
    load_material_units,
)
from app.embedding_client import EmbeddingConfig
from app.index_manifest import (
    DISTANCE_METRIC,
    EMBEDDING_PROVIDER,
    IndexRuntimeConfig,
    check_index_compatibility,
    load_index_manifest,
)
from app.langchain_store import (
    BAILIAN_EMBEDDING_BATCH_SIZE,
    COLLECTION_NAME,
    VECTOR_STORE_DIR,
    close_vector_store,
    create_langchain_embeddings,
    rebuild_vector_store,
    sync_vector_store,
)
from app.material_ingestion import (
    MAX_UPLOAD_BYTES,
    PENDING_DELETIONS_DIR,
    PENDING_UPLOADS_DIR,
    MaterialManager,
    StagedMaterial,
)
from app.security_limits import (
    INDEX_MAX_BATCHES_PER_OPERATION,
    MAX_UPLOAD_FILES_PER_BATCH,
)


MIGRATION_SCHEMA_VERSION = 1
MIGRATION_ID_PATTERN = re.compile(r"[0-9a-f]{32}")
DEFAULT_MIGRATIONS_DIR = PROJECT_ROOT / "data" / "index_migrations"
PLAN_FILENAME = "plan.json"
STATE_FILENAME = "state.json"
BASE_DOCUMENTS_DIRNAME = "base_documents"
CANDIDATE_DOCUMENTS_DIRNAME = "candidate_documents"
CANDIDATE_VECTOR_STORE_DIRNAME = "candidate_vector_store"
INCREMENTAL_BACKUP_DIRNAME = "candidate_vector_store.before_incremental"
PROMOTION_BACKUP_DIRNAME = "promotion_backup"
ROLLED_BACK_CANDIDATE_DIRNAME = "rolled_back_candidate"
DEFAULT_EMBEDDING_MODEL = "text-embedding-v4"
DEFAULT_EMBEDDING_DIMENSIONS = 1024


class LegacyMigrationError(RuntimeError):
    """legacy 索引迁移无法在安全边界内继续。"""


@dataclass(frozen=True)
class MigrationLayout:
    documents_dir: Path = DOCUMENTS_DIR
    vector_store_dir: Path = VECTOR_STORE_DIR
    pending_uploads_dir: Path = PENDING_UPLOADS_DIR
    pending_deletions_dir: Path = PENDING_DELETIONS_DIR
    migrations_dir: Path = DEFAULT_MIGRATIONS_DIR


@dataclass(frozen=True)
class FileSnapshot:
    filename: str
    size_bytes: int
    sha256: str
    operation: str = "current"
    upload_id: str | None = None


@dataclass(frozen=True)
class LegacyMigrationPlan:
    schema_version: int
    migration_id: str
    created_at: str
    embedding_model: str
    embedding_dimensions: int
    legacy_record_count: int
    legacy_ids_sha256: str
    base_files: tuple[FileSnapshot, ...]
    staged_files: tuple[FileSnapshot, ...]
    base_chunk_count: int
    incremental_new_chunk_count: int
    final_chunk_count: int
    base_embedding_batches: int
    incremental_embedding_batches: int
    total_embedding_batches: int
    legacy_count_matches_base_chunks: bool


@dataclass(frozen=True)
class _PlannedSources:
    plan: LegacyMigrationPlan
    base_paths: tuple[Path, ...]
    staged: tuple[tuple[StagedMaterial, Path], ...]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(64 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ids_fingerprint(ids: Sequence[str]) -> str:
    payload = "\n".join(sorted(ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _snapshot(path: Path, *, operation: str, upload_id: str | None = None) -> FileSnapshot:
    if path.is_symlink() or path.is_junction() or not path.is_file():
        raise LegacyMigrationError(f"迁移来源不是普通文件：{path.name}。")
    return FileSnapshot(
        filename=path.name,
        size_bytes=path.stat().st_size,
        sha256=_sha256_file(path),
        operation=operation,
        upload_id=upload_id,
    )


def _runtime_config(plan: LegacyMigrationPlan) -> IndexRuntimeConfig:
    return IndexRuntimeConfig(
        embedding_provider=EMBEDDING_PROVIDER,
        embedding_model=plan.embedding_model,
        embedding_dimension=plan.embedding_dimensions,
        collection_name=COLLECTION_NAME,
        distance_metric=DISTANCE_METRIC,
        parser_version=PARSER_VERSION,
        cleaner_version=CLEANER_VERSION,
        chunker_version=CHUNKER_VERSION,
        metadata_schema_version=METADATA_SCHEMA_VERSION,
    )


def _read_record_ids(vector_store_dir: Path) -> tuple[str, ...]:
    if (
        vector_store_dir.is_symlink()
        or vector_store_dir.is_junction()
        or not vector_store_dir.is_dir()
    ):
        raise LegacyMigrationError("找不到需要迁移的 Chroma 索引目录。")
    database_path = vector_store_dir / "chroma.sqlite3"
    if not database_path.is_file():
        raise LegacyMigrationError("Chroma 索引缺少 chroma.sqlite3。")
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(
            f"file:{database_path.resolve().as_posix()}?mode=ro",
            uri=True,
        )
        connection.execute("PRAGMA query_only = ON")
        collections = connection.execute(
            "SELECT id FROM collections WHERE name = ?",
            (COLLECTION_NAME,),
        ).fetchall()
        if len(collections) != 1:
            raise LegacyMigrationError("Chroma 中找不到唯一的正式资料集合。")
        segments = connection.execute(
            "SELECT id FROM segments WHERE collection = ? AND scope = 'METADATA'",
            (collections[0][0],),
        ).fetchall()
        if len(segments) != 1:
            raise LegacyMigrationError("Chroma 正式资料集合缺少唯一 Metadata Segment。")
        rows = connection.execute(
            "SELECT embedding_id FROM embeddings WHERE segment_id = ?",
            (segments[0][0],),
        ).fetchall()
        return tuple(str(row[0]) for row in rows)
    except sqlite3.Error as error:
        raise LegacyMigrationError("无法以只读模式检查 Chroma 记录。") from error
    finally:
        if connection is not None:
            connection.close()


def _chunk_ids(chunks: Sequence[DocumentChunk]) -> tuple[str, ...]:
    ids = tuple(chunk.chunk_id for chunk in chunks)
    if len(ids) != len(set(ids)):
        raise LegacyMigrationError("资料切分生成了重复 chunk_id，迁移已停止。")
    return ids


def _current_paths(documents_dir: Path) -> tuple[Path, ...]:
    if not documents_dir.is_dir():
        raise LegacyMigrationError("找不到正式资料目录。")
    paths_list: list[Path] = []
    for path in sorted(documents_dir.iterdir(), key=lambda item: item.name):
        if path.suffix.casefold() not in SUPPORTED_DOCUMENT_SUFFIXES:
            continue
        if path.is_symlink() or path.is_junction() or not path.is_file():
            raise LegacyMigrationError(f"正式资料不是普通文件：{path.name}。")
        paths_list.append(path)
    paths = tuple(paths_list)
    if not paths:
        raise LegacyMigrationError("正式资料目录中没有可迁移的资料。")
    return paths


def discover_pending_upload_ids(pending_uploads_dir: Path) -> tuple[str, ...]:
    """只枚举合法的一层暂存目录，不读取或删除任何内容。"""
    if not pending_uploads_dir.is_dir():
        return ()
    return tuple(
        path.name
        for path in sorted(pending_uploads_dir.iterdir(), key=lambda item: item.name)
        if MIGRATION_ID_PATTERN.fullmatch(path.name)
        and path.is_dir()
        and not path.is_symlink()
        and not path.is_junction()
    )


def _load_staged(
    layout: MigrationLayout,
    upload_ids: Sequence[str],
) -> tuple[tuple[StagedMaterial, Path], ...]:
    ids = tuple(upload_ids)
    if not ids:
        raise LegacyMigrationError("至少需要选择一个已暂存资料。")
    if len(ids) > MAX_UPLOAD_FILES_PER_BATCH or len(ids) != len(set(ids)):
        raise LegacyMigrationError("暂存资料数量超限或包含重复 upload_id。")
    manager = MaterialManager(
        documents_dir=layout.documents_dir,
        pending_uploads_dir=layout.pending_uploads_dir,
        pending_deletions_dir=layout.pending_deletions_dir,
        max_file_size=MAX_UPLOAD_BYTES,
    )
    loaded: list[tuple[StagedMaterial, Path]] = []
    for upload_id in ids:
        staged = manager.inspect_staged(upload_id)
        path = layout.pending_uploads_dir / upload_id / staged.filename
        if path.is_symlink() or path.is_junction():
            raise LegacyMigrationError("暂存资料不能是符号链接或目录联接。")
        supported_paths = tuple(
            candidate
            for candidate in path.parent.iterdir()
            if candidate.suffix.casefold() in SUPPORTED_DOCUMENT_SUFFIXES
        )
        if supported_paths != (path,):
            raise LegacyMigrationError("暂存目录包含非预期的额外资料文件。")
        loaded.append((staged, path))

    filenames = [staged.filename.casefold() for staged, _ in loaded]
    if len(filenames) != len(set(filenames)):
        raise LegacyMigrationError("暂存资料包含重复文件名。")
    return tuple(loaded)


def _plan_sources(
    layout: MigrationLayout,
    upload_ids: Sequence[str],
    *,
    migration_id: str,
    embedding_model: str,
    embedding_dimensions: int,
) -> _PlannedSources:
    if MIGRATION_ID_PATTERN.fullmatch(migration_id) is None:
        raise LegacyMigrationError("migration_id 必须是 32 位小写十六进制字符串。")
    if not embedding_model.strip():
        raise LegacyMigrationError("Embedding 模型名称不能为空。")
    if (
        isinstance(embedding_dimensions, bool)
        or not isinstance(embedding_dimensions, int)
        or embedding_dimensions <= 0
    ):
        raise LegacyMigrationError("Embedding 维度必须是正整数。")

    if load_index_manifest(layout.vector_store_dir) is not None:
        raise LegacyMigrationError("当前索引已有 Manifest，不属于 legacy 迁移范围。")
    legacy_ids = _read_record_ids(layout.vector_store_dir)
    if not legacy_ids:
        raise LegacyMigrationError("当前索引没有记录，不需要执行 legacy 迁移。")

    base_paths = _current_paths(layout.documents_dir)
    staged = _load_staged(layout, upload_ids)
    base_names = {path.name.casefold() for path in base_paths}
    for item, _ in staged:
        exists = item.filename.casefold() in base_names
        if item.operation == "add" and exists:
            raise LegacyMigrationError(f"新增资料与正式资料重名：{item.filename}。")
        if item.operation == "replace" and not exists:
            raise LegacyMigrationError(f"替换目标不存在：{item.filename}。")

    base_chunks = build_chunks(load_material_units(layout.documents_dir))
    replacement_names = {
        item.filename.casefold() for item, _ in staged if item.operation == "replace"
    }
    final_chunks = [
        chunk
        for chunk in base_chunks
        if chunk.filename.casefold() not in replacement_names
    ]
    for _, path in staged:
        final_chunks.extend(build_chunks(load_material_units(path.parent)))

    base_chunk_ids = set(_chunk_ids(base_chunks))
    final_chunk_ids = set(_chunk_ids(final_chunks))
    incremental_count = len(final_chunk_ids - base_chunk_ids)
    base_batches = ceil(len(base_chunk_ids) / BAILIAN_EMBEDDING_BATCH_SIZE)
    incremental_batches = ceil(incremental_count / BAILIAN_EMBEDDING_BATCH_SIZE)
    if base_batches > INDEX_MAX_BATCHES_PER_OPERATION:
        raise LegacyMigrationError("基础索引超过单阶段 Embedding 费用保护上限。")
    if incremental_batches > INDEX_MAX_BATCHES_PER_OPERATION:
        raise LegacyMigrationError("增量索引超过单阶段 Embedding 费用保护上限。")

    base_files = tuple(_snapshot(path, operation="current") for path in base_paths)
    staged_files = tuple(
        _snapshot(path, operation=item.operation, upload_id=item.upload_id)
        for item, path in staged
    )
    plan = LegacyMigrationPlan(
        schema_version=MIGRATION_SCHEMA_VERSION,
        migration_id=migration_id,
        created_at=_utc_now(),
        embedding_model=embedding_model.strip(),
        embedding_dimensions=embedding_dimensions,
        legacy_record_count=len(legacy_ids),
        legacy_ids_sha256=_ids_fingerprint(legacy_ids),
        base_files=base_files,
        staged_files=staged_files,
        base_chunk_count=len(base_chunk_ids),
        incremental_new_chunk_count=incremental_count,
        final_chunk_count=len(final_chunk_ids),
        base_embedding_batches=base_batches,
        incremental_embedding_batches=incremental_batches,
        total_embedding_batches=base_batches + incremental_batches,
        legacy_count_matches_base_chunks=len(legacy_ids) == len(base_chunk_ids),
    )
    return _PlannedSources(plan=plan, base_paths=base_paths, staged=staged)


def plan_migration(
    layout: MigrationLayout,
    upload_ids: Sequence[str],
    *,
    migration_id: str | None = None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedding_dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
) -> LegacyMigrationPlan:
    """零费用、只读地计算迁移范围和两阶段真实批次数。"""
    return _plan_sources(
        layout,
        upload_ids,
        migration_id=migration_id or uuid4().hex,
        embedding_model=embedding_model,
        embedding_dimensions=embedding_dimensions,
    ).plan


def _write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _plan_payload(plan: LegacyMigrationPlan) -> dict[str, object]:
    return asdict(plan)


def _load_plan(workspace: Path) -> LegacyMigrationPlan:
    if workspace.is_symlink() or workspace.is_junction() or not workspace.is_dir():
        raise LegacyMigrationError("迁移工作区必须是普通目录。")
    try:
        payload = json.loads((workspace / PLAN_FILENAME).read_text(encoding="utf-8"))
        payload["base_files"] = tuple(
            FileSnapshot(**item) for item in payload["base_files"]
        )
        payload["staged_files"] = tuple(
            FileSnapshot(**item) for item in payload["staged_files"]
        )
        plan = LegacyMigrationPlan(**payload)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise LegacyMigrationError("迁移计划文件不存在或结构无效。") from error
    if (
        plan.schema_version != MIGRATION_SCHEMA_VERSION
        or MIGRATION_ID_PATTERN.fullmatch(plan.migration_id) is None
        or workspace.name != plan.migration_id
        or not plan.embedding_model
        or isinstance(plan.embedding_dimensions, bool)
        or not isinstance(plan.embedding_dimensions, int)
        or plan.embedding_dimensions <= 0
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (
                plan.legacy_record_count,
                plan.base_chunk_count,
                plan.incremental_new_chunk_count,
                plan.final_chunk_count,
                plan.base_embedding_batches,
                plan.incremental_embedding_batches,
                plan.total_embedding_batches,
            )
        )
        or not isinstance(plan.legacy_count_matches_base_chunks, bool)
        or not re.fullmatch(r"[0-9a-f]{64}", plan.legacy_ids_sha256)
        or plan.total_embedding_batches
        != plan.base_embedding_batches + plan.incremental_embedding_batches
    ):
        raise LegacyMigrationError("迁移计划版本或目录身份无效。")
    return plan


def _load_state(workspace: Path) -> dict[str, object]:
    try:
        payload = json.loads((workspace / STATE_FILENAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LegacyMigrationError("迁移状态文件不存在或结构无效。") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("phase"), str):
        raise LegacyMigrationError("迁移状态文件结构无效。")
    return payload


def _write_state(workspace: Path, phase: str, **details: object) -> None:
    _write_json(
        workspace / STATE_FILENAME,
        {"phase": phase, "updated_at": _utc_now(), **details},
    )


def _final_files(plan: LegacyMigrationPlan) -> tuple[FileSnapshot, ...]:
    files = {item.filename.casefold(): item for item in plan.base_files}
    for item in plan.staged_files:
        files[item.filename.casefold()] = FileSnapshot(
            filename=item.filename,
            size_bytes=item.size_bytes,
            sha256=item.sha256,
        )
    return tuple(sorted(files.values(), key=lambda item: item.filename))


def _validate_snapshot_directory(
    directory: Path,
    expected: Sequence[FileSnapshot],
) -> None:
    if directory.is_symlink() or directory.is_junction() or not directory.is_dir():
        raise LegacyMigrationError(f"迁移快照目录无效：{directory.name}。")
    paths = tuple(sorted(directory.iterdir(), key=lambda item: item.name))
    if any(
        path.is_symlink()
        or path.is_junction()
        or not path.is_file()
        or path.suffix.casefold() not in SUPPORTED_DOCUMENT_SUFFIXES
        for path in paths
    ):
        raise LegacyMigrationError(f"迁移快照包含非预期文件：{directory.name}。")
    expected_by_name = {item.filename: item for item in expected}
    if {path.name for path in paths} != set(expected_by_name):
        raise LegacyMigrationError(f"迁移快照文件集合发生变化：{directory.name}。")
    for path in paths:
        item = expected_by_name[path.name]
        if path.stat().st_size != item.size_bytes or _sha256_file(path) != item.sha256:
            raise LegacyMigrationError(f"迁移快照内容发生变化：{path.name}。")


def _copy_snapshot(source: Path, destination: Path) -> None:
    if source.is_symlink() or source.is_junction() or not source.is_file():
        raise LegacyMigrationError(f"迁移来源文件无效：{source.name}。")
    shutil.copy2(source, destination / source.name)


def _require_cli_workspace(workspace: Path, migrations_dir: Path) -> None:
    """限制 CLI 只能操作默认迁移根目录下的一层工作区。"""
    resolved_root = migrations_dir.resolve(strict=False)
    resolved_workspace = workspace.resolve(strict=False)
    if resolved_workspace.parent != resolved_root:
        raise LegacyMigrationError("迁移工作区超出 data/index_migrations 安全范围。")


def prepare_migration(
    layout: MigrationLayout,
    upload_ids: Sequence[str],
    *,
    migration_id: str | None = None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedding_dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
) -> Path:
    """只创建正式资料的不可变快照；暂存资料留到第二阶段再复制。"""
    planned = _plan_sources(
        layout,
        upload_ids,
        migration_id=migration_id or uuid4().hex,
        embedding_model=embedding_model,
        embedding_dimensions=embedding_dimensions,
    )
    if layout.migrations_dir.exists() and (
        layout.migrations_dir.is_symlink()
        or layout.migrations_dir.is_junction()
        or not layout.migrations_dir.is_dir()
    ):
        raise LegacyMigrationError("迁移工作区根目录无效。")
    layout.migrations_dir.mkdir(parents=True, exist_ok=True)
    workspace = layout.migrations_dir / planned.plan.migration_id
    if workspace.resolve(strict=False).parent != layout.migrations_dir.resolve():
        raise LegacyMigrationError("迁移工作区超出允许的根目录。")
    if workspace.exists():
        raise LegacyMigrationError("同名迁移工作区已经存在。")
    try:
        base_documents = workspace / BASE_DOCUMENTS_DIRNAME
        base_documents.mkdir(parents=True, exist_ok=False)
        for source in planned.base_paths:
            _copy_snapshot(source, base_documents)
        _validate_snapshot_directory(base_documents, planned.plan.base_files)
        _write_json(workspace / PLAN_FILENAME, _plan_payload(planned.plan))
        _write_state(workspace, "prepared")
    except Exception:
        shutil.rmtree(workspace, ignore_errors=True)
        raise
    return workspace


def prepare_staged_snapshot(workspace: Path, layout: MigrationLayout) -> None:
    """第二阶段前才固化暂存资料；不调用 Embedding 或修改正式资料。"""
    plan = _validate_prepared_workspace(workspace)
    if _load_state(workspace).get("phase") != "base_built":
        raise LegacyMigrationError("暂存资料快照只能从 base_built 状态开始。")
    candidate_documents = workspace / CANDIDATE_DOCUMENTS_DIRNAME
    if candidate_documents.exists():
        raise LegacyMigrationError("候选资料快照目录已经存在，拒绝覆盖。")
    upload_ids = tuple(
        item.upload_id
        for item in plan.staged_files
        if isinstance(item.upload_id, str)
    )
    if len(upload_ids) != len(plan.staged_files):
        raise LegacyMigrationError("迁移计划缺少暂存资料 upload_id。")
    loaded = _load_staged(layout, upload_ids)
    expected_staged = {item.upload_id: item for item in plan.staged_files}
    for staged, source in loaded:
        expected = expected_staged.get(staged.upload_id)
        actual = _snapshot(
            source,
            operation=staged.operation,
            upload_id=staged.upload_id,
        )
        if actual != expected:
            raise LegacyMigrationError(f"暂存资料发生变化：{staged.filename}。")

    try:
        candidate_documents.mkdir(exist_ok=False)
        for source in sorted(
            (workspace / BASE_DOCUMENTS_DIRNAME).iterdir(),
            key=lambda item: item.name,
        ):
            _copy_snapshot(source, candidate_documents)
        for _, source in loaded:
            destination = candidate_documents / source.name
            if destination.exists():
                destination.unlink()
            _copy_snapshot(source, candidate_documents)
        _validate_snapshot_directory(candidate_documents, _final_files(plan))
        _write_state(workspace, "staged_snapshot_ready")
    except Exception:
        shutil.rmtree(candidate_documents, ignore_errors=True)
        raise


def _validate_prepared_workspace(workspace: Path) -> LegacyMigrationPlan:
    plan = _load_plan(workspace)
    _validate_snapshot_directory(
        workspace / BASE_DOCUMENTS_DIRNAME,
        plan.base_files,
    )
    candidate_documents = workspace / CANDIDATE_DOCUMENTS_DIRNAME
    if candidate_documents.exists():
        _validate_snapshot_directory(candidate_documents, _final_files(plan))
    return plan


def _snapshot_chunks(workspace: Path, directory_name: str) -> list[DocumentChunk]:
    chunks = build_chunks(load_material_units(workspace / directory_name))
    _chunk_ids(chunks)
    return chunks


def _require_paid_confirmation(
    expected_batches: int,
    *,
    confirm_api_cost: bool,
    confirm_batches: int,
) -> None:
    if not confirm_api_cost or confirm_batches != expected_batches:
        raise LegacyMigrationError(
            "费用确认不完整：必须显式确认 API 费用并填写本阶段精确批次数。"
        )


def _load_paid_embeddings(
    plan: LegacyMigrationPlan,
    config_loader: Callable[[], EmbeddingConfig],
    embedding_factory: Callable[[EmbeddingConfig], Embeddings],
) -> Embeddings:
    config = config_loader()
    if (
        config.model != plan.embedding_model
        or config.dimensions != plan.embedding_dimensions
    ):
        raise LegacyMigrationError("当前 Embedding 配置与迁移计划不一致。")
    return embedding_factory(config)


def _validate_candidate_ids(workspace: Path, expected_ids: set[str]) -> None:
    actual_ids = set(_read_record_ids(workspace / CANDIDATE_VECTOR_STORE_DIRNAME))
    if actual_ids != expected_ids:
        raise LegacyMigrationError("候选索引记录与资料快照不一致。")


def build_base_candidate(
    workspace: Path,
    *,
    confirm_api_cost: bool,
    confirm_batches: int,
    config_loader: Callable[[], EmbeddingConfig] = EmbeddingConfig.from_environment,
    embedding_factory: Callable[[EmbeddingConfig], Embeddings] = create_langchain_embeddings,
) -> None:
    """付费阶段一：只在空候选目录中重建当前正式资料。"""
    plan = _validate_prepared_workspace(workspace)
    if _load_state(workspace).get("phase") != "prepared":
        raise LegacyMigrationError("基础候选索引只能从 prepared 状态开始。")
    _require_paid_confirmation(
        plan.base_embedding_batches,
        confirm_api_cost=confirm_api_cost,
        confirm_batches=confirm_batches,
    )
    chunks = _snapshot_chunks(workspace, BASE_DOCUMENTS_DIRNAME)
    expected_ids = set(_chunk_ids(chunks))
    actual_batches = ceil(len(expected_ids) / BAILIAN_EMBEDDING_BATCH_SIZE)
    if (
        len(expected_ids) != plan.base_chunk_count
        or actual_batches != plan.base_embedding_batches
        or actual_batches > INDEX_MAX_BATCHES_PER_OPERATION
    ):
        raise LegacyMigrationError("基础资料切分结果与迁移计划不一致。")
    candidate = workspace / CANDIDATE_VECTOR_STORE_DIRNAME
    if candidate.exists():
        raise LegacyMigrationError("候选索引目录已经存在，拒绝覆盖。")

    embeddings = _load_paid_embeddings(plan, config_loader, embedding_factory)
    vector_store = None
    try:
        vector_store = rebuild_vector_store(
            chunks,
            embeddings,
            candidate,
            runtime_config=_runtime_config(plan),
        )
        close_vector_store(vector_store)
        vector_store = None
        _validate_candidate_ids(workspace, expected_ids)
        _write_state(
            workspace,
            "base_built",
            candidate_record_count=len(expected_ids),
        )
    except Exception:
        if vector_store is not None:
            close_vector_store(vector_store)
        shutil.rmtree(candidate, ignore_errors=True)
        raise


def add_staged_to_candidate(
    workspace: Path,
    *,
    confirm_api_cost: bool,
    confirm_batches: int,
    config_loader: Callable[[], EmbeddingConfig] = EmbeddingConfig.from_environment,
    embedding_factory: Callable[[EmbeddingConfig], Embeddings] = create_langchain_embeddings,
) -> None:
    """付费阶段二：在已验证的基础候选索引上加入暂存资料。"""
    plan = _validate_prepared_workspace(workspace)
    if _load_state(workspace).get("phase") != "staged_snapshot_ready":
        raise LegacyMigrationError(
            "增量候选索引只能从 staged_snapshot_ready 状态开始。"
        )
    _require_paid_confirmation(
        plan.incremental_embedding_batches,
        confirm_api_cost=confirm_api_cost,
        confirm_batches=confirm_batches,
    )
    base_chunks = _snapshot_chunks(workspace, BASE_DOCUMENTS_DIRNAME)
    final_chunks = _snapshot_chunks(workspace, CANDIDATE_DOCUMENTS_DIRNAME)
    base_ids = set(_chunk_ids(base_chunks))
    final_ids = set(_chunk_ids(final_chunks))
    incremental_count = len(final_ids - base_ids)
    actual_batches = ceil(incremental_count / BAILIAN_EMBEDDING_BATCH_SIZE)
    if (
        len(final_ids) != plan.final_chunk_count
        or incremental_count != plan.incremental_new_chunk_count
        or actual_batches != plan.incremental_embedding_batches
        or actual_batches > INDEX_MAX_BATCHES_PER_OPERATION
    ):
        raise LegacyMigrationError("候选资料切分结果与迁移计划不一致。")
    _validate_candidate_ids(workspace, base_ids)

    candidate = workspace / CANDIDATE_VECTOR_STORE_DIRNAME
    backup = workspace / INCREMENTAL_BACKUP_DIRNAME
    if backup.exists():
        raise LegacyMigrationError("发现未处理的增量备份，拒绝继续。")
    shutil.copytree(candidate, backup)
    vector_store = None
    try:
        embeddings = _load_paid_embeddings(plan, config_loader, embedding_factory)
        result = sync_vector_store(
            final_chunks,
            embeddings,
            candidate,
            runtime_config=_runtime_config(plan),
        )
        vector_store = result.vector_store
        close_vector_store(vector_store)
        vector_store = None
        _validate_candidate_ids(workspace, final_ids)
        _write_state(
            workspace,
            "candidate_ready",
            candidate_record_count=len(final_ids),
        )
        try:
            shutil.rmtree(backup)
        except OSError:
            # 候选索引和成功状态已经持久化。残留备份只占用空间，不能把一次
            # 清理失败重新解释成索引失败，否则会破坏已经验证的候选版本。
            pass
    except Exception:
        if vector_store is not None:
            close_vector_store(vector_store)
        shutil.rmtree(candidate, ignore_errors=True)
        if backup.exists():
            backup.replace(candidate)
        raise


def validate_candidate(workspace: Path) -> LegacyMigrationPlan:
    """零费用验证候选资料、Manifest 与 Chroma 记录集合。"""
    plan = _validate_prepared_workspace(workspace)
    if _load_state(workspace).get("phase") != "candidate_ready":
        raise LegacyMigrationError("候选索引尚未完成两个构建阶段。")
    final_chunks = _snapshot_chunks(workspace, CANDIDATE_DOCUMENTS_DIRNAME)
    _validate_candidate_ids(workspace, set(_chunk_ids(final_chunks)))
    manifest = load_index_manifest(workspace / CANDIDATE_VECTOR_STORE_DIRNAME)
    if manifest is None:
        raise LegacyMigrationError("候选索引 Manifest 与迁移计划不一致。")
    try:
        check_index_compatibility(
            workspace / CANDIDATE_VECTOR_STORE_DIRNAME,
            _runtime_config(plan),
            has_records=True,
            access="read",
        )
    except Exception as error:
        raise LegacyMigrationError(
            "候选索引 Manifest 与迁移计划不一致。"
        ) from error
    return plan


def _validate_production_unchanged(layout: MigrationLayout, plan: LegacyMigrationPlan) -> None:
    _validate_snapshot_directory(layout.documents_dir, plan.base_files)
    if load_index_manifest(layout.vector_store_dir) is not None:
        raise LegacyMigrationError("正式索引已不再是计划中的 legacy 状态。")
    ids = _read_record_ids(layout.vector_store_dir)
    if (
        len(ids) != plan.legacy_record_count
        or _ids_fingerprint(ids) != plan.legacy_ids_sha256
    ):
        raise LegacyMigrationError("正式索引在准备迁移后发生了变化。")


def promote_candidate(
    workspace: Path,
    layout: MigrationLayout,
    *,
    confirm_service_stopped: bool,
    confirm_migration_id: str,
) -> Path:
    """零 API 调用地替换正式资料和索引，并保留完整备份。"""
    plan = validate_candidate(workspace)
    if not confirm_service_stopped or confirm_migration_id != plan.migration_id:
        raise LegacyMigrationError("提升确认不完整：必须停服并填写 migration_id。")
    _validate_production_unchanged(layout, plan)

    candidate_documents = workspace / CANDIDATE_DOCUMENTS_DIRNAME
    candidate_vector = workspace / CANDIDATE_VECTOR_STORE_DIRNAME
    backup_root = workspace / PROMOTION_BACKUP_DIRNAME
    backup_documents = backup_root / "documents"
    backup_vector = backup_root / "vector_store"
    if backup_root.exists():
        raise LegacyMigrationError("迁移备份目录已经存在，拒绝覆盖。")
    moved: list[tuple[Path, Path]] = []
    try:
        backup_root.mkdir()
        _write_state(workspace, "promoting")
        layout.documents_dir.replace(backup_documents)
        moved.append((backup_documents, layout.documents_dir))
        layout.vector_store_dir.replace(backup_vector)
        moved.append((backup_vector, layout.vector_store_dir))
        candidate_documents.replace(layout.documents_dir)
        moved.append((layout.documents_dir, candidate_documents))
        candidate_vector.replace(layout.vector_store_dir)
        moved.append((layout.vector_store_dir, candidate_vector))
        _write_state(
            workspace,
            "promoted",
            backup_directory=str(backup_root),
        )
    except Exception as error:
        rollback_errors: list[Exception] = []
        for source, destination in reversed(moved):
            try:
                if source.exists():
                    source.replace(destination)
            except OSError as rollback_error:
                rollback_errors.append(rollback_error)
        if not rollback_errors:
            _write_state(workspace, "candidate_ready")
            try:
                backup_root.rmdir()
            except OSError:
                pass
        if rollback_errors:
            raise LegacyMigrationError(
                "正式索引提升失败，且自动回滚未能完整完成。"
            ) from rollback_errors[0]
        raise LegacyMigrationError("正式索引提升失败，已恢复提升前状态。") from error
    return backup_root


def rollback_promotion(
    workspace: Path,
    layout: MigrationLayout,
    *,
    confirm_service_stopped: bool,
    confirm_migration_id: str,
) -> None:
    """把已提升版本移回迁移工作区，并恢复 promotion_backup。"""
    plan = _load_plan(workspace)
    if _load_state(workspace).get("phase") != "promoted":
        raise LegacyMigrationError("只有 promoted 状态可以执行回滚。")
    if not confirm_service_stopped or confirm_migration_id != plan.migration_id:
        raise LegacyMigrationError("回滚确认不完整：必须停服并填写 migration_id。")
    backup_root = workspace / PROMOTION_BACKUP_DIRNAME
    backup_documents = backup_root / "documents"
    backup_vector = backup_root / "vector_store"
    rolled_back_root = workspace / ROLLED_BACK_CANDIDATE_DIRNAME
    if (
        rolled_back_root.exists()
        or not backup_documents.exists()
        or not backup_vector.exists()
    ):
        raise LegacyMigrationError("回滚所需目录缺失或目标目录已存在。")
    _validate_snapshot_directory(layout.documents_dir, _final_files(plan))
    current_ids = _read_record_ids(layout.vector_store_dir)
    expected_current_ids = set(
        _chunk_ids(build_chunks(load_material_units(layout.documents_dir)))
    )
    if set(current_ids) != expected_current_ids:
        raise LegacyMigrationError("已提升的正式索引发生变化，拒绝自动回滚。")
    try:
        check_index_compatibility(
            layout.vector_store_dir,
            _runtime_config(plan),
            has_records=bool(current_ids),
            access="read",
        )
    except Exception as error:
        raise LegacyMigrationError("已提升的正式索引配置发生变化。") from error
    _validate_snapshot_directory(backup_documents, plan.base_files)
    if load_index_manifest(backup_vector) is not None:
        raise LegacyMigrationError("提升前备份不再是 legacy 索引。")
    backup_ids = _read_record_ids(backup_vector)
    if (
        len(backup_ids) != plan.legacy_record_count
        or _ids_fingerprint(backup_ids) != plan.legacy_ids_sha256
    ):
        raise LegacyMigrationError("提升前的索引备份发生变化。")
    rolled_back_root.mkdir()
    moved: list[tuple[Path, Path]] = []
    try:
        layout.documents_dir.replace(rolled_back_root / "documents")
        moved.append((rolled_back_root / "documents", layout.documents_dir))
        layout.vector_store_dir.replace(rolled_back_root / "vector_store")
        moved.append((rolled_back_root / "vector_store", layout.vector_store_dir))
        backup_documents.replace(layout.documents_dir)
        moved.append((layout.documents_dir, backup_documents))
        backup_vector.replace(layout.vector_store_dir)
        moved.append((layout.vector_store_dir, backup_vector))
        _write_state(workspace, "rolled_back")
    except Exception as error:
        rollback_errors: list[Exception] = []
        for source, destination in reversed(moved):
            try:
                if source.exists():
                    source.replace(destination)
            except OSError as rollback_error:
                rollback_errors.append(rollback_error)
        if rollback_errors:
            raise LegacyMigrationError(
                "迁移回滚失败，且恢复 promoted 状态也未完整完成。"
            ) from rollback_errors[0]
        raise LegacyMigrationError("迁移回滚失败，已恢复 promoted 状态。") from error


def _add_selection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--upload-id", action="append", default=[])
    parser.add_argument("--all-pending", action="store_true")
    parser.add_argument("--migration-id")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument(
        "--embedding-dimensions",
        type=int,
        default=DEFAULT_EMBEDDING_DIMENSIONS,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="安全地把 legacy Chroma 迁移到带 Manifest 的候选索引。"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan_parser = subparsers.add_parser("plan", help="只读 dry-run，不写文件、不调用 API")
    _add_selection_arguments(plan_parser)
    prepare_parser = subparsers.add_parser(
        "prepare",
        help="只创建现有正式资料的本地不可变快照",
    )
    _add_selection_arguments(prepare_parser)

    staged_snapshot_parser = subparsers.add_parser(
        "snapshot-staged",
        help="第二阶段前才复制暂存资料，不调用 API",
    )
    staged_snapshot_parser.add_argument("workspace", type=Path)

    for command, help_text in (
        ("build-base", "付费阶段一：构建基础候选索引"),
        ("add-staged", "付费阶段二：把暂存资料加入候选索引"),
    ):
        command_parser = subparsers.add_parser(command, help=help_text)
        command_parser.add_argument("workspace", type=Path)
        command_parser.add_argument("--confirm-api-cost", action="store_true")
        command_parser.add_argument("--confirm-batches", type=int, required=True)

    validate_parser = subparsers.add_parser("validate", help="零费用校验完整候选索引")
    validate_parser.add_argument("workspace", type=Path)
    for command, help_text in (
        ("promote", "停服后提升候选索引并备份正式版本"),
        ("rollback", "停服后恢复提升前的正式版本"),
    ):
        command_parser = subparsers.add_parser(command, help=help_text)
        command_parser.add_argument("workspace", type=Path)
        command_parser.add_argument("--confirm-service-stopped", action="store_true")
        command_parser.add_argument("--confirm-migration-id", required=True)
    return parser


def _selected_upload_ids(args: argparse.Namespace, layout: MigrationLayout) -> tuple[str, ...]:
    explicit = tuple(args.upload_id)
    if args.all_pending and explicit:
        raise LegacyMigrationError("--all-pending 与 --upload-id 不能同时使用。")
    ids = (
        discover_pending_upload_ids(layout.pending_uploads_dir)
        if args.all_pending
        else explicit
    )
    if not ids:
        raise LegacyMigrationError("没有选择任何暂存资料。")
    return ids


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    layout = MigrationLayout()
    try:
        if args.command not in {"plan", "prepare"}:
            _require_cli_workspace(args.workspace, layout.migrations_dir)
        if args.command in {"plan", "prepare"}:
            ids = _selected_upload_ids(args, layout)
            options = {
                "migration_id": args.migration_id,
                "embedding_model": args.embedding_model,
                "embedding_dimensions": args.embedding_dimensions,
            }
            if args.command == "plan":
                plan = plan_migration(layout, ids, **options)
                print(json.dumps(asdict(plan), ensure_ascii=False, indent=2))
            else:
                workspace = prepare_migration(layout, ids, **options)
                print(f"迁移快照已准备：{workspace}")
        elif args.command == "build-base":
            build_base_candidate(
                args.workspace,
                confirm_api_cost=args.confirm_api_cost,
                confirm_batches=args.confirm_batches,
            )
            print("基础候选索引已完成。")
        elif args.command == "snapshot-staged":
            prepare_staged_snapshot(args.workspace, layout)
            print("暂存资料快照已准备。")
        elif args.command == "add-staged":
            add_staged_to_candidate(
                args.workspace,
                confirm_api_cost=args.confirm_api_cost,
                confirm_batches=args.confirm_batches,
            )
            print("增量候选索引已完成。")
        elif args.command == "validate":
            plan = validate_candidate(args.workspace)
            print(f"候选索引验证通过：{plan.final_chunk_count} 条记录。")
        elif args.command == "promote":
            backup = promote_candidate(
                args.workspace,
                layout,
                confirm_service_stopped=args.confirm_service_stopped,
                confirm_migration_id=args.confirm_migration_id,
            )
            print(f"候选索引已提升；原版本备份：{backup}")
        elif args.command == "rollback":
            rollback_promotion(
                args.workspace,
                layout,
                confirm_service_stopped=args.confirm_service_stopped,
                confirm_migration_id=args.confirm_migration_id,
            )
            print("已恢复提升前的正式资料与索引。")
    except Exception as error:
        print(f"legacy 索引迁移失败：{error}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
