"""Chroma Index Manifest 的创建、读取与兼容性检查。"""

import json
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal


MANIFEST_FILENAME = "index_manifest.json"
MANIFEST_VERSION = 1
DISTANCE_METRIC = "cosine"
EMBEDDING_PROVIDER = "bailian-openai-compatible"


class IndexManifestError(RuntimeError):
    """Manifest 缺失、损坏或与当前运行配置不兼容。"""


class LegacyIndexError(IndexManifestError):
    """旧索引没有 Manifest，不能执行会改变索引的操作。"""


class IncompatibleIndexError(IndexManifestError):
    """当前运行配置无法安全复用现有索引。"""


class IndexCompatibilityStatus(StrEnum):
    EMPTY = "empty"
    COMPATIBLE = "compatible"
    LEGACY_READ_ONLY = "legacy_read_only"


@dataclass(frozen=True)
class IndexRuntimeConfig:
    embedding_provider: str
    embedding_model: str
    embedding_dimension: int | None
    collection_name: str
    distance_metric: str
    parser_version: str
    cleaner_version: str
    chunker_version: str
    metadata_schema_version: str


@dataclass(frozen=True)
class IndexManifest:
    manifest_version: int
    embedding_provider: str
    embedding_model: str
    embedding_dimension: int | None
    collection_name: str
    distance_metric: str
    parser_version: str
    cleaner_version: str
    chunker_version: str
    metadata_schema_version: str
    created_at: str
    updated_at: str

    @classmethod
    def create(cls, config: IndexRuntimeConfig, *, now: str | None = None) -> "IndexManifest":
        timestamp = now or _utc_now()
        return cls(
            manifest_version=MANIFEST_VERSION,
            **asdict(config),
            created_at=timestamp,
            updated_at=timestamp,
        )


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def manifest_path(persist_directory: Path) -> Path:
    return persist_directory / MANIFEST_FILENAME


def load_index_manifest(persist_directory: Path) -> IndexManifest | None:
    path = manifest_path(persist_directory)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        manifest = IndexManifest(**payload)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise IndexManifestError("Index Manifest 无法读取或结构无效。") from error
    if manifest.manifest_version != MANIFEST_VERSION:
        raise IncompatibleIndexError("Index Manifest 版本与当前程序不兼容。")
    if (
        not manifest.embedding_provider
        or not manifest.embedding_model
        or not manifest.collection_name
        or not manifest.distance_metric
        or not manifest.parser_version
        or not manifest.cleaner_version
        or not manifest.chunker_version
        or not manifest.metadata_schema_version
        or not isinstance(manifest.created_at, str)
        or not isinstance(manifest.updated_at, str)
        or len(manifest.created_at) > 80
        or len(manifest.updated_at) > 80
        or isinstance(manifest.embedding_dimension, bool)
        or (
            manifest.embedding_dimension is not None
            and (
                not isinstance(manifest.embedding_dimension, int)
                or manifest.embedding_dimension <= 0
            )
        )
    ):
        raise IndexManifestError("Index Manifest 字段无效。")
    try:
        created_at = datetime.fromisoformat(manifest.created_at.replace("Z", "+00:00"))
        updated_at = datetime.fromisoformat(manifest.updated_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise IndexManifestError("Index Manifest 时间字段无效。") from error
    if (
        created_at.tzinfo is None
        or updated_at.tzinfo is None
        or updated_at < created_at
    ):
        raise IndexManifestError("Index Manifest 时间字段无效。")
    return manifest


def write_index_manifest(persist_directory: Path, manifest: IndexManifest) -> None:
    """同目录临时文件替换，避免只写入半份 Manifest。"""
    persist_directory.mkdir(parents=True, exist_ok=True)
    path = manifest_path(persist_directory)
    temporary_path = persist_directory / f".{MANIFEST_FILENAME}.tmp"
    payload = json.dumps(asdict(manifest), ensure_ascii=False, indent=2) + "\n"
    try:
        temporary_path.write_text(payload, encoding="utf-8")
        temporary_path.replace(path)
    except OSError as error:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise IndexManifestError("Index Manifest 无法安全写入。") from error


def _mismatched_fields(
    manifest: IndexManifest,
    config: IndexRuntimeConfig,
) -> tuple[str, ...]:
    critical_fields = (
        "embedding_provider",
        "embedding_model",
        "embedding_dimension",
        "collection_name",
        "distance_metric",
        "parser_version",
        "cleaner_version",
        "chunker_version",
        "metadata_schema_version",
    )
    return tuple(
        field_name
        for field_name in critical_fields
        if getattr(manifest, field_name) != getattr(config, field_name)
    )


def check_index_compatibility(
    persist_directory: Path,
    config: IndexRuntimeConfig,
    *,
    has_records: bool,
    access: Literal["read", "write"],
) -> IndexCompatibilityStatus:
    manifest = load_index_manifest(persist_directory)
    if manifest is None:
        if not has_records:
            return IndexCompatibilityStatus.EMPTY
        if access == "read":
            return IndexCompatibilityStatus.LEGACY_READ_ONLY
        raise LegacyIndexError(
            "现有 Chroma 是没有 Manifest 的 legacy index；禁止自动修改，"
            "请显式迁移或重新索引。"
        )
    mismatches = _mismatched_fields(manifest, config)
    if mismatches:
        raise IncompatibleIndexError(
            "当前索引与运行配置不兼容，需要显式迁移或重新索引。"
            f"不兼容字段：{', '.join(mismatches)}。"
        )
    return IndexCompatibilityStatus.COMPATIBLE


def prepare_manifest_for_write(
    persist_directory: Path,
    config: IndexRuntimeConfig,
    *,
    has_records: bool,
) -> IndexManifest:
    """写索引前先固定配置身份；legacy index 会在这里明确失败。"""
    status = check_index_compatibility(
        persist_directory,
        config,
        has_records=has_records,
        access="write",
    )
    current = load_index_manifest(persist_directory)
    if status is IndexCompatibilityStatus.EMPTY or current is None:
        manifest = IndexManifest.create(config)
    else:
        manifest = replace(current, updated_at=_utc_now())
    write_index_manifest(persist_directory, manifest)
    return manifest
