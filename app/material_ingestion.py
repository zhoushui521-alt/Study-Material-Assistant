"""安全暂存学习资料，并复用现有解析与 Chroma 增量索引。"""

import hashlib
import json
import re
import shutil
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from io import BufferedIOBase
from math import ceil
from pathlib import Path
from typing import BinaryIO, Literal
from uuid import uuid4

if __package__:
    from app.chunk_documents import (
        DOCUMENTS_DIR,
        PROJECT_ROOT,
        SUPPORTED_DOCUMENT_SUFFIXES,
        DocumentChunk,
        DocumentLoadError,
        build_chunks,
        load_material_units,
    )
    from app.embedding_client import EmbeddingConfig
    from app.langchain_store import (
        BAILIAN_EMBEDDING_BATCH_SIZE,
        close_vector_store,
        create_langchain_embeddings,
        delete_material_documents,
        estimate_vector_store_sync_batches,
        runtime_index_config,
        source_belongs_to_material,
        sync_vector_store,
    )
    from app.index_manifest import IndexManifestError
    from app.security_limits import (
        INDEX_MAX_BATCHES_PER_OPERATION,
        MAX_UPLOAD_EXTRACTED_CHARACTERS,
        MAX_UPLOAD_PDF_PAGES,
        PENDING_UPLOAD_TTL_SECONDS,
    )
else:
    from chunk_documents import (
        DOCUMENTS_DIR,
        PROJECT_ROOT,
        SUPPORTED_DOCUMENT_SUFFIXES,
        DocumentChunk,
        DocumentLoadError,
        build_chunks,
        load_material_units,
    )
    from embedding_client import EmbeddingConfig
    from langchain_store import (
        BAILIAN_EMBEDDING_BATCH_SIZE,
        close_vector_store,
        create_langchain_embeddings,
        delete_material_documents,
        estimate_vector_store_sync_batches,
        runtime_index_config,
        source_belongs_to_material,
        sync_vector_store,
    )
    from index_manifest import IndexManifestError
    from security_limits import (
        INDEX_MAX_BATCHES_PER_OPERATION,
        MAX_UPLOAD_EXTRACTED_CHARACTERS,
        MAX_UPLOAD_PDF_PAGES,
        PENDING_UPLOAD_TTL_SECONDS,
    )


MAX_UPLOAD_BYTES = 10 * 1024 * 1024
UPLOAD_READ_SIZE = 64 * 1024
MAX_FILENAME_LENGTH = 120
PENDING_UPLOADS_DIR = PROJECT_ROOT / "data" / "pending_uploads"
PENDING_DELETIONS_DIR = PROJECT_ROOT / "data" / "pending_deletions"
UPLOAD_METADATA_FILENAME = "upload.json"
UPLOAD_ID_PATTERN = re.compile(r"[0-9a-f]{32}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
WINDOWS_INVALID_FILENAME_CHARACTERS = frozenset('<>:"/\\|?*')
WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
}
ALLOWED_CONTENT_TYPES = {
    ".txt": {"text/plain", "application/octet-stream"},
    ".md": {"text/markdown", "text/plain", "application/octet-stream"},
    ".pdf": {"application/pdf", "application/octet-stream"},
}

UploadOperation = Literal["add", "replace"]


class MaterialIngestionError(RuntimeError):
    """资料上传、索引或删除无法安全完成。"""


class MaterialValidationError(MaterialIngestionError):
    """上传文件名、类型或内容不符合要求。"""


class MaterialTooLargeError(MaterialValidationError):
    """上传文件超过大小限制。"""


class MaterialConflictError(MaterialIngestionError):
    """新增或替换操作与当前资料状态冲突。"""


class MaterialNotFoundError(MaterialIngestionError):
    """暂存上传或正式资料不存在。"""


class MaterialIndexError(MaterialIngestionError):
    """资料文件与 Chroma 索引未能同步。"""


class MaterialRollbackError(MaterialIndexError):
    """索引失败后的补偿回滚也未能完成。"""


@dataclass(frozen=True)
class StagedMaterial:
    upload_id: str
    filename: str
    operation: UploadOperation
    size_bytes: int
    sha256: str
    document_units: int
    chunk_count: int
    embedding_batch_count: int
    staged_at: str


@dataclass(frozen=True)
class IndexSyncSummary:
    added: int
    deleted: int
    unchanged: int


@dataclass(frozen=True)
class MaterialSyncResult:
    filename: str
    operation: UploadOperation
    added: int
    deleted: int
    unchanged: int
    cleanup_pending: bool


@dataclass(frozen=True)
class MaterialFile:
    filename: str
    size_bytes: int


@dataclass(frozen=True)
class MaterialDeleteResult:
    filename: str
    deleted_records: int
    cleanup_pending: bool


def validate_material_filename(filename: str | None) -> str:
    """只接受安全的单层 Windows 文件名，并保留可核对的原始来源名。"""
    if not isinstance(filename, str) or not filename.strip():
        raise MaterialValidationError("上传文件必须包含有效文件名。")
    if filename != filename.strip():
        raise MaterialValidationError("文件名首尾不能包含空白字符。")
    normalized = filename
    if len(normalized) > MAX_FILENAME_LENGTH:
        raise MaterialValidationError(
            f"文件名不能超过 {MAX_FILENAME_LENGTH} 个字符。"
        )
    if normalized != Path(normalized).name or any(
        character in WINDOWS_INVALID_FILENAME_CHARACTERS for character in normalized
    ):
        raise MaterialValidationError("文件名包含路径或 Windows 不支持的字符。")
    if normalized.endswith((".", " ")) or any(
        ord(character) < 32 for character in normalized
    ):
        raise MaterialValidationError("文件名包含 Windows 不支持的字符。")
    if normalized.split(".", maxsplit=1)[0].upper() in WINDOWS_RESERVED_NAMES:
        raise MaterialValidationError("文件名使用了 Windows 保留名称。")
    suffix = Path(normalized).suffix.lower()
    if suffix not in SUPPORTED_DOCUMENT_SUFFIXES:
        supported = "、".join(sorted(SUPPORTED_DOCUMENT_SUFFIXES))
        raise MaterialValidationError(f"只支持以下文件类型：{supported}。")
    return normalized


def _validate_content_type(filename: str, content_type: str | None) -> None:
    suffix = Path(filename).suffix.lower()
    normalized_type = (content_type or "application/octet-stream").split(
        ";", maxsplit=1
    )[0].strip().lower()
    if normalized_type not in ALLOWED_CONTENT_TYPES[suffix]:
        raise MaterialValidationError(
            f"文件声明的内容类型与 {suffix} 扩展名不匹配。"
        )


def _validate_file_signature(path: Path) -> None:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        with path.open("rb") as file:
            if file.read(5) != b"%PDF-":
                raise MaterialValidationError("PDF 文件签名无效。")
        return

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise MaterialValidationError("TXT 和 Markdown 必须使用 UTF-8 编码。") from error
    if "\x00" in text:
        raise MaterialValidationError("文本文件包含不允许的二进制空字符。")


def _write_upload_stream(
    stream: BinaryIO | BufferedIOBase,
    destination: Path,
    max_file_size: int,
) -> tuple[int, str]:
    digest = hashlib.sha256()
    total_bytes = 0
    with destination.open("xb") as output:
        while True:
            block = stream.read(UPLOAD_READ_SIZE)
            if not block:
                break
            total_bytes += len(block)
            if total_bytes > max_file_size:
                raise MaterialTooLargeError(
                    f"单文件不能超过 {max_file_size // (1024 * 1024)} MiB。"
                )
            digest.update(block)
            output.write(block)
    if total_bytes == 0:
        raise MaterialValidationError("不能上传空文件。")
    return total_bytes, digest.hexdigest()


def _production_sync_index(chunks: list[DocumentChunk]) -> IndexSyncSummary:
    config = EmbeddingConfig.from_environment()
    embeddings = create_langchain_embeddings(config)
    sync_result = sync_vector_store(
        chunks,
        embeddings,
        allow_empty=True,
        runtime_config=runtime_index_config(config),
    )
    try:
        return IndexSyncSummary(
            added=sync_result.added,
            deleted=sync_result.deleted,
            unchanged=sync_result.unchanged,
        )
    finally:
        close_vector_store(sync_result.vector_store)


def _production_estimate_index_batches(chunks: list[DocumentChunk]) -> int:
    config = EmbeddingConfig.from_environment()
    return estimate_vector_store_sync_batches(
        chunks,
        runtime_config=runtime_index_config(config),
    )


class MaterialManager:
    """协调资料暂存、文件提交、索引同步和失败补偿。"""

    def __init__(
        self,
        *,
        documents_dir: Path = DOCUMENTS_DIR,
        pending_uploads_dir: Path = PENDING_UPLOADS_DIR,
        pending_deletions_dir: Path = PENDING_DELETIONS_DIR,
        max_file_size: int = MAX_UPLOAD_BYTES,
        max_pdf_pages: int = MAX_UPLOAD_PDF_PAGES,
        max_extracted_characters: int = MAX_UPLOAD_EXTRACTED_CHARACTERS,
        max_embedding_batches: int = INDEX_MAX_BATCHES_PER_OPERATION,
        pending_upload_ttl_seconds: int = PENDING_UPLOAD_TTL_SECONDS,
        sync_index: Callable[[list[DocumentChunk]], IndexSyncSummary] | None = None,
        delete_index: Callable[[str], int] | None = None,
        estimate_index_batches: Callable[[list[DocumentChunk]], int] | None = None,
    ) -> None:
        numeric_limits = {
            "max_file_size": max_file_size,
            "max_pdf_pages": max_pdf_pages,
            "max_extracted_characters": max_extracted_characters,
            "max_embedding_batches": max_embedding_batches,
            "pending_upload_ttl_seconds": pending_upload_ttl_seconds,
        }
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in numeric_limits.values()
        ):
            raise ValueError("资料管理限制必须是正整数。")
        self.documents_dir = documents_dir
        self.pending_uploads_dir = pending_uploads_dir
        self.pending_deletions_dir = pending_deletions_dir
        self.max_file_size = max_file_size
        self.max_pdf_pages = max_pdf_pages
        self.max_extracted_characters = max_extracted_characters
        self.max_embedding_batches = max_embedding_batches
        self.pending_upload_ttl_seconds = pending_upload_ttl_seconds
        self._sync_index = sync_index or _production_sync_index
        self._delete_index = delete_index or delete_material_documents
        self._estimate_index_batches = (
            estimate_index_batches or _production_estimate_index_batches
        )

    def cleanup_stale_pending_uploads(self, *, now: datetime | None = None) -> int:
        """删除超过 TTL 的可识别暂存目录，不处理正式资料或删除隔离区。"""
        current_time = (now or datetime.now(UTC)).timestamp()
        try:
            candidates = tuple(self.pending_uploads_dir.iterdir())
        except OSError:
            return 0

        cleaned = 0
        for candidate in candidates:
            if (
                UPLOAD_ID_PATTERN.fullmatch(candidate.name) is None
                or candidate.is_symlink()
                or candidate.is_junction()
                or not candidate.is_dir()
            ):
                continue
            try:
                age_seconds = current_time - candidate.stat().st_mtime
                if age_seconds < self.pending_upload_ttl_seconds:
                    continue
                shutil.rmtree(candidate)
            except OSError:
                continue
            cleaned += 1
        return cleaned

    def _validate_operation(self, operation: object) -> UploadOperation:
        if not isinstance(operation, str) or operation not in {"add", "replace"}:
            raise MaterialValidationError("上传操作只能是 add 或 replace。")
        return operation

    def _validate_staged_metadata(self, staged: StagedMaterial) -> None:
        """重新校验磁盘中的暂存元数据，不能只依赖 dataclass 类型标注。"""
        if (
            not isinstance(staged.upload_id, str)
            or UPLOAD_ID_PATTERN.fullmatch(staged.upload_id) is None
        ):
            raise MaterialValidationError("暂存上传状态无效，请重新上传。")
        validate_material_filename(staged.filename)
        self._validate_operation(staged.operation)
        for value in (
            staged.size_bytes,
            staged.document_units,
            staged.chunk_count,
            staged.embedding_batch_count,
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise MaterialValidationError("暂存上传状态无效，请重新上传。")
        if staged.size_bytes > self.max_file_size:
            raise MaterialValidationError("暂存上传状态无效，请重新上传。")
        if staged.embedding_batch_count != ceil(
            staged.chunk_count / BAILIAN_EMBEDDING_BATCH_SIZE
        ):
            raise MaterialValidationError("暂存上传状态无效，请重新上传。")
        if staged.embedding_batch_count > self.max_embedding_batches:
            raise MaterialValidationError("暂存上传状态无效，请重新上传。")
        if (
            not isinstance(staged.sha256, str)
            or SHA256_PATTERN.fullmatch(staged.sha256) is None
            or not isinstance(staged.staged_at, str)
            or not staged.staged_at
            or len(staged.staged_at) > 80
        ):
            raise MaterialValidationError("暂存上传状态无效，请重新上传。")

    def _check_operation_state(
        self,
        filename: str,
        operation: UploadOperation,
    ) -> None:
        exists = (self.documents_dir / filename).is_file()
        if operation == "add" and exists:
            raise MaterialConflictError("同名资料已存在，请明确选择替换操作。")
        if operation == "replace" and not exists:
            raise MaterialConflictError("要替换的同名资料不存在。")

    def stage_upload(
        self,
        *,
        filename: str | None,
        content_type: str | None,
        stream: BinaryIO | BufferedIOBase,
        operation: str = "add",
    ) -> StagedMaterial:
        """零费用暂存并解析单个文件，不创建 Embedding 或打开 Chroma。"""
        self.cleanup_stale_pending_uploads()
        safe_filename = validate_material_filename(filename)
        safe_operation = self._validate_operation(operation)
        _validate_content_type(safe_filename, content_type)
        self._check_operation_state(safe_filename, safe_operation)

        upload_id = uuid4().hex
        upload_directory = self.pending_uploads_dir / upload_id
        incoming_path = upload_directory / ".incoming"
        staged_path = upload_directory / safe_filename
        try:
            upload_directory.mkdir(parents=True, exist_ok=False)
            size_bytes, digest = _write_upload_stream(
                stream,
                incoming_path,
                self.max_file_size,
            )
            incoming_path.replace(staged_path)
            _validate_file_signature(staged_path)
            documents = load_material_units(
                upload_directory,
                max_pdf_pages=self.max_pdf_pages,
                max_total_characters=self.max_extracted_characters,
            )
            chunks = build_chunks(documents)
            if not chunks:
                raise MaterialValidationError("资料中没有可建立索引的文本。")
            embedding_batch_count = ceil(
                len(chunks) / BAILIAN_EMBEDDING_BATCH_SIZE
            )
            if embedding_batch_count > self.max_embedding_batches:
                raise MaterialValidationError(
                    "资料切分后的预计 Embedding 批次数超过单次费用保护上限。"
                )
            staged = StagedMaterial(
                upload_id=upload_id,
                filename=safe_filename,
                operation=safe_operation,
                size_bytes=size_bytes,
                sha256=digest,
                document_units=len(documents),
                chunk_count=len(chunks),
                embedding_batch_count=embedding_batch_count,
                staged_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            )
            (upload_directory / UPLOAD_METADATA_FILENAME).write_text(
                json.dumps(asdict(staged), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            return staged
        except DocumentLoadError as error:
            shutil.rmtree(upload_directory, ignore_errors=True)
            raise MaterialValidationError(str(error)) from error
        except Exception:
            shutil.rmtree(upload_directory, ignore_errors=True)
            raise

    def _load_staged(self, upload_id: str) -> tuple[StagedMaterial, Path]:
        if UPLOAD_ID_PATTERN.fullmatch(upload_id) is None:
            raise MaterialNotFoundError("暂存上传不存在。")
        upload_directory = self.pending_uploads_dir / upload_id
        metadata_path = upload_directory / UPLOAD_METADATA_FILENAME
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            staged = StagedMaterial(**payload)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise MaterialNotFoundError("暂存上传不存在或状态无效。") from error
        try:
            self._validate_staged_metadata(staged)
        except MaterialValidationError as error:
            raise MaterialValidationError(
                "暂存上传状态无效，请重新上传。"
            ) from error
        if staged.upload_id != upload_id:
            raise MaterialNotFoundError("暂存上传状态无效。")
        safe_filename = validate_material_filename(staged.filename)
        staged_path = upload_directory / safe_filename
        if not staged_path.is_file():
            raise MaterialNotFoundError("暂存文件不存在。")
        _validate_file_signature(staged_path)
        if staged_path.stat().st_size != staged.size_bytes:
            raise MaterialValidationError("暂存文件大小发生变化，请重新上传。")
        digest = hashlib.sha256(staged_path.read_bytes()).hexdigest()
        if digest != staged.sha256:
            raise MaterialValidationError("暂存文件内容发生变化，请重新上传。")
        return staged, staged_path

    def inspect_staged(self, upload_id: str) -> StagedMaterial:
        """零费用读取并校验暂存摘要。"""
        staged, _ = self._load_staged(upload_id)
        return staged

    def estimate_index_batches(self, upload_id: str) -> int:
        """在正式提交前按 Chroma 现状计算实际待新增的批次数。"""
        staged, staged_path = self._load_staged(upload_id)
        self._check_operation_state(staged.filename, staged.operation)
        current_chunks = build_chunks(load_material_units(self.documents_dir))
        if staged.operation == "replace":
            current_chunks = [
                chunk
                for chunk in current_chunks
                if not source_belongs_to_material(chunk.source, staged.filename)
            ]
        staged_chunks = build_chunks(
            load_material_units(
                staged_path.parent,
                max_pdf_pages=self.max_pdf_pages,
                max_total_characters=self.max_extracted_characters,
            )
        )
        estimated_batches = self._estimate_index_batches(
            [*current_chunks, *staged_chunks]
        )
        if (
            isinstance(estimated_batches, bool)
            or not isinstance(estimated_batches, int)
            or estimated_batches < 0
        ):
            raise MaterialIndexError("无法计算安全的索引费用预算。")
        return estimated_batches

    def commit_staged(self, upload_id: str) -> MaterialSyncResult:
        """提交暂存文件并同步索引；调用方必须已经取得费用确认。"""
        staged, staged_path = self._load_staged(upload_id)
        self._check_operation_state(staged.filename, staged.operation)
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        final_path = self.documents_dir / staged.filename
        backup_directory = self.pending_deletions_dir / uuid4().hex
        backup_path = backup_directory / staged.filename
        old_chunks = build_chunks(load_material_units(self.documents_dir))
        moved_existing = False

        try:
            if staged.operation == "replace":
                backup_directory.mkdir(parents=True, exist_ok=False)
                final_path.replace(backup_path)
                moved_existing = True
            staged_path.replace(final_path)
            new_chunks = build_chunks(load_material_units(self.documents_dir))
            sync_result = self._sync_index(new_chunks)
        except Exception as error:
            try:
                if final_path.exists():
                    final_path.replace(staged_path)
                if moved_existing and backup_path.exists():
                    backup_path.replace(final_path)
                if not isinstance(error, IndexManifestError):
                    self._sync_index(old_chunks)
            except Exception as rollback_error:
                raise MaterialRollbackError(
                    "索引失败，且资料与索引的自动回滚未能完成。"
                ) from rollback_error
            raise MaterialIndexError("资料索引失败，已恢复上传前状态。") from error

        cleanup_pending = False
        for directory in (backup_directory, staged_path.parent):
            if not directory.exists():
                continue
            try:
                shutil.rmtree(directory)
            except OSError:
                cleanup_pending = True
        return MaterialSyncResult(
            filename=staged.filename,
            operation=staged.operation,
            added=sync_result.added,
            deleted=sync_result.deleted,
            unchanged=sync_result.unchanged,
            cleanup_pending=cleanup_pending,
        )

    def list_materials(self) -> tuple[MaterialFile, ...]:
        """列出正式资料目录中的受支持文件，不解析内容或打开 Chroma。"""
        if not self.documents_dir.exists():
            return ()
        return tuple(
            MaterialFile(filename=path.name, size_bytes=path.stat().st_size)
            for path in sorted(self.documents_dir.iterdir(), key=lambda item: item.name)
            if path.is_file() and path.suffix.lower() in SUPPORTED_DOCUMENT_SUFFIXES
        )

    def delete_material(self, filename: str) -> MaterialDeleteResult:
        """先隔离资料文件，再定向删除其 Chroma 记录；失败时恢复文件。"""
        safe_filename = validate_material_filename(filename)
        material_path = self.documents_dir / safe_filename
        if not material_path.is_file():
            raise MaterialNotFoundError("要删除的资料不存在。")

        quarantine_directory = self.pending_deletions_dir / uuid4().hex
        quarantine_path = quarantine_directory / safe_filename
        try:
            quarantine_directory.mkdir(parents=True, exist_ok=False)
            material_path.replace(quarantine_path)
        except OSError as error:
            shutil.rmtree(quarantine_directory, ignore_errors=True)
            raise MaterialIndexError(
                "资料文件无法安全隔离，未执行索引删除。"
            ) from error

        try:
            deleted_records = self._delete_index(safe_filename)
        except Exception as error:
            try:
                if quarantine_path.exists():
                    quarantine_path.replace(material_path)
            except OSError as rollback_error:
                raise MaterialRollbackError(
                    "删除索引失败，且资料文件自动恢复失败。"
                ) from rollback_error
            if isinstance(error, IndexManifestError):
                raise MaterialIndexError(
                    "索引配置不兼容，资料文件已恢复，未修改现有索引。"
                ) from error
            raise MaterialRollbackError(
                "删除索引失败，资料文件已恢复，但索引状态需要检查。"
            ) from error

        cleanup_pending = False
        try:
            shutil.rmtree(quarantine_directory)
        except OSError:
            cleanup_pending = True
        return MaterialDeleteResult(
            filename=safe_filename,
            deleted_records=deleted_records,
            cleanup_pending=cleanup_pending,
        )
