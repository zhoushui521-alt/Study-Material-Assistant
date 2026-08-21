"""Stage 5.2：持久化文档处理任务与单进程后台 Worker。"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import aiosqlite

from app.material_ingestion import (
    BatchMaterialSyncResult,
    MaterialIngestionError,
    MaterialManager,
    MaterialRollbackError,
    MaterialSyncResult,
)
from app.operation_guard import OperationGuard, OperationProtectionError


DOCUMENT_JOB_DB_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "jobs" / "document_jobs.sqlite3"
)
DOCUMENT_JOB_SCHEMA_VERSION = 1
MAX_JOB_ERROR_LENGTH = 500
JobStatus = Literal["pending", "processing", "completed", "failed"]


class DocumentJobError(RuntimeError):
    """文档任务无法按持久化契约完成。"""


class DocumentJobNotFoundError(DocumentJobError):
    """文档任务不存在。"""


class DocumentJobConflictError(DocumentJobError):
    """同一批暂存资料已经有活动任务。"""


@dataclass(frozen=True)
class DocumentJobRecord:
    job_id: str
    upload_ids: tuple[str, ...]
    filenames: tuple[str, ...]
    status: JobStatus
    progress: int
    error_message: str | None
    result: dict[str, Any] | None
    created_at: str
    started_at: str | None
    finished_at: str | None
    updated_at: str


MIGRATION_1 = """
BEGIN IMMEDIATE;

CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL DEFAULT (
        strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
    )
);

CREATE TABLE IF NOT EXISTS document_jobs (
    job_id TEXT PRIMARY KEY,
    request_key TEXT NOT NULL,
    upload_ids_json TEXT NOT NULL,
    filenames_json TEXT NOT NULL,
    status TEXT NOT NULL CHECK (
        status IN ('pending', 'processing', 'completed', 'failed')
    ),
    progress INTEGER NOT NULL CHECK (progress BETWEEN 0 AND 100),
    error_message TEXT,
    result_json TEXT,
    created_at TEXT NOT NULL DEFAULT (
        strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
    ),
    started_at TEXT,
    finished_at TEXT,
    updated_at TEXT NOT NULL DEFAULT (
        strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
    )
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_document_jobs_active_request
    ON document_jobs(request_key)
    WHERE status IN ('pending', 'processing');

CREATE INDEX IF NOT EXISTS idx_document_jobs_status_created
    ON document_jobs(status, created_at, job_id);

INSERT OR IGNORE INTO schema_migrations(version) VALUES (1);
COMMIT;
"""


def _canonical_job_id(value: str) -> str:
    try:
        return str(uuid.UUID(value))
    except (AttributeError, TypeError, ValueError) as error:
        raise DocumentJobNotFoundError("文档处理任务不存在。") from error


def _decode_string_tuple(value: str) -> tuple[str, ...]:
    try:
        payload = json.loads(value)
    except (TypeError, json.JSONDecodeError) as error:
        raise DocumentJobError("文档任务持久化数据无效。") from error
    if (
        not isinstance(payload, list)
        or not payload
        or any(not isinstance(item, str) or not item for item in payload)
    ):
        raise DocumentJobError("文档任务持久化数据无效。")
    return tuple(payload)


def _row_to_record(row: aiosqlite.Row) -> DocumentJobRecord:
    result: dict[str, Any] | None = None
    if row["result_json"] is not None:
        try:
            decoded = json.loads(row["result_json"])
        except (TypeError, json.JSONDecodeError) as error:
            raise DocumentJobError("文档任务结果数据无效。") from error
        if not isinstance(decoded, dict):
            raise DocumentJobError("文档任务结果数据无效。")
        result = decoded
    return DocumentJobRecord(
        job_id=row["job_id"],
        upload_ids=_decode_string_tuple(row["upload_ids_json"]),
        filenames=_decode_string_tuple(row["filenames_json"]),
        status=row["status"],
        progress=row["progress"],
        error_message=row["error_message"],
        result=result,
        created_at=row["created_at"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        updated_at=row["updated_at"],
    )


class DocumentJobStore:
    """用 SQLite 保存任务状态；文件正文和模型内容不进入任务数据库。"""

    def __init__(self, connection: aiosqlite.Connection) -> None:
        self._connection = connection
        self._write_lock = asyncio.Lock()
        self._closed = False

    @classmethod
    async def open(
        cls,
        database_path: Path = DOCUMENT_JOB_DB_PATH,
    ) -> "DocumentJobStore":
        database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = await aiosqlite.connect(database_path)
        try:
            connection.row_factory = aiosqlite.Row
            await connection.execute("PRAGMA busy_timeout = 5000")
            await connection.execute("PRAGMA journal_mode = WAL")
            await cls._apply_migrations(connection)
            return cls(connection)
        except Exception:
            await connection.close()
            raise

    @staticmethod
    async def _apply_migrations(connection: aiosqlite.Connection) -> None:
        await connection.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL DEFAULT (
                    strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                )
            )
            """
        )
        cursor = await connection.execute(
            "SELECT COALESCE(MAX(version), 0) FROM schema_migrations"
        )
        row = await cursor.fetchone()
        current_version = int(row[0]) if row is not None else 0
        if current_version > DOCUMENT_JOB_SCHEMA_VERSION:
            raise DocumentJobError("文档任务数据库版本高于当前程序支持范围。")
        if current_version < 1:
            try:
                await connection.executescript(MIGRATION_1)
            except Exception:
                try:
                    await connection.rollback()
                except Exception:
                    pass
                raise
        await connection.commit()

    async def schema_version(self) -> int:
        cursor = await self._connection.execute(
            "SELECT COALESCE(MAX(version), 0) FROM schema_migrations"
        )
        row = await cursor.fetchone()
        return int(row[0]) if row is not None else 0

    async def create_job(
        self,
        upload_ids: Sequence[str],
        filenames: Sequence[str],
    ) -> DocumentJobRecord:
        ids = tuple(upload_ids)
        names = tuple(filenames)
        if not ids or len(ids) != len(names):
            raise ValueError("任务必须包含一一对应的暂存 ID 和文件名。")
        request_key = json.dumps(
            sorted(ids),
            ensure_ascii=True,
            separators=(",", ":"),
        )
        job_id = str(uuid.uuid4())
        async with self._write_lock:
            try:
                await self._connection.execute(
                    """
                    INSERT INTO document_jobs(
                        job_id, request_key, upload_ids_json, filenames_json,
                        status, progress
                    ) VALUES (?, ?, ?, ?, 'pending', 0)
                    """,
                    (
                        job_id,
                        request_key,
                        json.dumps(ids, ensure_ascii=False, separators=(",", ":")),
                        json.dumps(names, ensure_ascii=False, separators=(",", ":")),
                    ),
                )
                await self._connection.commit()
            except sqlite3.IntegrityError as error:
                await self._connection.rollback()
                raise DocumentJobConflictError(
                    "这些暂存资料已经有等待或正在处理的任务。"
                ) from error
            except Exception:
                await self._connection.rollback()
                raise
        return await self.get_job(job_id)

    async def get_job(self, job_id: str) -> DocumentJobRecord:
        normalized = _canonical_job_id(job_id)
        cursor = await self._connection.execute(
            "SELECT * FROM document_jobs WHERE job_id = ?",
            (normalized,),
        )
        row = await cursor.fetchone()
        if row is None:
            raise DocumentJobNotFoundError("文档处理任务不存在。")
        return _row_to_record(row)

    async def recover_interrupted_jobs(self) -> int:
        """把上次进程中断时的 processing 任务落为明确失败，避免假运行。"""
        async with self._write_lock:
            cursor = await self._connection.execute(
                """
                UPDATE document_jobs
                SET status = 'failed',
                    error_message = ?,
                    finished_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
                    updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                WHERE status = 'processing'
                """,
                ("服务在处理期间中断，请重新提交任务。",),
            )
            await self._connection.commit()
            return max(0, cursor.rowcount)

    async def claim_next_pending(self) -> DocumentJobRecord | None:
        async with self._write_lock:
            try:
                await self._connection.execute("BEGIN IMMEDIATE")
                cursor = await self._connection.execute(
                    """
                    SELECT job_id FROM document_jobs
                    WHERE status = 'pending'
                    ORDER BY created_at, job_id
                    LIMIT 1
                    """
                )
                row = await cursor.fetchone()
                if row is None:
                    await self._connection.commit()
                    return None
                job_id = row["job_id"]
                await self._connection.execute(
                    """
                    UPDATE document_jobs
                    SET status = 'processing', progress = 10,
                        started_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
                        updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                    WHERE job_id = ? AND status = 'pending'
                    """,
                    (job_id,),
                )
                await self._connection.commit()
            except Exception:
                await self._connection.rollback()
                raise
        return await self.get_job(job_id)

    async def mark_completed(
        self,
        job_id: str,
        result: dict[str, Any],
    ) -> DocumentJobRecord:
        return await self._finish_job(
            job_id,
            status="completed",
            progress=100,
            error_message=None,
            result=result,
        )

    async def mark_failed(
        self,
        job_id: str,
        error_message: str,
    ) -> DocumentJobRecord:
        message = error_message.strip()[:MAX_JOB_ERROR_LENGTH]
        if not message:
            message = "资料后台处理失败。"
        return await self._finish_job(
            job_id,
            status="failed",
            progress=10,
            error_message=message,
            result=None,
        )

    async def _finish_job(
        self,
        job_id: str,
        *,
        status: Literal["completed", "failed"],
        progress: int,
        error_message: str | None,
        result: dict[str, Any] | None,
    ) -> DocumentJobRecord:
        normalized = _canonical_job_id(job_id)
        async with self._write_lock:
            cursor = await self._connection.execute(
                """
                UPDATE document_jobs
                SET status = ?, progress = ?, error_message = ?, result_json = ?,
                    finished_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
                    updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                WHERE job_id = ? AND status = 'processing'
                """,
                (
                    status,
                    progress,
                    error_message,
                    (
                        json.dumps(result, ensure_ascii=False, separators=(",", ":"))
                        if result is not None
                        else None
                    ),
                    normalized,
                ),
            )
            await self._connection.commit()
            if cursor.rowcount != 1:
                raise DocumentJobConflictError("文档任务状态已经发生变化。")
        return await self.get_job(normalized)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._connection.close()


def _safe_job_error(error: Exception) -> str:
    if isinstance(error, MaterialRollbackError):
        return "资料处理失败，且自动回滚未能确认完成；请检查资料与索引状态。"
    if isinstance(error, (MaterialIngestionError, OperationProtectionError)):
        return str(error)
    return "资料后台处理失败。"


class DocumentJobService:
    """串行消费持久化任务，并复用现有资料提交与索引保护。"""

    def __init__(
        self,
        *,
        store: DocumentJobStore,
        material_manager: MaterialManager,
        operation_guard: OperationGuard,
        invalidate_rag: Callable[[], None],
    ) -> None:
        self.store = store
        self._material_manager = material_manager
        self._operation_guard = operation_guard
        self._invalidate_rag = invalidate_rag
        self._wake_event = asyncio.Event()
        self._worker_task: asyncio.Task[None] | None = None
        self._closing = False

    @classmethod
    async def open(
        cls,
        *,
        material_manager: MaterialManager,
        operation_guard: OperationGuard,
        invalidate_rag: Callable[[], None],
        database_path: Path = DOCUMENT_JOB_DB_PATH,
    ) -> "DocumentJobService":
        store = await DocumentJobStore.open(database_path)
        service = cls(
            store=store,
            material_manager=material_manager,
            operation_guard=operation_guard,
            invalidate_rag=invalidate_rag,
        )
        await store.recover_interrupted_jobs()
        service._worker_task = asyncio.create_task(
            service._worker_loop(),
            name="document-job-worker",
        )
        service._wake_event.set()
        return service

    async def enqueue(self, upload_ids: Sequence[str]) -> DocumentJobRecord:
        ids = tuple(upload_ids)
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("任务必须包含不重复的暂存上传 ID。")
        staged = await asyncio.gather(
            *(
                asyncio.to_thread(self._material_manager.inspect_staged, upload_id)
                for upload_id in ids
            )
        )
        names = tuple(item.filename for item in staged)
        if len({name.casefold() for name in names}) != len(names):
            raise ValueError("任务不能包含重复文件名。")
        job = await self.store.create_job(ids, names)
        self._wake_event.set()
        return job

    async def get_job(self, job_id: str) -> DocumentJobRecord:
        return await self.store.get_job(job_id)

    def _execute_job(self, job: DocumentJobRecord) -> dict[str, Any]:
        with self._operation_guard.acquire("index") as lease:
            if len(job.upload_ids) == 1:
                upload_id = job.upload_ids[0]
                lease.reserve_units(
                    self._material_manager.estimate_index_batches(upload_id)
                )
                result: MaterialSyncResult = self._material_manager.commit_staged(
                    upload_id
                )
                return {
                    "filenames": [result.filename],
                    "added": result.added,
                    "deleted": result.deleted,
                    "unchanged": result.unchanged,
                    "cleanup_pending": result.cleanup_pending,
                }

            lease.reserve_units(
                self._material_manager.estimate_index_batches_batch(job.upload_ids)
            )
            batch_result: BatchMaterialSyncResult = (
                self._material_manager.commit_staged_batch(job.upload_ids)
            )
            return {
                "filenames": list(batch_result.filenames),
                "added": batch_result.added,
                "deleted": batch_result.deleted,
                "unchanged": batch_result.unchanged,
                "cleanup_pending": batch_result.cleanup_pending,
            }

    async def _process_job(self, job: DocumentJobRecord) -> None:
        try:
            result = await asyncio.to_thread(self._execute_job, job)
        except Exception as error:
            if isinstance(error, MaterialRollbackError):
                self._invalidate_rag()
            await self.store.mark_failed(job.job_id, _safe_job_error(error))
            return
        self._invalidate_rag()
        await self.store.mark_completed(job.job_id, result)

    async def _worker_loop(self) -> None:
        while not self._closing:
            self._wake_event.clear()
            while not self._closing:
                job = await self.store.claim_next_pending()
                if job is None:
                    break
                await self._process_job(job)
            if not self._closing:
                await self._wake_event.wait()

    async def close(self) -> None:
        if self._closing:
            return
        self._closing = True
        self._wake_event.set()
        try:
            if self._worker_task is not None:
                await self._worker_task
        finally:
            await self.store.close()
