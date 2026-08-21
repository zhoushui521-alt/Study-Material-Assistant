import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from app.document_jobs import (
    DOCUMENT_JOB_SCHEMA_VERSION,
    DocumentJobConflictError,
    DocumentJobService,
    DocumentJobStore,
)
from app.material_ingestion import (
    BatchMaterialSyncResult,
    MaterialIndexReadOnlyError,
    MaterialManager,
    MaterialSyncResult,
    StagedMaterial,
)
from app.operation_guard import OperationGuard


def staged(upload_id: str, filename: str) -> StagedMaterial:
    return StagedMaterial(
        upload_id=upload_id,
        filename=filename,
        operation="add",
        size_bytes=12,
        sha256="b" * 64,
        document_units=1,
        chunk_count=2,
        embedding_batch_count=1,
        staged_at="2026-08-21T00:00:00Z",
    )


class DocumentJobStoreTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "jobs.sqlite3"
        self.store = await DocumentJobStore.open(self.database_path)

    async def asyncTearDown(self) -> None:
        await self.store.close()
        self.temporary_directory.cleanup()

    async def test_create_job_has_persistent_pending_state_and_timestamps(self) -> None:
        created = await self.store.create_job(("a" * 32,), ("notes.md",))
        loaded = await self.store.get_job(created.job_id)

        self.assertEqual(await self.store.schema_version(), DOCUMENT_JOB_SCHEMA_VERSION)
        self.assertEqual(loaded.status, "pending")
        self.assertEqual(loaded.progress, 0)
        self.assertEqual(loaded.filenames, ("notes.md",))
        self.assertIsNotNone(loaded.created_at)
        self.assertIsNone(loaded.started_at)
        self.assertIsNone(loaded.finished_at)

    async def test_rejects_duplicate_active_job_for_same_uploads(self) -> None:
        await self.store.create_job(
            ("a" * 32, "b" * 32),
            ("notes.md", "chapter.pdf"),
        )

        with self.assertRaises(DocumentJobConflictError):
            await self.store.create_job(
                ("b" * 32, "a" * 32),
                ("chapter.pdf", "notes.md"),
            )

    async def test_recovery_marks_interrupted_processing_as_failed(self) -> None:
        created = await self.store.create_job(("a" * 32,), ("notes.md",))
        claimed = await self.store.claim_next_pending()
        self.assertEqual(claimed.status, "processing")
        await self.store.close()

        self.store = await DocumentJobStore.open(self.database_path)
        recovered = await self.store.recover_interrupted_jobs()
        loaded = await self.store.get_job(created.job_id)

        self.assertEqual(recovered, 1)
        self.assertEqual(loaded.status, "failed")
        self.assertIsNotNone(loaded.finished_at)
        self.assertIn("服务在处理期间中断", loaded.error_message)


class DocumentJobServiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "jobs.sqlite3"
        self.manager = Mock(spec=MaterialManager)
        self.invalidated = Mock()
        self.service = await DocumentJobService.open(
            material_manager=self.manager,
            operation_guard=OperationGuard(),
            invalidate_rag=self.invalidated,
            database_path=self.database_path,
        )

    async def asyncTearDown(self) -> None:
        await self.service.close()
        self.temporary_directory.cleanup()

    async def wait_finished(self, job_id: str):
        for _ in range(100):
            job = await self.service.get_job(job_id)
            if job.status in {"completed", "failed"}:
                return job
            await asyncio.sleep(0.01)
        self.fail("文档任务未在测试时限内结束。")

    async def test_worker_moves_pending_to_completed_and_saves_result(self) -> None:
        upload_id = "a" * 32
        self.manager.inspect_staged.return_value = staged(upload_id, "notes.md")
        self.manager.estimate_index_batches.return_value = 1
        self.manager.commit_staged.return_value = MaterialSyncResult(
            filename="notes.md",
            operation="add",
            added=2,
            deleted=0,
            unchanged=141,
            cleanup_pending=False,
        )

        created = await self.service.enqueue((upload_id,))
        finished = await self.wait_finished(created.job_id)

        self.assertEqual(finished.status, "completed")
        self.assertEqual(finished.progress, 100)
        self.assertEqual(finished.result["added"], 2)
        self.assertIsNotNone(finished.started_at)
        self.assertIsNotNone(finished.finished_at)
        self.manager.estimate_index_batches.assert_called_once_with(upload_id)
        self.manager.commit_staged.assert_called_once_with(upload_id)
        self.invalidated.assert_called_once_with()

    async def test_worker_records_safe_failure_and_never_leaves_processing(self) -> None:
        upload_id = "a" * 32
        self.manager.inspect_staged.return_value = staged(upload_id, "notes.md")
        self.manager.estimate_index_batches.side_effect = MaterialIndexReadOnlyError(
            "现有索引处于只读兼容性保护；未调用 Embedding。"
        )

        created = await self.service.enqueue((upload_id,))
        finished = await self.wait_finished(created.job_id)

        self.assertEqual(finished.status, "failed")
        self.assertIn("只读兼容性保护", finished.error_message)
        self.assertIsNotNone(finished.finished_at)
        self.manager.commit_staged.assert_not_called()
        self.invalidated.assert_not_called()

    async def test_worker_reuses_existing_atomic_batch_pipeline(self) -> None:
        upload_ids = ("a" * 32, "b" * 32)
        self.manager.inspect_staged.side_effect = (
            staged(upload_ids[0], "chapter.pdf"),
            staged(upload_ids[1], "notes.docx"),
        )
        self.manager.estimate_index_batches_batch.return_value = 2
        self.manager.commit_staged_batch.return_value = BatchMaterialSyncResult(
            filenames=("chapter.pdf", "notes.docx"),
            added=4,
            deleted=0,
            unchanged=141,
            cleanup_pending=False,
        )

        created = await self.service.enqueue(upload_ids)
        finished = await self.wait_finished(created.job_id)

        self.assertEqual(finished.status, "completed")
        self.assertEqual(finished.result["filenames"], ["chapter.pdf", "notes.docx"])
        self.manager.estimate_index_batches_batch.assert_called_once_with(upload_ids)
        self.manager.commit_staged_batch.assert_called_once_with(upload_ids)

    async def test_restart_resumes_persisted_pending_job(self) -> None:
        upload_id = "a" * 32
        await self.service.close()
        store = await DocumentJobStore.open(self.database_path)
        pending = await store.create_job((upload_id,), ("notes.md",))
        await store.close()
        self.manager.estimate_index_batches.return_value = 1
        self.manager.commit_staged.return_value = MaterialSyncResult(
            filename="notes.md",
            operation="add",
            added=1,
            deleted=0,
            unchanged=0,
            cleanup_pending=False,
        )

        self.service = await DocumentJobService.open(
            material_manager=self.manager,
            operation_guard=OperationGuard(),
            invalidate_rag=self.invalidated,
            database_path=self.database_path,
        )
        finished = await self.wait_finished(pending.job_id)

        self.assertEqual(finished.status, "completed")
        self.manager.commit_staged.assert_called_once_with(upload_id)


if __name__ == "__main__":
    unittest.main()
