import json
import os
import tempfile
import unittest
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path

from docx import Document as WordDocument

from app.index_manifest import LegacyIndexError

from app.material_ingestion import (
    IndexSyncSummary,
    MaterialConflictError,
    MaterialIndexError,
    MaterialManager,
    MaterialRollbackError,
    MaterialTooLargeError,
    MaterialValidationError,
    validate_material_filename,
)


def docx_bytes(text: str | None = "RAG 会先检索资料。") -> bytes:
    stream = BytesIO()
    document = WordDocument()
    if text is not None:
        document.add_paragraph(text)
    document.save(stream)
    return stream.getvalue()


class MaterialIngestionTests(unittest.TestCase):
    def make_manager(
        self,
        root: Path,
        *,
        sync_index=None,
        delete_index=None,
        max_file_size: int = 10 * 1024 * 1024,
        max_embedding_batches: int = 20,
        pending_upload_ttl_seconds: int = 24 * 60 * 60,
        estimate_index_batches=None,
    ) -> MaterialManager:
        return MaterialManager(
            documents_dir=root / "documents",
            pending_uploads_dir=root / "pending_uploads",
            pending_deletions_dir=root / "pending_deletions",
            max_file_size=max_file_size,
            max_embedding_batches=max_embedding_batches,
            pending_upload_ttl_seconds=pending_upload_ttl_seconds,
            sync_index=sync_index or (lambda chunks: IndexSyncSummary(len(chunks), 0, 0)),
            delete_index=delete_index or (lambda filename: 1),
            estimate_index_batches=estimate_index_batches or (lambda chunks: 1),
        )

    def test_rejects_path_traversal_and_windows_reserved_names(self) -> None:
        for filename in (
            "../secret.md",
            r"folder\secret.md",
            "C:secret.md",
            "CON.txt",
            "CON.backup.txt",
            "bad?.md",
            " notes.md",
        ):
            with self.subTest(filename=filename):
                with self.assertRaises(MaterialValidationError):
                    validate_material_filename(filename)

    def test_stages_valid_markdown_without_calling_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sync_calls = []
            manager = self.make_manager(
                root,
                sync_index=lambda chunks: sync_calls.append(chunks),
            )

            staged = manager.stage_upload(
                filename="notes.md",
                content_type="text/markdown",
                stream=BytesIO("RAG 上传后仍复用现有切分。".encode()),
            )

            self.assertEqual(staged.filename, "notes.md")
            self.assertEqual(staged.chunk_count, 1)
            self.assertEqual(staged.embedding_batch_count, 1)
            self.assertTrue(
                (root / "pending_uploads" / staged.upload_id / "notes.md").is_file()
            )
            self.assertEqual(sync_calls, [])

    def test_stages_valid_docx_without_calling_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sync_calls = []
            manager = self.make_manager(
                root,
                sync_index=lambda chunks: sync_calls.append(chunks),
            )

            staged = manager.stage_upload(
                filename="notes.docx",
                content_type=(
                    "application/vnd.openxmlformats-officedocument."
                    "wordprocessingml.document"
                ),
                stream=BytesIO(docx_bytes()),
            )

            self.assertEqual(staged.filename, "notes.docx")
            self.assertEqual(staged.document_units, 1)
            self.assertEqual(staged.chunk_count, 1)
            self.assertEqual(sync_calls, [])

    def test_rejects_spoofed_docx_and_cleans_pending_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manager = self.make_manager(root)

            with self.assertRaisesRegex(MaterialValidationError, "OOXML ZIP"):
                manager.stage_upload(
                    filename="fake.docx",
                    content_type=(
                        "application/vnd.openxmlformats-officedocument."
                        "wordprocessingml.document"
                    ),
                    stream=BytesIO(b"this is plain text"),
                )

            pending = root / "pending_uploads"
            self.assertFalse(pending.exists() and any(pending.iterdir()))

    def test_rejects_empty_docx_without_leaving_partial_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manager = self.make_manager(root)

            with self.assertRaisesRegex(MaterialValidationError, "没有可建立索引"):
                manager.stage_upload(
                    filename="empty.docx",
                    content_type="application/octet-stream",
                    stream=BytesIO(docx_bytes(None)),
                )

            pending = root / "pending_uploads"
            self.assertFalse(pending.exists() and any(pending.iterdir()))

    def test_rejects_legacy_doc_with_explicit_message(self) -> None:
        with self.assertRaisesRegex(MaterialValidationError, r"旧版 \.doc 暂不支持"):
            validate_material_filename("legacy.doc")

    def test_commits_docx_through_existing_sync_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            captured_chunks = []

            def sync_index(chunks):
                captured_chunks.extend(chunks)
                return IndexSyncSummary(added=len(chunks), deleted=0, unchanged=0)

            manager = self.make_manager(root, sync_index=sync_index)
            staged = manager.stage_upload(
                filename="notes.docx",
                content_type="application/octet-stream",
                stream=BytesIO(docx_bytes("DOCX 复用现有同步链路。")),
            )

            result = manager.commit_staged(staged.upload_id)

            self.assertEqual(result.added, 1)
            self.assertEqual(captured_chunks[0].source_type, "docx")
            self.assertEqual(captured_chunks[0].paragraph_index, 1)
            self.assertIsNone(captured_chunks[0].page)
            self.assertTrue((root / "documents" / "notes.docx").is_file())

    def test_rejects_oversized_upload_without_leaving_partial_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manager = self.make_manager(root, max_file_size=4)

            with self.assertRaises(MaterialTooLargeError):
                manager.stage_upload(
                    filename="notes.txt",
                    content_type="text/plain",
                    stream=BytesIO(b"12345"),
                )

            pending = root / "pending_uploads"
            self.assertFalse(pending.exists() and any(pending.iterdir()))

    def test_rejects_upload_over_embedding_batch_limit_without_staging(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manager = self.make_manager(root, max_embedding_batches=1)

            with self.assertRaisesRegex(MaterialValidationError, "批次数"):
                manager.stage_upload(
                    filename="large.md",
                    content_type="text/markdown",
                    stream=BytesIO(("a" * 1801).encode()),
                )

            pending = root / "pending_uploads"
            self.assertFalse(pending.exists() and any(pending.iterdir()))

    def test_cleanup_removes_only_stale_recognized_pending_directories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manager = self.make_manager(root, pending_upload_ttl_seconds=60)
            pending = root / "pending_uploads"
            pending.mkdir()
            stale = pending / ("a" * 32)
            fresh = pending / ("b" * 32)
            unrelated = pending / "keep-me"
            stale.mkdir()
            fresh.mkdir()
            unrelated.mkdir()
            now = datetime(2026, 8, 10, 12, 0, tzinfo=UTC)
            os.utime(stale, (now.timestamp() - 61, now.timestamp() - 61))
            os.utime(fresh, (now.timestamp() - 10, now.timestamp() - 10))

            cleaned = manager.cleanup_stale_pending_uploads(now=now)

            self.assertEqual(cleaned, 1)
            self.assertFalse(stale.exists())
            self.assertTrue(fresh.is_dir())
            self.assertTrue(unrelated.is_dir())

    def test_rejects_non_positive_manager_limits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "正整数"):
                self.make_manager(root, max_embedding_batches=0)

    def test_estimates_replace_batches_from_candidate_documents(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            documents = root / "documents"
            documents.mkdir()
            (documents / "notes.md").write_text("旧资料", encoding="utf-8")
            (documents / "keep.md").write_text("保留资料", encoding="utf-8")
            captured = []
            manager = self.make_manager(
                root,
                estimate_index_batches=lambda chunks: captured.extend(chunks) or 2,
            )
            staged = manager.stage_upload(
                filename="notes.md",
                content_type="text/markdown",
                stream=BytesIO("新资料".encode()),
                operation="replace",
            )

            batches = manager.estimate_index_batches(staged.upload_id)

            self.assertEqual(batches, 2)
            contents = [chunk.content for chunk in captured]
            self.assertIn("保留资料", contents)
            self.assertIn("新资料", contents)
            self.assertNotIn("旧资料", contents)

    def test_rejects_spoofed_pdf_signature(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manager = self.make_manager(root)

            with self.assertRaisesRegex(MaterialValidationError, "PDF 文件签名"):
                manager.stage_upload(
                    filename="fake.pdf",
                    content_type="application/pdf",
                    stream=BytesIO(b"not-a-pdf"),
                )

    def test_add_conflict_and_replace_missing_are_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            documents = root / "documents"
            documents.mkdir()
            (documents / "notes.md").write_text("旧资料", encoding="utf-8")
            manager = self.make_manager(root)

            with self.assertRaises(MaterialConflictError):
                manager.stage_upload(
                    filename="notes.md",
                    content_type="text/markdown",
                    stream=BytesIO("新资料".encode()),
                    operation="add",
                )
            with self.assertRaises(MaterialConflictError):
                manager.stage_upload(
                    filename="missing.md",
                    content_type="text/markdown",
                    stream=BytesIO("新资料".encode()),
                    operation="replace",
                )

    def test_commits_add_and_returns_incremental_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            captured_chunks = []

            def sync_index(chunks):
                captured_chunks.extend(chunks)
                return IndexSyncSummary(added=1, deleted=0, unchanged=2)

            manager = self.make_manager(root, sync_index=sync_index)
            staged = manager.stage_upload(
                filename="notes.md",
                content_type="text/markdown",
                stream=BytesIO("新增 RAG 资料。".encode()),
            )

            result = manager.commit_staged(staged.upload_id)

            self.assertEqual(result.added, 1)
            self.assertEqual(result.unchanged, 2)
            self.assertTrue((root / "documents" / "notes.md").is_file())
            self.assertEqual(captured_chunks[0].source, "notes.md")
            self.assertFalse((root / "pending_uploads" / staged.upload_id).exists())

    def test_rejects_tampered_staged_operation_before_overwriting_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manager = self.make_manager(root)
            staged = manager.stage_upload(
                filename="notes.md",
                content_type="text/markdown",
                stream=BytesIO("新增 RAG 资料。".encode()),
            )
            metadata_path = (
                root / "pending_uploads" / staged.upload_id / "upload.json"
            )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["operation"] = "overwrite"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            documents = root / "documents"
            documents.mkdir()
            existing = documents / "notes.md"
            existing.write_text("并发出现的原资料", encoding="utf-8")

            with self.assertRaisesRegex(MaterialValidationError, "状态无效"):
                manager.commit_staged(staged.upload_id)

            self.assertEqual(existing.read_text(encoding="utf-8"), "并发出现的原资料")

    def test_index_failure_restores_pending_file_and_previous_documents(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calls = 0

            def fail_once(chunks):
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise RuntimeError("embedding failed")
                return IndexSyncSummary(0, 0, len(chunks))

            manager = self.make_manager(root, sync_index=fail_once)
            staged = manager.stage_upload(
                filename="notes.md",
                content_type="text/markdown",
                stream=BytesIO("新增 RAG 资料。".encode()),
            )

            with self.assertRaisesRegex(MaterialIndexError, "已恢复"):
                manager.commit_staged(staged.upload_id)

            self.assertFalse((root / "documents" / "notes.md").exists())
            self.assertTrue(
                (root / "pending_uploads" / staged.upload_id / "notes.md").is_file()
            )
            self.assertEqual(calls, 2)

    def test_replace_updates_file_only_after_explicit_operation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            documents = root / "documents"
            documents.mkdir()
            (documents / "notes.md").write_text("旧资料", encoding="utf-8")

            def sync_index(chunks):
                self.assertTrue(any("新资料" in chunk.content for chunk in chunks))
                return IndexSyncSummary(1, 1, 0)

            manager = self.make_manager(root, sync_index=sync_index)
            staged = manager.stage_upload(
                filename="notes.md",
                content_type="text/markdown",
                stream=BytesIO("新资料".encode()),
                operation="replace",
            )

            result = manager.commit_staged(staged.upload_id)

            self.assertEqual(result.operation, "replace")
            self.assertEqual(
                (documents / "notes.md").read_text(encoding="utf-8"),
                "新资料",
            )

    def test_delete_failure_restores_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            documents = root / "documents"
            documents.mkdir()
            material = documents / "notes.md"
            material.write_text("资料", encoding="utf-8")
            manager = self.make_manager(
                root,
                delete_index=lambda filename: (_ for _ in ()).throw(
                    RuntimeError("delete failed")
                ),
            )

            with self.assertRaisesRegex(MaterialRollbackError, "需要检查"):
                manager.delete_material("notes.md")

            self.assertTrue(material.is_file())

    def test_legacy_index_delete_restores_file_without_uncertain_rollback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            documents = root / "documents"
            documents.mkdir()
            material = documents / "notes.md"
            material.write_text("资料", encoding="utf-8")
            manager = self.make_manager(
                root,
                delete_index=lambda filename: (_ for _ in ()).throw(
                    LegacyIndexError("legacy index")
                ),
            )

            with self.assertRaisesRegex(MaterialIndexError, "未修改现有索引"):
                manager.delete_material("notes.md")

            self.assertTrue(material.is_file())

    def test_delete_success_removes_file_and_reports_index_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            documents = root / "documents"
            documents.mkdir()
            material = documents / "notes.md"
            material.write_text("资料", encoding="utf-8")
            manager = self.make_manager(root, delete_index=lambda filename: 3)

            result = manager.delete_material("notes.md")

            self.assertEqual(result.deleted_records, 3)
            self.assertFalse(material.exists())


if __name__ == "__main__":
    unittest.main()
