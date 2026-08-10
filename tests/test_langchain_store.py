import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from langchain_core.embeddings import Embeddings

from app.chunk_documents import DocumentChunk
from app.embedding_client import EmbeddingConfig
from app.langchain_store import (
    close_vector_store,
    create_langchain_embeddings,
    delete_material_documents,
    estimate_vector_store_sync_batches,
    open_vector_store,
    rebuild_vector_store,
    search_vector_store,
    source_belongs_to_material,
    sync_vector_store,
)


class KeywordEmbeddings(Embeddings):
    """测试用 Embedding：包含 RAG 时返回横向量，否则返回纵向量。"""

    @staticmethod
    def _embed(text: str) -> list[float]:
        return [1.0, 0.0] if "rag" in text.casefold() else [0.0, 1.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class CountingEmbeddings(KeywordEmbeddings):
    """记录被向量化的文档数量，用于验证增量索引没有重复调用。"""

    def __init__(self) -> None:
        self.document_count = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_count += len(texts)
        return super().embed_documents(texts)


class LangChainStoreTests(unittest.TestCase):
    def test_web_source_label_belongs_to_generated_material(self) -> None:
        source = "web-example-a1b2c3d4e5f6.md · 网页：https://example.com/rag"

        self.assertTrue(
            source_belongs_to_material(source, "web-example-a1b2c3d4e5f6.md")
        )
        self.assertFalse(source_belongs_to_material(source, "other.md"))

    @patch("app.langchain_store.OpenAIEmbeddings")
    def test_configures_bailian_embedding_batch_limit(
        self,
        mock_embeddings: Mock,
    ) -> None:
        config = EmbeddingConfig(
            api_key="test-key",
            base_url="https://example.com/v1",
            model="test-model",
            dimensions=1024,
        )

        result = create_langchain_embeddings(config)

        mock_embeddings.assert_called_once_with(
            api_key=config.api_key,
            base_url=config.base_url,
            model=config.model,
            dimensions=config.dimensions,
            check_embedding_ctx_length=False,
            chunk_size=10,
            timeout=30,
            max_retries=2,
        )
        self.assertIs(result, mock_embeddings.return_value)

    def test_estimates_only_new_embedding_batches_without_embedding(self) -> None:
        chunks = [DocumentChunk("notes.md", 1, "existing")]
        chunks.extend(
            DocumentChunk("notes.md", index, f"new-{index}")
            for index in range(2, 13)
        )

        with tempfile.TemporaryDirectory() as directory:
            persist_directory = Path(directory)
            vector_store = rebuild_vector_store(
                chunks[:1],
                KeywordEmbeddings(),
                persist_directory,
            )
            close_vector_store(vector_store)

            batches = estimate_vector_store_sync_batches(
                chunks,
                persist_directory,
            )

        self.assertEqual(batches, 2)

    def test_persists_metadata_and_retrieves_relevant_document(self) -> None:
        chunks = [
            DocumentChunk("rag.md", 1, "RAG 会先检索相关资料。"),
            DocumentChunk("python.md", 2, "Python 函数可以复用代码。"),
        ]

        with tempfile.TemporaryDirectory() as directory:
            vector_store = rebuild_vector_store(
                chunks,
                KeywordEmbeddings(),
                Path(directory),
            )
            try:
                results = search_vector_store("RAG 是什么？", vector_store, limit=1)

                self.assertEqual(len(vector_store.get()["ids"]), 2)
                self.assertEqual(results[0][0].metadata["source"], "rag.md")
                self.assertEqual(results[0][0].metadata["chunk_index"], 1)
            finally:
                vector_store.delete_collection()
                vector_store._client.close()

    def test_rejects_empty_chunk_list(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                rebuild_vector_store([], KeywordEmbeddings(), Path(directory))

    def test_sync_skips_unchanged_documents(self) -> None:
        chunks = [
            DocumentChunk("rag.md", 1, "RAG 会先检索相关资料。"),
            DocumentChunk("python.md", 1, "Python 函数可以复用代码。"),
        ]
        embeddings = CountingEmbeddings()

        with tempfile.TemporaryDirectory() as directory:
            first_result = sync_vector_store(chunks, embeddings, Path(directory))
            self.assertEqual(first_result.added, 2)
            first_result.vector_store._client.close()

            second_result = sync_vector_store(chunks, embeddings, Path(directory))
            try:
                self.assertEqual(embeddings.document_count, 2)
                self.assertEqual(second_result.added, 0)
                self.assertEqual(second_result.deleted, 0)
                self.assertEqual(second_result.unchanged, 2)
                self.assertEqual(len(second_result.vector_store.get(include=[])["ids"]), 2)
            finally:
                second_result.vector_store.delete_collection()
                second_result.vector_store._client.close()

    def test_sync_embeds_only_new_document(self) -> None:
        original_chunks = [DocumentChunk("rag.md", 1, "RAG 会先检索相关资料。")]
        expanded_chunks = [
            *original_chunks,
            DocumentChunk("python.md", 1, "Python 函数可以复用代码。"),
        ]
        embeddings = CountingEmbeddings()

        with tempfile.TemporaryDirectory() as directory:
            initial_result = sync_vector_store(original_chunks, embeddings, Path(directory))
            initial_result.vector_store._client.close()

            result = sync_vector_store(expanded_chunks, embeddings, Path(directory))
            try:
                self.assertEqual(result.added, 1)
                self.assertEqual(result.deleted, 0)
                self.assertEqual(result.unchanged, 1)
                self.assertEqual(embeddings.document_count, 2)
                self.assertEqual(len(result.vector_store.get(include=[])["ids"]), 2)
            finally:
                result.vector_store.delete_collection()
                result.vector_store._client.close()

    def test_sync_adds_changed_document_and_deletes_stale_records(self) -> None:
        original_chunks = [
            DocumentChunk("rag.md", 1, "RAG 会先检索相关资料。"),
            DocumentChunk("python.md", 1, "Python 函数可以复用代码。"),
        ]
        changed_chunks = [
            DocumentChunk("rag.md", 1, "RAG 会检索资料并生成回答。"),
        ]
        embeddings = CountingEmbeddings()

        with tempfile.TemporaryDirectory() as directory:
            initial_result = sync_vector_store(original_chunks, embeddings, Path(directory))
            initial_result.vector_store._client.close()

            result = sync_vector_store(changed_chunks, embeddings, Path(directory))
            try:
                stored = result.vector_store.get(include=["documents", "metadatas"])

                self.assertEqual(result.added, 1)
                self.assertEqual(result.deleted, 2)
                self.assertEqual(result.unchanged, 0)
                self.assertEqual(embeddings.document_count, 3)
                self.assertEqual(stored["documents"], ["RAG 会检索资料并生成回答。"])
                self.assertEqual(stored["metadatas"][0]["source"], "rag.md")
            finally:
                result.vector_store.delete_collection()
                result.vector_store._client.close()

    def test_sync_rejects_empty_chunk_list_without_deleting_existing_index(self) -> None:
        chunks = [DocumentChunk("rag.md", 1, "RAG 会先检索相关资料。")]
        embeddings = CountingEmbeddings()

        with tempfile.TemporaryDirectory() as directory:
            initial_result = sync_vector_store(chunks, embeddings, Path(directory))
            with self.assertRaises(ValueError):
                sync_vector_store([], embeddings, Path(directory))
            try:
                self.assertEqual(
                    len(initial_result.vector_store.get(include=[])["ids"]),
                    1,
                )
            finally:
                initial_result.vector_store.delete_collection()
                initial_result.vector_store._client.close()

    def test_sync_rejects_duplicate_document_ids(self) -> None:
        duplicate_chunk = DocumentChunk("rag.md", 1, "RAG 会先检索相关资料。")

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "重复 ID"):
                sync_vector_store(
                    [duplicate_chunk, duplicate_chunk],
                    CountingEmbeddings(),
                    Path(directory),
                )

    def test_sync_can_explicitly_clear_the_last_material(self) -> None:
        chunks = [DocumentChunk("rag.md", 1, "RAG 会先检索相关资料。")]
        embeddings = CountingEmbeddings()

        with tempfile.TemporaryDirectory() as directory:
            initial = sync_vector_store(chunks, embeddings, Path(directory))
            initial.vector_store._client.close()

            result = sync_vector_store(
                [],
                embeddings,
                Path(directory),
                allow_empty=True,
            )
            try:
                self.assertEqual(result.added, 0)
                self.assertEqual(result.deleted, 1)
                self.assertEqual(result.unchanged, 0)
                self.assertEqual(result.vector_store.get(include=[])["ids"], [])
            finally:
                result.vector_store.delete_collection()
                result.vector_store._client.close()

    def test_deletes_only_records_owned_by_one_material_without_embedding(self) -> None:
        chunks = [
            DocumentChunk("chapter.pdf · 第 1 页", 1, "RAG 第一页。"),
            DocumentChunk("chapter.pdf · 第 2 页", 1, "RAG 第二页。"),
            DocumentChunk("chapter-extra.pdf · 第 1 页", 1, "其他资料。"),
            DocumentChunk("chapter.pdf · 第 1 页.md", 1, "名字相似的 Markdown。"),
        ]

        with tempfile.TemporaryDirectory() as directory:
            initial = rebuild_vector_store(chunks, KeywordEmbeddings(), Path(directory))
            initial._client.close()

            deleted = delete_material_documents("chapter.pdf", Path(directory))

            remaining = open_vector_store(None, Path(directory))
            try:
                self.assertEqual(deleted, 2)
                stored = remaining.get(include=["metadatas"])
                self.assertEqual(len(stored["ids"]), 2)
                self.assertEqual(
                    {metadata["source"] for metadata in stored["metadatas"]},
                    {
                        "chapter-extra.pdf · 第 1 页",
                        "chapter.pdf · 第 1 页.md",
                    },
                )
            finally:
                remaining.delete_collection()
                close_vector_store(remaining)
