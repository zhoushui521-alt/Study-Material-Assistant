import unittest

from langchain_core.documents import Document

from app.hybrid_search import (
    expand_adjacent_documents,
    hybrid_search_vector_store,
    keyword_coverage,
)


class FakeVectorStore:
    def __init__(
        self,
        results: list[tuple[Document, float]],
        stored_documents: list[Document] | None = None,
    ) -> None:
        self.results = results
        self.stored_documents = stored_documents or []

    def similarity_search_with_relevance_scores(
        self,
        question: str,
        k: int,
    ) -> list[tuple[Document, float]]:
        return self.results[:k]

    def get(self, where: dict[str, str], include: list[str]) -> dict[str, list]:
        documents = [
            document
            for document in self.stored_documents
            if document.metadata.get("source") == where["source"]
        ]
        return {
            "documents": [document.page_content for document in documents],
            "metadatas": [document.metadata for document in documents],
        }


class HybridSearchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rag_document = Document(
            page_content="整份资料可能太长，把资料切成较小的文本块。",
            metadata={"source": "rag-intro.md", "chunk_index": 2},
        )
        self.chroma_document = Document(
            page_content=(
                "资料没有变化时，增量同步会复用已有向量；"
                "只有新增或修改的文本块才需要重新调用 Embedding。"
            ),
            metadata={"source": "chroma-intro.md", "chunk_index": 1},
        )

    def test_keyword_coverage_is_a_ratio(self) -> None:
        score = keyword_coverage("重新调用 Embedding", self.chroma_document.page_content)

        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_exact_terms_can_rerank_close_vector_results(self) -> None:
        vector_store = FakeVectorStore(
            [
                (self.rag_document, 0.4462),
                (self.chroma_document, 0.4273),
            ]
        )

        results = hybrid_search_vector_store(
            "资料没有变化时，为什么不需要重新调用 Embedding？",
            vector_store,
        )

        self.assertEqual(results[0].document.metadata["source"], "chroma-intro.md")
        self.assertGreater(results[0].keyword_score, results[1].keyword_score)

    def test_vector_order_is_preserved_without_keyword_matches(self) -> None:
        vector_store = FakeVectorStore(
            [
                (self.rag_document, 0.8),
                (self.chroma_document, 0.5),
            ]
        )

        results = hybrid_search_vector_store("oversized corpus remedy", vector_store)

        self.assertEqual(results[0].document.metadata["source"], "rag-intro.md")
        self.assertEqual(results[0].keyword_score, 0.0)

    def test_filters_candidates_below_relevance_threshold(self) -> None:
        vector_store = FakeVectorStore([(self.rag_document, 0.2)])

        results = hybrid_search_vector_store(
            "资料太长怎么办？",
            vector_store,
            relevance_threshold=0.25,
        )

        self.assertEqual(results, [])

    def test_returns_empty_list_for_empty_question(self) -> None:
        self.assertEqual(
            hybrid_search_vector_store("  ", FakeVectorStore([])),
            [],
        )

    def test_expands_adjacent_chunks_from_the_same_source(self) -> None:
        source = "chapter.pdf · 第 13 页"
        chunks = [
            Document(
                page_content=f"第 {index} 段",
                metadata={"source": source, "chunk_index": index},
            )
            for index in range(1, 4)
        ]
        other_source = Document(
            page_content="其他资料",
            metadata={"source": "other.pdf · 第 1 页", "chunk_index": 1},
        )
        vector_store = FakeVectorStore([], [*chunks, other_source])

        expanded = expand_adjacent_documents(
            vector_store,
            [chunks[1]],
            adjacent_window=1,
            context_limit=8,
        )

        self.assertEqual(
            [document.metadata["chunk_index"] for document in expanded],
            [2, 1, 3],
        )
        self.assertTrue(
            all(document.metadata["source"] == source for document in expanded)
        )

    def test_adjacent_expansion_deduplicates_and_respects_context_limit(self) -> None:
        source = "chapter.pdf · 第 13 页"
        chunks = [
            Document(
                page_content=f"第 {index} 段",
                metadata={"source": source, "chunk_index": index},
            )
            for index in range(1, 4)
        ]
        vector_store = FakeVectorStore([], chunks)

        expanded = expand_adjacent_documents(
            vector_store,
            [chunks[0], chunks[1]],
            adjacent_window=1,
            context_limit=3,
        )

        self.assertEqual(
            [document.metadata["chunk_index"] for document in expanded],
            [1, 2, 3],
        )

    def test_expansion_recovers_the_missing_chunk_next_to_a_ranked_seed(self) -> None:
        source = "第八章.pdf · 第 13 页"
        chunks = [
            Document(
                page_content=f"第 {index} 段",
                metadata={"source": source, "chunk_index": index},
            )
            for index in range(1, 6)
        ]
        vector_store = FakeVectorStore([], chunks)

        expanded = expand_adjacent_documents(
            vector_store,
            [chunks[3], chunks[0], chunks[1]],
            adjacent_window=1,
            context_limit=8,
        )

        self.assertEqual(
            [document.metadata["chunk_index"] for document in expanded],
            [4, 1, 2, 3, 5],
        )

    def test_two_chunk_window_recovers_a_section_definition(self) -> None:
        source = "第八章.pdf · 第 17 页"
        chunks = [
            Document(
                page_content=f"第 {index} 段",
                metadata={"source": source, "chunk_index": index},
            )
            for index in range(1, 7)
        ]
        vector_store = FakeVectorStore([], chunks)

        expanded = expand_adjacent_documents(
            vector_store,
            [chunks[5]],
            adjacent_window=2,
            context_limit=8,
        )

        self.assertEqual(
            [document.metadata["chunk_index"] for document in expanded],
            [6, 5, 4],
        )

    def test_adjacent_expansion_validates_window_and_limit(self) -> None:
        vector_store = FakeVectorStore([])

        with self.assertRaisesRegex(ValueError, "窗口"):
            expand_adjacent_documents(vector_store, [], adjacent_window=-1)
        with self.assertRaisesRegex(ValueError, "数量"):
            expand_adjacent_documents(vector_store, [], context_limit=0)


if __name__ == "__main__":
    unittest.main()
