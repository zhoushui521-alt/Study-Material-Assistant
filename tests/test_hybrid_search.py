import unittest

from langchain_core.documents import Document

from app.hybrid_search import (
    BM25Index,
    BM25SearchResult,
    HybridRRFRetriever,
    expand_adjacent_documents,
    hybrid_rrf_search_vector_store,
    hybrid_search_vector_store,
    keyword_coverage,
    reciprocal_rank_fusion,
    tokenize_for_bm25,
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

    def get(
        self,
        include: list[str],
        where: dict[str, str] | None = None,
    ) -> dict[str, list]:
        documents = [
            document
            for document in self.stored_documents
            if where is None or document.metadata.get("source") == where["source"]
        ]
        return {
            "ids": [str(index) for index, _ in enumerate(documents)],
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


class BM25SearchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.documents = [
            Document(
                page_content="LangChain LCEL 使用 Runnable 组合固定 RAG 流程。",
                metadata={"source": "langchain.md", "chunk_index": 1},
            ),
            Document(
                page_content="威沙特分布用于协方差矩阵建模。",
                metadata={"source": "statistics.md", "chunk_index": 1},
            ),
            Document(
                page_content="FastAPI 提供 OpenAPI 接口文档。",
                metadata={"source": "api.md", "chunk_index": 1},
            ),
        ]
        self.index = BM25Index.from_documents(self.documents)

    def test_tokenizer_preserves_technical_terms_and_chinese_bigrams(self) -> None:
        tokens = tokenize_for_bm25("FastAPI 与威沙特分布")

        self.assertIn("fastapi", tokens)
        self.assertIn("威沙", tokens)
        self.assertIn("沙特", tokens)

    def test_basic_recall_returns_score_and_rank(self) -> None:
        results = self.index.search("LCEL Runnable", limit=3)

        self.assertEqual(results[0].document.metadata["source"], "langchain.md")
        self.assertGreater(results[0].score, 0.0)
        self.assertEqual(results[0].rank, 1)

    def test_chinese_exact_term_and_api_name_are_retrievable(self) -> None:
        chinese = self.index.search("威沙特分布", limit=1)
        api = self.index.search("FastAPI OpenAPI", limit=1)

        self.assertEqual(chinese[0].document.metadata["source"], "statistics.md")
        self.assertEqual(api[0].document.metadata["source"], "api.md")

    def test_empty_or_unmatched_query_returns_no_results(self) -> None:
        self.assertEqual(self.index.search("  ", limit=3), [])
        self.assertEqual(self.index.search("PostgreSQL MVCC", limit=3), [])

    def test_duplicate_document_is_indexed_once(self) -> None:
        index = BM25Index.from_documents([self.documents[0], self.documents[0]])

        self.assertEqual(len(index.documents), 1)
        self.assertEqual(len(index.search("LCEL", limit=3)), 1)


class RRFFusionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dense_first = Document(
            page_content="语义相关内容",
            metadata={"source": "dense.md", "chunk_index": 1},
        )
        self.shared = Document(
            page_content="RRF 融合内容",
            metadata={"source": "shared.md", "chunk_index": 1},
        )
        self.sparse_only = Document(
            page_content="精确术语 LCEL",
            metadata={"source": "sparse.md", "chunk_index": 1},
        )

    def _bm25(self, document: Document, score: float, rank: int) -> BM25SearchResult:
        chunk_id = BM25Index.from_documents([document]).chunk_ids[0]
        return BM25SearchResult(document, chunk_id, score, rank)

    def test_two_lists_fuse_shared_chunk_and_use_rank_formula(self) -> None:
        results = reciprocal_rank_fusion(
            [(self.dense_first, 0.9), (self.shared, 0.8)],
            [self._bm25(self.shared, 4.2, 1), self._bm25(self.sparse_only, 3.1, 2)],
            rrf_k=60,
        )

        self.assertIs(results[0].document, self.shared)
        self.assertEqual(results[0].dense_rank, 2)
        self.assertEqual(results[0].bm25_rank, 1)
        self.assertAlmostEqual(results[0].rrf_score, 1 / 62 + 1 / 61)
        self.assertEqual(len(results), 3)

    def test_dense_only_and_bm25_only_results_are_retained(self) -> None:
        dense_only = reciprocal_rank_fusion([(self.dense_first, 0.9)], [])
        sparse_only = reciprocal_rank_fusion([], [self._bm25(self.sparse_only, 2.0, 1)])

        self.assertEqual(dense_only[0].dense_rank, 1)
        self.assertIsNone(dense_only[0].bm25_rank)
        self.assertEqual(sparse_only[0].bm25_rank, 1)
        self.assertIsNone(sparse_only[0].dense_rank)

    def test_duplicate_dense_chunk_is_counted_once(self) -> None:
        results = reciprocal_rank_fusion(
            [(self.shared, 0.9), (self.shared, 0.8)],
            [self._bm25(self.shared, 2.0, 1)],
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].dense_rank, 1)

    def test_equal_rrf_scores_have_stable_chunk_id_order(self) -> None:
        first = reciprocal_rank_fusion(
            [(self.dense_first, 0.9)],
            [self._bm25(self.sparse_only, 2.0, 1)],
        )
        second = reciprocal_rank_fusion(
            [(self.dense_first, 0.9)],
            [self._bm25(self.sparse_only, 2.0, 1)],
        )

        self.assertEqual(
            [result.chunk_id for result in first],
            [result.chunk_id for result in second],
        )

    def test_true_hybrid_can_recover_document_absent_from_dense_candidates(self) -> None:
        store = FakeVectorStore(
            [(self.dense_first, 0.9)],
            [self.dense_first, self.sparse_only],
        )

        results = hybrid_rrf_search_vector_store(
            "LCEL",
            store,
            limit=3,
            dense_candidate_limit=1,
            bm25_candidate_limit=1,
        )

        sources = [result.document.metadata["source"] for result in results]
        self.assertIn("sparse.md", sources)
        sparse_result = next(result for result in results if result.bm25_rank == 1)
        self.assertIsNone(sparse_result.dense_rank)

    def test_experimental_retriever_exposes_true_hybrid_results(self) -> None:
        store = FakeVectorStore(
            [(self.dense_first, 0.9)],
            [self.dense_first, self.sparse_only],
        )
        retriever = HybridRRFRetriever.model_construct(
            vector_store=store,
            bm25_index=BM25Index.from_documents(store.stored_documents),
            limit=3,
            dense_candidate_limit=1,
            bm25_candidate_limit=1,
        )

        results = retriever.invoke("LCEL")

        self.assertEqual(
            {document.metadata["source"] for document in results},
            {"dense.md", "sparse.md"},
        )


if __name__ == "__main__":
    unittest.main()
