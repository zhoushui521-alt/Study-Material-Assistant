import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from langchain_core.documents import Document

from app.hybrid_search import BM25Index
from app.evaluate_reranker import main as evaluate_reranker_main
from app.reranker import DeterministicMockReranker
from app.reranker_evaluation import (
    RERANKER_EXPERIMENT_NAME,
    RerankerEvaluationConfig,
    build_reranker_report,
    evaluate_reranker_case,
    write_reranker_report,
)
from app.retrieval_evaluation import (
    CurrentChunkMapping,
    RetrievalEvaluationCase,
    StableGoldMeaning,
)


MATERIAL_ID = "a" * 64


def make_document(number: int, content: str, source: str = "notes.md") -> Document:
    return Document(
        page_content=content,
        metadata={
            "source": source,
            "filename": source,
            "source_type": "markdown",
            "chunk_index": number,
            "chunk_id": f"{number:064x}",
            "material_id": MATERIAL_ID,
            "content_hash": f"{number + 100:064x}",
        },
    )


def make_case(gold: Document, question: str) -> RetrievalEvaluationCase:
    chunk_id = str(gold.metadata["chunk_id"])
    content_hash = str(gold.metadata["content_hash"])
    source = str(gold.metadata["source"])
    return RetrievalEvaluationCase(
        case_id="case",
        dataset_version="study-material-retrieval-v1",
        question=question,
        answerable=True,
        case_types=("ranking",),
        expected_material_ids=(MATERIAL_ID,),
        expected_sources=(source,),
        stable_gold_meanings=(
            StableGoldMeaning(
                gold_id="gold-case",
                material_id=MATERIAL_ID,
                source=source,
                locator=f"{source}#chunk=1",
                text_span=gold.page_content,
            ),
        ),
        current_chunk_mappings=(
            CurrentChunkMapping(
                gold_ids=("gold-case",),
                chunk_id=chunk_id,
                material_id=MATERIAL_ID,
                content_hash=content_hash,
                source=source,
                legacy_chunk_index=int(gold.metadata["chunk_index"]),
            ),
        ),
        annotation_notes="fixture",
    )


class FakeVectorStore:
    def __init__(self, dense_results: list[tuple[Document, float]], corpus: list[Document]) -> None:
        self.dense_results = dense_results
        self.corpus = corpus

    def similarity_search_with_relevance_scores(
        self, question: str, k: int
    ) -> list[tuple[Document, float]]:
        return self.dense_results[:k]

    def get(
        self, include: list[str], where: dict[str, str] | None = None
    ) -> dict[str, list]:
        documents = [
            document
            for document in self.corpus
            if where is None or document.metadata["source"] == where["source"]
        ]
        return {
            "ids": [str(document.metadata["chunk_id"]) for document in documents],
            "documents": [document.page_content for document in documents],
            "metadatas": [document.metadata for document in documents],
        }


class RerankerEvaluationTests(unittest.TestCase):
    def test_runner_requires_both_explicit_confirmations(self) -> None:
        self.assertEqual(evaluate_reranker_main([]), 2)
        self.assertEqual(
            evaluate_reranker_main(["--confirm-query-embedding-cost"]),
            2,
        )
        self.assertEqual(
            evaluate_reranker_main(
                ["--confirm-model-download-and-local-inference"]
            ),
            2,
        )

    def test_reranker_moves_gold_forward_and_records_trace(self) -> None:
        distractor = make_document(1, "ExactAPI ExactAPI")
        gold = make_document(2, "semantic gold")
        store = FakeVectorStore(
            [(gold, 0.9), (distractor, 0.8)],
            [gold, distractor],
        )
        reranker = DeterministicMockReranker(
            {str(gold.metadata["chunk_id"]): 0.9, str(distractor.metadata["chunk_id"]): 0.1}
        )

        result = evaluate_reranker_case(
            make_case(gold, "ExactAPI"),
            store,
            BM25Index.from_documents(store.corpus),
            reranker,
            RerankerEvaluationConfig(adjacent_window=0),
        )

        self.assertEqual(result.hybrid_metrics["recall_at_1"], 0.0)
        self.assertEqual(result.reranker_metrics["recall_at_1"], 1.0)
        self.assertEqual(result.case_analysis["reranker_vs_hybrid_outcome"], "ranking_improved")
        gold_trace = next(
            item
            for item in result.trace["reranker_ranking"]
            if item["chunk_id"] == gold.metadata["chunk_id"]
        )
        self.assertEqual(gold_trace["original_rank"], 2)
        self.assertEqual(gold_trace["reranker_rank"], 1)
        self.assertEqual(gold_trace["rank_change"], 1)
        self.assertNotIn("semantic gold", str(result.trace))

    def test_candidate_pool_is_bounded_and_context_metrics_are_reported(self) -> None:
        documents = [make_document(index, f"term {index}") for index in range(1, 5)]
        store = FakeVectorStore(
            [(document, 0.9 - index / 10) for index, document in enumerate(documents)],
            documents,
        )
        reranker = DeterministicMockReranker(
            {str(document.metadata["chunk_id"]): float(index) for index, document in enumerate(documents)}
        )
        config = RerankerEvaluationConfig(
            dense_candidate_limit=2,
            bm25_candidate_limit=2,
            reranker_candidate_limit=4,
            adjacent_window=0,
        )

        result = evaluate_reranker_case(
            make_case(documents[0], "term"),
            store,
            BM25Index.from_documents(store.corpus),
            reranker,
            config,
        )

        self.assertLessEqual(result.trace["candidate_pool"]["actual_size"], 4)
        self.assertFalse(result.trace["candidate_pool"]["corpus_reranked"])
        self.assertEqual(result.context_metrics["reranker"]["context_size"], 2)
        self.assertEqual(result.context_metrics["reranker"]["duplicate_ratio"], 0.0)
        self.assertIn("context_precision", result.context_metrics["reranker"])
        self.assertIn("final_context_recall", result.context_metrics["reranker"])

    def test_report_schema_compares_all_strategies_and_exposes_regression(self) -> None:
        gold = make_document(1, "gold")
        distractor = make_document(2, "distractor")
        store = FakeVectorStore([(gold, 0.9), (distractor, 0.8)], [gold, distractor])
        reranker = DeterministicMockReranker(
            {str(gold.metadata["chunk_id"]): 0.1, str(distractor.metadata["chunk_id"]): 0.9}
        )
        config = RerankerEvaluationConfig(adjacent_window=0)
        result = evaluate_reranker_case(
            make_case(gold, "gold"),
            store,
            BM25Index.from_documents(store.corpus),
            reranker,
            config,
        )

        report = build_reranker_report(
            [result],
            config=config,
            reranker=reranker,
            git_commit="d" * 40,
            stage3_2_start_commit="e" * 40,
            dataset_metadata={"dataset_version": "study-material-retrieval-v1"},
            embedding_model=None,
            query_embedding_calls=0,
            validation_level="fixture",
            index_fingerprint="f" * 64,
            generated_at=datetime(2026, 8, 20, tzinfo=UTC),
        )

        self.assertEqual(report["report_schema_version"], 1)
        self.assertEqual(report["experiment_name"], RERANKER_EXPERIMENT_NAME)
        self.assertEqual(set(report["metrics"]), {"baseline", "hybrid", "reranker"})
        self.assertLess(report["metric_deltas"]["reranker_vs_hybrid"]["recall_at_1"], 0)
        self.assertIn("reranker_average_incremental_ms", report["latency"])
        self.assertEqual(report["cost"]["query_embedding_calls"], 0)
        self.assertEqual(report["cost"]["chat_calls"], 0)

    def test_report_writer_never_overwrites(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            first = write_reranker_report({"value": "中文"}, output)
            second = write_reranker_report({"value": "中文"}, output)
            self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
