import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from langchain_core.documents import Document

from app.evaluate_hybrid_retrieval import _source_state, main as evaluate_hybrid_main
from app.hybrid_retrieval_evaluation import (
    HYBRID_BASELINE_NAME,
    HYBRID_EXPERIMENT_NAME,
    HybridRetrievalConfig,
    build_hybrid_report,
    evaluate_hybrid_case,
    write_hybrid_report,
)
from app.hybrid_search import BM25Index
from app.retrieval_evaluation import (
    CurrentChunkMapping,
    RetrievalEvaluationCase,
    StableGoldMeaning,
)


MATERIAL_ID = "a" * 64


def make_document(number: int, content: str, source: str = "notes.md") -> Document:
    chunk_id = f"{number:064x}"
    return Document(
        page_content=content,
        metadata={
            "source": source,
            "filename": source,
            "source_type": "markdown",
            "chunk_index": number,
            "chunk_id": chunk_id,
            "material_id": MATERIAL_ID,
            "content_hash": f"{number + 100:064x}",
        },
    )


def make_case(gold: Document, question: str, case_id: str = "case") -> RetrievalEvaluationCase:
    chunk_id = str(gold.metadata["chunk_id"])
    content_hash = str(gold.metadata["content_hash"])
    source = str(gold.metadata["source"])
    return RetrievalEvaluationCase(
        case_id=case_id,
        dataset_version="study-material-retrieval-v1",
        question=question,
        answerable=True,
        case_types=("exact_term",),
        expected_material_ids=(MATERIAL_ID,),
        expected_sources=(source,),
        stable_gold_meanings=(
            StableGoldMeaning(
                gold_id=f"gold-{case_id}",
                material_id=MATERIAL_ID,
                source=source,
                locator=f"{source}#chunk=1",
                text_span=gold.page_content,
                page=None,
                section=None,
            ),
        ),
        current_chunk_mappings=(
            CurrentChunkMapping(
                chunk_id=chunk_id,
                gold_ids=(f"gold-{case_id}",),
                material_id=MATERIAL_ID,
                content_hash=content_hash,
                source=source,
                page=None,
                section=None,
                legacy_chunk_index=int(gold.metadata["chunk_index"]),
            ),
        ),
        annotation_notes="fixture",
    )


class FakeVectorStore:
    def __init__(
        self,
        dense_results: list[tuple[Document, float]],
        corpus: list[Document],
    ) -> None:
        self.dense_results = dense_results
        self.corpus = corpus

    def similarity_search_with_relevance_scores(
        self, question: str, k: int
    ) -> list[tuple[Document, float]]:
        return self.dense_results[:k]

    def get(
        self,
        include: list[str],
        where: dict[str, str] | None = None,
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


class HybridEvaluationTests(unittest.TestCase):
    def test_bm25_adds_gold_that_dense_baseline_misses(self) -> None:
        distractor = make_document(1, "semantic distractor")
        gold = make_document(2, "LCEL RunnableParallel 精确术语")
        store = FakeVectorStore([(distractor, 0.9)], [distractor, gold])

        result = evaluate_hybrid_case(
            make_case(gold, "LCEL RunnableParallel"),
            store,
            BM25Index.from_documents(store.corpus),
            HybridRetrievalConfig(adjacent_window=0),
        )

        self.assertEqual(result.baseline_metrics["recall_at_10"], 0.0)
        self.assertEqual(result.metrics["recall_at_10"], 1.0)
        self.assertEqual(result.case_analysis["outcome"], "added_recall")
        self.assertEqual(result.failure_category, None)
        self.assertNotIn("LCEL RunnableParallel 精确术语", str(result.trace))

    def test_report_exposes_baseline_regression_instead_of_hiding_average(self) -> None:
        gold = make_document(1, "semantic evidence")
        distractor = make_document(2, "ExactAPI ExactAPI")
        store = FakeVectorStore(
            [(gold, 0.99), (distractor, 0.25)],
            [gold, distractor],
        )
        result = evaluate_hybrid_case(
            make_case(gold, "ExactAPI"),
            store,
            BM25Index.from_documents(store.corpus),
            HybridRetrievalConfig(adjacent_window=0),
        )

        self.assertEqual(result.baseline_metrics["recall_at_1"], 1.0)
        self.assertEqual(result.metrics["recall_at_1"], 0.0)
        self.assertEqual(result.metric_deltas["recall_at_1"], -1.0)
        self.assertEqual(result.case_analysis["outcome"], "ranking_regressed")

    def test_report_schema_contains_config_metrics_comparison_cases_and_latency(self) -> None:
        gold = make_document(1, "LCEL")
        store = FakeVectorStore([(gold, 0.9)], [gold])
        config = HybridRetrievalConfig(adjacent_window=0)
        result = evaluate_hybrid_case(
            make_case(gold, "LCEL"),
            store,
            BM25Index.from_documents(store.corpus),
            config,
        )

        report = build_hybrid_report(
            [result],
            config=config,
            git_commit="d" * 40,
            stage3_1_start_commit="e" * 40,
            dataset_metadata={"dataset_version": "study-material-retrieval-v1"},
            embedding_model=None,
            query_embedding_calls=0,
            validation_level="fixture",
            index_fingerprint="f" * 64,
            generated_at=datetime(2026, 8, 20, tzinfo=UTC),
        )

        self.assertEqual(report["report_schema_version"], 1)
        self.assertEqual(report["experiment_name"], HYBRID_EXPERIMENT_NAME)
        self.assertEqual(report["baseline"], HYBRID_BASELINE_NAME)
        self.assertIn("retrieval_config", report)
        self.assertIn("baseline_metrics", report)
        self.assertIn("metrics", report)
        self.assertIn("metric_deltas", report)
        self.assertIn("baseline_average_retrieval_ms", report["latency"])
        self.assertIn("experiment_average_retrieval_ms", report["latency"])
        self.assertEqual(len(report["per_case_results"]), 1)
        self.assertEqual(report["validation_level"], "fixture")
        self.assertEqual(report["run_status"], "completed")
        self.assertEqual(report["cost"]["query_embedding_calls"], 0)

    def test_report_writer_never_overwrites(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            first = write_hybrid_report({"value": "中文"}, output)
            second = write_hybrid_report({"value": "中文"}, output)

            self.assertNotEqual(first, second)
            self.assertEqual(len(list(output.glob("*.json"))), 2)

    def test_runner_requires_explicit_embedding_cost_confirmation(self) -> None:
        self.assertEqual(evaluate_hybrid_main([]), 2)

    def test_source_state_fingerprints_stage3_implementation_files(self) -> None:
        source_state = _source_state("d" * 40)

        self.assertEqual(source_state["base_git_commit"], "d" * 40)
        self.assertEqual(len(source_state["implementation_sha256"]), 64)
        self.assertEqual(len(source_state["implementation_files_sha256"]), 4)


if __name__ == "__main__":
    unittest.main()
