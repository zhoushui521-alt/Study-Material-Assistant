import json
import tempfile
import unittest
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock, patch

from langchain_core.documents import Document

from app.evaluate_retrieval import main as retrieval_main
from app.index_manifest import IndexCompatibilityStatus
from app.retrieval_evaluation import (
    DEFAULT_RETRIEVAL_EVALUATION_PATH,
    CurrentChunkMapping,
    FailureCategory,
    RetrievalConfig,
    RetrievalEvaluationCase,
    RetrievalEvaluationDataError,
    RetrievalTrace,
    StableGoldMeaning,
    build_retrieval_report,
    context_precision,
    diagnose_end_to_end_failure,
    diagnose_retrieval_failure,
    evaluate_citation_validity,
    evaluate_retrieval_cases,
    find_unresolved_gold_mappings,
    load_retrieval_cases,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    trace_retrieval,
    write_retrieval_report,
)


MATERIAL_ID = "a" * 64


def make_document(
    index: int,
    text: str,
    *,
    source: str = "notes.md",
) -> Document:
    return Document(
        page_content=text,
        metadata={
            "source": source,
            "filename": source,
            "source_type": "markdown",
            "chunk_index": index,
            "material_id": MATERIAL_ID,
            "chunk_id": f"{index:064x}",
            "content_hash": f"{index + 100:064x}",
        },
    )


def make_case(
    gold_indexes: tuple[int, ...] = (2,),
    *,
    answerable: bool = True,
) -> RetrievalEvaluationCase:
    mappings = tuple(
        CurrentChunkMapping(
            gold_ids=("gold_fact",),
            chunk_id=f"{index:064x}",
            material_id=MATERIAL_ID,
            content_hash=f"{index + 100:064x}",
            source="notes.md",
            legacy_chunk_index=index,
        )
        for index in gold_indexes
    )
    meanings = (
        StableGoldMeaning(
            gold_id="gold_fact",
            material_id=MATERIAL_ID,
            source="notes.md",
            locator="notes.md#gold",
            text_span="gold fact",
        ),
    ) if answerable else ()
    return RetrievalEvaluationCase(
        case_id="case",
        dataset_version="test-v1",
        question="gold question",
        answerable=answerable,
        case_types=("explicit_fact",) if answerable else ("unanswerable",),
        expected_material_ids=(MATERIAL_ID,) if answerable else (),
        expected_sources=("notes.md",) if answerable else (),
        stable_gold_meanings=meanings,
        current_chunk_mappings=mappings if answerable else (),
        annotation_notes="test",
    )


class FakeVectorStore:
    def __init__(
        self,
        results: list[tuple[Document, float]],
        stored_documents: list[Document] | None = None,
    ) -> None:
        self.results = results
        self.stored_documents = stored_documents or [document for document, _ in results]
        self.mutation_calls = 0

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

    def add_documents(self, *args: object, **kwargs: object) -> None:
        self.mutation_calls += 1
        raise AssertionError("Evaluation 不得写入 legacy index。")

    delete = add_documents
    reset_collection = add_documents


def empty_trace() -> RetrievalTrace:
    return RetrievalTrace(
        query={"case_id": "case", "question": "q"},
        raw_candidates=(),
        filtering={
            "threshold": 0.25,
            "before_count": 0,
            "after_count": 0,
            "candidates": (),
        },
        hybrid_ranking=(),
        seeds=(),
        adjacent_expansion=(),
        final_context={"final_chunk_ids": [], "chunk_count": 0},
        latency_ms={
            "raw_vector_search": 0,
            "filter_and_hybrid_ranking": 0,
            "context_construction": 0,
            "total_retrieval": 0,
        },
    )


class RetrievalDatasetTests(unittest.TestCase):
    def test_default_dataset_is_versioned_and_covers_required_case_types(self) -> None:
        cases = load_retrieval_cases(DEFAULT_RETRIEVAL_EVALUATION_PATH)

        self.assertEqual(len(cases), 10)
        self.assertEqual(len({case.case_id for case in cases}), 10)
        types = {item for case in cases for item in case.case_types}
        self.assertTrue(
            {
                "explicit_fact",
                "semantic_paraphrase",
                "exact_term",
                "cross_chunk",
                "similar_material_distractor",
                "unanswerable",
                "citation_sensitive",
            }.issubset(types)
        )
        self.assertTrue(all(case.dataset_version == "study-material-retrieval-v1" for case in cases))

    def test_rejects_duplicate_case_id(self) -> None:
        payload = json.loads(DEFAULT_RETRIEVAL_EVALUATION_PATH.read_text(encoding="utf-8"))
        payload["cases"].append(payload["cases"][0])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.json"
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            with self.assertRaisesRegex(RetrievalEvaluationDataError, "重复 case_id"):
                load_retrieval_cases(path)

    def test_rejects_unanswerable_case_with_fake_gold(self) -> None:
        payload = json.loads(DEFAULT_RETRIEVAL_EVALUATION_PATH.read_text(encoding="utf-8"))
        payload["cases"][0]["answerable"] = False
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.json"
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            with self.assertRaisesRegex(RetrievalEvaluationDataError, "不可回答"):
                load_retrieval_cases(path)

    def test_reports_gold_mapping_that_cannot_be_located(self) -> None:
        case = make_case((2,))

        missing = find_unresolved_gold_mappings([case], [make_document(1, "other")])

        self.assertEqual(missing, (f"case:{2:064x}",))


class RetrievalTraceTests(unittest.TestCase):
    def test_trace_records_raw_filter_rank_seed_adjacent_and_final_context(self) -> None:
        documents = [
            make_document(1, "other"),
            make_document(2, "gold gold"),
            make_document(3, "neighbor"),
        ]
        store = FakeVectorStore(
            [(documents[0], 0.9), (documents[1], 0.8), (documents[2], 0.2)],
            documents,
        )

        trace = trace_retrieval(
            make_case(),
            store,
            RetrievalConfig(retrieval_limit=1, adjacent_window=1),
        )

        self.assertEqual([item["raw_rank"] for item in trace.raw_candidates], [1, 2, 3])
        self.assertEqual(trace.filtering["before_count"], 3)
        self.assertEqual(trace.filtering["after_count"], 2)
        self.assertEqual(trace.hybrid_ranking[0]["chunk_id"], f"{2:064x}")
        self.assertEqual(trace.seeds[0]["seed_rank"], 1)
        self.assertEqual(
            [item["role"] for item in trace.adjacent_expansion],
            ["seed", "adjacent", "adjacent"],
        )
        self.assertEqual(trace.final_context["chunk_count"], 3)
        self.assertIsNone(trace.final_context["token_size"])
        self.assertNotIn("gold gold", json.dumps(trace.__dict__, ensure_ascii=False))
        self.assertEqual(store.mutation_calls, 0)

    def test_empty_retrieval_has_complete_zero_count_trace(self) -> None:
        trace = trace_retrieval(
            make_case(),
            FakeVectorStore([]),
            RetrievalConfig(),
        )

        self.assertEqual(trace.raw_candidates, ())
        self.assertEqual(trace.filtering["before_count"], 0)
        self.assertEqual(trace.final_context["chunk_count"], 0)


class RetrievalMetricTests(unittest.TestCase):
    def test_recall_mrr_and_ndcg_use_multi_gold_binary_relevance(self) -> None:
        ranked = ["other", "g2", "g1"]
        gold = {"g1", "g2"}

        self.assertEqual(recall_at_k(ranked, gold, 1), 0.0)
        self.assertEqual(recall_at_k(ranked, gold, 3), 1.0)
        self.assertEqual(reciprocal_rank(ranked, gold), 0.5)
        self.assertGreater(ndcg_at_k(ranked, gold, 3), 0.6)
        self.assertLess(ndcg_at_k(ranked, gold, 3), 1.0)

    def test_context_precision_handles_multi_gold_empty_and_unanswerable(self) -> None:
        self.assertEqual(context_precision(["g1", "noise", "g2"], {"g1", "g2"}, answerable=True), 2 / 3)
        self.assertEqual(context_precision([], {"g1"}, answerable=True), 0.0)
        self.assertEqual(context_precision([], set(), answerable=False), 1.0)
        self.assertEqual(context_precision(["noise"], set(), answerable=False), 0.0)

    def test_case_metrics_include_raw_and_ranked_recall(self) -> None:
        gold = make_document(2, "gold")
        result = evaluate_retrieval_cases(
            [make_case()],
            FakeVectorStore([(gold, 0.9)], [gold]),
            RetrievalConfig(adjacent_window=0),
        )[0]

        self.assertEqual(result.metrics["raw_recall_at_1"], 1.0)
        self.assertEqual(result.metrics["ranked_mrr"], 1.0)
        self.assertEqual(result.metrics["context_precision"], 1.0)


class FailureTaxonomyTests(unittest.TestCase):
    def test_classifies_recall_filtering_and_ranking_failures(self) -> None:
        case = make_case()
        self.assertEqual(
            diagnose_retrieval_failure(case, empty_trace()),
            FailureCategory.RECALL_FAILURE,
        )

        filtered = replace(
            empty_trace(),
            raw_candidates=({"chunk_id": f"{2:064x}"},),
        )
        self.assertEqual(
            diagnose_retrieval_failure(case, filtered),
            FailureCategory.FILTERING_FAILURE,
        )

        ranked = replace(
            filtered,
            hybrid_ranking=({"chunk_id": f"{2:064x}"},),
        )
        self.assertEqual(
            diagnose_retrieval_failure(case, ranked),
            FailureCategory.RANKING_FAILURE,
        )

    def test_classifies_context_and_unanswerable_failures(self) -> None:
        case = make_case()
        gold_id = f"{2:064x}"
        trace = replace(
            empty_trace(),
            raw_candidates=({"chunk_id": gold_id},),
            hybrid_ranking=({"chunk_id": gold_id},),
            seeds=({"chunk_id": gold_id},),
        )
        self.assertEqual(
            diagnose_retrieval_failure(case, trace),
            FailureCategory.CONTEXT_CONSTRUCTION_FAILURE,
        )
        unanswerable_trace = replace(
            empty_trace(),
            final_context={"final_chunk_ids": ["noise"], "chunk_count": 1},
        )
        self.assertEqual(
            diagnose_retrieval_failure(make_case((), answerable=False), unanswerable_trace),
            FailureCategory.UNANSWERABLE_HANDLING_FAILURE,
        )

    def test_end_to_end_taxonomy_uses_explicit_parser_generation_and_citation_signals(self) -> None:
        answerable = make_case()
        good_trace = replace(
            empty_trace(),
            raw_candidates=({"chunk_id": f"{2:064x}"},),
            hybrid_ranking=({"chunk_id": f"{2:064x}"},),
            seeds=({"chunk_id": f"{2:064x}"},),
            final_context={"final_chunk_ids": [f"{2:064x}"], "chunk_count": 1},
        )

        self.assertEqual(
            diagnose_end_to_end_failure(answerable, good_trace, parsing_failed=True),
            FailureCategory.PARSING_FAILURE,
        )
        self.assertEqual(
            diagnose_end_to_end_failure(answerable, good_trace, generation_failed=True),
            FailureCategory.GENERATION_FAILURE,
        )
        self.assertEqual(
            diagnose_end_to_end_failure(answerable, good_trace, citation_valid=False),
            FailureCategory.CITATION_FAILURE,
        )
        self.assertEqual(
            diagnose_end_to_end_failure(answerable, good_trace, citation_valid=True),
            FailureCategory.NEEDS_MANUAL_REVIEW,
        )


class CitationAndReportTests(unittest.TestCase):
    def test_citation_validity_does_not_claim_coverage_or_support(self) -> None:
        result = evaluate_citation_validity(["S1", "S9", "S1"], {"S1", "S2"})

        self.assertEqual(result["valid_ids"], ["S1"])
        self.assertEqual(result["invalid_ids"], ["S9"])
        self.assertEqual(result["validity_ratio"], 0.5)
        self.assertIsNone(result["coverage"])
        self.assertIsNone(result["support"])

    def test_report_schema_marks_fixture_level_and_omits_token_cost_claims(self) -> None:
        gold = make_document(2, "gold")
        results = evaluate_retrieval_cases(
            [make_case()],
            FakeVectorStore([(gold, 0.9)], [gold]),
            RetrievalConfig(adjacent_window=0),
        )

        report = build_retrieval_report(
            results,
            dataset_path=DEFAULT_RETRIEVAL_EVALUATION_PATH,
            config=RetrievalConfig(adjacent_window=0),
            git_commit="d" * 64,
            stage2_start_commit="e" * 64,
            generated_at=datetime(2026, 8, 19, tzinfo=UTC),
            validation_level="fixture",
            embedding_model=None,
            query_embedding_calls=0,
        )

        self.assertEqual(report["report_schema_version"], 2)
        self.assertEqual(report["evaluation_type"], "retrieval")
        self.assertEqual(report["validation_level"], "fixture")
        self.assertFalse(report["tokens"]["measured"])
        self.assertFalse(report["cost"]["measured"])
        self.assertIn("per_case_results", report)

    def test_report_writer_never_overwrites(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = write_retrieval_report({"value": "中文"}, Path(directory))
            second = write_retrieval_report({"value": "中文"}, Path(directory))

            self.assertNotEqual(first, second)
            self.assertEqual(len(list(Path(directory).glob("*.json"))), 2)


class RetrievalRunnerTests(unittest.TestCase):
    @patch("app.evaluate_retrieval.open_vector_store")
    def test_cost_gate_stops_before_opening_index(self, open_vector_store: Mock) -> None:
        with patch("builtins.print"):
            exit_code = retrieval_main([])

        self.assertEqual(exit_code, 2)
        open_vector_store.assert_not_called()

    @patch("app.evaluate_retrieval.write_retrieval_report")
    @patch("app.evaluate_retrieval.build_retrieval_report")
    @patch("app.evaluate_retrieval.evaluate_retrieval_cases", return_value=())
    @patch("app.evaluate_retrieval.find_unresolved_gold_mappings", return_value=())
    @patch("app.evaluate_retrieval.check_index_compatibility")
    @patch("app.evaluate_retrieval.open_vector_store")
    @patch("app.evaluate_retrieval.create_langchain_embeddings")
    @patch("app.evaluate_retrieval.EmbeddingConfig.from_environment")
    @patch("app.evaluate_retrieval.close_vector_store")
    def test_legacy_index_evaluation_is_read_only(
        self,
        close_vector_store: Mock,
        from_environment: Mock,
        create_embeddings: Mock,
        open_vector_store: Mock,
        check_compatibility: Mock,
        find_unresolved: Mock,
        evaluate_cases: Mock,
        build_report: Mock,
        write_report: Mock,
    ) -> None:
        config = Mock(model="embedding-model")
        from_environment.return_value = config
        store = Mock()
        store.get.side_effect = [
            {"ids": ["legacy"]},
            {"ids": [], "documents": [], "metadatas": []},
            {"ids": [], "documents": [], "metadatas": []},
        ]
        open_vector_store.return_value = store
        check_compatibility.return_value = IndexCompatibilityStatus.LEGACY_READ_ONLY
        build_report.return_value = {"report_schema_version": 2}
        write_report.return_value = Path("report.json")

        with patch("app.evaluate_retrieval._git_commit", return_value="d" * 64), patch(
            "builtins.print"
        ):
            exit_code = retrieval_main(["--confirm-query-embedding-cost"])

        self.assertEqual(exit_code, 0)
        self.assertEqual(check_compatibility.call_args.kwargs["access"], "read")
        store.add_documents.assert_not_called()
        store.delete.assert_not_called()
        store.reset_collection.assert_not_called()
        close_vector_store.assert_called_once_with(store)


if __name__ == "__main__":
    unittest.main()
