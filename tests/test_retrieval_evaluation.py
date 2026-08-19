import json
import tempfile
import unittest
from contextlib import nullcontext
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock, patch

from langchain_core.documents import Document

from app.evaluate_retrieval import (
    EvaluationIndexSnapshot,
    disposable_evaluation_snapshot,
    main as retrieval_main,
)
from app.index_manifest import IndexCompatibilityStatus
from app.retrieval_evaluation import (
    DEFAULT_RETRIEVAL_EVALUATION_PATH,
    CurrentChunkMapping,
    FailureCategory,
    RetrievalConfig,
    RetrievalEvaluationCase,
    RetrievalEvaluationDataError,
    RetrievalEvaluationReportError,
    RetrievalEvaluationRun,
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
    load_persisted_retrieval_report,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    rebuild_retrieval_report_aggregates,
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

    def test_run_persists_each_case_and_rebuilds_aggregates(self) -> None:
        good_result = evaluate_retrieval_cases(
            [make_case()],
            FakeVectorStore([(make_document(2, "gold"), 0.9)]),
            RetrievalConfig(adjacent_window=0),
        )[0]
        second_case = replace(make_case(), case_id="case-2")
        missed_result = evaluate_retrieval_cases(
            [second_case],
            FakeVectorStore([]),
            RetrievalConfig(adjacent_window=0),
        )[0]

        with tempfile.TemporaryDirectory() as directory:
            run = RetrievalEvaluationRun.create(
                dataset_path=DEFAULT_RETRIEVAL_EVALUATION_PATH,
                output_directory=Path(directory),
                config=RetrievalConfig(adjacent_window=0),
                git_commit="d" * 40,
                stage2_start_commit="e" * 40,
                started_at=datetime(2026, 8, 19, tzinfo=UTC),
                validation_level="fixture",
                embedding_model=None,
                original_index_fingerprint="f" * 64,
            )
            self.assertEqual(
                load_persisted_retrieval_report(run.path)["run_status"],
                "ready",
            )

            run.persist_case_result(good_result)
            after_first = load_persisted_retrieval_report(run.path)
            self.assertEqual(len(after_first["per_case_results"]), 1)
            self.assertEqual(after_first["cost"]["query_embedding_calls"], 1)

            run.persist_case_result(missed_result)
            after_second = load_persisted_retrieval_report(run.path)
            self.assertEqual(len(after_second["per_case_results"]), 2)
            self.assertEqual(after_second["aggregate_metrics"]["raw_recall_at_1"], 0.5)

            without_aggregates = dict(after_second)
            without_aggregates["aggregate_metrics"] = {"stale": 999}
            rebuilt = rebuild_retrieval_report_aggregates(without_aggregates)
            self.assertNotIn("stale", rebuilt["aggregate_metrics"])
            self.assertEqual(rebuilt["aggregate_metrics"]["raw_recall_at_1"], 0.5)

    def test_finalization_failure_keeps_completed_cases_recoverable(self) -> None:
        result = evaluate_retrieval_cases(
            [make_case()],
            FakeVectorStore([(make_document(2, "gold"), 0.9)]),
            RetrievalConfig(adjacent_window=0),
        )[0]
        with tempfile.TemporaryDirectory() as directory:
            run = RetrievalEvaluationRun.create(
                dataset_path=DEFAULT_RETRIEVAL_EVALUATION_PATH,
                output_directory=Path(directory),
                config=RetrievalConfig(adjacent_window=0),
                git_commit="d" * 40,
                stage2_start_commit="e" * 40,
                started_at=datetime(2026, 8, 19, tzinfo=UTC),
                validation_level="fixture",
                embedding_model=None,
                original_index_fingerprint="f" * 64,
            )
            run.persist_case_result(result)

            with patch(
                "app.retrieval_evaluation._atomic_replace_report",
                side_effect=RetrievalEvaluationReportError("finalization failed"),
            ):
                with self.assertRaisesRegex(
                    RetrievalEvaluationReportError,
                    "finalization failed",
                ):
                    run.finalize(original_index_unchanged=True)

            recovered = load_persisted_retrieval_report(run.path)
            self.assertEqual(recovered["run_status"], "running")
            self.assertEqual(len(recovered["per_case_results"]), 1)
            self.assertIsNone(
                recovered["index_isolation"]["original_index_unchanged"]
            )


class RetrievalRunnerTests(unittest.TestCase):
    @patch("app.evaluate_retrieval.open_vector_store")
    def test_cost_gate_stops_before_opening_index(self, open_vector_store: Mock) -> None:
        with patch("builtins.print"):
            exit_code = retrieval_main([])

        self.assertEqual(exit_code, 2)
        open_vector_store.assert_not_called()

    def test_git_preflight_fails_before_embedding_client_is_created(self) -> None:
        with patch(
            "app.evaluate_retrieval._git_commit",
            side_effect=RuntimeError("git unavailable"),
        ), patch(
            "app.evaluate_retrieval.create_langchain_embeddings"
        ) as create_embeddings, patch(
            "builtins.print"
        ):
            exit_code = retrieval_main(["--confirm-query-embedding-cost"])

        self.assertEqual(exit_code, 2)
        create_embeddings.assert_not_called()

    def test_output_preflight_failure_happens_before_embedding_call(self) -> None:
        snapshot = EvaluationIndexSnapshot(Path("snapshot"), "f" * 64)
        config = Mock(model="embedding-model")
        with patch("app.evaluate_retrieval._git_commit", return_value="d" * 40), patch(
            "app.evaluate_retrieval.EmbeddingConfig.from_environment",
            return_value=config,
        ), patch(
            "app.evaluate_retrieval.disposable_evaluation_snapshot",
            return_value=nullcontext(snapshot),
        ), patch(
            "app.evaluate_retrieval._preflight_snapshot",
            return_value=IndexCompatibilityStatus.LEGACY_READ_ONLY,
        ), patch.object(
            RetrievalEvaluationRun,
            "create",
            side_effect=RetrievalEvaluationReportError("output unavailable"),
        ), patch(
            "app.evaluate_retrieval.create_langchain_embeddings"
        ) as create_embeddings, patch(
            "builtins.print"
        ):
            exit_code = retrieval_main(["--confirm-query-embedding-cost"])

        self.assertEqual(exit_code, 2)
        create_embeddings.assert_not_called()

    def test_evaluation_queries_snapshot_path_only(self) -> None:
        snapshot = EvaluationIndexSnapshot(Path("snapshot"), "f" * 64)
        config = Mock(model="embedding-model")
        query_store = Mock()
        result = evaluate_retrieval_cases(
            [make_case()],
            FakeVectorStore([(make_document(2, "gold"), 0.9)]),
            RetrievalConfig(adjacent_window=0),
        )[0]
        run = Mock(path=Path("report.json"))
        run.finalize.return_value = {
            "per_case_results": [{"failure_category": None}]
        }
        with patch(
            "app.evaluate_retrieval.load_retrieval_cases",
            return_value=(make_case(),),
        ), patch("app.evaluate_retrieval._git_commit", return_value="d" * 40), patch(
            "app.evaluate_retrieval.EmbeddingConfig.from_environment",
            return_value=config,
        ), patch(
            "app.evaluate_retrieval.disposable_evaluation_snapshot",
            return_value=nullcontext(snapshot),
        ), patch(
            "app.evaluate_retrieval._preflight_snapshot",
            return_value=IndexCompatibilityStatus.LEGACY_READ_ONLY,
        ), patch.object(
            RetrievalEvaluationRun,
            "create",
            return_value=run,
        ), patch(
            "app.evaluate_retrieval.create_langchain_embeddings",
            return_value=Mock(),
        ), patch(
            "app.evaluate_retrieval.open_vector_store",
            return_value=query_store,
        ) as open_vector_store, patch(
            "app.evaluate_retrieval.evaluate_retrieval_cases",
            return_value=(result,),
        ), patch("app.evaluate_retrieval.close_vector_store"), patch(
            "builtins.print"
        ):
            exit_code = retrieval_main(["--confirm-query-embedding-cost"])

        self.assertEqual(exit_code, 0)
        self.assertEqual(open_vector_store.call_args.kwargs["persist_directory"], snapshot.path)
        self.assertNotEqual(
            open_vector_store.call_args.kwargs["persist_directory"],
            Path("data/vector_store"),
        )
        run.persist_case_result.assert_called_once_with(result)

    def test_disposable_snapshot_can_mutate_without_changing_original(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            original = Path(directory) / "original"
            nested = original / "segment"
            nested.mkdir(parents=True)
            original_file = nested / "data.bin"
            original_file.write_bytes(b"original bytes")

            with disposable_evaluation_snapshot(original) as snapshot:
                (snapshot.path / "segment" / "data.bin").write_bytes(b"changed")
                (snapshot.path / "query-cache.bin").write_bytes(b"snapshot only")

            self.assertEqual(original_file.read_bytes(), b"original bytes")
            self.assertFalse((original / "query-cache.bin").exists())

    def test_disposable_snapshot_detects_original_index_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            original = Path(directory) / "original"
            original.mkdir()
            original_file = original / "data.bin"
            original_file.write_bytes(b"before")

            with self.assertRaisesRegex(RuntimeError, "filesystem 指纹发生变化"):
                with disposable_evaluation_snapshot(original):
                    original_file.write_bytes(b"after")


if __name__ == "__main__":
    unittest.main()
