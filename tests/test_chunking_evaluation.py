import json
import tempfile
import unittest
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from app.chunk_documents import DocumentChunk, Material, MaterialUnit, build_chunks
from app.chunking_evaluation import (
    ChunkingMappingError,
    ChunkingStrategyDescriptor,
    build_chunking_report,
    citation_localization_summary,
    remap_cases_to_chunks,
    write_chunking_report,
)
from app.embedding_client import EmbeddingConfig
from app.evaluate_chunking import (
    STAGE3_3_START_COMMIT,
    main as evaluate_chunking_main,
)
from app.evaluate_retrieval import EvaluationIndexSnapshot
from app.index_manifest import IndexCompatibilityStatus
from app.retrieval_evaluation import (
    CurrentChunkMapping,
    RetrievalCaseResult,
    RetrievalConfig,
    RetrievalEvaluationCase,
    RetrievalTrace,
    StableGoldMeaning,
)
from app.structure_aware_chunking import build_structure_aware_chunks
from langchain_core.embeddings import Embeddings


MATERIAL_ID = "a" * 64


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text) % 7 + 1), 1.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text) % 7 + 1), 1.0]


def make_unit(text: str) -> MaterialUnit:
    return MaterialUnit(
        material=Material(
            material_id=MATERIAL_ID,
            source_type="markdown",
            filename="notes.md",
            content_hash="b" * 64,
        ),
        source="notes.md",
        text=text,
    )


def make_case(baseline: DocumentChunk, *, text_span: str = "semantic meaning") -> RetrievalEvaluationCase:
    return RetrievalEvaluationCase(
        case_id="case",
        dataset_version="v1",
        question="What is the fact?",
        answerable=True,
        case_types=("citation_sensitive",),
        expected_material_ids=(MATERIAL_ID,),
        expected_sources=("notes.md",),
        stable_gold_meanings=(
            StableGoldMeaning(
                gold_id="gold",
                material_id=MATERIAL_ID,
                source="notes.md",
                locator="notes.md",
                text_span=text_span,
            ),
        ),
        current_chunk_mappings=(
            CurrentChunkMapping(
                gold_ids=("gold",),
                chunk_id=baseline.chunk_id,
                material_id=MATERIAL_ID,
                content_hash=baseline.content_hash,
                source="notes.md",
                legacy_chunk_index=baseline.index,
            ),
        ),
        annotation_notes="fixture",
    )


def make_result(case: RetrievalEvaluationCase, value: float) -> RetrievalCaseResult:
    metrics = {
        "raw_recall_at_1": value,
        "raw_recall_at_3": value,
        "raw_recall_at_5": value,
        "raw_recall_at_10": value,
        "ranked_recall_at_1": value,
        "ranked_recall_at_3": value,
        "ranked_recall_at_5": value,
        "ranked_recall_at_10": value,
        "ranked_mrr": value,
        "ranked_ndcg_at_5": value,
        "context_precision": value,
        "final_context_recall": value,
    }
    trace = RetrievalTrace(
        query={},
        raw_candidates=(),
        filtering={},
        hybrid_ranking=(),
        seeds=(),
        adjacent_expansion=(),
        final_context={"chunk_count": 1},
        latency_ms={
            "raw_vector_search": 1,
            "filter_and_hybrid_ranking": 2,
            "context_construction": 3,
            "total_retrieval": 6,
        },
    )
    return RetrievalCaseResult(case, trace, metrics, None)


class ChunkingEvaluationTests(unittest.TestCase):
    def test_semantic_gold_projects_only_when_current_gold_is_fully_contained(self) -> None:
        unit = make_unit("# RAG\n\nPrefix factual content suffix.")
        baseline = build_chunks([unit])
        experimental = build_structure_aware_chunks([unit])

        remapped = remap_cases_to_chunks(
            [make_case(baseline[0])],
            experimental,
            baseline_chunks=baseline,
        )

        mapping = remapped[0].current_chunk_mappings[0]
        self.assertEqual(mapping.chunk_id, experimental[0].chunk_id)
        self.assertEqual(mapping.section, "RAG")

    def test_exact_stable_span_takes_precedence(self) -> None:
        unit = make_unit("# First\n\nAlpha fact.\n\n# Second\n\nExact gold fact.")
        baseline = build_chunks([unit])
        experimental = build_structure_aware_chunks([unit])

        remapped = remap_cases_to_chunks(
            [make_case(baseline[0], text_span="Exact gold fact.")],
            experimental,
            baseline_chunks=baseline,
        )

        mapping = remapped[0].current_chunk_mappings[0]
        self.assertEqual(mapping.section, "Second")

    def test_unresolvable_projection_fails_instead_of_guessing(self) -> None:
        baseline = build_chunks([make_unit("baseline content")])
        experimental = build_structure_aware_chunks([make_unit("different content")])

        with self.assertRaisesRegex(ChunkingMappingError, "无法完整投影"):
            remap_cases_to_chunks(
                [make_case(baseline[0])],
                experimental,
                baseline_chunks=baseline,
            )

    def test_citation_localization_reports_resolution_without_support_claim(self) -> None:
        unit = make_unit("# RAG\n\nExact gold fact.")
        baseline = build_chunks([unit])
        experimental = build_structure_aware_chunks([unit])
        remapped = remap_cases_to_chunks(
            [make_case(baseline[0], text_span="Exact gold fact.")],
            experimental,
            baseline_chunks=baseline,
        )

        summary = citation_localization_summary(remapped, experimental)

        self.assertTrue(summary["all_mappings_resolved"])
        self.assertEqual(summary["material_match_ratio"], 1.0)
        self.assertEqual(summary["source_match_ratio"], 1.0)
        self.assertFalse(summary["citation_support_validated"])

    def test_report_records_controlled_variable_metrics_and_no_content(self) -> None:
        unit = make_unit("# RAG\n\nExact gold fact.")
        baseline = build_chunks([unit])
        experimental = build_structure_aware_chunks([unit])
        case = make_case(baseline[0], text_span="Exact gold fact.")
        remapped_case = remap_cases_to_chunks(
            [case], experimental, baseline_chunks=baseline
        )[0]
        descriptor = ChunkingStrategyDescriptor("a", "v1", 180, 0, ("fixed",))
        report = build_chunking_report(
            [make_result(case, 0.5)],
            [make_result(remapped_case, 1.0)],
            baseline_descriptor=descriptor,
            experimental_descriptor=ChunkingStrategyDescriptor(
                "b", "v2", 600, 0, ("heading", "sentence")
            ),
            retrieval_config=RetrievalConfig(),
            baseline_chunks=baseline,
            experimental_chunks=experimental,
            baseline_localization=citation_localization_summary([case], baseline),
            experimental_localization=citation_localization_summary(
                [remapped_case], experimental
            ),
            deterministic_rebuild=True,
            git_commit="c" * 40,
            stage3_3_start_commit="d" * 40,
            dataset_metadata={"dataset_version": "v1", "sha256": "e" * 64},
            embedding_model="embedding-model",
            query_embedding_calls=2,
            baseline_index_embedding_texts=1,
            baseline_index_embedding_batches=1,
            experimental_index_embedding_texts=1,
            experimental_index_embedding_batches=1,
            production_index_fingerprint="2" * 64,
            baseline_index_fingerprint="f" * 64,
            experimental_index_fingerprint="1" * 64,
            generated_at=datetime(2026, 8, 21, tzinfo=UTC),
            validation_level="fixture",
        )

        self.assertEqual(report["controlled_variable"], "chunking_strategy")
        self.assertEqual(
            report["metric_deltas"]["structure_aware_vs_baseline"]["ranked_mrr"],
            0.5,
        )
        self.assertEqual(report["cost"]["chat_calls"], 0)
        self.assertNotIn("Exact gold fact", json.dumps(report, ensure_ascii=False))

    def test_report_writer_never_overwrites_an_existing_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = write_chunking_report({"run_status": "completed"}, Path(directory))
            second = write_chunking_report({"run_status": "completed"}, Path(directory))

        self.assertNotEqual(first, second)

    def test_runner_requires_both_cost_confirmations_before_api_or_index(self) -> None:
        unit = make_unit("# RAG\n\nExact gold fact.")
        baseline = build_chunks([unit])
        case = make_case(baseline[0], text_span="Exact gold fact.")
        with tempfile.TemporaryDirectory() as directory, patch(
            "app.evaluate_chunking.load_retrieval_cases", return_value=(case,)
        ), patch(
            "app.evaluate_chunking.load_material_units", return_value=[unit]
        ), patch(
            "app.evaluate_chunking.create_langchain_embeddings"
        ) as create_embeddings, patch(
            "app.evaluate_chunking.open_vector_store"
        ) as open_vector_store, patch("builtins.print"):
            base_args = ["--results-dir", directory]
            self.assertEqual(evaluate_chunking_main(base_args), 2)
            self.assertEqual(
                evaluate_chunking_main(
                    base_args + ["--confirm-query-embedding-cost"]
                ),
                2,
            )
            self.assertEqual(
                evaluate_chunking_main(
                    base_args + ["--confirm-controlled-index-embedding-cost"]
                ),
                2,
            )

        create_embeddings.assert_not_called()
        open_vector_store.assert_not_called()

    def test_runner_builds_both_temporary_indexes_with_same_embedding_config(self) -> None:
        unit = make_unit("# RAG\n\nExact gold fact.")
        baseline = build_chunks([unit])
        case = make_case(baseline[0], text_span="Exact gold fact.")
        snapshot = EvaluationIndexSnapshot(Path("unused"), "f" * 64)
        config = EmbeddingConfig(
            api_key="unused",
            base_url="https://unused.invalid",
            model="embedding-model",
            dimensions=2,
        )
        with tempfile.TemporaryDirectory() as directory, patch(
            "app.evaluate_chunking.load_retrieval_cases", return_value=(case,)
        ), patch(
            "app.evaluate_chunking.load_material_units", return_value=[unit]
        ), patch(
            "app.evaluate_chunking._git_commit",
            return_value=STAGE3_3_START_COMMIT,
        ), patch(
            "app.evaluate_chunking._source_state", return_value={"fixture": True}
        ), patch(
            "app.evaluate_chunking.EmbeddingConfig.from_environment",
            return_value=config,
        ), patch(
            "app.evaluate_chunking.disposable_evaluation_snapshot",
            return_value=nullcontext(snapshot),
        ), patch(
            "app.evaluate_chunking._preflight_snapshot",
            return_value=IndexCompatibilityStatus.LEGACY_READ_ONLY,
        ), patch(
            "app.evaluate_chunking._verify_baseline_index"
        ), patch(
            "app.evaluate_chunking.create_langchain_embeddings",
            return_value=FakeEmbeddings(),
        ):
            exit_code = evaluate_chunking_main(
                [
                    "--results-dir",
                    directory,
                    "--confirm-controlled-index-embedding-cost",
                    "--confirm-query-embedding-cost",
                ]
            )

            reports = list(Path(directory).glob("chunking-*.json"))
            payload = json.loads(reports[0].read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(reports), 1)
        self.assertEqual(payload["cost"]["index_embedding_texts"]["total"], 2)
        self.assertFalse(payload["index_isolation"]["production_index_mutated"])


if __name__ == "__main__":
    unittest.main()
