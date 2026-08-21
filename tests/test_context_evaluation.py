import json
import tempfile
import unittest
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

from langchain_core.documents import Document

from app.context_evaluation import (
    ContextEvaluationConfig,
    ContextEvaluationError,
    build_context_experiment_report,
    load_context_baseline_report,
    write_context_experiment_report,
)
from app.retrieval_evaluation import (
    CurrentChunkMapping,
    RetrievalEvaluationCase,
    StableGoldMeaning,
)


MATERIAL_ID = "a" * 64


def make_document(number: int, content: str) -> Document:
    return Document(
        page_content=content,
        metadata={
            "source": "notes.md",
            "filename": "notes.md",
            "source_type": "markdown",
            "chunk_index": number,
            "chunk_id": f"{number:064x}",
            "material_id": MATERIAL_ID,
            "content_hash": f"{number + 100:064x}",
        },
    )


def make_case(
    case_id: str,
    *,
    answerable: bool,
    gold: Document | None = None,
) -> RetrievalEvaluationCase:
    mappings = ()
    meanings = ()
    if gold is not None:
        chunk_id = str(gold.metadata["chunk_id"])
        meanings = (
            StableGoldMeaning(
                gold_id=f"gold-{case_id}",
                material_id=MATERIAL_ID,
                source="notes.md",
                locator="notes.md#chunk=1",
                text_span=gold.page_content,
            ),
        )
        mappings = (
            CurrentChunkMapping(
                gold_ids=(f"gold-{case_id}",),
                chunk_id=chunk_id,
                material_id=MATERIAL_ID,
                content_hash=str(gold.metadata["content_hash"]),
                source="notes.md",
                legacy_chunk_index=int(gold.metadata["chunk_index"]),
            ),
        )
    return RetrievalEvaluationCase(
        case_id=case_id,
        dataset_version="study-material-retrieval-v1",
        question="target",
        answerable=answerable,
        case_types=("fixture",),
        expected_material_ids=(MATERIAL_ID,) if answerable else (),
        expected_sources=("notes.md",) if answerable else (),
        stable_gold_meanings=meanings,
        current_chunk_mappings=mappings,
        annotation_notes="fixture",
    )


def trace_item(document: Document, role: str) -> dict[str, object]:
    return {
        "chunk_id": document.metadata["chunk_id"],
        "source": document.metadata["source"],
        "chunk_index": document.metadata["chunk_index"],
        "role": role,
    }


class ContextEvaluationTests(unittest.TestCase):
    def test_reports_precision_recall_size_and_unchanged_retrieval(self) -> None:
        gold = make_document(1, "gold")
        relevant_adjacent = make_document(2, "target")
        noise = make_document(3, "noise")
        unanswerable_seed = make_document(4, "unrelated")
        unanswerable_adjacent = make_document(5, "also unrelated")
        cases = (
            make_case("answerable", answerable=True, gold=gold),
            make_case("unanswerable", answerable=False),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_path = root / "retrieval_cases.json"
            dataset_path.write_text('{"fixture": true}\n', encoding="utf-8")
            dataset_hash = sha256(dataset_path.read_bytes()).hexdigest()
            baseline_path = root / "baseline.json"
            baseline_report = {
                "report_schema_version": 2,
                "evaluation_type": "retrieval",
                "run_status": "completed",
                "git_commit": "c" * 40,
                "validation_level": "fixture",
                "evaluation_dataset": {
                    "dataset_version": "study-material-retrieval-v1",
                    "sha256": dataset_hash,
                },
                "aggregate_metrics": {
                    "raw_recall_at_1": 1.0,
                    "ranked_recall_at_1": 1.0,
                    "ranked_mrr": 1.0,
                },
                "retrieval_config": {"retrieval_limit": 1},
                "index_isolation": {
                    "original_index_fingerprint_sha256": "b" * 64,
                },
                "per_case_results": [
                    {
                        "case_id": "answerable",
                        "failure_category": "ranking_failure",
                        "trace": {
                            "adjacent_expansion": [
                                trace_item(gold, "seed"),
                                trace_item(relevant_adjacent, "adjacent"),
                                trace_item(noise, "adjacent"),
                            ],
                            "final_context": {
                                "char_size": sum(
                                    len(item.page_content)
                                    for item in (gold, relevant_adjacent, noise)
                                )
                            },
                        },
                    },
                    {
                        "case_id": "unanswerable",
                        "failure_category": "unanswerable_handling_failure",
                        "trace": {
                            "adjacent_expansion": [
                                trace_item(unanswerable_seed, "seed"),
                                trace_item(unanswerable_adjacent, "adjacent"),
                            ],
                            "final_context": {
                                "char_size": sum(
                                    len(item.page_content)
                                    for item in (
                                        unanswerable_seed,
                                        unanswerable_adjacent,
                                    )
                                )
                            },
                        },
                    },
                ],
            }
            baseline_path.write_text(
                json.dumps(baseline_report),
                encoding="utf-8",
            )

            report = build_context_experiment_report(
                cases,
                baseline_report,
                [
                    gold,
                    relevant_adjacent,
                    noise,
                    unanswerable_seed,
                    unanswerable_adjacent,
                ],
                config=ContextEvaluationConfig(
                    seed_count=1,
                    adjacent_per_seed=1,
                    latency_repetitions=2,
                ),
                dataset_path=dataset_path,
                baseline_report_path=baseline_path,
                stage3_4_start_commit="d" * 40,
                git_head="d" * 40,
                current_index_fingerprint="e" * 64,
                implementation_hashes={"app/context_selector.py": "f" * 64},
                generated_at=datetime(2026, 8, 21, tzinfo=UTC),
            )

        metrics = report["context_metrics"]
        self.assertGreater(
            metrics["optimized"]["context_precision"],
            metrics["baseline"]["context_precision"],
        )
        self.assertEqual(metrics["optimized"]["final_context_recall"], 1.0)
        self.assertLess(
            metrics["optimized"]["average_chunk_count"],
            metrics["baseline"]["average_chunk_count"],
        )
        self.assertLess(
            metrics["optimized"]["average_char_size"],
            metrics["baseline"]["average_char_size"],
        )
        self.assertTrue(report["retrieval_invariants"]["unchanged_by_design"])
        self.assertTrue(
            all(
                delta == 0.0
                for delta in report["retrieval_invariants"]["deltas"].values()
            )
        )
        self.assertEqual(report["cost"]["query_embedding_calls"], 0)
        self.assertEqual(report["cost"]["chat_calls"], 0)
        self.assertEqual(
            report["per_case_results"][0]["optimized_failure_category"],
            "ranking_failure",
        )
        self.assertEqual(
            report["per_case_results"][1]["optimized_failure_category"],
            "unanswerable_handling_failure",
        )
        self.assertNotIn("also unrelated", json.dumps(report))

    def test_rejects_context_content_that_no_longer_matches_trace(self) -> None:
        document = make_document(1, "changed")
        case = make_case("case", answerable=True, gold=document)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_path = root / "cases.json"
            dataset_path.write_text("{}", encoding="utf-8")
            baseline_path = root / "baseline.json"
            baseline_path.write_text("{}", encoding="utf-8")
            baseline = {
                "evaluation_dataset": {
                    "sha256": sha256(dataset_path.read_bytes()).hexdigest(),
                },
                "aggregate_metrics": {},
                "retrieval_config": {"retrieval_limit": 3},
                "per_case_results": [
                    {
                        "case_id": "case",
                        "trace": {
                            "adjacent_expansion": [trace_item(document, "seed")],
                            "final_context": {"char_size": 999},
                        },
                    }
                ],
            }

            with self.assertRaisesRegex(ContextEvaluationError, "char_size"):
                build_context_experiment_report(
                    [case],
                    baseline,
                    [document],
                    config=ContextEvaluationConfig(latency_repetitions=1),
                    dataset_path=dataset_path,
                    baseline_report_path=baseline_path,
                    stage3_4_start_commit="d" * 40,
                    git_head="d" * 40,
                    current_index_fingerprint="e" * 64,
                    implementation_hashes={},
                )

    def test_loader_rejects_incomplete_retrieval_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(
                json.dumps(
                    {
                        "report_schema_version": 2,
                        "evaluation_type": "retrieval",
                        "run_status": "running",
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ContextEvaluationError, "尚未完成"):
                load_context_baseline_report(path)

    def test_report_writer_never_overwrites(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            first = write_context_experiment_report({"value": "中文"}, output)
            second = write_context_experiment_report({"value": "中文"}, output)

            self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
