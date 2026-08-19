import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.compare_rag_reports import main
from app.evaluation_comparison import (
    EvaluationComparisonError,
    compare_evaluation_reports,
    compare_retrieval_reports,
    load_evaluation_report,
    write_comparison_report,
)


def report(dataset_sha: str, cases: list[dict]) -> dict:
    return {
        "report_schema_version": 1,
        "evaluation_dataset": {
            "file": "rag_cases.json",
            "version": 1,
            "sha256": dataset_sha,
        },
        "cases": cases,
    }


def retrieval_report(
    dataset_sha: str,
    *,
    metric: float,
    failure: str | None = None,
) -> dict:
    return {
        "report_schema_version": 2,
        "evaluation_type": "retrieval",
        "evaluation_dataset": {
            "file": "retrieval_cases.json",
            "dataset_version": "retrieval-v1",
            "sha256": dataset_sha,
        },
        "aggregate_metrics": {"raw_recall_at_5": metric},
        "per_case_results": [
            {
                "case_id": "case",
                "metrics": {"raw_recall_at_5": metric},
                "failure_category": failure,
                "trace": {"latency_ms": {"total_retrieval": 5}},
            }
        ],
    }


class EvaluationComparisonTests(unittest.TestCase):
    def test_identifies_new_failures_recoveries_and_elapsed_delta(self) -> None:
        baseline = report(
            "same-sha",
            [
                {"id": "stable", "passed": True, "elapsed_ms": 100},
                {"id": "regressed", "passed": True, "elapsed_ms": 80},
                {"id": "recovered", "passed": False, "elapsed_ms": 70},
            ],
        )
        current = report(
            "same-sha",
            [
                {"id": "stable", "passed": True, "elapsed_ms": 90},
                {"id": "regressed", "passed": False, "elapsed_ms": 120},
                {"id": "recovered", "passed": True, "elapsed_ms": 60},
            ],
        )

        comparison = compare_evaluation_reports(baseline, current)

        self.assertTrue(comparison["same_evaluation_dataset"])
        self.assertTrue(comparison["same_case_set"])
        self.assertTrue(comparison["totals_directly_comparable"])
        self.assertEqual(comparison["newly_failed"], ["regressed"])
        self.assertEqual(comparison["recovered"], ["recovered"])
        self.assertEqual(comparison["summary"]["elapsed_delta_ms"], 20)

    def test_warns_when_dataset_hashes_differ_and_tracks_case_set(self) -> None:
        comparison = compare_evaluation_reports(
            report("old", [{"id": "removed", "passed": True, "elapsed_ms": 1}]),
            report("new", [{"id": "added", "passed": True, "elapsed_ms": 1}]),
        )

        self.assertFalse(comparison["same_evaluation_dataset"])
        self.assertFalse(comparison["same_case_set"])
        self.assertFalse(comparison["totals_directly_comparable"])
        self.assertIsNotNone(comparison["warning"])
        self.assertEqual(comparison["added_cases"], ["added"])
        self.assertEqual(comparison["removed_cases"], ["removed"])

    def test_warns_when_same_dataset_report_has_different_case_set(self) -> None:
        comparison = compare_evaluation_reports(
            report("same", [{"id": "shared", "passed": True, "elapsed_ms": 1}]),
            report(
                "same",
                [
                    {"id": "shared", "passed": True, "elapsed_ms": 1},
                    {"id": "added", "passed": True, "elapsed_ms": 1},
                ],
            ),
        )

        self.assertTrue(comparison["same_evaluation_dataset"])
        self.assertFalse(comparison["same_case_set"])
        self.assertFalse(comparison["totals_directly_comparable"])
        self.assertIn("案例集合不同", comparison["warning"])

    def test_rejects_duplicate_case_ids(self) -> None:
        duplicate = report(
            "sha",
            [
                {"id": "same", "passed": True, "elapsed_ms": 1},
                {"id": "same", "passed": False, "elapsed_ms": 2},
            ],
        )

        with self.assertRaisesRegex(EvaluationComparisonError, "重复案例"):
            compare_evaluation_reports(duplicate, report("sha", []))

    def test_loads_and_exclusively_writes_utf8_reports(self) -> None:
        payload = report("sha", [{"id": "case", "passed": True, "elapsed_ms": 1}])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.json"
            output = root / "comparison.json"
            source.write_text(json.dumps(payload), encoding="utf-8")

            self.assertEqual(load_evaluation_report(source), payload)
            write_comparison_report({"warning": "中文提示"}, output)
            with self.assertRaises(EvaluationComparisonError):
                write_comparison_report({"warning": "不能覆盖"}, output)

            self.assertIn("中文提示", output.read_text(encoding="utf-8"))

    def test_compares_retrieval_metrics_and_failure_changes(self) -> None:
        comparison = compare_retrieval_reports(
            retrieval_report("same", metric=0.5),
            retrieval_report("same", metric=0.75, failure="filtering_failure"),
        )

        self.assertEqual(comparison["comparison_schema_version"], 2)
        self.assertEqual(comparison["metric_changes"]["raw_recall_at_5"]["delta"], 0.25)
        self.assertEqual(comparison["newly_failed"], ["case"])
        self.assertTrue(comparison["totals_directly_comparable"])

    def test_dispatches_retrieval_reports_and_rejects_mixed_report_types(self) -> None:
        comparison = compare_evaluation_reports(
            retrieval_report("same", metric=0.5),
            retrieval_report("same", metric=0.6),
        )

        self.assertEqual(comparison["evaluation_type"], "retrieval")
        with self.assertRaisesRegex(EvaluationComparisonError, "不能直接比较"):
            compare_evaluation_reports(
                report("same", []),
                retrieval_report("same", metric=0.6),
            )

    def test_cli_returns_one_when_current_report_has_new_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline.json"
            current = root / "current.json"
            baseline.write_text(
                json.dumps(report("sha", [{"id": "case", "passed": True, "elapsed_ms": 1}])),
                encoding="utf-8",
            )
            current.write_text(
                json.dumps(report("sha", [{"id": "case", "passed": False, "elapsed_ms": 2}])),
                encoding="utf-8",
            )

            with patch("builtins.print"):
                exit_code = main([str(baseline), str(current)])

            self.assertEqual(exit_code, 1)


if __name__ == "__main__":
    unittest.main()
