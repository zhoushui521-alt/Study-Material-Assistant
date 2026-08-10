"""读取并比较两份 RAG 结构化评测报告。"""

import json
from collections.abc import Mapping
from pathlib import Path


COMPARISON_SCHEMA_VERSION = 1


class EvaluationComparisonError(ValueError):
    """评测报告缺少比较所需的稳定结构。"""


def load_evaluation_report(path: Path) -> dict[str, object]:
    """读取单份 JSON 报告并校验比较所需的最小字段。"""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise EvaluationComparisonError(f"无法读取评测报告 {path}。") from error
    except json.JSONDecodeError as error:
        raise EvaluationComparisonError(f"评测报告 {path} 不是合法 JSON。") from error
    if not isinstance(payload, dict):
        raise EvaluationComparisonError("评测报告根节点必须是对象。")
    if payload.get("report_schema_version") != 1:
        raise EvaluationComparisonError("只支持 report_schema_version=1 的报告。")
    if not isinstance(payload.get("evaluation_dataset"), dict):
        raise EvaluationComparisonError("评测报告缺少 evaluation_dataset。")
    if not isinstance(payload.get("cases"), list):
        raise EvaluationComparisonError("评测报告缺少 cases 列表。")
    return payload


def _case_map(report: Mapping[str, object], label: str) -> dict[str, dict]:
    raw_cases = report.get("cases")
    if not isinstance(raw_cases, list):
        raise EvaluationComparisonError(f"{label}报告缺少 cases 列表。")
    cases: dict[str, dict] = {}
    for raw_case in raw_cases:
        if not isinstance(raw_case, dict):
            raise EvaluationComparisonError(f"{label}报告包含无效案例。")
        case_id = raw_case.get("id")
        passed = raw_case.get("passed")
        elapsed_ms = raw_case.get("elapsed_ms")
        if not isinstance(case_id, str) or not case_id:
            raise EvaluationComparisonError(f"{label}报告包含无效案例 id。")
        if case_id in cases:
            raise EvaluationComparisonError(f"{label}报告包含重复案例 id：{case_id}。")
        if not isinstance(passed, bool):
            raise EvaluationComparisonError(f"案例 {case_id} 缺少 passed 布尔值。")
        if (
            isinstance(elapsed_ms, bool)
            or not isinstance(elapsed_ms, int)
            or elapsed_ms < 0
        ):
            raise EvaluationComparisonError(f"案例 {case_id} 的 elapsed_ms 无效。")
        cases[case_id] = raw_case
    return cases


def _dataset_identity(report: Mapping[str, object]) -> dict[str, object]:
    dataset = report.get("evaluation_dataset")
    if not isinstance(dataset, dict):
        raise EvaluationComparisonError("评测报告缺少 evaluation_dataset。")
    return {
        "file": dataset.get("file"),
        "version": dataset.get("version"),
        "sha256": dataset.get("sha256"),
    }


def compare_evaluation_reports(
    baseline_report: Mapping[str, object],
    current_report: Mapping[str, object],
    *,
    baseline_name: str = "baseline",
    current_name: str = "current",
) -> dict[str, object]:
    """比较通过状态、案例集合和耗时，不把固定案例外推为普遍效果。"""
    baseline_cases = _case_map(baseline_report, "基线")
    current_cases = _case_map(current_report, "当前")
    baseline_ids = set(baseline_cases)
    current_ids = set(current_cases)
    shared_ids = sorted(baseline_ids & current_ids)

    newly_failed = [
        case_id
        for case_id in shared_ids
        if baseline_cases[case_id]["passed"] and not current_cases[case_id]["passed"]
    ]
    recovered = [
        case_id
        for case_id in shared_ids
        if not baseline_cases[case_id]["passed"] and current_cases[case_id]["passed"]
    ]
    unchanged_failed = [
        case_id
        for case_id in shared_ids
        if not baseline_cases[case_id]["passed"]
        and not current_cases[case_id]["passed"]
    ]
    baseline_passed = sum(bool(case["passed"]) for case in baseline_cases.values())
    current_passed = sum(bool(case["passed"]) for case in current_cases.values())
    baseline_elapsed = sum(int(case["elapsed_ms"]) for case in baseline_cases.values())
    current_elapsed = sum(int(case["elapsed_ms"]) for case in current_cases.values())
    baseline_dataset = _dataset_identity(baseline_report)
    current_dataset = _dataset_identity(current_report)
    same_dataset = (
        isinstance(baseline_dataset["sha256"], str)
        and baseline_dataset["sha256"] == current_dataset["sha256"]
    )
    same_case_set = baseline_ids == current_ids
    totals_directly_comparable = same_dataset and same_case_set

    warning = None
    if not same_dataset:
        warning = "两份报告的评测集 SHA-256 不同，只能参考共同案例，不能直接比较总通过率。"
    elif not same_case_set:
        warning = "两份报告的案例集合不同，只能参考共同案例，不能直接比较总通过率和总耗时。"

    return {
        "comparison_schema_version": COMPARISON_SCHEMA_VERSION,
        "baseline": {"name": baseline_name, "dataset": baseline_dataset},
        "current": {"name": current_name, "dataset": current_dataset},
        "same_evaluation_dataset": same_dataset,
        "same_case_set": same_case_set,
        "totals_directly_comparable": totals_directly_comparable,
        "warning": warning,
        "summary": {
            "baseline_total": len(baseline_cases),
            "current_total": len(current_cases),
            "baseline_passed": baseline_passed,
            "current_passed": current_passed,
            "passed_delta": current_passed - baseline_passed,
            "baseline_elapsed_ms": baseline_elapsed,
            "current_elapsed_ms": current_elapsed,
            "elapsed_delta_ms": current_elapsed - baseline_elapsed,
            "newly_failed_count": len(newly_failed),
            "recovered_count": len(recovered),
        },
        "newly_failed": newly_failed,
        "recovered": recovered,
        "unchanged_failed": unchanged_failed,
        "added_cases": sorted(current_ids - baseline_ids),
        "removed_cases": sorted(baseline_ids - current_ids),
        "case_changes": [
            {
                "id": case_id,
                "baseline_passed": baseline_cases[case_id]["passed"],
                "current_passed": current_cases[case_id]["passed"],
                "baseline_elapsed_ms": baseline_cases[case_id]["elapsed_ms"],
                "current_elapsed_ms": current_cases[case_id]["elapsed_ms"],
                "elapsed_delta_ms": (
                    current_cases[case_id]["elapsed_ms"]
                    - baseline_cases[case_id]["elapsed_ms"]
                ),
            }
            for case_id in shared_ids
        ],
    }


def compare_evaluation_report_files(
    baseline_path: Path,
    current_path: Path,
) -> dict[str, object]:
    return compare_evaluation_reports(
        load_evaluation_report(baseline_path),
        load_evaluation_report(current_path),
        baseline_name=baseline_path.name,
        current_name=current_path.name,
    )


def write_comparison_report(report: Mapping[str, object], output_path: Path) -> None:
    """以独占创建写入比较结果，防止覆盖已有证据。"""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("x", encoding="utf-8", newline="\n") as output:
            output.write(json.dumps(dict(report), ensure_ascii=False, indent=2) + "\n")
    except (OSError, TypeError, ValueError) as error:
        raise EvaluationComparisonError(f"无法保存评测对比报告到 {output_path}。") from error
