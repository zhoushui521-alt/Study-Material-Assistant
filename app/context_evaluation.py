"""复用已完成的 Retrieval Trace，对 Context Selector 做零费用 A/B 评测。"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from time import perf_counter_ns

from langchain_core.documents import Document

if __package__:
    from app.context_selector import (
        BaselineContextSelector,
        ContextSelector,
        EvidenceScoreContextSelector,
    )
    from app.evidence import evidence_from_document
    from app.retrieval_evaluation import (
        DEFAULT_RETRIEVAL_EVALUATION_PATH,
        RetrievalEvaluationCase,
        context_precision,
    )
else:
    from context_selector import (
        BaselineContextSelector,
        ContextSelector,
        EvidenceScoreContextSelector,
    )
    from evidence import evidence_from_document
    from retrieval_evaluation import (
        DEFAULT_RETRIEVAL_EVALUATION_PATH,
        RetrievalEvaluationCase,
        context_precision,
    )


CONTEXT_EXPERIMENT_NAME = "stage3_4_evidence_score_selection"
CONTEXT_REPORT_SCHEMA_VERSION = 1


class ContextEvaluationError(ValueError):
    """Baseline Report、Dataset 或当前 Index 不能支持可信 A/B 时抛出。"""


@dataclass(frozen=True)
class ContextEvaluationConfig:
    seed_count: int = 3
    adjacent_per_seed: int = 1
    latency_repetitions: int = 100

    def __post_init__(self) -> None:
        if self.seed_count <= 0:
            raise ContextEvaluationError("Seed 数量必须大于 0。")
        if self.adjacent_per_seed < 0:
            raise ContextEvaluationError("每个 Seed 的相邻 Evidence 数量不能小于 0。")
        if self.latency_repetitions <= 0:
            raise ContextEvaluationError("Latency 重复次数必须大于 0。")


def _sha256_file(path: Path) -> str:
    digest = sha256()
    try:
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise ContextEvaluationError(f"无法读取文件 {path}。") from error
    return digest.hexdigest()


def load_context_baseline_report(path: Path) -> dict[str, object]:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ContextEvaluationError(f"无法读取 Baseline Report：{path}。") from error
    except json.JSONDecodeError as error:
        raise ContextEvaluationError("Baseline Report 不是合法 JSON。") from error
    if not isinstance(report, dict):
        raise ContextEvaluationError("Baseline Report 根节点必须是对象。")
    if report.get("report_schema_version") != 2:
        raise ContextEvaluationError("只支持 Stage 2 Retrieval Report schema v2。")
    if report.get("evaluation_type") != "retrieval":
        raise ContextEvaluationError("Baseline 必须是 Retrieval Report。")
    if report.get("run_status") != "completed":
        raise ContextEvaluationError("Baseline Retrieval Report 尚未完成。")
    if not isinstance(report.get("per_case_results"), list):
        raise ContextEvaluationError("Baseline Report 缺少 per_case_results。")
    if not isinstance(report.get("aggregate_metrics"), dict):
        raise ContextEvaluationError("Baseline Report 缺少 aggregate_metrics。")
    return report


def _average(values: Sequence[object]) -> float | None:
    numeric = [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    return sum(numeric) / len(numeric) if numeric else None


def _chunk_id(document: Document) -> str:
    return evidence_from_document(document, "S1").chunk_id


def _document_map(indexed_documents: Sequence[Document]) -> dict[str, Document]:
    documents: dict[str, Document] = {}
    for document in indexed_documents:
        chunk_id = _chunk_id(document)
        if chunk_id in documents:
            raise ContextEvaluationError(f"当前 Index 包含重复 Chunk ID：{chunk_id}。")
        documents[chunk_id] = document
    return documents


def _case_map(
    baseline_report: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    raw_results = baseline_report.get("per_case_results")
    if not isinstance(raw_results, list):
        raise ContextEvaluationError("Baseline Report 缺少 per_case_results。")
    results = {}
    for raw_result in raw_results:
        if not isinstance(raw_result, dict):
            raise ContextEvaluationError("Baseline Report 包含无效案例。")
        case_id = raw_result.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ContextEvaluationError("Baseline Report 包含无效 case_id。")
        if case_id in results:
            raise ContextEvaluationError(f"Baseline Report 包含重复案例：{case_id}。")
        results[case_id] = raw_result
    return results


def _trace_items(raw_result: Mapping[str, object]) -> list[Mapping[str, object]]:
    trace = raw_result.get("trace")
    if not isinstance(trace, dict):
        raise ContextEvaluationError("Baseline 案例缺少 trace。")
    items = trace.get("adjacent_expansion")
    if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
        raise ContextEvaluationError("Baseline 案例缺少 adjacent_expansion。")
    return items


def _context_documents(
    raw_result: Mapping[str, object],
    documents: Mapping[str, Document],
) -> list[Document]:
    selected = []
    for item in _trace_items(raw_result):
        chunk_id = item.get("chunk_id")
        if not isinstance(chunk_id, str) or chunk_id not in documents:
            raise ContextEvaluationError(
                f"当前 Index 无法解析 Baseline Context Chunk：{chunk_id}。"
            )
        selected.append(documents[chunk_id])
    trace = raw_result.get("trace")
    assert isinstance(trace, dict)
    final_context = trace.get("final_context")
    if not isinstance(final_context, dict):
        raise ContextEvaluationError("Baseline 案例缺少 final_context。")
    expected_char_size = final_context.get("char_size")
    actual_char_size = sum(len(document.page_content) for document in selected)
    if expected_char_size != actual_char_size:
        raise ContextEvaluationError(
            "当前 Index 的 Context 内容与 Baseline Trace char_size 不一致。"
        )
    return selected


def _measure_selector(
    selector: ContextSelector,
    query: str,
    documents: Sequence[Document],
    repetitions: int,
) -> tuple[list[Document], float]:
    expected_ids: list[str] | None = None
    selected: list[Document] = []
    started = perf_counter_ns()
    for _ in range(repetitions):
        selected = selector.select(query, documents)
        selected_ids = [_chunk_id(document) for document in selected]
        if expected_ids is None:
            expected_ids = selected_ids
        elif selected_ids != expected_ids:
            raise ContextEvaluationError("Context Selector 重复运行结果不稳定。")
    elapsed_ms = (perf_counter_ns() - started) / 1_000_000 / repetitions
    return selected, elapsed_ms


def _duplicate_ratio(documents: Sequence[Document]) -> float:
    if not documents:
        return 0.0
    unique = len({_chunk_id(document) for document in documents})
    return 1.0 - unique / len(documents)


def _context_metrics(
    case: RetrievalEvaluationCase,
    documents: Sequence[Document],
) -> dict[str, object]:
    chunk_ids = [_chunk_id(document) for document in documents]
    gold_ids = set(case.gold_chunk_ids)
    hits = len(set(chunk_ids) & gold_ids)
    return {
        "context_precision": context_precision(
            chunk_ids,
            gold_ids,
            answerable=case.answerable,
        ),
        "final_context_recall": hits / len(gold_ids) if gold_ids else None,
        "chunk_count": len(documents),
        "char_size": sum(len(document.page_content) for document in documents),
        "token_size": None,
        "token_count_method": None,
        "duplicate_ratio": _duplicate_ratio(documents),
    }


def _failure_category(
    case: RetrievalEvaluationCase,
    metrics: Mapping[str, object],
    baseline_failure: object,
) -> str | None:
    if not case.answerable:
        return (
            "unanswerable_handling_failure"
            if metrics.get("chunk_count")
            else None
        )
    if metrics.get("final_context_recall") == 0:
        return (
            baseline_failure
            if isinstance(baseline_failure, str)
            else "context_construction_failure"
        )
    return baseline_failure if isinstance(baseline_failure, str) else None


def _aggregate_context_metrics(
    results: Sequence[Mapping[str, object]],
    strategy: str,
) -> dict[str, object]:
    metrics = [result[strategy] for result in results]
    if not all(isinstance(item, dict) for item in metrics):
        raise ContextEvaluationError("Context 评测结果缺少策略指标。")
    unanswerable = [
        item
        for result, item in zip(results, metrics, strict=True)
        if result["answerable"] is False
    ]
    return {
        "context_precision": _average(
            [item["context_precision"] for item in metrics]
        ),
        "final_context_recall": _average(
            [item["final_context_recall"] for item in metrics]
        ),
        "average_chunk_count": _average([item["chunk_count"] for item in metrics]),
        "average_char_size": _average([item["char_size"] for item in metrics]),
        "average_token_size": None,
        "average_duplicate_ratio": _average(
            [item["duplicate_ratio"] for item in metrics]
        ),
        "unanswerable_empty_context_rate": _average(
            [1.0 if item["chunk_count"] == 0 else 0.0 for item in unanswerable]
        ),
    }


def _metric_deltas(
    baseline: Mapping[str, object],
    optimized: Mapping[str, object],
) -> dict[str, float | None]:
    deltas = {}
    for name in sorted(set(baseline) | set(optimized)):
        before = baseline.get(name)
        after = optimized.get(name)
        deltas[name] = (
            float(after) - float(before)
            if isinstance(before, (int, float))
            and not isinstance(before, bool)
            and isinstance(after, (int, float))
            and not isinstance(after, bool)
            else None
        )
    return deltas


def build_context_experiment_report(
    cases: Sequence[RetrievalEvaluationCase],
    baseline_report: Mapping[str, object],
    indexed_documents: Sequence[Document],
    *,
    config: ContextEvaluationConfig,
    dataset_path: Path = DEFAULT_RETRIEVAL_EVALUATION_PATH,
    baseline_report_path: Path,
    stage3_4_start_commit: str,
    git_head: str,
    current_index_fingerprint: str,
    implementation_hashes: Mapping[str, str],
    generated_at: datetime | None = None,
) -> dict[str, object]:
    """在同一历史 Retrieval Trace 上比较 A/B Context，不执行任何 Retrieval。"""
    generated_at = generated_at or datetime.now(UTC)
    baseline_cases = _case_map(baseline_report)
    case_ids = [case.case_id for case in cases]
    if set(case_ids) != set(baseline_cases):
        raise ContextEvaluationError("Dataset 与 Baseline Report 的案例集合不同。")
    dataset = baseline_report.get("evaluation_dataset")
    if not isinstance(dataset, dict):
        raise ContextEvaluationError("Baseline Report 缺少 evaluation_dataset。")
    dataset_hash = _sha256_file(dataset_path)
    if dataset.get("sha256") != dataset_hash:
        raise ContextEvaluationError("Dataset SHA-256 与 Baseline Report 不一致。")
    retrieval_config = baseline_report.get("retrieval_config")
    if not isinstance(retrieval_config, dict):
        raise ContextEvaluationError("Baseline Report 缺少 retrieval_config。")
    if retrieval_config.get("retrieval_limit") != config.seed_count:
        raise ContextEvaluationError(
            "Context Selector 的 Seed 数必须与 Baseline retrieval_limit 一致。"
        )

    documents = _document_map(indexed_documents)
    baseline_selector = BaselineContextSelector()
    optimized_selector = EvidenceScoreContextSelector(
        seed_count=config.seed_count,
        adjacent_per_seed=config.adjacent_per_seed,
    )
    per_case_results = []
    for case in cases:
        raw_result = baseline_cases[case.case_id]
        context_documents = _context_documents(raw_result, documents)
        baseline_documents, baseline_latency = _measure_selector(
            baseline_selector,
            case.question,
            context_documents,
            config.latency_repetitions,
        )
        optimized_documents, optimized_latency = _measure_selector(
            optimized_selector,
            case.question,
            context_documents,
            config.latency_repetitions,
        )
        baseline_metrics = _context_metrics(case, baseline_documents)
        optimized_metrics = _context_metrics(case, optimized_documents)
        optimized_ids = {_chunk_id(document) for document in optimized_documents}
        gold_ids = set(case.gold_chunk_ids)
        removed = []
        for item in _trace_items(raw_result):
            chunk_id = str(item["chunk_id"])
            if chunk_id in optimized_ids:
                continue
            removed.append(
                {
                    "chunk_id": chunk_id,
                    "source": item.get("source"),
                    "chunk_index": item.get("chunk_index"),
                    "role": item.get("role"),
                    "was_gold": chunk_id in gold_ids,
                }
            )
        per_case_results.append(
            {
                "case_id": case.case_id,
                "answerable": case.answerable,
                "baseline": baseline_metrics,
                "optimized": optimized_metrics,
                "metric_deltas": _metric_deltas(
                    baseline_metrics,
                    optimized_metrics,
                ),
                "baseline_failure_category": raw_result.get("failure_category"),
                "optimized_failure_category": _failure_category(
                    case,
                    optimized_metrics,
                    raw_result.get("failure_category"),
                ),
                "gold_preservation": {
                    "gold_chunk_ids": sorted(gold_ids),
                    "preserved_gold_chunk_ids": sorted(gold_ids & optimized_ids),
                    "removed_gold_chunk_ids": sorted(gold_ids - optimized_ids),
                },
                "removed_context": removed,
                "latency_ms": {
                    "baseline_selector_average": baseline_latency,
                    "optimized_selector_average": optimized_latency,
                    "incremental": optimized_latency - baseline_latency,
                },
            }
        )

    baseline_metrics = _aggregate_context_metrics(per_case_results, "baseline")
    optimized_metrics = _aggregate_context_metrics(per_case_results, "optimized")
    retrieval_metrics = baseline_report.get("aggregate_metrics")
    assert isinstance(retrieval_metrics, dict)
    invariant_names = [
        name
        for name in retrieval_metrics
        if name.startswith("raw_") or name.startswith("ranked_")
    ]
    baseline_index = baseline_report.get("index_isolation")
    baseline_index_fingerprint = (
        baseline_index.get("original_index_fingerprint_sha256")
        if isinstance(baseline_index, dict)
        else None
    )
    return {
        "report_schema_version": CONTEXT_REPORT_SCHEMA_VERSION,
        "evaluation_type": "context_construction",
        "experiment_name": CONTEXT_EXPERIMENT_NAME,
        "stage3_4_start_commit": stage3_4_start_commit,
        "git_head_at_evaluation": git_head,
        "evaluation_dataset": {
            "file": dataset_path.name,
            "dataset_version": dataset.get("dataset_version"),
            "sha256": dataset_hash,
            "case_count": len(cases),
        },
        "baseline_retrieval_report": {
            "file": baseline_report_path.name,
            "sha256": _sha256_file(baseline_report_path),
            "git_commit": baseline_report.get("git_commit"),
            "validation_level": baseline_report.get("validation_level"),
            "index_fingerprint_sha256": baseline_index_fingerprint,
        },
        "index_replay": {
            "current_index_fingerprint_sha256": current_index_fingerprint,
            "same_filesystem_fingerprint_as_baseline": (
                current_index_fingerprint == baseline_index_fingerprint
            ),
            "all_context_chunk_ids_and_char_sizes_resolved": True,
            "original_index_unchanged": True,
        },
        "implementation_hashes": dict(implementation_hashes),
        "controlled_variables": {
            "retrieval_replayed_not_rerun": True,
            "retrieval_config": retrieval_config,
            "chunking_unchanged": True,
            "embedding_calls": 0,
            "chat_calls": 0,
            "reranker_calls": 0,
        },
        "selector_config": {
            "baseline": "pass_through_current_context",
            "optimized": "evidence_score_per_seed_adjacent_selection",
            "seed_count": config.seed_count,
            "adjacent_per_seed": config.adjacent_per_seed,
            "score": "query_keyword_coverage",
            "tie_break": ["smaller_adjacent_distance", "original_context_order"],
            "token_budget": None,
            "token_budget_reason": "no_reliable_chat_model_tokenizer_configured",
        },
        "retrieval_invariants": {
            "unchanged_by_design": True,
            "metrics": {name: retrieval_metrics[name] for name in invariant_names},
            "deltas": {name: 0.0 for name in invariant_names},
        },
        "context_metrics": {
            "baseline": baseline_metrics,
            "optimized": optimized_metrics,
            "deltas": _metric_deltas(baseline_metrics, optimized_metrics),
        },
        "latency": {
            "measurement_scope": "selector_only_local_wall_clock",
            "repetitions_per_case": config.latency_repetitions,
            "baseline_selector_average_ms": _average(
                [result["latency_ms"]["baseline_selector_average"] for result in per_case_results]
            ),
            "optimized_selector_average_ms": _average(
                [result["latency_ms"]["optimized_selector_average"] for result in per_case_results]
            ),
            "average_incremental_ms": _average(
                [result["latency_ms"]["incremental"] for result in per_case_results]
            ),
        },
        "tokens": {
            "measured": False,
            "total": None,
            "reason": "no_reliable_chat_model_tokenizer_configured",
        },
        "cost": {
            "query_embedding_calls": 0,
            "chat_calls": 0,
            "reranker_calls": 0,
            "measured_amount": None,
        },
        "per_case_results": per_case_results,
        "generated_at": generated_at.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        "validation_level": "local_historical_retrieval_trace_replay",
    }


def write_context_experiment_report(
    report: Mapping[str, object],
    output_directory: Path,
) -> Path:
    output_directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    base = output_directory / f"context-optimization-{timestamp}.json"
    for suffix in range(1000):
        path = base if suffix == 0 else base.with_stem(f"{base.stem}-{suffix}")
        try:
            with path.open("x", encoding="utf-8", newline="\n") as target:
                json.dump(report, target, ensure_ascii=False, indent=2)
                target.write("\n")
            return path
        except FileExistsError:
            continue
        except OSError as error:
            raise ContextEvaluationError(f"无法写入 Context 评测报告：{path}。") from error
    raise ContextEvaluationError("无法生成唯一的 Context 评测报告文件名。")
