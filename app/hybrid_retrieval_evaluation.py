"""Stage 3.1：在同一 Dense 召回上比较旧基线与 BM25 + Dense + RRF。"""

import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Iterable, Mapping, Sequence
from uuid import uuid4

from langchain_core.documents import Document

if __package__:
    from app.evidence import evidence_from_document
    from app.hybrid_search import (
        DEFAULT_BM25_B,
        DEFAULT_BM25_K1,
        DEFAULT_CANDIDATE_LIMIT,
        DEFAULT_KEYWORD_WEIGHT,
        DEFAULT_RELEVANCE_THRESHOLD,
        DEFAULT_RRF_K,
        BM25Index,
        expand_adjacent_documents,
        rank_vector_candidates,
        reciprocal_rank_fusion,
    )
    from app.langchain_store import search_vector_store
    from app.retrieval_evaluation import (
        RECALL_K_VALUES,
        RetrievalEvaluationCase,
        ndcg_at_k,
        recall_at_k,
        reciprocal_rank,
    )
else:
    from evidence import evidence_from_document
    from hybrid_search import (
        DEFAULT_BM25_B,
        DEFAULT_BM25_K1,
        DEFAULT_CANDIDATE_LIMIT,
        DEFAULT_KEYWORD_WEIGHT,
        DEFAULT_RELEVANCE_THRESHOLD,
        DEFAULT_RRF_K,
        BM25Index,
        expand_adjacent_documents,
        rank_vector_candidates,
        reciprocal_rank_fusion,
    )
    from langchain_store import search_vector_store
    from retrieval_evaluation import (
        RECALL_K_VALUES,
        RetrievalEvaluationCase,
        ndcg_at_k,
        recall_at_k,
        reciprocal_rank,
    )


HYBRID_REPORT_SCHEMA_VERSION = 1
HYBRID_EXPERIMENT_NAME = "stage3_1_hybrid_rrf"
HYBRID_BASELINE_NAME = "stage2_baseline"


@dataclass(frozen=True)
class HybridRetrievalConfig:
    retrieval_limit: int = 3
    dense_candidate_limit: int = DEFAULT_CANDIDATE_LIMIT
    bm25_candidate_limit: int = DEFAULT_CANDIDATE_LIMIT
    relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD
    baseline_vector_weight: float = 1.0 - DEFAULT_KEYWORD_WEIGHT
    baseline_keyword_weight: float = DEFAULT_KEYWORD_WEIGHT
    rrf_k: int = DEFAULT_RRF_K
    bm25_k1: float = DEFAULT_BM25_K1
    bm25_b: float = DEFAULT_BM25_B
    adjacent_window: int = 2
    context_limit: int = 8

    def __post_init__(self) -> None:
        if (
            self.retrieval_limit <= 0
            or self.dense_candidate_limit <= 0
            or self.bm25_candidate_limit <= 0
        ):
            raise ValueError("检索数量必须大于 0。")
        if not 0.0 <= self.relevance_threshold <= 1.0:
            raise ValueError("相关度阈值必须在 0 到 1 之间。")
        if not 0.0 <= self.baseline_vector_weight <= 1.0 or not 0.0 <= (
            self.baseline_keyword_weight
        ) <= 1.0:
            raise ValueError("Baseline 权重必须在 0 到 1 之间。")
        if not math.isclose(
            self.baseline_vector_weight + self.baseline_keyword_weight,
            1.0,
            abs_tol=1e-9,
        ):
            raise ValueError("Baseline 向量权重与关键词权重之和必须为 1。")
        if self.rrf_k <= 0 or self.bm25_k1 <= 0 or not 0.0 <= self.bm25_b <= 1.0:
            raise ValueError("BM25 与 RRF 参数无效。")
        if self.adjacent_window < 0 or self.context_limit <= 0:
            raise ValueError("相邻窗口不能为负数，上下文上限必须大于 0。")


@dataclass(frozen=True)
class HybridEvaluationCaseResult:
    case_id: str
    answerable: bool
    baseline_metrics: dict[str, float | None]
    metrics: dict[str, float | None]
    metric_deltas: dict[str, float | None]
    case_analysis: dict[str, object]
    baseline_failure_category: str | None
    failure_category: str | None
    trace: dict[str, object]


def _identity(document: Document) -> dict[str, object]:
    evidence = evidence_from_document(document, "S1")
    return {
        "chunk_id": evidence.chunk_id,
        "material_id": evidence.material_id,
        "source": evidence.source,
        "page": evidence.page,
        "section": evidence.section,
        "chunk_index": evidence.chunk_index,
        "content_hash": evidence.content_hash,
        "locator": evidence.locator,
    }


def _ranking_metrics(
    ranked_chunk_ids: Sequence[str],
    gold_chunk_ids: set[str],
    *,
    answerable: bool,
) -> dict[str, float | None]:
    if not answerable:
        return {
            **{f"recall_at_{k}": None for k in RECALL_K_VALUES},
            "mrr": None,
            "ndcg_at_5": None,
        }
    return {
        **{
            f"recall_at_{k}": recall_at_k(ranked_chunk_ids, gold_chunk_ids, k)
            for k in RECALL_K_VALUES
        },
        "mrr": reciprocal_rank(ranked_chunk_ids, gold_chunk_ids),
        "ndcg_at_5": ndcg_at_k(ranked_chunk_ids, gold_chunk_ids, 5),
    }


def _metric_deltas(
    baseline: Mapping[str, float | None],
    experiment: Mapping[str, float | None],
) -> dict[str, float | None]:
    return {
        name: (
            float(experiment[name]) - float(baseline[name])
            if isinstance(experiment.get(name), (float, int))
            and isinstance(baseline.get(name), (float, int))
            else None
        )
        for name in experiment
    }


def _gold_ranks(
    ranked_chunk_ids: Sequence[str], gold_chunk_ids: set[str]
) -> dict[str, int | None]:
    positions = {chunk_id: rank for rank, chunk_id in enumerate(ranked_chunk_ids, start=1)}
    return {chunk_id: positions.get(chunk_id) for chunk_id in sorted(gold_chunk_ids)}


def _best_gold_rank(ranks: Mapping[str, int | None]) -> int | None:
    present = [rank for rank in ranks.values() if isinstance(rank, int)]
    return min(present) if present else None


def _case_outcome(
    *,
    answerable: bool,
    baseline_ranks: Mapping[str, int | None],
    experiment_ranks: Mapping[str, int | None],
) -> str:
    if not answerable:
        return "unanswerable"
    baseline_best = _best_gold_rank(baseline_ranks)
    experiment_best = _best_gold_rank(experiment_ranks)
    if baseline_best is None and experiment_best is not None:
        return "added_recall"
    if baseline_best is not None and experiment_best is None:
        return "lost_recall"
    if baseline_best is None:
        return "unchanged_miss"
    if experiment_best < baseline_best:
        return "ranking_improved"
    if experiment_best > baseline_best:
        return "ranking_regressed"
    return "unchanged"


def _failure_category(
    *,
    answerable: bool,
    ranked_chunk_ids: Sequence[str],
    final_chunk_ids: Sequence[str],
    gold_chunk_ids: set[str],
    retrieval_limit: int,
) -> str | None:
    if not answerable:
        return "unanswerable_handling_failure" if final_chunk_ids else None
    ranked = set(ranked_chunk_ids)
    if not ranked & gold_chunk_ids:
        return "recall_failure"
    if not set(ranked_chunk_ids[:retrieval_limit]) & gold_chunk_ids:
        return "ranking_failure"
    if not set(final_chunk_ids) & gold_chunk_ids:
        return "context_construction_failure"
    return None


def _expand_context(
    vector_store: object,
    seed_documents: list[Document],
    config: HybridRetrievalConfig,
) -> list[Document]:
    return (
        expand_adjacent_documents(
            vector_store,
            seed_documents,
            adjacent_window=config.adjacent_window,
            context_limit=config.context_limit,
        )
        if config.adjacent_window
        else seed_documents
    )


def evaluate_hybrid_case(
    case: RetrievalEvaluationCase,
    vector_store: object,
    bm25_index: BM25Index,
    config: HybridRetrievalConfig,
) -> HybridEvaluationCaseResult:
    """同一次 Dense Search 同时供 Baseline 与实验使用，避免额外模型变量。"""
    total_started = perf_counter()
    dense_started = perf_counter()
    dense_results = search_vector_store(
        case.question,
        vector_store,
        limit=config.dense_candidate_limit,
    )
    dense_elapsed = round((perf_counter() - dense_started) * 1000)

    baseline_started = perf_counter()
    baseline_ranked = rank_vector_candidates(
        case.question,
        dense_results,
        relevance_threshold=config.relevance_threshold,
        keyword_weight=config.baseline_keyword_weight,
    )
    baseline_elapsed = round((perf_counter() - baseline_started) * 1000)

    bm25_started = perf_counter()
    bm25_results = bm25_index.search(case.question, limit=config.bm25_candidate_limit)
    bm25_elapsed = round((perf_counter() - bm25_started) * 1000)

    retained_dense = [
        (document, score)
        for document, score in dense_results
        if score >= config.relevance_threshold
    ]
    fusion_started = perf_counter()
    fused = reciprocal_rank_fusion(retained_dense, bm25_results, rrf_k=config.rrf_k)
    fusion_elapsed = round((perf_counter() - fusion_started) * 1000)

    baseline_documents = [result.document for result in baseline_ranked]
    experiment_documents = [result.document for result in fused]
    baseline_ids = [_identity(document)["chunk_id"] for document in baseline_documents]
    experiment_ids = [result.chunk_id for result in fused]
    baseline_seed_documents = baseline_documents[: config.retrieval_limit]
    experiment_seed_documents = experiment_documents[: config.retrieval_limit]

    baseline_context_started = perf_counter()
    baseline_context = _expand_context(vector_store, baseline_seed_documents, config)
    baseline_context_elapsed = round((perf_counter() - baseline_context_started) * 1000)
    experiment_context_started = perf_counter()
    experiment_context = _expand_context(vector_store, experiment_seed_documents, config)
    experiment_context_elapsed = round((perf_counter() - experiment_context_started) * 1000)

    baseline_final_ids = [str(_identity(document)["chunk_id"]) for document in baseline_context]
    experiment_final_ids = [
        str(_identity(document)["chunk_id"]) for document in experiment_context
    ]
    gold = set(case.gold_chunk_ids)
    baseline_metrics = _ranking_metrics(
        baseline_ids, gold, answerable=case.answerable
    )
    metrics = _ranking_metrics(experiment_ids, gold, answerable=case.answerable)
    baseline_ranks = _gold_ranks(baseline_ids, gold)
    experiment_ranks = _gold_ranks(experiment_ids, gold)
    added_gold = sorted(
        chunk_id
        for chunk_id in gold
        if chunk_id in experiment_ids and chunk_id not in baseline_ids
    )
    lost_gold = sorted(
        chunk_id
        for chunk_id in gold
        if chunk_id in baseline_ids and chunk_id not in experiment_ids
    )
    total_elapsed = round((perf_counter() - total_started) * 1000)
    baseline_total_elapsed = dense_elapsed + baseline_elapsed + baseline_context_elapsed
    experiment_total_elapsed = (
        dense_elapsed + bm25_elapsed + fusion_elapsed + experiment_context_elapsed
    )

    return HybridEvaluationCaseResult(
        case_id=case.case_id,
        answerable=case.answerable,
        baseline_metrics=baseline_metrics,
        metrics=metrics,
        metric_deltas=_metric_deltas(baseline_metrics, metrics),
        case_analysis={
            "outcome": _case_outcome(
                answerable=case.answerable,
                baseline_ranks=baseline_ranks,
                experiment_ranks=experiment_ranks,
            ),
            "baseline_gold_ranks": baseline_ranks,
            "experiment_gold_ranks": experiment_ranks,
            "baseline_best_gold_rank": _best_gold_rank(baseline_ranks),
            "experiment_best_gold_rank": _best_gold_rank(experiment_ranks),
            "added_gold_chunk_ids": added_gold,
            "lost_gold_chunk_ids": lost_gold,
        },
        baseline_failure_category=_failure_category(
            answerable=case.answerable,
            ranked_chunk_ids=baseline_ids,
            final_chunk_ids=baseline_final_ids,
            gold_chunk_ids=gold,
            retrieval_limit=config.retrieval_limit,
        ),
        failure_category=_failure_category(
            answerable=case.answerable,
            ranked_chunk_ids=experiment_ids,
            final_chunk_ids=experiment_final_ids,
            gold_chunk_ids=gold,
            retrieval_limit=config.retrieval_limit,
        ),
        trace={
            "query": {"case_id": case.case_id, "question": case.question},
            "dense_candidates": [
                {
                    **_identity(document),
                    "dense_rank": rank,
                    "vector_score": score,
                    "retained": score >= config.relevance_threshold,
                }
                for rank, (document, score) in enumerate(dense_results, start=1)
            ],
            "bm25_candidates": [
                {
                    **_identity(result.document),
                    "bm25_rank": result.rank,
                    "bm25_score": result.score,
                }
                for result in bm25_results
            ],
            "baseline_ranking": [
                {
                    **_identity(result.document),
                    "rank": rank,
                    "dense_rank": result.raw_rank,
                    "vector_score": result.vector_score,
                    "keyword_coverage": result.keyword_score,
                    "combined_score": result.combined_score,
                }
                for rank, result in enumerate(baseline_ranked, start=1)
            ],
            "rrf_ranking": [
                {
                    **_identity(result.document),
                    "rank": rank,
                    "rrf_score": result.rrf_score,
                    "dense_rank": result.dense_rank,
                    "bm25_rank": result.bm25_rank,
                    "vector_score": result.vector_score,
                    "bm25_score": result.bm25_score,
                }
                for rank, result in enumerate(fused, start=1)
            ],
            "baseline_final_context_chunk_ids": baseline_final_ids,
            "experiment_final_context_chunk_ids": experiment_final_ids,
            "latency_ms": {
                "dense_search": dense_elapsed,
                "baseline_ranking": baseline_elapsed,
                "bm25_search": bm25_elapsed,
                "rrf_fusion": fusion_elapsed,
                "baseline_context": baseline_context_elapsed,
                "experiment_context": experiment_context_elapsed,
                "baseline_total": baseline_total_elapsed,
                "experiment_total": experiment_total_elapsed,
                "ab_total": total_elapsed,
            },
        },
    )


def evaluate_hybrid_cases(
    cases: Sequence[RetrievalEvaluationCase],
    vector_store: object,
    bm25_index: BM25Index,
    config: HybridRetrievalConfig,
) -> tuple[HybridEvaluationCaseResult, ...]:
    return tuple(
        evaluate_hybrid_case(case, vector_store, bm25_index, config) for case in cases
    )


def _average_metrics(
    results: Sequence[HybridEvaluationCaseResult], field: str
) -> dict[str, float | None]:
    metric_names = sorted(
        {
            name
            for result in results
            for name in getattr(result, field)
        }
    )
    averages: dict[str, float | None] = {}
    for name in metric_names:
        values = [
            metrics[name]
            for result in results
            for metrics in [getattr(result, field)]
            if isinstance(metrics.get(name), (float, int))
        ]
        averages[name] = sum(float(value) for value in values) / len(values) if values else None
    return averages


def build_hybrid_report(
    results: Sequence[HybridEvaluationCaseResult],
    *,
    config: HybridRetrievalConfig,
    git_commit: str,
    stage3_1_start_commit: str,
    dataset_metadata: Mapping[str, object],
    embedding_model: str | None,
    query_embedding_calls: int,
    validation_level: str,
    index_fingerprint: str,
    generated_at: datetime,
) -> dict[str, object]:
    if generated_at.tzinfo is None:
        raise ValueError("generated_at 必须包含时区。")
    if query_embedding_calls < 0:
        raise ValueError("Query Embedding 调用数不能为负数。")
    baseline_metrics = _average_metrics(results, "baseline_metrics")
    experiment_metrics = _average_metrics(results, "metrics")
    ab_latencies = [int(result.trace["latency_ms"]["ab_total"]) for result in results]
    baseline_latencies = [
        int(result.trace["latency_ms"]["baseline_total"]) for result in results
    ]
    experiment_latencies = [
        int(result.trace["latency_ms"]["experiment_total"]) for result in results
    ]
    bm25_latencies = [int(result.trace["latency_ms"]["bm25_search"]) for result in results]
    report = {
        "report_schema_version": HYBRID_REPORT_SCHEMA_VERSION,
        "experiment_name": HYBRID_EXPERIMENT_NAME,
        "baseline": HYBRID_BASELINE_NAME,
        "baseline_retrieval": "dense_keyword_coverage_rerank",
        "git_commit": git_commit,
        "stage3_1_start_commit": stage3_1_start_commit,
        "evaluation_dataset": dict(dataset_metadata),
        "index_fingerprint_sha256": index_fingerprint,
        "retrieval_config": asdict(config),
        "baseline_metrics": baseline_metrics,
        "metrics": experiment_metrics,
        "metric_deltas": _metric_deltas(baseline_metrics, experiment_metrics),
        "per_case_results": [asdict(result) for result in results],
        "failure_counts": {
            "baseline": _failure_counts(
                result.baseline_failure_category for result in results
            ),
            "experiment": _failure_counts(result.failure_category for result in results),
        },
        "latency": {
            "baseline_average_retrieval_ms": (
                sum(baseline_latencies) / len(baseline_latencies)
                if baseline_latencies
                else None
            ),
            "experiment_average_retrieval_ms": (
                sum(experiment_latencies) / len(experiment_latencies)
                if experiment_latencies
                else None
            ),
            "average_retrieval_delta_ms": (
                (sum(experiment_latencies) - sum(baseline_latencies))
                / len(experiment_latencies)
                if experiment_latencies
                else None
            ),
            "average_bm25_ms": (
                sum(bm25_latencies) / len(bm25_latencies) if bm25_latencies else None
            ),
            "ab_run_total_ms": sum(ab_latencies),
            "measurement": "local_wall_clock",
        },
        "models": {
            "embedding": {
                "model": embedding_model,
                "validation_level": (
                    "real_embedding"
                    if validation_level == "local_real_retrieval"
                    else validation_level
                ),
            },
            "chat": None,
        },
        "cost": {
            "measured": False,
            "amount": None,
            "currency": None,
            "query_embedding_calls": query_embedding_calls,
        },
        "validation_level": validation_level,
        "run_status": "completed",
        "generated_at": generated_at.astimezone(UTC).isoformat().replace("+00:00", "Z"),
    }
    return report


def _failure_counts(categories: Iterable[str | None]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for category in categories:
        if category is not None:
            counts[category] = counts.get(category, 0) + 1
    return counts


def write_hybrid_report(report: Mapping[str, object], output_directory: Path) -> Path:
    output_directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    destination = (
        output_directory / f"hybrid-retrieval-{timestamp}-{uuid4().hex[:8]}.json"
    )
    with destination.open("x", encoding="utf-8", newline="\n") as output:
        output.write(json.dumps(dict(report), ensure_ascii=False, indent=2) + "\n")
    return destination
