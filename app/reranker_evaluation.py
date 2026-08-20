"""Stage 3.2：比较 Baseline、Hybrid 与 Hybrid + Reranker。"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
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
    from app.reranker import Reranker
    from app.retrieval_evaluation import (
        RECALL_K_VALUES,
        RetrievalEvaluationCase,
        context_precision,
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
    from reranker import Reranker
    from retrieval_evaluation import (
        RECALL_K_VALUES,
        RetrievalEvaluationCase,
        context_precision,
        ndcg_at_k,
        recall_at_k,
        reciprocal_rank,
    )


RERANKER_REPORT_SCHEMA_VERSION = 1
RERANKER_EXPERIMENT_NAME = "stage3_2_cross_encoder_reranker"
STAGE2_BASELINE_NAME = "stage2_baseline"
STAGE3_1_HYBRID_NAME = "stage3_1_hybrid_rrf"
DEFAULT_RERANKER_CANDIDATE_LIMIT = 20


@dataclass(frozen=True)
class RerankerEvaluationConfig:
    retrieval_limit: int = 3
    dense_candidate_limit: int = DEFAULT_CANDIDATE_LIMIT
    bm25_candidate_limit: int = DEFAULT_CANDIDATE_LIMIT
    reranker_candidate_limit: int = DEFAULT_RERANKER_CANDIDATE_LIMIT
    relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD
    baseline_vector_weight: float = 1.0 - DEFAULT_KEYWORD_WEIGHT
    baseline_keyword_weight: float = DEFAULT_KEYWORD_WEIGHT
    rrf_k: int = DEFAULT_RRF_K
    bm25_k1: float = DEFAULT_BM25_K1
    bm25_b: float = DEFAULT_BM25_B
    adjacent_window: int = 2
    context_limit: int = 8

    def __post_init__(self) -> None:
        if min(
            self.retrieval_limit,
            self.dense_candidate_limit,
            self.bm25_candidate_limit,
            self.reranker_candidate_limit,
            self.context_limit,
        ) <= 0:
            raise ValueError("检索、候选池与上下文数量必须大于 0。")
        maximum_union = self.dense_candidate_limit + self.bm25_candidate_limit
        if self.reranker_candidate_limit > maximum_union:
            raise ValueError("Reranker Candidate Pool 不能超过 Dense 与 BM25 候选上限之和。")
        if not 0.0 <= self.relevance_threshold <= 1.0:
            raise ValueError("相关度阈值必须在 0 到 1 之间。")
        if not 0.0 <= self.baseline_keyword_weight <= 1.0:
            raise ValueError("Baseline 关键词权重必须在 0 到 1 之间。")
        if not math.isclose(
            self.baseline_vector_weight + self.baseline_keyword_weight,
            1.0,
            abs_tol=1e-9,
        ):
            raise ValueError("Baseline 向量权重与关键词权重之和必须为 1。")
        if self.rrf_k <= 0 or self.bm25_k1 <= 0 or not 0.0 <= self.bm25_b <= 1.0:
            raise ValueError("BM25 与 RRF 参数无效。")
        if self.adjacent_window < 0:
            raise ValueError("相邻窗口不能为负数。")


@dataclass(frozen=True)
class RerankerEvaluationCaseResult:
    case_id: str
    answerable: bool
    baseline_metrics: dict[str, float | None]
    hybrid_metrics: dict[str, float | None]
    reranker_metrics: dict[str, float | None]
    hybrid_vs_baseline_deltas: dict[str, float | None]
    reranker_vs_hybrid_deltas: dict[str, float | None]
    reranker_vs_baseline_deltas: dict[str, float | None]
    context_metrics: dict[str, dict[str, float | int | None]]
    case_analysis: dict[str, object]
    failure_categories: dict[str, str | None]
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


def _context_metrics(
    final_chunk_ids: Sequence[str],
    gold_chunk_ids: set[str],
    *,
    answerable: bool,
) -> dict[str, float | int | None]:
    duplicate_count = len(final_chunk_ids) - len(set(final_chunk_ids))
    return {
        "context_precision": context_precision(
            final_chunk_ids,
            gold_chunk_ids,
            answerable=answerable,
        ),
        "final_context_recall": (
            len(set(final_chunk_ids) & gold_chunk_ids) / len(gold_chunk_ids)
            if gold_chunk_ids
            else None
        ),
        "context_size": len(final_chunk_ids),
        "duplicate_ratio": (
            duplicate_count / len(final_chunk_ids) if final_chunk_ids else 0.0
        ),
    }


def _metric_deltas(
    baseline: Mapping[str, float | None],
    experiment: Mapping[str, float | None],
) -> dict[str, float | None]:
    return {
        name: (
            float(experiment[name]) - float(baseline[name])
            if isinstance(experiment.get(name), (float, int))
            and not isinstance(experiment.get(name), bool)
            and isinstance(baseline.get(name), (float, int))
            and not isinstance(baseline.get(name), bool)
            else None
        )
        for name in sorted(set(baseline) | set(experiment))
    }


def _gold_ranks(
    ranked_chunk_ids: Sequence[str],
    gold_chunk_ids: set[str],
) -> dict[str, int | None]:
    positions = {chunk_id: rank for rank, chunk_id in enumerate(ranked_chunk_ids, start=1)}
    return {chunk_id: positions.get(chunk_id) for chunk_id in sorted(gold_chunk_ids)}


def _best_gold_rank(ranks: Mapping[str, int | None]) -> int | None:
    present = [rank for rank in ranks.values() if isinstance(rank, int)]
    return min(present) if present else None


def _rank_outcome(before: int | None, after: int | None, *, answerable: bool) -> str:
    if not answerable:
        return "unanswerable"
    if before is None and after is not None:
        return "added_recall"
    if before is not None and after is None:
        return "lost_recall"
    if before is None:
        return "unchanged_miss"
    if after < before:
        return "ranking_improved"
    if after > before:
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
    if not set(ranked_chunk_ids) & gold_chunk_ids:
        return "recall_failure"
    if not set(ranked_chunk_ids[:retrieval_limit]) & gold_chunk_ids:
        return "ranking_failure"
    if not set(final_chunk_ids) & gold_chunk_ids:
        return "context_construction_failure"
    return None


def _expand_context(
    vector_store: object,
    documents: list[Document],
    config: RerankerEvaluationConfig,
) -> list[Document]:
    return (
        expand_adjacent_documents(
            vector_store,
            documents,
            adjacent_window=config.adjacent_window,
            context_limit=config.context_limit,
        )
        if config.adjacent_window
        else documents
    )


def evaluate_reranker_case(
    case: RetrievalEvaluationCase,
    vector_store: object,
    bm25_index: BM25Index,
    reranker: Reranker,
    config: RerankerEvaluationConfig,
) -> RerankerEvaluationCaseResult:
    """用同一次 Dense Search 构建 A/B/C 三个实验对象。"""
    ab_total_started = perf_counter()
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
    baseline_ranking_elapsed = round((perf_counter() - baseline_started) * 1000)

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
    candidate_pool = fused[: config.reranker_candidate_limit]

    reranker_started = perf_counter()
    reranked = reranker.rerank(
        case.question,
        [result.document for result in candidate_pool],
    )
    reranker_elapsed = round((perf_counter() - reranker_started) * 1000)

    baseline_documents = [result.document for result in baseline_ranked]
    hybrid_documents = [result.document for result in fused]
    reranker_documents = [result.document for result in reranked]
    baseline_ids = [str(_identity(document)["chunk_id"]) for document in baseline_documents]
    hybrid_ids = [result.chunk_id for result in fused]
    reranker_ids = [result.chunk_id for result in reranked]

    context_documents: dict[str, list[Document]] = {}
    context_latencies: dict[str, int] = {}
    for name, documents in (
        ("baseline", baseline_documents),
        ("hybrid", hybrid_documents),
        ("reranker", reranker_documents),
    ):
        context_started = perf_counter()
        context_documents[name] = _expand_context(
            vector_store,
            documents[: config.retrieval_limit],
            config,
        )
        context_latencies[name] = round((perf_counter() - context_started) * 1000)

    final_ids = {
        name: [str(_identity(document)["chunk_id"]) for document in documents]
        for name, documents in context_documents.items()
    }
    gold = set(case.gold_chunk_ids)
    baseline_metrics = _ranking_metrics(baseline_ids, gold, answerable=case.answerable)
    hybrid_metrics = _ranking_metrics(hybrid_ids, gold, answerable=case.answerable)
    reranker_metrics = _ranking_metrics(reranker_ids, gold, answerable=case.answerable)
    ranking_ids = {
        "baseline": baseline_ids,
        "hybrid": hybrid_ids,
        "reranker": reranker_ids,
    }
    gold_ranks = {
        name: _gold_ranks(ids, gold)
        for name, ids in ranking_ids.items()
    }
    best_gold_ranks = {
        name: _best_gold_rank(ranks)
        for name, ranks in gold_ranks.items()
    }
    context = {
        name: _context_metrics(ids, gold, answerable=case.answerable)
        for name, ids in final_ids.items()
    }
    failure_categories = {
        name: _failure_category(
            answerable=case.answerable,
            ranked_chunk_ids=ranking_ids[name],
            final_chunk_ids=final_ids[name],
            gold_chunk_ids=gold,
            retrieval_limit=config.retrieval_limit,
        )
        for name in ("baseline", "hybrid", "reranker")
    }
    ab_total_elapsed = round((perf_counter() - ab_total_started) * 1000)

    pool_by_chunk_id = {result.chunk_id: result for result in candidate_pool}
    return RerankerEvaluationCaseResult(
        case_id=case.case_id,
        answerable=case.answerable,
        baseline_metrics=baseline_metrics,
        hybrid_metrics=hybrid_metrics,
        reranker_metrics=reranker_metrics,
        hybrid_vs_baseline_deltas=_metric_deltas(baseline_metrics, hybrid_metrics),
        reranker_vs_hybrid_deltas=_metric_deltas(hybrid_metrics, reranker_metrics),
        reranker_vs_baseline_deltas=_metric_deltas(baseline_metrics, reranker_metrics),
        context_metrics=context,
        case_analysis={
            "gold_ranks": gold_ranks,
            "best_gold_ranks": best_gold_ranks,
            "hybrid_vs_baseline_outcome": _rank_outcome(
                best_gold_ranks["baseline"],
                best_gold_ranks["hybrid"],
                answerable=case.answerable,
            ),
            "reranker_vs_hybrid_outcome": _rank_outcome(
                best_gold_ranks["hybrid"],
                best_gold_ranks["reranker"],
                answerable=case.answerable,
            ),
        },
        failure_categories=failure_categories,
        trace={
            "query": {"query_id": case.case_id},
            "candidate_pool": {
                "configured_limit": config.reranker_candidate_limit,
                "actual_size": len(candidate_pool),
                "corpus_reranked": False,
            },
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
            "reranker_ranking": [
                {
                    **_identity(result.document),
                    "original_rank": result.original_rank,
                    "original_rrf_score": pool_by_chunk_id[result.chunk_id].rrf_score,
                    "reranker_score": result.reranker_score,
                    "reranker_rank": result.reranker_rank,
                    "rank_change": result.rank_change,
                }
                for result in reranked
            ],
            "final_context_chunk_ids": final_ids,
            "latency_ms": {
                "dense_search": dense_elapsed,
                "baseline_ranking": baseline_ranking_elapsed,
                "bm25_search": bm25_elapsed,
                "rrf_fusion": fusion_elapsed,
                "reranker": reranker_elapsed,
                "baseline_context": context_latencies["baseline"],
                "hybrid_context": context_latencies["hybrid"],
                "reranker_context": context_latencies["reranker"],
                "baseline_total": (
                    dense_elapsed
                    + baseline_ranking_elapsed
                    + context_latencies["baseline"]
                ),
                "hybrid_total": (
                    dense_elapsed
                    + bm25_elapsed
                    + fusion_elapsed
                    + context_latencies["hybrid"]
                ),
                "reranker_total": (
                    dense_elapsed
                    + bm25_elapsed
                    + fusion_elapsed
                    + reranker_elapsed
                    + context_latencies["reranker"]
                ),
                "abc_total": ab_total_elapsed,
            },
        },
    )


def evaluate_reranker_cases(
    cases: Sequence[RetrievalEvaluationCase],
    vector_store: object,
    bm25_index: BM25Index,
    reranker: Reranker,
    config: RerankerEvaluationConfig,
) -> tuple[RerankerEvaluationCaseResult, ...]:
    return tuple(
        evaluate_reranker_case(case, vector_store, bm25_index, reranker, config)
        for case in cases
    )


def _average_mapping(
    results: Sequence[RerankerEvaluationCaseResult],
    field: str,
) -> dict[str, float | None]:
    mappings = [getattr(result, field) for result in results]
    names = sorted({name for mapping in mappings for name in mapping})
    return {
        name: (
            sum(float(mapping[name]) for mapping in mappings if mapping[name] is not None)
            / sum(mapping[name] is not None for mapping in mappings)
            if any(mapping[name] is not None for mapping in mappings)
            else None
        )
        for name in names
    }


def _average_context(
    results: Sequence[RerankerEvaluationCaseResult],
    strategy: str,
) -> dict[str, float | None]:
    names = sorted(
        {
            name
            for result in results
            for name in result.context_metrics[strategy]
        }
    )
    averages: dict[str, float | None] = {}
    for name in names:
        values = [
            float(result.context_metrics[strategy][name])
            for result in results
            if result.context_metrics[strategy][name] is not None
        ]
        averages[name] = sum(values) / len(values) if values else None
    return averages


def _failure_counts(categories: Iterable[str | None]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for category in categories:
        if category is not None:
            counts[category] = counts.get(category, 0) + 1
    return counts


def build_reranker_report(
    results: Sequence[RerankerEvaluationCaseResult],
    *,
    config: RerankerEvaluationConfig,
    reranker: Reranker,
    git_commit: str,
    stage3_2_start_commit: str,
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

    metrics = {
        "baseline": _average_mapping(results, "baseline_metrics"),
        "hybrid": _average_mapping(results, "hybrid_metrics"),
        "reranker": _average_mapping(results, "reranker_metrics"),
    }
    context_metrics = {
        strategy: _average_context(results, strategy)
        for strategy in ("baseline", "hybrid", "reranker")
    }
    latencies = {
        strategy: [
            int(result.trace["latency_ms"][f"{strategy}_total"])
            for result in results
        ]
        for strategy in ("baseline", "hybrid", "reranker")
    }
    pair_count = sum(
        int(result.trace["candidate_pool"]["actual_size"])
        for result in results
    )
    return {
        "report_schema_version": RERANKER_REPORT_SCHEMA_VERSION,
        "experiment_name": RERANKER_EXPERIMENT_NAME,
        "strategies": {
            "baseline": STAGE2_BASELINE_NAME,
            "hybrid": STAGE3_1_HYBRID_NAME,
            "reranker": RERANKER_EXPERIMENT_NAME,
        },
        "git_commit": git_commit,
        "stage3_2_start_commit": stage3_2_start_commit,
        "evaluation_dataset": dict(dataset_metadata),
        "index_fingerprint_sha256": index_fingerprint,
        "retrieval_config": asdict(config),
        "metrics": metrics,
        "metric_deltas": {
            "hybrid_vs_baseline": _metric_deltas(metrics["baseline"], metrics["hybrid"]),
            "reranker_vs_hybrid": _metric_deltas(metrics["hybrid"], metrics["reranker"]),
            "reranker_vs_baseline": _metric_deltas(metrics["baseline"], metrics["reranker"]),
        },
        "context_metrics": context_metrics,
        "context_metric_deltas": {
            "hybrid_vs_baseline": _metric_deltas(
                context_metrics["baseline"], context_metrics["hybrid"]
            ),
            "reranker_vs_hybrid": _metric_deltas(
                context_metrics["hybrid"], context_metrics["reranker"]
            ),
            "reranker_vs_baseline": _metric_deltas(
                context_metrics["baseline"], context_metrics["reranker"]
            ),
        },
        "failure_counts": {
            strategy: _failure_counts(
                result.failure_categories[strategy] for result in results
            )
            for strategy in ("baseline", "hybrid", "reranker")
        },
        "latency": {
            f"{strategy}_average_retrieval_ms": (
                sum(values) / len(values) if values else None
            )
            for strategy, values in latencies.items()
        }
        | {
            "reranker_average_incremental_ms": (
                (
                    sum(latencies["reranker"])
                    - sum(latencies["hybrid"])
                )
                / len(latencies["reranker"])
                if latencies["reranker"]
                else None
            ),
            "measurement": "local_wall_clock",
        },
        "models": {
            "embedding": {
                "model": embedding_model,
                "validation_level": (
                    "real_embedding"
                    if validation_level
                    in {"local_real_retrieval", "local_real_cross_encoder"}
                    else validation_level
                ),
            },
            "reranker": {
                "model": reranker.model_name,
                "source": reranker.model_source,
                "validation_level": (
                    "real_cross_encoder"
                    if validation_level == "local_real_cross_encoder"
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
            "reranker_pair_scores": pair_count,
            "chat_calls": 0,
        },
        "per_case_results": [asdict(result) for result in results],
        "validation_level": validation_level,
        "run_status": "completed",
        "generated_at": generated_at.astimezone(UTC).isoformat().replace("+00:00", "Z"),
    }


def write_reranker_report(
    report: Mapping[str, object],
    output_directory: Path,
) -> Path:
    output_directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    destination = (
        output_directory / f"reranker-retrieval-{timestamp}-{uuid4().hex[:8]}.json"
    )
    with destination.open("x", encoding="utf-8", newline="\n") as output:
        output.write(json.dumps(dict(report), ensure_ascii=False, indent=2) + "\n")
    return destination
