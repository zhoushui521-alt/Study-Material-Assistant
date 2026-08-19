"""Stage 2 的独立 Retrieval Evaluation、Trace 与失败诊断。"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from langchain_core.documents import Document

if __package__:
    from app.config import PROJECT_ROOT
    from app.evidence import evidence_from_document
    from app.hybrid_search import (
        DEFAULT_CANDIDATE_LIMIT,
        DEFAULT_KEYWORD_WEIGHT,
        DEFAULT_RELEVANCE_THRESHOLD,
        expand_adjacent_documents,
        rank_vector_candidates,
    )
    from app.langchain_store import search_vector_store
else:
    from config import PROJECT_ROOT
    from evidence import evidence_from_document
    from hybrid_search import (
        DEFAULT_CANDIDATE_LIMIT,
        DEFAULT_KEYWORD_WEIGHT,
        DEFAULT_RELEVANCE_THRESHOLD,
        expand_adjacent_documents,
        rank_vector_candidates,
    )
    from langchain_store import search_vector_store


DEFAULT_RETRIEVAL_EVALUATION_PATH = (
    PROJECT_ROOT / "evaluation" / "retrieval_cases.json"
)
DEFAULT_RETRIEVAL_RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
RETRIEVAL_DATASET_SCHEMA_VERSION = 1
RETRIEVAL_REPORT_SCHEMA_VERSION = 2
RETRIEVAL_EVALUATION_VERSION = "study-material-retrieval-eval-v1"
CASE_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]*")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
GIT_COMMIT_PATTERN = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
RECALL_K_VALUES = (1, 3, 5, 10)


class RetrievalEvaluationDataError(ValueError):
    """Retrieval Dataset 或 Gold Mapping 不符合稳定契约。"""


class RetrievalEvaluationReportError(RuntimeError):
    """Retrieval Report 无法安全生成或写入。"""


class FailureCategory(StrEnum):
    PARSING_FAILURE = "parsing_failure"
    CHUNKING_FAILURE = "chunking_failure"
    RECALL_FAILURE = "recall_failure"
    FILTERING_FAILURE = "filtering_failure"
    RANKING_FAILURE = "ranking_failure"
    CONTEXT_CONSTRUCTION_FAILURE = "context_construction_failure"
    GENERATION_FAILURE = "generation_failure"
    CITATION_FAILURE = "citation_failure"
    UNANSWERABLE_HANDLING_FAILURE = "unanswerable_handling_failure"
    NEEDS_MANUAL_REVIEW = "needs_manual_review"


@dataclass(frozen=True)
class StableGoldMeaning:
    """不依赖当前切块编号的人工 Gold 语义定位。"""

    gold_id: str
    material_id: str
    source: str
    locator: str
    text_span: str
    page: int | None = None
    section: str | None = None


@dataclass(frozen=True)
class CurrentChunkMapping:
    """当前 Chunker 对 Stable Gold Meaning 的可更新映射。"""

    gold_ids: tuple[str, ...]
    chunk_id: str
    material_id: str
    content_hash: str
    source: str
    page: int | None = None
    section: str | None = None
    legacy_chunk_index: int | None = None


@dataclass(frozen=True)
class RetrievalEvaluationCase:
    case_id: str
    dataset_version: str
    question: str
    answerable: bool
    case_types: tuple[str, ...]
    expected_material_ids: tuple[str, ...]
    expected_sources: tuple[str, ...]
    stable_gold_meanings: tuple[StableGoldMeaning, ...]
    current_chunk_mappings: tuple[CurrentChunkMapping, ...]
    annotation_notes: str

    @property
    def gold_chunk_ids(self) -> frozenset[str]:
        return frozenset(mapping.chunk_id for mapping in self.current_chunk_mappings)


@dataclass(frozen=True)
class RetrievalConfig:
    retrieval_limit: int = 3
    candidate_limit: int = DEFAULT_CANDIDATE_LIMIT
    relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD
    vector_weight: float = 1.0 - DEFAULT_KEYWORD_WEIGHT
    keyword_weight: float = DEFAULT_KEYWORD_WEIGHT
    adjacent_window: int = 2
    context_limit: int = 8

    def __post_init__(self) -> None:
        if self.retrieval_limit <= 0 or self.candidate_limit <= 0:
            raise ValueError("检索数量必须大于 0。")
        if not 0.0 <= self.relevance_threshold <= 1.0:
            raise ValueError("相关度阈值必须在 0 到 1 之间。")
        if not 0.0 <= self.keyword_weight <= 1.0:
            raise ValueError("关键词权重必须在 0 到 1 之间。")
        if not math.isclose(
            self.vector_weight + self.keyword_weight,
            1.0,
            abs_tol=1e-9,
        ):
            raise ValueError("向量权重与关键词权重之和必须为 1。")
        if self.adjacent_window < 0 or self.context_limit <= 0:
            raise ValueError("相邻窗口不能为负数，上下文上限必须大于 0。")


@dataclass(frozen=True)
class RetrievalTrace:
    query: dict[str, object]
    raw_candidates: tuple[dict[str, object], ...]
    filtering: dict[str, object]
    hybrid_ranking: tuple[dict[str, object], ...]
    seeds: tuple[dict[str, object], ...]
    adjacent_expansion: tuple[dict[str, object], ...]
    final_context: dict[str, object]
    latency_ms: dict[str, int]


@dataclass(frozen=True)
class RetrievalCaseResult:
    case: RetrievalEvaluationCase
    trace: RetrievalTrace
    metrics: dict[str, object]
    failure_category: str | None
    failure_notes: tuple[str, ...] = ()


def _require_text(value: object, field: str, case_id: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RetrievalEvaluationDataError(
            f"案例 {case_id} 的 {field} 必须是非空字符串。"
        )
    return value.strip()


def _require_sha256(value: object, field: str, case_id: str) -> str:
    text = _require_text(value, field, case_id)
    if SHA256_PATTERN.fullmatch(text) is None:
        raise RetrievalEvaluationDataError(
            f"案例 {case_id} 的 {field} 必须是 64 位小写 SHA-256。"
        )
    return text


def _optional_positive_int(value: object, field: str, case_id: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RetrievalEvaluationDataError(
            f"案例 {case_id} 的 {field} 必须是正整数或 null。"
        )
    return value


def _parse_text_list(value: object, field: str, case_id: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise RetrievalEvaluationDataError(f"案例 {case_id} 的 {field} 必须是列表。")
    items = tuple(_require_text(item, field, case_id) for item in value)
    if len(items) != len(set(items)):
        raise RetrievalEvaluationDataError(f"案例 {case_id} 的 {field} 包含重复值。")
    return items


def _parse_stable_gold(
    value: object,
    case_id: str,
) -> tuple[StableGoldMeaning, ...]:
    if not isinstance(value, list):
        raise RetrievalEvaluationDataError(
            f"案例 {case_id} 的 stable_gold_meanings 必须是列表。"
        )
    result = []
    for position, raw in enumerate(value, start=1):
        if not isinstance(raw, dict):
            raise RetrievalEvaluationDataError(
                f"案例 {case_id} 的第 {position} 个 stable_gold_meanings 必须是对象。"
            )
        result.append(
            StableGoldMeaning(
                gold_id=_require_text(raw.get("gold_id"), "stable_gold.gold_id", case_id),
                material_id=_require_sha256(
                    raw.get("material_id"), "stable_gold.material_id", case_id
                ),
                source=_require_text(raw.get("source"), "stable_gold.source", case_id),
                locator=_require_text(
                    raw.get("locator"), "stable_gold.locator", case_id
                ),
                text_span=_require_text(
                    raw.get("text_span"), "stable_gold.text_span", case_id
                ),
                page=_optional_positive_int(raw.get("page"), "stable_gold.page", case_id),
                section=(
                    _require_text(raw.get("section"), "stable_gold.section", case_id)
                    if raw.get("section") is not None
                    else None
                ),
            )
        )
    return tuple(result)


def _parse_chunk_mappings(
    value: object,
    case_id: str,
) -> tuple[CurrentChunkMapping, ...]:
    if not isinstance(value, list):
        raise RetrievalEvaluationDataError(
            f"案例 {case_id} 的 current_chunk_mappings 必须是列表。"
        )
    result = []
    seen_ids = set()
    for position, raw in enumerate(value, start=1):
        if not isinstance(raw, dict):
            raise RetrievalEvaluationDataError(
                f"案例 {case_id} 的第 {position} 个 current_chunk_mappings 必须是对象。"
            )
        chunk_id = _require_sha256(raw.get("chunk_id"), "mapping.chunk_id", case_id)
        if chunk_id in seen_ids:
            raise RetrievalEvaluationDataError(
                f"案例 {case_id} 包含重复的 current_chunk_mappings chunk_id。"
            )
        seen_ids.add(chunk_id)
        result.append(
            CurrentChunkMapping(
                gold_ids=_parse_text_list(
                    raw.get("gold_ids"), "mapping.gold_ids", case_id
                ),
                chunk_id=chunk_id,
                material_id=_require_sha256(
                    raw.get("material_id"), "mapping.material_id", case_id
                ),
                content_hash=_require_sha256(
                    raw.get("content_hash"), "mapping.content_hash", case_id
                ),
                source=_require_text(raw.get("source"), "mapping.source", case_id),
                page=_optional_positive_int(raw.get("page"), "mapping.page", case_id),
                section=(
                    _require_text(raw.get("section"), "mapping.section", case_id)
                    if raw.get("section") is not None
                    else None
                ),
                legacy_chunk_index=_optional_positive_int(
                    raw.get("legacy_chunk_index"),
                    "mapping.legacy_chunk_index",
                    case_id,
                ),
            )
        )
    return tuple(result)


def load_retrieval_cases(
    path: Path = DEFAULT_RETRIEVAL_EVALUATION_PATH,
) -> tuple[RetrievalEvaluationCase, ...]:
    """读取独立 Retrieval Dataset，并严格区分稳定 Gold 与当前 Chunk 映射。"""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise RetrievalEvaluationDataError(f"无法读取 Retrieval Dataset {path}。") from error
    except json.JSONDecodeError as error:
        raise RetrievalEvaluationDataError("Retrieval Dataset 不是合法 JSON。") from error
    if not isinstance(payload, dict):
        raise RetrievalEvaluationDataError("Retrieval Dataset 根节点必须是对象。")
    if payload.get("evaluation_version") != RETRIEVAL_DATASET_SCHEMA_VERSION:
        raise RetrievalEvaluationDataError("Retrieval Dataset 必须使用 evaluation_version=1。")
    dataset_version = _require_text(
        payload.get("dataset_version"), "dataset_version", "根对象"
    )
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise RetrievalEvaluationDataError("Retrieval Dataset cases 必须是非空列表。")

    cases = []
    seen_ids = set()
    for position, raw_case in enumerate(raw_cases, start=1):
        placeholder = f"第 {position} 条"
        if not isinstance(raw_case, dict):
            raise RetrievalEvaluationDataError(f"{placeholder}案例必须是对象。")
        case_id = _require_text(raw_case.get("case_id"), "case_id", placeholder)
        if CASE_ID_PATTERN.fullmatch(case_id) is None:
            raise RetrievalEvaluationDataError(f"案例 {case_id} 的 case_id 格式无效。")
        if case_id in seen_ids:
            raise RetrievalEvaluationDataError(f"Retrieval Dataset 包含重复 case_id：{case_id}。")
        seen_ids.add(case_id)
        case_dataset_version = _require_text(
            raw_case.get("dataset_version"), "dataset_version", case_id
        )
        if case_dataset_version != dataset_version:
            raise RetrievalEvaluationDataError(
                f"案例 {case_id} 的 dataset_version 与根对象不一致。"
            )
        answerable = raw_case.get("answerable")
        if not isinstance(answerable, bool):
            raise RetrievalEvaluationDataError(f"案例 {case_id} 的 answerable 必须是布尔值。")
        scope = raw_case.get("expected_scope")
        if not isinstance(scope, dict):
            raise RetrievalEvaluationDataError(f"案例 {case_id} 缺少 expected_scope。")
        expected_material_ids = tuple(
            _require_sha256(item, "expected_scope.material_ids", case_id)
            for item in scope.get("material_ids", [])
        )
        expected_sources = _parse_text_list(
            scope.get("sources"), "expected_scope.sources", case_id
        )
        stable_gold = _parse_stable_gold(raw_case.get("stable_gold_meanings"), case_id)
        mappings = _parse_chunk_mappings(raw_case.get("current_chunk_mappings"), case_id)
        if answerable and (
            not expected_material_ids
            or not expected_sources
            or not stable_gold
            or not mappings
        ):
            raise RetrievalEvaluationDataError(
                f"可回答案例 {case_id} 必须声明 scope、Stable Gold 和 Current Mapping。"
            )
        if not answerable and (
            expected_material_ids or expected_sources or stable_gold or mappings
        ):
            raise RetrievalEvaluationDataError(
                f"不可回答案例 {case_id} 不能伪造 Gold Evidence。"
            )
        if any(
            mapping.material_id not in expected_material_ids for mapping in mappings
        ):
            raise RetrievalEvaluationDataError(
                f"案例 {case_id} 的 Current Mapping 超出 expected material scope。"
            )
        if any(mapping.source not in expected_sources for mapping in mappings):
            raise RetrievalEvaluationDataError(
                f"案例 {case_id} 的 Current Mapping 超出 expected source scope。"
            )
        if any(
            meaning.material_id not in expected_material_ids
            or meaning.source not in expected_sources
            for meaning in stable_gold
        ):
            raise RetrievalEvaluationDataError(
                f"案例 {case_id} 的 Stable Gold Meaning 超出 expected scope。"
            )
        stable_gold_ids = {meaning.gold_id for meaning in stable_gold}
        if len(stable_gold_ids) != len(stable_gold):
            raise RetrievalEvaluationDataError(
                f"案例 {case_id} 包含重复的 stable_gold gold_id。"
            )
        referenced_gold_ids = {
            gold_id for mapping in mappings for gold_id in mapping.gold_ids
        }
        if any(not mapping.gold_ids for mapping in mappings) or (
            referenced_gold_ids != stable_gold_ids
        ):
            raise RetrievalEvaluationDataError(
                f"案例 {case_id} 的 Current Mapping 必须完整引用 Stable Gold gold_id。"
            )
        cases.append(
            RetrievalEvaluationCase(
                case_id=case_id,
                dataset_version=dataset_version,
                question=_require_text(raw_case.get("question"), "question", case_id),
                answerable=answerable,
                case_types=_parse_text_list(raw_case.get("case_types"), "case_types", case_id),
                expected_material_ids=expected_material_ids,
                expected_sources=expected_sources,
                stable_gold_meanings=stable_gold,
                current_chunk_mappings=mappings,
                annotation_notes=_require_text(
                    raw_case.get("annotation_notes"), "annotation_notes", case_id
                ),
            )
        )
    return tuple(cases)


def _document_trace_identity(document: Document) -> dict[str, object]:
    evidence = evidence_from_document(document, "S1")
    return {
        "evidence_id": evidence.evidence_id,
        "chunk_id": evidence.chunk_id,
        "material_id": evidence.material_id,
        "source": evidence.source,
        "page": evidence.page,
        "section": evidence.section,
        "chunk_index": evidence.chunk_index,
        "content_hash": evidence.content_hash,
        "locator": evidence.locator,
    }


def find_unresolved_gold_mappings(
    cases: Sequence[RetrievalEvaluationCase],
    indexed_documents: Sequence[Document],
) -> tuple[str, ...]:
    """在任何 Query Embedding 前验证 Current Mapping 仍能定位当前索引。"""
    available = {
        (
            identity["chunk_id"],
            identity["material_id"],
            identity["content_hash"],
            identity["source"],
            identity["page"],
            identity["section"],
        )
        for identity in (_document_trace_identity(document) for document in indexed_documents)
    }
    missing = []
    for case in cases:
        for mapping in case.current_chunk_mappings:
            key = (
                mapping.chunk_id,
                mapping.material_id,
                mapping.content_hash,
                mapping.source,
                mapping.page,
                mapping.section,
            )
            if key not in available:
                missing.append(f"{case.case_id}:{mapping.chunk_id}")
    return tuple(sorted(missing))


def _candidate_record(
    document: Document,
    *,
    raw_rank: int,
    vector_score: float,
) -> dict[str, object]:
    return {
        **_document_trace_identity(document),
        "raw_rank": raw_rank,
        "vector_score": vector_score,
    }


def trace_retrieval(
    case: RetrievalEvaluationCase,
    vector_store: object,
    config: RetrievalConfig,
) -> RetrievalTrace:
    """只执行当前生产 Retrieval 基线，并记录不含正文的 Evaluation Trace。"""
    total_started = perf_counter()
    raw_started = perf_counter()
    vector_results = search_vector_store(
        case.question,
        vector_store,
        limit=max(config.retrieval_limit, config.candidate_limit),
    )
    raw_elapsed = round((perf_counter() - raw_started) * 1000)
    raw_candidates = tuple(
        _candidate_record(document, raw_rank=rank, vector_score=score)
        for rank, (document, score) in enumerate(vector_results, start=1)
    )

    ranking_started = perf_counter()
    ranked = rank_vector_candidates(
        case.question,
        vector_results,
        relevance_threshold=config.relevance_threshold,
        keyword_weight=config.keyword_weight,
    )
    ranking_elapsed = round((perf_counter() - ranking_started) * 1000)
    retained_raw_ranks = {result.raw_rank for result in ranked}
    filtering_candidates = tuple(
        {
            "chunk_id": candidate["chunk_id"],
            "raw_rank": candidate["raw_rank"],
            "vector_score": candidate["vector_score"],
            "retained": candidate["raw_rank"] in retained_raw_ranks,
            "reason": (
                "vector_score_at_or_above_threshold"
                if candidate["raw_rank"] in retained_raw_ranks
                else "vector_score_below_threshold"
            ),
        }
        for candidate in raw_candidates
    )
    hybrid_ranking = tuple(
        {
            **_document_trace_identity(result.document),
            "vector_score": result.vector_score,
            "keyword_coverage": result.keyword_score,
            "combined_score": result.combined_score,
            "rank_before": result.raw_rank,
            "rank_after": rank,
        }
        for rank, result in enumerate(ranked, start=1)
    )
    seed_results = ranked[: config.retrieval_limit]
    seed_documents = [result.document for result in seed_results]
    seeds = tuple(
        {
            **_document_trace_identity(result.document),
            "seed_rank": rank,
            "rank_before": result.raw_rank,
            "combined_score": result.combined_score,
        }
        for rank, result in enumerate(seed_results, start=1)
    )

    context_started = perf_counter()
    final_documents = (
        expand_adjacent_documents(
            vector_store,
            seed_documents,
            adjacent_window=config.adjacent_window,
            context_limit=config.context_limit,
        )
        if config.adjacent_window
        else seed_documents
    )
    context_elapsed = round((perf_counter() - context_started) * 1000)
    seed_identities = [_document_trace_identity(document) for document in seed_documents]
    seed_ids = {identity["chunk_id"] for identity in seed_identities}
    expansion = []
    for order, document in enumerate(final_documents, start=1):
        identity = _document_trace_identity(document)
        if identity["chunk_id"] in seed_ids:
            source_seed = identity["chunk_id"]
            role = "seed"
            reason = "hybrid_top_n"
        else:
            matching_seed = next(
                (
                    seed
                    for seed in seed_identities
                    if seed["source"] == identity["source"]
                    and isinstance(seed["chunk_index"], int)
                    and isinstance(identity["chunk_index"], int)
                    and 0
                    < abs(seed["chunk_index"] - identity["chunk_index"])
                    <= config.adjacent_window
                ),
                None,
            )
            source_seed = matching_seed["chunk_id"] if matching_seed else None
            role = "adjacent"
            reason = (
                "same_source_adjacent_chunk"
                if matching_seed is not None
                else "adjacency_provenance_unresolved"
            )
        expansion.append(
            {
                **identity,
                "context_order": order,
                "role": role,
                "source_seed_chunk_id": source_seed,
                "adjacency_reason": reason,
            }
        )
    final_context = {
        "final_evidence_ids": [item["evidence_id"] for item in expansion],
        "final_chunk_ids": [item["chunk_id"] for item in expansion],
        "context_order": [item["chunk_id"] for item in expansion],
        "chunk_count": len(expansion),
        "source_diversity": len({item["source"] for item in expansion}),
        "material_diversity": len({item["material_id"] for item in expansion}),
        "char_size": sum(len(document.page_content) for document in final_documents),
        "token_size": None,
        "token_count_method": None,
    }
    total_elapsed = round((perf_counter() - total_started) * 1000)
    return RetrievalTrace(
        query={"case_id": case.case_id, "question": case.question},
        raw_candidates=raw_candidates,
        filtering={
            "threshold": config.relevance_threshold,
            "before_count": len(raw_candidates),
            "after_count": len(ranked),
            "retained_count": len(ranked),
            "filtered_count": len(raw_candidates) - len(ranked),
            "candidates": filtering_candidates,
        },
        hybrid_ranking=hybrid_ranking,
        seeds=seeds,
        adjacent_expansion=tuple(expansion),
        final_context=final_context,
        latency_ms={
            "raw_vector_search": raw_elapsed,
            "filter_and_hybrid_ranking": ranking_elapsed,
            "context_construction": context_elapsed,
            "total_retrieval": total_elapsed,
        },
    )


def recall_at_k(ranked_chunk_ids: Sequence[str], gold_chunk_ids: set[str], k: int) -> float:
    if k <= 0:
        raise ValueError("K 必须大于 0。")
    if not gold_chunk_ids:
        raise ValueError("Recall@K 只适用于至少一个 Gold Evidence 的案例。")
    retrieved = set(ranked_chunk_ids[:k])
    return len(retrieved & gold_chunk_ids) / len(gold_chunk_ids)


def reciprocal_rank(ranked_chunk_ids: Sequence[str], gold_chunk_ids: set[str]) -> float:
    if not gold_chunk_ids:
        raise ValueError("MRR 只适用于至少一个 Gold Evidence 的案例。")
    for rank, chunk_id in enumerate(ranked_chunk_ids, start=1):
        if chunk_id in gold_chunk_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_chunk_ids: Sequence[str], gold_chunk_ids: set[str], k: int) -> float:
    if k <= 0:
        raise ValueError("K 必须大于 0。")
    if not gold_chunk_ids:
        raise ValueError("nDCG 只适用于至少一个 Gold Evidence 的案例。")
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, chunk_id in enumerate(ranked_chunk_ids[:k], start=1)
        if chunk_id in gold_chunk_ids
    )
    ideal_hits = min(len(gold_chunk_ids), k)
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / ideal_dcg if ideal_dcg else 0.0


def context_precision(
    final_chunk_ids: Sequence[str],
    gold_chunk_ids: set[str],
    *,
    answerable: bool,
) -> float:
    """可回答题计算 Gold/Context；不可回答题以空 Context 为正确结果。"""
    if not answerable:
        return 1.0 if not final_chunk_ids else 0.0
    if not final_chunk_ids:
        return 0.0
    relevant = len(set(final_chunk_ids) & gold_chunk_ids)
    return relevant / len(final_chunk_ids)


def _case_metrics(case: RetrievalEvaluationCase, trace: RetrievalTrace) -> dict[str, object]:
    raw_ids = [str(item["chunk_id"]) for item in trace.raw_candidates]
    ranked_ids = [str(item["chunk_id"]) for item in trace.hybrid_ranking]
    final_ids = [str(item) for item in trace.final_context["final_chunk_ids"]]
    gold = set(case.gold_chunk_ids)
    metrics: dict[str, object] = {
        "gold_count": len(gold),
        "context_precision": context_precision(
            final_ids, gold, answerable=case.answerable
        ),
        "final_context_recall": (
            len(set(final_ids) & gold) / len(gold) if gold else None
        ),
    }
    for stage, ids in (("raw", raw_ids), ("ranked", ranked_ids)):
        for k in RECALL_K_VALUES:
            metrics[f"{stage}_recall_at_{k}"] = (
                recall_at_k(ids, gold, k) if gold else None
            )
        metrics[f"{stage}_mrr"] = reciprocal_rank(ids, gold) if gold else None
        metrics[f"{stage}_ndcg_at_5"] = ndcg_at_k(ids, gold, 5) if gold else None
    return metrics


def diagnose_retrieval_failure(
    case: RetrievalEvaluationCase,
    trace: RetrievalTrace,
    *,
    gold_mapping_resolved: bool = True,
) -> str | None:
    """只根据确定性 Pipeline 状态归因；不猜 Parser/Chunker 的隐含原因。"""
    if not gold_mapping_resolved:
        return FailureCategory.NEEDS_MANUAL_REVIEW
    final_ids = set(str(item) for item in trace.final_context["final_chunk_ids"])
    if not case.answerable:
        return (
            FailureCategory.UNANSWERABLE_HANDLING_FAILURE if final_ids else None
        )
    gold = set(case.gold_chunk_ids)
    raw_ids = {str(item["chunk_id"]) for item in trace.raw_candidates}
    retained_ids = {str(item["chunk_id"]) for item in trace.hybrid_ranking}
    seed_ids = {str(item["chunk_id"]) for item in trace.seeds}
    if not raw_ids & gold:
        return FailureCategory.RECALL_FAILURE
    if not retained_ids & gold:
        return FailureCategory.FILTERING_FAILURE
    if not seed_ids & gold:
        return FailureCategory.RANKING_FAILURE
    if not final_ids & gold:
        return FailureCategory.CONTEXT_CONSTRUCTION_FAILURE
    return None


def diagnose_end_to_end_failure(
    case: RetrievalEvaluationCase,
    trace: RetrievalTrace,
    *,
    parsing_failed: bool = False,
    chunking_failed: bool = False,
    generation_failed: bool = False,
    citation_valid: bool | None = None,
    citation_support: bool | None = None,
) -> str | None:
    """组合显式信号；Citation Support 未判断时绝不伪造自动结论。"""
    if parsing_failed:
        return FailureCategory.PARSING_FAILURE
    if chunking_failed:
        return FailureCategory.CHUNKING_FAILURE
    retrieval_failure = diagnose_retrieval_failure(case, trace)
    if retrieval_failure is not None:
        return retrieval_failure
    if generation_failed:
        return FailureCategory.GENERATION_FAILURE
    if citation_valid is False or citation_support is False:
        return FailureCategory.CITATION_FAILURE
    if citation_valid is True and citation_support is None:
        return FailureCategory.NEEDS_MANUAL_REVIEW
    return None


def evaluate_citation_validity(
    citation_ids: Sequence[str],
    evidence_map_ids: set[str],
) -> dict[str, object]:
    """只验证 Citation ID 是否存在；不把存在性冒充支持正确性。"""
    unique_ids = tuple(dict.fromkeys(citation_ids))
    valid = tuple(item for item in unique_ids if item in evidence_map_ids)
    invalid = tuple(item for item in unique_ids if item not in evidence_map_ids)
    return {
        "citation_count": len(unique_ids),
        "valid_count": len(valid),
        "valid_ids": list(valid),
        "invalid_ids": list(invalid),
        "validity_ratio": len(valid) / len(unique_ids) if unique_ids else None,
        "coverage": None,
        "support": None,
        "coverage_validation": "manual_or_future_judge_required",
        "support_validation": "manual_or_future_judge_required",
    }


def evaluate_retrieval_cases(
    cases: Sequence[RetrievalEvaluationCase],
    vector_store: object,
    config: RetrievalConfig,
) -> tuple[RetrievalCaseResult, ...]:
    results = []
    for case in cases:
        trace = trace_retrieval(case, vector_store, config)
        results.append(
            RetrievalCaseResult(
                case=case,
                trace=trace,
                metrics=_case_metrics(case, trace),
                failure_category=diagnose_retrieval_failure(case, trace),
            )
        )
    return tuple(results)


def _average(values: Sequence[object]) -> float | None:
    numeric = [float(value) for value in values if isinstance(value, (int, float))]
    return sum(numeric) / len(numeric) if numeric else None


def _dataset_metadata(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    return {
        "file": path.name,
        "evaluation_version": payload.get("evaluation_version"),
        "dataset_version": payload.get("dataset_version"),
        "sha256": sha256(raw).hexdigest(),
    }


def serialize_retrieval_case_result(
    result: RetrievalCaseResult,
) -> dict[str, object]:
    """把单个 Case 转成可独立持久化、可重建聚合的报告记录。"""
    return {
        "case_id": result.case.case_id,
        "answerable": result.case.answerable,
        "case_types": list(result.case.case_types),
        "metrics": result.metrics,
        "failure_category": result.failure_category,
        "failure_notes": list(result.failure_notes),
        "trace": asdict(result.trace),
    }


def rebuild_retrieval_report_aggregates(
    report: Mapping[str, object],
) -> dict[str, object]:
    """只依赖已持久化的 per-case records 重建聚合指标、失败与耗时。"""
    rebuilt = dict(report)
    raw_results = rebuilt.get("per_case_results")
    if not isinstance(raw_results, list):
        raise RetrievalEvaluationReportError("Retrieval Report 缺少 per_case_results。")

    results: list[dict[str, object]] = []
    seen_case_ids = set()
    for raw_result in raw_results:
        if not isinstance(raw_result, dict):
            raise RetrievalEvaluationReportError("per_case_results 包含无效记录。")
        case_id = raw_result.get("case_id")
        metrics = raw_result.get("metrics")
        trace = raw_result.get("trace")
        latency = trace.get("latency_ms") if isinstance(trace, dict) else None
        total_latency = latency.get("total_retrieval") if isinstance(latency, dict) else None
        if not isinstance(case_id, str) or not case_id:
            raise RetrievalEvaluationReportError("per_case_results 包含无效 case_id。")
        if case_id in seen_case_ids:
            raise RetrievalEvaluationReportError(f"per_case_results 包含重复案例：{case_id}。")
        if not isinstance(metrics, dict):
            raise RetrievalEvaluationReportError(f"案例 {case_id} 缺少 metrics。")
        if (
            isinstance(total_latency, bool)
            or not isinstance(total_latency, (int, float))
            or total_latency < 0
        ):
            raise RetrievalEvaluationReportError(
                f"案例 {case_id} 缺少有效 total_retrieval latency。"
            )
        seen_case_ids.add(case_id)
        results.append(raw_result)

    metric_names = sorted(
        {
            name
            for result in results
            for name in result["metrics"]
            if name != "gold_count"
        }
    )
    rebuilt["aggregate_metrics"] = {
        name: _average([result["metrics"].get(name) for result in results])
        for name in metric_names
    }
    failure_counts = Counter(
        result.get("failure_category")
        for result in results
        if isinstance(result.get("failure_category"), str)
    )
    rebuilt["failure_categories"] = {
        "counts": dict(sorted(failure_counts.items())),
        "supported": [category.value for category in FailureCategory],
    }
    total_latencies = [
        result["trace"]["latency_ms"]["total_retrieval"] for result in results
    ]
    rebuilt["latency"] = {
        "total_retrieval_ms": sum(total_latencies),
        "average_retrieval_ms": _average(total_latencies),
        "measurement": "local_wall_clock",
    }
    return rebuilt


def build_retrieval_report(
    results: Sequence[RetrievalCaseResult],
    *,
    dataset_path: Path,
    config: RetrievalConfig,
    git_commit: str,
    stage2_start_commit: str,
    generated_at: datetime,
    validation_level: str,
    embedding_model: str | None,
    query_embedding_calls: int,
) -> dict[str, object]:
    if generated_at.tzinfo is None:
        raise RetrievalEvaluationReportError("generated_at 必须包含时区。")
    if not GIT_COMMIT_PATTERN.fullmatch(git_commit):
        raise RetrievalEvaluationReportError("git_commit 必须是完整 commit hash。")
    if not GIT_COMMIT_PATTERN.fullmatch(stage2_start_commit):
        raise RetrievalEvaluationReportError("stage2_start_commit 必须是完整 commit hash。")
    dataset_metadata = _dataset_metadata(dataset_path)
    report = {
        "report_schema_version": RETRIEVAL_REPORT_SCHEMA_VERSION,
        "evaluation_type": "retrieval",
        "evaluation_version": RETRIEVAL_EVALUATION_VERSION,
        "dataset_version": dataset_metadata["dataset_version"],
        "evaluation_dataset": dataset_metadata,
        "git_commit": git_commit,
        "stage2_start_commit": stage2_start_commit,
        "retrieval_config": asdict(config),
        "aggregate_metrics": {},
        "per_case_results": [serialize_retrieval_case_result(result) for result in results],
        "failure_categories": {},
        "latency": {},
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
        "tokens": {"measured": False, "total": None},
        "cost": {
            "measured": False,
            "amount": None,
            "currency": None,
            "query_embedding_calls": query_embedding_calls,
        },
        "citation_evaluation": {
            "validity": "deterministic_id_membership_supported",
            "coverage": "manual_or_future_judge_required",
            "support": "manual_or_future_judge_required",
            "warning": "valid_citation_id_does_not_prove_evidence_support",
        },
        "generated_at": generated_at.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        "validation_level": validation_level,
    }
    return rebuild_retrieval_report_aggregates(report)


def write_retrieval_report(
    report: Mapping[str, object],
    output_directory: Path = DEFAULT_RETRIEVAL_RESULTS_DIR,
) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    path = output_directory / f"retrieval-evaluation-{timestamp}-{uuid4().hex[:8]}.json"
    created = False
    try:
        output_directory.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8", newline="\n") as output:
            created = True
            output.write(json.dumps(dict(report), ensure_ascii=False, indent=2) + "\n")
            output.flush()
            os.fsync(output.fileno())
    except (OSError, TypeError, ValueError) as error:
        if created:
            try:
                path.unlink()
            except OSError:
                pass
        raise RetrievalEvaluationReportError(
            f"无法保存 Retrieval Report 到 {output_directory}。"
        ) from error
    return path


def load_persisted_retrieval_report(path: Path) -> dict[str, object]:
    """读取 crash-safe Retrieval run state。"""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RetrievalEvaluationReportError(f"无法读取 Retrieval run state：{path}。") from error
    if not isinstance(payload, dict):
        raise RetrievalEvaluationReportError("Retrieval run state 根节点必须是对象。")
    return payload


def _atomic_replace_report(path: Path, report: Mapping[str, object]) -> None:
    temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary_path.open("x", encoding="utf-8", newline="\n") as output:
            output.write(json.dumps(dict(report), ensure_ascii=False, indent=2) + "\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    except (OSError, TypeError, ValueError) as error:
        try:
            temporary_path.unlink()
        except OSError:
            pass
        raise RetrievalEvaluationReportError(
            f"无法原子更新 Retrieval run state：{path}。"
        ) from error


@dataclass(frozen=True)
class RetrievalEvaluationRun:
    """一个轻量 JSON journal；每完成一个 Case 就原子替换持久化状态。"""

    path: Path

    @classmethod
    def create(
        cls,
        *,
        dataset_path: Path,
        output_directory: Path,
        config: RetrievalConfig,
        git_commit: str,
        stage2_start_commit: str,
        started_at: datetime,
        validation_level: str,
        embedding_model: str | None,
        original_index_fingerprint: str,
    ) -> "RetrievalEvaluationRun":
        report = build_retrieval_report(
            (),
            dataset_path=dataset_path,
            config=config,
            git_commit=git_commit,
            stage2_start_commit=stage2_start_commit,
            generated_at=started_at,
            validation_level=validation_level,
            embedding_model=embedding_model,
            query_embedding_calls=0,
        )
        timestamp = started_at.astimezone(UTC).isoformat().replace("+00:00", "Z")
        report.update(
            {
                "run_status": "ready",
                "started_at": timestamp,
                "last_persisted_at": timestamp,
                "completed_at": None,
                "index_isolation": {
                    "mode": "disposable_snapshot",
                    "original_index_fingerprint_sha256": original_index_fingerprint,
                    "original_index_unchanged": None,
                },
            }
        )
        return cls(write_retrieval_report(report, output_directory))

    def persist_case_result(
        self,
        result: RetrievalCaseResult,
        *,
        persisted_at: datetime | None = None,
    ) -> dict[str, object]:
        report = load_persisted_retrieval_report(self.path)
        raw_results = report.get("per_case_results")
        if not isinstance(raw_results, list):
            raise RetrievalEvaluationReportError("Retrieval run state 缺少 per_case_results。")
        if any(
            isinstance(item, dict) and item.get("case_id") == result.case.case_id
            for item in raw_results
        ):
            raise RetrievalEvaluationReportError(
                f"Retrieval run state 已包含案例：{result.case.case_id}。"
            )
        raw_results.append(serialize_retrieval_case_result(result))
        report["per_case_results"] = raw_results
        report = rebuild_retrieval_report_aggregates(report)
        cost = report.get("cost")
        if not isinstance(cost, dict):
            raise RetrievalEvaluationReportError("Retrieval run state 缺少 cost metadata。")
        cost["query_embedding_calls"] = len(raw_results)
        report["cost"] = cost
        report["run_status"] = "running"
        now = persisted_at or datetime.now(UTC)
        report["last_persisted_at"] = now.astimezone(UTC).isoformat().replace(
            "+00:00", "Z"
        )
        _atomic_replace_report(self.path, report)
        return report

    def finalize(
        self,
        *,
        original_index_unchanged: bool,
        completed_at: datetime | None = None,
    ) -> dict[str, object]:
        if not original_index_unchanged:
            raise RetrievalEvaluationReportError(
                "原始 Index 字节指纹发生变化，拒绝完成 Retrieval Report。"
            )
        report = rebuild_retrieval_report_aggregates(
            load_persisted_retrieval_report(self.path)
        )
        now = completed_at or datetime.now(UTC)
        timestamp = now.astimezone(UTC).isoformat().replace("+00:00", "Z")
        isolation = report.get("index_isolation")
        if not isinstance(isolation, dict):
            raise RetrievalEvaluationReportError("Retrieval run state 缺少 index_isolation。")
        isolation["original_index_unchanged"] = True
        report["index_isolation"] = isolation
        report["run_status"] = "completed"
        report["completed_at"] = timestamp
        report["last_persisted_at"] = timestamp
        report["generated_at"] = timestamp
        _atomic_replace_report(self.path, report)
        return report
