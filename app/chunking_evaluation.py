"""Stage 3.3：固定 Retrieval，仅比较 Current 与 Structure-aware Chunking。"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from langchain_core.documents import Document

if __package__:
    from app.chunk_documents import DocumentChunk
    from app.evidence import evidence_from_document
    from app.langchain_store import chunks_to_documents
    from app.retrieval_evaluation import (
        CurrentChunkMapping,
        RetrievalCaseResult,
        RetrievalConfig,
        RetrievalEvaluationCase,
        StableGoldMeaning,
    )
else:
    from chunk_documents import DocumentChunk
    from evidence import evidence_from_document
    from langchain_store import chunks_to_documents
    from retrieval_evaluation import (
        CurrentChunkMapping,
        RetrievalCaseResult,
        RetrievalConfig,
        RetrievalEvaluationCase,
        StableGoldMeaning,
    )


CHUNKING_REPORT_SCHEMA_VERSION = 1
CHUNKING_EXPERIMENT_NAME = "stage3_3_structure_aware_chunking"
TRACKED_METRICS = (
    "raw_recall_at_1",
    "raw_recall_at_3",
    "raw_recall_at_5",
    "raw_recall_at_10",
    "ranked_recall_at_1",
    "ranked_recall_at_3",
    "ranked_recall_at_5",
    "ranked_recall_at_10",
    "ranked_mrr",
    "ranked_ndcg_at_5",
    "context_precision",
    "final_context_recall",
)


class ChunkingMappingError(RuntimeError):
    """Stable Gold 无法确定性映射到候选 Chunk。"""


@dataclass(frozen=True)
class ChunkingStrategyDescriptor:
    name: str
    chunker_version: str
    max_chars: int
    overlap_chars: int
    structure_order: tuple[str, ...]


def _normalized_match_text(value: str) -> str:
    return " ".join(value.split())


def _document_matches_meaning(document: Document, meaning: StableGoldMeaning) -> bool:
    metadata = document.metadata
    if metadata.get("material_id") != meaning.material_id:
        return False
    if metadata.get("source") != meaning.source:
        return False
    if meaning.page is not None and metadata.get("page") != meaning.page:
        return False
    if (
        meaning.section is not None
        and metadata.get("section") is not None
        and metadata.get("section") != meaning.section
    ):
        return False
    needle = _normalized_match_text(meaning.text_span)
    return bool(needle) and needle in _normalized_match_text(document.page_content)


def _mapping_for_document(
    document: Document,
    gold_ids: Sequence[str],
) -> CurrentChunkMapping:
    evidence = evidence_from_document(document, "S1")
    return CurrentChunkMapping(
        gold_ids=tuple(sorted(gold_ids)),
        chunk_id=evidence.chunk_id,
        material_id=evidence.material_id,
        content_hash=evidence.content_hash,
        source=evidence.source,
        page=evidence.page,
        section=evidence.section,
        legacy_chunk_index=evidence.chunk_index,
    )


def remap_cases_to_chunks(
    cases: Sequence[RetrievalEvaluationCase],
    chunks: Sequence[DocumentChunk],
    *,
    baseline_chunks: Sequence[DocumentChunk],
) -> tuple[RetrievalEvaluationCase, ...]:
    """优先精确 Stable Gold；否则投影已人工确认的当前 Gold Chunk。"""
    documents = chunks_to_documents(list(chunks))
    baseline_documents = chunks_to_documents(list(baseline_chunks))
    baseline_by_id = {
        evidence_from_document(document, "S1").chunk_id: document
        for document in baseline_documents
    }
    remapped: list[RetrievalEvaluationCase] = []
    for case in cases:
        if not case.answerable:
            remapped.append(replace(case, current_chunk_mappings=()))
            continue
        gold_by_chunk: dict[str, tuple[Document, list[str]]] = {}
        for meaning in case.stable_gold_meanings:
            matches = [
                document
                for document in documents
                if _document_matches_meaning(document, meaning)
            ]
            if not matches:
                current_mappings = [
                    mapping
                    for mapping in case.current_chunk_mappings
                    if meaning.gold_id in mapping.gold_ids
                ]
                if len(current_mappings) != 1:
                    raise ChunkingMappingError(
                        f"案例 {case.case_id} 的 Stable Gold {meaning.gold_id} "
                        "缺少唯一 Current Chunk Mapping。"
                    )
                current_mapping = current_mappings[0]
                baseline_document = baseline_by_id.get(current_mapping.chunk_id)
                if baseline_document is None:
                    raise ChunkingMappingError(
                        f"案例 {case.case_id} 的 Current Chunk Mapping "
                        f"无法定位基线 Chunk {current_mapping.chunk_id}。"
                    )
                baseline_text = _normalized_match_text(
                    baseline_document.page_content
                )
                matches = [
                    document
                    for document in documents
                    if document.metadata.get("material_id") == meaning.material_id
                    and document.metadata.get("source") == meaning.source
                    and (
                        current_mapping.page is None
                        or document.metadata.get("page") == current_mapping.page
                    )
                    and baseline_text
                    in _normalized_match_text(document.page_content)
                ]
                if not matches:
                    raise ChunkingMappingError(
                        f"案例 {case.case_id} 的 Stable Gold {meaning.gold_id} "
                        "既没有精确文本定位，也无法完整投影当前 Gold Chunk。"
                    )
            if len(matches) > 1:
                raise ChunkingMappingError(
                    f"案例 {case.case_id} 的 Stable Gold {meaning.gold_id} "
                    "映射到多个候选 Chunk，无法确定性评测。"
                )
            document = matches[0]
            chunk_id = evidence_from_document(document, "S1").chunk_id
            if chunk_id not in gold_by_chunk:
                gold_by_chunk[chunk_id] = (document, [])
            gold_by_chunk[chunk_id][1].append(meaning.gold_id)
        mappings = tuple(
            _mapping_for_document(document, gold_ids)
            for _, (document, gold_ids) in sorted(gold_by_chunk.items())
        )
        remapped.append(replace(case, current_chunk_mappings=mappings))
    return tuple(remapped)


def chunk_statistics(chunks: Sequence[DocumentChunk]) -> dict[str, object]:
    lengths = sorted(len(chunk.content) for chunk in chunks)
    if not lengths:
        return {
            "chunk_count": 0,
            "min_chars": None,
            "median_chars": None,
            "p95_chars": None,
            "max_chars": None,
            "average_chars": None,
            "chunks_by_source_type": {},
        }
    middle = len(lengths) // 2
    median = (
        float(lengths[middle])
        if len(lengths) % 2
        else (lengths[middle - 1] + lengths[middle]) / 2
    )
    return {
        "chunk_count": len(chunks),
        "min_chars": lengths[0],
        "median_chars": median,
        "p95_chars": lengths[max(0, math.ceil(len(lengths) * 0.95) - 1)],
        "max_chars": lengths[-1],
        "average_chars": sum(lengths) / len(lengths),
        "chunks_by_source_type": dict(
            sorted(Counter(chunk.source_type for chunk in chunks).items())
        ),
    }


def citation_localization_summary(
    cases: Sequence[RetrievalEvaluationCase],
    chunks: Sequence[DocumentChunk],
) -> dict[str, object]:
    documents = chunks_to_documents(list(chunks))
    by_id = {
        evidence_from_document(document, "S1").chunk_id: document
        for document in documents
    }
    meanings = {
        meaning.gold_id: meaning
        for case in cases
        for meaning in case.stable_gold_meanings
    }
    locator_values: list[str] = []
    material_matches = 0
    source_matches = 0
    page_checks = 0
    page_matches = 0
    section_checks = 0
    section_matches = 0
    section_available = 0
    mapping_count = 0
    for case in cases:
        for mapping in case.current_chunk_mappings:
            document = by_id.get(mapping.chunk_id)
            if document is None:
                raise ChunkingMappingError(
                    f"案例 {case.case_id} 的 Mapping 无法定位 Chunk {mapping.chunk_id}。"
                )
            evidence = evidence_from_document(document, "S1")
            mapping_count += 1
            locator_values.append(evidence.locator)
            related = [meanings[gold_id] for gold_id in mapping.gold_ids]
            if all(evidence.material_id == meaning.material_id for meaning in related):
                material_matches += 1
            if all(evidence.source == meaning.source for meaning in related):
                source_matches += 1
            for meaning in related:
                if meaning.page is not None:
                    page_checks += 1
                    page_matches += int(evidence.page == meaning.page)
                if meaning.section is not None:
                    section_checks += 1
                    if evidence.section is not None:
                        section_available += 1
                        section_matches += int(evidence.section == meaning.section)
    return {
        "mapping_count": mapping_count,
        "all_mappings_resolved": True,
        "material_match_ratio": material_matches / mapping_count if mapping_count else None,
        "source_match_ratio": source_matches / mapping_count if mapping_count else None,
        "page_match_ratio": page_matches / page_checks if page_checks else None,
        "section_match_ratio": section_matches / section_checks if section_checks else None,
        "section_available_ratio": (
            section_available / section_checks if section_checks else None
        ),
        "locator_count": len(locator_values),
        "unique_locator_count": len(set(locator_values)),
        "locators_unique": len(locator_values) == len(set(locator_values)),
        "citation_support_validated": False,
    }


def _average_metrics(results: Sequence[RetrievalCaseResult]) -> dict[str, float | None]:
    output: dict[str, float | None] = {}
    for name in TRACKED_METRICS:
        values = [
            float(result.metrics[name])
            for result in results
            if isinstance(result.metrics.get(name), (int, float))
            and not isinstance(result.metrics.get(name), bool)
        ]
        output[name] = sum(values) / len(values) if values else None
    return output


def _metric_deltas(
    baseline: Mapping[str, float | None],
    experiment: Mapping[str, float | None],
) -> dict[str, float | None]:
    return {
        name: (
            float(experiment[name]) - float(baseline[name])
            if isinstance(experiment.get(name), (int, float))
            and not isinstance(experiment.get(name), bool)
            and isinstance(baseline.get(name), (int, float))
            and not isinstance(baseline.get(name), bool)
            else None
        )
        for name in TRACKED_METRICS
    }


def _failure_counts(results: Sequence[RetrievalCaseResult]) -> dict[str, int]:
    return dict(
        sorted(
            Counter(
                result.failure_category
                for result in results
                if result.failure_category is not None
            ).items()
        )
    )


def _context_sizes(results: Sequence[RetrievalCaseResult]) -> dict[str, float | int | None]:
    sizes = [int(result.trace.final_context["chunk_count"]) for result in results]
    return {
        "average": sum(sizes) / len(sizes) if sizes else None,
        "min": min(sizes) if sizes else None,
        "max": max(sizes) if sizes else None,
    }


def _latency_summary(
    results: Sequence[RetrievalCaseResult],
) -> dict[str, float | None]:
    fields = (
        "raw_vector_search",
        "filter_and_hybrid_ranking",
        "context_construction",
        "total_retrieval",
    )
    return {
        field: (
            sum(float(result.trace.latency_ms[field]) for result in results)
            / len(results)
            if results
            else None
        )
        for field in fields
    }


def build_chunking_report(
    baseline_results: Sequence[RetrievalCaseResult],
    experimental_results: Sequence[RetrievalCaseResult],
    *,
    baseline_descriptor: ChunkingStrategyDescriptor,
    experimental_descriptor: ChunkingStrategyDescriptor,
    retrieval_config: RetrievalConfig,
    baseline_chunks: Sequence[DocumentChunk],
    experimental_chunks: Sequence[DocumentChunk],
    baseline_localization: Mapping[str, object],
    experimental_localization: Mapping[str, object],
    deterministic_rebuild: bool,
    git_commit: str,
    stage3_3_start_commit: str,
    dataset_metadata: Mapping[str, object],
    embedding_model: str | None,
    query_embedding_calls: int,
    baseline_index_embedding_texts: int,
    baseline_index_embedding_batches: int,
    experimental_index_embedding_texts: int,
    experimental_index_embedding_batches: int,
    production_index_fingerprint: str,
    baseline_index_fingerprint: str,
    experimental_index_fingerprint: str,
    generated_at: datetime,
    validation_level: str,
) -> dict[str, object]:
    if generated_at.tzinfo is None:
        raise ValueError("generated_at 必须包含时区。")
    baseline_metrics = _average_metrics(baseline_results)
    experimental_metrics = _average_metrics(experimental_results)
    return {
        "report_schema_version": CHUNKING_REPORT_SCHEMA_VERSION,
        "experiment_name": CHUNKING_EXPERIMENT_NAME,
        "git_commit": git_commit,
        "stage3_3_start_commit": stage3_3_start_commit,
        "controlled_variable": "chunking_strategy",
        "frozen_variables": {
            "embedding_model": embedding_model,
            "retrieval_config": asdict(retrieval_config),
            "retrieval_algorithm": "stage2_dense_keyword_coverage_baseline",
            "bm25": "unchanged_not_used_by_production_baseline",
            "rrf": "unchanged_not_used_by_production_baseline",
            "reranker": "unchanged_not_used_by_production_baseline",
            "query_rewrite": False,
            "context_construction": "top3_same_source_adjacent_window2_context_limit8",
        },
        "strategies": {
            "baseline": asdict(baseline_descriptor),
            "structure_aware": asdict(experimental_descriptor),
        },
        "evaluation_dataset": dict(dataset_metadata),
        "gold_mapping_contract": (
            "exact_stable_span_else_full_containment_of_current_gold_chunk"
        ),
        "chunk_statistics": {
            "baseline": chunk_statistics(baseline_chunks),
            "structure_aware": chunk_statistics(experimental_chunks),
        },
        "metrics": {
            "baseline": baseline_metrics,
            "structure_aware": experimental_metrics,
        },
        "metric_deltas": {
            "structure_aware_vs_baseline": _metric_deltas(
                baseline_metrics,
                experimental_metrics,
            )
        },
        "context_size": {
            "baseline": _context_sizes(baseline_results),
            "structure_aware": _context_sizes(experimental_results),
        },
        "average_latency_ms": {
            "baseline": _latency_summary(baseline_results),
            "structure_aware": _latency_summary(experimental_results),
        },
        "failure_counts": {
            "baseline": _failure_counts(baseline_results),
            "structure_aware": _failure_counts(experimental_results),
        },
        "citation_localization": {
            "baseline": dict(baseline_localization),
            "structure_aware": dict(experimental_localization),
            "deterministic_rebuild": deterministic_rebuild,
            "citation_support_validated": False,
        },
        "index_isolation": {
            "production_index_fingerprint_sha256": production_index_fingerprint,
            "baseline_temporary_index_fingerprint_sha256": baseline_index_fingerprint,
            "structure_aware_index_fingerprint_sha256": experimental_index_fingerprint,
            "production_index_mutated": False,
        },
        "cost": {
            "measured": False,
            "amount": None,
            "currency": None,
            "query_embedding_calls": query_embedding_calls,
            "index_embedding_texts": {
                "baseline": baseline_index_embedding_texts,
                "structure_aware": experimental_index_embedding_texts,
                "total": (
                    baseline_index_embedding_texts
                    + experimental_index_embedding_texts
                ),
            },
            "index_embedding_batches": {
                "baseline": baseline_index_embedding_batches,
                "structure_aware": experimental_index_embedding_batches,
                "total": (
                    baseline_index_embedding_batches
                    + experimental_index_embedding_batches
                ),
            },
            "chat_calls": 0,
        },
        "validation_level": validation_level,
        "per_case_results": {
            "baseline": [
                {
                    "case_id": result.case.case_id,
                    "metrics": result.metrics,
                    "failure_category": result.failure_category,
                }
                for result in baseline_results
            ],
            "structure_aware": [
                {
                    "case_id": result.case.case_id,
                    "metrics": result.metrics,
                    "failure_category": result.failure_category,
                }
                for result in experimental_results
            ],
        },
        "run_status": "completed",
        "generated_at": generated_at.astimezone(UTC).isoformat().replace("+00:00", "Z"),
    }


def write_chunking_report(
    report: Mapping[str, object],
    output_directory: Path,
) -> Path:
    output_directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    destination = output_directory / f"chunking-{timestamp}-{uuid4().hex[:8]}.json"
    with destination.open("x", encoding="utf-8", newline="\n") as output:
        output.write(json.dumps(dict(report), ensure_ascii=False, indent=2) + "\n")
    return destination
