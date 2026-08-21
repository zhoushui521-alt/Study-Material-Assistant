"""提供旧检索基线，以及 BM25 + Dense Vector + RRF 双路检索。"""

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

from langchain_chroma import Chroma
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

if __package__:
    from app.evidence import evidence_from_document
    from app.langchain_store import search_vector_store
    from app.search_documents import extract_terms
else:
    from evidence import evidence_from_document
    from langchain_store import search_vector_store
    from search_documents import extract_terms


DEFAULT_CANDIDATE_LIMIT = 10
DEFAULT_KEYWORD_WEIGHT = 0.2
DEFAULT_RELEVANCE_THRESHOLD = 0.25
DEFAULT_BM25_K1 = 1.5
DEFAULT_BM25_B = 0.75
DEFAULT_RRF_K = 60
CONTEXT_ROLE_METADATA_KEY = "_context_role"
CONTEXT_SEED_RANK_METADATA_KEY = "_context_seed_rank"


@dataclass(frozen=True)
class HybridSearchResult:
    """一条包含向量分、关键词覆盖率和混合分的检索结果。"""

    document: Document
    raw_rank: int
    vector_score: float
    keyword_score: float
    combined_score: float


@dataclass(frozen=True)
class BM25SearchResult:
    """一条稀疏检索结果；rank 从 1 开始。"""

    document: Document
    chunk_id: str
    score: float
    rank: int


@dataclass(frozen=True)
class RRFFusionResult:
    """一条只按两路排名融合、不混合原始分数的结果。"""

    document: Document
    chunk_id: str
    rrf_score: float
    dense_rank: int | None
    bm25_rank: int | None
    vector_score: float | None
    bm25_score: float | None


def _chunk_id(document: Document) -> str:
    """复用 Stage 1 身份规则，兼容 legacy read-only 索引。"""
    return evidence_from_document(document, "S1").chunk_id


def tokenize_for_bm25(text: str) -> list[str]:
    """提取英文技术词和连续中文双字词，保留词频供 BM25 使用。"""
    normalized = text.strip().casefold()
    if not normalized:
        return []
    english_terms = re.findall(r"[a-z0-9_+#.-]+", normalized)
    chinese_groups = re.findall(r"[\u4e00-\u9fff]+", normalized)
    chinese_terms = [
        group[index : index + 2]
        for group in chinese_groups
        for index in range(len(group) - 1)
    ]
    return english_terms + chinese_terms


@dataclass(frozen=True)
class BM25Index:
    """面向当前本地 Chunk Corpus 的只读 BM25 索引。"""

    documents: tuple[Document, ...]
    chunk_ids: tuple[str, ...]
    term_frequencies: tuple[Counter[str], ...]
    document_lengths: tuple[int, ...]
    document_frequencies: dict[str, int]
    average_document_length: float
    k1: float = DEFAULT_BM25_K1
    b: float = DEFAULT_BM25_B

    @classmethod
    def from_documents(
        cls,
        documents: Sequence[Document],
        *,
        k1: float = DEFAULT_BM25_K1,
        b: float = DEFAULT_BM25_B,
    ) -> "BM25Index":
        if k1 <= 0:
            raise ValueError("BM25 k1 必须大于 0。")
        if not 0.0 <= b <= 1.0:
            raise ValueError("BM25 b 必须在 0 到 1 之间。")

        unique_documents: dict[str, Document] = {}
        for document in documents:
            chunk_id = _chunk_id(document)
            existing = unique_documents.get(chunk_id)
            if existing is not None and (
                existing.page_content != document.page_content
                or existing.metadata != document.metadata
            ):
                raise ValueError(f"chunk_id {chunk_id} 对应了不一致的重复文档。")
            unique_documents.setdefault(chunk_id, document)

        ordered = tuple(sorted(unique_documents.items(), key=lambda item: item[0]))
        chunk_ids = tuple(chunk_id for chunk_id, _ in ordered)
        indexed_documents = tuple(document for _, document in ordered)
        term_frequencies = tuple(
            Counter(tokenize_for_bm25(document.page_content))
            for document in indexed_documents
        )
        document_lengths = tuple(sum(frequencies.values()) for frequencies in term_frequencies)
        document_frequencies: Counter[str] = Counter()
        for frequencies in term_frequencies:
            document_frequencies.update(frequencies.keys())
        average_document_length = (
            sum(document_lengths) / len(document_lengths) if document_lengths else 0.0
        )
        return cls(
            documents=indexed_documents,
            chunk_ids=chunk_ids,
            term_frequencies=term_frequencies,
            document_lengths=document_lengths,
            document_frequencies=dict(document_frequencies),
            average_document_length=average_document_length,
            k1=k1,
            b=b,
        )

    def search(self, query: str, *, limit: int) -> list[BM25SearchResult]:
        if limit <= 0:
            raise ValueError("BM25 检索数量必须大于 0。")
        query_terms = tuple(dict.fromkeys(tokenize_for_bm25(query)))
        if not query_terms or not self.documents or self.average_document_length == 0:
            return []

        document_count = len(self.documents)
        scored: list[tuple[float, str, Document]] = []
        for document, chunk_id, frequencies, document_length in zip(
            self.documents,
            self.chunk_ids,
            self.term_frequencies,
            self.document_lengths,
            strict=True,
        ):
            score = 0.0
            for term in query_terms:
                term_frequency = frequencies.get(term, 0)
                if term_frequency == 0:
                    continue
                document_frequency = self.document_frequencies.get(term, 0)
                inverse_document_frequency = math.log(
                    1.0
                    + (document_count - document_frequency + 0.5)
                    / (document_frequency + 0.5)
                )
                length_normalization = self.k1 * (
                    1.0
                    - self.b
                    + self.b * document_length / self.average_document_length
                )
                score += inverse_document_frequency * (
                    term_frequency * (self.k1 + 1.0)
                    / (term_frequency + length_normalization)
                )
            if score > 0.0:
                scored.append((score, chunk_id, document))

        ranked = sorted(scored, key=lambda item: (-item[0], item[1]))[:limit]
        return [
            BM25SearchResult(document, chunk_id, score, rank)
            for rank, (score, chunk_id, document) in enumerate(ranked, start=1)
        ]


def build_bm25_index(vector_store: Chroma) -> BM25Index:
    """从完整 Chroma Corpus 构建稀疏索引，不触发 Embedding。"""
    stored = vector_store.get(include=["documents", "metadatas"])
    documents = [
        Document(page_content=content, metadata=metadata)
        for content, metadata in zip(
            stored.get("documents") or [],
            stored.get("metadatas") or [],
            strict=True,
        )
        if isinstance(content, str) and isinstance(metadata, dict)
    ]
    return BM25Index.from_documents(documents)


def reciprocal_rank_fusion(
    dense_results: Sequence[tuple[Document, float]],
    bm25_results: Sequence[BM25SearchResult],
    *,
    rrf_k: int = DEFAULT_RRF_K,
) -> list[RRFFusionResult]:
    """用 chunk_id 去重并融合两路排名；不比较 BM25 与 Dense 原始分数。"""
    if rrf_k <= 0:
        raise ValueError("RRF k 必须大于 0。")

    fused: dict[str, dict[str, object]] = {}
    for dense_rank, (document, vector_score) in enumerate(dense_results, start=1):
        chunk_id = _chunk_id(document)
        if chunk_id in fused and fused[chunk_id]["dense_rank"] is not None:
            continue
        entry = fused.setdefault(
            chunk_id,
            {
                "document": document,
                "dense_rank": None,
                "bm25_rank": None,
                "vector_score": None,
                "bm25_score": None,
            },
        )
        entry["dense_rank"] = dense_rank
        entry["vector_score"] = vector_score

    for result in bm25_results:
        entry = fused.setdefault(
            result.chunk_id,
            {
                "document": result.document,
                "dense_rank": None,
                "bm25_rank": None,
                "vector_score": None,
                "bm25_score": None,
            },
        )
        if entry["bm25_rank"] is None:
            entry["bm25_rank"] = result.rank
            entry["bm25_score"] = result.score

    results = []
    for chunk_id, entry in fused.items():
        dense_rank = entry["dense_rank"]
        bm25_rank = entry["bm25_rank"]
        rrf_score = sum(
            1.0 / (rrf_k + rank)
            for rank in (dense_rank, bm25_rank)
            if isinstance(rank, int)
        )
        results.append(
            RRFFusionResult(
                document=entry["document"],
                chunk_id=chunk_id,
                rrf_score=rrf_score,
                dense_rank=dense_rank if isinstance(dense_rank, int) else None,
                bm25_rank=bm25_rank if isinstance(bm25_rank, int) else None,
                vector_score=(
                    float(entry["vector_score"])
                    if isinstance(entry["vector_score"], (float, int))
                    else None
                ),
                bm25_score=(
                    float(entry["bm25_score"])
                    if isinstance(entry["bm25_score"], (float, int))
                    else None
                ),
            )
        )
    return sorted(results, key=lambda result: (-result.rrf_score, result.chunk_id))


def keyword_coverage(question: str, text: str) -> float:
    """计算问题词项中有多少比例出现在候选文本里。"""
    terms = extract_terms(question)
    if not terms:
        return 0.0

    normalized_text = text.casefold()
    matched_count = sum(term in normalized_text for term in terms)
    return matched_count / len(terms)


def rank_vector_candidates(
    question: str,
    vector_results: list[tuple[Document, float]],
    *,
    relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
    keyword_weight: float = DEFAULT_KEYWORD_WEIGHT,
) -> list[HybridSearchResult]:
    """按当前生产公式过滤并重排已经召回的向量候选。"""
    if not 0.0 <= relevance_threshold <= 1.0:
        raise ValueError("相关度阈值必须在 0 到 1 之间。")
    if not 0.0 <= keyword_weight <= 1.0:
        raise ValueError("关键词权重必须在 0 到 1 之间。")

    hybrid_results = []
    for raw_rank, (document, vector_score) in enumerate(vector_results, start=1):
        if vector_score < relevance_threshold:
            continue
        keyword_score = keyword_coverage(question, document.page_content)
        combined_score = (
            (1.0 - keyword_weight) * vector_score
            + keyword_weight * keyword_score
        )
        hybrid_results.append(
            HybridSearchResult(
                document=document,
                raw_rank=raw_rank,
                vector_score=vector_score,
                keyword_score=keyword_score,
                combined_score=combined_score,
            )
        )

    return sorted(
        hybrid_results,
        key=lambda result: (result.combined_score, result.vector_score),
        reverse=True,
    )


def hybrid_search_vector_store(
    question: str,
    vector_store: Chroma,
    limit: int = 3,
    candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
    relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
    keyword_weight: float = DEFAULT_KEYWORD_WEIGHT,
) -> list[HybridSearchResult]:
    """先用 Chroma 召回候选，再结合关键词覆盖率重新排序。"""
    if not question.strip():
        return []
    if limit <= 0 or candidate_limit <= 0:
        raise ValueError("检索数量必须大于 0。")
    if not 0.0 <= relevance_threshold <= 1.0:
        raise ValueError("相关度阈值必须在 0 到 1 之间。")
    if not 0.0 <= keyword_weight <= 1.0:
        raise ValueError("关键词权重必须在 0 到 1 之间。")

    vector_results = search_vector_store(
        question,
        vector_store,
        limit=max(limit, candidate_limit),
    )
    return rank_vector_candidates(
        question,
        vector_results,
        relevance_threshold=relevance_threshold,
        keyword_weight=keyword_weight,
    )[:limit]


def hybrid_rrf_search_vector_store(
    question: str,
    vector_store: Chroma,
    *,
    bm25_index: BM25Index | None = None,
    limit: int = 3,
    dense_candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
    bm25_candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
    relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
    rrf_k: int = DEFAULT_RRF_K,
) -> list[RRFFusionResult]:
    """让 Dense 与 BM25 独立召回，再用 RRF 产生最终候选排名。"""
    if not question.strip():
        return []
    if limit <= 0 or dense_candidate_limit <= 0 or bm25_candidate_limit <= 0:
        raise ValueError("检索数量必须大于 0。")
    if not 0.0 <= relevance_threshold <= 1.0:
        raise ValueError("相关度阈值必须在 0 到 1 之间。")

    dense_results = search_vector_store(
        question,
        vector_store,
        limit=dense_candidate_limit,
    )
    retained_dense_results = [
        (document, vector_score)
        for document, vector_score in dense_results
        if vector_score >= relevance_threshold
    ]
    sparse_index = bm25_index or build_bm25_index(vector_store)
    bm25_results = sparse_index.search(question, limit=bm25_candidate_limit)
    return reciprocal_rank_fusion(
        retained_dense_results,
        bm25_results,
        rrf_k=rrf_k,
    )[:limit]


def expand_adjacent_documents(
    vector_store: Chroma,
    documents: list[Document],
    adjacent_window: int = 1,
    context_limit: int = 8,
) -> list[Document]:
    """为检索种子补充同一来源中的相邻文本块，并保持种子优先。"""
    if adjacent_window < 0:
        raise ValueError("相邻文本块窗口不能小于 0。")
    if context_limit <= 0:
        raise ValueError("上下文文本块数量必须大于 0。")

    expanded: list[Document] = []
    seen: set[tuple[str, object]] = set()

    def append_document(
        document: Document,
        *,
        role: str,
        seed_rank: int | None = None,
    ) -> None:
        source = str(document.metadata.get("source", ""))
        chunk_index = document.metadata.get("chunk_index")
        identity = (
            source,
            chunk_index if chunk_index is not None else document.page_content,
        )
        if identity not in seen and len(expanded) < context_limit:
            seen.add(identity)
            metadata = {**document.metadata, CONTEXT_ROLE_METADATA_KEY: role}
            if seed_rank is not None:
                metadata[CONTEXT_SEED_RANK_METADATA_KEY] = seed_rank
            expanded.append(
                Document(page_content=document.page_content, metadata=metadata)
            )

    for seed_rank, document in enumerate(documents, start=1):
        append_document(document, role="seed", seed_rank=seed_rank)
    if adjacent_window == 0 or len(expanded) >= context_limit:
        return expanded

    source_cache: dict[str, dict[int, Document]] = {}
    for seed in documents:
        source = seed.metadata.get("source")
        chunk_index = seed.metadata.get("chunk_index")
        if not isinstance(source, str) or not isinstance(chunk_index, int):
            continue

        if source not in source_cache:
            stored = vector_store.get(
                where={"source": source},
                include=["documents", "metadatas"],
            )
            source_documents: dict[int, Document] = {}
            for content, metadata in zip(
                stored.get("documents") or [],
                stored.get("metadatas") or [],
                strict=True,
            ):
                if not isinstance(content, str) or not isinstance(metadata, dict):
                    continue
                stored_index = metadata.get("chunk_index")
                if isinstance(stored_index, int):
                    source_documents[stored_index] = Document(
                        page_content=content,
                        metadata=metadata,
                    )
            source_cache[source] = source_documents

        source_documents = source_cache[source]
        for distance in range(1, adjacent_window + 1):
            for neighbor_index in (chunk_index - distance, chunk_index + distance):
                neighbor = source_documents.get(neighbor_index)
                if neighbor is not None:
                    append_document(neighbor, role="adjacent")
                if len(expanded) >= context_limit:
                    return expanded

    return expanded


class HybridRetriever(BaseRetriever):
    """保留 Stage 2 的 Dense + 关键词覆盖率重排生产基线。"""

    vector_store: Chroma
    limit: int = 3
    candidate_limit: int = DEFAULT_CANDIDATE_LIMIT
    relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD
    keyword_weight: float = DEFAULT_KEYWORD_WEIGHT
    adjacent_window: int = 0
    context_limit: int = 8

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        results = hybrid_search_vector_store(
            query,
            self.vector_store,
            limit=self.limit,
            candidate_limit=self.candidate_limit,
            relevance_threshold=self.relevance_threshold,
            keyword_weight=self.keyword_weight,
        )
        documents = [result.document for result in results]
        if self.adjacent_window == 0:
            return documents
        return expand_adjacent_documents(
            self.vector_store,
            documents,
            adjacent_window=self.adjacent_window,
            context_limit=self.context_limit,
        )


class HybridRRFRetriever(BaseRetriever):
    """封装 Stage 3.1 的 BM25 + Dense + RRF 受控实验策略。"""

    vector_store: Chroma
    bm25_index: BM25Index | None = None
    limit: int = 3
    dense_candidate_limit: int = DEFAULT_CANDIDATE_LIMIT
    bm25_candidate_limit: int = DEFAULT_CANDIDATE_LIMIT
    relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD
    rrf_k: int = DEFAULT_RRF_K
    adjacent_window: int = 0
    context_limit: int = 8

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        results = hybrid_rrf_search_vector_store(
            query,
            self.vector_store,
            bm25_index=self.bm25_index,
            limit=self.limit,
            dense_candidate_limit=self.dense_candidate_limit,
            bm25_candidate_limit=self.bm25_candidate_limit,
            relevance_threshold=self.relevance_threshold,
            rrf_k=self.rrf_k,
        )
        documents = [result.document for result in results]
        if self.adjacent_window == 0:
            return documents
        return expand_adjacent_documents(
            self.vector_store,
            documents,
            adjacent_window=self.adjacent_window,
            context_limit=self.context_limit,
        )
