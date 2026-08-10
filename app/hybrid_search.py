"""结合向量相关度与关键词覆盖率，对 Chroma 候选结果重新排序。"""

from dataclasses import dataclass

from langchain_chroma import Chroma
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

if __package__:
    from app.langchain_store import search_vector_store
    from app.search_documents import extract_terms
else:
    from langchain_store import search_vector_store
    from search_documents import extract_terms


DEFAULT_CANDIDATE_LIMIT = 10
DEFAULT_KEYWORD_WEIGHT = 0.2
DEFAULT_RELEVANCE_THRESHOLD = 0.25


@dataclass(frozen=True)
class HybridSearchResult:
    """一条包含向量分、关键词覆盖率和混合分的检索结果。"""

    document: Document
    vector_score: float
    keyword_score: float
    combined_score: float


def keyword_coverage(question: str, text: str) -> float:
    """计算问题词项中有多少比例出现在候选文本里。"""
    terms = extract_terms(question)
    if not terms:
        return 0.0

    normalized_text = text.casefold()
    matched_count = sum(term in normalized_text for term in terms)
    return matched_count / len(terms)


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
    hybrid_results = []
    for document, vector_score in vector_results:
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
                vector_score=vector_score,
                keyword_score=keyword_score,
                combined_score=combined_score,
            )
        )

    return sorted(
        hybrid_results,
        key=lambda result: (result.combined_score, result.vector_score),
        reverse=True,
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

    def append_document(document: Document) -> None:
        source = str(document.metadata.get("source", ""))
        chunk_index = document.metadata.get("chunk_index")
        identity = (
            source,
            chunk_index if chunk_index is not None else document.page_content,
        )
        if identity not in seen and len(expanded) < context_limit:
            seen.add(identity)
            expanded.append(document)

    for document in documents:
        append_document(document)
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
                    append_document(neighbor)
                if len(expanded) >= context_limit:
                    return expanded

    return expanded


class HybridRetriever(BaseRetriever):
    """把混合检索和可选的相邻块扩展包装成 LangChain 标准 Retriever。"""

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
