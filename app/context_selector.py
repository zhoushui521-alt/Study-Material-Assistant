"""在已有 Retrieval Evidence 中构造更聚焦的 LLM Context。"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from hashlib import sha256

from langchain_core.documents import Document

if __package__:
    from app.hybrid_search import CONTEXT_ROLE_METADATA_KEY, keyword_coverage
else:
    from hybrid_search import CONTEXT_ROLE_METADATA_KEY, keyword_coverage


TokenCounter = Callable[[str], int]


class ContextSelectionError(ValueError):
    """Context Selector 配置或 Token Counter 返回无效结果。"""


class ContextSelector(ABC):
    """只从已有 Retrieval Evidence 中选择 LLM Context，不允许重新召回。"""

    @abstractmethod
    def select(
        self,
        query: str,
        evidences: Sequence[Document],
    ) -> list[Document]:
        """返回 ``evidences`` 的有序子集。"""


@dataclass(frozen=True)
class BaselineContextSelector(ContextSelector):
    """A 组：完整保留 Retriever 已构造的当前 Context。"""

    def select(
        self,
        query: str,
        evidences: Sequence[Document],
    ) -> list[Document]:
        del query
        return list(evidences)


def _document_identity(document: Document) -> tuple[str, ...]:
    chunk_id = document.metadata.get("chunk_id")
    if isinstance(chunk_id, str) and chunk_id:
        return ("chunk_id", chunk_id)
    source = str(document.metadata.get("source") or "")
    chunk_index = document.metadata.get("chunk_index")
    return (
        "fallback",
        source,
        str(chunk_index),
        sha256(document.page_content.encode("utf-8")).hexdigest(),
    )


def _deduplicate(evidences: Sequence[Document]) -> list[Document]:
    selected = []
    seen: set[tuple[str, ...]] = set()
    for evidence in evidences:
        identity = _document_identity(evidence)
        if identity in seen:
            continue
        seen.add(identity)
        selected.append(evidence)
    return selected


def _adjacent_distance(seed: Document, candidate: Document) -> int | None:
    if seed.metadata.get("source") != candidate.metadata.get("source"):
        return None
    seed_index = seed.metadata.get("chunk_index")
    candidate_index = candidate.metadata.get("chunk_index")
    if (
        isinstance(seed_index, bool)
        or not isinstance(seed_index, int)
        or isinstance(candidate_index, bool)
        or not isinstance(candidate_index, int)
    ):
        return None
    distance = abs(seed_index - candidate_index)
    return distance if distance > 0 else None


@dataclass(frozen=True)
class EvidenceScoreContextSelector(ContextSelector):
    """B 组：保留 Seed，并为每个 Seed 选择一个最相关的已有相邻 Evidence。

    ``HybridRetriever`` 会先返回所有 Seed，再返回相邻扩展结果。因此这里只读取已有
    Context 的顺序、正文和 metadata，不查询 Vector Store，也不改变 Retrieval。
    相邻 Evidence 先按 query keyword coverage 排序，再以距离和原顺序稳定打破并列。

    ``token_budget`` 只有在注入与目标 ChatModel 匹配的可靠 ``token_counter`` 时才可
    启用；Stage 3.4 默认不启用，避免把字符估算伪装成真实 Token 数。
    """

    seed_count: int = 3
    adjacent_per_seed: int = 1
    token_budget: int | None = None
    token_counter: TokenCounter | None = None

    def __post_init__(self) -> None:
        if self.seed_count <= 0:
            raise ContextSelectionError("Seed 数量必须大于 0。")
        if self.adjacent_per_seed < 0:
            raise ContextSelectionError("每个 Seed 的相邻 Evidence 数量不能小于 0。")
        if self.token_budget is not None:
            if self.token_budget <= 0:
                raise ContextSelectionError("Token Budget 必须大于 0。")
            if self.token_counter is None:
                raise ContextSelectionError(
                    "启用 Token Budget 时必须提供可靠的 Token Counter。"
                )
        elif self.token_counter is not None:
            raise ContextSelectionError(
                "只有启用 Token Budget 时才应提供 Token Counter。"
            )

    def select(
        self,
        query: str,
        evidences: Sequence[Document],
    ) -> list[Document]:
        unique = _deduplicate(evidences)
        if not unique:
            return []

        has_provenance = any(
            evidence.metadata.get(CONTEXT_ROLE_METADATA_KEY) in {"seed", "adjacent"}
            for evidence in unique
        )
        if has_provenance:
            if not all(
                evidence.metadata.get(CONTEXT_ROLE_METADATA_KEY)
                in {"seed", "adjacent"}
                for evidence in unique
            ):
                raise ContextSelectionError(
                    "Context Evidence 的 Seed/Adjacent provenance 不完整。"
                )
            seeds = [
                evidence
                for evidence in unique
                if evidence.metadata.get(CONTEXT_ROLE_METADATA_KEY) == "seed"
            ]
            adjacent = [
                evidence
                for evidence in unique
                if evidence.metadata.get(CONTEXT_ROLE_METADATA_KEY) == "adjacent"
            ]
        else:
            seeds = unique[: self.seed_count]
            adjacent = unique[self.seed_count :]
        selected_ids = {_document_identity(seed) for seed in seeds}
        adjacent_positions = {
            _document_identity(document): position
            for position, document in enumerate(adjacent)
        }

        for seed in seeds:
            for _ in range(self.adjacent_per_seed):
                candidates = []
                for candidate in adjacent:
                    identity = _document_identity(candidate)
                    if identity in selected_ids:
                        continue
                    distance = _adjacent_distance(seed, candidate)
                    if distance is None:
                        continue
                    candidates.append(
                        (
                            keyword_coverage(query, candidate.page_content),
                            -distance,
                            -adjacent_positions[identity],
                            candidate,
                        )
                    )
                if not candidates:
                    break
                selected_ids.add(_document_identity(max(candidates, key=lambda item: item[:3])[3]))

        preferred = [
            evidence
            for evidence in unique
            if _document_identity(evidence) in selected_ids
        ]
        if self.token_budget is None:
            return preferred
        return self._apply_token_budget(preferred)

    def _apply_token_budget(self, evidences: Sequence[Document]) -> list[Document]:
        assert self.token_budget is not None
        assert self.token_counter is not None
        selected = []
        used_tokens = 0
        for evidence in evidences:
            token_count = self.token_counter(evidence.page_content)
            if isinstance(token_count, bool) or not isinstance(token_count, int):
                raise ContextSelectionError("Token Counter 必须返回非负整数。")
            if token_count < 0:
                raise ContextSelectionError("Token Counter 必须返回非负整数。")
            if used_tokens + token_count > self.token_budget:
                continue
            selected.append(evidence)
            used_tokens += token_count
        return selected
