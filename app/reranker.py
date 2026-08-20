"""Stage 3.2 的模型无关 Reranker 契约与确定性测试实现。"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from langchain_core.documents import Document

if __package__:
    from app.evidence import evidence_from_document
else:
    from evidence import evidence_from_document


class RerankerError(RuntimeError):
    """Reranker 无法可靠完成打分或返回了无效结果。"""


@dataclass(frozen=True)
class RerankResult:
    """一条候选在 Reranker 前后的稳定排序信息。"""

    document: Document
    chunk_id: str
    original_rank: int
    reranker_score: float
    reranker_rank: int
    rank_change: int


class Reranker(ABC):
    """业务 Retrieval 依赖的模型无关接口。"""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """用于 Trace 与报告的模型身份。"""

    @property
    @abstractmethod
    def model_source(self) -> str:
        """模型或实现来源；不得包含密钥、令牌或本地敏感路径。"""

    @abstractmethod
    def score(
        self,
        query: str,
        documents: Sequence[Document],
    ) -> Sequence[float]:
        """为每个 Query/Document pair 返回一个同位置 relevance score。"""

    def rerank(
        self,
        query: str,
        documents: Sequence[Document],
    ) -> list[RerankResult]:
        """统一校验分数并稳定重排；分数相同时保留原候选顺序。"""
        if not query.strip():
            raise ValueError("Reranker query 不能为空。")
        candidates = tuple(documents)
        if not candidates:
            return []

        chunk_ids = tuple(
            evidence_from_document(document, "S1").chunk_id
            for document in candidates
        )
        if len(set(chunk_ids)) != len(chunk_ids):
            raise RerankerError("Reranker Candidate Pool 包含重复 chunk_id。")

        try:
            scores = tuple(self.score(query, candidates))
        except RerankerError:
            raise
        except Exception as error:
            raise RerankerError(f"Reranker 打分失败：{error}") from error
        if len(scores) != len(candidates):
            raise RerankerError(
                "Reranker 返回的分数数量与 Candidate Pool 不一致。"
            )

        scored: list[tuple[float, int, str, Document]] = []
        for original_rank, (document, chunk_id, score) in enumerate(
            zip(candidates, chunk_ids, scores, strict=True),
            start=1,
        ):
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise RerankerError("Reranker score 必须是有限数值。")
            numeric_score = float(score)
            if not math.isfinite(numeric_score):
                raise RerankerError("Reranker score 必须是有限数值。")
            scored.append((numeric_score, original_rank, chunk_id, document))

        ordered = sorted(scored, key=lambda item: (-item[0], item[1], item[2]))
        return [
            RerankResult(
                document=document,
                chunk_id=chunk_id,
                original_rank=original_rank,
                reranker_score=score,
                reranker_rank=reranker_rank,
                rank_change=original_rank - reranker_rank,
            )
            for reranker_rank, (score, original_rank, chunk_id, document) in enumerate(
                ordered,
                start=1,
            )
        ]


class DeterministicMockReranker(Reranker):
    """仅供测试和 Pipeline 验证使用；不能作为真实 Cross-Encoder 证据。"""

    def __init__(self, scores_by_chunk_id: Mapping[str, float]) -> None:
        self._scores_by_chunk_id = dict(scores_by_chunk_id)

    @property
    def model_name(self) -> str:
        return "deterministic-mock-reranker"

    @property
    def model_source(self) -> str:
        return "fixture"

    def score(
        self,
        query: str,
        documents: Sequence[Document],
    ) -> Sequence[float]:
        scores = []
        for document in documents:
            chunk_id = evidence_from_document(document, "S1").chunk_id
            if chunk_id not in self._scores_by_chunk_id:
                raise RerankerError(f"Mock 缺少 chunk_id {chunk_id} 的分数。")
            scores.append(self._scores_by_chunk_id[chunk_id])
        return tuple(scores)


class _FlagRerankerBackend(Protocol):
    def compute_score(
        self,
        sentence_pairs: list[tuple[str, str]],
        **kwargs: object,
    ) -> object: ...


class FlagEmbeddingCrossEncoderReranker(Reranker):
    """把本地 FlagEmbedding Cross-Encoder 隔离在模型无关接口之后。"""

    def __init__(
        self,
        backend: _FlagRerankerBackend,
        *,
        model_id: str,
        model_revision: str,
        max_length: int = 512,
        batch_size: int = 16,
        device: str = "cpu",
    ) -> None:
        if not model_id.strip() or not model_revision.strip():
            raise ValueError("Reranker model_id 与 revision 不能为空。")
        if max_length <= 0 or batch_size <= 0:
            raise ValueError("Reranker max_length 与 batch_size 必须大于 0。")
        self._backend = backend
        self._model_id = model_id
        self._model_revision = model_revision
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device

    @classmethod
    def from_local_model(
        cls,
        model_path: Path,
        *,
        model_id: str,
        model_revision: str,
        max_length: int = 512,
        batch_size: int = 16,
        device: str = "cpu",
    ) -> "FlagEmbeddingCrossEncoderReranker":
        """只从已下载的本地目录加载，不允许运行时隐式联网。"""
        if not model_path.is_dir():
            raise RerankerError(f"本地 Reranker 模型目录不存在：{model_path}。")
        required = ("config.json", "model.safetensors", "tokenizer_config.json")
        missing = [name for name in required if not (model_path / name).is_file()]
        if missing:
            raise RerankerError(
                "本地 Reranker 模型不完整，缺少：" + "、".join(missing)
            )
        try:
            from FlagEmbedding import FlagReranker

            backend = FlagReranker(
                str(model_path),
                use_fp16=False,
                trust_remote_code=False,
                devices=device,
                batch_size=batch_size,
                max_length=max_length,
                local_files_only=True,
            )
        except Exception as error:
            raise RerankerError(f"本地 Cross-Encoder 加载失败：{error}") from error
        return cls(
            backend,
            model_id=model_id,
            model_revision=model_revision,
            max_length=max_length,
            batch_size=batch_size,
            device=device,
        )

    @property
    def model_name(self) -> str:
        return f"{self._model_id}@{self._model_revision}"

    @property
    def model_source(self) -> str:
        return f"huggingface:{self._model_id}"

    def score(
        self,
        query: str,
        documents: Sequence[Document],
    ) -> Sequence[float]:
        pairs = [(query, document.page_content) for document in documents]
        raw_scores = self._backend.compute_score(
            pairs,
            batch_size=self.batch_size,
            max_length=self.max_length,
            normalize=False,
        )
        if isinstance(raw_scores, (int, float)) and not isinstance(raw_scores, bool):
            return (float(raw_scores),)
        if hasattr(raw_scores, "tolist"):
            raw_scores = raw_scores.tolist()
        if isinstance(raw_scores, (str, bytes)) or not isinstance(raw_scores, Iterable):
            raise RerankerError("Cross-Encoder 返回了无效的分数结构。")
        return tuple(float(score) for score in raw_scores)
