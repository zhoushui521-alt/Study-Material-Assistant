"""构造并管理可由 CLI 与 API 复用的 LangChain RAG 服务。"""

import logging
from pathlib import Path
from dataclasses import dataclass, field
from time import perf_counter

from langchain_chroma import Chroma
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable

if __package__:
    from app.config import get_settings
    from app.context_selector import ContextSelector, EvidenceScoreContextSelector
    from app.embedding_client import EmbeddingConfig
    from app.hybrid_search import HybridRetriever
    from app.langchain_rag import RAGAnswer, create_rag_chain
    from app.model_gateway import ModelGateway
    from app.observability import log_event
    from app.langchain_store import (
        VECTOR_STORE_DIR,
        create_langchain_embeddings,
        open_vector_store,
        runtime_index_config,
    )
    from app.index_manifest import (
        IndexCompatibilityStatus,
        check_index_compatibility,
    )
else:
    from config import get_settings
    from context_selector import ContextSelector, EvidenceScoreContextSelector
    from embedding_client import EmbeddingConfig
    from hybrid_search import HybridRetriever
    from langchain_rag import RAGAnswer, create_rag_chain
    from model_gateway import ModelGateway
    from observability import log_event
    from langchain_store import (
        VECTOR_STORE_DIR,
        create_langchain_embeddings,
        open_vector_store,
        runtime_index_config,
    )
    from index_manifest import IndexCompatibilityStatus, check_index_compatibility


RETRIEVAL_LIMIT = 3
ADJACENT_WINDOW = 2
CONTEXT_LIMIT = 8
RELEVANCE_THRESHOLD = 0.25


class RAGServiceInitializationError(RuntimeError):
    """RAG 配置、向量库或组件初始化失败时抛出的错误。"""


def _close_vector_store(vector_store: Chroma) -> None:
    """释放当前 Chroma 客户端；兼容其暂未提供公共 close 的版本。"""
    client = getattr(vector_store, "_client", None)
    close = getattr(client, "close", None)
    if callable(close):
        close()


@dataclass
class RAGService:
    """持有一次进程内复用的 Retriever、ChatModel 和向量库资源。"""

    vector_store: Chroma
    retriever: BaseRetriever
    chat_model: BaseChatModel
    index_status: IndexCompatibilityStatus = IndexCompatibilityStatus.COMPATIBLE
    context_selector: ContextSelector = field(
        default_factory=lambda: EvidenceScoreContextSelector(
            seed_count=RETRIEVAL_LIMIT,
            adjacent_per_seed=1,
        )
    )
    rag_chain: Runnable[str, RAGAnswer] = field(init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.rag_chain = create_rag_chain(
            self.retriever,
            self.chat_model,
            context_selector=self.context_selector,
        )

    def ask(self, question: str) -> RAGAnswer:
        """沿当前进程复用的 LCEL RAG 管道回答问题。"""
        started = perf_counter()
        log_event(
            "rag_request_started",
            details={"component": "rag_service"},
        )
        try:
            result = self.rag_chain.invoke(question)
        except Exception as error:
            log_event(
                "rag_request_failed",
                level=logging.ERROR,
                duration_ms=round((perf_counter() - started) * 1000),
                details={
                    "component": "rag_service",
                    "error_type": type(error).__name__,
                },
            )
            raise
        log_event(
            "rag_request_completed",
            duration_ms=round((perf_counter() - started) * 1000),
            details={"component": "rag_service"},
        )
        return result

    def close(self) -> None:
        """幂等释放向量库客户端。"""
        if self._closed:
            return
        _close_vector_store(self.vector_store)
        self._closed = True


def create_rag_service(
    persist_directory: Path = VECTOR_STORE_DIR,
    *,
    model_gateway: ModelGateway | None = None,
    user_id: str | None = None,
) -> RAGService:
    """从环境配置、本地 Chroma 与 Model Gateway 构造正式 RAG 服务。"""
    vector_store = None
    try:
        embedding_config = EmbeddingConfig.from_environment()
        embeddings = create_langchain_embeddings(embedding_config)
        vector_store = open_vector_store(embeddings, persist_directory)
        has_records = bool(vector_store.get(limit=1)["ids"])
        if not has_records:
            raise RAGServiceInitializationError(
                "向量库中没有资料，请先运行 app.index_langchain 建立索引。"
            )
        index_status = check_index_compatibility(
            persist_directory,
            runtime_index_config(embedding_config),
            has_records=has_records,
            access="read",
        )
        if index_status is IndexCompatibilityStatus.LEGACY_READ_ONLY:
            log_event(
                "legacy_index_read_only",
                level=logging.WARNING,
                details={"component": "vector_store"},
            )

        retriever = HybridRetriever(
            vector_store=vector_store,
            limit=RETRIEVAL_LIMIT,
            relevance_threshold=RELEVANCE_THRESHOLD,
            adjacent_window=ADJACENT_WINDOW,
            context_limit=CONTEXT_LIMIT,
        )
        gateway = model_gateway or ModelGateway.from_settings(get_settings())
        chat_model = gateway.create_chat_model(user_id)
        return RAGService(
            vector_store=vector_store,
            retriever=retriever,
            chat_model=chat_model,
            index_status=index_status,
        )
    except Exception as error:
        if vector_store is not None:
            try:
                _close_vector_store(vector_store)
            except Exception as cleanup_error:
                log_event(
                    "rag_initialization_cleanup_failed",
                    level=logging.ERROR,
                    details={
                        "component": "vector_store",
                        "error_type": type(cleanup_error).__name__,
                    },
                )
        if isinstance(error, RAGServiceInitializationError):
            raise
        raise RAGServiceInitializationError(f"初始化 RAG 服务失败：{error}") from error
