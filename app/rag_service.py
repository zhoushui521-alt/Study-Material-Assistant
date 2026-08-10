"""构造并管理可由 CLI 与 API 复用的 LangChain RAG 服务。"""

import logging
from dataclasses import dataclass, field

from langchain_chroma import Chroma
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable

if __package__:
    from app.chat_client import ChatConfig
    from app.embedding_client import EmbeddingConfig
    from app.hybrid_search import HybridRetriever
    from app.langchain_rag import (
        RAGAnswer,
        create_langchain_chat_model,
        create_rag_chain,
    )
    from app.langchain_store import create_langchain_embeddings, open_vector_store
else:
    from chat_client import ChatConfig
    from embedding_client import EmbeddingConfig
    from hybrid_search import HybridRetriever
    from langchain_rag import (
        RAGAnswer,
        create_langchain_chat_model,
        create_rag_chain,
    )
    from langchain_store import create_langchain_embeddings, open_vector_store


RETRIEVAL_LIMIT = 3
ADJACENT_WINDOW = 2
CONTEXT_LIMIT = 8
RELEVANCE_THRESHOLD = 0.25
logger = logging.getLogger(__name__)


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
    rag_chain: Runnable[str, RAGAnswer] = field(init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.rag_chain = create_rag_chain(self.retriever, self.chat_model)

    def ask(self, question: str) -> RAGAnswer:
        """沿当前进程复用的 LCEL RAG 管道回答问题。"""
        return self.rag_chain.invoke(question)

    def close(self) -> None:
        """幂等释放向量库客户端。"""
        if self._closed:
            return
        _close_vector_store(self.vector_store)
        self._closed = True


def create_rag_service() -> RAGService:
    """从环境配置和本地 Chroma 索引构造正式 RAG 服务。"""
    vector_store = None
    try:
        embeddings = create_langchain_embeddings(EmbeddingConfig.from_environment())
        vector_store = open_vector_store(embeddings)
        if not vector_store.get(limit=1)["ids"]:
            raise RAGServiceInitializationError(
                "向量库中没有资料，请先运行 app.index_langchain 建立索引。"
            )

        retriever = HybridRetriever(
            vector_store=vector_store,
            limit=RETRIEVAL_LIMIT,
            relevance_threshold=RELEVANCE_THRESHOLD,
            adjacent_window=ADJACENT_WINDOW,
            context_limit=CONTEXT_LIMIT,
        )
        chat_model = create_langchain_chat_model(ChatConfig.from_environment())
        return RAGService(
            vector_store=vector_store,
            retriever=retriever,
            chat_model=chat_model,
        )
    except Exception as error:
        if vector_store is not None:
            try:
                _close_vector_store(vector_store)
            except Exception:
                logger.exception("RAG 初始化失败后释放 Chroma 客户端失败。")
        if isinstance(error, RAGServiceInitializationError):
            raise
        raise RAGServiceInitializationError(f"初始化 RAG 服务失败：{error}") from error
