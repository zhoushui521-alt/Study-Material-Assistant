"""使用 LangChain 和 Chroma 管理持久化学习资料向量库。"""

import hashlib
import re
from dataclasses import dataclass
from math import ceil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

if __package__:
    from app.chunk_documents import DocumentChunk, PROJECT_ROOT
    from app.embedding_client import EmbeddingConfig
    from app.security_limits import EMBEDDING_TIMEOUT_SECONDS, EXTERNAL_API_MAX_RETRIES
else:
    from chunk_documents import DocumentChunk, PROJECT_ROOT
    from embedding_client import EmbeddingConfig
    from security_limits import EMBEDDING_TIMEOUT_SECONDS, EXTERNAL_API_MAX_RETRIES


VECTOR_STORE_DIR = PROJECT_ROOT / "data" / "vector_store"
COLLECTION_NAME = "study_materials"
# 当前百炼 Embedding 接口实测单次最多接受 10 条文本。
BAILIAN_EMBEDDING_BATCH_SIZE = 10


@dataclass(frozen=True)
class VectorStoreSyncResult:
    """一次增量索引同步的结果。"""

    vector_store: Chroma
    added: int
    deleted: int
    unchanged: int


def create_langchain_embeddings(config: EmbeddingConfig) -> OpenAIEmbeddings:
    """用百炼的 OpenAI 兼容地址创建 LangChain Embedding 组件。"""
    return OpenAIEmbeddings(
        api_key=config.api_key,
        base_url=config.base_url,
        model=config.model,
        dimensions=config.dimensions,
        check_embedding_ctx_length=False,
        chunk_size=BAILIAN_EMBEDDING_BATCH_SIZE,
        timeout=EMBEDDING_TIMEOUT_SECONDS,
        max_retries=EXTERNAL_API_MAX_RETRIES,
    )


def open_vector_store(
    embedding_function: Embeddings | None,
    persist_directory: Path = VECTOR_STORE_DIR,
) -> Chroma:
    """打开或创建一个本地持久化 Chroma 集合。"""
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_function,
        persist_directory=str(persist_directory),
        collection_metadata={"hnsw:space": "cosine"},
    )


def close_vector_store(vector_store: Chroma) -> None:
    """释放当前 Chroma 客户端；兼容其暂未提供公共 close 的版本。"""
    client = getattr(vector_store, "_client", None)
    close = getattr(client, "close", None)
    if callable(close):
        close()


def chunks_to_documents(chunks: list[DocumentChunk]) -> list[Document]:
    """把手写阶段的文本块转换成 LangChain 标准 Document。"""
    return [
        Document(
            page_content=chunk.content,
            metadata={"source": chunk.source, "chunk_index": chunk.index},
        )
        for chunk in chunks
    ]


def document_id(document: Document) -> str:
    """根据来源、段号和内容生成稳定 ID，便于识别同一资料块。"""
    raw_value = (
        f"{document.metadata['source']}:{document.metadata['chunk_index']}:"
        f"{document.page_content}"
    )
    return hashlib.sha256(raw_value.encode("utf-8")).hexdigest()


def rebuild_vector_store(
    chunks: list[DocumentChunk],
    embedding_function: Embeddings,
    persist_directory: Path = VECTOR_STORE_DIR,
) -> Chroma:
    """清空旧索引，并把当前资料重新写入 Chroma。"""
    if not chunks:
        raise ValueError("没有可建立索引的资料文本块。")

    vector_store = open_vector_store(embedding_function, persist_directory)
    vector_store.reset_collection()
    documents = chunks_to_documents(chunks)
    vector_store.add_documents(
        documents=documents,
        ids=[document_id(document) for document in documents],
    )
    return vector_store


def sync_vector_store(
    chunks: list[DocumentChunk],
    embedding_function: Embeddings,
    persist_directory: Path = VECTOR_STORE_DIR,
    *,
    allow_empty: bool = False,
) -> VectorStoreSyncResult:
    """只添加新增或变化的文本块，并删除当前资料中已不存在的旧记录。"""
    if not chunks and not allow_empty:
        raise ValueError("没有可同步索引的资料文本块。")

    documents = chunks_to_documents(chunks)
    documents_by_id = {document_id(document): document for document in documents}
    if len(documents_by_id) != len(documents):
        raise ValueError("资料文本块生成了重复 ID，无法安全同步索引。")

    vector_store = open_vector_store(embedding_function, persist_directory)
    try:
        stored_ids = set(vector_store.get(include=[])["ids"])
        current_ids = set(documents_by_id)

        new_ids = [item_id for item_id in documents_by_id if item_id not in stored_ids]
        stale_ids = sorted(stored_ids - current_ids)
        unchanged_count = len(stored_ids & current_ids)

        # 先写入新记录，再删除旧记录。若写入失败，原索引仍保持可用。
        if new_ids:
            vector_store.add_documents(
                documents=[documents_by_id[item_id] for item_id in new_ids],
                ids=new_ids,
            )
        if stale_ids:
            vector_store.delete(ids=stale_ids)
    except Exception:
        close_vector_store(vector_store)
        raise

    return VectorStoreSyncResult(
        vector_store=vector_store,
        added=len(new_ids),
        deleted=len(stale_ids),
        unchanged=unchanged_count,
    )


def estimate_vector_store_sync_batches(
    chunks: list[DocumentChunk],
    persist_directory: Path = VECTOR_STORE_DIR,
) -> int:
    """零费用计算下一次增量同步真正需要新增的 Embedding 批次数。"""
    documents = chunks_to_documents(chunks)
    documents_by_id = {document_id(document): document for document in documents}
    if len(documents_by_id) != len(documents):
        raise ValueError("资料文本块生成了重复 ID，无法安全同步索引。")

    vector_store = open_vector_store(None, persist_directory)
    try:
        stored_ids = set(vector_store.get(include=[])["ids"])
    finally:
        close_vector_store(vector_store)
    new_document_count = len(set(documents_by_id) - stored_ids)
    return ceil(new_document_count / BAILIAN_EMBEDDING_BATCH_SIZE)


def source_belongs_to_material(source: object, filename: str) -> bool:
    """判断 Chroma 来源标签是否属于指定资料文件。"""
    if not isinstance(source, str):
        return False
    if source == filename:
        return True
    if source.startswith(f"{filename} · 网页：http://") or source.startswith(
        f"{filename} · 网页：https://"
    ):
        return True
    if Path(filename).suffix.lower() != ".pdf":
        return False
    return re.fullmatch(rf"{re.escape(filename)} · 第 [1-9][0-9]* 页", source) is not None


def delete_material_documents(
    filename: str,
    persist_directory: Path = VECTOR_STORE_DIR,
) -> int:
    """不调用 Embedding，按来源标签删除一个资料文件对应的 Chroma 记录。"""
    vector_store = open_vector_store(None, persist_directory)
    try:
        stored = vector_store.get(include=["metadatas"])
        matching_ids = [
            item_id
            for item_id, metadata in zip(
                stored.get("ids") or [],
                stored.get("metadatas") or [],
                strict=True,
            )
            if isinstance(metadata, dict)
            and source_belongs_to_material(metadata.get("source"), filename)
        ]
        if matching_ids:
            vector_store.delete(ids=matching_ids)
        return len(matching_ids)
    finally:
        close_vector_store(vector_store)


def search_vector_store(
    question: str,
    vector_store: Chroma,
    limit: int = 3,
) -> list[tuple[Document, float]]:
    """从已建立的 Chroma 索引中检索资料，返回文档和相关度分数。"""
    if not question.strip():
        return []
    return vector_store.similarity_search_with_relevance_scores(question, k=limit)
