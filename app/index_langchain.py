"""将学习资料写入 LangChain + Chroma 持久化向量库。"""

if __package__:
    from app.chunk_documents import DOCUMENTS_DIR, build_chunks, load_documents
    from app.embedding_client import EmbeddingConfig
    from app.langchain_store import (
        VECTOR_STORE_DIR,
        create_langchain_embeddings,
        sync_vector_store,
    )
else:
    from chunk_documents import DOCUMENTS_DIR, build_chunks, load_documents
    from embedding_client import EmbeddingConfig
    from langchain_store import (
        VECTOR_STORE_DIR,
        create_langchain_embeddings,
        sync_vector_store,
    )


def main() -> None:
    try:
        chunks = build_chunks(load_documents(DOCUMENTS_DIR))
        config = EmbeddingConfig.from_environment()
        embeddings = create_langchain_embeddings(config)
        sync_result = sync_vector_store(chunks, embeddings)
    except Exception as error:
        print(f"同步索引失败：{error}")
        return

    print("资料索引同步成功。")
    print(f"文本块数量：{len(chunks)}")
    print(f"新增或更新：{sync_result.added}")
    print(f"删除旧记录：{sync_result.deleted}")
    print(f"未变化：{sync_result.unchanged}")
    print(f"向量库位置：{VECTOR_STORE_DIR}")
    print(f"Chroma 记录数：{len(sync_result.vector_store.get(include=[])['ids'])}")


if __name__ == "__main__":
    main()
