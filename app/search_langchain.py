"""使用 LangChain 从持久化 Chroma 向量库检索资料。"""

import sys

if __package__:
    from app.embedding_client import EmbeddingConfig
    from app.hybrid_search import hybrid_search_vector_store
    from app.langchain_store import create_langchain_embeddings, open_vector_store
else:
    from embedding_client import EmbeddingConfig
    from hybrid_search import hybrid_search_vector_store
    from langchain_store import create_langchain_embeddings, open_vector_store


def main() -> None:
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        question = input("请输入关于学习资料的问题：").strip()

    try:
        config = EmbeddingConfig.from_environment()
        embeddings = create_langchain_embeddings(config)
        vector_store = open_vector_store(embeddings)
        results = hybrid_search_vector_store(question, vector_store)
    except Exception as error:
        print(f"检索失败：{error}")
        return

    if not results:
        print("没有检索到资料。请先运行 index_langchain.py 建立索引。")
        return

    print(f"找到 {len(results)} 个相关资料片段：\n")
    for result in results:
        document = result.document
        print(
            f"[{document.metadata['source']} · 第 {document.metadata['chunk_index']} 段"
            f" · 混合分 {result.combined_score:.4f}"
            f" · 向量分 {result.vector_score:.4f}"
            f" · 关键词覆盖 {result.keyword_score:.4f}]"
        )
        print(f"{document.page_content}\n")


if __name__ == "__main__":
    main()
