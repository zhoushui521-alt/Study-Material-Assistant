"""端到端 RAG：语义检索资料后，基于资料生成带来源的回答。"""

import sys

if __package__:
    from app.chat_client import ChatAPIError, ChatConfig, generate_answer
    from app.chunk_documents import DOCUMENTS_DIR, build_chunks, load_documents
    from app.embedding_client import EmbeddingAPIError, EmbeddingConfig
    from app.vector_search import search_by_vector
else:
    from chat_client import ChatAPIError, ChatConfig, generate_answer
    from chunk_documents import DOCUMENTS_DIR, build_chunks, load_documents
    from embedding_client import EmbeddingAPIError, EmbeddingConfig
    from vector_search import search_by_vector


def main() -> None:
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        question = input("请输入关于学习资料的问题：").strip()

    try:
        chunks = build_chunks(load_documents(DOCUMENTS_DIR))
        results = search_by_vector(question, chunks, EmbeddingConfig.from_environment())
        sources = [result.chunk for result in results]
        answer = generate_answer(question, sources, ChatConfig.from_environment())
    except (EmbeddingAPIError, ChatAPIError, ValueError) as error:
        print(f"RAG 调用失败：{error}")
        return

    print("回答：")
    print(answer)
    print("\n本次检索来源：")
    for result in results:
        print(f"- {result.chunk.source} · 第 {result.chunk.index} 段（相似度 {result.similarity:.4f}）")


if __name__ == "__main__":
    main()
