"""完整 LangChain RAG：从 Chroma 检索资料并生成带来源的回答。"""

import sys

if __package__:
    from app.langchain_rag import source_label
    from app.rag_service import (
        ADJACENT_WINDOW,
        CONTEXT_LIMIT,
        RELEVANCE_THRESHOLD,
        RETRIEVAL_LIMIT,
        RAGService,
        create_rag_service,
    )
else:
    from langchain_rag import source_label
    from rag_service import (
        ADJACENT_WINDOW,
        CONTEXT_LIMIT,
        RELEVANCE_THRESHOLD,
        RETRIEVAL_LIMIT,
        RAGService,
        create_rag_service,
    )


def main() -> None:
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        question = input("请输入关于学习资料的问题：").strip()

    service: RAGService | None = None
    try:
        service = create_rag_service()
        result = service.ask(question)
    except Exception as error:
        print(f"LangChain RAG 调用失败：{error}")
        return
    finally:
        if service is not None:
            try:
                service.close()
            except Exception as error:
                print(f"RAG 资源释放失败：{error}", file=sys.stderr)

    print("回答：")
    print(result.answer)
    if result.sources:
        print("\n本次检索候选资料：")
        for document in result.sources:
            print(f"- {source_label(document)}")


if __name__ == "__main__":
    main()
