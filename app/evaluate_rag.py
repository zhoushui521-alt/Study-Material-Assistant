"""运行固定 RAG 评测集；该命令会调用真实 Embedding 和 Chat API。"""

import argparse
from datetime import UTC, datetime
from pathlib import Path

if __package__:
    from app.ask_langchain import (
        ADJACENT_WINDOW,
        CONTEXT_LIMIT,
        RELEVANCE_THRESHOLD,
        RETRIEVAL_LIMIT,
    )
    from app.chat_client import ChatConfig
    from app.embedding_client import EmbeddingConfig
    from app.hybrid_search import (
        DEFAULT_CANDIDATE_LIMIT,
        DEFAULT_KEYWORD_WEIGHT,
        HybridRetriever,
    )
    from app.langchain_rag import create_langchain_chat_model
    from app.langchain_store import create_langchain_embeddings, open_vector_store
    from app.rag_evaluation import (
        DEFAULT_EVALUATION_PATH,
        DEFAULT_EVALUATION_RESULTS_DIR,
        build_evaluation_report,
        evaluate_cases,
        find_missing_index_sources,
        load_evaluation_cases,
        write_evaluation_report,
    )
else:
    from ask_langchain import (
        ADJACENT_WINDOW,
        CONTEXT_LIMIT,
        RELEVANCE_THRESHOLD,
        RETRIEVAL_LIMIT,
    )
    from chat_client import ChatConfig
    from embedding_client import EmbeddingConfig
    from hybrid_search import (
        DEFAULT_CANDIDATE_LIMIT,
        DEFAULT_KEYWORD_WEIGHT,
        HybridRetriever,
    )
    from langchain_rag import create_langchain_chat_model
    from langchain_store import create_langchain_embeddings, open_vector_store
    from rag_evaluation import (
        DEFAULT_EVALUATION_PATH,
        DEFAULT_EVALUATION_RESULTS_DIR,
        build_evaluation_report,
        evaluate_cases,
        find_missing_index_sources,
        load_evaluation_cases,
        write_evaluation_report,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="运行固定 RAG 端到端评测集。")
    parser.add_argument(
        "--cases",
        type=Path,
        default=DEFAULT_EVALUATION_PATH,
        help="评测集 JSON 路径。",
    )
    parser.add_argument(
        "--confirm-api-cost",
        action="store_true",
        help="确认本次批量评测会产生真实 Embedding 和 Chat API 调用。",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_EVALUATION_RESULTS_DIR,
        help="结构化评测报告的保存目录。",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    if not args.confirm_api_cost:
        print(
            "未执行评测：该命令会为每个案例调用 Embedding，并在检索到资料时调用 Chat API。"
        )
        print("确认费用后请增加参数：--confirm-api-cost")
        return 2

    started_at = datetime.now(UTC)
    vector_store = None
    try:
        cases = load_evaluation_cases(args.cases)
        embedding_config = EmbeddingConfig.from_environment()
        embeddings = create_langchain_embeddings(embedding_config)
        vector_store = open_vector_store(embeddings)
        stored = vector_store.get(include=["metadatas"])
        if not stored.get("ids"):
            print("RAG 评测启动失败：向量库中没有资料，请先建立索引。")
            return 2

        missing_sources = find_missing_index_sources(
            cases,
            stored.get("metadatas") or [],
        )
        if missing_sources:
            print("RAG 评测启动失败：以下预期来源已不在当前索引中：")
            for source in missing_sources:
                print(f"- {source}")
            return 2

        retriever = HybridRetriever(
            vector_store=vector_store,
            limit=RETRIEVAL_LIMIT,
            candidate_limit=DEFAULT_CANDIDATE_LIMIT,
            relevance_threshold=RELEVANCE_THRESHOLD,
            keyword_weight=DEFAULT_KEYWORD_WEIGHT,
            adjacent_window=ADJACENT_WINDOW,
            context_limit=CONTEXT_LIMIT,
        )
        chat_config = ChatConfig.from_environment()
        chat_model = create_langchain_chat_model(chat_config)

        print(f"开始评测：{len(cases)} 个固定案例。")
        print(
            "检索配置："
            f"limit={RETRIEVAL_LIMIT}, threshold={RELEVANCE_THRESHOLD}, "
            f"adjacent_window={ADJACENT_WINDOW}, context_limit={CONTEXT_LIMIT}"
        )
        results = evaluate_cases(cases, retriever, chat_model)
        completed_at = datetime.now(UTC)
    except Exception as error:
        print(f"RAG 评测启动失败：{error}")
        return 2
    finally:
        if vector_store is not None:
            vector_store._client.close()

    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(
            f"\n[{status}] {result.case.case_id} "
            f"({result.case.category}, {result.elapsed_ms} ms)"
        )
        print(f"问题：{result.case.question}")
        if result.answer:
            print(f"回答：{result.answer}")
        if result.retrieved_sources:
            print("候选来源：")
            for source, chunk_index in result.retrieved_sources:
                print(f"- [{source} · 第 {chunk_index} 段]")
        for failure in result.failures:
            print(f"失败原因：{failure}")

    passed_count = sum(result.passed for result in results)
    print(f"\n评测汇总：{passed_count}/{len(results)} 通过。")
    try:
        report = build_evaluation_report(
            results,
            evaluation_path=args.cases,
            retrieval_parameters={
                "retrieval_limit": RETRIEVAL_LIMIT,
                "candidate_limit": DEFAULT_CANDIDATE_LIMIT,
                "relevance_threshold": RELEVANCE_THRESHOLD,
                "vector_weight": 1.0 - DEFAULT_KEYWORD_WEIGHT,
                "keyword_weight": DEFAULT_KEYWORD_WEIGHT,
                "adjacent_window": ADJACENT_WINDOW,
                "context_limit": CONTEXT_LIMIT,
            },
            embedding_model=embedding_config.model,
            chat_model=chat_config.model,
            started_at=started_at,
            completed_at=completed_at,
        )
        report_path = write_evaluation_report(report, args.results_dir)
    except Exception as error:
        print(f"评测报告保存失败：{error}")
        return 2

    print(f"结构化评测报告：{report_path}")
    return 0 if passed_count == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
