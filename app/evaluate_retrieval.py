"""运行 Stage 2 Retrieval Benchmark；只调用 Query Embedding，不调用 ChatModel。"""

import argparse
import json
import subprocess
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

from langchain_core.documents import Document

if __package__:
    from app.embedding_client import EmbeddingConfig
    from app.index_manifest import check_index_compatibility, load_index_manifest
    from app.langchain_store import (
        VECTOR_STORE_DIR,
        close_vector_store,
        create_langchain_embeddings,
        open_vector_store,
        runtime_index_config,
    )
    from app.rag_service import (
        ADJACENT_WINDOW,
        CONTEXT_LIMIT,
        RELEVANCE_THRESHOLD,
        RETRIEVAL_LIMIT,
    )
    from app.retrieval_evaluation import (
        DEFAULT_RETRIEVAL_EVALUATION_PATH,
        DEFAULT_RETRIEVAL_RESULTS_DIR,
        RetrievalConfig,
        build_retrieval_report,
        evaluate_retrieval_cases,
        find_unresolved_gold_mappings,
        load_retrieval_cases,
        write_retrieval_report,
    )
else:
    from embedding_client import EmbeddingConfig
    from index_manifest import check_index_compatibility, load_index_manifest
    from langchain_store import (
        VECTOR_STORE_DIR,
        close_vector_store,
        create_langchain_embeddings,
        open_vector_store,
        runtime_index_config,
    )
    from rag_service import (
        ADJACENT_WINDOW,
        CONTEXT_LIMIT,
        RELEVANCE_THRESHOLD,
        RETRIEVAL_LIMIT,
    )
    from retrieval_evaluation import (
        DEFAULT_RETRIEVAL_EVALUATION_PATH,
        DEFAULT_RETRIEVAL_RESULTS_DIR,
        RetrievalConfig,
        build_retrieval_report,
        evaluate_retrieval_cases,
        find_unresolved_gold_mappings,
        load_retrieval_cases,
        write_retrieval_report,
    )

if __package__:
    from app.config import PROJECT_ROOT
    from app.hybrid_search import DEFAULT_CANDIDATE_LIMIT, DEFAULT_KEYWORD_WEIGHT
else:
    from config import PROJECT_ROOT
    from hybrid_search import DEFAULT_CANDIDATE_LIMIT, DEFAULT_KEYWORD_WEIGHT


STAGE2_START_COMMIT = "b99f4a73de123c1428c7a88320765186cf3c727b"


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="运行独立 Retrieval Evaluation 与本地结构化 Trace。"
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=DEFAULT_RETRIEVAL_EVALUATION_PATH,
        help="Retrieval Dataset JSON 路径。",
    )
    parser.add_argument(
        "--confirm-query-embedding-cost",
        action="store_true",
        help="确认每个案例会调用一次真实 Query Embedding。",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RETRIEVAL_RESULTS_DIR,
        help="Retrieval Report 保存目录。",
    )
    return parser


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        text=True,
        encoding="utf-8",
    ).strip()


def _read_index_state(
    vector_store: object,
) -> tuple[tuple[tuple[str, str, str], ...], list[Document]]:
    stored = vector_store.get(include=["documents", "metadatas"])
    ids = stored.get("ids") or []
    documents = [
        Document(page_content=content, metadata=metadata)
        for content, metadata in zip(
            stored.get("documents") or [],
            stored.get("metadatas") or [],
            strict=True,
        )
        if isinstance(content, str) and isinstance(metadata, dict)
    ]
    snapshot = tuple(
        sorted(
            (
                str(item_id),
                sha256(document.page_content.encode("utf-8")).hexdigest(),
                sha256(
                    json.dumps(
                        document.metadata,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest(),
            )
            for item_id, document in zip(ids, documents, strict=True)
        )
    )
    return snapshot, documents


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        cases = load_retrieval_cases(args.cases)
    except Exception as error:
        print(f"Retrieval Evaluation 启动失败：{error}")
        return 2
    if not args.confirm_query_embedding_cost:
        print("未执行真实 Retrieval Benchmark。")
        print(f"案例数：{len(cases)}")
        print(f"预计 Query Embedding 次数：{len(cases)}")
        print("索引访问：仅以 legacy_read_only/compatible 读取，不执行写入。")
        print("Chat/LLM 调用：0")
        print("确认费用后请增加参数：--confirm-query-embedding-cost")
        return 2

    vector_store = None
    started_at = datetime.now(UTC)
    try:
        embedding_config = EmbeddingConfig.from_environment()
        embeddings = create_langchain_embeddings(embedding_config)
        vector_store = open_vector_store(embeddings)
        stored = vector_store.get(include=[])
        has_records = bool(stored.get("ids"))
        if not has_records:
            print("Retrieval Evaluation 启动失败：当前索引没有记录。")
            return 2
        index_status = check_index_compatibility(
            VECTOR_STORE_DIR,
            runtime_index_config(embedding_config),
            has_records=has_records,
            access="read",
        )
        manifest_before = load_index_manifest(VECTOR_STORE_DIR)
        index_snapshot_before, indexed_documents = _read_index_state(vector_store)
        unresolved = find_unresolved_gold_mappings(cases, indexed_documents)
        if unresolved:
            print("Retrieval Evaluation 启动失败：Current Chunk Mapping 无法定位：")
            for item in unresolved:
                print(f"- {item}")
            return 2
        config = RetrievalConfig(
            retrieval_limit=RETRIEVAL_LIMIT,
            candidate_limit=DEFAULT_CANDIDATE_LIMIT,
            relevance_threshold=RELEVANCE_THRESHOLD,
            vector_weight=1.0 - DEFAULT_KEYWORD_WEIGHT,
            keyword_weight=DEFAULT_KEYWORD_WEIGHT,
            adjacent_window=ADJACENT_WINDOW,
            context_limit=CONTEXT_LIMIT,
        )
        print(f"开始 Retrieval Evaluation：{len(cases)} 个案例。")
        print(f"索引状态：{index_status.value}；只读执行，不迁移、不补 Manifest。")
        results = evaluate_retrieval_cases(cases, vector_store, config)
        index_snapshot_after, _ = _read_index_state(vector_store)
        manifest_after = load_index_manifest(VECTOR_STORE_DIR)
        if index_snapshot_after != index_snapshot_before or manifest_after != manifest_before:
            raise RuntimeError("Retrieval Evaluation 检测到索引或 Manifest 发生变化。")
        report = build_retrieval_report(
            results,
            dataset_path=args.cases,
            config=config,
            git_commit=_git_commit(),
            stage2_start_commit=STAGE2_START_COMMIT,
            generated_at=datetime.now(UTC),
            validation_level="local_real_retrieval",
            embedding_model=embedding_config.model,
            query_embedding_calls=len(cases),
        )
        report_path = write_retrieval_report(report, args.results_dir)
    except Exception as error:
        print(f"Retrieval Evaluation 执行失败：{error}")
        return 2
    finally:
        if vector_store is not None:
            close_vector_store(vector_store)

    failures = [result for result in results if result.failure_category]
    for result in results:
        status = result.failure_category or "ok"
        print(f"[{status}] {result.case.case_id}")
    print(f"Retrieval Report：{report_path}")
    print("本次未调用 ChatModel；Citation Coverage/Support 未自动评估。")
    print(f"总耗时：{round((datetime.now(UTC) - started_at).total_seconds() * 1000)} ms")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
