"""运行 Stage 3.1 BM25 + Dense + RRF 受控 Retrieval 实验。"""

import argparse
import subprocess
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

if __package__:
    from app.embedding_client import EmbeddingConfig
    from app.evaluate_retrieval import (
        MAX_QUERY_EMBEDDING_CALLS,
        _git_commit,
        _preflight_snapshot,
        disposable_evaluation_snapshot,
    )
    from app.hybrid_retrieval_evaluation import (
        HybridRetrievalConfig,
        build_hybrid_report,
        evaluate_hybrid_cases,
        write_hybrid_report,
    )
    from app.hybrid_search import build_bm25_index
    from app.langchain_store import (
        VECTOR_STORE_DIR,
        close_vector_store,
        create_langchain_embeddings,
        open_vector_store,
    )
    from app.retrieval_evaluation import (
        DEFAULT_RETRIEVAL_EVALUATION_PATH,
        DEFAULT_RETRIEVAL_RESULTS_DIR,
        _dataset_metadata,
        load_retrieval_cases,
    )
else:
    from embedding_client import EmbeddingConfig
    from evaluate_retrieval import (
        MAX_QUERY_EMBEDDING_CALLS,
        _git_commit,
        _preflight_snapshot,
        disposable_evaluation_snapshot,
    )
    from hybrid_retrieval_evaluation import (
        HybridRetrievalConfig,
        build_hybrid_report,
        evaluate_hybrid_cases,
        write_hybrid_report,
    )
    from hybrid_search import build_bm25_index
    from langchain_store import (
        VECTOR_STORE_DIR,
        close_vector_store,
        create_langchain_embeddings,
        open_vector_store,
    )
    from retrieval_evaluation import (
        DEFAULT_RETRIEVAL_EVALUATION_PATH,
        DEFAULT_RETRIEVAL_RESULTS_DIR,
        _dataset_metadata,
        load_retrieval_cases,
    )


STAGE3_1_START_COMMIT = "43e54de209e3d46d86f274f7b405f5c50f4b9677"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION_PATHS = (
    Path("app/evaluate_hybrid_retrieval.py"),
    Path("app/hybrid_retrieval_evaluation.py"),
    Path("app/hybrid_search.py"),
    Path("app/rag_service.py"),
)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="比较 Stage 2 Baseline 与 BM25 + Dense + RRF。"
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=DEFAULT_RETRIEVAL_EVALUATION_PATH,
        help="Stage 2 Retrieval Dataset 路径。",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RETRIEVAL_RESULTS_DIR,
        help="实验报告输出目录。",
    )
    parser.add_argument(
        "--confirm-query-embedding-cost",
        action="store_true",
        help="明确确认每个案例一次真实 Query Embedding 费用。",
    )
    return parser


def _config() -> HybridRetrievalConfig:
    return HybridRetrievalConfig()


def _source_state(git_commit: str) -> dict[str, object]:
    """记录未提交实验代码的内容身份，避免把结果误归因给 base commit。"""
    repository = str(PROJECT_ROOT.resolve()).replace("\\", "/")
    status = subprocess.check_output(
        [
            "git",
            "-c",
            f"safe.directory={repository}",
            "status",
            "--porcelain",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        encoding="utf-8",
        stderr=subprocess.STDOUT,
    )
    combined = sha256()
    file_hashes: dict[str, str] = {}
    for relative_path in IMPLEMENTATION_PATHS:
        content = (PROJECT_ROOT / relative_path).read_bytes()
        file_hash = sha256(content).hexdigest()
        normalized_path = relative_path.as_posix()
        file_hashes[normalized_path] = file_hash
        combined.update(normalized_path.encode("utf-8"))
        combined.update(b"\0")
        combined.update(content)
        combined.update(b"\0")
    return {
        "base_git_commit": git_commit,
        "worktree_clean": not bool(status.strip()),
        "implementation_sha256": combined.hexdigest(),
        "implementation_files_sha256": file_hashes,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        cases = load_retrieval_cases(args.cases)
    except Exception as error:
        print(f"Stage 3.1 Retrieval Experiment 启动失败：{error}")
        return 2
    if len(cases) > MAX_QUERY_EMBEDDING_CALLS:
        print(
            "Stage 3.1 Retrieval Experiment 启动失败："
            f"{len(cases)} 个案例超过 {MAX_QUERY_EMBEDDING_CALLS} 次 Query Embedding 上限。"
        )
        return 2
    if not args.confirm_query_embedding_cost:
        print("未执行真实 Stage 3.1 Retrieval Experiment。")
        print(f"案例数：{len(cases)}")
        print(f"预计 Query Embedding 次数：{len(cases)}")
        print("每个 Query 的同一次 Dense 结果同时供 Baseline 与 Hybrid 使用。")
        print("BM25、RRF 与报告聚合均为本地计算；Chat/LLM 调用：0。")
        print("确认费用后请增加参数：--confirm-query-embedding-cost")
        return 2

    vector_store = None
    started_at = datetime.now(UTC)
    try:
        git_commit = _git_commit()
        source_state = _source_state(git_commit)
        embedding_config = EmbeddingConfig.from_environment()
        config = _config()
        with disposable_evaluation_snapshot(VECTOR_STORE_DIR) as snapshot:
            index_status = _preflight_snapshot(snapshot.path, embedding_config, cases)
            embeddings = create_langchain_embeddings(embedding_config)
            vector_store = open_vector_store(
                embeddings,
                persist_directory=snapshot.path,
            )
            try:
                bm25_index = build_bm25_index(vector_store)
                print(f"开始 Stage 3.1 Retrieval Experiment：{len(cases)} 个案例。")
                print(
                    f"索引状态：{index_status.value}；Dense 与 BM25 都查询同一 disposable snapshot。"
                )
                results = evaluate_hybrid_cases(cases, vector_store, bm25_index, config)
            finally:
                close_vector_store(vector_store)
                vector_store = None
            report = build_hybrid_report(
                results,
                config=config,
                git_commit=git_commit,
                stage3_1_start_commit=STAGE3_1_START_COMMIT,
                dataset_metadata=_dataset_metadata(args.cases),
                embedding_model=embedding_config.model,
                query_embedding_calls=len(cases),
                validation_level="local_real_retrieval",
                index_fingerprint=snapshot.original_fingerprint,
                generated_at=datetime.now(UTC),
            )
            report["source_state"] = source_state
        report_path = write_hybrid_report(report, args.results_dir)
    except Exception as error:
        print(f"Stage 3.1 Retrieval Experiment 执行失败：{error}")
        return 2
    finally:
        if vector_store is not None:
            close_vector_store(vector_store)

    regressions = [
        result
        for result in report["per_case_results"]
        if result["case_analysis"]["outcome"] in {"lost_recall", "ranking_regressed"}
    ]
    print(f"Stage 3.1 Retrieval Report：{report_path}")
    print("原始 Index：运行前后 filesystem SHA-256 指纹一致。")
    print("本次未调用 ChatModel；Evidence、Citation 与 Generation 未参与实验。")
    print(
        f"总耗时：{round((datetime.now(UTC) - started_at).total_seconds() * 1000)} ms；"
        f"回归案例：{len(regressions)}。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
