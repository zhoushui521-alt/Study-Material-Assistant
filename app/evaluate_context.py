"""运行 Stage 3.4 Baseline / Evidence Score Context 受控实验。"""

from __future__ import annotations

import argparse
import subprocess
from hashlib import sha256
from pathlib import Path

from langchain_core.documents import Document

if __package__:
    from app.config import PROJECT_ROOT
    from app.context_evaluation import (
        ContextEvaluationConfig,
        build_context_experiment_report,
        load_context_baseline_report,
        write_context_experiment_report,
    )
    from app.evaluate_retrieval import (
        _git_commit,
        disposable_evaluation_snapshot,
    )
    from app.langchain_store import (
        VECTOR_STORE_DIR,
        close_vector_store,
        open_vector_store,
    )
    from app.retrieval_evaluation import (
        DEFAULT_RETRIEVAL_EVALUATION_PATH,
        DEFAULT_RETRIEVAL_RESULTS_DIR,
        load_retrieval_cases,
    )
else:
    from config import PROJECT_ROOT
    from context_evaluation import (
        ContextEvaluationConfig,
        build_context_experiment_report,
        load_context_baseline_report,
        write_context_experiment_report,
    )
    from evaluate_retrieval import _git_commit, disposable_evaluation_snapshot
    from langchain_store import VECTOR_STORE_DIR, close_vector_store, open_vector_store
    from retrieval_evaluation import (
        DEFAULT_RETRIEVAL_EVALUATION_PATH,
        DEFAULT_RETRIEVAL_RESULTS_DIR,
        load_retrieval_cases,
    )


STAGE3_4_START_COMMIT = "b4757b29264aae54a06994207ac4e0f7be476971"
DEFAULT_BASELINE_REPORT = (
    DEFAULT_RETRIEVAL_RESULTS_DIR
    / "retrieval-evaluation-20260819T104043883459Z-13d190ed.json"
)
IMPLEMENTATION_PATHS = (
    Path("app/context_selector.py"),
    Path("app/context_evaluation.py"),
    Path("app/evaluate_context.py"),
    Path("app/hybrid_search.py"),
    Path("app/langchain_rag.py"),
    Path("app/rag_service.py"),
)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "复用已完成的 Stage 2 Retrieval Trace，只比较 Context Construction A/B。"
        )
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=DEFAULT_RETRIEVAL_EVALUATION_PATH,
        help="复用的 Stage 2 Retrieval Dataset。",
    )
    parser.add_argument(
        "--baseline-report",
        type=Path,
        default=DEFAULT_BASELINE_REPORT,
        help="已完成且包含 Final Context Trace 的 Stage 2 Retrieval Report。",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RETRIEVAL_RESULTS_DIR,
        help="Context Evaluation Report 输出目录。",
    )
    parser.add_argument(
        "--latency-repetitions",
        type=int,
        default=100,
        help="每个案例重复执行 Selector 的次数，只用于稳定测量本地微小延迟。",
    )
    return parser


def _implementation_hashes() -> dict[str, str]:
    hashes = {}
    for relative_path in IMPLEMENTATION_PATHS:
        path = PROJECT_ROOT / relative_path
        hashes[relative_path.as_posix()] = sha256(path.read_bytes()).hexdigest()
    return hashes


def _worktree_status() -> str:
    repository = PROJECT_ROOT.resolve().as_posix()
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={repository}", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
        text=True,
        encoding="utf-8",
        stderr=subprocess.STDOUT,
    )


def _stage_start_is_ancestor(git_head: str) -> bool:
    repository = PROJECT_ROOT.resolve().as_posix()
    result = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repository}",
            "merge-base",
            "--is-ancestor",
            STAGE3_4_START_COMMIT,
            git_head,
        ],
        cwd=PROJECT_ROOT,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def _read_index_documents(vector_store: object) -> list[Document]:
    stored = vector_store.get(include=["documents", "metadatas"])
    return [
        Document(page_content=content, metadata=metadata)
        for content, metadata in zip(
            stored.get("documents") or [],
            stored.get("metadatas") or [],
            strict=True,
        )
        if isinstance(content, str) and isinstance(metadata, dict)
    ]


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        git_head = _git_commit()
        if not _stage_start_is_ancestor(git_head):
            raise RuntimeError(
                "Stage 3.4 起点不是当前 HEAD 的祖先；无法确认实验边界。"
            )
        baseline_report = load_context_baseline_report(args.baseline_report)
        cases = load_retrieval_cases(args.cases)
        config = ContextEvaluationConfig(
            latency_repetitions=args.latency_repetitions,
        )
        with disposable_evaluation_snapshot(VECTOR_STORE_DIR) as snapshot:
            vector_store = open_vector_store(None, snapshot.path)
            try:
                indexed_documents = _read_index_documents(vector_store)
                report = build_context_experiment_report(
                    cases,
                    baseline_report,
                    indexed_documents,
                    config=config,
                    dataset_path=args.cases,
                    baseline_report_path=args.baseline_report,
                    stage3_4_start_commit=STAGE3_4_START_COMMIT,
                    git_head=git_head,
                    current_index_fingerprint=snapshot.original_fingerprint,
                    implementation_hashes=_implementation_hashes(),
                )
            finally:
                close_vector_store(vector_store)
        report["source_state"] = {
            "worktree_clean": not bool(_worktree_status().strip()),
            "note": "Stage implementation is expected to be uncommitted during evaluation.",
        }
        output = write_context_experiment_report(report, args.results_dir)
    except Exception as error:
        print(f"Stage 3.4 Context Evaluation 失败：{error}")
        return 2

    metrics = report["context_metrics"]
    baseline = metrics["baseline"]
    optimized = metrics["optimized"]
    latency = report["latency"]
    print(f"Context Evaluation Report：{output}")
    print(
        "Context Precision："
        f"{baseline['context_precision']:.4f} -> "
        f"{optimized['context_precision']:.4f}"
    )
    print(
        "Final Context Recall："
        f"{baseline['final_context_recall']:.4f} -> "
        f"{optimized['final_context_recall']:.4f}"
    )
    print(
        "Average Context Chunks："
        f"{baseline['average_chunk_count']:.2f} -> "
        f"{optimized['average_chunk_count']:.2f}"
    )
    print(
        "Selector Incremental Latency："
        f"{latency['average_incremental_ms']:.4f} ms"
    )
    print("Query Embedding / Chat / Reranker Calls：0 / 0 / 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
