"""运行 Stage 2 Retrieval Benchmark；只调用 Query Embedding，不调用 ChatModel。"""

import argparse
import json
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

from langchain_core.documents import Document

if __package__:
    from app.embedding_client import EmbeddingConfig
    from app.index_manifest import check_index_compatibility
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
        RetrievalEvaluationRun,
        evaluate_retrieval_cases,
        find_unresolved_gold_mappings,
        load_retrieval_cases,
    )
else:
    from embedding_client import EmbeddingConfig
    from index_manifest import check_index_compatibility
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
        RetrievalEvaluationRun,
        evaluate_retrieval_cases,
        find_unresolved_gold_mappings,
        load_retrieval_cases,
    )

if __package__:
    from app.config import PROJECT_ROOT
    from app.hybrid_search import DEFAULT_CANDIDATE_LIMIT, DEFAULT_KEYWORD_WEIGHT
else:
    from config import PROJECT_ROOT
    from hybrid_search import DEFAULT_CANDIDATE_LIMIT, DEFAULT_KEYWORD_WEIGHT


STAGE2_START_COMMIT = "b99f4a73de123c1428c7a88320765186cf3c727b"
MAX_QUERY_EMBEDDING_CALLS = 10


@dataclass(frozen=True)
class EvaluationIndexSnapshot:
    path: Path
    original_fingerprint: str


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
    """用单次命令显式信任当前 checkout，不修改全局 Git 配置。"""
    repository = PROJECT_ROOT.resolve().as_posix()
    commit = subprocess.check_output(
        ["git", "-c", f"safe.directory={repository}", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        text=True,
        encoding="utf-8",
        stderr=subprocess.STDOUT,
    ).strip()
    if len(commit) != 40 or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise RuntimeError("无法可靠获得当前 Git commit。")
    return commit


def _directory_fingerprint(directory: Path) -> str:
    """对目录结构与全部文件内容生成稳定 SHA-256 指纹。"""
    if not directory.is_dir():
        raise RuntimeError(f"Index 目录不存在：{directory}。")
    digest = sha256()
    entries = sorted(
        directory.rglob("*"),
        key=lambda item: item.relative_to(directory).as_posix(),
    )
    for entry in entries:
        relative = entry.relative_to(directory).as_posix().encode("utf-8")
        if entry.is_symlink():
            digest.update(b"L\0" + relative + b"\0")
            digest.update(str(entry.readlink()).encode("utf-8"))
        elif entry.is_dir():
            digest.update(b"D\0" + relative + b"\0")
        elif entry.is_file():
            digest.update(b"F\0" + relative + b"\0")
            with entry.open("rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(block)
        else:
            raise RuntimeError(f"Index 包含无法指纹化的目录项：{entry}。")
    return digest.hexdigest()


@contextmanager
def disposable_evaluation_snapshot(
    original_directory: Path,
) -> Iterator[EvaluationIndexSnapshot]:
    """复制原始 Index 供评测使用，并在退出时验证原目录字节级不变。"""
    original_fingerprint = _directory_fingerprint(original_directory)
    with tempfile.TemporaryDirectory(prefix="study-material-retrieval-eval-") as directory:
        snapshot_path = Path(directory) / "vector_store"
        shutil.copytree(original_directory, snapshot_path, copy_function=shutil.copy2)
        if _directory_fingerprint(snapshot_path) != original_fingerprint:
            raise RuntimeError("Evaluation snapshot 与原始 Index 的初始指纹不一致。")
        try:
            yield EvaluationIndexSnapshot(snapshot_path, original_fingerprint)
        finally:
            if _directory_fingerprint(original_directory) != original_fingerprint:
                raise RuntimeError("Evaluation 期间原始 Index 的 filesystem 指纹发生变化。")


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


def _retrieval_config() -> RetrievalConfig:
    return RetrievalConfig(
        retrieval_limit=RETRIEVAL_LIMIT,
        candidate_limit=DEFAULT_CANDIDATE_LIMIT,
        relevance_threshold=RELEVANCE_THRESHOLD,
        vector_weight=1.0 - DEFAULT_KEYWORD_WEIGHT,
        keyword_weight=DEFAULT_KEYWORD_WEIGHT,
        adjacent_window=ADJACENT_WINDOW,
        context_limit=CONTEXT_LIMIT,
    )


def _preflight_snapshot(
    snapshot_path: Path,
    embedding_config: EmbeddingConfig,
    cases: tuple,
) -> object:
    """不创建 Embedding client，先验证 Index、配置与 Gold Mapping。"""
    vector_store = open_vector_store(None, persist_directory=snapshot_path)
    try:
        stored = vector_store.get(include=[])
        has_records = bool(stored.get("ids"))
        if not has_records:
            raise RuntimeError("当前索引没有记录。")
        index_status = check_index_compatibility(
            snapshot_path,
            runtime_index_config(embedding_config),
            has_records=has_records,
            access="read",
        )
        _, indexed_documents = _read_index_state(vector_store)
        unresolved = find_unresolved_gold_mappings(cases, indexed_documents)
        if unresolved:
            details = "、".join(unresolved)
            raise RuntimeError(f"Current Chunk Mapping 无法定位：{details}")
        return index_status
    finally:
        close_vector_store(vector_store)


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        cases = load_retrieval_cases(args.cases)
    except Exception as error:
        print(f"Retrieval Evaluation 启动失败：{error}")
        return 2
    if len(cases) > MAX_QUERY_EMBEDDING_CALLS:
        print(
            "Retrieval Evaluation 启动失败："
            f"{len(cases)} 个案例会超过 {MAX_QUERY_EMBEDDING_CALLS} 次 Query Embedding 上限。"
        )
        return 2
    if not args.confirm_query_embedding_cost:
        print("未执行真实 Retrieval Benchmark。")
        print(f"案例数：{len(cases)}")
        print(f"预计 Query Embedding 次数：{len(cases)}")
        print("索引访问：Evaluation 只查询 disposable snapshot；原始 Index 仅做指纹校验。")
        print("Chat/LLM 调用：0")
        print("确认费用后请增加参数：--confirm-query-embedding-cost")
        return 2

    started_at = datetime.now(UTC)
    run: RetrievalEvaluationRun | None = None
    vector_store = None
    try:
        git_commit = _git_commit()
        embedding_config = EmbeddingConfig.from_environment()
        config = _retrieval_config()
        with disposable_evaluation_snapshot(VECTOR_STORE_DIR) as snapshot:
            index_status = _preflight_snapshot(snapshot.path, embedding_config, cases)
            run = RetrievalEvaluationRun.create(
                dataset_path=args.cases,
                output_directory=args.results_dir,
                config=config,
                git_commit=git_commit,
                stage2_start_commit=STAGE2_START_COMMIT,
                started_at=started_at,
                validation_level="local_real_retrieval",
                embedding_model=embedding_config.model,
                original_index_fingerprint=snapshot.original_fingerprint,
            )
            print(f"开始 Retrieval Evaluation：{len(cases)} 个案例。")
            print(
                f"索引状态：{index_status.value}；只查询 disposable snapshot，"
                "不迁移、不补 Manifest。"
            )
            embeddings = create_langchain_embeddings(embedding_config)
            vector_store = open_vector_store(
                embeddings,
                persist_directory=snapshot.path,
            )
            try:
                for case in cases:
                    result = evaluate_retrieval_cases((case,), vector_store, config)[0]
                    run.persist_case_result(result)
                    status = result.failure_category or "ok"
                    print(f"[{status}] {result.case.case_id}（已持久化）")
            finally:
                close_vector_store(vector_store)
                vector_store = None
        report = run.finalize(original_index_unchanged=True)
    except Exception as error:
        print(f"Retrieval Evaluation 执行失败：{error}")
        if run is not None:
            print(f"可恢复 Retrieval run state：{run.path}")
        return 2
    finally:
        if vector_store is not None:
            close_vector_store(vector_store)

    failures = [
        result
        for result in report["per_case_results"]
        if result.get("failure_category")
    ]
    print(f"Retrieval Report：{run.path}")
    print("原始 Index：运行前后 filesystem SHA-256 指纹一致。")
    print("本次未调用 ChatModel；Citation Coverage/Support 未自动评估。")
    print(f"总耗时：{round((datetime.now(UTC) - started_at).total_seconds() * 1000)} ms")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
