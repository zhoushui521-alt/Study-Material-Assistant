"""运行 Stage 3.2 Baseline / Hybrid / Cross-Encoder 受控实验。"""

from __future__ import annotations

import argparse
import os
import subprocess
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from time import perf_counter

if __package__:
    from app.config import PROJECT_ROOT
    from app.embedding_client import EmbeddingConfig
    from app.evaluate_retrieval import (
        MAX_QUERY_EMBEDDING_CALLS,
        _git_commit,
        _preflight_snapshot,
        disposable_evaluation_snapshot,
    )
    from app.hybrid_search import build_bm25_index
    from app.langchain_store import (
        VECTOR_STORE_DIR,
        close_vector_store,
        create_langchain_embeddings,
        open_vector_store,
    )
    from app.reranker import FlagEmbeddingCrossEncoderReranker
    from app.reranker_evaluation import (
        RerankerEvaluationConfig,
        build_reranker_report,
        evaluate_reranker_cases,
        write_reranker_report,
    )
    from app.retrieval_evaluation import (
        DEFAULT_RETRIEVAL_EVALUATION_PATH,
        DEFAULT_RETRIEVAL_RESULTS_DIR,
        _dataset_metadata,
        load_retrieval_cases,
    )
else:
    from config import PROJECT_ROOT
    from embedding_client import EmbeddingConfig
    from evaluate_retrieval import (
        MAX_QUERY_EMBEDDING_CALLS,
        _git_commit,
        _preflight_snapshot,
        disposable_evaluation_snapshot,
    )
    from hybrid_search import build_bm25_index
    from langchain_store import (
        VECTOR_STORE_DIR,
        close_vector_store,
        create_langchain_embeddings,
        open_vector_store,
    )
    from reranker import FlagEmbeddingCrossEncoderReranker
    from reranker_evaluation import (
        RerankerEvaluationConfig,
        build_reranker_report,
        evaluate_reranker_cases,
        write_reranker_report,
    )
    from retrieval_evaluation import (
        DEFAULT_RETRIEVAL_EVALUATION_PATH,
        DEFAULT_RETRIEVAL_RESULTS_DIR,
        _dataset_metadata,
        load_retrieval_cases,
    )


STAGE3_2_START_COMMIT = "2d526c5efdf7d9bb92fef4c53fdadc0856d4cbd9"
RERANKER_MODEL_ID = "BAAI/bge-reranker-base"
RERANKER_MODEL_REVISION = "2cfc18c9415c912f9d8155881c133215df768a70"
RERANKER_MODEL_SHA256 = (
    "ced967c45fd1902eb92716c9ceeca7c95a936770ea9db611f5a841b926e33fbd"
)
RERANKER_MODEL_DIR = PROJECT_ROOT / "data" / "models" / "bge-reranker-base"
RERANKER_MAX_LENGTH = 512
RERANKER_BATCH_SIZE = 16
RERANKER_DEVICE = "cpu"
MODEL_ALLOW_PATTERNS = (
    "config.json",
    "model.safetensors",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
IMPLEMENTATION_PATHS = (
    Path("app/evaluate_reranker.py"),
    Path("app/reranker.py"),
    Path("app/reranker_evaluation.py"),
)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="比较 Stage 2 Baseline、Stage 3.1 Hybrid 与 Cross-Encoder。"
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=DEFAULT_RETRIEVAL_EVALUATION_PATH,
        help="复用的 Stage 2 Retrieval Dataset。",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RETRIEVAL_RESULTS_DIR,
        help="Evaluation Report 输出目录。",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=RERANKER_MODEL_DIR,
        help="项目内本地 Reranker 模型目录。",
    )
    parser.add_argument(
        "--confirm-query-embedding-cost",
        action="store_true",
        help="确认每个 Case 调用一次真实 Query Embedding。",
    )
    parser.add_argument(
        "--confirm-model-download-and-local-inference",
        action="store_true",
        help="确认下载本地模型并执行 CPU Cross-Encoder 推理。",
    )
    return parser


def _source_state(git_commit: str) -> dict[str, object]:
    repository = PROJECT_ROOT.resolve().as_posix()
    status = subprocess.check_output(
        ["git", "-c", f"safe.directory={repository}", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
        text=True,
        encoding="utf-8",
        stderr=subprocess.STDOUT,
    )
    combined = sha256()
    file_hashes: dict[str, str] = {}
    for relative_path in IMPLEMENTATION_PATHS:
        content = (PROJECT_ROOT / relative_path).read_bytes()
        normalized_path = relative_path.as_posix()
        file_hashes[normalized_path] = sha256(content).hexdigest()
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


def _directory_size(directory: Path) -> int:
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_reranker_model(model_directory: Path) -> tuple[str, int]:
    """固定远端 revision 后只下载推理必需文件到项目目录。"""
    model_path = model_directory / "model.safetensors"
    if model_path.is_file():
        if _file_sha256(model_path) != RERANKER_MODEL_SHA256:
            raise RuntimeError("本地 Reranker 权重 SHA-256 与固定模型不一致。")
        return RERANKER_MODEL_REVISION, _directory_size(model_directory)

    # Windows 上 Xet 曾出现长时间无任何落盘进度；标准 HTTP 支持本地 incomplete 续传，
    # 进度与失败状态也更容易审计。只影响模型下载，不改变模型内容或 revision。
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=RERANKER_MODEL_ID,
        revision=RERANKER_MODEL_REVISION,
        local_dir=model_directory,
        allow_patterns=list(MODEL_ALLOW_PATTERNS),
        max_workers=2,
    )
    required = ("config.json", "model.safetensors", "tokenizer_config.json")
    missing = [name for name in required if not (model_directory / name).is_file()]
    if missing:
        raise RuntimeError("Reranker 模型下载不完整，缺少：" + "、".join(missing))
    if _file_sha256(model_path) != RERANKER_MODEL_SHA256:
        raise RuntimeError("下载后的 Reranker 权重 SHA-256 校验失败。")
    return RERANKER_MODEL_REVISION, _directory_size(model_directory)


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        cases = load_retrieval_cases(args.cases)
    except Exception as error:
        print(f"Stage 3.2 Cross-Encoder Experiment 启动失败：{error}")
        return 2
    if len(cases) > MAX_QUERY_EMBEDDING_CALLS:
        print(
            "Stage 3.2 Cross-Encoder Experiment 启动失败："
            f"{len(cases)} 个案例超过 {MAX_QUERY_EMBEDDING_CALLS} 次 Query Embedding 上限。"
        )
        return 2
    if not (
        args.confirm_query_embedding_cost
        and args.confirm_model_download_and_local_inference
    ):
        print("未执行真实 Stage 3.2 Cross-Encoder Experiment。")
        print(f"案例数：{len(cases)}")
        print(f"预计 Query Embedding 次数：{len(cases)}")
        print("Reranker Candidate Pool：每个 Case 最多 20 个，全部本地 CPU Pair Scoring。")
        print(f"模型：{RERANKER_MODEL_ID}；Chat/LLM 调用：0。")
        print("需要同时确认 Query Embedding 费用与本地模型下载/推理。")
        return 2

    vector_store = None
    started_at = datetime.now(UTC)
    try:
        git_commit = _git_commit()
        source_state = _source_state(git_commit)
        embedding_config = EmbeddingConfig.from_environment()
        config = RerankerEvaluationConfig()

        download_started = perf_counter()
        model_revision, model_size_bytes = download_reranker_model(args.model_dir)
        model_download_elapsed = round((perf_counter() - download_started) * 1000)
        load_started = perf_counter()
        reranker = FlagEmbeddingCrossEncoderReranker.from_local_model(
            args.model_dir,
            model_id=RERANKER_MODEL_ID,
            model_revision=model_revision,
            max_length=RERANKER_MAX_LENGTH,
            batch_size=RERANKER_BATCH_SIZE,
            device=RERANKER_DEVICE,
        )
        model_load_elapsed = round((perf_counter() - load_started) * 1000)

        with disposable_evaluation_snapshot(VECTOR_STORE_DIR) as snapshot:
            index_status = _preflight_snapshot(snapshot.path, embedding_config, cases)
            embeddings = create_langchain_embeddings(embedding_config)
            vector_store = open_vector_store(
                embeddings,
                persist_directory=snapshot.path,
            )
            try:
                bm25_index = build_bm25_index(vector_store)
                print(f"开始 Stage 3.2 Cross-Encoder Experiment：{len(cases)} 个案例。")
                print(
                    f"索引状态：{index_status.value}；A/B/C 复用同一 disposable snapshot。"
                )
                results = evaluate_reranker_cases(
                    cases,
                    vector_store,
                    bm25_index,
                    reranker,
                    config,
                )
            finally:
                close_vector_store(vector_store)
                vector_store = None
            report = build_reranker_report(
                results,
                config=config,
                reranker=reranker,
                git_commit=git_commit,
                stage3_2_start_commit=STAGE3_2_START_COMMIT,
                dataset_metadata=_dataset_metadata(args.cases),
                embedding_model=embedding_config.model,
                query_embedding_calls=len(cases),
                validation_level="local_real_cross_encoder",
                index_fingerprint=snapshot.original_fingerprint,
                generated_at=datetime.now(UTC),
            )
            report["source_state"] = source_state
            report["model_runtime"] = {
                "model_id": RERANKER_MODEL_ID,
                "revision": model_revision,
                "model_directory_size_bytes": model_size_bytes,
                "device": RERANKER_DEVICE,
                "torch_precision": "fp32",
                "max_length": RERANKER_MAX_LENGTH,
                "batch_size": RERANKER_BATCH_SIZE,
                "download_and_verify_ms": model_download_elapsed,
                "load_ms": model_load_elapsed,
            }
        report_path = write_reranker_report(report, args.results_dir)
    except Exception as error:
        print(f"Stage 3.2 Cross-Encoder Experiment 执行失败：{error}")
        return 2
    finally:
        if vector_store is not None:
            close_vector_store(vector_store)

    print(f"Stage 3.2 Reranker Evaluation Report：{report_path}")
    print("原始 Index：运行前后 filesystem SHA-256 指纹一致。")
    print("本次 Chat/LLM 调用：0；生产 Retrieval 接线未修改。")
    print(
        f"总耗时：{round((datetime.now(UTC) - started_at).total_seconds() * 1000)} ms。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
