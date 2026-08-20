"""运行 Stage 3.3 Current / Structure-aware Chunking 受控实验。"""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from dataclasses import replace
from datetime import UTC, datetime
from hashlib import sha256
from math import ceil
from pathlib import Path

if __package__:
    from app.chunk_documents import (
        CHUNKER_VERSION,
        DOCUMENTS_DIR,
        build_chunks,
        load_material_units,
    )
    from app.chunking_evaluation import (
        ChunkingStrategyDescriptor,
        build_chunking_report,
        citation_localization_summary,
        remap_cases_to_chunks,
        write_chunking_report,
    )
    from app.config import PROJECT_ROOT
    from app.embedding_client import EmbeddingConfig
    from app.evidence import evidence_from_document
    from app.evaluate_retrieval import (
        MAX_QUERY_EMBEDDING_CALLS,
        _directory_fingerprint,
        _git_commit,
        _preflight_snapshot,
        _retrieval_config,
        disposable_evaluation_snapshot,
    )
    from app.index_manifest import load_index_manifest
    from app.langchain_store import (
        BAILIAN_EMBEDDING_BATCH_SIZE,
        VECTOR_STORE_DIR,
        close_vector_store,
        create_langchain_embeddings,
        open_vector_store,
        rebuild_vector_store,
        runtime_index_config,
    )
    from app.retrieval_evaluation import (
        DEFAULT_RETRIEVAL_EVALUATION_PATH,
        DEFAULT_RETRIEVAL_RESULTS_DIR,
        _dataset_metadata,
        evaluate_retrieval_cases,
        load_retrieval_cases,
    )
    from app.structure_aware_chunking import (
        DEFAULT_MAX_CHARS,
        DEFAULT_OVERLAP_CHARS,
        STRUCTURE_AWARE_CHUNKER_VERSION,
        StructureAwareChunkingConfig,
        build_structure_aware_chunks,
    )
else:
    from chunk_documents import (
        CHUNKER_VERSION,
        DOCUMENTS_DIR,
        build_chunks,
        load_material_units,
    )
    from chunking_evaluation import (
        ChunkingStrategyDescriptor,
        build_chunking_report,
        citation_localization_summary,
        remap_cases_to_chunks,
        write_chunking_report,
    )
    from config import PROJECT_ROOT
    from embedding_client import EmbeddingConfig
    from evidence import evidence_from_document
    from evaluate_retrieval import (
        MAX_QUERY_EMBEDDING_CALLS,
        _directory_fingerprint,
        _git_commit,
        _preflight_snapshot,
        _retrieval_config,
        disposable_evaluation_snapshot,
    )
    from index_manifest import load_index_manifest
    from langchain_store import (
        BAILIAN_EMBEDDING_BATCH_SIZE,
        VECTOR_STORE_DIR,
        close_vector_store,
        create_langchain_embeddings,
        open_vector_store,
        rebuild_vector_store,
        runtime_index_config,
    )
    from retrieval_evaluation import (
        DEFAULT_RETRIEVAL_EVALUATION_PATH,
        DEFAULT_RETRIEVAL_RESULTS_DIR,
        _dataset_metadata,
        evaluate_retrieval_cases,
        load_retrieval_cases,
    )
    from structure_aware_chunking import (
        DEFAULT_MAX_CHARS,
        DEFAULT_OVERLAP_CHARS,
        STRUCTURE_AWARE_CHUNKER_VERSION,
        StructureAwareChunkingConfig,
        build_structure_aware_chunks,
    )


STAGE3_3_START_COMMIT = "cc6e344363f36e542f1f5a52cec3ada7731f6969"
IMPLEMENTATION_PATHS = (
    Path("app/evaluate_chunking.py"),
    Path("app/chunking_evaluation.py"),
    Path("app/structure_aware_chunking.py"),
)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="固定 Retrieval Pipeline，仅比较 Current 与 Structure-aware Chunking。"
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
        "--confirm-controlled-index-embedding-cost",
        action="store_true",
        help="确认用同一配置为 A/B 临时索引调用真实 Embedding API。",
    )
    parser.add_argument(
        "--confirm-query-embedding-cost",
        action="store_true",
        help="确认 A/B 各执行一轮真实 Query Embedding。",
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


def _chunk_signature(chunks: list) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            chunk.chunk_id,
            chunk.content_hash,
            chunk.material_id,
            chunk.source,
            chunk.page,
            chunk.section,
            chunk.index,
            chunk.chunker_version,
        )
        for chunk in chunks
    )


def _preflight_output_directory(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    if not directory.is_dir() or not os.access(directory, os.W_OK):
        raise RuntimeError(f"Evaluation Report 目录不可写：{directory}。")


def _verify_baseline_index(snapshot_path: Path, baseline_chunks: list) -> None:
    vector_store = open_vector_store(None, persist_directory=snapshot_path)
    try:
        stored = vector_store.get(include=["documents", "metadatas"])
    finally:
        close_vector_store(vector_store)
    from langchain_core.documents import Document

    indexed_documents = [
        Document(page_content=content, metadata=metadata)
        for content, metadata in zip(
            stored.get("documents") or [],
            stored.get("metadatas") or [],
            strict=True,
        )
        if isinstance(content, str) and isinstance(metadata, dict)
    ]
    indexed_ids = [
        evidence_from_document(document, "S1").chunk_id
        for document in indexed_documents
    ]
    expected_ids = [chunk.chunk_id for chunk in baseline_chunks]
    if len(indexed_ids) != len(set(indexed_ids)):
        raise RuntimeError("Current Index 包含重复 Chunk ID。")
    if set(indexed_ids) != set(expected_ids):
        raise RuntimeError("Current Index 与当前生产 Chunk Pipeline 的 Chunk ID 不一致。")


def _baseline_descriptor() -> ChunkingStrategyDescriptor:
    return ChunkingStrategyDescriptor(
        name="current_fixed_character",
        chunker_version=CHUNKER_VERSION,
        max_chars=180,
        overlap_chars=0,
        structure_order=("normalized_text", "fixed_character_window"),
    )


def _experimental_descriptor() -> ChunkingStrategyDescriptor:
    return ChunkingStrategyDescriptor(
        name="structure_aware",
        chunker_version=STRUCTURE_AWARE_CHUNKER_VERSION,
        max_chars=DEFAULT_MAX_CHARS,
        overlap_chars=DEFAULT_OVERLAP_CHARS,
        structure_order=(
            "heading",
            "paragraph",
            "list_or_code_lines",
            "sentence",
            "word",
            "character",
        ),
    )


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        cases = load_retrieval_cases(args.cases)
        if len(cases) > MAX_QUERY_EMBEDDING_CALLS:
            raise RuntimeError(
                f"{len(cases)} 个案例超过单策略 {MAX_QUERY_EMBEDDING_CALLS} 次上限。"
            )
        units = load_material_units(DOCUMENTS_DIR)
        baseline_chunks = build_chunks(units)
        structure_config = StructureAwareChunkingConfig()
        experimental_chunks = build_structure_aware_chunks(units, structure_config)
        repeated_chunks = build_structure_aware_chunks(units, structure_config)
        deterministic_rebuild = (
            _chunk_signature(experimental_chunks) == _chunk_signature(repeated_chunks)
        )
        if not deterministic_rebuild:
            raise RuntimeError("Structure-aware Chunker 重复构建结果不稳定。")
        experimental_cases = remap_cases_to_chunks(
            cases,
            experimental_chunks,
            baseline_chunks=baseline_chunks,
        )
    except Exception as error:
        print(f"Stage 3.3 Chunking Experiment 启动失败：{error}")
        return 2

    query_embedding_calls = len(cases) * 2
    baseline_index_embedding_texts = len(baseline_chunks)
    experimental_index_embedding_texts = len(experimental_chunks)
    baseline_index_embedding_batches = ceil(
        baseline_index_embedding_texts / BAILIAN_EMBEDDING_BATCH_SIZE
    )
    experimental_index_embedding_batches = ceil(
        experimental_index_embedding_texts / BAILIAN_EMBEDDING_BATCH_SIZE
    )
    if not (
        args.confirm_controlled_index_embedding_cost
        and args.confirm_query_embedding_cost
    ):
        print("未执行真实 Stage 3.3 Chunking Experiment。")
        print(
            f"Chunk 数：Current={len(baseline_chunks)}，"
            f"Structure-aware={experimental_index_embedding_texts}。"
        )
        print(
            "A/B 临时索引 Embedding："
            f"{baseline_index_embedding_texts} + "
            f"{experimental_index_embedding_texts} = "
            f"{baseline_index_embedding_texts + experimental_index_embedding_texts} 条文本，"
            f"预计 {baseline_index_embedding_batches + experimental_index_embedding_batches} 个批次。"
        )
        print(f"A/B Query Embedding：{query_embedding_calls} 次；Chat/LLM：0 次。")
        print("生产 Index：只复制与指纹校验；候选 Index 仅写入临时目录。")
        print("需要同时确认实验索引 Embedding 与 Query Embedding 费用。")
        return 2

    started_at = datetime.now(UTC)
    baseline_store = None
    experimental_store = None
    try:
        _preflight_output_directory(args.results_dir)
        git_commit = _git_commit()
        if git_commit != STAGE3_3_START_COMMIT:
            raise RuntimeError(
                "当前 HEAD 不是 study-material-v2-stage3-part2 对应基线，拒绝归因实验结果。"
            )
        source_state = _source_state(git_commit)
        embedding_config = EmbeddingConfig.from_environment()
        retrieval_config = _retrieval_config()
        with disposable_evaluation_snapshot(VECTOR_STORE_DIR) as snapshot:
            index_status = _preflight_snapshot(snapshot.path, embedding_config, cases)
            _verify_baseline_index(snapshot.path, baseline_chunks)
            embeddings = create_langchain_embeddings(embedding_config)
            with tempfile.TemporaryDirectory(
                prefix="study-material-chunking-eval-"
            ) as directory:
                baseline_index_path = Path(directory) / "baseline_vector_store"
                experimental_index_path = Path(directory) / "vector_store"
                baseline_runtime_config = runtime_index_config(embedding_config)
                experimental_runtime_config = replace(
                    baseline_runtime_config,
                    chunker_version=STRUCTURE_AWARE_CHUNKER_VERSION,
                )
                baseline_store = rebuild_vector_store(
                    baseline_chunks,
                    embeddings,
                    persist_directory=baseline_index_path,
                    runtime_config=baseline_runtime_config,
                )
                try:
                    baseline_ids = baseline_store.get(include=[]).get("ids") or []
                    if len(baseline_ids) != len(baseline_chunks):
                        raise RuntimeError("A 临时 Index 的记录数与当前 Chunk 数不一致。")
                    baseline_results = evaluate_retrieval_cases(
                        cases,
                        baseline_store,
                        retrieval_config,
                    )
                finally:
                    close_vector_store(baseline_store)
                    baseline_store = None
                baseline_manifest = load_index_manifest(baseline_index_path)
                if (
                    baseline_manifest is None
                    or baseline_manifest.chunker_version != CHUNKER_VERSION
                ):
                    raise RuntimeError("A 临时 Index Manifest 未记录当前 Chunker 身份。")
                baseline_index_fingerprint = _directory_fingerprint(
                    baseline_index_path
                )
                experimental_store = rebuild_vector_store(
                    experimental_chunks,
                    embeddings,
                    persist_directory=experimental_index_path,
                    runtime_config=experimental_runtime_config,
                )
                try:
                    stored_ids = experimental_store.get(include=[]).get("ids") or []
                    if len(stored_ids) != len(experimental_chunks):
                        raise RuntimeError("实验 Index 的记录数与候选 Chunk 数不一致。")
                    experimental_results = evaluate_retrieval_cases(
                        experimental_cases,
                        experimental_store,
                        retrieval_config,
                    )
                finally:
                    close_vector_store(experimental_store)
                    experimental_store = None
                manifest = load_index_manifest(experimental_index_path)
                if (
                    manifest is None
                    or manifest.chunker_version != STRUCTURE_AWARE_CHUNKER_VERSION
                ):
                    raise RuntimeError("实验 Index Manifest 未记录候选 Chunker 身份。")
                experimental_index_fingerprint = _directory_fingerprint(
                    experimental_index_path
                )

                report = build_chunking_report(
                    baseline_results,
                    experimental_results,
                    baseline_descriptor=_baseline_descriptor(),
                    experimental_descriptor=_experimental_descriptor(),
                    retrieval_config=retrieval_config,
                    baseline_chunks=baseline_chunks,
                    experimental_chunks=experimental_chunks,
                    baseline_localization=citation_localization_summary(
                        cases,
                        baseline_chunks,
                    ),
                    experimental_localization=citation_localization_summary(
                        experimental_cases,
                        experimental_chunks,
                    ),
                    deterministic_rebuild=deterministic_rebuild,
                    git_commit=git_commit,
                    stage3_3_start_commit=STAGE3_3_START_COMMIT,
                    dataset_metadata=_dataset_metadata(args.cases),
                    embedding_model=embedding_config.model,
                    query_embedding_calls=query_embedding_calls,
                    baseline_index_embedding_texts=baseline_index_embedding_texts,
                    baseline_index_embedding_batches=baseline_index_embedding_batches,
                    experimental_index_embedding_texts=experimental_index_embedding_texts,
                    experimental_index_embedding_batches=experimental_index_embedding_batches,
                    production_index_fingerprint=snapshot.original_fingerprint,
                    baseline_index_fingerprint=baseline_index_fingerprint,
                    experimental_index_fingerprint=experimental_index_fingerprint,
                    generated_at=datetime.now(UTC),
                    validation_level="local_real_structure_aware_chunking",
                )
                report["source_state"] = source_state
                report["baseline_index_status"] = index_status.value
        report_path = write_chunking_report(report, args.results_dir)
    except Exception as error:
        print(f"Stage 3.3 Chunking Experiment 执行失败：{error}")
        return 2
    finally:
        if baseline_store is not None:
            close_vector_store(baseline_store)
        if experimental_store is not None:
            close_vector_store(experimental_store)

    print(f"Stage 3.3 Chunking Report：{report_path}")
    print("原始 Index：运行前后 filesystem SHA-256 指纹一致。")
    print("候选 Index：仅存在于临时目录，运行结束后自动清理。")
    print("本次未调用 ChatModel；Citation Support 未自动评估。")
    print(
        f"总耗时：{round((datetime.now(UTC) - started_at).total_seconds() * 1000)} ms。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
