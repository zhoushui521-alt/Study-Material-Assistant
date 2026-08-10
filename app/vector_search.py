"""RAG 第三步：使用 Embedding 向量进行语义检索。"""

import math
import sys
from dataclasses import dataclass

if __package__:
    from app.chunk_documents import DOCUMENTS_DIR, DocumentChunk, build_chunks, load_documents
    from app.embedding_client import EmbeddingConfig, embed_texts
else:
    from chunk_documents import DOCUMENTS_DIR, DocumentChunk, build_chunks, load_documents
    from embedding_client import EmbeddingConfig, embed_texts


@dataclass
class VectorSearchResult:
    """一个按语义相似度排序的资料片段。"""

    chunk: DocumentChunk
    similarity: float


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """计算两个向量夹角的余弦值；越接近 1，语义通常越接近。"""
    if len(left) != len(right):
        raise ValueError("两个向量维度必须一致。")

    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        raise ValueError("不能计算零向量的相似度。")

    dot_product = sum(a * b for a, b in zip(left, right, strict=True))
    return dot_product / (left_norm * right_norm)


def search_by_vector(
    query: str,
    chunks: list[DocumentChunk],
    config: EmbeddingConfig,
    limit: int = 3,
) -> list[VectorSearchResult]:
    """将问题和文本块一起向量化，再返回语义最接近的文本块。"""
    if not query.strip():
        return []
    if not chunks:
        return []

    vectors = embed_texts([query, *(chunk.content for chunk in chunks)], config)
    query_vector, chunk_vectors = vectors[0], vectors[1:]
    results = [
        VectorSearchResult(chunk=chunk, similarity=cosine_similarity(query_vector, vector))
        for chunk, vector in zip(chunks, chunk_vectors, strict=True)
    ]
    return sorted(results, key=lambda result: result.similarity, reverse=True)[:limit]


def main() -> None:
    query = " ".join(sys.argv[1:]).strip()
    if not query:
        query = input("请输入问题（例如：资料太长怎么办？）：").strip()

    chunks = build_chunks(load_documents(DOCUMENTS_DIR))
    results = search_by_vector(query, chunks, EmbeddingConfig.from_environment())
    print(f"找到 {len(results)} 个语义最相关的资料片段：\n")
    for result in results:
        print(f"[{result.chunk.source} · 第 {result.chunk.index} 段 · 相似度 {result.similarity:.4f}]")
        print(f"{result.chunk.content}\n")


if __name__ == "__main__":
    main()
