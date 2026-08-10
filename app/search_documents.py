"""RAG 第二步：从文本块中找出与关键词最相关的内容。"""

import re
import sys
from dataclasses import dataclass

# VS Code 点击“运行 Python 文件”时，会直接执行本文件；
# 此时只能从同一 app 文件夹导入。用模块方式运行时则从 app 包导入。
if __package__:
    from app.chunk_documents import DOCUMENTS_DIR, DocumentChunk, build_chunks, load_documents
else:
    from chunk_documents import DOCUMENTS_DIR, DocumentChunk, build_chunks, load_documents


@dataclass
class SearchResult:
    """一次检索命中的文本块及其匹配次数。"""

    chunk: DocumentChunk
    score: int


def extract_terms(query: str) -> list[str]:
    """从问题中提取英文词和连续中文的双字词。

    双字词只是为了让“RAG 为什么要切分文本”这类问题也能匹配到“切分”“文本”。
    它仍然是字面匹配，不理解语义。
    """
    normalized = query.strip().casefold()
    english_terms = re.findall(r"[a-z0-9_+#.-]+", normalized)
    chinese_groups = re.findall(r"[\u4e00-\u9fff]+", normalized)
    chinese_terms = [
        group[index : index + 2]
        for group in chinese_groups
        for index in range(len(group) - 1)
    ]
    return list(dict.fromkeys(english_terms + chinese_terms))


def search_chunks(query: str, chunks: list[DocumentChunk], limit: int = 3) -> list[SearchResult]:
    """使用问题中提取出的词给文本块打分，返回分数最高的结果。

    这是教学用的词重叠检索器，不是向量检索。
    """
    terms = extract_terms(query)
    if not terms:
        return []

    results = [
        SearchResult(
            chunk=chunk,
            score=sum(chunk.content.casefold().count(term) for term in terms),
        )
        for chunk in chunks
    ]
    matched_results = [result for result in results if result.score > 0]
    return sorted(matched_results, key=lambda result: result.score, reverse=True)[:limit]


def main() -> None:
    query = " ".join(sys.argv[1:]).strip()
    if not query:
        query = input("请输入问题（例如：RAG 为什么要切分文本？）：").strip()

    chunks = build_chunks(load_documents(DOCUMENTS_DIR))
    results = search_chunks(query, chunks)

    if not results:
        print(f"没有找到与“{query}”有字面重叠的资料片段。")
        return

    print(f"找到 {len(results)} 个与“{query}”相关的文本块：\n")
    for result in results:
        print(f"[{result.chunk.source} · 第 {result.chunk.index} 段 · 匹配 {result.score} 次]")
        print(f"{result.chunk.content}\n")


if __name__ == "__main__":
    main()
