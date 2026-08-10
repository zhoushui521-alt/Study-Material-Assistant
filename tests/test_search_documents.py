import unittest

from app.chunk_documents import DocumentChunk
from app.search_documents import extract_terms, search_chunks


class SearchChunksTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chunks = [
            DocumentChunk(source="rag.md", index=1, content="RAG 会先检索资料。"),
            DocumentChunk(source="langchain.md", index=1, content="LangChain 可以组装 RAG 流程。"),
        ]

    def test_returns_matching_chunk(self) -> None:
        results = search_chunks("RAG", self.chunks)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].chunk.source, "rag.md")

    def test_returns_empty_list_for_unknown_keyword(self) -> None:
        self.assertEqual(search_chunks("向量数据库", self.chunks), [])

    def test_extracts_terms_from_a_natural_language_question(self) -> None:
        terms = extract_terms("RAG 如何检索资料？")
        self.assertIn("rag", terms)
        self.assertIn("检索", terms)

    def test_can_search_with_a_natural_language_question(self) -> None:
        results = search_chunks("RAG 如何检索资料？", self.chunks)
        self.assertEqual(results[0].chunk.source, "rag.md")


if __name__ == "__main__":
    unittest.main()
