import unittest
from unittest.mock import patch

from app.chunk_documents import DocumentChunk
from app.embedding_client import EmbeddingConfig
from app.vector_search import cosine_similarity, search_by_vector


class VectorSearchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = EmbeddingConfig("test-key", "https://example.com/v1", "test-model", 2)
        self.chunks = [
            DocumentChunk("rag.md", 1, "RAG 资料"),
            DocumentChunk("other.md", 1, "其他资料"),
        ]

    def test_cosine_similarity(self) -> None:
        self.assertAlmostEqual(cosine_similarity([1, 0], [1, 0]), 1.0)
        self.assertAlmostEqual(cosine_similarity([1, 0], [0, 1]), 0.0)

    @patch("app.vector_search.embed_texts")
    def test_ranks_most_similar_chunk_first(self, mock_embed_texts: object) -> None:
        mock_embed_texts.return_value = [[1, 0], [0.9, 0.1], [0, 1]]

        results = search_by_vector("RAG 是什么", self.chunks, self.config)

        self.assertEqual(results[0].chunk.source, "rag.md")
        self.assertGreater(results[0].similarity, results[1].similarity)
