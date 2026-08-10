import json
import unittest
from unittest.mock import patch

from app.embedding_client import EmbeddingConfig, embed_texts


class FakeResponse:
    def __init__(self, body: dict) -> None:
        self.body = body

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.body).encode("utf-8")


class EmbeddingClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = EmbeddingConfig(
            api_key="test-key",
            base_url="https://example.com/v1",
            model="test-embedding",
            dimensions=3,
        )

    @patch("app.embedding_client.urlopen")
    def test_returns_vectors_in_input_order(self, mock_urlopen: object) -> None:
        mock_urlopen.return_value = FakeResponse(
            {"data": [{"index": 1, "embedding": [4, 5, 6]}, {"index": 0, "embedding": [1, 2, 3]}]}
        )

        vectors = embed_texts(["第一条", "第二条"], self.config)

        self.assertEqual(vectors, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        request = mock_urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "https://example.com/v1/embeddings")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")

    def test_rejects_empty_text(self) -> None:
        with self.assertRaises(ValueError):
            embed_texts(["  "], self.config)
