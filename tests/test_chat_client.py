import json
import unittest
from unittest.mock import patch

from app.chat_client import ChatConfig, build_messages, generate_answer
from app.chunk_documents import DocumentChunk


class FakeResponse:
    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps({"choices": [{"message": {"content": "这是基于资料的回答。"}}]}).encode()


class ChatClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ChatConfig("test-key", "https://example.com/v1", "test-model")
        self.sources = [DocumentChunk("rag.md", 2, "RAG 会先检索资料。")]

    def test_build_messages_keeps_source_label(self) -> None:
        messages = build_messages("RAG 是什么？", self.sources)
        self.assertIn("[rag.md · 第 2 段]", messages[1]["content"])

    @patch("app.chat_client.urlopen")
    def test_returns_answer_from_api_response(self, mock_urlopen: object) -> None:
        mock_urlopen.return_value = FakeResponse()

        answer = generate_answer("RAG 是什么？", self.sources, self.config)

        self.assertEqual(answer, "这是基于资料的回答。")
        request = mock_urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "https://example.com/v1/chat/completions")
