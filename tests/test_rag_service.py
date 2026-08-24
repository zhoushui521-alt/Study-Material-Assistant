import json
import unittest
from unittest.mock import Mock, patch

from langchain_core.documents import Document

from app.context_selector import EvidenceScoreContextSelector
from app.langchain_rag import RAGAnswer
from app.langchain_store import VECTOR_STORE_DIR
from app.index_manifest import IndexCompatibilityStatus
from app.observability import OBSERVABILITY_LOGGER_NAME, observation_context
from app.rag_service import (
    ADJACENT_WINDOW,
    CONTEXT_LIMIT,
    RELEVANCE_THRESHOLD,
    RETRIEVAL_LIMIT,
    RAGService,
    RAGServiceInitializationError,
    create_rag_service,
)


class RAGServiceTests(unittest.TestCase):
    @patch(
        "app.rag_service.check_index_compatibility",
        return_value=IndexCompatibilityStatus.COMPATIBLE,
    )
    @patch("app.rag_service.HybridRetriever")
    @patch("app.rag_service.open_vector_store")
    @patch("app.rag_service.create_langchain_embeddings")
    @patch("app.rag_service.EmbeddingConfig.from_environment")
    def test_builds_service_with_stage2_baseline_retriever(
        self,
        embedding_config_from_environment: Mock,
        create_embeddings: Mock,
        open_store: Mock,
        hybrid_retriever: Mock,
        check_index_compatibility: Mock,
    ) -> None:
        vector_store = Mock()
        vector_store.get.return_value = {"ids": ["document-id"]}
        open_store.return_value = vector_store
        model_gateway = Mock()

        service = create_rag_service(
            model_gateway=model_gateway,
            user_id="11111111-1111-4111-8111-111111111111",
        )

        embedding_config_from_environment.assert_called_once_with()
        create_embeddings.assert_called_once_with(
            embedding_config_from_environment.return_value
        )
        open_store.assert_called_once_with(create_embeddings.return_value, VECTOR_STORE_DIR)
        model_gateway.create_chat_model.assert_called_once_with(
            "11111111-1111-4111-8111-111111111111"
        )
        hybrid_retriever.assert_called_once_with(
            vector_store=vector_store,
            limit=RETRIEVAL_LIMIT,
            relevance_threshold=RELEVANCE_THRESHOLD,
            adjacent_window=ADJACENT_WINDOW,
            context_limit=CONTEXT_LIMIT,
        )
        self.assertIs(service.vector_store, vector_store)
        self.assertIs(service.retriever, hybrid_retriever.return_value)
        self.assertIsInstance(service.context_selector, EvidenceScoreContextSelector)
        self.assertEqual(service.context_selector.seed_count, RETRIEVAL_LIMIT)
        self.assertEqual(service.context_selector.adjacent_per_seed, 1)
        self.assertIsNotNone(service.rag_chain)
        self.assertEqual(service.index_status, IndexCompatibilityStatus.COMPATIBLE)
        check_index_compatibility.assert_called_once()

    @patch("app.rag_service.open_vector_store")
    @patch("app.rag_service.create_langchain_embeddings")
    @patch("app.rag_service.EmbeddingConfig.from_environment")
    def test_empty_index_fails_and_releases_vector_store(
        self,
        embedding_config_from_environment: Mock,
        create_embeddings: Mock,
        open_store: Mock,
    ) -> None:
        vector_store = Mock()
        vector_store.get.return_value = {"ids": []}
        open_store.return_value = vector_store

        with self.assertRaisesRegex(RAGServiceInitializationError, "没有资料"):
            create_rag_service()

        vector_store._client.close.assert_called_once_with()

    @patch("app.rag_service.open_vector_store")
    @patch("app.rag_service.create_langchain_embeddings")
    @patch("app.rag_service.EmbeddingConfig.from_environment")
    def test_cleanup_failure_does_not_hide_initialization_error(
        self,
        embedding_config_from_environment: Mock,
        create_embeddings: Mock,
        open_store: Mock,
    ) -> None:
        vector_store = Mock()
        vector_store.get.return_value = {"ids": []}
        vector_store._client.close.side_effect = RuntimeError("close failed")
        open_store.return_value = vector_store

        with self.assertLogs(OBSERVABILITY_LOGGER_NAME, level="ERROR") as logs:
            with self.assertRaisesRegex(RAGServiceInitializationError, "没有资料"):
                create_rag_service()

        payload = json.loads(logs.records[-1].getMessage())
        self.assertEqual(payload["event"], "rag_initialization_cleanup_failed")
        self.assertEqual(payload["error_type"], "RuntimeError")

    @patch("app.rag_service.create_rag_chain")
    def test_ask_reuses_one_lcel_chain_across_questions(
        self,
        create_chain: Mock,
    ) -> None:
        vector_store = Mock()
        retriever = Mock()
        chat_model = Mock()
        expected = RAGAnswer(
            answer="RAG 会先检索资料。",
            sources=(
                Document(
                    page_content="RAG 会先检索资料。",
                    metadata={"source": "rag.md", "chunk_index": 2},
                ),
            ),
        )
        chain = create_chain.return_value
        chain.invoke.return_value = expected
        service = RAGService(vector_store, retriever, chat_model)

        first_result = service.ask("RAG 是什么？")
        second_result = service.ask("为什么要检索？")

        self.assertIs(first_result, expected)
        self.assertIs(second_result, expected)
        create_chain.assert_called_once_with(
            retriever,
            chat_model,
            context_selector=service.context_selector,
        )
        self.assertEqual(
            [call.args[0] for call in chain.invoke.call_args_list],
            ["RAG 是什么？", "为什么要检索？"],
        )

    @patch("app.rag_service.create_rag_chain")
    def test_ask_records_service_trace_without_logging_question(
        self,
        create_chain: Mock,
    ) -> None:
        expected = RAGAnswer(answer="回答", sources=())
        create_chain.return_value.invoke.return_value = expected
        service = RAGService(Mock(), Mock(), Mock())

        with observation_context(request_id="request-service", user_id="user-service"):
            with self.assertLogs(
                OBSERVABILITY_LOGGER_NAME, level="INFO"
            ) as captured:
                result = service.ask("不应进入日志的问题")

        payloads = [json.loads(record.getMessage()) for record in captured.records]
        self.assertIs(result, expected)
        self.assertEqual(
            [payload["event"] for payload in payloads],
            ["rag_request_started", "rag_request_completed"],
        )
        self.assertTrue(
            all(payload["request_id"] == "request-service" for payload in payloads)
        )
        self.assertNotIn(
            "不应进入日志",
            "\n".join(record.getMessage() for record in captured.records),
        )

    def test_close_is_idempotent(self) -> None:
        vector_store = Mock()
        service = RAGService(vector_store, Mock(), Mock())

        service.close()
        service.close()

        vector_store._client.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
