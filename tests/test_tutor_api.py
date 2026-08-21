import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from fastapi.testclient import TestClient

from app.api import (
    app,
    get_tutor_service_provider,
    invalidate_rag_service,
)
from app.operation_guard import OperationGuard, OperationPolicy
from app.rag_service import RAGService
from app.request_history import RequestHistoryWriter
from app.tutor_workflow import (
    TutorExecutionError,
    TutorResult,
    TutorTimeoutError,
    TutorWorkflowService,
)


SESSION_ID = "11111111-1111-4111-8111-111111111111"


def tutor_result() -> TutorResult:
    return TutorResult(
        session_id=SESSION_ID,
        intent="quiz",
        topic="Embedding",
        answer="资料要点和一道练习 [S1]",
        sources=("[rag.md · 第 2 段]",),
        citations=(
            {
                "citation_id": "S1",
                "evidence_id": "e" * 64,
                "material_id": "m" * 64,
                "chunk_id": "c" * 64,
                "source": "rag.md",
                "filename": "rag.md",
                "page": None,
                "chunk_index": 2,
                "excerpt": "private-evidence",
                "locator": "rag.md#chunk=2",
            },
        ),
        evidence=(
            {
                "context_id": "S1",
                "evidence_id": "e" * 64,
                "material_id": "m" * 64,
                "chunk_id": "c" * 64,
                "source": "rag.md",
                "filename": "rag.md",
                "source_type": "markdown",
                "page": None,
                "section": None,
                "chunk_index": 2,
                "excerpt": "private-evidence",
                "locator": "rag.md#chunk=2",
                "content_hash": "h" * 64,
                "canonical_url": None,
            },
        ),
        learning_action="practice_quiz",
        quiz={
            "question": "Embedding 的作用是什么？",
            "options": ["表示语义", "删除资料"],
            "answer": "表示语义",
            "explanation": "可由资料直接支持。",
        },
        summary=None,
        tools_used=("knowledge_retrieval", "quiz_generator"),
        route_trace=("intent:quiz", "response:completed"),
    )


class TutorAPITests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        app.dependency_overrides.clear()
        app.state.rag_service = None
        app.state.agent_service = None
        app.state.tutor_service = None
        app.state.operation_guard = OperationGuard()
        app.state.request_history_writer = RequestHistoryWriter(
            Path(self.temporary_directory.name) / "requests.jsonl"
        )

    def tearDown(self) -> None:
        app.dependency_overrides.clear()
        app.state.rag_service = None
        app.state.agent_service = None
        app.state.tutor_service = None
        self.temporary_directory.cleanup()

    @staticmethod
    def make_service() -> Mock:
        service = Mock(spec=TutorWorkflowService)
        service.chat = AsyncMock()
        return service

    def test_tutor_chat_returns_learning_action_quiz_and_evidence(self) -> None:
        service = self.make_service()
        service.chat.return_value = tutor_result()
        app.dependency_overrides[get_tutor_service_provider] = lambda: lambda: service

        with TestClient(app) as client:
            response = client.post(
                "/api/tutor/chat",
                json={
                    "message": " 帮我出题练习 Embedding ",
                    "session_id": SESSION_ID,
                    "confirm_api_cost": True,
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["session_id"], SESSION_ID)
        self.assertEqual(payload["learning_action"], "practice_quiz")
        self.assertEqual(payload["quiz"]["answer"], "表示语义")
        self.assertEqual(payload["citations"][0]["citation_id"], "S1")
        self.assertEqual(payload["evidence"][0]["context_id"], "S1")
        service.chat.assert_awaited_once_with(
            SESSION_ID, "帮我出题练习 Embedding"
        )

    def test_cost_confirmation_and_session_id_are_required_before_initialization(self) -> None:
        with patch("app.api.create_tutor_workflow_service") as create_service:
            with TestClient(app) as client:
                missing_cost = client.post(
                    "/api/tutor/chat",
                    json={"message": "解释 RAG", "session_id": SESSION_ID},
                )
                missing_session = client.post(
                    "/api/tutor/chat",
                    json={"message": "解释 RAG", "confirm_api_cost": True},
                )
                extra_field = client.post(
                    "/api/tutor/chat",
                    json={
                        "message": "解释 RAG",
                        "session_id": SESSION_ID,
                        "confirm_api_cost": True,
                        "api_key": "must-not-be-accepted",
                    },
                )

        self.assertEqual(missing_cost.status_code, 422)
        self.assertEqual(missing_session.status_code, 422)
        self.assertEqual(extra_field.status_code, 422)
        create_service.assert_not_called()

    def test_tutor_timeout_and_failure_return_generic_errors(self) -> None:
        service = self.make_service()
        app.dependency_overrides[get_tutor_service_provider] = lambda: lambda: service
        request = {
            "message": "解释 RAG",
            "session_id": SESSION_ID,
            "confirm_api_cost": True,
        }

        with TestClient(app) as client:
            service.chat.side_effect = TutorTimeoutError("secret-timeout")
            timeout = client.post("/api/tutor/chat", json=request)
            service.chat.side_effect = TutorExecutionError("secret-failure")
            failure = client.post("/api/tutor/chat", json=request)

        self.assertEqual(timeout.status_code, 504)
        self.assertEqual(timeout.json(), {"detail": "Tutor 处理超时。"})
        self.assertNotIn("secret-timeout", timeout.text)
        self.assertEqual(failure.status_code, 502)
        self.assertEqual(failure.json(), {"detail": "Tutor 处理失败。"})
        self.assertNotIn("secret-failure", failure.text)

    def test_tutor_uses_agent_rate_and_budget_boundary(self) -> None:
        service = self.make_service()
        service.chat.return_value = tutor_result()
        app.dependency_overrides[get_tutor_service_provider] = lambda: lambda: service
        app.state.operation_guard = OperationGuard(
            policies={
                "agent": OperationPolicy(
                    window_seconds=60,
                    max_calls_per_window=1,
                    max_units_per_operation=1,
                    max_units_per_process=10,
                )
            }
        )
        request = {
            "message": "解释 RAG",
            "session_id": SESSION_ID,
            "confirm_api_cost": True,
        }

        with TestClient(app) as client:
            accepted = client.post("/api/tutor/chat", json=request)
            rejected = client.post("/api/tutor/chat", json=request)

        self.assertEqual(accepted.status_code, 200)
        self.assertEqual(rejected.status_code, 429)
        self.assertEqual(service.chat.await_count, 1)

    def test_request_history_excludes_session_message_answer_and_evidence(self) -> None:
        service = self.make_service()
        service.chat.return_value = tutor_result()
        app.dependency_overrides[get_tutor_service_provider] = lambda: lambda: service

        with TestClient(app) as client:
            response = client.post(
                "/api/tutor/chat",
                json={
                    "message": "private-question",
                    "session_id": SESSION_ID,
                    "confirm_api_cost": True,
                },
            )

        self.assertEqual(response.status_code, 200)
        log_text = (
            Path(self.temporary_directory.name) / "requests.jsonl"
        ).read_text(encoding="utf-8")
        self.assertNotIn("private-question", log_text)
        self.assertNotIn("private-evidence", log_text)
        self.assertNotIn(SESSION_ID, log_text)
        self.assertEqual(
            json.loads(log_text.splitlines()[-1])["path"],
            "/api/tutor/chat",
        )

    def test_rag_invalidation_also_invalidates_tutor(self) -> None:
        rag_service = Mock(spec=RAGService)
        app.state.rag_service = rag_service
        app.state.tutor_service = self.make_service()

        invalidate_rag_service(app)

        rag_service.close.assert_called_once_with()
        self.assertIsNone(app.state.rag_service)
        self.assertIsNone(app.state.tutor_service)

    def test_tutor_service_is_reused_with_current_rag_service(self) -> None:
        rag_service = Mock(spec=RAGService)
        tutor_service = self.make_service()
        tutor_service.chat.return_value = tutor_result()
        app.state.rag_service = rag_service

        with patch(
            "app.api.create_tutor_workflow_service",
            return_value=tutor_service,
        ) as create:
            with TestClient(app) as client:
                request = {
                    "message": "解释 Embedding",
                    "session_id": SESSION_ID,
                    "confirm_api_cost": True,
                }
                first = client.post("/api/tutor/chat", json=request)
                second = client.post("/api/tutor/chat", json=request)

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        create.assert_called_once_with(rag_service)
        self.assertEqual(tutor_service.chat.await_count, 2)
        rag_service.close.assert_called_once_with()
        self.assertIsNone(app.state.tutor_service)


if __name__ == "__main__":
    unittest.main()
