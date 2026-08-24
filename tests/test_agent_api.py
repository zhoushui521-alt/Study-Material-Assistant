import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from fastapi.testclient import TestClient

from app.agent_service import (
    ANSWER_TOOL_NAME,
    AgentExecutionError,
    AgentResult,
    AgentService,
    AgentTimeoutError,
)
from app.api import (
    app,
    get_agent_service_provider,
    invalidate_rag_service,
)
from app.model_gateway import ModelAuthenticationError
from app.operation_guard import OperationGuard, OperationPolicy
from app.rag_service import RAGService
from app.request_history import RequestHistoryWriter
from tests.auth_helpers import TEST_USER, clear_user_services, install_authenticated_user


class AgentAPITests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        app.dependency_overrides.clear()
        install_authenticated_user()
        app.state.rag_service = None
        app.state.agent_service = None
        app.state.operation_guard = OperationGuard()
        app.state.request_history_writer = RequestHistoryWriter(
            Path(self.temporary_directory.name) / "requests.jsonl"
        )

    def tearDown(self) -> None:
        app.dependency_overrides.clear()
        clear_user_services()
        app.state.rag_service = None
        app.state.agent_service = None
        self.temporary_directory.cleanup()

    @staticmethod
    def make_service() -> Mock:
        service = Mock(spec=AgentService)
        service.ask = AsyncMock()
        return service

    def test_agent_returns_bounded_result_and_forwards_preview_authorization(self) -> None:
        service = self.make_service()
        service.ask.return_value = AgentResult(
            answer="资料中的原始回答",
            sources=("[rag.md · 第 2 段]",),
            tools_used=(ANSWER_TOOL_NAME,),
        )
        app.dependency_overrides[get_agent_service_provider] = lambda: lambda: service

        with TestClient(app) as client:
            response = client.post(
                "/api/agent",
                json={
                    "message": " 请根据资料回答 ",
                    "confirm_api_cost": True,
                    "allow_web_preview": True,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "answer": "资料中的原始回答",
                "sources": ["[rag.md · 第 2 段]"],
                "tools_used": [ANSWER_TOOL_NAME],
            },
        )
        service.ask.assert_awaited_once_with(
            "请根据资料回答",
            allow_web_preview=True,
        )

    def test_agent_requires_explicit_cost_confirmation_before_initialization(self) -> None:
        with patch("app.api.create_agent_service") as create_service:
            with TestClient(app) as client:
                missing = client.post(
                    "/api/agent",
                    json={"message": "列出资料"},
                )
                rejected = client.post(
                    "/api/agent",
                    json={"message": "列出资料", "confirm_api_cost": False},
                )

        self.assertEqual(missing.status_code, 422)
        self.assertEqual(rejected.status_code, 422)
        create_service.assert_not_called()

    def test_agent_rejects_extra_fields_before_initialization(self) -> None:
        with patch("app.api.create_agent_service") as create_service:
            with TestClient(app) as client:
                response = client.post(
                    "/api/agent",
                    json={
                        "message": "列出资料",
                        "confirm_api_cost": True,
                        "api_key": "must-not-be-accepted",
                    },
                )

        self.assertEqual(response.status_code, 422)
        create_service.assert_not_called()

    def test_agent_timeout_returns_generic_504_without_internal_detail(self) -> None:
        service = self.make_service()
        service.ask.side_effect = AgentTimeoutError(
            "upstream secret-token timed out"
        )
        app.dependency_overrides[get_agent_service_provider] = lambda: lambda: service

        with TestClient(app) as client:
            response = client.post(
                "/api/agent",
                json={"message": "列出资料", "confirm_api_cost": True},
            )

        self.assertEqual(response.status_code, 504)
        self.assertEqual(response.json(), {"detail": "Agent 处理超时。"})
        self.assertNotIn("secret-token", response.text)

    def test_agent_execution_failure_returns_generic_502(self) -> None:
        service = self.make_service()
        service.ask.side_effect = AgentExecutionError(
            "BAILIAN_API_KEY=secret-value"
        )
        app.dependency_overrides[get_agent_service_provider] = lambda: lambda: service

        with TestClient(app) as client:
            response = client.post(
                "/api/agent",
                json={"message": "列出资料", "confirm_api_cost": True},
            )

        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.json(), {"detail": "Agent 处理失败。"})
        self.assertNotIn("secret-value", response.text)

    def test_wrapped_model_authentication_failure_returns_stable_502(self) -> None:
        service = self.make_service()
        authentication_error = ModelAuthenticationError(
            "provider detail must not escape"
        )
        wrapped_error = AgentExecutionError("Agent 处理失败。")
        wrapped_error.__cause__ = authentication_error
        service.ask.side_effect = wrapped_error
        app.dependency_overrides[get_agent_service_provider] = lambda: lambda: service

        with TestClient(app) as client:
            response = client.post(
                "/api/agent",
                json={"message": "列出资料", "confirm_api_cost": True},
            )

        self.assertEqual(response.status_code, 502)
        self.assertEqual(
            response.json(),
            {"detail": "Model authentication failed."},
        )
        self.assertNotIn("provider detail", response.text)

    def test_agent_rate_limit_prevents_extra_service_call(self) -> None:
        service = self.make_service()
        service.ask.return_value = AgentResult(
            answer="完成",
            sources=(),
            tools_used=(),
        )
        app.dependency_overrides[get_agent_service_provider] = lambda: lambda: service
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

        with TestClient(app) as client:
            accepted = client.post(
                "/api/agent",
                json={"message": "第一次", "confirm_api_cost": True},
            )
            rejected = client.post(
                "/api/agent",
                json={"message": "第二次", "confirm_api_cost": True},
            )

        self.assertEqual(accepted.status_code, 200)
        self.assertEqual(rejected.status_code, 429)
        self.assertEqual(rejected.headers["retry-after"], "60")
        self.assertEqual(service.ask.await_count, 1)

    def test_agent_request_history_excludes_message_answer_and_source(self) -> None:
        service = self.make_service()
        service.ask.return_value = AgentResult(
            answer="private-answer",
            sources=("[private-source.md · 第 1 段]",),
            tools_used=(ANSWER_TOOL_NAME,),
        )
        app.dependency_overrides[get_agent_service_provider] = lambda: lambda: service

        with TestClient(app) as client:
            response = client.post(
                "/api/agent",
                json={"message": "private-question", "confirm_api_cost": True},
            )

        self.assertEqual(response.status_code, 200)
        log_text = (
            Path(self.temporary_directory.name) / "requests.jsonl"
        ).read_text(encoding="utf-8")
        self.assertNotIn("private-question", log_text)
        self.assertNotIn("private-answer", log_text)
        self.assertNotIn("private-source", log_text)
        self.assertEqual(json.loads(log_text.splitlines()[-1])["path"], "/api/agent")

    def test_agent_service_is_reused_with_current_rag_service(self) -> None:
        rag_service = Mock(spec=RAGService)
        agent_service = self.make_service()
        agent_service.ask.return_value = AgentResult(
            answer="完成",
            sources=(),
            tools_used=(),
        )
        app.state.rag_services[TEST_USER.user_id] = rag_service

        with patch("app.api.create_agent_service", return_value=agent_service) as create:
            with TestClient(app) as client:
                first = client.post(
                    "/api/agent",
                    json={"message": "第一次", "confirm_api_cost": True},
                )
                second = client.post(
                    "/api/agent",
                    json={"message": "第二次", "confirm_api_cost": True},
                )

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        create.assert_called_once()
        self.assertIs(create.call_args.args[0], rag_service)
        rag_service.close.assert_called_once_with()
        self.assertNotIn(TEST_USER.user_id, app.state.agent_services)

    def test_rag_invalidation_also_invalidates_agent(self) -> None:
        rag_service = Mock(spec=RAGService)
        app.state.rag_services[TEST_USER.user_id] = rag_service
        app.state.agent_services[TEST_USER.user_id] = self.make_service()

        invalidate_rag_service(app, TEST_USER.user_id)

        rag_service.close.assert_called_once_with()
        self.assertNotIn(TEST_USER.user_id, app.state.rag_services)
        self.assertNotIn(TEST_USER.user_id, app.state.agent_services)


if __name__ == "__main__":
    unittest.main()
