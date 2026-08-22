import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch
from uuid import UUID

from fastapi.testclient import TestClient

from app.agent_service import AgentService
from app.api import (
    app,
    get_agent_service_provider,
    get_study_workflow_service,
)
from app.operation_guard import OperationGuard, OperationPolicy
from app.request_history import RequestHistoryWriter
from tests.auth_helpers import TEST_USER, clear_user_services, install_authenticated_user
from app.study_workflow import (
    StudyWorkflowConflictError,
    StudyWorkflowNotFoundError,
    StudyWorkflowResult,
    StudyWorkflowService,
)


WORKFLOW_ID = "11111111-1111-4111-8111-111111111111"


def workflow_result(
    *,
    status: str = "awaiting_confirmation",
    approval_required: bool = True,
) -> StudyWorkflowResult:
    return StudyWorkflowResult(
        workflow_id=WORKFLOW_ID,
        goal="学习 RAG",
        status=status,
        evidence_summary="基于资料的证据",
        sources=("[rag.md · 第 1 段]",),
        tasks=(
            {"id": "task-1", "title": "阅读资料", "status": "pending"},
            {"id": "task-2", "title": "解释概念", "status": "pending"},
            {"id": "task-3", "title": "完成练习", "status": "pending"},
        ),
        current_task_index=0,
        progress_history=(),
        retry_count=0,
        review="",
        route_trace=("plan:drafted",),
        error="",
        approval_required=approval_required,
    )


class StudyWorkflowAPITests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        app.dependency_overrides.clear()
        install_authenticated_user()
        app.state.rag_service = None
        app.state.agent_service = None
        app.state.study_workflow_service = None
        app.state.operation_guard = OperationGuard()
        app.state.request_history_writer = RequestHistoryWriter(
            Path(self.temporary_directory.name) / "requests.jsonl"
        )

    def tearDown(self) -> None:
        app.dependency_overrides.clear()
        clear_user_services()
        app.state.rag_service = None
        app.state.agent_service = None
        app.state.study_workflow_service = None
        self.temporary_directory.cleanup()

    @staticmethod
    def make_workflow_service() -> Mock:
        service = Mock(spec=StudyWorkflowService)
        service.start = AsyncMock()
        service.confirm = AsyncMock()
        service.record_progress = AsyncMock()
        service.retry = AsyncMock()
        service.assert_retryable = AsyncMock()
        service.get = AsyncMock()
        service.delete = AsyncMock()
        service.close = AsyncMock()
        return service

    def override_services(self, workflow_service: Mock, agent_service: Mock) -> None:
        app.dependency_overrides[get_study_workflow_service] = (
            lambda: workflow_service
        )
        app.dependency_overrides[get_agent_service_provider] = (
            lambda: lambda: agent_service
        )

    def test_start_requires_cost_confirmation_and_returns_paused_plan(self) -> None:
        workflow_service = self.make_workflow_service()
        workflow_service.start.return_value = workflow_result()
        agent_service = Mock(spec=AgentService)
        self.override_services(workflow_service, agent_service)

        with TestClient(app) as client:
            response = client.post(
                "/api/study-workflows",
                json={"goal": " 学习 RAG ", "confirm_api_cost": True},
            )

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json()["status"], "awaiting_confirmation")
        self.assertTrue(response.json()["approval_required"])
        self.assertEqual(response.json()["sources"], ["[rag.md · 第 1 段]"])
        workflow_service.start.assert_awaited_once()
        call = workflow_service.start.await_args
        self.assertEqual(call.args[0], TEST_USER.user_id)
        UUID(call.args[1])
        self.assertEqual(call.args[2], "学习 RAG")
        self.assertIs(call.kwargs["agent_service"], agent_service)

    def test_invalid_start_never_initializes_agent(self) -> None:
        workflow_service = self.make_workflow_service()
        agent_provider = Mock()
        app.dependency_overrides[get_study_workflow_service] = (
            lambda: workflow_service
        )
        app.dependency_overrides[get_agent_service_provider] = (
            lambda: agent_provider
        )

        with TestClient(app) as client:
            missing = client.post(
                "/api/study-workflows",
                json={"goal": "学习 RAG"},
            )
            rejected = client.post(
                "/api/study-workflows",
                json={"goal": "学习 RAG", "confirm_api_cost": False},
            )

        self.assertEqual(missing.status_code, 422)
        self.assertEqual(rejected.status_code, 422)
        agent_provider.assert_not_called()
        workflow_service.start.assert_not_awaited()

    def test_confirm_and_progress_are_zero_model_workflow_operations(self) -> None:
        workflow_service = self.make_workflow_service()
        workflow_service.confirm.return_value = workflow_result(
            status="in_progress",
            approval_required=False,
        )
        workflow_service.record_progress.return_value = workflow_result(
            status="in_progress",
            approval_required=False,
        )
        agent_service = Mock(spec=AgentService)
        agent_provider = Mock(return_value=agent_service)
        app.dependency_overrides[get_study_workflow_service] = (
            lambda: workflow_service
        )
        app.dependency_overrides[get_agent_service_provider] = (
            lambda: agent_provider
        )

        with TestClient(app) as client:
            confirmed = client.post(
                f"/api/study-workflows/{WORKFLOW_ID}/confirm",
                json={"decision": "approve"},
            )
            progressed = client.post(
                f"/api/study-workflows/{WORKFLOW_ID}/progress",
                json={"note": " 完成阅读 ", "complete_current_task": True},
            )

        self.assertEqual(confirmed.status_code, 200)
        self.assertEqual(progressed.status_code, 200)
        workflow_service.confirm.assert_awaited_once_with(
            TEST_USER.user_id, WORKFLOW_ID, "approve"
        )
        workflow_service.record_progress.assert_awaited_once_with(
            TEST_USER.user_id,
            WORKFLOW_ID,
            "完成阅读",
            complete_current_task=True,
        )
        agent_provider.assert_not_called()

    def test_retry_requires_new_cost_confirmation(self) -> None:
        workflow_service = self.make_workflow_service()
        workflow_service.retry.return_value = workflow_result()
        agent_service = Mock(spec=AgentService)
        self.override_services(workflow_service, agent_service)

        with TestClient(app) as client:
            rejected = client.post(
                f"/api/study-workflows/{WORKFLOW_ID}/retry",
                json={"confirm_api_cost": False},
            )
            accepted = client.post(
                f"/api/study-workflows/{WORKFLOW_ID}/retry",
                json={"confirm_api_cost": True},
            )

        self.assertEqual(rejected.status_code, 422)
        self.assertEqual(accepted.status_code, 200)
        workflow_service.retry.assert_awaited_once_with(
            TEST_USER.user_id,
            WORKFLOW_ID,
            agent_service=agent_service,
        )
        workflow_service.assert_retryable.assert_awaited_once_with(
            TEST_USER.user_id, WORKFLOW_ID
        )

    def test_non_retryable_workflow_never_initializes_agent(self) -> None:
        workflow_service = self.make_workflow_service()
        workflow_service.assert_retryable.side_effect = StudyWorkflowConflictError(
            "当前工作流不能再次重试。"
        )
        agent_provider = Mock()
        app.dependency_overrides[get_study_workflow_service] = (
            lambda: workflow_service
        )
        app.dependency_overrides[get_agent_service_provider] = (
            lambda: agent_provider
        )

        with TestClient(app) as client:
            response = client.post(
                f"/api/study-workflows/{WORKFLOW_ID}/retry",
                json={"confirm_api_cost": True},
            )

        self.assertEqual(response.status_code, 409)
        agent_provider.assert_not_called()
        workflow_service.retry.assert_not_awaited()

    def test_get_maps_not_found_and_conflict_is_safe(self) -> None:
        workflow_service = self.make_workflow_service()
        workflow_service.get.side_effect = StudyWorkflowNotFoundError(
            "database path secret-value"
        )
        workflow_service.confirm.side_effect = StudyWorkflowConflictError(
            "database path secret-value"
        )
        self.override_services(workflow_service, Mock(spec=AgentService))

        with TestClient(app) as client:
            missing = client.get(f"/api/study-workflows/{WORKFLOW_ID}")
            conflict = client.post(
                f"/api/study-workflows/{WORKFLOW_ID}/confirm",
                json={"decision": "approve"},
            )

        self.assertEqual(missing.status_code, 404)
        self.assertEqual(missing.json(), {"detail": "学习工作流不存在。"})
        self.assertNotIn("secret-value", missing.text)
        self.assertEqual(conflict.status_code, 409)
        self.assertEqual(
            conflict.json(),
            {"detail": "当前学习工作流状态不允许此操作。"},
        )
        self.assertNotIn("secret-value", conflict.text)

    def test_lazy_workflow_service_is_reused_and_closed_by_lifespan(self) -> None:
        workflow_service = self.make_workflow_service()
        workflow_service.start.return_value = workflow_result()
        workflow_service.get.return_value = workflow_result()
        agent_service = Mock(spec=AgentService)
        app.dependency_overrides[get_agent_service_provider] = (
            lambda: lambda: agent_service
        )

        with patch(
            "app.api.open_sqlite_study_workflow_service",
            new=AsyncMock(return_value=workflow_service),
        ) as opener:
            with TestClient(app) as client:
                first = client.post(
                    "/api/study-workflows",
                    json={"goal": "学习 RAG", "confirm_api_cost": True},
                )
                second = client.get(f"/api/study-workflows/{WORKFLOW_ID}")

        self.assertEqual(first.status_code, 201)
        self.assertEqual(second.status_code, 200)
        opener.assert_awaited_once_with()
        workflow_service.close.assert_awaited_once_with()

    def test_delete_requires_confirmation_and_removes_checkpoint_thread(self) -> None:
        workflow_service = self.make_workflow_service()
        self.override_services(workflow_service, Mock(spec=AgentService))

        with TestClient(app) as client:
            rejected = client.request(
                "DELETE",
                f"/api/study-workflows/{WORKFLOW_ID}",
                json={"confirm_delete": False},
            )
            accepted = client.request(
                "DELETE",
                f"/api/study-workflows/{WORKFLOW_ID}",
                json={"confirm_delete": True},
            )

        self.assertEqual(rejected.status_code, 422)
        self.assertEqual(accepted.status_code, 200)
        self.assertEqual(
            accepted.json(),
            {"workflow_id": WORKFLOW_ID, "status": "deleted"},
        )
        workflow_service.delete.assert_awaited_once_with(
            TEST_USER.user_id, WORKFLOW_ID
        )

    def test_workflow_rate_limit_blocks_second_progress_update(self) -> None:
        workflow_service = self.make_workflow_service()
        workflow_service.record_progress.return_value = workflow_result(
            status="in_progress",
            approval_required=False,
        )
        self.override_services(workflow_service, Mock(spec=AgentService))
        app.state.operation_guard = OperationGuard(
            policies={
                "workflow": OperationPolicy(
                    window_seconds=60,
                    max_calls_per_window=1,
                )
            }
        )

        payload = {"note": "记录进度", "complete_current_task": False}
        with TestClient(app) as client:
            accepted = client.post(
                f"/api/study-workflows/{WORKFLOW_ID}/progress",
                json=payload,
            )
            rejected = client.post(
                f"/api/study-workflows/{WORKFLOW_ID}/progress",
                json=payload,
            )

        self.assertEqual(accepted.status_code, 200)
        self.assertEqual(rejected.status_code, 429)
        self.assertEqual(workflow_service.record_progress.await_count, 1)

    def test_request_history_excludes_goal_evidence_progress_and_id(self) -> None:
        workflow_service = self.make_workflow_service()
        workflow_service.record_progress.return_value = workflow_result(
            status="in_progress",
            approval_required=False,
        )
        self.override_services(workflow_service, Mock(spec=AgentService))

        with TestClient(app) as client:
            response = client.post(
                f"/api/study-workflows/{WORKFLOW_ID}/progress",
                json={
                    "note": "private-progress",
                    "complete_current_task": False,
                },
            )

        self.assertEqual(response.status_code, 200)
        log_text = (
            Path(self.temporary_directory.name) / "requests.jsonl"
        ).read_text(encoding="utf-8")
        self.assertNotIn("private-progress", log_text)
        self.assertNotIn("基于资料的证据", log_text)
        self.assertNotIn(WORKFLOW_ID, log_text)
        record = json.loads(log_text.splitlines()[-1])
        self.assertEqual(
            record["path"],
            "/api/study-workflows/{workflow_id}/progress",
        )


if __name__ == "__main__":
    unittest.main()
