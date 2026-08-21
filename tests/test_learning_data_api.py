import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

from app.api import app, get_learning_data_store
from app.learning_data import LearningDataStore
from app.operation_guard import OperationGuard
from app.request_history import RequestHistoryWriter


class LearningDataAPITests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = (
            Path(self.temporary_directory.name) / "learning.sqlite3"
        )
        self.store: LearningDataStore | None = None
        app.dependency_overrides.clear()
        app.state.learning_data_store = None
        app.state.rag_service = None
        app.state.agent_service = None
        app.state.tutor_service = None
        app.state.operation_guard = OperationGuard()
        app.state.request_history_writer = RequestHistoryWriter(
            Path(self.temporary_directory.name) / "requests.jsonl"
        )

        async def provide_store() -> LearningDataStore:
            if self.store is None:
                self.store = await LearningDataStore.open(self.database_path)
                app.state.learning_data_store = self.store
            return self.store

        app.dependency_overrides[get_learning_data_store] = provide_store

    def tearDown(self) -> None:
        app.dependency_overrides.clear()
        app.state.learning_data_store = None
        self.store = None
        self.temporary_directory.cleanup()

    def test_create_and_query_user(self) -> None:
        with TestClient(app) as client:
            created = client.post("/api/users")
            loaded = client.get(f"/api/users/{created.json()['user_id']}")

        self.assertEqual(created.status_code, 201)
        self.assertEqual(loaded.status_code, 200)
        self.assertEqual(loaded.json(), created.json())

    def test_create_list_sessions_and_hide_other_user_session(self) -> None:
        with TestClient(app) as client:
            first_user = client.post("/api/users").json()["user_id"]
            second_user = client.post("/api/users").json()["user_id"]
            created = client.post(
                f"/api/users/{first_user}/sessions",
                json={"topic": " Embedding "},
            )
            first_sessions = client.get(f"/api/users/{first_user}/sessions")
            second_sessions = client.get(f"/api/users/{second_user}/sessions")
            hidden = client.get(
                f"/api/users/{second_user}/history",
                params={"session_id": created.json()["session_id"]},
            )

        self.assertEqual(created.status_code, 201)
        self.assertEqual(created.json()["topic"], "Embedding")
        self.assertEqual(len(first_sessions.json()["sessions"]), 1)
        self.assertEqual(second_sessions.json(), {"sessions": []})
        self.assertEqual(hidden.status_code, 404)
        self.assertEqual(hidden.json(), {"detail": "用户或学习会话不存在。"})

    def test_history_returns_persisted_messages_and_learning_records(self) -> None:
        with TestClient(app) as client:
            user_id = client.post("/api/users").json()["user_id"]
            session_id = client.post(
                f"/api/users/{user_id}/sessions",
                json={"topic": "RAG"},
            ).json()["session_id"]

            async def seed_history() -> None:
                assert self.store is not None
                await self.store.record_tutor_exchange(
                    user_id=user_id,
                    session_id=session_id,
                    topic="RAG",
                    intent="summary",
                    user_content="总结本次学习",
                    tutor_content="本次学习了 RAG。",
                    activity_type="summarize_learning",
                    metadata={"tools_used": ["learning_summary"]},
                )

            client.portal.call(seed_history)
            history = client.get(
                f"/api/users/{user_id}/history",
                params={"session_id": session_id},
            )

        self.assertEqual(history.status_code, 200)
        payload = history.json()
        self.assertEqual([item["role"] for item in payload["messages"]], ["user", "tutor"])
        self.assertEqual(
            payload["learning_records"][0]["activity_type"],
            "summarize_learning",
        )
        self.assertEqual(
            payload["learning_records"][0]["metadata"],
            {"tools_used": ["learning_summary"]},
        )

    def test_database_failure_returns_generic_error_without_internal_detail(self) -> None:
        class FailingStore:
            async def get_user(self, user_id: str):
                del user_id
                raise RuntimeError("database-path-and-secret")

        async def provide_failing_store() -> FailingStore:
            return FailingStore()

        app.dependency_overrides[get_learning_data_store] = provide_failing_store

        with TestClient(app) as client:
            response = client.get(f"/api/users/{uuid4()}")

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json(), {"detail": "学习数据服务暂不可用。"})
        self.assertNotIn("database-path-and-secret", response.text)


if __name__ == "__main__":
    unittest.main()
