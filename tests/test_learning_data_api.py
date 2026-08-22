import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

from app.api import app, get_learning_data_store
from app.learning_data import LearningDataStore
from app.operation_guard import OperationGuard

def register(
    client: TestClient,
    email: str,
    *,
    display_name: str = "学习者",
) -> dict:
    response = client.post(
        "/api/auth/register",
        json={
            "email": email,
            "password": "safe-test-password",
            "display_name": display_name,
        },
    )
    return response.json()

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

    def test_register_duplicate_login_logout_and_password_safety(self) -> None:
        credentials = {
            "email": "learner@example.com",
            "password": "correct-horse-battery",
            "display_name": "水哥",
        }
        with TestClient(app) as client:
            created = client.post("/api/auth/register", json=credentials)
            current = client.get("/api/auth/me")
            duplicate = client.post("/api/auth/register", json=credentials)

            async def password_state() -> tuple[str, str | None]:
                assert self.store is not None
                user = await self.store.get_user_by_email(credentials["email"])
                return credentials["password"], user.password_hash

            plain_password, password_hash = client.portal.call(password_state)
            logged_out = client.post("/api/auth/logout")
            anonymous = client.get("/api/auth/me")
            wrong_password = client.post(
                "/api/auth/login",
                json={
                    "email": credentials["email"],
                    "password": "definitely-wrong",
                },
            )
            logged_in = client.post(
                "/api/auth/login",
                json={
                    "email": credentials["email"],
                    "password": credentials["password"],
                },
            )

        self.assertEqual(created.status_code, 201)
        self.assertEqual(current.json(), created.json())
        self.assertEqual(duplicate.status_code, 409)
        self.assertNotIn("password", created.json())
        self.assertIsNotNone(password_hash)
        self.assertNotEqual(password_hash, plain_password)
        self.assertTrue(password_hash.startswith("scrypt$"))
        self.assertEqual(logged_out.status_code, 204)
        self.assertEqual(anonymous.status_code, 401)
        self.assertEqual(wrong_password.status_code, 401)
        self.assertEqual(logged_in.status_code, 200)

    def test_create_list_sessions_and_hide_other_user_session(self) -> None:
        with TestClient(app) as client:
            first_user = register(client, "first@example.com")["user_id"]
            created = client.post(
                "/api/sessions",
                json={"topic": " Embedding "},
            )
            first_sessions = client.get("/api/sessions")
            client.post("/api/auth/logout")
            second_user = register(client, "second@example.com")["user_id"]
            second_sessions = client.get("/api/sessions")
            hidden = client.get(
                "/api/history",
                params={"session_id": created.json()["session_id"]},
            )

        self.assertNotEqual(first_user, second_user)
        self.assertEqual(created.status_code, 201)
        self.assertEqual(created.json()["topic"], "Embedding")
        self.assertEqual(len(first_sessions.json()["sessions"]), 1)
        self.assertEqual(second_sessions.json(), {"sessions": []})
        self.assertEqual(hidden.status_code, 404)
        self.assertEqual(hidden.json(), {"detail": "用户或学习会话不存在。"})

    def test_history_returns_persisted_messages_and_learning_records(self) -> None:
        with TestClient(app) as client:
            user_id = register(client, "history@example.com")["user_id"]
            session_id = client.post(
                "/api/sessions",
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
                "/api/history",
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

    def test_invalid_bearer_token_is_rejected(self) -> None:
        with TestClient(app) as client:
            response = client.get(
                "/api/auth/me",
                headers={"Authorization": "Bearer forged-token"},
            )

        self.assertEqual(response.status_code, 401)
        self.assertEqual(
            response.json(),
            {"detail": "登录状态无效或已过期。"},
        )


if __name__ == "__main__":
    unittest.main()
