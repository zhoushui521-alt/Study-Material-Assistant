import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from cryptography.fernet import Fernet
from fastapi.testclient import TestClient

from app.api import app, get_current_user, get_rag_service_provider
from app.config import Settings
from app.learning_data import UserRecord
from app.model_gateway import ModelAuthenticationError, ModelGateway
from app.operation_guard import OperationGuard
from app.request_history import RequestHistoryWriter
from tests.auth_helpers import TEST_USER, clear_user_services, install_authenticated_user


class ModelGatewayAPITests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.original_gateway = app.state.model_gateway
        app.dependency_overrides.clear()
        install_authenticated_user()
        app.state.operation_guard = OperationGuard()
        app.state.request_history_writer = RequestHistoryWriter(
            self.root / "requests.jsonl"
        )
        settings = Settings.from_mapping(
            {
                "APP_DATA_DIR": str(self.root / "data"),
                "DEFAULT_PROVIDER": "qwen",
                "DEFAULT_MODEL": "qwen-plus",
                "MODEL_API_KEY": "sk-system",
                "MODEL_BASE_URL": "https://workspace.example.com/v1",
                "MODEL_CREDENTIAL_ENCRYPTION_KEY": (
                    Fernet.generate_key().decode("ascii")
                ),
            },
            project_root=self.root,
        )
        self.settings = settings
        app.state.model_gateway = ModelGateway.from_settings(settings)

    def tearDown(self) -> None:
        app.dependency_overrides.clear()
        clear_user_services()
        app.state.tutor_services.clear()
        app.state.model_gateway = self.original_gateway
        self.temporary_directory.cleanup()

    def test_authenticated_user_can_store_read_and_delete_only_metadata(self) -> None:
        secret = "sk-user-api-secret"
        cached_rag = Mock()
        app.state.rag_services[TEST_USER.user_id] = cached_rag
        app.state.agent_services[TEST_USER.user_id] = Mock()
        app.state.tutor_services[TEST_USER.user_id] = Mock()

        with self.assertLogs("zhixing.observability", level="INFO") as captured:
            with TestClient(app) as client:
                before = client.get("/api/model-gateway/credential")
                saved = client.put(
                    "/api/model-gateway/credential",
                    json={
                        "provider": "deepseek",
                        "model_name": "deepseek-v4-flash",
                        "api_key": secret,
                    },
                )
                loaded = client.get("/api/model-gateway/credential")
                deleted = client.delete("/api/model-gateway/credential")

        before_payload = before.json()
        saved_payload = saved.json()
        serialized = json.dumps(
            {
                "saved": saved_payload,
                "loaded": loaded.json(),
                "deleted": deleted.json(),
            },
            ensure_ascii=False,
        )
        logs = "\n".join(record.getMessage() for record in captured.records)
        database_bytes = self.settings.model_credentials_database_path.read_bytes()

        self.assertEqual(before.status_code, 200)
        self.assertFalse(before_payload["byok_configured"])
        self.assertEqual(before_payload["credential_source"], "system")
        self.assertEqual(saved.status_code, 200)
        self.assertTrue(saved_payload["byok_configured"])
        self.assertEqual(saved_payload["provider"], "deepseek")
        self.assertEqual(saved_payload["credential_source"], "byok")
        self.assertEqual(loaded.json(), saved_payload)
        self.assertEqual(deleted.status_code, 200)
        self.assertTrue(deleted.json()["deleted"])
        self.assertEqual(deleted.json()["active"]["credential_source"], "system")
        self.assertNotIn(secret, serialized)
        self.assertNotIn(secret, logs)
        self.assertNotIn(secret.encode("utf-8"), database_bytes)
        cached_rag.close.assert_called_once_with()
        self.assertNotIn(TEST_USER.user_id, app.state.rag_services)
        self.assertNotIn(TEST_USER.user_id, app.state.agent_services)
        self.assertNotIn(TEST_USER.user_id, app.state.tutor_services)

    def test_api_uses_authenticated_user_scope_for_credentials(self) -> None:
        other_user = UserRecord(
            user_id="33333333-3333-4333-8333-333333333333",
            email="other@example.com",
            password_hash="test-only-hash",
            display_name="其他用户",
            updated_at="2026-08-24T00:00:00.000Z",
            created_at="2026-08-24T00:00:00.000Z",
        )

        with TestClient(app) as client:
            saved = client.put(
                "/api/model-gateway/credential",
                json={
                    "provider": "qwen",
                    "model_name": "qwen-plus",
                    "api_key": "sk-first-user",
                },
            )
            app.dependency_overrides[get_current_user] = lambda: other_user
            other_selection = client.get("/api/model-gateway/credential")

        self.assertEqual(saved.status_code, 200)
        self.assertTrue(saved.json()["byok_configured"])
        self.assertEqual(other_selection.status_code, 200)
        self.assertFalse(other_selection.json()["byok_configured"])
        self.assertEqual(other_selection.json()["credential_source"], "system")

    def test_request_cannot_supply_arbitrary_base_url_or_extra_fields(self) -> None:
        with TestClient(app) as client:
            response = client.put(
                "/api/model-gateway/credential",
                json={
                    "provider": "openai_compatible",
                    "model_name": "model",
                    "api_key": "sk-user",
                    "base_url": "http://127.0.0.1:8000/private",
                },
            )

        self.assertEqual(response.status_code, 422)
        self.assertFalse(
            app.state.model_gateway.current_selection(TEST_USER.user_id).byok_configured
        )

    def test_missing_encryption_key_rejects_storage_without_leaking_key(self) -> None:
        secret = "sk-not-stored"
        app.state.model_gateway = ModelGateway.from_settings(
            Settings.from_mapping(
                {
                    "APP_DATA_DIR": str(self.root / "unconfigured"),
                    "DEFAULT_PROVIDER": "qwen",
                    "DEFAULT_MODEL": "qwen-plus",
                    "MODEL_API_KEY": "sk-system",
                    "MODEL_CREDENTIAL_ENCRYPTION_KEY": "",
                },
                project_root=self.root,
            )
        )

        with TestClient(app) as client:
            response = client.put(
                "/api/model-gateway/credential",
                json={
                    "provider": "qwen",
                    "model_name": "qwen-plus",
                    "api_key": secret,
                },
            )

        self.assertEqual(response.status_code, 503)
        self.assertNotIn(secret, response.text)
        self.assertFalse(
            app.state.model_gateway.current_selection(TEST_USER.user_id).byok_configured
        )

    def test_model_authentication_error_has_stable_safe_response(self) -> None:
        service = Mock()
        service.ask.side_effect = ModelAuthenticationError(
            "provider detail must not escape"
        )
        app.dependency_overrides[get_rag_service_provider] = lambda: lambda: service

        with TestClient(app) as client:
            response = client.post(
                "/api/ask",
                json={"question": "RAG 是什么？"},
            )

        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.json()["detail"], "Model authentication failed.")
        self.assertNotIn("provider detail", response.text)


if __name__ == "__main__":
    unittest.main()
