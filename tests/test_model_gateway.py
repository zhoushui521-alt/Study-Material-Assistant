import sqlite3
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import Mock, patch

from cryptography.fernet import Fernet

from app.config import Settings
from app.model_gateway import (
    ModelCredentialDecryptionError,
    ModelCredentialEncryptionUnavailableError,
    ModelCredentialStoreError,
    ModelGateway,
    OpenAICompatibleProviderAdapter,
    UnsupportedModelProviderError,
    is_model_authentication_error,
)
from app.model_gateway.credentials import ModelCredentialStore
from app.model_gateway.providers import ModelRoute


class ModelGatewayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.project_root = Path(self.temporary_directory.name)
        self.user_id = str(uuid.uuid4())
        self.encryption_key = Fernet.generate_key().decode("ascii")

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def settings(self, **overrides: str) -> Settings:
        environment = {
            "APP_DATA_DIR": str(self.project_root / "data"),
            "DEFAULT_PROVIDER": "qwen",
            "DEFAULT_MODEL": "qwen-plus",
            "MODEL_API_KEY": "sk-system-secret",
            "MODEL_BASE_URL": "https://workspace.example.com/compatible-mode/v1/",
            "MODEL_TIMEOUT": "45",
            "MODEL_MAX_TOKENS": "2048",
            "MODEL_TEMPERATURE": "0.3",
            "MODEL_CREDENTIAL_ENCRYPTION_KEY": self.encryption_key,
        }
        environment.update(overrides)
        return Settings.from_mapping(environment, project_root=self.project_root)

    def test_system_route_uses_unified_configuration(self) -> None:
        gateway = ModelGateway.from_settings(self.settings())

        route = gateway.route(self.user_id)

        self.assertEqual(route.provider, "qwen")
        self.assertEqual(route.model_name, "qwen-plus")
        self.assertEqual(route.base_url, "https://workspace.example.com/compatible-mode/v1")
        self.assertEqual(route.timeout_seconds, 45)
        self.assertEqual(route.max_tokens, 2048)
        self.assertEqual(route.temperature, 0.3)
        self.assertEqual(route.credential_source, "system")
        self.assertNotIn("sk-system-secret", repr(route))

    def test_byok_is_encrypted_routed_and_deleted(self) -> None:
        settings = self.settings()
        gateway = ModelGateway.from_settings(settings)
        secret = "sk-user-secret-value"

        metadata = gateway.save_credential(
            user_id=self.user_id,
            provider="deepseek",
            model_name="deepseek-v4-flash",
            api_key=secret,
        )
        route = gateway.route(self.user_id)
        database_bytes = settings.model_credentials_database_path.read_bytes()

        self.assertEqual(metadata.user_id, self.user_id)
        self.assertEqual(route.provider, "deepseek")
        self.assertEqual(route.model_name, "deepseek-v4-flash")
        self.assertEqual(route.base_url, "https://api.deepseek.com")
        self.assertEqual(route.api_key, secret)
        self.assertEqual(route.credential_source, "byok")
        self.assertNotIn(secret.encode("utf-8"), database_bytes)
        self.assertFalse(hasattr(metadata, "encrypted_api_key"))
        self.assertTrue(gateway.delete_credential(self.user_id))
        self.assertEqual(gateway.route(self.user_id).credential_source, "system")
        self.assertFalse(gateway.delete_credential(self.user_id))

    def test_credentials_are_strictly_user_scoped(self) -> None:
        gateway = ModelGateway.from_settings(self.settings())
        other_user_id = str(uuid.uuid4())
        gateway.save_credential(
            user_id=self.user_id,
            provider="openai",
            model_name="gpt-4.1-mini",
            api_key="sk-user-a",
        )

        first = gateway.current_selection(self.user_id)
        second = gateway.current_selection(other_user_id)

        self.assertTrue(first.byok_configured)
        self.assertEqual(first.credential_source, "byok")
        self.assertFalse(second.byok_configured)
        self.assertEqual(second.credential_source, "system")

    def test_missing_or_rotated_encryption_key_fails_without_system_fallback(self) -> None:
        gateway = ModelGateway.from_settings(self.settings())
        gateway.save_credential(
            user_id=self.user_id,
            provider="qwen",
            model_name="qwen-plus",
            api_key="sk-user-a",
        )
        rotated = ModelGateway.from_settings(
            self.settings(
                MODEL_CREDENTIAL_ENCRYPTION_KEY=Fernet.generate_key().decode("ascii")
            )
        )
        unconfigured = ModelGateway.from_settings(
            self.settings(MODEL_CREDENTIAL_ENCRYPTION_KEY="")
        )

        with self.assertRaises(ModelCredentialDecryptionError):
            rotated.route(self.user_id)
        with self.assertRaises(ModelCredentialDecryptionError):
            unconfigured.route(self.user_id)
        with self.assertRaises(ModelCredentialEncryptionUnavailableError):
            unconfigured.save_credential(
                user_id=str(uuid.uuid4()),
                provider="qwen",
                model_name="qwen-plus",
                api_key="sk-user-b",
            )

    def test_provider_and_model_inputs_are_bounded(self) -> None:
        gateway = ModelGateway.from_settings(self.settings())

        with self.assertRaises(UnsupportedModelProviderError):
            gateway.save_credential(
                user_id=self.user_id,
                provider="arbitrary_remote",
                model_name="model",
                api_key="sk-user-a",
            )
        with self.assertRaises(ValueError):
            gateway.save_credential(
                user_id=self.user_id,
                provider="qwen",
                model_name="bad model name",
                api_key="sk-user-a",
            )
        with self.assertRaises(ValueError):
            gateway.save_credential(
                user_id=self.user_id,
                provider="qwen",
                model_name="qwen-plus",
                api_key="contains whitespace",
            )

    def test_openai_compatible_adapter_only_translates_model_parameters(self) -> None:
        route = ModelRoute(
            provider="qwen",
            model_name="qwen-test",
            api_key="sk-secret",
            base_url="https://example.com/v1",
            temperature=0.2,
            timeout_seconds=30,
            max_tokens=512,
            credential_source="byok",
        )
        model = Mock()
        with patch(
            "app.model_gateway.providers.ChatOpenAI",
            return_value=model,
        ) as chat_openai:
            actual = OpenAICompatibleProviderAdapter().create_chat_model(route)

        self.assertIs(actual, model)
        chat_openai.assert_called_once_with(
            api_key="sk-secret",
            model="qwen-test",
            temperature=0.2,
            timeout=30,
            max_retries=2,
            base_url="https://example.com/v1",
            max_completion_tokens=512,
        )
        self.assertEqual(model._model_gateway_provider, "qwen")
        self.assertEqual(model._model_gateway_credential_source, "byok")

    def test_connection_is_closed_when_schema_initialization_fails(self) -> None:
        store = ModelCredentialStore(self.project_root / "broken" / "credentials.sqlite3")
        connection = Mock()

        with patch(
            "app.model_gateway.credentials.sqlite3.connect",
            return_value=connection,
        ), patch.object(
            store,
            "_apply_schema",
            side_effect=RuntimeError("schema failed"),
        ):
            with self.assertRaises(ModelCredentialStoreError):
                store._connect()

        connection.close.assert_called_once_with()

    def test_authentication_error_detection_uses_status_not_message(self) -> None:
        error = RuntimeError("secret provider detail")
        error.status_code = 401

        self.assertTrue(is_model_authentication_error(error))
        self.assertFalse(is_model_authentication_error(RuntimeError("401 in text")))


if __name__ == "__main__":
    unittest.main()
