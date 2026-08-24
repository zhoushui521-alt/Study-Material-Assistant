import tempfile
import unittest
from pathlib import Path

from app.config import ConfigurationError, Settings, load_local_env


class SettingsTests(unittest.TestCase):
    def test_defaults_keep_local_single_instance_layout(self) -> None:
        settings = Settings.from_mapping({})

        self.assertEqual(settings.app_host, "127.0.0.1")
        self.assertEqual(settings.app_port, 8000)
        self.assertEqual(
            settings.learning_database_path,
            settings.data_dir / "learning" / "learning.sqlite3",
        )
        self.assertEqual(
            settings.document_job_database_path,
            settings.data_dir / "jobs" / "document_jobs.sqlite3",
        )
        self.assertEqual(
            settings.study_workflow_database_path,
            settings.data_dir / "study_workflows" / "checkpoints.sqlite3",
        )
        self.assertEqual(
            settings.user_workspaces_dir,
            settings.data_dir / "user_workspaces",
        )
        self.assertEqual(settings.default_model_provider, "qwen")
        self.assertEqual(settings.default_model_name, "qwen-plus")
        self.assertEqual(settings.model_timeout_seconds, 60)
        self.assertIsNone(settings.model_max_tokens)
        self.assertEqual(
            settings.model_credentials_database_path,
            settings.data_dir / "model_gateway" / "model_credentials.sqlite3",
        )

    def test_custom_data_directory_derives_all_persistent_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            settings = Settings.from_mapping(
                {
                    "APP_HOST": "0.0.0.0",
                    "APP_PORT": "8012",
                    "APP_DATA_DIR": "runtime-data",
                    "BAILIAN_BASE_URL": "https://example.com/v1/",
                    "BAILIAN_API_KEY": "test-secret-key",
                    "MODEL_BASE_URL": "",
                    "MODEL_API_KEY": "",
                },
                project_root=project_root,
            )

        self.assertEqual(settings.app_host, "0.0.0.0")
        self.assertEqual(settings.app_port, 8012)
        self.assertEqual(settings.data_dir, (project_root / "runtime-data").resolve())
        self.assertEqual(settings.bailian_base_url, "https://example.com/v1")
        self.assertEqual(settings.model_base_url, "https://example.com/v1")
        self.assertEqual(settings.model_api_key, "test-secret-key")
        self.assertNotIn("test-secret-key", repr(settings))
        self.assertEqual(
            settings.request_log_path,
            settings.data_dir / "request_logs" / "http_requests.jsonl",
        )
        self.assertEqual(
            settings.crawl4ai_runtime_dir,
            settings.data_dir / "crawl4ai_runtime",
        )

    def test_invalid_runtime_configuration_fails_clearly(self) -> None:
        with self.assertRaisesRegex(ConfigurationError, "APP_PORT"):
            Settings.from_mapping({"APP_PORT": "not-a-port"})
        with self.assertRaisesRegex(ConfigurationError, "APP_DATA_DIR"):
            Settings.from_mapping({"APP_DATA_DIR": "   "})
        with self.assertRaisesRegex(ConfigurationError, "MODEL_TIMEOUT"):
            Settings.from_mapping({"MODEL_TIMEOUT": "0"})
        with self.assertRaisesRegex(ConfigurationError, "MODEL_MAX_TOKENS"):
            Settings.from_mapping({"MODEL_MAX_TOKENS": "many"})
        with self.assertRaisesRegex(ConfigurationError, "MODEL_TEMPERATURE"):
            Settings.from_mapping({"MODEL_TEMPERATURE": "2.1"})

    def test_model_gateway_configuration_is_normalized_without_secret_repr(self) -> None:
        settings = Settings.from_mapping(
            {
                "DEFAULT_PROVIDER": " DeepSeek ",
                "DEFAULT_MODEL": " deepseek-v4-flash ",
                "MODEL_API_KEY": "sk-model-secret",
                "MODEL_BASE_URL": "https://api.example.com/v1/",
                "MODEL_TIMEOUT": "45",
                "MODEL_MAX_TOKENS": "4096",
                "MODEL_TEMPERATURE": "0.4",
                "MODEL_CREDENTIAL_ENCRYPTION_KEY": "credential-master-secret",
            }
        )

        self.assertEqual(settings.default_model_provider, "deepseek")
        self.assertEqual(settings.default_model_name, "deepseek-v4-flash")
        self.assertEqual(settings.model_base_url, "https://api.example.com/v1")
        self.assertEqual(settings.model_timeout_seconds, 45)
        self.assertEqual(settings.model_max_tokens, 4096)
        self.assertEqual(settings.model_temperature, 0.4)
        self.assertNotIn("sk-model-secret", repr(settings))
        self.assertNotIn("credential-master-secret", repr(settings))

    def test_legacy_qwen_key_never_falls_back_to_another_provider(self) -> None:
        settings = Settings.from_mapping(
            {
                "DEFAULT_PROVIDER": "deepseek",
                "DEFAULT_MODEL": "deepseek-chat",
                "MODEL_API_KEY": "",
                "MODEL_BASE_URL": "",
                "BAILIAN_API_KEY": "sk-qwen-must-not-cross-provider",
                "BAILIAN_BASE_URL": "https://dashscope.example.com/v1",
            }
        )

        self.assertEqual(settings.default_model_provider, "deepseek")
        self.assertEqual(settings.model_api_key, "")
        self.assertEqual(settings.model_base_url, "")
        self.assertNotIn("sk-qwen-must-not-cross-provider", repr(settings))

    def test_local_env_does_not_override_process_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text(
                "EXISTING=file-value\nNEW_VALUE=from-file\n",
                encoding="utf-8",
            )
            environment = {"EXISTING": "process-value"}

            load_local_env(env_file, environ=environment)

        self.assertEqual(environment["EXISTING"], "process-value")
        self.assertEqual(environment["NEW_VALUE"], "from-file")


if __name__ == "__main__":
    unittest.main()
