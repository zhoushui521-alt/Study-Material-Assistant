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
                },
                project_root=project_root,
            )

        self.assertEqual(settings.app_host, "0.0.0.0")
        self.assertEqual(settings.app_port, 8012)
        self.assertEqual(settings.data_dir, (project_root / "runtime-data").resolve())
        self.assertEqual(settings.bailian_base_url, "https://example.com/v1")
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
