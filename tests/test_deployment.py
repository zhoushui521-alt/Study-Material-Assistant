import unittest
from pathlib import Path
from unittest.mock import patch

from app.config import Settings
from app.server import main


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DeploymentContractTests(unittest.TestCase):
    def test_dockerfile_runs_non_root_server_with_healthcheck(self) -> None:
        dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")

        self.assertIn("USER 10001:10001", dockerfile)
        self.assertIn('CMD ["python", "-m", "app.server"]', dockerfile)
        self.assertIn("/health", dockerfile)
        self.assertNotIn("COPY . .", dockerfile)

    def test_compose_keeps_current_single_process_architecture(self) -> None:
        compose = (PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8")

        self.assertIn("\n  app:\n", compose)
        self.assertIn("APP_DATA_DIR:", compose)
        self.assertIn("zhixing-data:/app/data", compose)
        self.assertIn("/health", compose)
        self.assertNotIn("\n  frontend:\n", compose)
        self.assertNotIn("\n  worker:\n", compose)
        self.assertNotIn("\n  postgres:\n", compose)
        self.assertNotIn("\n  redis:\n", compose)

    def test_docker_context_excludes_local_secrets_and_data(self) -> None:
        ignored = {
            line.strip()
            for line in (PROJECT_ROOT / ".dockerignore")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        }

        self.assertIn(".env", ignored)
        self.assertIn("data", ignored)
        self.assertIn(".git", ignored)

    @patch("app.server.uvicorn.run")
    @patch("app.server.get_settings")
    def test_server_uses_settings_and_disables_raw_access_log(
        self,
        get_settings,
        uvicorn_run,
    ) -> None:
        settings = Settings.from_mapping(
            {
                "APP_HOST": "0.0.0.0",
                "APP_PORT": "8012",
            }
        )
        get_settings.return_value = settings

        main()

        uvicorn_run.assert_called_once_with(
            "app.api:app",
            host="0.0.0.0",
            port=8012,
            access_log=False,
        )


if __name__ == "__main__":
    unittest.main()
