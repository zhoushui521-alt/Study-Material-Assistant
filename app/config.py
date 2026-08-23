"""统一读取本地与部署环境配置；真实密钥只保存在环境或项目根目录 .env。"""

from __future__ import annotations

import os
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"


class ConfigurationError(RuntimeError):
    """应用运行配置无效。"""


def load_local_env(
    env_file: Path = ENV_FILE,
    *,
    environ: MutableMapping[str, str] | None = None,
) -> None:
    """读取简单 KEY=VALUE .env，不覆盖进程已经提供的环境变量。"""
    target = os.environ if environ is None else environ
    if not env_file.exists():
        return

    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", maxsplit=1)
        normalized_key = key.strip()
        if normalized_key:
            target.setdefault(
                normalized_key,
                value.strip().strip('"').strip("'"),
            )


def _required_runtime_value(
    environment: Mapping[str, str],
    key: str,
    default: str,
) -> str:
    value = environment.get(key, default).strip()
    if not value:
        raise ConfigurationError(f"{key} 不能为空。")
    return value


def _runtime_port(environment: Mapping[str, str]) -> int:
    value = _required_runtime_value(environment, "APP_PORT", "8000")
    try:
        port = int(value)
    except ValueError as error:
        raise ConfigurationError("APP_PORT 必须是整数。") from error
    if not 1 <= port <= 65535:
        raise ConfigurationError("APP_PORT 必须介于 1 和 65535 之间。")
    return port


def _runtime_path(
    environment: Mapping[str, str],
    key: str,
    default: str,
    *,
    project_root: Path,
) -> Path:
    raw_value = _required_runtime_value(environment, key, default)
    path = Path(raw_value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


@dataclass(frozen=True)
class Settings:
    """单进程 FastAPI、SQLite、Chroma 与模型客户端共用的配置入口。"""

    app_host: str
    app_port: int
    data_dir: Path
    bailian_api_key: str = field(repr=False)
    bailian_base_url: str
    bailian_embedding_model: str
    bailian_embedding_dimensions: str
    bailian_chat_model: str

    @classmethod
    def from_mapping(
        cls,
        environment: Mapping[str, str],
        *,
        project_root: Path = PROJECT_ROOT,
    ) -> "Settings":
        return cls(
            app_host=_required_runtime_value(
                environment,
                "APP_HOST",
                "127.0.0.1",
            ),
            app_port=_runtime_port(environment),
            data_dir=_runtime_path(
                environment,
                "APP_DATA_DIR",
                "data",
                project_root=project_root,
            ),
            bailian_api_key=environment.get("BAILIAN_API_KEY", "").strip(),
            bailian_base_url=environment.get("BAILIAN_BASE_URL", "")
            .strip()
            .rstrip("/"),
            bailian_embedding_model=environment.get(
                "BAILIAN_EMBEDDING_MODEL",
                "text-embedding-v4",
            ).strip(),
            bailian_embedding_dimensions=environment.get(
                "BAILIAN_EMBEDDING_DIMENSIONS",
                "1024",
            ).strip(),
            bailian_chat_model=environment.get(
                "BAILIAN_CHAT_MODEL",
                "qwen-plus",
            ).strip(),
        )

    @property
    def learning_database_path(self) -> Path:
        return self.data_dir / "learning" / "learning.sqlite3"

    @property
    def document_job_database_path(self) -> Path:
        return self.data_dir / "jobs" / "document_jobs.sqlite3"

    @property
    def study_workflow_database_path(self) -> Path:
        return self.data_dir / "study_workflows" / "checkpoints.sqlite3"

    @property
    def user_workspaces_dir(self) -> Path:
        return self.data_dir / "user_workspaces"

    @property
    def request_log_path(self) -> Path:
        return self.data_dir / "request_logs" / "http_requests.jsonl"

    @property
    def crawl4ai_runtime_dir(self) -> Path:
        return self.data_dir / "crawl4ai_runtime"


def get_settings() -> Settings:
    """加载 .env 后返回当前环境快照；不缓存，便于测试和显式进程配置。"""
    load_local_env()
    return Settings.from_mapping(os.environ)
