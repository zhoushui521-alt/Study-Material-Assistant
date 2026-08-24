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


def _runtime_int(
    environment: Mapping[str, str],
    key: str,
    default: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    value = _required_runtime_value(environment, key, default)
    try:
        parsed = int(value)
    except ValueError as error:
        raise ConfigurationError(f"{key} 必须是整数。") from error
    if not minimum <= parsed <= maximum:
        raise ConfigurationError(f"{key} 必须介于 {minimum} 和 {maximum} 之间。")
    return parsed


def _optional_runtime_int(
    environment: Mapping[str, str],
    key: str,
    *,
    minimum: int,
    maximum: int,
) -> int | None:
    value = environment.get(key, "").strip()
    if not value:
        return None
    return _runtime_int(
        environment,
        key,
        value,
        minimum=minimum,
        maximum=maximum,
    )


def _runtime_float(
    environment: Mapping[str, str],
    key: str,
    default: str,
    *,
    minimum: float,
    maximum: float,
) -> float:
    value = _required_runtime_value(environment, key, default)
    try:
        parsed = float(value)
    except ValueError as error:
        raise ConfigurationError(f"{key} 必须是数字。") from error
    if not minimum <= parsed <= maximum:
        raise ConfigurationError(f"{key} 必须介于 {minimum} 和 {maximum} 之间。")
    return parsed


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
    default_model_provider: str
    default_model_name: str
    model_api_key: str = field(repr=False)
    model_base_url: str
    model_timeout_seconds: int
    model_max_tokens: int | None
    model_temperature: float
    model_credential_encryption_key: str = field(repr=False)

    @classmethod
    def from_mapping(
        cls,
        environment: Mapping[str, str],
        *,
        project_root: Path = PROJECT_ROOT,
    ) -> "Settings":
        default_model_provider = environment.get(
            "DEFAULT_PROVIDER",
            "qwen",
        ).strip().lower()
        use_legacy_qwen_config = default_model_provider == "qwen"
        default_model_name = environment.get("DEFAULT_MODEL", "").strip()
        model_api_key = environment.get("MODEL_API_KEY", "").strip()
        model_base_url = environment.get("MODEL_BASE_URL", "").strip()
        if use_legacy_qwen_config:
            default_model_name = default_model_name or environment.get(
                "BAILIAN_CHAT_MODEL",
                "qwen-plus",
            ).strip()
            model_api_key = model_api_key or environment.get(
                "BAILIAN_API_KEY",
                "",
            ).strip()
            model_base_url = model_base_url or environment.get(
                "BAILIAN_BASE_URL",
                "",
            ).strip()

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
            default_model_provider=default_model_provider,
            default_model_name=default_model_name,
            model_api_key=model_api_key,
            model_base_url=model_base_url.rstrip("/"),
            model_timeout_seconds=_runtime_int(
                environment,
                "MODEL_TIMEOUT",
                "60",
                minimum=1,
                maximum=600,
            ),
            model_max_tokens=_optional_runtime_int(
                environment,
                "MODEL_MAX_TOKENS",
                minimum=1,
                maximum=1_000_000,
            ),
            model_temperature=_runtime_float(
                environment,
                "MODEL_TEMPERATURE",
                "0.2",
                minimum=0.0,
                maximum=2.0,
            ),
            model_credential_encryption_key=environment.get(
                "MODEL_CREDENTIAL_ENCRYPTION_KEY",
                "",
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

    @property
    def model_credentials_database_path(self) -> Path:
        return self.data_dir / "model_gateway" / "model_credentials.sqlite3"


def get_settings() -> Settings:
    """加载 .env 后返回当前环境快照；不缓存，便于测试和显式进程配置。"""
    load_local_env()
    return Settings.from_mapping(os.environ)
