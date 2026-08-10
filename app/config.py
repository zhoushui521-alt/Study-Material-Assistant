"""读取本地环境变量；真实密钥只保存在项目根目录的 .env 中。"""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"


def load_local_env() -> None:
    """读取简单的 KEY=VALUE 格式 .env 文件，不覆盖系统已有环境变量。"""
    if not ENV_FILE.exists():
        return

    for raw_line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", maxsplit=1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
