"""从统一 Settings 启动 FastAPI；供本地与容器共用。"""

from __future__ import annotations

import uvicorn

from app.config import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "app.api:app",
        host=settings.app_host,
        port=settings.app_port,
        access_log=False,
    )


if __name__ == "__main__":
    main()
