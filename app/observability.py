"""知行单实例运行时的结构化日志基础能力。"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import TextIO


SERVICE_NAME = "zhixing"
OBSERVABILITY_LOGGER_NAME = "zhixing.observability"
_HANDLER_MARKER = "_zhixing_observability_handler"
_EVENT_PATTERN = re.compile(r"^[a-z][a-z0-9_]{1,79}$")
_ALLOWED_DETAIL_FIELDS = frozenset(
    {
        "component",
        "count",
        "empty_retrieval",
        "error_category",
        "error_type",
        "input_tokens",
        "job_id",
        "job_status",
        "method",
        "model",
        "output_tokens",
        "path",
        "retrieved_count",
        "status_code",
        "token_usage_available",
        "total_tokens",
    }
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _safe_text(value: object, *, maximum: int = 160) -> str | None:
    if value is None:
        return None
    text = str(value).replace("\r", " ").replace("\n", " ").strip()
    if not text:
        return None
    return text[:maximum]


def build_log_record(
    event: str,
    *,
    level: int = logging.INFO,
    request_id: object = None,
    user_id: object = None,
    duration_ms: int | None = None,
    details: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """构造固定基线字段，并只接纳显式允许的非内容型详情。"""
    if not _EVENT_PATTERN.fullmatch(event):
        raise ValueError("可观测性事件名称无效。")
    if duration_ms is not None and (isinstance(duration_ms, bool) or duration_ms < 0):
        raise ValueError("duration_ms 必须是非负整数。")

    record: dict[str, object] = {
        "time": _utc_now(),
        "level": logging.getLevelName(level),
        "service": SERVICE_NAME,
        "request_id": _safe_text(request_id, maximum=80),
        "user_id": _safe_text(user_id, maximum=80),
        "event": event,
        "duration_ms": duration_ms,
    }
    for key, value in (details or {}).items():
        if key not in _ALLOWED_DETAIL_FIELDS:
            continue
        if isinstance(value, str):
            record[key] = _safe_text(value)
        elif value is None or isinstance(value, (bool, int, float)):
            record[key] = value
    return record


class JsonLogFormatter(logging.Formatter):
    """输出纯 JSON，方便 Docker、文件或未来 Collector 直接解析。"""

    def format(self, record: logging.LogRecord) -> str:
        payload = getattr(record, "observability_payload", None)
        if not isinstance(payload, dict):
            payload = build_log_record(
                "unstructured_log_rejected",
                level=record.levelno,
                details={"component": record.name},
            )
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def configure_observability_logging(
    *,
    level: int = logging.INFO,
    stream: TextIO | None = None,
) -> logging.Logger:
    """幂等初始化独立 Logger；不依赖 Uvicorn 的日志配置。"""
    logger = logging.getLogger(OBSERVABILITY_LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False
    logger.disabled = False
    if not any(getattr(handler, _HANDLER_MARKER, False) for handler in logger.handlers):
        handler = logging.StreamHandler(stream)
        handler.setFormatter(JsonLogFormatter())
        setattr(handler, _HANDLER_MARKER, True)
        logger.addHandler(handler)
    return logger


def log_event(
    event: str,
    *,
    level: int = logging.INFO,
    request_id: object = None,
    user_id: object = None,
    duration_ms: int | None = None,
    details: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """记录结构化事件并返回 payload，便于测试精确验证日志契约。"""
    payload = build_log_record(
        event,
        level=level,
        request_id=request_id,
        user_id=user_id,
        duration_ms=duration_ms,
        details=details,
    )
    configure_observability_logging().log(
        level,
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        extra={"observability_payload": payload},
    )
    return payload


configure_observability_logging()
