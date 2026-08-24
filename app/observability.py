"""知行单实例运行时的结构化日志基础能力。"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar, Token
from datetime import UTC, datetime
from time import perf_counter
from typing import TextIO

from app.metrics import runtime_metrics
from app.model_gateway import ModelAuthenticationError, is_model_authentication_error


SERVICE_NAME = "zhixing"
OBSERVABILITY_LOGGER_NAME = "zhixing.observability"
_HANDLER_MARKER = "_zhixing_observability_handler"
_EVENT_PATTERN = re.compile(r"^[a-z][a-z0-9_]{1,79}$")
_ALLOWED_DETAIL_FIELDS = frozenset(
    {
        "component",
        "count",
        "credential_source",
        "empty_retrieval",
        "error_category",
        "error_type",
        "input_tokens",
        "job_id",
        "job_status",
        "method",
        "model",
        "operation",
        "output_tokens",
        "path",
        "provider",
        "retrieved_count",
        "status_code",
        "token_usage_available",
        "total_tokens",
    }
)
_CURRENT_REQUEST_ID: ContextVar[str | None] = ContextVar(
    "observability_request_id", default=None
)
_CURRENT_USER_ID: ContextVar[str | None] = ContextVar(
    "observability_user_id", default=None
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


def set_request_context(request_id: object) -> Token[str | None]:
    """绑定当前请求 ID；asyncio.to_thread 会安全复制该上下文。"""
    return _CURRENT_REQUEST_ID.set(_safe_text(request_id, maximum=80))


def set_user_context(user_id: object) -> Token[str | None]:
    """在后端认证成功后绑定可信用户 ID。"""
    return _CURRENT_USER_ID.set(_safe_text(user_id, maximum=80))


def reset_request_context(token: Token[str | None]) -> None:
    _CURRENT_REQUEST_ID.reset(token)


def reset_user_context(token: Token[str | None]) -> None:
    _CURRENT_USER_ID.reset(token)


@contextmanager
def observation_context(
    *, request_id: object = None, user_id: object = None
) -> Iterator[None]:
    """为测试、CLI 或后台操作显式建立并在退出时清理观察上下文。"""
    request_token = set_request_context(request_id)
    user_token = set_user_context(user_id)
    try:
        yield
    finally:
        reset_user_context(user_token)
        reset_request_context(request_token)


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
    if duration_ms is not None and (
        isinstance(duration_ms, bool)
        or not isinstance(duration_ms, int)
        or duration_ms < 0
    ):
        raise ValueError("duration_ms 必须是非负整数。")
    if request_id is None:
        request_id = _CURRENT_REQUEST_ID.get()
    if user_id is None:
        user_id = _CURRENT_USER_ID.get()

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


def model_identifier(model: object) -> str:
    """只读取非敏感模型标识；无法识别时退回类名。"""
    for attribute in ("model_name", "model"):
        value = getattr(model, attribute, None)
        if isinstance(value, str) and value.strip():
            return _safe_text(value) or type(model).__name__
    params = getattr(model, "_identifying_params", None)
    if isinstance(params, Mapping):
        for key in ("model_name", "model"):
            value = params.get(key)
            if isinstance(value, str) and value.strip():
                return _safe_text(value) or type(model).__name__
    return type(model).__name__


def provider_identifier(model: object) -> str:
    """读取网关附加的非敏感 Provider 标识。"""
    value = getattr(model, "_model_gateway_provider", None)
    return _safe_text(value) if isinstance(value, str) else "unknown"


def credential_source_identifier(model: object) -> str | None:
    """只允许 system/byok 枚举进入日志。"""
    value = getattr(model, "_model_gateway_credential_source", None)
    return value if value in {"system", "byok"} else None


def token_usage_details(result: object) -> dict[str, object]:
    """兼容 LangChain 常见 usage 位置；不可获得时明确标记而不估算。"""
    usage = getattr(result, "usage_metadata", None)
    if not isinstance(usage, Mapping):
        response_metadata = getattr(result, "response_metadata", None)
        if isinstance(response_metadata, Mapping):
            candidate = response_metadata.get("token_usage")
            usage = candidate if isinstance(candidate, Mapping) else None
    if not isinstance(usage, Mapping):
        return {"token_usage_available": False}

    def token_value(*keys: str) -> int | None:
        for key in keys:
            value = usage.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                return value
        return None

    input_tokens = token_value("input_tokens", "prompt_tokens")
    output_tokens = token_value("output_tokens", "completion_tokens")
    total_tokens = token_value("total_tokens")
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    available = any(
        value is not None for value in (input_tokens, output_tokens, total_tokens)
    )
    return {
        "token_usage_available": available,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def invoke_observed_model(
    model: object,
    input_value: object,
    *,
    component: str,
    model_name: str | None = None,
    provider_name: str | None = None,
) -> object:
    """保持原 invoke 契约，只在调用边界增加耗时、错误与 usage 事件。"""
    identifier = model_name or model_identifier(model)
    provider = provider_name or provider_identifier(model)
    credential_source = credential_source_identifier(model)
    details: dict[str, object] = {
        "component": component,
        "model": identifier,
        "provider": provider,
    }
    if credential_source is not None:
        details["credential_source"] = credential_source
    started = perf_counter()
    log_event("llm_call_started", details=details)
    try:
        result = model.invoke(input_value)
    except Exception as error:
        duration_ms = round((perf_counter() - started) * 1000)
        runtime_metrics.record_llm(
            model=identifier,
            provider=provider,
            duration_ms=duration_ms,
            failed=True,
        )
        log_event(
            "llm_call_failed",
            level=logging.ERROR,
            duration_ms=duration_ms,
            details={
                **details,
                "error_type": type(error).__name__,
            },
        )
        if is_model_authentication_error(error):
            raise ModelAuthenticationError("Model authentication failed.") from error
        raise
    duration_ms = round((perf_counter() - started) * 1000)
    usage = token_usage_details(result)
    runtime_metrics.record_llm(
        model=identifier,
        provider=provider,
        duration_ms=duration_ms,
        token_usage_available=bool(usage["token_usage_available"]),
        input_tokens=usage.get("input_tokens"),
        output_tokens=usage.get("output_tokens"),
        total_tokens=usage.get("total_tokens"),
    )
    log_event(
        "llm_call_completed",
        duration_ms=duration_ms,
        details={
            **details,
            **usage,
        },
    )
    return result


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
