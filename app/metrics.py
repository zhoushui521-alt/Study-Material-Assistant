"""知行单实例进程内的匿名运行指标聚合。"""

from __future__ import annotations

from datetime import UTC, datetime
from threading import Lock
from time import perf_counter


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _average(total: int, count: int) -> float:
    return round(total / count, 2) if count else 0.0


def _rate(count: int, total: int) -> float:
    return round(count / total, 4) if total else 0.0


def _non_negative_int(value: int, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} 必须是非负整数。")
    return value


class RuntimeMetrics:
    """线程安全的单进程累计指标；重启后清零，不承担长期存储。"""

    def __init__(self) -> None:
        self._lock = Lock()
        self.reset()

    def reset(self) -> None:
        """清空指标；仅供进程初始化和隔离测试使用。"""
        with self._lock:
            self._started_at = _utc_now()
            self._started = perf_counter()
            self._http_total = 0
            self._http_succeeded = 0
            self._http_failed = 0
            self._http_duration_ms = 0
            self._retrieval_total = 0
            self._retrieval_failed = 0
            self._retrieval_empty = 0
            self._retrieved_chunks = 0
            self._retrieval_duration_ms = 0
            self._llm_total = 0
            self._llm_failed = 0
            self._llm_duration_ms = 0
            self._llm_usage_available = 0
            self._input_tokens = 0
            self._output_tokens = 0
            self._total_tokens = 0
            self._models: set[str] = set()
            self._document_completed = 0
            self._document_failed = 0
            self._document_duration_ms = 0

    def record_http(self, *, status_code: int, duration_ms: int) -> None:
        _non_negative_int(duration_ms, field="duration_ms")
        if isinstance(status_code, bool) or not isinstance(status_code, int):
            raise ValueError("status_code 必须是整数。")
        with self._lock:
            self._http_total += 1
            self._http_duration_ms += duration_ms
            if status_code < 400:
                self._http_succeeded += 1
            else:
                self._http_failed += 1

    def record_retrieval(
        self,
        *,
        retrieved_count: int,
        duration_ms: int,
        failed: bool = False,
    ) -> None:
        _non_negative_int(retrieved_count, field="retrieved_count")
        _non_negative_int(duration_ms, field="duration_ms")
        with self._lock:
            self._retrieval_total += 1
            self._retrieval_duration_ms += duration_ms
            self._retrieved_chunks += retrieved_count
            if failed:
                self._retrieval_failed += 1
            elif retrieved_count == 0:
                self._retrieval_empty += 1

    def record_llm(
        self,
        *,
        model: str,
        duration_ms: int,
        failed: bool = False,
        token_usage_available: bool = False,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        total_tokens: int | None = None,
    ) -> None:
        _non_negative_int(duration_ms, field="duration_ms")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model 不能为空。")
        token_values = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }
        for field, value in token_values.items():
            if value is not None:
                _non_negative_int(value, field=field)
        with self._lock:
            self._llm_total += 1
            self._llm_duration_ms += duration_ms
            self._models.add(model.strip()[:160])
            if failed:
                self._llm_failed += 1
            if token_usage_available:
                self._llm_usage_available += 1
                self._input_tokens += input_tokens or 0
                self._output_tokens += output_tokens or 0
                self._total_tokens += total_tokens or 0

    def record_document_job(self, *, status: str, duration_ms: int) -> None:
        _non_negative_int(duration_ms, field="duration_ms")
        if status not in {"completed", "failed"}:
            raise ValueError("只记录 completed 或 failed 文档任务。")
        with self._lock:
            self._document_duration_ms += duration_ms
            if status == "completed":
                self._document_completed += 1
            else:
                self._document_failed += 1

    def snapshot(self) -> dict[str, object]:
        """返回不含 request_id、user_id 或内容数据的稳定快照。"""
        with self._lock:
            http_total = self._http_total
            retrieval_total = self._retrieval_total
            llm_total = self._llm_total
            document_total = self._document_completed + self._document_failed
            return {
                "generated_at": _utc_now(),
                "process_started_at": self._started_at,
                "uptime_seconds": round(perf_counter() - self._started, 3),
                "system": {
                    "requests_total": http_total,
                    "requests_succeeded": self._http_succeeded,
                    "requests_failed": self._http_failed,
                    "success_rate": _rate(self._http_succeeded, http_total),
                    "error_rate": _rate(self._http_failed, http_total),
                    "average_duration_ms": _average(
                        self._http_duration_ms, http_total
                    ),
                },
                "retrieval": {
                    "calls_total": retrieval_total,
                    "calls_failed": self._retrieval_failed,
                    "empty_results": self._retrieval_empty,
                    "retrieved_chunks_total": self._retrieved_chunks,
                    "average_retrieved_chunks": _average(
                        self._retrieved_chunks, retrieval_total
                    ),
                    "average_duration_ms": _average(
                        self._retrieval_duration_ms, retrieval_total
                    ),
                },
                "llm": {
                    "calls_total": llm_total,
                    "calls_failed": self._llm_failed,
                    "models": sorted(self._models),
                    "average_duration_ms": _average(
                        self._llm_duration_ms, llm_total
                    ),
                    "token_usage_available_calls": self._llm_usage_available,
                    "input_tokens_total": self._input_tokens,
                    "output_tokens_total": self._output_tokens,
                    "tokens_total": self._total_tokens,
                },
                "document_processing": {
                    "completed_total": self._document_completed,
                    "failed_total": self._document_failed,
                    "average_duration_ms": _average(
                        self._document_duration_ms, document_total
                    ),
                },
            }


runtime_metrics = RuntimeMetrics()
