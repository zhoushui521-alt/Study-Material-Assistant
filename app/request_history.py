"""将隐私安全的 HTTP 元数据写入可轮转 JSONL 历史。"""

import json
from collections.abc import Mapping
from pathlib import Path
from threading import Lock

if __package__:
    from app.config import PROJECT_ROOT
else:
    from config import PROJECT_ROOT


DEFAULT_REQUEST_LOG_PATH = (
    PROJECT_ROOT / "data" / "request_logs" / "http_requests.jsonl"
)
DEFAULT_MAX_LOG_BYTES = 1024 * 1024
DEFAULT_LOG_BACKUP_COUNT = 3
REQUEST_LOG_FIELDS = (
    "timestamp",
    "event",
    "request_id",
    "method",
    "path",
    "status_code",
    "elapsed_ms",
    "error_category",
)


class RequestHistoryError(RuntimeError):
    """请求历史记录不安全或无法写入。"""


def validate_request_record(record: Mapping[str, object]) -> None:
    """只允许固定的非内容字段进入持久化日志。"""
    if set(record) != set(REQUEST_LOG_FIELDS):
        raise RequestHistoryError("请求日志字段不符合隐私安全约定。")
    if record.get("event") != "http_request_completed":
        raise RequestHistoryError("请求日志事件类型无效。")
    path = record.get("path")
    if not isinstance(path, str) or "?" in path or len(path) > 120:
        raise RequestHistoryError("请求日志路由字段无效。")
    if path != "unmatched" and not path.startswith("/"):
        raise RequestHistoryError("请求日志只能记录路由模板。")
    method = record.get("method")
    if not isinstance(method, str) or not method.isalpha() or len(method) > 10:
        raise RequestHistoryError("请求日志方法字段无效。")
    for field in ("status_code", "elapsed_ms"):
        value = record.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RequestHistoryError(f"请求日志 {field} 字段无效。")
    for field in ("timestamp", "request_id"):
        value = record.get(field)
        if not isinstance(value, str) or not value or len(value) > 80:
            raise RequestHistoryError(f"请求日志 {field} 字段无效。")
    error_category = record.get("error_category")
    if error_category is not None and (
        not isinstance(error_category, str) or len(error_category) > 60
    ):
        raise RequestHistoryError("请求日志错误类别无效。")


class RequestHistoryWriter:
    """线程安全追加 JSONL，并按固定大小保留有限数量的历史文件。"""

    def __init__(
        self,
        path: Path = DEFAULT_REQUEST_LOG_PATH,
        *,
        max_bytes: int = DEFAULT_MAX_LOG_BYTES,
        backup_count: int = DEFAULT_LOG_BACKUP_COUNT,
    ) -> None:
        if max_bytes <= 0:
            raise ValueError("请求日志最大大小必须大于 0。")
        if backup_count < 0:
            raise ValueError("请求日志备份数量不能小于 0。")
        self.path = path
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self._lock = Lock()

    def _rotate(self) -> None:
        if self.backup_count == 0:
            self.path.write_text("", encoding="utf-8")
            return
        oldest = self.path.with_name(f"{self.path.name}.{self.backup_count}")
        if oldest.exists():
            oldest.unlink()
        for number in range(self.backup_count - 1, 0, -1):
            source = self.path.with_name(f"{self.path.name}.{number}")
            if source.exists():
                source.replace(self.path.with_name(f"{self.path.name}.{number + 1}"))
        if self.path.exists():
            self.path.replace(self.path.with_name(f"{self.path.name}.1"))

    def write(self, record: Mapping[str, object]) -> None:
        validate_request_record(record)
        serialized = json.dumps(
            dict(record),
            ensure_ascii=False,
            separators=(",", ":"),
        ) + "\n"
        encoded_size = len(serialized.encode("utf-8"))
        try:
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                current_size = self.path.stat().st_size if self.path.exists() else 0
                if current_size and current_size + encoded_size > self.max_bytes:
                    self._rotate()
                with self.path.open("a", encoding="utf-8", newline="\n") as output:
                    output.write(serialized)
        except OSError as error:
            raise RequestHistoryError("请求历史无法写入。") from error
