import json
import tempfile
import unittest
from pathlib import Path

from app.request_history import (
    RequestHistoryError,
    RequestHistoryWriter,
)


def safe_record(request_id: str = "request-1") -> dict[str, object]:
    return {
        "timestamp": "2026-08-09T10:00:00Z",
        "event": "http_request_completed",
        "request_id": request_id,
        "method": "POST",
        "path": "/api/ask",
        "status_code": 200,
        "elapsed_ms": 15,
        "error_category": None,
    }


class RequestHistoryTests(unittest.TestCase):
    def test_writes_one_utf8_json_object_per_line(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "requests.jsonl"
            writer = RequestHistoryWriter(path)

            writer.write(safe_record())

            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            self.assertEqual(json.loads(lines[0]), safe_record())

    def test_rejects_content_and_unknown_fields(self) -> None:
        record = {**safe_record(), "question": "不应记录的问题"}

        with self.assertRaises(RequestHistoryError):
            RequestHistoryWriter(Path("unused.jsonl")).write(record)

    def test_rotates_and_limits_backup_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "requests.jsonl"
            writer = RequestHistoryWriter(path, max_bytes=250, backup_count=2)

            writer.write(safe_record("request-1"))
            writer.write(safe_record("request-2"))
            writer.write(safe_record("request-3"))
            writer.write(safe_record("request-4"))

            self.assertTrue(path.is_file())
            self.assertTrue(path.with_name("requests.jsonl.1").is_file())
            self.assertTrue(path.with_name("requests.jsonl.2").is_file())
            self.assertFalse(path.with_name("requests.jsonl.3").exists())


if __name__ == "__main__":
    unittest.main()
