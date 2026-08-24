import io
import json
import logging
import unittest

from app.observability import (
    JsonLogFormatter,
    build_log_record,
    configure_observability_logging,
)


class ObservabilityLoggingTests(unittest.TestCase):
    def test_log_record_contains_required_structured_fields(self) -> None:
        payload = build_log_record(
            "retrieval_completed",
            request_id="request-1",
            user_id="user-1",
            duration_ms=12,
            details={"retrieved_count": 3},
        )

        self.assertTrue(
            set(payload).issuperset(
                {
                    "time",
                    "level",
                    "service",
                    "request_id",
                    "user_id",
                    "event",
                    "duration_ms",
                }
            )
        )
        self.assertEqual(payload["event"], "retrieval_completed")
        self.assertEqual(payload["retrieved_count"], 3)
        self.assertNotIn("question", payload)

    def test_unknown_content_fields_are_not_serialized(self) -> None:
        payload = build_log_record(
            "http_request_completed",
            details={
                "path": "/api/ask",
                "question": "不应进入日志的问题内容",
            },
        )

        self.assertEqual(payload["path"], "/api/ask")
        self.assertNotIn("question", payload)
        self.assertNotIn("不应进入日志", json.dumps(payload, ensure_ascii=False))

    def test_formatter_outputs_one_json_object(self) -> None:
        payload = build_log_record("http_request_started", request_id="request-1")
        record = logging.LogRecord(
            "test",
            logging.INFO,
            __file__,
            1,
            "ignored",
            (),
            None,
        )
        record.observability_payload = payload

        serialized = JsonLogFormatter().format(record)

        self.assertEqual(json.loads(serialized), payload)
        self.assertNotIn("\n", serialized)

    def test_logging_initialization_is_idempotent(self) -> None:
        stream = io.StringIO()
        logger = configure_observability_logging(stream=stream)
        marked_before = [
            handler
            for handler in logger.handlers
            if getattr(handler, "_zhixing_observability_handler", False)
        ]

        configure_observability_logging(stream=stream)

        marked_after = [
            handler
            for handler in logger.handlers
            if getattr(handler, "_zhixing_observability_handler", False)
        ]
        self.assertEqual(len(marked_before), 1)
        self.assertEqual(marked_after, marked_before)


if __name__ == "__main__":
    unittest.main()
