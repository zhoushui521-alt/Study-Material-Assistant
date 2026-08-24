import io
import json
import logging
import unittest
from unittest.mock import Mock

from langchain_core.messages import AIMessage

from app.metrics import runtime_metrics
from app.observability import (
    OBSERVABILITY_LOGGER_NAME,
    JsonLogFormatter,
    build_log_record,
    configure_observability_logging,
    invoke_observed_model,
    log_event,
    observation_context,
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


    def test_context_is_applied_to_events_and_restored(self) -> None:
        with self.assertLogs(OBSERVABILITY_LOGGER_NAME, level="INFO") as captured:
            with observation_context(request_id="request-1", user_id="user-1"):
                log_event("context_probe")
            log_event("context_probe")

        inside, outside = [json.loads(record.getMessage()) for record in captured.records]
        self.assertEqual(inside["request_id"], "request-1")
        self.assertEqual(inside["user_id"], "user-1")
        self.assertIsNone(outside["request_id"])
        self.assertIsNone(outside["user_id"])

    def test_model_invoke_records_available_token_usage(self) -> None:
        model = Mock()
        model.model_name = "qwen-test"
        result = AIMessage(
            content="answer",
            usage_metadata={
                "input_tokens": 4,
                "output_tokens": 2,
                "total_tokens": 6,
            },
        )
        model.invoke.return_value = result

        with observation_context(request_id="request-2", user_id="user-2"):
            with self.assertLogs(
                OBSERVABILITY_LOGGER_NAME, level="INFO"
            ) as captured:
                actual = invoke_observed_model(
                    model,
                    "safe prompt placeholder",
                    component="rag",
                )

        payloads = [json.loads(record.getMessage()) for record in captured.records]
        self.assertIs(actual, result)
        self.assertEqual(
            [payload["event"] for payload in payloads],
            ["llm_call_started", "llm_call_completed"],
        )
        self.assertEqual(payloads[-1]["model"], "qwen-test")
        self.assertEqual(payloads[-1]["input_tokens"], 4)
        self.assertEqual(payloads[-1]["output_tokens"], 2)
        self.assertEqual(payloads[-1]["total_tokens"], 6)
        self.assertTrue(payloads[-1]["token_usage_available"])
        self.assertTrue(all(item["request_id"] == "request-2" for item in payloads))

    def test_model_failure_log_does_not_include_exception_detail(self) -> None:
        runtime_metrics.reset()
        model = Mock()
        model.model_name = "qwen-test"
        model.invoke.side_effect = RuntimeError("api_key=secret-value")

        with self.assertLogs(OBSERVABILITY_LOGGER_NAME, level="ERROR") as captured:
            with self.assertRaises(RuntimeError):
                invoke_observed_model(model, "prompt", component="rag")

        serialized = captured.records[-1].getMessage()
        payload = json.loads(serialized)
        self.assertEqual(payload["event"], "llm_call_failed")
        self.assertEqual(payload["error_type"], "RuntimeError")
        self.assertNotIn("secret-value", serialized)
        llm_metrics = runtime_metrics.snapshot()["llm"]
        self.assertEqual(llm_metrics["calls_total"], 1)
        self.assertEqual(llm_metrics["calls_failed"], 1)

if __name__ == "__main__":
    unittest.main()
