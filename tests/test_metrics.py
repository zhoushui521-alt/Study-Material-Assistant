import unittest

from app.metrics import RuntimeMetrics


class RuntimeMetricsTests(unittest.TestCase):
    def test_snapshot_aggregates_system_retrieval_llm_and_document_metrics(
        self,
    ) -> None:
        metrics = RuntimeMetrics()

        metrics.record_http(status_code=200, duration_ms=10)
        metrics.record_http(status_code=500, duration_ms=30)
        metrics.record_retrieval(retrieved_count=3, duration_ms=10)
        metrics.record_retrieval(retrieved_count=0, duration_ms=20)
        metrics.record_retrieval(
            retrieved_count=0,
            duration_ms=30,
            failed=True,
        )
        metrics.record_llm(
            model="qwen-test",
            duration_ms=40,
            token_usage_available=True,
            input_tokens=8,
            output_tokens=4,
            total_tokens=12,
        )
        metrics.record_llm(
            model="qwen-test",
            duration_ms=60,
            failed=True,
        )
        metrics.record_document_job(status="completed", duration_ms=100)
        metrics.record_document_job(status="failed", duration_ms=300)

        snapshot = metrics.snapshot()

        self.assertEqual(
            snapshot["system"],
            {
                "requests_total": 2,
                "requests_succeeded": 1,
                "requests_failed": 1,
                "success_rate": 0.5,
                "error_rate": 0.5,
                "average_duration_ms": 20.0,
            },
        )
        self.assertEqual(
            snapshot["retrieval"],
            {
                "calls_total": 3,
                "calls_failed": 1,
                "empty_results": 1,
                "retrieved_chunks_total": 3,
                "average_retrieved_chunks": 1.0,
                "average_duration_ms": 20.0,
            },
        )
        self.assertEqual(snapshot["llm"]["calls_total"], 2)
        self.assertEqual(snapshot["llm"]["calls_failed"], 1)
        self.assertEqual(snapshot["llm"]["models"], ["qwen-test"])
        self.assertEqual(snapshot["llm"]["average_duration_ms"], 50.0)
        self.assertEqual(snapshot["llm"]["token_usage_available_calls"], 1)
        self.assertEqual(snapshot["llm"]["input_tokens_total"], 8)
        self.assertEqual(snapshot["llm"]["output_tokens_total"], 4)
        self.assertEqual(snapshot["llm"]["tokens_total"], 12)
        self.assertEqual(
            snapshot["document_processing"],
            {
                "completed_total": 1,
                "failed_total": 1,
                "average_duration_ms": 200.0,
            },
        )
        self.assertNotIn("request_id", snapshot)
        self.assertNotIn("user_id", snapshot)

    def test_reset_clears_aggregates(self) -> None:
        metrics = RuntimeMetrics()
        metrics.record_http(status_code=204, duration_ms=1)

        metrics.reset()

        snapshot = metrics.snapshot()
        self.assertEqual(snapshot["system"]["requests_total"], 0)
        self.assertEqual(snapshot["retrieval"]["calls_total"], 0)
        self.assertEqual(snapshot["llm"]["calls_total"], 0)
        self.assertEqual(snapshot["document_processing"]["completed_total"], 0)

    def test_invalid_values_are_rejected_before_mutating_counters(self) -> None:
        metrics = RuntimeMetrics()

        with self.assertRaises(ValueError):
            metrics.record_retrieval(retrieved_count=-1, duration_ms=1)
        with self.assertRaises(ValueError):
            metrics.record_llm(model="", duration_ms=1)
        with self.assertRaises(ValueError):
            metrics.record_document_job(status="processing", duration_ms=1)

        snapshot = metrics.snapshot()
        self.assertEqual(snapshot["retrieval"]["calls_total"], 0)
        self.assertEqual(snapshot["llm"]["calls_total"], 0)
        self.assertEqual(snapshot["document_processing"]["completed_total"], 0)
        self.assertEqual(snapshot["document_processing"]["failed_total"], 0)


if __name__ == "__main__":
    unittest.main()
