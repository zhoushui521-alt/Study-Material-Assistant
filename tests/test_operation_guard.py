import unittest

from app.operation_guard import (
    OperationBudgetExceededError,
    OperationBusyError,
    OperationGuard,
    OperationPolicy,
    OperationRateLimitError,
)


class MutableClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class OperationGuardTests(unittest.TestCase):
    def make_guard(
        self,
        clock: MutableClock,
        *,
        max_calls: int = 2,
        per_operation: int = 2,
        per_process: int = 3,
    ) -> OperationGuard:
        return OperationGuard(
            policies={
                "ask": OperationPolicy(
                    window_seconds=60,
                    max_calls_per_window=max_calls,
                    max_units_per_operation=per_operation,
                    max_units_per_process=per_process,
                )
            },
            clock=clock,
        )

    def test_rolling_window_rejects_then_allows_after_expiry(self) -> None:
        clock = MutableClock()
        guard = self.make_guard(clock)
        with guard.acquire("ask", units=1):
            pass
        clock.now = 1
        with guard.acquire("ask", units=1):
            pass

        with self.assertRaises(OperationRateLimitError) as captured:
            with guard.acquire("ask", units=0):
                pass

        self.assertEqual(captured.exception.retry_after, 59)
        clock.now = 60
        with guard.acquire("ask", units=0):
            pass

    def test_rejects_single_operation_over_budget(self) -> None:
        guard = self.make_guard(MutableClock())
        with self.assertRaises(OperationBudgetExceededError):
            with guard.acquire("ask", units=3):
                pass
        self.assertEqual(guard.used_units("ask"), 0)

    def test_rejects_process_budget_and_keeps_conservative_count(self) -> None:
        guard = self.make_guard(MutableClock(), max_calls=10)
        with self.assertRaisesRegex(RuntimeError, "operation failed"):
            with guard.acquire("ask", units=2):
                raise RuntimeError("operation failed")

        self.assertEqual(guard.used_units("ask"), 2)
        with self.assertRaises(OperationBudgetExceededError):
            with guard.acquire("ask", units=2):
                pass
        self.assertEqual(guard.used_units("ask"), 2)

    def test_busy_operation_is_rejected_without_queueing(self) -> None:
        guard = self.make_guard(MutableClock())
        with guard.acquire("ask", units=1):
            with self.assertRaises(OperationBusyError) as captured:
                with guard.acquire("ask", units=1):
                    pass

        self.assertEqual(captured.exception.retry_after, 1)
        self.assertEqual(guard.used_units("ask"), 1)

    def test_active_lease_can_reserve_actual_units_before_external_call(self) -> None:
        guard = self.make_guard(MutableClock(), max_calls=10)
        with guard.acquire("ask", units=0) as lease:
            lease.reserve_units(2)
            self.assertEqual(lease.reserved_units, 2)

        self.assertEqual(guard.used_units("ask"), 2)
        with self.assertRaises(RuntimeError):
            lease.reserve_units(1)


if __name__ == "__main__":
    unittest.main()
