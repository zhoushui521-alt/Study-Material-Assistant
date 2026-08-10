"""为本地 API 提供有限并发、频率和进程级费用保护。"""

from collections import deque
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from math import ceil
from threading import Lock
from time import monotonic
from typing import Literal

from app.security_limits import (
    AGENT_MAX_CALLS_PER_WINDOW,
    AGENT_MAX_UNITS_PER_PROCESS,
    AGENT_WINDOW_SECONDS,
    ASK_MAX_CALLS_PER_WINDOW,
    ASK_MAX_UNITS_PER_PROCESS,
    ASK_WINDOW_SECONDS,
    DELETE_MAX_CALLS_PER_WINDOW,
    DELETE_WINDOW_SECONDS,
    INDEX_MAX_BATCHES_PER_OPERATION,
    INDEX_MAX_BATCHES_PER_PROCESS,
    INDEX_MAX_CALLS_PER_WINDOW,
    INDEX_WINDOW_SECONDS,
    STAGE_MAX_CALLS_PER_WINDOW,
    STAGE_WINDOW_SECONDS,
    WEB_PREVIEW_MAX_CALLS_PER_WINDOW,
    WEB_PREVIEW_WINDOW_SECONDS,
    WORKFLOW_MAX_CALLS_PER_WINDOW,
    WORKFLOW_WINDOW_SECONDS,
)


OperationName = Literal[
    "ask",
    "agent",
    "index",
    "stage",
    "delete",
    "web_preview",
    "workflow",
]


@dataclass(frozen=True)
class OperationPolicy:
    window_seconds: float
    max_calls_per_window: int
    max_units_per_operation: int | None = None
    max_units_per_process: int | None = None


DEFAULT_OPERATION_POLICIES: Mapping[OperationName, OperationPolicy] = {
    "ask": OperationPolicy(
        window_seconds=ASK_WINDOW_SECONDS,
        max_calls_per_window=ASK_MAX_CALLS_PER_WINDOW,
        max_units_per_operation=1,
        max_units_per_process=ASK_MAX_UNITS_PER_PROCESS,
    ),
    "agent": OperationPolicy(
        window_seconds=AGENT_WINDOW_SECONDS,
        max_calls_per_window=AGENT_MAX_CALLS_PER_WINDOW,
        max_units_per_operation=1,
        max_units_per_process=AGENT_MAX_UNITS_PER_PROCESS,
    ),
    "index": OperationPolicy(
        window_seconds=INDEX_WINDOW_SECONDS,
        max_calls_per_window=INDEX_MAX_CALLS_PER_WINDOW,
        max_units_per_operation=INDEX_MAX_BATCHES_PER_OPERATION,
        max_units_per_process=INDEX_MAX_BATCHES_PER_PROCESS,
    ),
    "stage": OperationPolicy(
        window_seconds=STAGE_WINDOW_SECONDS,
        max_calls_per_window=STAGE_MAX_CALLS_PER_WINDOW,
    ),
    "delete": OperationPolicy(
        window_seconds=DELETE_WINDOW_SECONDS,
        max_calls_per_window=DELETE_MAX_CALLS_PER_WINDOW,
    ),
    "web_preview": OperationPolicy(
        window_seconds=WEB_PREVIEW_WINDOW_SECONDS,
        max_calls_per_window=WEB_PREVIEW_MAX_CALLS_PER_WINDOW,
    ),
    "workflow": OperationPolicy(
        window_seconds=WORKFLOW_WINDOW_SECONDS,
        max_calls_per_window=WORKFLOW_MAX_CALLS_PER_WINDOW,
    ),
}


class OperationProtectionError(RuntimeError):
    """操作被并发、频率或费用保护拒绝。"""

    def __init__(self, message: str, *, retry_after: int | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class OperationBusyError(OperationProtectionError):
    """另一个受保护操作仍在执行。"""


class OperationRateLimitError(OperationProtectionError):
    """滚动时间窗口内的操作次数已到上限。"""


class OperationBudgetExceededError(OperationProtectionError):
    """单次或当前进程累计费用单位已到上限。"""


class OperationLease:
    """一次独占操作持有的费用预留凭证。"""

    def __init__(
        self,
        guard: "OperationGuard",
        operation: OperationName,
        reserved_units: int,
    ) -> None:
        self._guard = guard
        self.operation = operation
        self._reserved_units = reserved_units
        self._active = True

    @property
    def reserved_units(self) -> int:
        return self._reserved_units

    def reserve_units(self, units: int) -> None:
        """在独占操作内追加真实费用单位，失败时不执行后续外部调用。"""
        self._guard._reserve_lease_units(self, units)


class OperationGuard:
    """以非阻塞互斥锁和内存计数保护昂贵或有状态操作。"""

    def __init__(
        self,
        *,
        policies: Mapping[OperationName, OperationPolicy] = DEFAULT_OPERATION_POLICIES,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self._policies = dict(policies)
        self._clock = clock
        self._exclusive_lock = Lock()
        self._state_lock = Lock()
        self._events = {name: deque[float]() for name in self._policies}
        self._used_units = {name: 0 for name in self._policies}
        self._active_lease: OperationLease | None = None

    @staticmethod
    def _validate_units(units: int) -> None:
        if isinstance(units, bool) or not isinstance(units, int) or units < 0:
            raise ValueError("操作费用单位必须是非负整数。")

    def _policy_for(self, operation: OperationName) -> OperationPolicy:
        try:
            return self._policies[operation]
        except KeyError as error:
            raise ValueError(f"未知受保护操作：{operation}。") from error

    def _reserve_call(self, operation: OperationName) -> None:
        policy = self._policy_for(operation)
        now = self._clock()
        events = self._events[operation]
        cutoff = now - policy.window_seconds
        while events and events[0] <= cutoff:
            events.popleft()
        if len(events) >= policy.max_calls_per_window:
            retry_after = max(1, ceil(events[0] + policy.window_seconds - now))
            raise OperationRateLimitError(
                "操作过于频繁，请稍后重试。",
                retry_after=retry_after,
            )
        events.append(now)

    def _reserve_units(
        self,
        operation: OperationName,
        units: int,
        *,
        already_reserved: int,
    ) -> None:
        self._validate_units(units)
        policy = self._policy_for(operation)
        if (
            policy.max_units_per_operation is not None
            and already_reserved + units > policy.max_units_per_operation
        ):
            raise OperationBudgetExceededError("单次操作超过费用保护上限。")
        used_units = self._used_units[operation]
        if (
            policy.max_units_per_process is not None
            and used_units + units > policy.max_units_per_process
        ):
            raise OperationBudgetExceededError(
                "当前服务进程的费用保护额度已用完，请确认后再重启服务。"
            )

        self._used_units[operation] = used_units + units

    def _reserve_lease_units(self, lease: OperationLease, units: int) -> None:
        with self._state_lock:
            if not lease._active or self._active_lease is not lease:
                raise RuntimeError("操作费用凭证已失效。")
            self._reserve_units(
                lease.operation,
                units,
                already_reserved=lease._reserved_units,
            )
            lease._reserved_units += units

    @contextmanager
    def acquire(
        self,
        operation: OperationName,
        *,
        units: int = 0,
    ) -> Iterator[OperationLease]:
        """立即取得独占执行权；繁忙时不排队，成功后保守计入调用额度。"""
        if not self._exclusive_lock.acquire(blocking=False):
            raise OperationBusyError(
                "服务正在处理另一个资料或问答操作，请稍后重试。",
                retry_after=1,
            )
        lease: OperationLease | None = None
        try:
            with self._state_lock:
                self._validate_units(units)
                self._reserve_call(operation)
                self._reserve_units(operation, units, already_reserved=0)
                lease = OperationLease(self, operation, units)
                self._active_lease = lease
            yield lease
        finally:
            if lease is not None:
                with self._state_lock:
                    lease._active = False
                    if self._active_lease is lease:
                        self._active_lease = None
            self._exclusive_lock.release()

    def used_units(self, operation: OperationName) -> int:
        """返回当前进程已保守计入的费用单位，仅用于状态展示和测试。"""
        with self._state_lock:
            if operation not in self._used_units:
                raise ValueError(f"未知受保护操作：{operation}。")
            return self._used_units[operation]
