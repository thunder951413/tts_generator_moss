"""Cross-client playback admission and global stop coordination."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class _PlaybackWaiter:
    sequence: int
    job_id: str
    caller_kind: str
    stop_epoch: int


class PlaybackCoordinator:
    """Grant one audible playback lease with internal-priority FIFO fairness."""

    def __init__(self, *, max_waiters: int = 64, internal_burst_limit: int = 4) -> None:
        self.max_waiters = max(1, int(max_waiters))
        self.internal_burst_limit = max(1, int(internal_burst_limit))
        self._condition = threading.Condition()
        self._waiters: list[_PlaybackWaiter] = []
        self._sequence = 0
        self._active_job_id = ""
        self._active_caller_kind = ""
        self._internal_burst = 0
        self._stop_epoch = 0

    @staticmethod
    def _kind(value: str) -> str:
        return "external" if str(value).lower() == "external" else "internal"

    def _selected_waiter(self) -> _PlaybackWaiter | None:
        internal = [item for item in self._waiters if item.caller_kind == "internal"]
        external = [item for item in self._waiters if item.caller_kind == "external"]
        if internal and (not external or self._internal_burst < self.internal_burst_limit):
            return internal[0]
        if external:
            return external[0]
        return internal[0] if internal else None

    def acquire(
        self,
        job_id: str,
        caller_kind: str,
        *,
        cancelled: Callable[[], bool] | None = None,
        timeout: float = 600.0,
    ) -> bool:
        caller_kind = self._kind(caller_kind)
        deadline = time.monotonic() + max(1.0, float(timeout))
        with self._condition:
            if self._active_job_id == job_id:
                return True
            if len(self._waiters) >= self.max_waiters:
                return False
            self._sequence += 1
            waiter = _PlaybackWaiter(
                sequence=self._sequence,
                job_id=str(job_id),
                caller_kind=caller_kind,
                stop_epoch=self._stop_epoch,
            )
            self._waiters.append(waiter)
            try:
                while True:
                    if waiter.stop_epoch != self._stop_epoch or (cancelled and cancelled()):
                        return False
                    selected = self._selected_waiter()
                    if not self._active_job_id and selected == waiter:
                        self._waiters.remove(waiter)
                        self._active_job_id = waiter.job_id
                        self._active_caller_kind = waiter.caller_kind
                        if waiter.caller_kind == "internal" and any(
                            item.caller_kind == "external" for item in self._waiters
                        ):
                            self._internal_burst += 1
                        elif waiter.caller_kind == "external":
                            self._internal_burst = 0
                        return True
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return False
                    self._condition.wait(timeout=min(0.25, remaining))
            finally:
                if waiter in self._waiters:
                    self._waiters.remove(waiter)
                    self._condition.notify_all()

    def release(self, job_id: str) -> bool:
        with self._condition:
            if self._active_job_id != str(job_id):
                return False
            self._active_job_id = ""
            self._active_caller_kind = ""
            self._condition.notify_all()
            return True

    def force_stop(self) -> dict[str, object]:
        with self._condition:
            active_job_id = self._active_job_id
            self._stop_epoch += 1
            self._active_job_id = ""
            self._active_caller_kind = ""
            self._waiters.clear()
            self._internal_burst = 0
            self._condition.notify_all()
            return {
                "playback_epoch": self._stop_epoch,
                "stopped_playback_job_id": active_job_id or None,
            }

    def internal_playback_active(self) -> bool:
        with self._condition:
            return self._active_caller_kind == "internal" or any(
                item.caller_kind == "internal" for item in self._waiters
            )

    def status(self) -> dict[str, object]:
        with self._condition:
            return {
                "active_job_id": self._active_job_id or None,
                "active_caller_kind": self._active_caller_kind or None,
                "waiting_internal": sum(
                    item.caller_kind == "internal" for item in self._waiters
                ),
                "waiting_external": sum(
                    item.caller_kind == "external" for item in self._waiters
                ),
                "playback_epoch": self._stop_epoch,
                "internal_burst": self._internal_burst,
                "internal_burst_limit": self.internal_burst_limit,
            }
