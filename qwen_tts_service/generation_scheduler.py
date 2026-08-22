# coding=utf-8
"""GPU inference concurrency scheduling with interactive priority for local Qwen3-TTS."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Any, Callable


class GpuGenerationScheduler:
    """Bound GPU inference concurrency with interactive priority and fairness."""

    def __init__(
        self,
        max_parallel: int = 1,
        interactive_burst_limit: int = 4,
        external_queue_limit: int = 64,
        external_blocked: Callable[[], bool] | None = None,
    ) -> None:
        self.max_parallel = max(1, int(max_parallel))
        self.interactive_burst_limit = max(1, int(interactive_burst_limit))
        self._condition = threading.Condition()
        self._active = 0
        self._waiting_interactive = 0
        self._waiting_document = 0
        self._interactive_burst = 0
        self._exclusive = False
        self._waiting_exclusive = 0
        self._waiting_external = 0
        self._active_external = 0
        self._internal_api_burst = 0
        self._external_queue_limit = max(1, int(external_queue_limit))
        self._external_blocked = external_blocked
        self._external_sequence = 0
        self._external_tickets: list[int] = []

    def _acquire(
        self,
        priority: str,
        cancelled: Callable[[], bool] | None = None,
    ) -> None:
        interactive = priority == "interactive"
        with self._condition:
            if interactive:
                self._waiting_interactive += 1
            else:
                self._waiting_document += 1
            try:
                while (
                    self._exclusive
                    or self._waiting_exclusive > 0
                    or self._active_external > 0
                    or self._active >= self.max_parallel
                    or (
                        self._waiting_external > 0
                        and self._internal_api_burst >= self.interactive_burst_limit
                    )
                    or (
                        interactive
                        and self._waiting_document > 0
                        and self._interactive_burst >= self.interactive_burst_limit
                    )
                    or (
                        not interactive
                        and self._waiting_interactive > 0
                        and self._interactive_burst < self.interactive_burst_limit
                    )
                ):
                    if cancelled and cancelled():
                        raise RuntimeError("generation cancelled while queued")
                    self._condition.wait(timeout=0.25)
                if cancelled and cancelled():
                    raise RuntimeError("generation cancelled while queued")
                self._active += 1
                if self._waiting_external > 0:
                    self._internal_api_burst += 1
                else:
                    self._internal_api_burst = 0
                if interactive and self._waiting_document > 0:
                    self._interactive_burst += 1
                elif not interactive:
                    self._interactive_burst = 0
            finally:
                if interactive:
                    self._waiting_interactive -= 1
                else:
                    self._waiting_document -= 1

    def _acquire_external(self, cancelled: Callable[[], bool] | None = None) -> None:
        with self._condition:
            if self._waiting_external >= self._external_queue_limit:
                raise RuntimeError("external generation queue is full")
            self._external_sequence += 1
            ticket = self._external_sequence
            self._external_tickets.append(ticket)
            self._waiting_external += 1
            try:
                while (
                    self._exclusive
                    or self._waiting_exclusive > 0
                    or self._active > 0
                    or (
                        (self._waiting_interactive > 0 or self._waiting_document > 0)
                        and self._internal_api_burst < self.interactive_burst_limit
                    )
                    or self._external_tickets[0] != ticket
                    or bool(self._external_blocked and self._external_blocked())
                ):
                    if cancelled and cancelled():
                        raise RuntimeError("generation cancelled while queued")
                    self._condition.wait(timeout=0.25)
                if cancelled and cancelled():
                    raise RuntimeError("generation cancelled while queued")
                self._active += 1
                self._active_external += 1
                self._internal_api_burst = 0
            finally:
                self._waiting_external -= 1
                if ticket in self._external_tickets:
                    self._external_tickets.remove(ticket)

    def _release(self, priority: str = "document") -> None:
        with self._condition:
            self._active = max(0, self._active - 1)
            if priority == "external":
                self._active_external = max(0, self._active_external - 1)
            self._condition.notify_all()

    def __enter__(self) -> GpuGenerationScheduler:
        self._acquire("document")
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self._release()

    @contextmanager
    def interactive_slot(self):
        self._acquire("interactive")
        try:
            yield
        finally:
            self._release()

    @contextmanager
    def api_slot(
        self,
        caller_kind: str,
        *,
        cancelled: Callable[[], bool] | None = None,
    ):
        priority = "external" if str(caller_kind).lower() == "external" else "interactive"
        if priority == "external":
            self._acquire_external(cancelled)
        else:
            self._acquire("interactive", cancelled)
        try:
            yield
        finally:
            self._release(priority)

    @contextmanager
    def exclusive_slot(self):
        with self._condition:
            self._waiting_exclusive += 1
            try:
                while self._active > 0 or self._exclusive:
                    self._condition.wait()
                self._exclusive = True
            finally:
                self._waiting_exclusive -= 1
        try:
            yield
        finally:
            with self._condition:
                self._exclusive = False
                self._condition.notify_all()

    def configure_max_parallel(self, value: int) -> int:
        with self._condition:
            self.max_parallel = max(1, min(2, int(value)))
            self._condition.notify_all()
            return self.max_parallel

    def status(self) -> dict[str, int]:
        with self._condition:
            return {
                "max_parallel": self.max_parallel,
                "active": self._active,
                "waiting_interactive": self._waiting_interactive,
                "waiting_document": self._waiting_document,
                "waiting_external": self._waiting_external,
                "active_external": self._active_external,
                "internal_api_burst": self._internal_api_burst,
                "interactive_burst": self._interactive_burst,
                "interactive_burst_limit": self.interactive_burst_limit,
                "performance_test_active": self._exclusive,
            }
