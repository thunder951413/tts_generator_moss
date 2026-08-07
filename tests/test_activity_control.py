from __future__ import annotations

import sys
import threading
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SERVICE_DIR = ROOT / "qwen_tts_service"
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

from activity_control import PlaybackCoordinator


def test_playback_coordinator_uses_internal_priority_fifo_with_fairness() -> None:
    coordinator = PlaybackCoordinator(internal_burst_limit=2)
    assert coordinator.acquire("active", "external", timeout=1)
    order: list[str] = []

    def wait(job_id: str, caller_kind: str) -> None:
        assert coordinator.acquire(job_id, caller_kind, timeout=3)
        order.append(job_id)
        coordinator.release(job_id)

    threads = [
        threading.Thread(target=wait, args=("external-1", "external")),
        threading.Thread(target=wait, args=("internal-1", "internal")),
        threading.Thread(target=wait, args=("internal-2", "internal")),
        threading.Thread(target=wait, args=("internal-3", "internal")),
    ]
    for thread in threads:
        thread.start()
        time.sleep(0.02)
    coordinator.release("active")
    for thread in threads:
        thread.join(timeout=4)
        assert not thread.is_alive()

    assert order == ["internal-1", "internal-2", "external-1", "internal-3"]


def test_force_stop_revokes_active_and_waiting_playback() -> None:
    coordinator = PlaybackCoordinator()
    assert coordinator.acquire("active", "internal", timeout=1)
    acquired: list[bool] = []
    waiter = threading.Thread(
        target=lambda: acquired.append(coordinator.acquire("waiting", "external", timeout=3))
    )
    waiter.start()
    time.sleep(0.05)

    stopped = coordinator.force_stop()
    waiter.join(timeout=2)

    assert acquired == [False]
    assert stopped["stopped_playback_job_id"] == "active"
    assert stopped["playback_epoch"] == 1
    assert coordinator.status()["active_job_id"] is None


def test_duplicate_job_needs_explicit_reentrant_lease() -> None:
    coordinator = PlaybackCoordinator()
    assert coordinator.acquire("same-job", "external", timeout=1)
    assert not coordinator.acquire("same-job", "external", timeout=1)
    assert coordinator.acquire(
        "same-job", "external", timeout=1, allow_reentrant=True
    )
    assert coordinator.release("same-job")


def test_release_cancels_a_waiting_media_lease() -> None:
    coordinator = PlaybackCoordinator()
    assert coordinator.acquire("active", "internal", timeout=1)
    acquired: list[bool] = []
    waiter = threading.Thread(
        target=lambda: acquired.append(coordinator.acquire("media:session", "external", timeout=3))
    )
    waiter.start()
    time.sleep(0.05)

    assert coordinator.release("media:session")
    waiter.join(timeout=1)

    assert acquired == [False]
    assert not waiter.is_alive()
    assert coordinator.release("active")
