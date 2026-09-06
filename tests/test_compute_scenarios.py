from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "qwen_tts_service"))
from generation_scheduler import GpuGenerationScheduler


def until(predicate):
    deadline = time.monotonic() + 2
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.005)
    assert predicate()


def test_playback_blocked_external_does_not_deadlock_reader_fairness():
    listening = threading.Event()
    listening.set()
    scheduler = GpuGenerationScheduler(external_blocked=listening.is_set, interactive_burst_limit=1)
    external_done = threading.Event()

    def external():
        with scheduler.api_slot("external"):
            external_done.set()

    worker = threading.Thread(target=external, daemon=True)
    worker.start()
    until(lambda: scheduler.status()["waiting_external"] == 1)
    # Cross the fairness quota while the external request cannot run because
    # it is waiting for this reader to finish.
    for _ in range(3):
        with scheduler.api_slot("internal"):
            assert not external_done.is_set()
    listening.clear()
    worker.join(2)
    assert external_done.is_set()


def test_stt_waits_for_listener_without_blocking_next_tts_block():
    listening = threading.Event()
    listening.set()
    scheduler = GpuGenerationScheduler(external_blocked=listening.is_set)
    stt_started = threading.Event()

    def transcribe():
        with scheduler.exclusive_slot(kind="stt", wait_for_playback=True):
            stt_started.set()
            assert scheduler.status()["active"] == 0

    worker = threading.Thread(target=transcribe, daemon=True)
    worker.start()
    until(lambda: scheduler.status()["waiting_stt"] == 1)
    with scheduler.api_slot("internal"):
        assert not stt_started.is_set()
    listening.clear()
    worker.join(2)
    assert stt_started.is_set()
    assert not scheduler.status()["stt_active"]


def test_stt_exclusive_at_document_boundary_then_tts_resumes():
    scheduler = GpuGenerationScheduler(max_parallel=2)
    started = threading.Event()
    release = threading.Event()
    tts_started = threading.Event()

    def transcribe():
        with scheduler.exclusive_slot(kind="stt"):
            assert scheduler.status()["active"] == 0
            started.set()
            release.wait(2)

    def tts():
        with scheduler.api_slot("internal"):
            tts_started.set()

    with scheduler:
        worker = threading.Thread(target=transcribe, daemon=True)
        worker.start()
        until(lambda: scheduler.status()["waiting_stt"] == 1)
        assert not started.is_set()
    assert started.wait(2)
    another = threading.Thread(target=tts, daemon=True)
    another.start()
    until(lambda: scheduler.status()["waiting_interactive"] == 1)
    assert not tts_started.is_set()
    release.set()
    worker.join(2)
    another.join(2)
    assert tts_started.is_set()


def test_global_stop_invalidates_queued_stt_and_allows_new_requests():
    listening = threading.Event()
    listening.set()
    scheduler = GpuGenerationScheduler(external_blocked=listening.is_set)
    errors = []

    def transcribe():
        try:
            with scheduler.exclusive_slot(kind="stt", wait_for_playback=True):
                raise AssertionError("stopped work must never run")
        except RuntimeError as error:
            errors.append(str(error))

    worker = threading.Thread(target=transcribe, daemon=True)
    worker.start()
    until(lambda: scheduler.status()["waiting_stt"] == 1)
    scheduler.cancel_pending()
    worker.join(2)
    assert errors and "cancelled" in errors[0]
    assert scheduler.status()["waiting_stt"] == 0
    listening.clear()
    with scheduler.exclusive_slot(kind="stt"):
        assert scheduler.status()["stt_active"]


def test_stt_disconnect_cancels_wait_without_running():
    scheduler = GpuGenerationScheduler()
    cancelled = threading.Event()
    errors = []

    def request():
        try:
            with scheduler.exclusive_slot(kind="stt", cancelled=cancelled.is_set):
                raise AssertionError("disconnected request must not run")
        except RuntimeError as error:
            errors.append(str(error))

    with scheduler:
        worker = threading.Thread(target=request, daemon=True)
        worker.start()
        until(lambda: scheduler.status()["waiting_stt"] == 1)
        cancelled.set()
        worker.join(2)
        assert errors
    assert scheduler.status()["active"] == 0
