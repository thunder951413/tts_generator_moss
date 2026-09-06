from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SERVICE_DIR = ROOT / "qwen_tts_service"
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

import stt_runtime
from stt_runtime import STTUnavailableError, WhisperCppRuntime


class FakeProcess:
    def __init__(self, number: int, *, returncode: int | None = None) -> None:
        self.number = number
        self.pid = 10_000 + number
        self.returncode = returncode
        self.terminated = False

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = 0

    def kill(self) -> None:
        self.returncode = -9

    def wait(self, timeout: float | None = None) -> int | None:
        return self.returncode


class BlockingTerminationProcess(FakeProcess):
    def __init__(
        self,
        number: int,
        termination_started: threading.Event,
        allow_termination: threading.Event,
    ) -> None:
        super().__init__(number)
        self.termination_started = termination_started
        self.allow_termination = allow_termination

    def terminate(self) -> None:
        self.terminated = True
        self.termination_started.set()

    def wait(self, timeout: float | None = None) -> int | None:
        assert self.allow_termination.wait(2)
        self.returncode = 0
        return self.returncode


class FakeRuntime(WhisperCppRuntime):
    def __init__(self) -> None:
        super().__init__(binary="fake-whisper", model="fake-model")
        self.ready_processes: set[int] = set()

    def available(self) -> tuple[bool, str | None]:
        return True, None

    def _resolve_binary(self) -> str:
        return "fake-whisper-server"

    def _subprocess_environment(self) -> dict[str, str]:
        return {}

    def _is_ready(self, *, timeout: float = 0.5) -> bool:
        process = self._process
        return bool(process is not None and process.number in self.ready_processes)


def _install_fake_popen(monkeypatch, factory):
    spawned: list[FakeProcess] = []

    def fake_popen(*args, **kwargs):
        process = factory(len(spawned) + 1)
        spawned.append(process)
        return process

    monkeypatch.setattr(stt_runtime.subprocess, "Popen", fake_popen)
    return spawned


def _wait_until(predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("condition was not reached before timeout")
        time.sleep(0.005)


def test_concurrent_start_is_single_flight(monkeypatch) -> None:
    runtime = FakeRuntime()
    first_spawned = threading.Event()
    allow_ready = threading.Event()

    def factory(number: int) -> FakeProcess:
        if number == 1:
            first_spawned.set()
        return FakeProcess(number)

    spawned = _install_fake_popen(monkeypatch, factory)

    def mark_ready() -> None:
        assert first_spawned.wait(1)
        time.sleep(0.05)
        runtime.ready_processes.add(1)
        allow_ready.set()

    ready_thread = threading.Thread(target=mark_ready)
    ready_thread.start()
    starts = [threading.Thread(target=lambda: runtime.start(timeout=1)) for _ in range(2)]
    for thread in starts:
        thread.start()
    for thread in starts:
        thread.join(2)
        assert not thread.is_alive()
    ready_thread.join(1)

    assert allow_ready.is_set()
    assert len(spawned) == 1
    assert runtime._process is spawned[0]


def test_close_during_start_invalidates_attempt_and_allows_later_retry(monkeypatch) -> None:
    runtime = FakeRuntime()
    first_spawned = threading.Event()
    spawned = _install_fake_popen(
        monkeypatch,
        lambda number: (first_spawned.set() or FakeProcess(number)) if number == 1 else FakeProcess(number),
    )
    outcome: list[BaseException] = []

    def start() -> None:
        try:
            runtime.start(timeout=1)
        except BaseException as exc:  # assertion keeps the thread error visible
            outcome.append(exc)

    thread = threading.Thread(target=start)
    thread.start()
    assert first_spawned.wait(1)
    runtime.close()
    thread.join(2)

    assert not thread.is_alive()
    assert len(outcome) == 1
    assert isinstance(outcome[0], STTUnavailableError)
    assert spawned[0].terminated is True
    assert runtime._process is None

    runtime.ready_processes.add(2)
    runtime.start(timeout=1)
    assert len(spawned) == 2
    assert runtime._process is spawned[1]


def test_failed_start_is_cleared_for_a_deterministic_retry(monkeypatch) -> None:
    runtime = FakeRuntime()

    def factory(number: int) -> FakeProcess:
        if number == 1:
            return FakeProcess(number, returncode=23)
        runtime.ready_processes.add(number)
        return FakeProcess(number)

    spawned = _install_fake_popen(monkeypatch, factory)

    with pytest.raises(STTUnavailableError, match="exited during startup with code 23"):
        runtime.start(timeout=1)

    assert runtime._process is None
    runtime.start(timeout=1)
    assert len(spawned) == 2
    assert runtime._process is spawned[1]
    assert runtime.status()["ready"] is True


def test_close_keeps_port_handoff_closed_until_old_process_exits(monkeypatch) -> None:
    runtime = FakeRuntime()
    termination_started = threading.Event()
    allow_termination = threading.Event()
    old_process = BlockingTerminationProcess(
        0, termination_started, allow_termination
    )
    runtime._process = old_process

    def factory(number: int) -> FakeProcess:
        runtime.ready_processes.add(number)
        return FakeProcess(number)

    spawned = _install_fake_popen(monkeypatch, factory)
    closer = threading.Thread(target=runtime.close)
    closer.start()
    assert termination_started.wait(1)
    starter = threading.Thread(target=lambda: runtime.start(timeout=1))
    starter.start()

    time.sleep(0.05)
    assert spawned == []
    allow_termination.set()
    closer.join(2)
    starter.join(2)

    assert not closer.is_alive()
    assert not starter.is_alive()
    assert len(spawned) == 1
    assert runtime._process is spawned[0]


def test_unhealthy_retirement_is_single_flight_before_replacement(monkeypatch) -> None:
    runtime = FakeRuntime()
    termination_started = threading.Event()
    allow_termination = threading.Event()
    runtime._process = BlockingTerminationProcess(
        0, termination_started, allow_termination
    )

    def factory(number: int) -> FakeProcess:
        runtime.ready_processes.add(number)
        return FakeProcess(number)

    spawned = _install_fake_popen(monkeypatch, factory)
    starts = [threading.Thread(target=lambda: runtime.start(timeout=1)) for _ in range(2)]
    starts[0].start()
    assert termination_started.wait(1)
    starts[1].start()

    time.sleep(0.05)
    assert spawned == []
    allow_termination.set()
    for thread in starts:
        thread.join(2)
        assert not thread.is_alive()

    assert len(spawned) == 1
    assert runtime._process is spawned[0]


def test_close_during_unhealthy_retirement_prevents_stale_restart(monkeypatch) -> None:
    runtime = FakeRuntime()
    termination_started = threading.Event()
    allow_termination = threading.Event()
    runtime._process = BlockingTerminationProcess(
        0, termination_started, allow_termination
    )
    spawned = _install_fake_popen(monkeypatch, lambda number: FakeProcess(number))
    outcomes: list[BaseException] = []

    def start() -> None:
        try:
            runtime.start(timeout=1)
        except BaseException as exc:
            outcomes.append(exc)

    starter = threading.Thread(target=start)
    starter.start()
    assert termination_started.wait(1)
    closer = threading.Thread(target=runtime.close)
    closer.start()
    _wait_until(lambda: runtime._closing)
    allow_termination.set()
    starter.join(2)
    closer.join(2)

    assert not starter.is_alive()
    assert not closer.is_alive()
    assert len(outcomes) == 1
    assert isinstance(outcomes[0], STTUnavailableError)
    assert spawned == []
    assert runtime._process is None


def test_timeout_cleanup_finishes_before_retry_can_spawn(monkeypatch) -> None:
    runtime = FakeRuntime()
    termination_started = threading.Event()
    allow_termination = threading.Event()

    def factory(number: int) -> FakeProcess:
        if number == 1:
            return BlockingTerminationProcess(
                number, termination_started, allow_termination
            )
        runtime.ready_processes.add(number)
        return FakeProcess(number)

    spawned = _install_fake_popen(monkeypatch, factory)
    outcomes: list[BaseException] = []

    def start_with_outcome() -> None:
        try:
            runtime.start(timeout=0.01)
        except BaseException as exc:
            outcomes.append(exc)

    failed_start = threading.Thread(target=start_with_outcome)
    failed_start.start()
    assert termination_started.wait(2)
    retry = threading.Thread(target=lambda: runtime.start(timeout=1))
    retry.start()

    time.sleep(0.05)
    assert len(spawned) == 1
    allow_termination.set()
    failed_start.join(2)
    retry.join(2)

    assert not failed_start.is_alive()
    assert not retry.is_alive()
    assert len(outcomes) == 1
    assert isinstance(outcomes[0], STTUnavailableError)
    assert len(spawned) == 2
    assert runtime._process is spawned[1]


def test_transcriptions_are_serialized_and_keep_responses_paired(monkeypatch) -> None:
    runtime = FakeRuntime()
    process = FakeProcess(1)
    runtime._process = process
    runtime.ready_processes.add(1)
    active = 0
    peak_active = 0
    lock = threading.Lock()

    class Response:
        def __init__(self, marker: str) -> None:
            self.marker = marker
            self.headers = self

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps({"text": self.marker}).encode()

        def get_content_type(self) -> str:
            return "application/json"

    def fake_urlopen(request, timeout):
        nonlocal active, peak_active
        marker = "one" if b"one" in request.data else "two"
        with lock:
            active += 1
            peak_active = max(peak_active, active)
        time.sleep(0.04)
        with lock:
            active -= 1
        return Response(marker)

    monkeypatch.setattr(stt_runtime.urllib.request, "urlopen", fake_urlopen)
    results: dict[str, str] = {}

    def transcribe(marker: str) -> None:
        payload, _ = runtime.transcribe(audio=marker.encode(), filename=f"{marker}.wav")
        results[marker] = json.loads(payload)["text"]

    threads = [threading.Thread(target=transcribe, args=(marker,)) for marker in ("one", "two")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(2)
        assert not thread.is_alive()

    assert peak_active == 1
    assert results == {"one": "one", "two": "two"}


def test_queued_transcription_honors_cancellation_before_it_starts() -> None:
    runtime = FakeRuntime()
    cancelled = threading.Event()
    outcome: list[BaseException] = []
    assert runtime._transcribe_lock.acquire()

    def transcribe() -> None:
        try:
            runtime.transcribe(
                audio=b"queued", filename="queued.wav", cancelled=cancelled.is_set
            )
        except BaseException as exc:
            outcome.append(exc)

    thread = threading.Thread(target=transcribe)
    thread.start()
    time.sleep(0.05)
    cancelled.set()
    thread.join(1)
    runtime._transcribe_lock.release()

    assert not thread.is_alive()
    assert len(outcome) == 1
    assert isinstance(outcome[0], STTUnavailableError)
    assert "cancelled" in str(outcome[0]).lower()
