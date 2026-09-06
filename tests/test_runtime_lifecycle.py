from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
SERVICE_DIR = ROOT / "qwen_tts_service"
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

import runtime_manager as runtime_manager_module
from runtime_manager import RuntimeManager, RuntimeSessionCancelled


class FakeRuntime:
    instances: list["FakeRuntime"] = []

    def __init__(self, **kwargs: Any) -> None:
        self.profile_id = str(kwargs["profile_id"])
        self.closed = False
        self.interrupted = 0
        self.attn_implementation = "fake"
        self.codec_weight_dtype = "fake"
        self.n_vq = 16
        self.sample_rate = 24000
        self.reference_audio_cache: dict[str, object] = {}
        self.reference_audio_cache_hits = 0
        self.reference_audio_cache_misses = 0
        self.instances.append(self)

    def close(self) -> None:
        self.closed = True

    def interrupt_all(self) -> int:
        self.interrupted += 1
        return 1


def _manager(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> RuntimeManager:
    FakeRuntime.instances = []
    monkeypatch.setattr(runtime_manager_module, "QwenWorkerRuntime", FakeRuntime)
    monkeypatch.setattr(runtime_manager_module.gc, "collect", lambda: 0)
    return RuntimeManager(
        qwen_python=sys.executable,
        qwen_worker_script=__file__,
        qwen_0_6b_model_dir=str(tmp_path / "small"),
        qwen_1_7b_model_dir=str(tmp_path / "large"),
        qwen_0_6b_lanes=2,
        qwen_1_7b_lanes=2,
        qwen_backend="fake",
        qwen_quant="fake",
        qwentts_library="",
    )


def _wait_until(predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("condition was not reached before timeout")
        time.sleep(0.005)


def test_same_profile_sessions_share_runtime_concurrently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = _manager(tmp_path, monkeypatch)
    entered = threading.Event()
    release = threading.Event()
    seen: list[object] = []

    def use_small() -> None:
        with manager.session("qwen_0_6b") as runtime:
            seen.append(runtime)
            entered.set()
            assert release.wait(2)

    with manager.session("qwen_0_6b") as runtime:
        peer = threading.Thread(target=use_small)
        peer.start()
        assert entered.wait(2)
        assert manager._session_count == 2
        assert seen == [runtime]
        release.set()
        peer.join(timeout=2)

    assert not peer.is_alive()
    assert manager._session_count == 0
    assert len(FakeRuntime.instances) == 1


def test_profile_switch_is_fifo_and_late_same_profile_cannot_starve_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = _manager(tmp_path, monkeypatch)
    order: list[str] = []
    large_entered = threading.Event()
    release_large = threading.Event()
    late_small_entered = threading.Event()

    def use_large() -> None:
        with manager.session("qwen_1_7b"):
            order.append("large")
            large_entered.set()
            assert release_large.wait(2)

    def use_late_small() -> None:
        with manager.session("qwen_0_6b"):
            order.append("late-small")
            late_small_entered.set()

    with manager.session("qwen_0_6b"):
        large = threading.Thread(target=use_large)
        large.start()
        _wait_until(lambda: len(manager._session_waiters) == 1)
        late_small = threading.Thread(target=use_late_small)
        late_small.start()
        _wait_until(lambda: len(manager._session_waiters) == 2)

    assert large_entered.wait(2)
    assert not late_small_entered.wait(0.05)
    release_large.set()
    large.join(timeout=2)
    late_small.join(timeout=2)

    assert not large.is_alive()
    assert not late_small.is_alive()
    assert order == ["large", "late-small"]
    assert [runtime.profile_id for runtime in FakeRuntime.instances] == [
        "qwen_0_6b",
        "qwen_1_7b",
        "qwen_0_6b",
    ]


def test_cancelled_callback_removes_waiting_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = _manager(tmp_path, monkeypatch)
    cancel = threading.Event()
    errors: list[BaseException] = []

    def wait_for_large() -> None:
        try:
            with manager.session("qwen_1_7b", cancelled=cancel.is_set):
                raise AssertionError("cancelled waiter must not enter")
        except BaseException as exc:
            errors.append(exc)

    with manager.session("qwen_0_6b"):
        waiter = threading.Thread(target=wait_for_large)
        waiter.start()
        _wait_until(lambda: len(manager._session_waiters) == 1)
        cancel.set()
        waiter.join(timeout=2)
        assert not waiter.is_alive()
        assert manager._session_count == 1

    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeSessionCancelled)
    assert not manager._session_waiters


def test_close_invalidates_waiters_and_old_finalizer_cannot_touch_new_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = _manager(tmp_path, monkeypatch)
    old_entered = threading.Event()
    release_old = threading.Event()
    waiter_errors: list[BaseException] = []

    def old_small_session() -> None:
        with manager.session("qwen_0_6b"):
            old_entered.set()
            assert release_old.wait(2)

    def queued_large_session() -> None:
        try:
            with manager.session("qwen_1_7b"):
                raise AssertionError("pre-close waiter must not enter")
        except BaseException as exc:
            waiter_errors.append(exc)

    old = threading.Thread(target=old_small_session)
    old.start()
    assert old_entered.wait(2)
    queued = threading.Thread(target=queued_large_session)
    queued.start()
    _wait_until(lambda: len(manager._session_waiters) == 1)

    old_runtime = FakeRuntime.instances[0]
    manager.close()
    queued.join(timeout=2)
    assert old_runtime.closed
    assert len(waiter_errors) == 1
    assert isinstance(waiter_errors[0], RuntimeSessionCancelled)

    with manager.session("qwen_0_6b"):
        assert manager._session_count == 1
        release_old.set()
        old.join(timeout=2)
        assert not old.is_alive()
        assert manager._session_count == 1

    assert manager._session_count == 0


def test_close_during_load_discards_worker_and_never_yields_stale_pointer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    load_started = threading.Event()
    finish_load = threading.Event()

    class BlockingRuntime(FakeRuntime):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            load_started.set()
            assert finish_load.wait(2)

    manager = _manager(tmp_path, monkeypatch)
    monkeypatch.setattr(runtime_manager_module, "QwenWorkerRuntime", BlockingRuntime)
    yielded: list[object] = []
    errors: list[BaseException] = []

    def load_session() -> None:
        try:
            with manager.session("qwen_1_7b") as runtime:
                yielded.append(runtime)
        except BaseException as exc:
            errors.append(exc)

    loader = threading.Thread(target=load_session)
    loader.start()
    assert load_started.wait(2)
    status_results: list[dict[str, Any]] = []
    status_reader = threading.Thread(target=lambda: status_results.append(manager.status()))
    status_reader.start()
    status_reader.join(timeout=0.2)
    assert not status_reader.is_alive(), "health status must not wait for model construction"
    assert status_results[0]["state"] == "loading"
    assert status_results[0]["active_profile"] is None
    closer = threading.Thread(target=manager.close)
    closer.start()
    _wait_until(lambda: manager._closing)
    finish_load.set()
    loader.join(timeout=2)
    closer.join(timeout=2)

    assert not loader.is_alive()
    assert not closer.is_alive()
    assert yielded == []
    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeSessionCancelled)
    assert len(FakeRuntime.instances) == 1
    assert FakeRuntime.instances[0].closed
    assert manager._runtime is None
    assert manager._active_profile is None
    assert manager.status()["state"] == "not_loaded"


def test_status_snapshots_runtime_coherently_before_concurrent_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    status_reading = threading.Event()
    allow_status = threading.Event()

    class BlockingStatusRuntime(FakeRuntime):
        @property
        def attn_implementation(self) -> str:
            status_reading.set()
            assert allow_status.wait(2)
            return self._attn_implementation

        @attn_implementation.setter
        def attn_implementation(self, value: str) -> None:
            self._attn_implementation = value

    manager = _manager(tmp_path, monkeypatch)
    monkeypatch.setattr(runtime_manager_module, "QwenWorkerRuntime", BlockingStatusRuntime)
    with manager.session("qwen_0_6b") as runtime:
        pass

    snapshots: list[dict[str, Any]] = []
    reader = threading.Thread(target=lambda: snapshots.append(manager.status()))
    reader.start()
    assert status_reading.wait(1)
    closer = threading.Thread(target=manager.close)
    closer.start()
    time.sleep(0.05)
    assert runtime.closed is False

    allow_status.set()
    reader.join(2)
    closer.join(2)

    assert not reader.is_alive()
    assert not closer.is_alive()
    assert snapshots[0]["active_profile"] == "qwen_0_6b"
    loaded = {profile["id"]: profile["loaded"] for profile in snapshots[0]["profiles"]}
    assert loaded == {"qwen_0_6b": True, "qwen_1_7b": False}
    assert snapshots[0]["attn_implementation"] == "fake"
    assert runtime.closed is True
