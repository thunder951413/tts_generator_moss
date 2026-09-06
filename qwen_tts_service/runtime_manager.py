# coding=utf-8
"""Model profile controller owning isolated Qwen3-TTS Metal worker runtimes."""

from __future__ import annotations

import gc
import logging
import os
import threading
import time
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

import torch

from qwen_runtime import QwenWorkerRuntime

REPO_ROOT = Path(__file__).resolve().parent.parent

MODEL_PROFILE_LABELS = {
    "qwen_0_6b": "Qwen3-TTS 0.6B（Metal 极速克隆）",
    "qwen_1_7b": "Qwen3-TTS 1.7B（Metal 高质量克隆）",
}
DEFAULT_MODEL_PROFILE = "qwen_0_6b"


class RuntimeSessionCancelled(RuntimeError):
    """Raised when a runtime session is invalidated before it is admitted."""


class RuntimeManager:
    def __init__(
        self,
        *,
        qwen_python: str,
        qwen_worker_script: str,
        qwen_0_6b_model_dir: str,
        qwen_1_7b_model_dir: str,
        qwen_0_6b_lanes: int,
        qwen_1_7b_lanes: int,
        qwen_backend: str,
        qwen_quant: str,
        qwentts_library: str,
    ) -> None:
        self.profiles = {
            "qwen_0_6b": {
                "label": MODEL_PROFILE_LABELS["qwen_0_6b"],
                "model_dir": str(qwen_0_6b_model_dir),
                "codec_dir": str(qwen_0_6b_model_dir),
                "sample_rate": 24000,
                "channels": 1,
                "streaming": True,
                "backend": "qwen",
                "family": "qwen",
                "lanes": max(1, int(qwen_0_6b_lanes)),
                "base_port": 7870,
                "runtime_backend": qwen_backend,
                "quant": qwen_quant,
            },
            "qwen_1_7b": {
                "label": MODEL_PROFILE_LABELS["qwen_1_7b"],
                "model_dir": str(qwen_1_7b_model_dir),
                "codec_dir": str(qwen_1_7b_model_dir),
                "sample_rate": 24000,
                "channels": 1,
                "streaming": True,
                "backend": "qwen",
                "family": "qwen",
                "lanes": max(1, int(qwen_1_7b_lanes)),
                "base_port": 7880,
                "runtime_backend": qwen_backend,
                "quant": qwen_quant,
            },
        }
        self.qwen_python = str(qwen_python)
        self.qwen_worker_script = str(qwen_worker_script)
        self.qwen_backend = str(qwen_backend)
        self.qwen_quant = str(qwen_quant)
        self.qwentts_library = str(qwentts_library)
        self.device = "metal"
        self.tts_device = "metal"
        self.codec_device = "metal"
        self.dtype = "gguf"
        self.attn_implementation = "ggml_metal"
        self.codec_weight_dtype = "gguf"
        self.codec_compute_dtype = "ggml"
        self._lock = threading.Lock()
        self._session_condition = threading.Condition()
        self._session_count = 0
        self._switching = False
        self._closing = False
        self._close_count = 0
        self._generation = 0
        self._session_waiters: deque[object] = deque()
        self._active_profile: str | None = None
        self._status_lock = threading.Lock()
        self._runtime: QwenWorkerRuntime | None = None
        self._loader_thread: threading.Thread | None = None
        self._state = "not_loaded"
        self._error: str | None = None
        self._load_started_at: float | None = None
        self._ready_at: float | None = None
        self._session_wait_timeout = max(
            30.0,
            float(os.environ.get("QWEN_TTS_SESSION_WAIT_TIMEOUT", "600")),
        )

    def _set_status(self, *, state: str, error: str | None = None) -> None:
        with self._status_lock:
            self._state = state
            self._error = error
            if state == "loading":
                self._load_started_at = time.time()
                self._ready_at = None
            elif state == "ready":
                self._ready_at = time.time()

    def status(self) -> dict[str, Any]:
        with self._status_lock:
            state = self._state
            error = self._error
            load_started_at = self._load_started_at
            ready_at = self._ready_at
        # Model construction and worker shutdown hold ``_lock`` for a long
        # time, so health checks snapshot the published runtime under the
        # short-lived session condition instead.  Detach/commit use the same
        # condition, preventing a close from turning a checked runtime into
        # ``None`` halfway through this snapshot.
        with self._session_condition:
            active_profile = self._active_profile
            runtime = self._runtime
            if runtime is None:
                runtime_status = {
                    "attn_implementation": self.attn_implementation,
                    "codec_weight_dtype": self.codec_weight_dtype,
                    "n_vq": None,
                    "sample_rate": None,
                    "reference_cache_entries": 0,
                    "reference_cache_hits": 0,
                    "reference_cache_misses": 0,
                }
            else:
                runtime_status = {
                    "attn_implementation": runtime.attn_implementation,
                    "codec_weight_dtype": runtime.codec_weight_dtype,
                    "n_vq": int(runtime.n_vq),
                    "sample_rate": int(runtime.sample_rate),
                    "reference_cache_entries": len(runtime.reference_audio_cache),
                    "reference_cache_hits": int(runtime.reference_audio_cache_hits),
                    "reference_cache_misses": int(runtime.reference_audio_cache_misses),
                }
        elapsed = None
        if load_started_at is not None:
            elapsed = max(0.0, (ready_at or time.time()) - load_started_at)
        return {
            "state": state,
            "error": error,
            "load_started_at": load_started_at,
            "ready_at": ready_at,
            "load_elapsed_seconds": elapsed,
            "active_profile": active_profile,
            "active_profile_label": MODEL_PROFILE_LABELS.get(active_profile or "", ""),
            "model_dir": None if active_profile is None else self.profiles[active_profile]["model_dir"],
            "codec_dir": None if active_profile is None else self.profiles[active_profile]["codec_dir"],
            "device": self.device,
            "tts_device": self.tts_device,
            "codec_device": self.codec_device,
            "dtype": self.dtype,
            "requested_attn_implementation": self.attn_implementation,
            "attn_implementation": runtime_status["attn_implementation"],
            "codec_weight_dtype": runtime_status["codec_weight_dtype"],
            "codec_compute_dtype": self.codec_compute_dtype,
            "n_vq": runtime_status["n_vq"],
            "sample_rate": runtime_status["sample_rate"],
            "reference_cache_entries": runtime_status["reference_cache_entries"],
            "reference_cache_hits": runtime_status["reference_cache_hits"],
            "reference_cache_misses": runtime_status["reference_cache_misses"],
            "profiles": [
                {
                    "id": profile_id,
                    **profile,
                    "available": (
                        Path(self.qwen_python).is_file()
                        and Path(self.qwen_worker_script).is_file()
                        and (
                            not self.qwentts_library
                            or Path(self.qwentts_library).expanduser().is_file()
                        )
                    ),
                    "loaded": profile_id == active_profile and runtime is not None,
                }
                for profile_id, profile in self.profiles.items()
            ],
        }

    def preload_async(self) -> None:
        with self._status_lock:
            if self._runtime is not None or self._state == "loading":
                return
            if self._loader_thread is not None and self._loader_thread.is_alive():
                return

        def _load() -> None:
            try:
                with self.session(DEFAULT_MODEL_PROFILE):
                    pass
            except Exception:
                logging.exception("failed to preload Qwen3-TTS Metal runtime")

        self._loader_thread = threading.Thread(target=_load, name="qwen3-tts-metal-runtime-loader", daemon=True)
        self._loader_thread.start()

    def _release_runtime(self) -> None:
        with self._session_condition:
            runtime = self._runtime
            self._runtime = None
            self._active_profile = None
        self._close_runtime(runtime)

    @staticmethod
    def _close_runtime(runtime: Any | None) -> None:
        if runtime is not None and hasattr(runtime, "close"):
            try:
                runtime.close()
            except Exception:
                logging.exception("failed to close runtime")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def close(self) -> None:
        # Invalidate current sessions and pending loads before waiting for the
        # loader lock.  A loader that finishes meanwhile will observe the new
        # generation and discard its worker instead of publishing it.
        with self._session_condition:
            self._generation += 1
            self._session_count = 0
            self._switching = False
            self._session_waiters.clear()
            self._close_count += 1
            self._closing = True
            self._session_condition.notify_all()
        try:
            with self._lock:
                self._release_runtime()
                self._set_status(state="not_loaded")
        finally:
            with self._session_condition:
                self._close_count = max(0, self._close_count - 1)
                self._closing = self._close_count > 0
                self._session_condition.notify_all()

    def interrupt_active(self) -> int:
        """Hard-stop active Metal worker processes without unloading the model controller."""
        with self._lock:
            runtime = self._runtime
            if runtime is None or not hasattr(runtime, "interrupt_all"):
                return 0
            return int(runtime.interrupt_all())

    def _load(self, profile_id: str, generation: int | None = None) -> QwenWorkerRuntime:
        if profile_id not in self.profiles:
            raise ValueError(f"unknown model profile: {profile_id}")
        profile = self.profiles[profile_id]
        if generation is None:
            with self._session_condition:
                generation = self._generation
        with self._lock:
            with self._session_condition:
                if generation != self._generation or self._closing:
                    raise RuntimeSessionCancelled("runtime session was invalidated")
            if self._runtime is None or self._active_profile != profile_id:
                self._set_status(state="loading")
                candidate: QwenWorkerRuntime | None = None
                try:
                    self._release_runtime()
                    if not Path(self.qwen_python).is_file():
                        raise RuntimeError(f"Qwen Python环境不存在：{self.qwen_python}")
                    if self.qwentts_library and not Path(self.qwentts_library).expanduser().is_file():
                        raise RuntimeError(f"qwentts.cpp Metal动态库不存在：{self.qwentts_library}")
                    candidate = QwenWorkerRuntime(
                        profile_id=profile_id,
                        model_dir=profile["model_dir"],
                        backend=str(profile["runtime_backend"]),
                        quant=str(profile["quant"]),
                        library_path=self.qwentts_library or None,
                        python_executable=self.qwen_python,
                        worker_script=self.qwen_worker_script,
                        lanes=int(profile.get("lanes") or 1),
                        base_port=int(profile["base_port"]),
                        log_dir=REPO_ROOT / "logs" / "qwen-workers",
                    )
                    with self._session_condition:
                        stale = generation != self._generation or self._closing
                        if not stale:
                            self._runtime = candidate
                            self._active_profile = profile_id
                except Exception as exc:
                    with self._session_condition:
                        stale = generation != self._generation or self._closing
                    if not stale:
                        self._set_status(state="error", error=str(exc))
                    raise
                if stale:
                    self._close_runtime(candidate)
                    raise RuntimeSessionCancelled("runtime session was invalidated while loading")
                self._set_status(state="ready")
            return self._runtime

    @contextmanager
    def session(
        self,
        profile_id: str,
        cancelled: Callable[[], bool] | None = None,
    ):
        """Allow same-model concurrency and fairly serialize model switches."""
        profile_id = profile_id if profile_id in self.profiles else DEFAULT_MODEL_PROFILE
        waiter = object()
        needs_load = False
        runtime: QwenWorkerRuntime | None = None
        with self._session_condition:
            generation = self._generation
            self._session_waiters.append(waiter)
            wait_deadline = time.monotonic() + self._session_wait_timeout
            while True:
                if generation != self._generation:
                    if waiter in self._session_waiters:
                        self._session_waiters.remove(waiter)
                    self._session_condition.notify_all()
                    raise RuntimeSessionCancelled("runtime session was invalidated")
                try:
                    cancel_requested = cancelled is not None and cancelled()
                except Exception:
                    if waiter in self._session_waiters:
                        self._session_waiters.remove(waiter)
                    self._session_condition.notify_all()
                    raise
                if cancel_requested:
                    if waiter in self._session_waiters:
                        self._session_waiters.remove(waiter)
                    self._session_condition.notify_all()
                    raise RuntimeSessionCancelled("runtime session was cancelled")

                is_first = bool(self._session_waiters) and self._session_waiters[0] is waiter
                same_runtime = self._active_profile == profile_id and self._runtime is not None
                can_enter = (
                    is_first
                    and not self._closing
                    and not self._switching
                    and (self._session_count == 0 or same_runtime)
                )
                if can_enter:
                    self._session_waiters.popleft()
                    if same_runtime:
                        self._session_count += 1
                        runtime = self._runtime
                    else:
                        self._switching = True
                        needs_load = True
                    self._session_condition.notify_all()
                    break

                remaining = wait_deadline - time.monotonic()
                if remaining <= 0:
                    if waiter in self._session_waiters:
                        self._session_waiters.remove(waiter)
                    self._session_condition.notify_all()
                    raise TimeoutError(
                        f"等待切换到 {profile_id} 超过 {self._session_wait_timeout:.0f} 秒"
                    )
                poll_interval = 0.1 if cancelled is not None else 1.0
                self._session_condition.wait(timeout=min(poll_interval, remaining))
        if needs_load:
            try:
                runtime = self._load(profile_id, generation)
            except Exception:
                with self._session_condition:
                    if generation == self._generation:
                        self._switching = False
                    self._session_condition.notify_all()
                raise
            with self._session_condition:
                if generation != self._generation or self._closing:
                    self._session_condition.notify_all()
                    raise RuntimeSessionCancelled("runtime session was invalidated while loading")
                self._switching = False
                if cancelled is not None and cancelled():
                    self._session_condition.notify_all()
                    raise RuntimeSessionCancelled("runtime session was cancelled")
                self._session_count += 1
                self._session_condition.notify_all()
        try:
            yield runtime
        finally:
            with self._session_condition:
                if generation == self._generation:
                    self._session_count = max(0, self._session_count - 1)
                self._session_condition.notify_all()
