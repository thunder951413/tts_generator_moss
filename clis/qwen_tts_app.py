# coding=utf-8
"""Persistent Qwen3-TTS web service for Apple Silicon."""

from __future__ import annotations

import argparse
import asyncio
import csv
import gc
import hashlib
import hmac
import html as html_lib
import json
import logging
import mimetypes
import os
import queue
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote, unquote

import orjson
import torch
import uvicorn
from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

REPO_ROOT = Path(__file__).resolve().parent.parent
STREAMING_MODULE_DIR = REPO_ROOT / "qwen_tts_service"
if str(STREAMING_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(STREAMING_MODULE_DIR))

from document_projects import DocumentProjectManager
from presets import VoicePresetStore
from qwen_protocol import StreamingRequest
from qwen_runtime import QwenWorkerRuntime
from stt_runtime import STTUnavailableError, WhisperCppRuntime

DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "qwen_tts_streaming"
DEFAULT_UPLOAD_DIR = REPO_ROOT / "outputs" / "qwen_tts_uploads"
DEFAULT_DOCUMENT_PROJECT_DIR = REPO_ROOT / "outputs" / "qwen_tts_document_projects"
DEFAULT_SERVICE_JOB_DIR = REPO_ROOT / "outputs" / "qwen_tts_service_jobs"
DEFAULT_PRESET_DIR = REPO_ROOT / "outputs" / "qwen_tts_presets"
NOVEL_READER_WEB_DIR = REPO_ROOT / "web" / "novel_reader"
SERVICE_AUTH_COOKIE = "qwen_tts_service_session"
DEFAULT_QWEN_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
DEFAULT_QWEN_WORKER_SCRIPT = STREAMING_MODULE_DIR / "qwen_worker.py"
DEFAULT_QWEN_0_6B_MODEL_DIR = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
DEFAULT_QWEN_1_7B_MODEL_DIR = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
DEFAULT_QWEN_BACKEND = "ggml"
DEFAULT_QWEN_QUANT = "Q4_K_M"
DEFAULT_QWENTTS_LIBRARY = REPO_ROOT / ".runtime" / "qwentts.cpp" / "build-metal" / "libqwen.dylib"
DEFAULT_WHISPER_SERVER = Path("/opt/homebrew/bin/whisper-server")
DEFAULT_WHISPER_MODEL = REPO_ROOT / ".runtime" / "whisper.cpp" / "models" / "ggml-small.bin"
DEFAULT_WHISPER_PORT = 7890
DEFAULT_MODEL_PROFILE = "qwen_0_6b"
MODEL_PROFILE_LABELS = {
    "qwen_0_6b": "Qwen3-TTS 0.6B（Metal 极速克隆）",
    "qwen_1_7b": "Qwen3-TTS 1.7B（Metal 高质量克隆）",
}
DEFAULT_MAX_NEW_TOKENS = 7500
MODE_CLONE = "Clone"
MODE_CONTINUE = "Continuation"
MODE_CONTINUE_CLONE = "Continuation + Clone"
CONTINUATION_NOTICE = (
    "Continuation mode is active. Fill Reference Audio Transcript with the transcript of the reference audio."
)
ZH_TOKENS_PER_CHAR = 3.098411951313033
EN_TOKENS_PER_CHAR = 0.8673376262755219
REFERENCE_AUDIO_DIR = REPO_ROOT / "assets" / "audio"
EXAMPLE_TEXTS_JSONL_PATH = REPO_ROOT / "assets" / "text" / "qwen_tts_example_texts.jsonl"
BAILIAN_VOICES_TSV_PATH = REFERENCE_AUDIO_DIR / "bailian" / "voices.tsv"
LANGUAGE_TAG_AUTO = "Auto (omit)"
LANGUAGE_TAG_CHOICES = [
    LANGUAGE_TAG_AUTO,
    "Chinese",
    "Cantonese",
    "English",
    "Arabic",
    "Czech",
    "Danish",
    "Dutch",
    "Finnish",
    "French",
    "German",
    "Greek",
    "Hebrew",
    "Hindi",
    "Hungarian",
    "Italian",
    "Japanese",
    "Korean",
    "Macedonian",
    "Malay",
    "Persian (Farsi)",
    "Polish",
    "Portuguese",
    "Romanian",
    "Russian",
    "Spanish",
    "Swahili",
    "Swedish",
    "Tagalog",
    "Thai",
    "Turkish",
    "Vietnamese",
]


def _parse_example_id(example_id: str) -> tuple[str, int] | None:
    matched = re.fullmatch(r"(zh|en)/(\d+)", (example_id or "").strip())
    if matched is None:
        return None
    return matched.group(1), int(matched.group(2))


def _resolve_reference_audio_path(language: str, index: int) -> Path | None:
    stem = f"reference_{language}_{index}"
    for ext in (".wav", ".mp3", ".m4a"):
        audio_path = REFERENCE_AUDIO_DIR / f"{stem}{ext}"
        if audio_path.exists():
            return audio_path
    return None


def build_example_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not EXAMPLE_TEXTS_JSONL_PATH.exists():
        return rows
    with open(EXAMPLE_TEXTS_JSONL_PATH, "rb") as f:
        for line in f:
            if not line.strip():
                continue
            sample = orjson.loads(line)
            parsed = _parse_example_id(str(sample.get("id", "")))
            if parsed is None:
                continue
            language, index = parsed
            audio_path = _resolve_reference_audio_path(language, index)
            if audio_path is None:
                continue
            rows.append(
                {
                    "role": str(sample.get("role", "")).strip(),
                    "audio_path": str(audio_path),
                    "text": str(sample.get("text", "")).strip(),
                    "language": "Chinese" if language == "zh" else "English",
                }
            )
    return rows


EXAMPLE_ROWS = build_example_rows()


def build_bailian_voice_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not BAILIAN_VOICES_TSV_PATH.exists():
        return rows
    with open(BAILIAN_VOICES_TSV_PATH, "r", encoding="utf-8-sig", newline="") as f:
        for item in csv.DictReader(f, delimiter="\t"):
            name = str(item.get("name") or "").strip()
            description = str(item.get("description") or "").strip()
            filename = str(item.get("filename") or "").strip()
            if not name or not filename:
                continue
            audio_path = BAILIAN_VOICES_TSV_PATH.parent / filename
            if audio_path.exists():
                rows.append(
                    {
                        "name": name,
                        "description": description,
                        "audio_path": str(audio_path),
                        "language": str(item.get("language") or "Chinese").strip(),
                        "transcript": str(item.get("transcript") or "").strip(),
                        "transcript_source": str(item.get("transcript_source") or "").strip(),
                    }
                )
    return rows


BAILIAN_VOICE_ROWS = build_bailian_voice_rows()
DEFAULT_CLONE_AUDIO_PATH = next(
    (row["audio_path"] for row in BAILIAN_VOICE_ROWS if row["name"] == "龙嫱"),
    BAILIAN_VOICE_ROWS[0]["audio_path"] if BAILIAN_VOICE_ROWS else "",
)


def _normalize_language(language_tag: str | None) -> str:
    value = (language_tag or "").strip()
    return "" if value == LANGUAGE_TAG_AUTO else value


def _safe_int(value: Any, *, default: int, minimum: int, maximum: int | None = None) -> int:
    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        parsed = int(default)
    parsed = max(int(minimum), parsed)
    if maximum is not None:
        parsed = min(int(maximum), parsed)
    return parsed


def _safe_aac_bitrate(value: Any, *, default: str = "80k") -> str:
    bitrate = str(value or default).strip().lower()
    return bitrate if bitrate in {"48k", "64k", "80k", "96k", "128k", "192k"} else default


def _safe_float(value: Any, *, default: float, minimum: float, maximum: float | None = None) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = float(default)
    parsed = max(float(minimum), parsed)
    if maximum is not None:
        parsed = min(float(maximum), parsed)
    return parsed


def _decode_reference_path(path: str) -> str:
    decoded = str(path or "")
    for _ in range(2):
        next_decoded = unquote(decoded)
        if next_decoded == decoded:
            break
        decoded = next_decoded
    return decoded


def _resolve_allowed_reference_audio_path(path: str, *roots: Path) -> Path:
    """Resolve a reference file without allowing an arbitrary local-file read."""
    try:
        candidate = Path(_decode_reference_path(path)).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise FileNotFoundError(path) from exc
    for root in roots:
        try:
            resolved_root = root.resolve(strict=True)
        except (OSError, RuntimeError):
            continue
        if candidate.is_file() and (candidate == resolved_root or resolved_root in candidate.parents):
            return candidate
    raise PermissionError("reference audio path is not allowed")


def _pcm16le_bytes(waveform: torch.Tensor, channels: int = 2) -> bytes:
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    target_channels = 1 if int(channels) == 1 else 2
    if target_channels == 1:
        waveform = waveform.mean(dim=0, keepdim=True) if waveform.shape[0] > 1 else waveform[:1]
    elif waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    elif waveform.shape[0] > 2:
        waveform = waveform[:2]
    pcm = waveform.detach().cpu().to(torch.float32).clamp(-1.0, 1.0)
    pcm = (pcm * 32767.0).round().to(torch.int16)
    return pcm.transpose(0, 1).contiguous().numpy().tobytes()


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
        elapsed = None
        if load_started_at is not None:
            elapsed = max(0.0, (ready_at or time.time()) - load_started_at)
        return {
            "state": state,
            "error": error,
            "load_started_at": load_started_at,
            "ready_at": ready_at,
            "load_elapsed_seconds": elapsed,
            "active_profile": self._active_profile,
            "active_profile_label": MODEL_PROFILE_LABELS.get(self._active_profile or "", ""),
            "model_dir": None if self._active_profile is None else self.profiles[self._active_profile]["model_dir"],
            "codec_dir": None if self._active_profile is None else self.profiles[self._active_profile]["codec_dir"],
            "device": self.device,
            "tts_device": self.tts_device,
            "codec_device": self.codec_device,
            "dtype": self.dtype,
            "requested_attn_implementation": self.attn_implementation,
            "attn_implementation": (
                self.attn_implementation
                if self._runtime is None
                else self._runtime.attn_implementation
            ),
            "codec_weight_dtype": (
                self.codec_weight_dtype
                if self._runtime is None
                else self._runtime.codec_weight_dtype
            ),
            "codec_compute_dtype": self.codec_compute_dtype,
            "n_vq": None if self._runtime is None else int(self._runtime.n_vq),
            "sample_rate": None if self._runtime is None else int(self._runtime.sample_rate),
            "reference_cache_entries": 0 if self._runtime is None else len(self._runtime.reference_audio_cache),
            "reference_cache_hits": 0 if self._runtime is None else int(self._runtime.reference_audio_cache_hits),
            "reference_cache_misses": 0 if self._runtime is None else int(self._runtime.reference_audio_cache_misses),
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
                    "loaded": profile_id == self._active_profile and self._runtime is not None,
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
        runtime = self._runtime
        if runtime is not None and hasattr(runtime, "close"):
            try:
                runtime.close()
            except Exception:
                logging.exception("failed to close runtime")
        self._runtime = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def close(self) -> None:
        with self._lock:
            self._release_runtime()
            self._active_profile = None
            self._set_status(state="not_loaded")

    def _load(self, profile_id: str) -> QwenWorkerRuntime:
        if profile_id not in self.profiles:
            raise ValueError(f"unknown model profile: {profile_id}")
        profile = self.profiles[profile_id]
        with self._lock:
            if self._runtime is None or self._active_profile != profile_id:
                self._set_status(state="loading")
                try:
                    self._release_runtime()
                    if not Path(self.qwen_python).is_file():
                        raise RuntimeError(f"Qwen Python环境不存在：{self.qwen_python}")
                    if self.qwentts_library and not Path(self.qwentts_library).expanduser().is_file():
                        raise RuntimeError(f"qwentts.cpp Metal动态库不存在：{self.qwentts_library}")
                    self._runtime = QwenWorkerRuntime(
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
                    self._active_profile = profile_id
                except Exception as exc:
                    self._set_status(state="error", error=str(exc))
                    raise
                self._set_status(state="ready")
            return self._runtime

    @contextmanager
    def session(self, profile_id: str):
        """Allow parallel requests for one model, but switch only when idle."""
        profile_id = profile_id if profile_id in self.profiles else DEFAULT_MODEL_PROFILE
        needs_load = False
        with self._session_condition:
            wait_deadline = time.monotonic() + self._session_wait_timeout
            while self._switching or (self._session_count > 0 and self._active_profile != profile_id):
                remaining = wait_deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"等待切换到 {profile_id} 超过 {self._session_wait_timeout:.0f} 秒"
                    )
                self._session_condition.wait(timeout=min(1.0, remaining))
            if self._active_profile != profile_id or self._runtime is None:
                self._switching = True
                needs_load = True
            else:
                self._session_count += 1
        if needs_load:
            try:
                runtime = self._load(profile_id)
            except Exception:
                with self._session_condition:
                    self._switching = False
                    self._session_condition.notify_all()
                raise
            with self._session_condition:
                self._switching = False
                self._session_count = 1
                self._session_condition.notify_all()
        else:
            runtime = self._runtime
        try:
            yield runtime
        finally:
            with self._session_condition:
                self._session_count = max(0, self._session_count - 1)
                self._session_condition.notify_all()


class GpuGenerationScheduler:
    """Bound GPU inference concurrency with interactive priority and fairness."""

    def __init__(self, max_parallel: int = 1, interactive_burst_limit: int = 4) -> None:
        self.max_parallel = max(1, int(max_parallel))
        self.interactive_burst_limit = max(1, int(interactive_burst_limit))
        self._condition = threading.Condition()
        self._active = 0
        self._waiting_interactive = 0
        self._waiting_document = 0
        self._interactive_burst = 0

    def _acquire(self, priority: str) -> None:
        interactive = priority == "interactive"
        with self._condition:
            if interactive:
                self._waiting_interactive += 1
            else:
                self._waiting_document += 1
            try:
                while (
                    self._active >= self.max_parallel
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
                    self._condition.wait()
                self._active += 1
                if interactive and self._waiting_document > 0:
                    self._interactive_burst += 1
                elif not interactive:
                    self._interactive_burst = 0
            finally:
                if interactive:
                    self._waiting_interactive -= 1
                else:
                    self._waiting_document -= 1

    def _release(self) -> None:
        with self._condition:
            self._active = max(0, self._active - 1)
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

    def status(self) -> dict[str, int]:
        with self._condition:
            return {
                "max_parallel": self.max_parallel,
                "active": self._active,
                "waiting_interactive": self._waiting_interactive,
                "waiting_document": self._waiting_document,
                "interactive_burst": self._interactive_burst,
                "interactive_burst_limit": self.interactive_burst_limit,
            }


class StreamingJob:
    def __init__(
        self,
        job_id: str,
        *,
        status: dict[str, Any] | None = None,
        result: dict[str, Any] | None = None,
        persist_callback: Callable[["StreamingJob"], None] | None = None,
    ) -> None:
        self.job_id = job_id
        self.audio_queue: queue.Queue[bytes | None] = queue.Queue(maxsize=64)
        self.status_lock = threading.Lock()
        default_status: dict[str, Any] = {
            "job_id": job_id,
            "state": "queued",
            "created_at": time.time(),
            "updated_at": time.time(),
            "started_at": None,
            "first_audio_at": None,
            "sample_rate": 48000,
            "channels": 2,
            "generated_frames": 0,
            "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
            "generated_audio_seconds": 0.0,
            "emitted_audio_seconds": 0.0,
            "lead_seconds": 0.0,
            "error": None,
            "closed": False,
        }
        if status:
            default_status.update(status)
        self.status = default_status
        self.result = result
        self.thread: threading.Thread | None = None
        self.is_closed = bool(self.status.get("closed", False))
        self._persist_callback = persist_callback
        self._last_persist_at = 0.0
        self._persist_lock = threading.Lock()

    def update(self, **kwargs: Any) -> None:
        should_persist = False
        with self.status_lock:
            self.status.update(kwargs)
            self.status["updated_at"] = time.time()
            should_persist = (
                str(self.status.get("state")) in {"finished", "error", "closed", "interrupted"}
                or self.status["updated_at"] - self._last_persist_at >= 0.5
            )
        if should_persist:
            self.persist()

    def set_result(self, result: dict[str, Any]) -> None:
        with self.status_lock:
            self.result = result
            self.status["updated_at"] = time.time()
        self.persist()

    def snapshot(self) -> dict[str, Any]:
        with self.status_lock:
            return {**self.status, "result_ready": self.result is not None}

    def persist(self) -> None:
        if self._persist_callback is not None:
            with self._persist_lock:
                self._persist_callback(self)
                self._last_persist_at = time.time()

    def manifest(self) -> dict[str, Any]:
        with self.status_lock:
            return {"status": dict(self.status), "result": self.result}


class StreamingJobManager:
    def __init__(self, root_dir: str | Path) -> None:
        self._jobs: dict[str, StreamingJob] = {}
        self._lock = threading.Lock()
        self.root_dir = Path(root_dir).resolve()
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._load_existing()

    def _manifest_path(self, job_id: str) -> Path:
        if not re.fullmatch(r"[a-f0-9]{32}", job_id or ""):
            raise ValueError("invalid job id")
        return self.root_dir / f"{job_id}.json"

    def _load_existing(self) -> None:
        for path in self.root_dir.glob("*.json"):
            try:
                manifest = json.loads(path.read_text(encoding="utf-8"))
                status = dict(manifest.get("status") or {})
                job_id = str(status.get("job_id") or path.stem)
                if status.get("state") in {"queued", "loading_runtime", "running"}:
                    status.update(
                        state="interrupted",
                        error="服务重启时任务仍未完成，请重新提交",
                        closed=True,
                        updated_at=time.time(),
                    )
                job = StreamingJob(
                    job_id,
                    status=status,
                    result=manifest.get("result"),
                    persist_callback=self._persist,
                )
                self._jobs[job_id] = job
                job.persist()
            except Exception:
                logging.exception("failed to restore service job manifest: %s", path)

    def _persist(self, job: StreamingJob) -> None:
        path = self._manifest_path(job.job_id)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(job.manifest(), ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, path)

    def create(self, *, summary: dict[str, Any] | None = None) -> StreamingJob:
        job = StreamingJob(uuid.uuid4().hex, persist_callback=self._persist)
        if summary:
            job.update(**summary)
        with self._lock:
            self._jobs[job.job_id] = job
        job.persist()
        return job

    def list(self, *, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            jobs = list(self._jobs.values())
        snapshots = [job.snapshot() for job in jobs]
        snapshots.sort(key=lambda item: float(item.get("updated_at") or 0), reverse=True)
        return snapshots[: max(1, int(limit))]

    def get(self, job_id: str) -> StreamingJob:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"stream job not found: {job_id}")
        return job

    def close(self, job_id: str) -> StreamingJob:
        job = self.get(job_id)
        with job.status_lock:
            job.is_closed = True
            job.status["closed"] = True
            if job.status.get("state") not in {"finished", "error"}:
                job.status["state"] = "closed"
            try:
                job.audio_queue.put_nowait(None)
            except queue.Full:
                pass
        job.persist()
        return job


def create_app(
    *,
    qwen_python: str | Path = DEFAULT_QWEN_PYTHON,
    qwen_worker_script: str | Path = DEFAULT_QWEN_WORKER_SCRIPT,
    qwen_0_6b_model_dir: str | Path = DEFAULT_QWEN_0_6B_MODEL_DIR,
    qwen_1_7b_model_dir: str | Path = DEFAULT_QWEN_1_7B_MODEL_DIR,
    qwen_0_6b_lanes: int = 1,
    qwen_1_7b_lanes: int = 1,
    qwen_backend: str = DEFAULT_QWEN_BACKEND,
    qwen_quant: str = DEFAULT_QWEN_QUANT,
    qwentts_library: str | Path = DEFAULT_QWENTTS_LIBRARY,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    upload_dir: str | Path = DEFAULT_UPLOAD_DIR,
    preset_dir: str | Path = DEFAULT_PRESET_DIR,
    preload: bool = True,
    max_parallel_generations: int = 1,
    document_parallel_generations: int = 2,
    access_password: str = "",
    stt_enabled: bool = True,
    stt_preload: bool | None = None,
    whisper_server: str | Path = DEFAULT_WHISPER_SERVER,
    whisper_model: str | Path = DEFAULT_WHISPER_MODEL,
    whisper_port: int = DEFAULT_WHISPER_PORT,
    whisper_threads: int = 8,
) -> FastAPI:
    runtime_manager = RuntimeManager(
        qwen_python=str(qwen_python),
        qwen_worker_script=str(qwen_worker_script),
        qwen_0_6b_model_dir=str(qwen_0_6b_model_dir),
        qwen_1_7b_model_dir=str(qwen_1_7b_model_dir),
        qwen_0_6b_lanes=max(1, int(qwen_0_6b_lanes)),
        qwen_1_7b_lanes=max(1, int(qwen_1_7b_lanes)),
        qwen_backend=str(qwen_backend),
        qwen_quant=str(qwen_quant),
        qwentts_library=str(qwentts_library),
    )
    jobs = StreamingJobManager(DEFAULT_SERVICE_JOB_DIR)
    output_dir = Path(output_dir)
    upload_dir = Path(upload_dir)
    reader_temp_dir = output_dir / "reader-temporary-audio"
    output_dir.mkdir(parents=True, exist_ok=True)
    upload_dir.mkdir(parents=True, exist_ok=True)
    reader_temp_dir.mkdir(parents=True, exist_ok=True)
    preset_store = VoicePresetStore(preset_dir)
    generation_scheduler = GpuGenerationScheduler(max_parallel=max_parallel_generations)
    ffmpeg_path = shutil.which("ffmpeg") or "ffmpeg"
    stt_runtime = WhisperCppRuntime(
        binary=whisper_server,
        model=whisper_model,
        port=int(whisper_port),
        threads=max(1, int(whisper_threads)),
        log_path=REPO_ROOT / "logs" / "whisper-server.log",
    )
    should_preload_stt = bool(preload if stt_preload is None else stt_preload)
    def synthesize_for_profile_runtime(runtime: Any, request: StreamingRequest, *, output_dir: str | Path):
        yield from runtime.synthesize(request, output_dir=output_dir)

    document_projects = DocumentProjectManager(
        root_dir=DEFAULT_DOCUMENT_PROJECT_DIR,
        runtime_session=runtime_manager.session,
        synthesize_fn=synthesize_for_profile_runtime,
        request_cls=StreamingRequest,
        generation_lock=generation_scheduler,
        ffmpeg_path=ffmpeg_path,
        synthesis_workers=min(
            max(1, int(document_parallel_generations)),
            max(1, int(max_parallel_generations)),
        ),
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if preload:
            with runtime_manager.session(DEFAULT_MODEL_PROFILE):
                pass
        if stt_enabled and should_preload_stt:
            try:
                stt_runtime.start()
            except STTUnavailableError:
                logging.exception("STT preload failed; TTS service will remain available")
        try:
            yield
        finally:
            stt_runtime.close()
            runtime_manager.close()

    app = FastAPI(title="Qwen3-TTS Apple Silicon Service", lifespan=lifespan)
    app.state.stt_runtime = stt_runtime
    app.mount(
        "/reader-assets",
        StaticFiles(directory=str(NOVEL_READER_WEB_DIR)),
        name="novel-reader-assets",
    )
    configured_cors = [
        origin.strip()
        for origin in os.environ.get("QWEN_TTS_CORS_ORIGINS", "").split(",")
        if origin.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        # The reader and native app are same-origin/local clients. Cross-origin
        # browser access is opt-in so a random website cannot probe a LAN-bound
        # service with a leaked API key.
        allow_origins=configured_cors,
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-API-Key"],
        expose_headers=[
            "X-Audio-Codec",
            "X-Audio-Sample-Rate",
            "X-Audio-Channels",
            "X-Stream-Id",
        ],
    )
    resolved_access_password = str(access_password or "")
    expected_session = hmac.new(
        resolved_access_password.encode("utf-8"), b"qwen-tts-service-session", hashlib.sha256
    ).hexdigest()

    @app.middleware("http")
    async def require_service_login(request: Request, call_next):
        if (
            not resolved_access_password
            or request.method == "OPTIONS"
            or request.url.path in {"/login", "/api/health"}
        ):
            return await call_next(request)
        supplied = request.cookies.get(SERVICE_AUTH_COOKIE, "")
        authorization = request.headers.get("authorization", "")
        bearer = authorization[7:].strip() if authorization.lower().startswith("bearer ") else ""
        api_key = request.headers.get("x-api-key", "")
        if (
            hmac.compare_digest(supplied, expected_session)
            or hmac.compare_digest(bearer, resolved_access_password)
            or hmac.compare_digest(api_key, resolved_access_password)
        ):
            return await call_next(request)
        if request.url.path.startswith("/api/"):
            return JSONResponse({"detail": "authentication required"}, status_code=401)
        next_path = request.url.path if request.url.path.startswith("/") else "/"
        return RedirectResponse(url=f"/login?next={next_path}", status_code=303)

    @app.get("/login", response_class=HTMLResponse)
    async def login_page(request: Request):
        if not resolved_access_password:
            return RedirectResponse(url="/", status_code=303)
        supplied = request.cookies.get(SERVICE_AUTH_COOKIE, "")
        if hmac.compare_digest(supplied, expected_session):
            return RedirectResponse(url="/", status_code=303)
        next_path = request.query_params.get("next", "/")
        if not next_path.startswith("/") or next_path.startswith("//"):
            next_path = "/"
        return HTMLResponse(_login_html(next_path=next_path, error=""))

    @app.post("/login", response_class=HTMLResponse)
    async def login_submit(password: str = Form(""), next_path: str = Form("/")):
        if not resolved_access_password:
            return RedirectResponse(url="/", status_code=303)
        if not hmac.compare_digest(str(password), resolved_access_password):
            return HTMLResponse(_login_html(next_path=next_path, error="密码不正确"), status_code=401)
        if not next_path.startswith("/") or next_path.startswith("//"):
            next_path = "/"
        response = RedirectResponse(url=next_path, status_code=303)
        response.set_cookie(
            SERVICE_AUTH_COOKIE,
            expected_session,
            max_age=30 * 24 * 60 * 60,
            httponly=True,
            samesite="strict",
        )
        return response

    @app.post("/logout")
    async def logout() -> RedirectResponse:
        response = RedirectResponse(url="/login", status_code=303)
        response.delete_cookie(SERVICE_AUTH_COOKIE)
        return response

    @app.get("/")
    async def index() -> RedirectResponse:
        return RedirectResponse(url="/reader", status_code=307)

    @app.get("/reader")
    async def novel_reader() -> FileResponse:
        return FileResponse(
            str(NOVEL_READER_WEB_DIR / "index.html"),
            media_type="text/html",
        )

    def _put_stream_audio(job: StreamingJob, pcm_bytes: bytes) -> None:
        with job.status_lock:
            if job.is_closed:
                return
        try:
            job.audio_queue.put_nowait(pcm_bytes)
        except queue.Full:
            # Browser playback is best-effort. Never let a disconnected or
            # slow client block the server-side generation task.
            try:
                job.audio_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                job.audio_queue.put_nowait(pcm_bytes)
            except queue.Full:
                pass

    def _finish_stream_audio(job: StreamingJob) -> None:
        while True:
            try:
                job.audio_queue.put_nowait(None)
                return
            except queue.Full:
                try:
                    job.audio_queue.get_nowait()
                except queue.Empty:
                    return

    def _remove_generated_result_files(result: dict[str, Any] | None) -> None:
        if not result:
            return
        resolved_output = output_dir.resolve()
        for key in ("audio_path", "tokens_path", "metadata_path"):
            raw_path = result.get(key)
            if not raw_path:
                continue
            try:
                candidate = Path(str(raw_path)).resolve()
                candidate.relative_to(resolved_output)
            except (OSError, RuntimeError, ValueError):
                continue
            if candidate.is_file():
                candidate.unlink(missing_ok=True)

    def _run_job(
        job: StreamingJob, request: StreamingRequest, mode_name: str,
        streaming_generation: bool, model_profile: str,
    ) -> None:
        try:
            job.update(
                state="loading_runtime",
                started_at=time.time(),
                max_new_tokens=int(request.max_new_frames),
                mode=mode_name,
                model_profile=model_profile,
                streaming_generation=streaming_generation,
            )
            with generation_scheduler.interactive_slot():
              with runtime_manager.session(model_profile) as runtime:
                channels = int(runtime_manager.profiles[model_profile]["channels"])
                job.update(state="running", sample_rate=runtime.sample_rate, channels=channels, n_vq=runtime.n_vq)
                for event in synthesize_for_profile_runtime(runtime, request, output_dir=output_dir):
                    with job.status_lock:
                        if job.is_closed:
                            if event.type == "result" and job.status.get("ephemeral_audio"):
                                _remove_generated_result_files(event.data)
                            break
                    if event.type == "metadata":
                        job.update(**event.data)
                    elif event.type == "progress":
                        job.update(**event.data)
                    elif event.type == "audio":
                        waveform = event.data["waveform"]
                        channels = 1 if waveform.ndim == 1 else int(min(2, waveform.shape[0]))
                        with job.status_lock:
                            if job.status.get("first_audio_at") is None:
                                job.status["first_audio_at"] = time.time()
                        if streaming_generation:
                            _put_stream_audio(job, _pcm16le_bytes(waveform, channels))
                        job.update(
                            generated_frames=event.data.get("generated_frames", job.snapshot().get("generated_frames", 0)),
                            emitted_audio_seconds=event.data.get("emitted_audio_seconds", 0.0),
                            generated_audio_seconds=event.data.get("generated_audio_seconds", 0.0),
                            sample_rate=event.data.get("sample_rate", runtime.sample_rate),
                            channels=channels,
                            lead_seconds=event.data.get("lead_seconds", 0.0),
                            generation_lead_seconds=event.data.get("generation_lead_seconds", 0.0),
                            playback_lead_seconds=event.data.get("playback_lead_seconds"),
                            generation_realtime_factor=event.data.get("generation_realtime_factor", 0.0),
                            post_first_generation_realtime_factor=event.data.get(
                                "post_first_generation_realtime_factor"
                            ),
                            first_audio_latency_seconds=event.data.get("first_audio_latency_seconds"),
                            decode_chunks_submitted=event.data.get("decode_chunks_submitted", 0),
                            decode_queue_depth=event.data.get("decode_queue_depth", 0),
                            pending_decode_frames=event.data.get("pending_decode_frames", 0),
                            chunk_frames=event.data.get("chunk_frames", 0),
                        )
                    elif event.type == "result":
                        metadata = dict(event.data["metadata"])
                        task_seed_status = job.snapshot()
                        metadata["seed"] = int(request.seed) if request.seed is not None else None
                        metadata["configured_seed"] = task_seed_status.get("configured_seed")
                        metadata["seed_mode"] = task_seed_status.get("seed_mode")
                        job.set_result({
                            "audio_path": event.data["audio_path"],
                            "tokens_path": event.data["tokens_path"],
                            "metadata_path": event.data["metadata_path"],
                            "metadata": metadata,
                        })
                        job.update(
                            state="finished",
                            generated_frames=metadata.get("generated_frames", 0),
                            emitted_audio_seconds=metadata.get("duration_seconds", 0.0),
                            audio_path=event.data["audio_path"],
                        )
            _finish_stream_audio(job)
        except Exception as exc:  # noqa: BLE001
            job.update(state="error", error=str(exc))
            _finish_stream_audio(job)

    @app.post("/api/generate-stream/start")
    async def generate_stream_start(
        mode: str = Form("voice_clone"),
        language: str = Form(""),
        text: str = Form(...),
        prompt_text: str = Form(""),
        max_new_tokens: int | None = Form(None),
        codec_chunk_frames: int | None = Form(None),
        seed: int | None = Form(None),
        tokens_control: int = Form(0),
        tokens: int = Form(0),
        temperature: float | None = Form(None),
        top_p: float | None = Form(None),
        top_k: int | None = Form(None),
        repetition_penalty: float | None = Form(None),
        model_profile: str = Form(""),
        voice_name: str = Form(""),
        streaming_generation: int | None = Form(None),
        qwen_clone_mode: str = Form(""),
        qwen_reference_text: str = Form(""),
        qwen_non_streaming_mode: int | None = Form(None),
        qwen_append_silence: int | None = Form(None),
        qwen_instruct: str = Form(""),
        qwen_min_new_tokens: int | None = Form(None),
        example_audio_path: str = Form(""),
        prompt_audio: UploadFile | None = File(None),
        use_service_settings: int = Form(1),
        ephemeral_audio: int = Form(0),
    ) -> JSONResponse:
        text = (text or "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="text must not be empty")

        mode = "voice_clone"
        language = "Chinese"
        mode_name = MODE_CLONE
        service_settings = dict(active_service_settings().get("settings") or {})
        use_applied_service_settings = bool(
            _safe_int(use_service_settings, default=1, minimum=0, maximum=1)
        )
        def service_value(explicit: Any, key: str, fallback: Any) -> Any:
            if use_applied_service_settings:
                return service_settings.get(key, fallback)
            return explicit if explicit is not None and explicit != "" else fallback

        max_new_tokens = service_value(max_new_tokens, "qwen_max_new_tokens", 2048)
        codec_chunk_frames = service_value(codec_chunk_frames, "qwen_chunk_size", 8)
        seed = service_value(seed, "qwen_seed", 1234)
        temperature = service_value(temperature, "qwen_temperature", 0.9)
        top_p = service_value(top_p, "qwen_top_p", 1.0)
        top_k = service_value(top_k, "qwen_top_k", 50)
        repetition_penalty = (
            service_value(repetition_penalty, "qwen_repetition_penalty", 1.05)
        )
        model_profile = str(service_value(model_profile, "model_profile", DEFAULT_MODEL_PROFILE))
        voice_name = str(service_value(voice_name, "voice_name", "本地克隆音色"))
        # Delivery mode belongs to the caller. The applied service setting owns
        # the voice and synthesis parameters, while a reader may still choose
        # realtime streaming or complete-block playback.
        streaming_generation = _safe_int(
            streaming_generation if streaming_generation is not None
            else service_settings.get("qwen_streaming_generation", 1),
            default=1,
            minimum=0,
            maximum=1,
        )
        qwen_clone_mode = str(service_value(qwen_clone_mode, "qwen_clone_mode", "xvec"))
        qwen_reference_text = str(service_value(
            qwen_reference_text, "qwen_reference_text", ""
        ))
        qwen_non_streaming_mode = _safe_int(
            qwen_non_streaming_mode if qwen_non_streaming_mode is not None
            else service_settings.get("qwen_non_streaming_mode", 0),
            default=0,
            minimum=0,
            maximum=1,
        )
        qwen_append_silence = int(bool(service_value(
            qwen_append_silence, "qwen_append_silence", True
        )))
        qwen_min_new_tokens = service_value(qwen_min_new_tokens, "qwen_min_new_tokens", 2)
        example_audio_path = str(service_value(
            example_audio_path, "reference_audio_path", ""
        ))
        if use_applied_service_settings:
            prompt_audio = None

        prompt_audio_path = ""
        if prompt_audio is not None and prompt_audio.filename:
            suffix = Path(prompt_audio.filename).suffix or ".wav"
            prompt_path = upload_dir / f"{uuid.uuid4().hex}{suffix}"
            prompt_path.write_bytes(await prompt_audio.read())
            prompt_audio_path = str(prompt_path)
        elif example_audio_path:
            try:
                prompt_audio_path = str(
                    _resolve_allowed_reference_audio_path(
                        example_audio_path,
                        REFERENCE_AUDIO_DIR,
                        preset_store.audio_dir,
                    )
                )
            except (FileNotFoundError, PermissionError):
                prompt_audio_path = ""

        if not prompt_audio_path and DEFAULT_CLONE_AUDIO_PATH:
            prompt_audio_path = DEFAULT_CLONE_AUDIO_PATH
        if not prompt_audio_path:
            raise HTTPException(status_code=500, detail="no clone reference audio is available")

        max_new_tokens = _safe_int(
            max_new_tokens,
            default=DEFAULT_MAX_NEW_TOKENS,
            minimum=1,
            maximum=DEFAULT_MAX_NEW_TOKENS,
        )
        codec_chunk_frames = _safe_int(codec_chunk_frames, default=16, minimum=0, maximum=32)
        streaming_generation_enabled = bool(_safe_int(streaming_generation, default=1, minimum=0, maximum=1))
        if model_profile not in runtime_manager.profiles:
            raise HTTPException(status_code=400, detail="invalid model profile")
        if runtime_manager.profiles[model_profile].get("backend") == "qwen":
            max_new_tokens = _safe_int(max_new_tokens, default=2048, minimum=2, maximum=2048)
            codec_chunk_frames = _safe_int(codec_chunk_frames, default=8, minimum=1, maximum=24)
        configured_seed = _safe_int(seed, default=1234, minimum=-1, maximum=999999)
        seed_mode = "random" if configured_seed < 0 else "fixed"
        resolved_seed = secrets.randbelow(1_000_000) if configured_seed < 0 else configured_seed
        if not runtime_manager.profiles[model_profile]["streaming"]:
            streaming_generation_enabled = False
        request = StreamingRequest(
            text=text,
            mode="voice_clone",
            prompt_text="",
            prompt_audio_path=prompt_audio_path,
            language="Chinese",
            tokens_control=bool(int(tokens_control)),
            tokens=_safe_int(tokens, default=0, minimum=0),
            max_new_frames=max_new_tokens,
            do_sample=True,
            temperature=_safe_float(temperature, default=1.7, minimum=0.1, maximum=3.0),
            top_p=_safe_float(top_p, default=0.8, minimum=0.1, maximum=1.0),
            top_k=_safe_int(top_k, default=25, minimum=1, maximum=200),
            repetition_penalty=_safe_float(repetition_penalty, default=1.0, minimum=0.8, maximum=2.0),
            seed=None if resolved_seed < 0 else resolved_seed,
            codec_chunk_frames=codec_chunk_frames,
            qwen_xvec_only=str(qwen_clone_mode or "xvec") != "icl",
            qwen_reference_text=str(qwen_reference_text or "").strip(),
            qwen_non_streaming_mode=bool(
                _safe_int(qwen_non_streaming_mode, default=0, minimum=0, maximum=1)
            ),
            qwen_append_silence=bool(
                _safe_int(qwen_append_silence, default=1, minimum=0, maximum=1)
            ),
            qwen_instruct="",
            qwen_min_new_tokens=_safe_int(
                qwen_min_new_tokens, default=2, minimum=2, maximum=256
            ),
        )
        job = jobs.create(
            summary={
                "task_type": "text_generation",
                "title": text[:80],
                "text_preview": text[:240],
                "model_profile": model_profile,
                "model_label": MODEL_PROFILE_LABELS[model_profile],
                "voice_name": str(voice_name or "本地克隆音色")[:120],
                "seed": resolved_seed,
                "configured_seed": configured_seed,
                "seed_mode": seed_mode,
                "ephemeral_audio": bool(
                    _safe_int(ephemeral_audio, default=0, minimum=0, maximum=1)
                ),
            }
        )
        thread = threading.Thread(
            target=_run_job,
            args=(job, request, mode_name, streaming_generation_enabled, model_profile),
            daemon=True,
        )
        job.thread = thread
        thread.start()
        return JSONResponse(
            {
                "job_id": job.job_id,
                "audio_url": f"/api/generate-stream/{job.job_id}/audio",
                "status_url": f"/api/generate-stream/{job.job_id}/status",
                "result_url": f"/api/generate-stream/{job.job_id}/result",
                "sample_rate": runtime_manager.profiles[model_profile]["sample_rate"],
                "channels": runtime_manager.profiles[model_profile]["channels"],
                "model_profile": model_profile,
                "streaming_generation": streaming_generation_enabled,
                "seed": resolved_seed,
                "configured_seed": configured_seed,
                "seed_mode": seed_mode,
            }
        )

    @app.get("/api/reference-audio")
    async def reference_audio(path: str) -> FileResponse:
        try:
            candidate = _resolve_allowed_reference_audio_path(
                path,
                REFERENCE_AUDIO_DIR,
                preset_store.audio_dir,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="reference audio not found") from exc
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail="reference audio path is not allowed") from exc
        media_type = mimetypes.guess_type(str(candidate))[0] or "application/octet-stream"
        return FileResponse(str(candidate), media_type=media_type, filename=candidate.name)

    def preset_payload(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        name = str(payload.get("name") or "").strip()
        settings = payload.get("settings")
        if not isinstance(settings, dict):
            raise HTTPException(status_code=400, detail="预设设置必须是对象")
        copied = dict(settings)
        model_profile = str(copied.get("model_profile") or DEFAULT_MODEL_PROFILE)
        if model_profile not in runtime_manager.profiles:
            raise HTTPException(status_code=400, detail="无效的模型预设")
        copied["model_profile"] = model_profile
        reference_path = str(copied.get("reference_audio_path") or "").strip()
        if reference_path:
            try:
                copied["reference_audio_path"] = str(
                    _resolve_allowed_reference_audio_path(
                        reference_path,
                        REFERENCE_AUDIO_DIR,
                        preset_store.audio_dir,
                    )
                )
            except FileNotFoundError as exc:
                raise HTTPException(status_code=400, detail="预设参考音频不存在") from exc
            except PermissionError as exc:
                raise HTTPException(status_code=400, detail="预设参考音频不在允许目录") from exc
        return name, copied

    @app.get("/api/presets")
    async def list_presets() -> JSONResponse:
        active = preset_store.active()
        return JSONResponse(
            {
                "presets": preset_store.list(),
                "active_preset_id": active["id"] if active else "",
            }
        )

    def active_service_settings() -> dict[str, Any]:
        configuration = preset_store.service_configuration()
        if configuration is not None:
            return {
                "active_preset_id": str(configuration.get("preset_id") or ""),
                "name": str(configuration.get("name") or "服务设置"),
                "source": "preset" if configuration.get("preset_id") else "studio",
                "settings": dict(configuration.get("settings") or {}),
                "updated_at": configuration.get("updated_at"),
            }
        active = preset_store.active()
        if active is not None:
            return {
                "active_preset_id": active["id"],
                "name": active["name"],
                "source": "preset",
                "settings": dict(active.get("settings") or {}),
            }
        default_voice = next(
            (voice for voice in BAILIAN_VOICE_ROWS if voice.get("audio_path") == DEFAULT_CLONE_AUDIO_PATH),
            BAILIAN_VOICE_ROWS[0] if BAILIAN_VOICE_ROWS else {},
        )
        return {
            "active_preset_id": "",
            "name": str(default_voice.get("name") or "服务默认音色"),
            "source": "default",
            "settings": {
                "model_profile": DEFAULT_MODEL_PROFILE,
                "voice_name": str(default_voice.get("name") or "服务默认音色"),
                "reference_audio_path": str(default_voice.get("audio_path") or DEFAULT_CLONE_AUDIO_PATH),
                "qwen_clone_mode": "xvec",
                "qwen_reference_text": "",
                "qwen_temperature": 0.9,
                "qwen_top_p": 1.0,
                "qwen_top_k": 50,
                "qwen_repetition_penalty": 1.05,
                "qwen_max_new_tokens": 2048,
                "qwen_chunk_size": 8,
                "qwen_min_new_tokens": 2,
                "qwen_seed": 1234,
                "qwen_append_silence": True,
                "qwen_aac_bitrate": "80k",
            },
        }

    @app.get("/api/service-settings")
    async def get_service_settings() -> JSONResponse:
        return JSONResponse(active_service_settings())

    @app.put("/api/service-settings/active-preset")
    async def activate_service_preset(payload: dict[str, Any] = Body(...)) -> JSONResponse:
        try:
            preset = preset_store.activate(str(payload.get("preset_id") or ""))
            return JSONResponse(
                {
                    "active_preset_id": preset["id"],
                    "name": preset["name"],
                    "source": "preset",
                    "settings": dict(preset.get("settings") or {}),
                }
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="preset not found") from exc

    @app.put("/api/service-settings")
    async def apply_service_settings(payload: dict[str, Any] = Body(...)) -> JSONResponse:
        try:
            name, settings = preset_payload(payload)
            configuration = preset_store.apply_configuration(name=name or "音频工作台设置", settings=settings)
            return JSONResponse(
                {
                    "active_preset_id": "",
                    "name": configuration["name"],
                    "source": "studio",
                    "settings": dict(configuration.get("settings") or {}),
                    "updated_at": configuration.get("updated_at"),
                }
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.delete("/api/service-settings")
    async def reset_service_settings() -> JSONResponse:
        preset_store.clear_configuration()
        return JSONResponse(active_service_settings())

    def builtin_reference_id(path: str) -> str:
        return "builtin-" + hashlib.sha256(str(Path(path).resolve()).encode()).hexdigest()[:24]

    def reference_audio_library(*, include_hidden: bool) -> list[dict[str, Any]]:
        hidden_builtin = preset_store.hidden_builtin_references()
        rows: list[dict[str, Any]] = []
        for voice in BAILIAN_VOICE_ROWS:
            path = str(Path(str(voice.get("audio_path") or "")).resolve())
            hidden = path in hidden_builtin
            if hidden and not include_hidden:
                continue
            rows.append(
                {
                    "id": builtin_reference_id(path),
                    "kind": "builtin",
                    "name": str(voice.get("name") or Path(path).stem),
                    "description": str(voice.get("description") or ""),
                    "path": path,
                    "audio_path": path,
                    "language": str(voice.get("language") or "Chinese"),
                    "transcript": str(voice.get("transcript") or ""),
                    "transcript_source": str(voice.get("transcript_source") or ""),
                    "hidden": hidden,
                    "in_use": bool(preset_store.reference_usage(path)),
                    "usages": preset_store.reference_usage(path),
                }
            )
        for item in preset_store.list_reference_audio(include_hidden=include_hidden):
            path = str(item.get("path") or "")
            rows.append(
                {
                    **item,
                    "kind": "custom",
                    "audio_path": path,
                    "description": "用户参考音频",
                    "language": "Chinese",
                    "transcript": "",
                    "transcript_source": "",
                    "in_use": bool(preset_store.reference_usage(path)),
                    "usages": preset_store.reference_usage(path),
                }
            )
        return rows

    @app.get("/api/voices")
    async def list_native_voices() -> JSONResponse:
        visible_references = reference_audio_library(include_hidden=False)
        return JSONResponse(
            {
                "voices": visible_references,
                "default_reference_audio_path": DEFAULT_CLONE_AUDIO_PATH,
                "models": [
                    {
                        "id": profile_id,
                        "label": MODEL_PROFILE_LABELS[profile_id],
                    }
                    for profile_id in ("qwen_0_6b", "qwen_1_7b")
                ],
            }
        )

    @app.get("/api/reference-audio-library")
    async def list_reference_audio_library(include_hidden: bool = False) -> JSONResponse:
        return JSONResponse(
            {"references": reference_audio_library(include_hidden=include_hidden)}
        )

    @app.put("/api/reference-audio-library/{reference_id}/visibility")
    async def update_reference_audio_visibility(
        reference_id: str,
        payload: dict[str, Any] = Body(...),
    ) -> JSONResponse:
        hidden = bool(payload.get("hidden"))
        builtin = next(
            (
                voice
                for voice in BAILIAN_VOICE_ROWS
                if builtin_reference_id(str(voice.get("audio_path") or "")) == reference_id
            ),
            None,
        )
        if builtin is not None:
            preset_store.set_builtin_hidden(str(builtin["audio_path"]), hidden)
        else:
            try:
                preset_store.set_reference_hidden(reference_id, hidden)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail="reference audio not found") from exc
        return JSONResponse({"ok": True, "hidden": hidden})

    @app.delete("/api/reference-audio-library/{reference_id}")
    async def delete_reference_audio(reference_id: str) -> JSONResponse:
        if any(
            builtin_reference_id(str(voice.get("audio_path") or "")) == reference_id
            for voice in BAILIAN_VOICE_ROWS
        ):
            raise HTTPException(status_code=400, detail="内置参考音频只能隐藏，不能删除")
        try:
            preset_store.delete_reference_audio(reference_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="reference audio not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return JSONResponse({"ok": True})

    @app.post("/api/presets")
    async def create_preset(payload: dict[str, Any] = Body(...)) -> JSONResponse:
        try:
            name, settings = preset_payload(payload)
            return JSONResponse(preset_store.create(name=name, settings=settings), status_code=201)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.put("/api/presets/{preset_id}")
    async def update_preset(preset_id: str, payload: dict[str, Any] = Body(...)) -> JSONResponse:
        try:
            name, settings = preset_payload(payload)
            return JSONResponse(preset_store.update(preset_id, name=name, settings=settings))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="preset not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.delete("/api/presets/{preset_id}")
    async def delete_preset(preset_id: str) -> JSONResponse:
        try:
            preset_store.delete(preset_id)
            return JSONResponse({"ok": True})
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="preset not found") from exc

    @app.post("/api/presets/reference-audio")
    async def import_preset_reference_audio(audio: UploadFile = File(...)) -> JSONResponse:
        filename = audio.filename or "reference.wav"
        data = await audio.read()
        if not data:
            raise HTTPException(status_code=400, detail="参考音频为空")
        if len(data) > 100 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="参考音频不能超过 100 MB")
        temporary = upload_dir / f"preset-{uuid.uuid4().hex}{Path(filename).suffix or '.wav'}"
        normalized = upload_dir / f"preset-normalized-{uuid.uuid4().hex}.wav"
        try:
            temporary.write_bytes(data)
            conversion = subprocess.run(
                [
                    ffmpeg_path,
                    "-y",
                    "-v",
                    "error",
                    "-i",
                    str(temporary),
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    "24000",
                    "-c:a",
                    "pcm_s16le",
                    str(normalized),
                ],
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            if conversion.returncode != 0 or not normalized.is_file() or normalized.stat().st_size <= 44:
                detail = (conversion.stderr or "").strip().splitlines()
                reason = detail[-1] if detail else "无法识别音频编码"
                raise ValueError(f"参考音频无法解码：{reason[:240]}")
            imported = preset_store.import_reference_audio(
                filename=f"{Path(filename).stem}.wav",
                temporary_path=normalized,
            )
        except subprocess.TimeoutExpired as exc:
            raise HTTPException(status_code=400, detail="参考音频转码超时") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            temporary.unlink(missing_ok=True)
            normalized.unlink(missing_ok=True)
        return JSONResponse(
            {
                "reference_audio_path": str(imported),
                "audio_url": f"/api/reference-audio?path={quote(str(imported), safe='')}",
                "reference": next(
                    (
                        item
                        for item in preset_store.list_reference_audio(include_hidden=True)
                        if item.get("path") == str(imported)
                    ),
                    None,
                ),
            }
        )

    @app.get("/api/generate-stream/{job_id}/audio")
    async def generate_stream_audio(job_id: str, request: Request) -> StreamingResponse:
        job = jobs.get(job_id)

        async def iterator():
            try:
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        item = await asyncio.to_thread(job.audio_queue.get, True, 0.25)
                    except queue.Empty:
                        continue
                    if item is None:
                        break
                    yield item
            finally:
                snapshot = job.snapshot()
                if snapshot.get("state") not in {"finished", "error", "closed"}:
                    jobs.close(job_id)

        snapshot = job.snapshot()
        return StreamingResponse(
            iterator(),
            media_type="application/octet-stream",
            headers={
                "X-Audio-Codec": "pcm_s16le",
                "X-Audio-Sample-Rate": str(snapshot.get("sample_rate", 48000)),
                "X-Audio-Channels": str(snapshot.get("channels", 2)),
                "X-Stream-Id": job_id,
            },
        )

    @app.get("/api/generate-stream/{job_id}/status")
    async def generate_stream_status(job_id: str) -> JSONResponse:
        return JSONResponse(jobs.get(job_id).snapshot())

    @app.get("/api/generate-stream/{job_id}/result")
    async def generate_stream_result(job_id: str) -> JSONResponse:
        job = jobs.get(job_id)
        if job.result is None:
            raise HTTPException(status_code=404, detail="result is not ready")
        return JSONResponse(job.result)

    @app.get("/api/generate-stream/{job_id}/result-audio")
    async def generate_stream_result_audio(job_id: str) -> FileResponse:
        job = jobs.get(job_id)
        if job.result is None:
            raise HTTPException(status_code=404, detail="result is not ready")
        audio_path = Path(str(job.result.get("audio_path") or ""))
        if not audio_path.is_file():
            raise HTTPException(status_code=404, detail="generated audio is missing")
        return FileResponse(str(audio_path), media_type="audio/wav", filename="generated.wav")

    @app.get("/api/generate-stream/{job_id}/result-audio-aac")
    async def generate_stream_result_audio_aac(
        job_id: str,
        bitrate: str = "80k",
    ) -> FileResponse:
        job = jobs.get(job_id)
        if job.result is None:
            raise HTTPException(status_code=404, detail="result is not ready")
        source = Path(str(job.result.get("audio_path") or ""))
        if not source.is_file():
            raise HTTPException(status_code=404, detail="generated audio is missing")
        selected_bitrate = _safe_aac_bitrate(bitrate)
        target = reader_temp_dir / f"{job_id}-{selected_bitrate}.m4a"
        if not target.is_file():
            completed = subprocess.run(
                [
                    ffmpeg_path,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(source),
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    "48000",
                    "-c:a",
                    "aac",
                    "-b:a",
                    selected_bitrate,
                    "-movflags",
                    "+faststart",
                    str(target),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0 or not target.is_file():
                target.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=500,
                    detail=(completed.stderr or "AAC encoding failed").strip(),
                )
        return FileResponse(
            str(target),
            media_type="audio/mp4",
            filename=f"reader-block-{job_id}.m4a",
        )

    def _remove_ephemeral_job_audio(job: StreamingJob) -> None:
        for target in reader_temp_dir.glob(f"{job.job_id}-*.m4a"):
            target.unlink(missing_ok=True)
        if not bool(job.snapshot().get("ephemeral_audio")) or job.result is None:
            return
        _remove_generated_result_files(job.result)
        with job.status_lock:
            job.result = None
            job.status["audio_path"] = None
            job.status["ephemeral_cleaned"] = True
            job.status["updated_at"] = time.time()
        job.persist()

    @app.delete("/api/generate-stream/{job_id}/ephemeral-audio")
    async def generate_stream_delete_ephemeral_audio(job_id: str) -> JSONResponse:
        job = jobs.close(job_id)
        _remove_ephemeral_job_audio(job)
        return JSONResponse({"ok": True, "job_id": job_id})

    @app.post("/api/generate-stream/{job_id}/ephemeral-audio/close")
    async def generate_stream_close_ephemeral_audio(job_id: str) -> JSONResponse:
        job = jobs.close(job_id)
        _remove_ephemeral_job_audio(job)
        return JSONResponse({"ok": True, "job_id": job_id})

    @app.post("/api/generate-stream/{job_id}/close")
    async def generate_stream_close(job_id: str) -> JSONResponse:
        jobs.close(job_id)
        return JSONResponse({"ok": True})

    @app.get("/api/runtime")
    async def runtime_info() -> JSONResponse:
        return JSONResponse(
            {
                "output_dir": str(output_dir),
                "upload_dir": str(upload_dir),
                "backend": runtime_manager.qwen_backend,
                "quant": runtime_manager.qwen_quant,
                "qwentts_library": runtime_manager.qwentts_library,
                "device": runtime_manager.device,
                "dtype": runtime_manager.dtype,
                "attn_implementation": runtime_manager.attn_implementation,
                "runtime": runtime_manager.status(),
                "generation_scheduler": {
                    **generation_scheduler.status(),
                    "document_parallel": document_projects.synthesis_workers,
                },
                "stt": stt_runtime.status() if stt_enabled else {
                    "state": "disabled",
                    "ready": False,
                    "available": False,
                },
            }
        )

    @app.get("/api/stt/status")
    async def stt_status() -> JSONResponse:
        if not stt_enabled:
            return JSONResponse({"state": "disabled", "ready": False, "available": False})
        return JSONResponse(stt_runtime.status())

    @app.post("/v1/audio/transcriptions")
    async def transcribe_audio(
        file: UploadFile = File(...),
        model: str = Form("whisper-small"),
        language: str = Form("auto"),
        prompt: str = Form(""),
        response_format: str = Form("json"),
    ) -> Response:
        del model  # OpenAI-compatible field; this service has one resident model.
        if not stt_enabled:
            raise HTTPException(status_code=503, detail="STT is disabled")
        if response_format not in {"json", "verbose_json", "text", "srt", "vtt"}:
            raise HTTPException(status_code=400, detail="unsupported response_format")
        audio = await file.read()
        if not audio:
            raise HTTPException(status_code=400, detail="audio file is empty")
        if len(audio) > 500 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="audio file exceeds 500 MB")
        try:
            payload, media_type = await asyncio.to_thread(
                stt_runtime.transcribe,
                audio=audio,
                filename=file.filename or "audio.wav",
                language=(language or "auto").strip(),
                prompt=(prompt or "").strip(),
                response_format=response_format,
            )
        except (STTUnavailableError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return Response(content=payload, media_type=media_type)

    @app.get("/api/health")
    async def health() -> JSONResponse:
        return JSONResponse(
            {
                **runtime_manager.status(),
                "generation_scheduler": {
                    **generation_scheduler.status(),
                    "document_parallel": document_projects.synthesis_workers,
                },
                "stt": stt_runtime.status() if stt_enabled else {
                    "state": "disabled",
                    "ready": False,
                    "available": False,
                },
            }
        )

    @app.get("/api/service/tasks")
    async def service_tasks() -> JSONResponse:
        tasks: list[dict[str, Any]] = []
        for item in jobs.list(limit=100):
            state = str(item.get("state") or "unknown")
            tasks.append(
                {
                    "type": "text",
                    "id": item["job_id"],
                    "title": item.get("title") or item.get("text_preview") or "文本生成",
                    "state": state,
                    "active": state in {"queued", "loading_runtime", "running"},
                    "created_at": item.get("created_at"),
                    "updated_at": item.get("updated_at"),
                    "model_profile": item.get("model_profile") or DEFAULT_MODEL_PROFILE,
                    "model_label": item.get("model_label") or MODEL_PROFILE_LABELS.get(
                        str(item.get("model_profile") or DEFAULT_MODEL_PROFILE), ""
                    ),
                    "voice_name": item.get("voice_name") or "",
                    "seed": item.get("seed"),
                    "configured_seed": item.get("configured_seed"),
                    "seed_mode": item.get("seed_mode"),
                    "generated_frames": item.get("generated_frames", 0),
                    "max_new_tokens": item.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS),
                    "duration_seconds": item.get("emitted_audio_seconds", 0.0),
                    "result_ready": bool(item.get("result_ready")),
                    "error": item.get("error"),
                }
            )
        for project in document_projects.list_projects():
            state = str(project.get("state") or "unknown")
            stats = project.get("stats") or {}
            tasks.append(
                {
                    "type": "document",
                    "id": project["id"],
                    "title": project.get("name") or "文档项目",
                    "state": state,
                    "active": state in {"running", "stopping"},
                    "created_at": project.get("created_at"),
                    "updated_at": project.get("updated_at"),
                    "model_profile": (project.get("settings") or {}).get("model_profile", DEFAULT_MODEL_PROFILE),
                    "model_label": (project.get("settings") or {}).get("model_label") or MODEL_PROFILE_LABELS[
                        (project.get("settings") or {}).get("model_profile", DEFAULT_MODEL_PROFILE)
                    ],
                    "voice_name": (project.get("settings") or {}).get("voice_name", ""),
                    "seed": (project.get("settings") or {}).get("seed"),
                    "configured_seed": (project.get("settings") or {}).get("configured_seed"),
                    "seed_mode": (project.get("settings") or {}).get("seed_mode"),
                    "completed_segments": stats.get("completed_segments", 0),
                    "total_segments": stats.get("total_segments", 0),
                    "progress": stats.get("progress", 0.0),
                    "eta_seconds": stats.get("eta_seconds"),
                    "error": project.get("message") if state == "error" else None,
                }
            )
        tasks.sort(
            key=lambda item: (bool(item.get("active")), float(item.get("updated_at") or 0)),
            reverse=True,
        )
        return JSONResponse(
            {
                "tasks": tasks[:100],
                "active_count": sum(1 for item in tasks if item.get("active")),
                "server_time": time.time(),
            }
        )

    def document_settings(raw: str) -> dict[str, Any]:
        try:
            incoming = json.loads(raw or "{}")
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="invalid document settings") from exc
        service_settings = dict(active_service_settings().get("settings") or {})
        inherited = {
            "model_profile": service_settings.get("model_profile"),
            "reference_audio_path": service_settings.get("reference_audio_path"),
            "voice_name": service_settings.get("voice_name"),
            "qwen_clone_mode": service_settings.get("qwen_clone_mode"),
            "qwen_reference_text": service_settings.get("qwen_reference_text"),
            "temperature": service_settings.get("qwen_temperature"),
            "top_p": service_settings.get("qwen_top_p"),
            "top_k": service_settings.get("qwen_top_k"),
            "repetition_penalty": service_settings.get("qwen_repetition_penalty"),
            "max_new_tokens": service_settings.get("qwen_max_new_tokens"),
            "codec_chunk_frames": service_settings.get("qwen_chunk_size"),
            "qwen_min_new_tokens": service_settings.get("qwen_min_new_tokens"),
            "seed": service_settings.get("qwen_seed"),
            "qwen_append_silence": service_settings.get("qwen_append_silence"),
            "aac_bitrate": service_settings.get("qwen_aac_bitrate"),
        }
        incoming = {key: value for key, value in inherited.items() if value is not None} | incoming
        reference_audio_path = str(incoming.get("reference_audio_path") or DEFAULT_CLONE_AUDIO_PATH)
        try:
            reference_candidate = _resolve_allowed_reference_audio_path(
                reference_audio_path,
                REFERENCE_AUDIO_DIR,
                preset_store.audio_dir,
            )
        except (FileNotFoundError, PermissionError) as exc:
            raise HTTPException(status_code=400, detail="invalid clone reference audio") from exc
        model_profile = str(incoming.get("model_profile") or DEFAULT_MODEL_PROFILE)
        if model_profile not in runtime_manager.profiles:
            raise HTTPException(status_code=400, detail="invalid model profile")
        qwen_profile = runtime_manager.profiles[model_profile].get("backend") == "qwen"
        qwen_clone_mode = "icl" if str(incoming.get("qwen_clone_mode") or "xvec") == "icl" else "xvec"
        qwen_reference_text = str(incoming.get("qwen_reference_text") or "").strip()
        if qwen_profile and qwen_clone_mode == "icl" and not qwen_reference_text:
            raise HTTPException(status_code=400, detail="Qwen ICL克隆模式必须填写参考音频文字")
        configured_seed = _safe_int(incoming.get("seed"), default=1234, minimum=-1, maximum=999999)
        seed_mode = "random" if configured_seed < 0 else "fixed"
        resolved_seed = secrets.randbelow(1_000_000) if configured_seed < 0 else configured_seed
        return {
            "model_profile": model_profile,
            "model_label": MODEL_PROFILE_LABELS[model_profile],
            "runtime_backend": str(runtime_manager.profiles[model_profile]["runtime_backend"]),
            "quant": str(runtime_manager.profiles[model_profile]["quant"]),
            "reference_audio_path": str(reference_candidate),
            "voice_name": str(incoming.get("voice_name") or "本地克隆音色")[:120],
            "language": "Chinese",
            "temperature": _safe_float(incoming.get("temperature"), default=1.7, minimum=0.1, maximum=3.0),
            "top_p": _safe_float(incoming.get("top_p"), default=0.8, minimum=0.1, maximum=1.0),
            "top_k": _safe_int(incoming.get("top_k"), default=25, minimum=1, maximum=200),
            "repetition_penalty": _safe_float(
                incoming.get("repetition_penalty"), default=1.0, minimum=0.8, maximum=2.0
            ),
            "max_new_tokens": _safe_int(
                incoming.get("max_new_tokens"),
                default=2048 if qwen_profile else DEFAULT_MAX_NEW_TOKENS,
                minimum=24 if qwen_profile else 80,
                maximum=2048 if qwen_profile else DEFAULT_MAX_NEW_TOKENS,
            ),
            "codec_chunk_frames": _safe_int(
                incoming.get("codec_chunk_frames"),
                default=8 if qwen_profile else 16,
                minimum=1,
                maximum=24 if qwen_profile else 32,
            ),
            "seed": resolved_seed,
            "configured_seed": configured_seed,
            "seed_mode": seed_mode,
            "qwen_clone_mode": qwen_clone_mode,
            "qwen_reference_text": qwen_reference_text,
            "qwen_non_streaming_mode": bool(incoming.get("qwen_non_streaming_mode", False)),
            "qwen_append_silence": bool(incoming.get("qwen_append_silence", True)),
            "qwen_instruct": "",
            "qwen_min_new_tokens": _safe_int(
                incoming.get("qwen_min_new_tokens"), default=2, minimum=2, maximum=256
            ),
            "aac_bitrate": _safe_aac_bitrate(incoming.get("aac_bitrate"), default="80k"),
            "sample_rate": 48000,
            "channels": 1,
        }

    @app.get("/api/document-projects")
    async def list_document_projects() -> JSONResponse:
        return JSONResponse({"projects": document_projects.list_projects()})

    @app.get("/api/document-projects/{project_id}")
    async def get_document_project(project_id: str) -> JSONResponse:
        try:
            return JSONResponse(document_projects.get_project(project_id))
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="project not found") from exc

    @app.post("/api/document-projects")
    async def create_document_project(
        document: UploadFile = File(...),
        name: str = Form(""),
        settings_json: str = Form("{}"),
        max_chars: int = Form(200),
    ) -> JSONResponse:
        try:
            project = document_projects.create_project(
                name=name,
                filename=document.filename or "document.txt",
                data=await document.read(),
                settings=document_settings(settings_json),
                max_chars=_safe_int(max_chars, default=200, minimum=40, maximum=500),
            )
            return JSONResponse(project)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/document-projects/{project_id}/append")
    async def append_document_project(project_id: str, document: UploadFile = File(...)) -> JSONResponse:
        try:
            return JSONResponse(
                document_projects.append_document(
                    project_id, filename=document.filename or "document.txt", data=await document.read()
                )
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="project not found") from exc
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/document-projects/{project_id}/start")
    async def start_document_project(project_id: str) -> JSONResponse:
        try:
            # A project is immutable with respect to voice/model parameters.
            # Resume always uses its saved settings, never the page's current controls.
            return JSONResponse(document_projects.start(project_id, settings=None))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="project not found") from exc
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/document-projects/{project_id}/start-selection")
    async def start_document_project_selection(
        project_id: str,
        segment_start: int = Form(0),
        segment_end: int = Form(0),
        settings_json: str = Form(""),
    ) -> JSONResponse:
        try:
            project = document_projects.get_project(project_id)
            maximum = max(0, len(project.get("segments", [])) - 1)
            start_index = _safe_int(segment_start, default=0, minimum=0, maximum=maximum)
            end_index = _safe_int(
                segment_end,
                default=start_index,
                minimum=start_index,
                maximum=maximum,
            )
            return JSONResponse(
                document_projects.start(
                    project_id,
                    settings=document_settings(settings_json) if settings_json.strip() else None,
                    segment_indices=list(range(start_index, end_index + 1)),
                )
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="project not found") from exc
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/document-projects/{project_id}/stop")
    async def stop_document_project(project_id: str) -> JSONResponse:
        try:
            return JSONResponse(document_projects.stop(project_id))
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="project not found") from exc

    @app.delete("/api/document-projects/{project_id}")
    async def delete_document_project(project_id: str) -> JSONResponse:
        try:
            document_projects.delete(project_id)
            return JSONResponse({"ok": True})
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="project not found") from exc
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/document-projects/{project_id}/playback")
    async def save_document_playback(
        project_id: str,
        segment_index: int = Form(0),
        offset_seconds: float = Form(0.0),
    ) -> JSONResponse:
        try:
            return JSONResponse(
                document_projects.update_playback(
                    project_id, segment_index=segment_index, offset_seconds=offset_seconds
                )
            )
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="project not found") from exc

    @app.put("/api/document-projects/{project_id}/segments/{segment_index}")
    async def update_document_segment(
        project_id: str,
        segment_index: int,
        payload: dict[str, Any] = Body(...),
    ) -> JSONResponse:
        try:
            return JSONResponse(
                document_projects.update_segment_text(
                    project_id,
                    segment_index=segment_index,
                    text=str(payload.get("text") or ""),
                )
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="project not found") from exc
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/document-projects/{project_id}/media")
    async def document_project_media(project_id: str, path: str) -> FileResponse:
        try:
            media_path = document_projects.media_path(project_id, _decode_reference_path(path))
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="media not found") from exc
        media_type = "audio/mp4" if media_path.suffix.lower() == ".m4a" else "application/octet-stream"
        return FileResponse(str(media_path), media_type=media_type, filename=media_path.name)

    return app


def _login_html(*, next_path: str, error: str) -> str:
    safe_next = html_lib.escape(next_path, quote=True)
    safe_error = html_lib.escape(error)
    error_block = f'<div class="error">{safe_error}</div>' if safe_error else ""
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Qwen3-TTS 服务登录</title><style>
body{{margin:0;background:#f3f4f6;font-family:Inter,"Microsoft YaHei",sans-serif;color:#171717;display:grid;place-items:center;min-height:100vh}}
.card{{width:min(420px,calc(100vw - 40px));background:#fff;border:1px solid #ddd;border-radius:12px;padding:28px;box-shadow:0 12px 35px #00000012}}
h1{{font-size:22px;margin:0 0 8px}}p{{color:#666;margin:0 0 22px}}label{{display:block;font-weight:700;margin-bottom:8px}}
input{{width:100%;box-sizing:border-box;padding:11px;border:1px solid #bbb;border-radius:7px;font-size:16px}}
button{{width:100%;margin-top:16px;padding:11px;border:0;border-radius:7px;background:#166534;color:#fff;font-weight:700;font-size:15px;cursor:pointer}}
.error{{color:#b91c1c;background:#fef2f2;padding:9px;border-radius:6px;margin-bottom:14px}}
</style></head><body><form class="card" method="post" action="/login">
<h1>Qwen3-TTS 服务登录</h1><p>输入服务密码后可查看和管理所有生成任务。</p>{error_block}
<input type="hidden" name="next_path" value="{safe_next}"><label for="password">服务密码</label>
<input id="password" name="password" type="password" autocomplete="current-password" required autofocus>
<button type="submit">登录</button></form></body></html>"""


def _html(*, defaults: dict[str, Any], examples: list[dict[str, str]], voices: list[dict[str, str]], languages: list[str], runtime: dict[str, Any]) -> str:
    replacements = {
        "__DEFAULT_TEXT__": json.dumps(defaults["text"], ensure_ascii=False),
        "__DEFAULT_MAX_NEW_TOKENS__": str(defaults["max_new_tokens"]),
        "__DEFAULT_SEED__": str(defaults["seed"]),
        "__EXAMPLES_JSON__": json.dumps(examples, ensure_ascii=False),
        "__VOICES_JSON__": json.dumps(voices, ensure_ascii=False),
        "__LANGUAGES_JSON__": json.dumps(languages, ensure_ascii=False),
        "__RUNTIME_JSON__": json.dumps(runtime, ensure_ascii=False),
    }
    html = INDEX_HTML
    for key, value in replacements.items():
        html = html.replace(key, value)
    return html


INDEX_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Qwen3-TTS Apple Silicon</title>
  <style>
    :root {
      --bg: #f6f7f8;
      --panel: #ffffff;
      --ink: #111418;
      --muted: #4d5562;
      --line: #e5e7eb;
      --accent: #0f766e;
      --orange: #f97316;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: linear-gradient(180deg, #f7f8fa 0%, #f3f5f7 100%);
      color: var(--ink);
      font: 14px/1.45 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .page { max-width: 1840px; margin: 0 auto; padding: 22px 58px 48px; }
    .app-card {
      border: 1px solid var(--line);
      border-radius: 16px;
      background: var(--panel);
      padding: 14px;
      margin-bottom: 16px;
    }
    .app-title { font-size: 22px; font-weight: 700; margin-bottom: 6px; letter-spacing: 0.2px; }
    .app-subtitle { color: var(--muted); font-size: 14px; }
    .app-header-row { display: flex; justify-content: space-between; gap: 12px; align-items: flex-start; }
    .app-header-actions { display: flex; align-items: center; gap: 8px; }
    .reader-link { display: inline-flex; align-items: center; min-height: 31px; padding: 0 12px; border: 1px solid #d8d4f3; border-radius: 7px; background: #f1efff; color: #5143c7; font-size: 12px; font-weight: 700; text-decoration: none; }
    .reader-link:hover { background: #e8e4ff; }
    .model-profile-row { display: grid; grid-template-columns: 150px minmax(260px, 420px); gap: 10px; align-items: center; margin-top: 12px; }
    .model-profile-row select { min-width: 0; }
    .logout-button { padding: 7px 11px; background: #f3f4f6; color: var(--muted); font-size: 12px; }
    .service-task-center { border: 1px solid var(--line); background: var(--panel); border-radius: 8px; padding: 12px; margin-bottom: 14px; }
    .service-task-header { display: flex; justify-content: space-between; align-items: center; gap: 12px; margin-bottom: 9px; }
    .service-task-title { font-weight: 700; font-size: 15px; }
    .service-task-list { max-height: 230px; overflow-y: auto; border: 1px solid var(--line); border-radius: 5px; }
    .service-task-item { display: grid; grid-template-columns: 86px minmax(180px,1fr) 180px 160px 72px; gap: 10px; align-items: center; padding: 9px 10px; border-bottom: 1px solid #eef0f3; }
    .service-task-item:last-child { border-bottom: 0; }
    .service-task-item.active { background: #ecfdf5; }
    .service-task-kind { color: var(--muted); font-size: 12px; }
    .service-task-name { overflow: hidden; white-space: nowrap; text-overflow: ellipsis; font-weight: 600; }
    .service-task-meta { color: var(--muted); font-size: 12px; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }
    .service-task-open { padding: 6px 9px; background: #e5e7eb; color: var(--ink); font-size: 12px; }
    .service-task-empty { color: var(--muted); padding: 18px; text-align: center; }
    .layout { display: grid; grid-template-columns: minmax(0, 3fr) minmax(360px, 2fr); gap: 16px; align-items: start; }
    .stack { display: flex; flex-direction: column; gap: 16px; }
    .panel {
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 4px;
      padding: 12px;
    }
    label { display: block; color: var(--muted); font-size: 13px; margin-bottom: 8px; }
    textarea, select, input[type="number"], input[type="text"] {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 4px;
      background: #fff;
      color: var(--ink);
      font: inherit;
      padding: 10px 12px;
    }
    textarea { min-height: 190px; resize: vertical; }
    .small-textarea { min-height: 76px; }
    .hint { color: var(--muted); font-size: 12px; margin-top: -3px; }
    .drop-zone {
      border: 1px solid var(--line);
      border-radius: 4px;
      min-height: 158px;
      display: grid;
      place-items: center;
      color: #6b7280;
      background: #fff;
      position: relative;
      overflow: hidden;
    }
    .drop-zone input { position: absolute; inset: 0; opacity: 0; cursor: pointer; z-index: 1; }
    .drop-zone.hidden { display: none; }
    .drop-zone.has-reference { min-height: 0; display: block; padding: 0; overflow: visible; }
    .drop-zone.has-reference input { display: none; }
    .drop-zone.has-reference .drop-copy { display: none; }
    .reference-preview { display: block; width: 100%; position: relative; z-index: 2; pointer-events: auto; }
    audio[disabled] { opacity: 0.55; pointer-events: none; }
    .drop-copy { text-align: center; pointer-events: none; }
    .selected-reference { margin-top: 8px; color: var(--muted); font-size: 12px; overflow-wrap: anywhere; text-align: center; }
    .reference-action-row { display: flex; justify-content: center; align-items: center; gap: 12px; margin-top: 8px; flex-wrap: wrap; }
    .reference-source-row { display: flex; justify-content: center; margin-top: 0; }
    .reference-source-toggle { display: inline-flex; gap: 4px; border: 1px solid var(--line); border-radius: 8px; padding: 4px; background: #fff; }
    .reference-source-button { display: inline-flex; align-items: center; gap: 7px; border-radius: 6px; padding: 8px 12px; background: transparent; color: var(--muted); }
    .reference-source-button.active { background: var(--accent); color: #fff; }
    .reference-source-button svg { width: 18px; height: 18px; stroke: currentColor; fill: none; stroke-width: 2; stroke-linecap: round; stroke-linejoin: round; }
    .reference-record-controls { display: flex; justify-content: center; align-items: center; gap: 10px; margin-top: 12px; flex-wrap: wrap; }
    .reference-record-controls.hidden { display: none; }
    .record-button { display: inline-flex; align-items: center; gap: 8px; background: var(--accent); color: #fff; }
    .record-button.recording { background: #fee2e2; color: #b91c1c; }
    .record-dot { width: 10px; height: 10px; border-radius: 999px; background: currentColor; }
    .record-status { color: var(--muted); font-size: 12px; }
    .radio-row { display: flex; gap: 8px; flex-wrap: wrap; }
    .radio-pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      border: 1px solid var(--line);
      border-radius: 4px;
      padding: 8px 12px;
      cursor: pointer;
      background: #fff;
    }
    .radio-pill input { accent-color: var(--orange); }
    .mode-hint { margin-top: 12px; color: var(--ink); }
    .accordion {
      border: 1px solid var(--line);
      border-radius: 4px;
      background: #fff;
      padding: 0;
    }
    .accordion summary {
      cursor: pointer;
      list-style: none;
      padding: 12px;
      color: var(--ink);
      border-bottom: 1px solid var(--line);
    }
    .accordion summary::-webkit-details-marker { display: none; }
    .accordion summary::after { content: "▾"; float: right; }
    .accordion[open] summary::after { content: "▴"; }
    .accordion-body { padding: 12px; display: grid; gap: 14px; }
    .control-row { display: grid; grid-template-columns: 1fr 92px; gap: 12px; align-items: center; }
    .control-row input[type="range"] { width: 100%; accent-color: var(--orange); }
    .field-block {
      display: grid;
      gap: 8px;
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: #fafafa;
    }
    .field-block > label { margin: 0; color: var(--ink); font-weight: 700; }
    .field-heading { display: flex; align-items: center; justify-content: space-between; gap: 10px; flex-wrap: wrap; }
    .field-heading > label { margin: 0; color: var(--ink); font-weight: 700; }
    .icl-status {
      display: inline-flex;
      align-items: center;
      min-height: 24px;
      padding: 3px 8px;
      border-radius: 999px;
      background: #ecfdf5;
      color: #047857;
      font-size: 11px;
      font-weight: 700;
    }
    .icl-status.needs-text { background: #fff7ed; color: #c2410c; }
    .icl-transcript { min-height: 92px; line-height: 1.65; }
    .field-footer { display: flex; align-items: center; justify-content: space-between; gap: 10px; flex-wrap: wrap; }
    .field-footer .hint { margin: 0; }
    .text-button { padding: 6px 9px; background: #eef2f7; color: var(--muted); font-size: 12px; }
    .qwen-toggle-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }
    .toggle-card {
      display: flex;
      align-items: flex-start;
      gap: 9px;
      min-height: 48px;
      margin: 0;
      padding: 11px 12px;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: #fafafa;
      color: var(--ink);
      cursor: pointer;
    }
    .toggle-card input { flex: 0 0 auto; margin-top: 3px; accent-color: var(--accent); }
    .compact-field-row { display: grid; grid-template-columns: minmax(0, 1fr) 160px; gap: 12px; align-items: center; }
    .compact-field-row label { margin: 0; }
    .range-label { color: var(--muted); font-size: 13px; margin-bottom: 4px; }
    .range-minmax { display: flex; justify-content: space-between; color: #9ca3af; font-size: 11px; margin-top: 2px; }
    .button-row { display: grid; grid-template-columns: 1fr 150px 120px; gap: 10px; }
    button {
      border: none;
      border-radius: 4px;
      padding: 12px 14px;
      font-weight: 700;
      cursor: pointer;
    }
    .primary { background: var(--accent); color: #fff; }
    .secondary { background: #e5e7eb; color: var(--ink); }
    .small-button { padding: 7px 10px; font-size: 12px; font-weight: 600; }
    button:disabled { opacity: 0.55; cursor: not-allowed; }
    audio { width: 100%; }
    .audio-panel { min-height: 118px; display: flex; flex-direction: column; gap: 8px; justify-content: center; }
    .status-box {
      min-height: 110px;
      max-height: 260px;
      overflow: auto;
      white-space: pre-wrap;
      border: 1px solid var(--line);
      border-radius: 4px;
      padding: 10px;
      background: #fff;
      color: var(--ink);
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
    }
    .summary { color: var(--muted); margin-bottom: 8px; }
    .preset-controls { display: grid; grid-template-columns: minmax(0, 1fr) auto auto; gap: 8px; align-items: center; }
    .preset-save-row { display: grid; grid-template-columns: minmax(0, 1fr) auto auto; gap: 8px; margin-top: 8px; }
    .preset-status { min-height: 18px; margin-top: 8px; color: var(--muted); font-size: 12px; }
    .meter { height: 7px; background: #e5e7eb; border-radius: 999px; overflow: hidden; margin-bottom: 8px; }
    .meter > div { height: 100%; width: 0%; background: var(--orange); transition: width 0.2s ease; }
    .clone-voice-tabs { display: flex; gap: 6px; margin-bottom: 10px; }
    .clone-voice-tab { flex: 1; padding: 8px 12px; background: #f3f4f6; color: var(--muted); }
    .clone-voice-tab.active { background: var(--accent); color: #fff; }
    .clone-voices-wrap { height: 360px; overflow-y: auto; border: 1px solid var(--line); border-radius: 4px; background: #fff; }
    .clone-voice-item { display: flex; align-items: center; gap: 12px; padding: 10px 12px; border-bottom: 1px solid #eef0f3; cursor: pointer; }
    .clone-voice-item:last-child { border-bottom: 0; }
    .clone-voice-item:hover { background: #f8fafc; }
    .clone-voice-item.active { background: #ecfdf5; box-shadow: inset 3px 0 0 var(--accent); }
    .clone-voice-info { min-width: 0; flex: 1; }
    .clone-voice-name { color: var(--ink); font-weight: 700; }
    .clone-voice-description { color: var(--muted); font-size: 12px; margin-top: 2px; }
    .clone-voice-meta { color: #047857; font-size: 11px; margin-top: 3px; }
    .clone-voice-controls { display: flex; align-items: center; gap: 6px; flex: 0 0 auto; }
    .clone-voice-action { padding: 7px 10px; background: #fff; color: var(--muted); border: 1px solid var(--line); font-size: 12px; }
    .clone-voice-action:hover { background: #f3f4f6; color: var(--ink); }
    .clone-voice-preview { flex: 0 0 auto; padding: 7px 12px; background: #e5e7eb; color: var(--ink); font-size: 12px; }
    .clone-voice-preview:hover { background: #d1d5db; }
    .clone-voices-empty { display: grid; place-items: center; height: 100%; color: var(--muted); }
    .workspace-tabs { display: flex; gap: 8px; margin: 16px 0; padding: 5px; border: 1px solid var(--line); border-radius: 7px; background: #f3f4f6; }
    .workspace-tab { flex: 1; padding: 11px 16px; background: transparent; color: var(--muted); }
    .workspace-tab:hover { background: #e5e7eb; color: var(--ink); }
    .workspace-tab.active { background: var(--accent); color: #fff; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.12); }
    .workspace-tab.active:hover { background: var(--accent); color: #fff; }
    .document-workspace { margin-top: 0; }
    .document-toolbar { display: grid; grid-template-columns: minmax(240px, 1fr) minmax(260px, 1fr); gap: 12px; }
    .document-drop-zone { position: relative; min-height: 96px; border: 1px dashed #9ca3af; border-radius: 6px; display: grid; place-items: center; text-align: center; color: var(--muted); background: #fafafa; padding: 14px; }
    .document-drop-zone input { position: absolute; inset: 0; width: 100%; height: 100%; opacity: 0; cursor: pointer; }
    .document-project-controls { display: grid; gap: 9px; align-content: start; }
    .document-inline { display: grid; grid-template-columns: 1fr auto auto; gap: 8px; align-items: center; }
    .document-actions { display: flex; gap: 8px; flex-wrap: wrap; }
    .document-progress { height: 12px; background: #e5e7eb; border-radius: 999px; overflow: hidden; margin: 12px 0 8px; }
    .document-progress > div { height: 100%; width: 0%; background: var(--accent); transition: width 0.3s ease; }
    .document-stats { color: var(--muted); font-size: 12px; min-height: 20px; }
    .document-body { display: grid; grid-template-columns: minmax(0, 3fr) minmax(300px, 2fr); gap: 12px; margin-top: 12px; }
    .document-segments { height: 340px; overflow-y: auto; border: 1px solid var(--line); border-radius: 4px; }
    .document-segment { display: grid; grid-template-columns: 64px 1fr 90px 74px; gap: 10px; align-items: center; padding: 9px 10px; border-bottom: 1px solid #eef0f3; }
    .document-segment:last-child { border-bottom: 0; }
    .document-segment.playable { cursor: pointer; }
    .document-segment.playable:hover { background: #f8fafc; }
    .document-segment.playable:focus-visible { outline: 2px solid var(--accent); outline-offset: -2px; }
    .document-segment.playing { background: #ecfdf5; }
    .document-segment.playing:hover { background: #ecfdf5; }
    .document-segment-text { overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }
    .document-segment-status { color: var(--muted); font-size: 12px; }
    .document-segment-play { color: var(--accent); font-size: 12px; font-weight: 700; text-align: right; }
    .document-project-status { min-height: 90px; max-height: 180px; overflow: auto; white-space: pre-wrap; border: 1px solid var(--line); border-radius: 4px; padding: 10px; background: #fff; font-size: 12px; }
    .document-player-wrap { display: grid; gap: 10px; align-content: start; }
    table { width: 100%; border-collapse: collapse; background: #fff; font-size: 13px; }
    th, td { border-bottom: 1px solid #eef0f3; padding: 10px; text-align: left; vertical-align: top; }
    th { position: sticky; top: 0; background: #fff; z-index: 1; font-weight: 700; }
    tr { cursor: pointer; }
    tr:hover td { background: #f8fafc; }
    .role-cell { width: 160px; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    .download { display: none; margin-top: 8px; color: var(--accent); font-weight: 700; text-decoration: none; }
    .hidden { display: none; }
    @media (max-width: 1100px) {
      .page { padding: 16px; }
      .layout { grid-template-columns: 1fr; }
      .button-row { grid-template-columns: 1fr; }
      .document-toolbar, .document-body { grid-template-columns: 1fr; }
      .document-inline { grid-template-columns: 1fr; }
      .service-task-item { grid-template-columns: 72px minmax(0,1fr) 72px; }
      .service-task-item .service-task-meta:nth-of-type(n+2) { display: none; }
    }
    @media (max-width: 640px) {
      .model-profile-row { grid-template-columns: 1fr; }
      .control-row, .compact-field-row, .qwen-toggle-grid { grid-template-columns: 1fr; }
      .control-row { gap: 7px; }
      .control-row > input[type="number"] { max-width: none; }
      .field-footer { align-items: stretch; }
      .field-footer .text-button { width: 100%; }
    }
  </style>
</head>
<body>
  <div class="page">
    <div class="app-card">
      <div class="app-header-row"><div><div class="app-title">Qwen3-TTS · Apple Silicon Metal</div>
      <div class="app-subtitle">后台服务持续执行任务；关闭页面后可从其他设备登录并查看进度</div></div>
      <div class="app-header-actions"><a class="reader-link" href="/reader">打开小说阅读器</a>
      <form method="post" action="/logout"><button class="logout-button" type="submit">退出登录</button></form></div></div>
      <div class="model-profile-row">
        <label for="model-profile" style="margin:0;font-weight:700;">生成模型</label>
        <select id="model-profile">
          <optgroup label="Qwen3-TTS / GGML Metal">
            <option value="qwen_0_6b" selected>Qwen3-TTS 0.6B（Metal 极速克隆）</option>
            <option value="qwen_1_7b">Qwen3-TTS 1.7B（Metal 高质量克隆）</option>
          </optgroup>
        </select>
      </div>
      <div id="model-profile-hint" class="hint" style="margin-top:6px;">Qwen 0.6B：GGML Metal 音色克隆。新建文件项目会锁定当前模型和参数。</div>
    </div>

    <div class="service-task-center">
      <div class="service-task-header"><div><span class="service-task-title">服务任务中心</span> <span id="service-active-count" class="hint">正在连接…</span></div>
      <button id="service-task-refresh" class="secondary small-button" type="button">刷新</button></div>
      <div id="service-task-list" class="service-task-list"><div class="service-task-empty">正在读取后台任务…</div></div>
    </div>

    <div class="workspace-tabs" role="tablist" aria-label="工作模式">
      <button id="workspace-tab-text" class="workspace-tab active" type="button" role="tab" aria-controls="workspace-text" aria-selected="true">本地文本测试</button>
      <button id="workspace-tab-document" class="workspace-tab" type="button" role="tab" aria-controls="workspace-document" aria-selected="false">文件生成</button>
    </div>

    <div id="workspace-text" class="workspace-panel" role="tabpanel" aria-labelledby="workspace-tab-text">
      <div class="layout">
      <div class="stack">
        <div class="panel">
          <label for="text">Text</label>
          <textarea id="text" placeholder="Enter text to synthesize or continue after the reference audio."></textarea>
        </div>

        <div class="panel">
          <label for="preset-select">语音预设</label>
          <div class="preset-controls">
            <select id="preset-select"><option value="">选择已保存预设…</option></select>
            <button id="preset-apply" class="secondary small-button" type="button">应用</button>
            <button id="preset-delete" class="secondary small-button" type="button" disabled>删除</button>
          </div>
          <div class="preset-save-row">
            <input id="preset-name" type="text" maxlength="120" placeholder="预设名称，例如：旁白·龙嫱·0.6B">
            <button id="preset-save" class="primary small-button" type="button">保存为新预设</button>
            <button id="preset-update" class="secondary small-button" type="button" disabled>更新选中预设</button>
          </div>
          <div id="preset-status" class="preset-status">预设会保存模型、参考音频、种子和全部 Qwen 生成参数；保存自定义音频时会一并导入本地服务。</div>
        </div>

        <div class="panel">
          <label>Reference Audio (Optional)</label>
          <label for="clone-voice" style="color: var(--ink); font-weight: 700;">本地克隆音色</label>
          <select id="clone-voice" style="margin-bottom: 10px;"></select>
          <div class="hint" style="margin-bottom: 10px;">选择后可先试听，生成时将作为 Clone 参考音频。</div>
          <div id="reference-drop-zone" class="drop-zone">
            <input id="prompt-audio" type="file" accept="audio/*,.wav,.mp3,.flac,.m4a,.ogg,.opus,.aac">
            <div class="drop-copy">Drop audio here<br>or<br>click to upload</div>
            <audio id="reference-audio-preview" class="reference-preview hidden" controls></audio>
          </div>
          <input id="example-audio-path" type="hidden" value="">
          <div id="reference-record-controls" class="reference-record-controls hidden">
            <button id="reference-record-button" class="record-button" type="button"><span class="record-dot"></span><span id="reference-record-button-label">Start Recording</span></button>
            <span id="reference-record-status" class="record-status">Ready to record.</span>
          </div>
          <div class="reference-action-row">
            <div class="reference-source-row">
              <div class="reference-source-toggle" role="group" aria-label="Reference audio source">
                <button id="reference-source-upload" class="reference-source-button active" type="button" aria-pressed="true" title="Upload">
                  <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 16V4"></path><path d="M7 9l5-5 5 5"></path><path d="M5 20h14"></path></svg>
                  <span>Upload</span>
                </button>
                <button id="reference-source-record" class="reference-source-button" type="button" aria-pressed="false" title="Record">
                  <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 14a3 3 0 0 0 3-3V6a3 3 0 0 0-6 0v5a3 3 0 0 0 3 3z"></path><path d="M19 11a7 7 0 0 1-14 0"></path><path d="M12 18v3"></path><path d="M8 21h8"></path></svg>
                  <span>Record</span>
                </button>
              </div>
            </div>
            <button id="clear-reference" class="secondary small-button" type="button">Clear Reference Audio</button>
          </div>
          <div id="selected-reference" class="selected-reference">No reference selected.</div>
        </div>

        <input class="hidden" type="radio" name="mode" value="voice_clone" checked>
        <select id="language" class="hidden"><option value="Chinese" selected>Chinese</option></select>
        <div id="mode-hint" class="hidden"></div>
        <div id="reference-transcript-panel" class="hidden"><textarea id="prompt-text"></textarea></div>

        <div id="moss-duration-panel" class="panel hidden">
          <label><input id="tokens-control" type="checkbox"> Enable Duration Control (Expected Audio Tokens)</label>
          <div id="tokens-wrap" class="hidden" style="margin-top: 10px;">
            <label for="tokens">expected_tokens</label>
            <input id="tokens" type="number" min="1" step="1" value="1">
          </div>
        </div>
        <div id="duration-hint" class="hint hidden">Duration control is disabled.</div>

        <details id="moss-params-panel" class="accordion hidden" open>
          <summary>MOSS Sampling Parameters</summary>
          <div class="accordion-body">
            <div class="control-row" data-pair="temperature">
              <div>
                <div class="range-label">temperature</div>
                <input id="temperature-range" type="range" min="0.1" max="3" step="0.05" value="1.7">
                <div class="range-minmax"><span>0.1</span><span>3</span></div>
              </div>
              <input id="temperature" type="number" min="0.1" max="3" step="0.05" value="1.7">
            </div>
            <div class="control-row" data-pair="top-p">
              <div>
                <div class="range-label">top_p</div>
                <input id="top-p-range" type="range" min="0.1" max="1" step="0.01" value="0.8">
                <div class="range-minmax"><span>0.1</span><span>1</span></div>
              </div>
              <input id="top-p" type="number" min="0.1" max="1" step="0.01" value="0.8">
            </div>
            <div class="control-row" data-pair="top-k">
              <div>
                <div class="range-label">top_k</div>
                <input id="top-k-range" type="range" min="1" max="200" step="1" value="25">
                <div class="range-minmax"><span>1</span><span>200</span></div>
              </div>
              <input id="top-k" type="number" min="1" max="200" step="1" value="25">
            </div>
            <div class="control-row" data-pair="repetition-penalty">
              <div>
                <div class="range-label">repetition_penalty</div>
                <input id="repetition-penalty-range" type="range" min="0.8" max="2" step="0.05" value="1.0">
                <div class="range-minmax"><span>0.8</span><span>2</span></div>
              </div>
              <input id="repetition-penalty" type="number" min="0.8" max="2" step="0.05" value="1.0">
            </div>
            <div class="control-row" data-pair="max-new-tokens">
              <div>
                <div class="range-label">max_new_tokens</div>
                <input id="max-new-tokens-range" type="range" min="1" max="7500" step="1" value="__DEFAULT_MAX_NEW_TOKENS__">
                <div class="range-minmax"><span>1</span><span>7500</span></div>
              </div>
              <input id="max-new-tokens" type="number" min="1" max="7500" step="1" value="__DEFAULT_MAX_NEW_TOKENS__">
            </div>
            <div class="control-row" data-pair="codec-chunk-frames">
              <div>
                <div class="range-label">Codec Chunk Frames (0=auto)</div>
                <input id="codec-chunk-frames-range" type="range" min="0" max="32" step="1" value="16">
                <div class="range-minmax"><span>0</span><span>32</span></div>
              </div>
              <input id="codec-chunk-frames" type="number" min="0" max="32" step="1" value="16">
            </div>
            <div class="control-row">
              <label for="initial-playback-delay">流式播放最小预缓冲（秒，自动调整）</label>
              <input id="initial-playback-delay" type="number" min="0.3" max="60" step="0.1" value="1.5">
            </div>
            <div class="control-row" data-pair="seed">
              <div>
                <div class="range-label">seed (-1=random)</div>
                <input id="seed-range" type="range" min="-1" max="999999" step="1" value="__DEFAULT_SEED__">
                <div class="range-minmax"><span>-1</span><span>999999</span></div>
              </div>
              <input id="seed" type="number" min="-1" step="1" value="__DEFAULT_SEED__">
            </div>
            <label style="margin-top: 14px;"><input id="streaming-generation" type="checkbox" checked> Enable Streaming Generation</label>
          </div>
        </details>

        <details id="qwen-params-panel" class="accordion" open>
          <summary>Qwen Sampling Parameters</summary>
          <div class="accordion-body">
            <div class="field-block">
              <label for="qwen-clone-mode">克隆方式</label>
              <select id="qwen-clone-mode">
                <option value="xvec" selected>X-vector（无需参考文字，速度与跨语言稳定）</option>
                <option value="icl">ICL（相似度更高，需要准确参考文字）</option>
              </select>
              <div class="hint">ICL 会使用所选音色的预置台词；上传或录制自定义音频时需要手动填写。</div>
            </div>
            <div id="qwen-reference-text-wrap" class="field-block hidden">
              <div class="field-heading">
                <label for="qwen-reference-text">参考音频准确文字</label>
                <span id="qwen-transcript-status" class="icl-status needs-text">需要填写</span>
              </div>
              <textarea id="qwen-reference-text" class="icl-transcript" placeholder="必须与参考音频逐字对应"></textarea>
              <div class="field-footer">
                <span id="qwen-transcript-hint" class="hint">选择本地音色后会自动填入预置台词。</span>
                <button id="qwen-transcript-reset" class="text-button" type="button">恢复预置台词</button>
              </div>
            </div>
            <input id="qwen-instruct" type="hidden" value="">
            <div class="control-row" data-pair="qwen-temperature">
              <div>
                <div class="range-label">temperature</div>
                <input id="qwen-temperature-range" type="range" min="0.1" max="2" step="0.05" value="0.9">
                <div class="range-minmax"><span>0.1</span><span>2</span></div>
              </div>
              <input id="qwen-temperature" type="number" min="0.1" max="2" step="0.05" value="0.9">
            </div>
            <div class="control-row" data-pair="qwen-top-p">
              <div>
                <div class="range-label">top_p</div>
                <input id="qwen-top-p-range" type="range" min="0.1" max="1" step="0.01" value="1">
                <div class="range-minmax"><span>0.1</span><span>1</span></div>
              </div>
              <input id="qwen-top-p" type="number" min="0.1" max="1" step="0.01" value="1">
            </div>
            <div class="control-row" data-pair="qwen-top-k">
              <div>
                <div class="range-label">top_k</div>
                <input id="qwen-top-k-range" type="range" min="1" max="200" step="1" value="50">
                <div class="range-minmax"><span>1</span><span>200</span></div>
              </div>
              <input id="qwen-top-k" type="number" min="1" max="200" step="1" value="50">
            </div>
            <div class="control-row" data-pair="qwen-repetition-penalty">
              <div>
                <div class="range-label">repetition_penalty</div>
                <input id="qwen-repetition-penalty-range" type="range" min="0.8" max="2" step="0.01" value="1.05">
                <div class="range-minmax"><span>0.8</span><span>2</span></div>
              </div>
              <input id="qwen-repetition-penalty" type="number" min="0.8" max="2" step="0.01" value="1.05">
            </div>
            <div class="control-row" data-pair="qwen-max-new-tokens">
              <div>
                <div class="range-label">max_new_tokens</div>
                <input id="qwen-max-new-tokens-range" type="range" min="24" max="2048" step="1" value="2048">
                <div class="range-minmax"><span>24</span><span>2048</span></div>
              </div>
              <input id="qwen-max-new-tokens" type="number" min="24" max="2048" step="1" value="2048">
            </div>
            <div class="control-row" data-pair="qwen-chunk-size">
              <div>
                <div class="range-label">流式块大小（帧）</div>
                <input id="qwen-chunk-size-range" type="range" min="1" max="24" step="1" value="8">
                <div class="range-minmax"><span>1</span><span>24</span></div>
              </div>
              <input id="qwen-chunk-size" type="number" min="1" max="24" step="1" value="8">
            </div>
            <div class="control-row" data-pair="qwen-min-new-tokens">
              <div>
                <div class="range-label">min_new_tokens</div>
                <input id="qwen-min-new-tokens-range" type="range" min="2" max="256" step="1" value="2">
                <div class="range-minmax"><span>2</span><span>256</span></div>
              </div>
              <input id="qwen-min-new-tokens" type="number" min="2" max="256" step="1" value="2">
            </div>
            <div class="control-row" data-pair="qwen-seed">
              <div>
                <div class="range-label">seed (-1=random)</div>
                <input id="qwen-seed-range" type="range" min="-1" max="999999" step="1" value="1234">
                <div class="range-minmax"><span>-1</span><span>999999</span></div>
              </div>
              <input id="qwen-seed" type="number" min="-1" max="999999" step="1" value="1234">
            </div>
            <div class="qwen-toggle-grid">
              <label class="toggle-card"><input id="qwen-non-streaming-mode" type="checkbox"> <span>一次性输入完整文本（音频仍可流式输出）</span></label>
              <label class="toggle-card"><input id="qwen-append-silence" type="checkbox" checked> <span>参考音频尾部自动补静音，降低首音节污染</span></label>
            </div>
            <div class="compact-field-row">
              <label for="qwen-initial-playback-delay">流式播放最小预缓冲（秒，自动调整）</label>
              <input id="qwen-initial-playback-delay" type="number" min="0.1" max="20" step="0.1" value="0.8">
            </div>
            <label class="toggle-card"><input id="qwen-streaming-generation" type="checkbox" checked> <span>启用流式生成与边生成边试听</span></label>
          </div>
        </details>

        <div class="button-row">
          <button id="start" class="primary" type="button">Generate Speech</button>
          <button id="pause" class="secondary" type="button" disabled>Pause Playback</button>
          <button id="stop" class="secondary" type="button" disabled>停止当前生成</button>
        </div>
      </div>

      <div class="stack">
        <div class="panel">
          <label style="color: var(--ink); font-weight: 700;">本地克隆音色</label>
          <div class="clone-voice-tabs" role="tablist" aria-label="本地克隆音色分类">
            <button id="clone-tab-favorites" class="clone-voice-tab active" type="button" role="tab" aria-selected="true">收藏</button>
            <button id="clone-tab-hidden" class="clone-voice-tab" type="button" role="tab" aria-selected="false">隐藏</button>
          </div>
          <div id="clone-voices-list" class="clone-voices-wrap"></div>
        </div>
        <div class="panel">
          <label>Status</label>
          <div id="runtime-summary" class="summary">Runtime: checking...</div>
          <div id="summary" class="summary"></div>
          <div class="meter"><div id="bar"></div></div>
          <div id="status" class="status-box">idle</div>
        </div>
        <div class="panel audio-panel">
          <label>Output Audio</label>
          <audio id="audio-output" controls disabled></audio>
          <a id="download" class="download" href="#">Download final wav</a>
        </div>
      </div>
    </div>
    </div>

    <div id="workspace-document" class="workspace-panel hidden" role="tabpanel" aria-labelledby="workspace-tab-document">
    <div class="panel document-workspace">
      <label style="color: var(--ink); font-size: 16px; font-weight: 700;">文档转音频项目</label>
      <div class="hint" style="margin-bottom: 12px;">面向整本书优化：自动分段后使用两个 Metal 通道并行生成，并行转 AAC，最终严格按原文顺序合并；支持追加文档、暂停和断点继续。</div>
      <div class="document-toolbar">
        <div>
          <input id="document-project-name" type="text" placeholder="项目名称（留空则使用文档名）" style="margin-bottom: 8px;">
          <div id="document-create-drop" class="document-drop-zone">
            <input id="document-create-file" type="file" accept=".txt,.md,.markdown,.docx,text/plain,text/markdown,application/vnd.openxmlformats-officedocument.wordprocessingml.document">
            <div><b>拖入文档创建项目</b><br><span class="hint">单文件最大 30 MB，自动按自然句分段</span></div>
          </div>
        </div>
        <div class="document-project-controls">
          <div class="document-inline">
            <select id="document-project-select"><option value="">暂无项目</option></select>
            <button id="document-refresh" class="secondary small-button" type="button">刷新</button>
            <label class="secondary small-button" style="margin:0; cursor:pointer; position:relative; overflow:hidden;">追加文档<input id="document-append-file" type="file" accept=".txt,.md,.markdown,.docx" style="position:absolute; inset:0; opacity:0; cursor:pointer;"></label>
          </div>
          <div class="document-inline">
            <label for="document-max-chars" style="margin:0;">每段最大字符</label>
            <input id="document-max-chars" type="number" min="40" max="500" step="10" value="200" style="width:100px;">
            <span></span>
          </div>
          <div class="document-actions">
            <button id="document-start" class="primary small-button" type="button" disabled>开始 / 继续</button>
            <button id="document-stop" class="secondary small-button" type="button" disabled>停止</button>
            <button id="document-delete" class="secondary small-button" type="button" disabled>删除项目</button>
            <a id="document-final-download" class="secondary small-button hidden" href="#" download="complete.m4a" style="text-decoration:none;">下载完整 M4A</a>
          </div>
        </div>
      </div>
      <div class="document-progress"><div id="document-progress-fill"></div></div>
      <div id="document-stats" class="document-stats">尚未选择项目。</div>
      <div class="document-body">
        <div id="document-segments" class="document-segments"><div class="clone-voices-empty">项目段落将在这里显示</div></div>
        <div class="document-player-wrap">
          <audio id="document-audio" controls></audio>
          <div class="document-actions">
            <button id="document-play-continue" class="secondary small-button" type="button" disabled>从记录位置连续播放</button>
          </div>
          <div id="document-project-status" class="document-project-status">idle</div>
        </div>
      </div>
    </div>
    </div>
  </div>

<script>
const EXAMPLES = __EXAMPLES_JSON__;
const CLONE_VOICES = __VOICES_JSON__;
const LANGUAGES = __LANGUAGES_JSON__;
const INITIAL_RUNTIME = __RUNTIME_JSON__;
const DEFAULT_TEXT = __DEFAULT_TEXT__;
const CONTINUATION_NOTICE = "Continuation mode is active. Fill Reference Audio Transcript with the transcript of the reference audio.";
const HIDDEN_CLONE_VOICES_STORAGE_KEY = "qwen-tts-hidden-clone-voices-v1";
const ICL_TRANSCRIPT_OVERRIDES_STORAGE_KEY = "qwen-tts-icl-transcript-overrides-v1";
const UI_STATE_STORAGE_KEY = "qwen-tts-ui-state-v1";
const UI_OPTIMIZATION_DEFAULTS_VERSION = 4;
const PERSISTED_VALUE_FIELDS = [
  "model-profile",
  "temperature", "top-p", "top-k", "repetition-penalty", "max-new-tokens",
  "codec-chunk-frames", "seed", "initial-playback-delay", "tokens",
  "qwen-clone-mode", "qwen-reference-text", "qwen-instruct",
  "qwen-temperature", "qwen-top-p", "qwen-top-k", "qwen-repetition-penalty",
  "qwen-max-new-tokens", "qwen-chunk-size", "qwen-min-new-tokens",
  "qwen-seed", "qwen-initial-playback-delay",
  "document-max-chars", "document-project-name"
];

let activeWorkspaceTab = "text";
let activeCloneVoiceTab = "favorites";
let hiddenCloneVoicePaths = new Set();
let iclTranscriptOverrides = {};
let uiStateReady = false;
try {
  const savedHiddenVoices = JSON.parse(localStorage.getItem(HIDDEN_CLONE_VOICES_STORAGE_KEY) || "[]");
  const availablePaths = new Set(CLONE_VOICES.map((voice) => voice.audio_path));
  hiddenCloneVoicePaths = new Set(savedHiddenVoices.filter((path) => availablePaths.has(path)));
} catch (err) {
  hiddenCloneVoicePaths = new Set();
}
try {
  const savedOverrides = JSON.parse(localStorage.getItem(ICL_TRANSCRIPT_OVERRIDES_STORAGE_KEY) || "{}");
  const availablePaths = new Set(CLONE_VOICES.map((voice) => voice.audio_path));
  if (savedOverrides && typeof savedOverrides === "object") {
    iclTranscriptOverrides = Object.fromEntries(
      Object.entries(savedOverrides).filter(
        ([path, text]) => availablePaths.has(path) && typeof text === "string"
      )
    );
  }
} catch (err) {
  iclTranscriptOverrides = {};
}

let currentJob = null;
let currentJobOwned = false;
let audioContext = null;
let nextPlaybackTime = 0;
let statusTimer = null;
let runtimeReady = false;
let generationActive = false;
let currentStreamAbortController = null;
let currentStreamingGenerationEnabled = true;
let playbackPaused = false;
let playbackCompletionTimer = null;
let currentInitialPlaybackDelaySeconds = 1.5;
let adaptiveRealtimeBufferTargetSeconds = 1.5;
let estimatedRealtimeAudioSeconds = 0;
let latestRealtimeGenerationRate = 0;
const MAX_ADAPTIVE_PLAYBACK_BUFFER_SECONDS = 60;
const ADAPTIVE_PLAYBACK_BUFFER_SAFETY_SECONDS = 0.75;
let pendingRealtimePcmChunks = [];
let pendingRealtimePcmSeconds = 0;
let realtimePlaybackStarted = false;
let currentReferenceObjectUrl = null;
let referenceSourceMode = "upload";
let recordedReferenceFile = null;
let referenceRecordingActive = false;
let referenceRecordingStream = null;
let referenceRecordingAudioContext = null;
let referenceRecordingSource = null;
let referenceRecordingProcessor = null;
let referenceRecordingChunks = [];
let referenceRecordingSampleRate = 48000;
let referenceRecordingStartedAt = 0;
let referenceRecordingTimer = null;
let currentDocumentProject = null;
let documentProjectPollTimer = null;
let serviceTasksPollTimer = null;
let documentPlaybackIndex = null;
let documentPlaybackActive = false;
let lastPlaybackSaveAt = 0;
let voicePresets = [];
let activePresetVoiceName = "";
const DOCUMENT_PROJECT_SELECTION_KEY = "qwen-tts-current-document-project-v1";

function field(id) { return document.getElementById(id); }
function apiUrl(path) {
  const cleanPath = String(path || "").replace(/^\/+/, "");
  const pagePath = window.location.pathname.endsWith("/") ? window.location.pathname : window.location.pathname + "/";
  return new URL(cleanPath, window.location.origin + pagePath).toString();
}
function referenceAudioUrl(path) {
  const url = new URL(apiUrl("api/reference-audio"));
  url.searchParams.set("path", path);
  return url.toString();
}
function selectedMode() {
  const selected = document.querySelector("input[name='mode']:checked");
  return selected ? selected.value : "voice_clone";
}
function selectedModeName() {
  const mode = selectedMode();
  if (mode === "continuation") return "Continuation";
  if (mode === "continuation_clone") return "Continuation + Clone";
  return "Clone";
}
function hasReference() {
  return Boolean(field("prompt-audio").files[0] || recordedReferenceFile || field("example-audio-path").value);
}
function setStatus(obj) {
  field("status").textContent = typeof obj === "string" ? obj : JSON.stringify(obj, null, 2);
  if (obj && (obj.generated_frames || obj.generated_frames === 0) && obj.max_new_tokens) {
    const pct = Math.min(100, 100 * Number(obj.generated_frames || 0) / Number(obj.max_new_tokens || 1));
    field("bar").style.width = pct.toFixed(1) + "%";
  }
}
function formatSeed(value, seedMode = "") {
  if (value === null || value === undefined || value === "") {
    return seedMode === "random" ? "等待生成" : "—";
  }
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return String(value);
  if (numeric < 0) return seedMode === "random" ? "等待生成" : "旧任务未记录";
  return `${Math.trunc(numeric)}${seedMode === "random" ? "（随机）" : ""}`;
}
function fetchJson(url, options) {
  return fetch(url, options).then(async (response) => {
    const text = await response.text();
    if (!response.ok) throw new Error(text || `HTTP ${response.status}`);
    return text ? JSON.parse(text) : {};
  });
}
function renderRuntime(status) {
  runtimeReady = status && status.state === "ready";
  const elapsed = status && status.load_elapsed_seconds != null ? ` | load=${Number(status.load_elapsed_seconds).toFixed(1)}s` : "";
  const extra = runtimeReady ? ` | n_vq=${status.n_vq} | sr=${status.sample_rate}` : "";
  const parallel = status && status.generation_scheduler
    ? ` | GPU并行=${status.generation_scheduler.max_parallel} | 整书并行=${status.generation_scheduler.document_parallel || 1}`
    : "";
  const error = status && status.error ? ` | error=${status.error}` : "";
  field("runtime-summary").textContent = `Runtime: ${(status && status.state) || "unknown"}${elapsed}${extra}${parallel}${error}`;
}
async function pollRuntime() {
  try {
    renderRuntime(await fetch(apiUrl("api/health")).then(r => r.json()));
  } catch (err) {
    runtimeReady = false;
    field("runtime-summary").textContent = `Runtime: unreachable (${err})`;
  }
}
function updateReferenceLabel() {
  const file = field("prompt-audio").files[0];
  const examplePath = field("example-audio-path").value;
  if (file) {
    field("selected-reference").textContent = `Uploaded reference: ${file.name}`;
  } else if (recordedReferenceFile) {
    field("selected-reference").textContent = `Recorded reference: ${recordedReferenceFile.name}`;
  } else if (examplePath) {
    field("selected-reference").textContent = `Example reference: ${examplePath}`;
  } else {
    field("selected-reference").textContent = "No reference selected.";
  }
  syncReferenceSourceControls();
  updateModeHint();
}
function clearReferencePreview() {
  if (currentReferenceObjectUrl) {
    URL.revokeObjectURL(currentReferenceObjectUrl);
    currentReferenceObjectUrl = null;
  }
  const preview = field("reference-audio-preview");
  preview.pause();
  preview.removeAttribute("src");
  preview.load();
  preview.classList.add("hidden");
  field("reference-drop-zone").classList.remove("has-reference");
  syncReferenceSourceControls();
}
function showReferencePreview(src, objectUrl = null) {
  if (currentReferenceObjectUrl) {
    URL.revokeObjectURL(currentReferenceObjectUrl);
    currentReferenceObjectUrl = null;
  }
  currentReferenceObjectUrl = objectUrl;
  const preview = field("reference-audio-preview");
  preview.src = src;
  preview.classList.remove("hidden");
  field("reference-drop-zone").classList.remove("hidden");
  field("reference-drop-zone").classList.add("has-reference");
  preview.load();
}
function updateReferencePreview() {
  const file = field("prompt-audio").files[0];
  const examplePath = field("example-audio-path").value;
  if (file) {
    const objectUrl = URL.createObjectURL(file);
    showReferencePreview(objectUrl, objectUrl);
    return;
  }
  if (recordedReferenceFile) {
    const objectUrl = URL.createObjectURL(recordedReferenceFile);
    showReferencePreview(objectUrl, objectUrl);
    return;
  }
  if (examplePath) {
    showReferencePreview(referenceAudioUrl(examplePath));
    return;
  }
  clearReferencePreview();
}
function syncReferenceSourceControls() {
  const uploadMode = referenceSourceMode === "upload";
  field("reference-source-upload").classList.toggle("active", uploadMode);
  field("reference-source-record").classList.toggle("active", !uploadMode);
  field("reference-source-upload").setAttribute("aria-pressed", uploadMode ? "true" : "false");
  field("reference-source-record").setAttribute("aria-pressed", uploadMode ? "false" : "true");
  field("reference-record-controls").classList.toggle("hidden", uploadMode);
  field("reference-drop-zone").classList.toggle("hidden", !uploadMode && !hasReference());
}
function setReferenceSourceMode(mode) {
  const nextMode = mode === "record" ? "record" : "upload";
  if (referenceRecordingActive && nextMode !== "record") stopReferenceRecording(true);
  referenceSourceMode = nextMode;
  if (nextMode === "upload") {
    recordedReferenceFile = null;
    field("reference-record-status").textContent = "Ready to record.";
  } else {
    field("prompt-audio").value = "";
    field("example-audio-path").value = "";
    field("clone-voice").value = "";
    activePresetVoiceName = "";
    syncCloneVoiceListSelection();
    clearIclTranscriptForCustomReference();
  }
  updateReferencePreview();
  updateReferenceLabel();
}
function formatRecordingDuration(seconds) {
  const safeSeconds = Math.max(0, Math.floor(seconds || 0));
  const minutes = Math.floor(safeSeconds / 60);
  const rest = safeSeconds % 60;
  return `${minutes}:${String(rest).padStart(2, "0")}`;
}
function updateRecordButtonState() {
  const button = field("reference-record-button");
  button.classList.toggle("recording", referenceRecordingActive);
  field("reference-record-button-label").textContent = referenceRecordingActive ? "Stop Recording" : "Start Recording";
}
function updateRecordingStatus() {
  if (!referenceRecordingActive) return;
  const elapsed = (Date.now() - referenceRecordingStartedAt) / 1000;
  field("reference-record-status").textContent = `Recording ${formatRecordingDuration(elapsed)}`;
}
function flattenRecordingChunks(chunks) {
  const length = chunks.reduce((total, chunk) => total + chunk.length, 0);
  const merged = new Float32Array(length);
  let offset = 0;
  for (const chunk of chunks) {
    merged.set(chunk, offset);
    offset += chunk.length;
  }
  return merged;
}
function writeAscii(view, offset, value) {
  for (let index = 0; index < value.length; index += 1) {
    view.setUint8(offset + index, value.charCodeAt(index));
  }
}
function encodeWavPcm16(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  writeAscii(view, 0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeAscii(view, 8, "WAVE");
  writeAscii(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeAscii(view, 36, "data");
  view.setUint32(40, samples.length * 2, true);
  let offset = 44;
  for (const sample of samples) {
    const clamped = Math.max(-1, Math.min(1, sample));
    view.setInt16(offset, clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff, true);
    offset += 2;
  }
  return buffer;
}
function cleanupReferenceRecordingNodes() {
  if (referenceRecordingTimer) {
    window.clearInterval(referenceRecordingTimer);
    referenceRecordingTimer = null;
  }
  if (referenceRecordingProcessor) {
    try { referenceRecordingProcessor.disconnect(); } catch (err) {}
    referenceRecordingProcessor.onaudioprocess = null;
    referenceRecordingProcessor = null;
  }
  if (referenceRecordingSource) {
    try { referenceRecordingSource.disconnect(); } catch (err) {}
    referenceRecordingSource = null;
  }
  if (referenceRecordingStream) {
    for (const track of referenceRecordingStream.getTracks()) track.stop();
    referenceRecordingStream = null;
  }
  if (referenceRecordingAudioContext) {
    referenceRecordingAudioContext.close().catch(() => {});
    referenceRecordingAudioContext = null;
  }
}
async function startReferenceRecording() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error("Microphone recording is not supported in this browser.");
  }
  field("prompt-audio").value = "";
  field("example-audio-path").value = "";
  field("clone-voice").value = "";
  activePresetVoiceName = "";
  syncCloneVoiceListSelection();
  clearIclTranscriptForCustomReference();
  recordedReferenceFile = null;
  clearReferencePreview();
  referenceRecordingChunks = [];
  referenceRecordingStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const AudioContextCtor = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextCtor) throw new Error("Web Audio recording is not supported in this browser.");
  referenceRecordingAudioContext = new AudioContextCtor();
  referenceRecordingSampleRate = referenceRecordingAudioContext.sampleRate || 48000;
  referenceRecordingSource = referenceRecordingAudioContext.createMediaStreamSource(referenceRecordingStream);
  referenceRecordingProcessor = referenceRecordingAudioContext.createScriptProcessor(4096, 1, 1);
  referenceRecordingActive = true;
  referenceRecordingStartedAt = Date.now();
  referenceRecordingProcessor.onaudioprocess = (event) => {
    if (!referenceRecordingActive) return;
    const input = event.inputBuffer.getChannelData(0);
    referenceRecordingChunks.push(new Float32Array(input));
    event.outputBuffer.getChannelData(0).fill(0);
  };
  referenceRecordingSource.connect(referenceRecordingProcessor);
  referenceRecordingProcessor.connect(referenceRecordingAudioContext.destination);
  referenceRecordingTimer = window.setInterval(updateRecordingStatus, 200);
  updateRecordingStatus();
  updateRecordButtonState();
}
function stopReferenceRecording(discard = false) {
  if (!referenceRecordingActive) return;
  referenceRecordingActive = false;
  const chunks = referenceRecordingChunks;
  referenceRecordingChunks = [];
  cleanupReferenceRecordingNodes();
  updateRecordButtonState();
  if (discard) {
    field("reference-record-status").textContent = "Ready to record.";
    return;
  }
  if (!chunks.length) {
    field("reference-record-status").textContent = "No audio captured.";
    return;
  }
  const samples = flattenRecordingChunks(chunks);
  const wavBuffer = encodeWavPcm16(samples, referenceRecordingSampleRate);
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  recordedReferenceFile = new File([wavBuffer], `recorded_reference_${stamp}.wav`, { type: "audio/wav" });
  field("reference-record-status").textContent = `Recorded ${formatRecordingDuration(samples.length / referenceRecordingSampleRate)}`;
  updateReferencePreview();
  updateReferenceLabel();
}
async function toggleReferenceRecording() {
  if (referenceRecordingActive) {
    stopReferenceRecording(false);
    return;
  }
  try {
    await startReferenceRecording();
  } catch (err) {
    cleanupReferenceRecordingNodes();
    referenceRecordingActive = false;
    updateRecordButtonState();
    field("reference-record-status").textContent = String(err);
  }
}
function clearReferenceAudio() {
  if (referenceRecordingActive) stopReferenceRecording(true);
  recordedReferenceFile = null;
  field("prompt-audio").value = "";
  field("example-audio-path").value = "";
  field("clone-voice").value = "";
  syncCloneVoiceListSelection();
  clearReferencePreview();
  updateReferenceLabel();
  const visibleVoices = favoriteCloneVoices();
  const fallbackVoice = visibleVoices.find((voice) => voice.name === "龙嫱") || visibleVoices[0];
  if (fallbackVoice) selectCloneVoice(fallbackVoice.audio_path, false);
}
function serviceTaskStateLabel(state) {
  return ({queued:"排队", loading_runtime:"加载模型", running:"执行中", stopping:"停止中", paused:"已暂停", ready:"就绪", completed:"已完成", finished:"已完成", error:"错误", interrupted:"已中断", closed:"已停止"})[state] || state || "未知";
}
function renderServiceTasks(data) {
  const list = field("service-task-list");
  const tasks = Array.isArray(data.tasks) ? data.tasks : [];
  field("service-active-count").textContent = `运行中 ${Number(data.active_count || 0)} · 最近 ${tasks.length}`;
  list.innerHTML = "";
  if (!tasks.length) {
    list.innerHTML = '<div class="service-task-empty">暂时没有后台任务</div>';
    return;
  }
  for (const task of tasks) {
    const row = document.createElement("div");
    row.className = "service-task-item" + (task.active ? " active" : "");
    const kind = document.createElement("div");
    kind.className = "service-task-kind";
    kind.textContent = task.type === "document" ? "文件项目" : "文本生成";
    const name = document.createElement("div");
    name.className = "service-task-name";
    name.textContent = task.title || "未命名任务";
    name.title = name.textContent;
    const state = document.createElement("div");
    state.className = "service-task-meta";
    if (task.type === "document") {
      state.textContent = `${serviceTaskStateLabel(task.state)} · ${task.completed_segments || 0}/${task.total_segments || 0} 段`;
    } else {
      state.textContent = `${serviceTaskStateLabel(task.state)} · ${task.generated_frames || 0}/${task.max_new_tokens || 0} 帧`;
    }
    const profile = document.createElement("div");
    profile.className = "service-task-meta";
    profile.textContent = `${task.model_label || ""}${task.voice_name ? " · " + task.voice_name : ""} · Seed ${formatSeed(task.seed, task.seed_mode)}`;
    profile.title = profile.textContent;
    const open = document.createElement("button");
    open.type = "button";
    open.className = "service-task-open";
    open.textContent = "查看";
    open.onclick = () => openServiceTask(task);
    row.append(kind, name, state, profile, open);
    list.appendChild(row);
  }
}
async function pollServiceTasks() {
  try {
    renderServiceTasks(await fetchJson(apiUrl("api/service/tasks")));
  } catch (err) {
    field("service-active-count").textContent = "连接失败";
  }
}
async function openServiceTask(task) {
  if (task.type === "document") {
    setWorkspaceTab("document");
    localStorage.setItem(DOCUMENT_PROJECT_SELECTION_KEY, task.id);
    await loadDocumentProjects(task.id);
    return;
  }
  setWorkspaceTab("text");
  currentJob = task.id;
  currentJobOwned = false;
  currentStreamingGenerationEnabled = false;
  if (statusTimer) clearInterval(statusTimer);
  statusTimer = setInterval(() => pollStatus(currentJob), 500);
  await pollStatus(currentJob);
  const status = await fetchJson(apiUrl(`api/generate-stream/${task.id}/status`));
  setGenerationActive(["queued", "loading_runtime", "running"].includes(status.state));
}
function selectedModelProfile() {
  const value = field("model-profile").value;
  return ["qwen_0_6b", "qwen_1_7b"].includes(value)
    ? value
    : "qwen_0_6b";
}
function isQwenProfile(profile = selectedModelProfile()) {
  return String(profile).startsWith("qwen_");
}
function activeGenerationParameters() {
  if (isQwenProfile()) {
    return {
      temperature: field("qwen-temperature").value,
      topP: field("qwen-top-p").value,
      topK: field("qwen-top-k").value,
      repetitionPenalty: field("qwen-repetition-penalty").value,
      maxNewTokens: field("qwen-max-new-tokens").value,
      chunkFrames: field("qwen-chunk-size").value,
      seed: field("qwen-seed").value,
      initialPlaybackDelay: field("qwen-initial-playback-delay").value,
      streaming: field("qwen-streaming-generation").checked,
    };
  }
  return {
    temperature: field("temperature").value,
    topP: field("top-p").value,
    topK: field("top-k").value,
    repetitionPenalty: field("repetition-penalty").value,
    maxNewTokens: field("max-new-tokens").value,
    chunkFrames: field("codec-chunk-frames").value,
    seed: field("seed").value,
    initialPlaybackDelay: field("initial-playback-delay").value,
    streaming: field("streaming-generation").checked,
  };
}
function applyModelProfileCapabilities(persist = true) {
  const profile = selectedModelProfile();
  const qwen = isQwenProfile(profile);
  field("moss-params-panel").classList.toggle("hidden", qwen);
  field("moss-duration-panel").classList.toggle("hidden", qwen);
  field("duration-hint").classList.toggle("hidden", qwen);
  field("qwen-params-panel").classList.toggle("hidden", !qwen);
  field("qwen-reference-text-wrap").classList.toggle(
    "hidden",
    !qwen || field("qwen-clone-mode").value !== "icl"
  );
  field("streaming-generation").disabled = false;
  field("qwen-streaming-generation").disabled = false;
  const hints = {
    qwen_0_6b: "Qwen 0.6B：GGML Metal 极速克隆，默认X-vector无需参考文字。新项目会锁定模型、量化和参数。",
    qwen_1_7b: "Qwen 1.7B：GGML Metal 高质量克隆，支持X-vector与ICL。ICL必须填写准确参考文字。",
  };
  field("model-profile-hint").textContent = hints[profile] || hints.qwen_0_6b;
  updateIclTranscriptStatus();
  updateDurationControls();
  if (persist) saveUiState();
}

function favoriteCloneVoices() {
  return CLONE_VOICES.filter((voice) => !hiddenCloneVoicePaths.has(voice.audio_path));
}

function hiddenCloneVoices() {
  return CLONE_VOICES.filter((voice) => hiddenCloneVoicePaths.has(voice.audio_path));
}

function persistHiddenCloneVoices() {
  try {
    localStorage.setItem(HIDDEN_CLONE_VOICES_STORAGE_KEY, JSON.stringify([...hiddenCloneVoicePaths]));
  } catch (err) {}
}

function setWorkspaceTab(tabName, persist = true) {
  activeWorkspaceTab = tabName === "document" ? "document" : "text";
  const textActive = activeWorkspaceTab === "text";
  field("workspace-text").classList.toggle("hidden", !textActive);
  field("workspace-document").classList.toggle("hidden", textActive);
  field("workspace-tab-text").classList.toggle("active", textActive);
  field("workspace-tab-document").classList.toggle("active", !textActive);
  field("workspace-tab-text").setAttribute("aria-selected", String(textActive));
  field("workspace-tab-document").setAttribute("aria-selected", String(!textActive));
  if (persist) saveUiState();
}

function saveUiState() {
  if (!uiStateReady) return;
  const values = {};
  for (const id of PERSISTED_VALUE_FIELDS) values[id] = field(id).value;
  const state = {
    text: field("text").value,
    selectedVoicePath: field("clone-voice").value,
    activeWorkspaceTab,
    activeCloneVoiceTab,
    tokensControl: field("tokens-control").checked,
    streamingGeneration: field("streaming-generation").checked,
    qwenStreamingGeneration: field("qwen-streaming-generation").checked,
    qwenNonStreamingMode: field("qwen-non-streaming-mode").checked,
    qwenAppendSilence: field("qwen-append-silence").checked,
    mossSamplingOpen: field("moss-params-panel").open,
    qwenSamplingOpen: field("qwen-params-panel").open,
    optimizationDefaultsVersion: UI_OPTIMIZATION_DEFAULTS_VERSION,
    values,
  };
  try {
    localStorage.setItem(UI_STATE_STORAGE_KEY, JSON.stringify(state));
  } catch (err) {}
}

function restoreUiState() {
  let state = {};
  try {
    state = JSON.parse(localStorage.getItem(UI_STATE_STORAGE_KEY) || "{}");
  } catch (err) {
    state = {};
  }
  if (typeof state.text === "string") field("text").value = state.text;
  if (state.values && typeof state.values === "object") {
    for (const id of PERSISTED_VALUE_FIELDS) {
      if (state.values[id] == null) continue;
      field(id).value = String(state.values[id]);
      const range = field(`${id}-range`);
      if (range) range.value = String(state.values[id]);
    }
  }
  if (Number(state.optimizationDefaultsVersion || 0) < 2) {
    field("codec-chunk-frames").value = "16";
    field("codec-chunk-frames-range").value = "16";
  }
  if (Number(state.optimizationDefaultsVersion || 0) < 3) {
    field("initial-playback-delay").value = "1.5";
  }
  if (Number(state.optimizationDefaultsVersion || 0) < 4) {
    field("model-profile").value = "qwen_0_6b";
  }
  if (typeof state.tokensControl === "boolean") field("tokens-control").checked = state.tokensControl;
  if (typeof state.streamingGeneration === "boolean") field("streaming-generation").checked = state.streamingGeneration;
  if (typeof state.qwenStreamingGeneration === "boolean") field("qwen-streaming-generation").checked = state.qwenStreamingGeneration;
  if (typeof state.qwenNonStreamingMode === "boolean") field("qwen-non-streaming-mode").checked = state.qwenNonStreamingMode;
  if (typeof state.qwenAppendSilence === "boolean") field("qwen-append-silence").checked = state.qwenAppendSilence;
  applyModelProfileCapabilities(false);
  if (typeof state.mossSamplingOpen === "boolean") field("moss-params-panel").open = state.mossSamplingOpen;
  else if (typeof state.samplingOpen === "boolean") field("moss-params-panel").open = state.samplingOpen;
  if (typeof state.qwenSamplingOpen === "boolean") field("qwen-params-panel").open = state.qwenSamplingOpen;
  setWorkspaceTab(state.activeWorkspaceTab, false);
  activeCloneVoiceTab = state.activeCloneVoiceTab === "hidden" ? "hidden" : "favorites";
  setupCloneVoices();
  renderCloneVoiceList();
  const visibleVoices = favoriteCloneVoices();
  const savedVoice = visibleVoices.find((voice) => voice.audio_path === state.selectedVoicePath);
  const fallbackVoice = visibleVoices.find((voice) => voice.name === "龙嫱") || visibleVoices[0];
  if (savedVoice || fallbackVoice) selectCloneVoice((savedVoice || fallbackVoice).audio_path, false);
  uiStateReady = true;
  updateDurationControls();
  syncCloneVoiceListSelection();
  saveUiState();
}

function setupUiStatePersistence() {
  field("text").addEventListener("input", saveUiState);
  field("clone-voice").addEventListener("change", saveUiState);
  field("tokens-control").addEventListener("change", saveUiState);
  field("streaming-generation").addEventListener("change", saveUiState);
  field("qwen-streaming-generation").addEventListener("change", saveUiState);
  field("qwen-non-streaming-mode").addEventListener("change", saveUiState);
  field("qwen-append-silence").addEventListener("change", saveUiState);
  field("qwen-clone-mode").addEventListener("change", () => applyModelProfileCapabilities(true));
  field("qwen-reference-text").addEventListener("input", saveCurrentIclTranscriptOverride);
  field("model-profile").addEventListener("change", () => applyModelProfileCapabilities(true));
  for (const id of PERSISTED_VALUE_FIELDS) {
    field(id).addEventListener("input", saveUiState);
    const range = field(`${id}-range`);
    if (range) range.addEventListener("input", saveUiState);
  }
  field("moss-params-panel").addEventListener("toggle", saveUiState);
  field("qwen-params-panel").addEventListener("toggle", saveUiState);
}

function setCloneVoiceTab(tabName) {
  activeCloneVoiceTab = tabName === "hidden" ? "hidden" : "favorites";
  renderCloneVoiceList();
  saveUiState();
}

function hideCloneVoice(voice) {
  hiddenCloneVoicePaths.add(voice.audio_path);
  persistHiddenCloneVoices();
  const wasSelected = field("clone-voice").value === voice.audio_path;
  setupCloneVoices();
  if (wasSelected) {
    const visibleVoices = favoriteCloneVoices();
    const fallbackVoice = visibleVoices.find((item) => item.name === "龙嫱") || visibleVoices[0];
    if (fallbackVoice) selectCloneVoice(fallbackVoice.audio_path, false);
  }
  renderCloneVoiceList();
  saveUiState();
}

function favoriteCloneVoice(voice) {
  hiddenCloneVoicePaths.delete(voice.audio_path);
  persistHiddenCloneVoices();
  setupCloneVoices();
  renderCloneVoiceList();
  saveUiState();
}

function syncCloneVoiceListSelection() {
  const selectedPath = field("clone-voice").value;
  for (const item of document.querySelectorAll(".clone-voice-item")) {
    item.classList.toggle("active", item.dataset.audioPath === selectedPath);
  }
}

function selectedCloneVoice() {
  const selectedPath = field("clone-voice").value;
  return CLONE_VOICES.find((voice) => voice.audio_path === selectedPath) || null;
}

function currentVoiceName() {
  return selectedCloneVoice()?.name || activePresetVoiceName || "本地克隆音色";
}

function persistIclTranscriptOverrides() {
  try {
    localStorage.setItem(
      ICL_TRANSCRIPT_OVERRIDES_STORAGE_KEY,
      JSON.stringify(iclTranscriptOverrides)
    );
  } catch (err) {}
}

function updateIclTranscriptStatus() {
  const voice = selectedCloneVoice();
  const textarea = field("qwen-reference-text");
  const status = field("qwen-transcript-status");
  const hint = field("qwen-transcript-hint");
  const reset = field("qwen-transcript-reset");
  const hasPreset = Boolean(voice && String(voice.transcript || "").trim());
  const hasText = Boolean(textarea.value.trim());
  const hasOverride = Boolean(
    voice && Object.prototype.hasOwnProperty.call(iclTranscriptOverrides, voice.audio_path)
  );
  status.classList.toggle("needs-text", !hasText);
  reset.disabled = !hasPreset;
  if (!voice) {
    status.textContent = hasText ? "自定义台词" : "需要填写";
    hint.textContent = "上传或录制的参考音频需要手动填写逐字台词。";
    return;
  }
  if (!hasText) {
    status.textContent = "需要填写";
    hint.textContent = `${voice.name} 暂无可用台词，请手动填写。`;
    return;
  }
  status.textContent = hasOverride ? "已保存修订" : "预置台词已匹配";
  const language = voice.language || "Chinese";
  hint.textContent = `${voice.name} · ${language} · 可直接用于 ICL，也可以在此修正。`;
}

function applySelectedVoiceTranscript(force = false) {
  const voice = selectedCloneVoice();
  if (!voice) {
    if (force) field("qwen-reference-text").value = "";
    field("qwen-reference-text").dataset.voicePath = "";
    updateIclTranscriptStatus();
    return;
  }
  const textarea = field("qwen-reference-text");
  const voiceChanged = textarea.dataset.voicePath !== voice.audio_path;
  if (force || voiceChanged) {
    const override = Object.prototype.hasOwnProperty.call(
      iclTranscriptOverrides, voice.audio_path
    ) ? iclTranscriptOverrides[voice.audio_path] : null;
    textarea.value = override == null ? String(voice.transcript || "") : String(override);
  }
  textarea.dataset.voicePath = voice.audio_path;
  updateIclTranscriptStatus();
}

function clearIclTranscriptForCustomReference() {
  field("qwen-reference-text").value = "";
  field("qwen-reference-text").dataset.voicePath = "";
  updateIclTranscriptStatus();
  saveUiState();
}

function saveCurrentIclTranscriptOverride() {
  const voice = selectedCloneVoice();
  if (!voice) {
    updateIclTranscriptStatus();
    saveUiState();
    return;
  }
  const text = field("qwen-reference-text").value;
  if (text.trim() && text.trim() !== String(voice.transcript || "").trim()) {
    iclTranscriptOverrides[voice.audio_path] = text;
  } else {
    delete iclTranscriptOverrides[voice.audio_path];
  }
  persistIclTranscriptOverrides();
  updateIclTranscriptStatus();
  saveUiState();
}

function resetCurrentIclTranscript() {
  const voice = selectedCloneVoice();
  if (!voice) return;
  delete iclTranscriptOverrides[voice.audio_path];
  persistIclTranscriptOverrides();
  field("qwen-reference-text").value = String(voice.transcript || "");
  field("qwen-reference-text").dataset.voicePath = voice.audio_path;
  updateIclTranscriptStatus();
  saveUiState();
}

function selectCloneVoice(audioPath, autoplay = false) {
  const select = field("clone-voice");
  if (!audioPath) {
    clearReferenceAudio();
    return;
  }
  if (referenceRecordingActive) stopReferenceRecording(true);
  recordedReferenceFile = null;
  referenceSourceMode = "upload";
  field("prompt-audio").value = "";
  field("example-audio-path").value = audioPath;
  select.value = audioPath;
  document.querySelector("input[name='mode'][value='voice_clone']").checked = true;
  syncReferenceSourceControls();
  updateReferencePreview();
  updateReferenceLabel();
  syncCloneVoiceListSelection();
  const selectedVoice = CLONE_VOICES.find((voice) => voice.audio_path === audioPath);
  activePresetVoiceName = selectedVoice?.name || "";
  applySelectedVoiceTranscript(true);
  setStatus(`已选择克隆音色：${selectedVoice ? `${selectedVoice.name} — ${selectedVoice.description}` : audioPath}`);
  if (autoplay) {
    const preview = field("reference-audio-preview");
    preview.play().catch((err) => setStatus(`试听失败：${err}`));
  }
  saveUiState();
}

function setPresetStatus(message) {
  field("preset-status").textContent = String(message || "");
}

function setPresetField(id, value) {
  if (value === undefined || value === null || !field(id)) return;
  field(id).value = String(value);
  const range = field(`${id}-range`);
  if (range) range.value = String(value);
}

function selectExternalPresetReference(audioPath, voiceName = "") {
  if (referenceRecordingActive) stopReferenceRecording(true);
  recordedReferenceFile = null;
  referenceSourceMode = "upload";
  field("prompt-audio").value = "";
  field("example-audio-path").value = audioPath;
  field("clone-voice").value = "";
  activePresetVoiceName = String(voiceName || "");
  syncReferenceSourceControls();
  updateReferencePreview();
  updateReferenceLabel();
  syncCloneVoiceListSelection();
  clearIclTranscriptForCustomReference();
}

function currentPresetSettings(referenceAudioPath) {
  const voice = selectedCloneVoice();
  const referencePath = referenceAudioPath || field("example-audio-path").value || voice?.audio_path || "";
  return {
    model_profile: selectedModelProfile(),
    voice_name: voice?.name || activePresetVoiceName || field("preset-name").value.trim() || "自定义克隆音色",
    reference_audio_path: referencePath,
    qwen_clone_mode: field("qwen-clone-mode").value,
    qwen_reference_text: field("qwen-reference-text").value,
    qwen_temperature: field("qwen-temperature").value,
    qwen_top_p: field("qwen-top-p").value,
    qwen_top_k: field("qwen-top-k").value,
    qwen_repetition_penalty: field("qwen-repetition-penalty").value,
    qwen_max_new_tokens: field("qwen-max-new-tokens").value,
    qwen_chunk_size: field("qwen-chunk-size").value,
    qwen_min_new_tokens: field("qwen-min-new-tokens").value,
    qwen_seed: field("qwen-seed").value,
    qwen_initial_playback_delay: field("qwen-initial-playback-delay").value,
    qwen_streaming_generation: field("qwen-streaming-generation").checked,
    qwen_non_streaming_mode: field("qwen-non-streaming-mode").checked,
    qwen_append_silence: field("qwen-append-silence").checked,
  };
}

function applyVoicePreset(preset) {
  if (!preset || !preset.settings) throw new Error("预设不存在或数据不完整");
  const settings = preset.settings;
  setPresetField("model-profile", settings.model_profile || "qwen_0_6b");
  for (const [setting, fieldId] of Object.entries({
    qwen_clone_mode: "qwen-clone-mode",
    qwen_reference_text: "qwen-reference-text",
    qwen_temperature: "qwen-temperature",
    qwen_top_p: "qwen-top-p",
    qwen_top_k: "qwen-top-k",
    qwen_repetition_penalty: "qwen-repetition-penalty",
    qwen_max_new_tokens: "qwen-max-new-tokens",
    qwen_chunk_size: "qwen-chunk-size",
    qwen_min_new_tokens: "qwen-min-new-tokens",
    qwen_seed: "qwen-seed",
    qwen_initial_playback_delay: "qwen-initial-playback-delay",
  })) setPresetField(fieldId, settings[setting]);
  for (const [setting, fieldId] of Object.entries({
    qwen_streaming_generation: "qwen-streaming-generation",
    qwen_non_streaming_mode: "qwen-non-streaming-mode",
    qwen_append_silence: "qwen-append-silence",
  })) {
    if (typeof settings[setting] === "boolean") field(fieldId).checked = settings[setting];
  }
  applyModelProfileCapabilities(false);
  const referencePath = String(settings.reference_audio_path || "");
  if (referencePath) {
    const builtInVoice = CLONE_VOICES.find((voice) => voice.audio_path === referencePath);
    if (builtInVoice) selectCloneVoice(referencePath, false);
    else selectExternalPresetReference(referencePath, settings.voice_name || preset.name);
  }
  if (settings.qwen_reference_text != null) field("qwen-reference-text").value = String(settings.qwen_reference_text);
  updateIclTranscriptStatus();
  field("preset-select").value = String(preset.id || "");
  field("preset-name").value = String(preset.name || "");
  field("preset-update").disabled = !preset.id;
  field("preset-delete").disabled = !preset.id;
  setPresetStatus(`已应用预设：${preset.name}`);
  saveUiState();
}

function renderVoicePresets(selectedId = field("preset-select").value) {
  const select = field("preset-select");
  select.innerHTML = '<option value="">选择已保存预设…</option>';
  for (const preset of voicePresets) {
    const option = document.createElement("option");
    option.value = preset.id;
    option.textContent = `${preset.name} · ${preset.settings?.model_profile === "qwen_1_7b" ? "1.7B" : "0.6B"}`;
    select.appendChild(option);
  }
  if (voicePresets.some((preset) => preset.id === selectedId)) select.value = selectedId;
  const selected = voicePresets.find((preset) => preset.id === select.value);
  field("preset-update").disabled = !selected;
  field("preset-delete").disabled = !selected;
}

async function loadVoicePresets(selectedId = "") {
  const data = await fetchJson(apiUrl("api/presets"));
  voicePresets = Array.isArray(data.presets) ? data.presets : [];
  renderVoicePresets(selectedId || field("preset-select").value);
  return voicePresets;
}

async function persistPresetReferenceAudio() {
  const file = field("prompt-audio").files[0] || recordedReferenceFile;
  if (!file) return field("example-audio-path").value || selectedCloneVoice()?.audio_path || "";
  setPresetStatus("正在导入自定义参考音频…");
  const form = new FormData();
  form.append("audio", file, file.name || "reference.wav");
  const stored = await fetchJson(apiUrl("api/presets/reference-audio"), { method: "POST", body: form });
  selectExternalPresetReference(stored.reference_audio_path);
  return String(stored.reference_audio_path || "");
}

async function saveVoicePreset(update = false) {
  const selectedId = field("preset-select").value;
  const name = field("preset-name").value.trim() || voicePresets.find((item) => item.id === selectedId)?.name || "";
  if (!name) throw new Error("请填写预设名称");
  const referenceAudioPath = await persistPresetReferenceAudio();
  if (!referenceAudioPath) throw new Error("请先选择、上传或录制参考音频");
  const response = await fetchJson(
    update && selectedId ? apiUrl(`api/presets/${encodeURIComponent(selectedId)}`) : apiUrl("api/presets"),
    {
      method: update && selectedId ? "PUT" : "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, settings: currentPresetSettings(referenceAudioPath) }),
    }
  );
  await loadVoicePresets(response.id);
  applyVoicePreset(response);
  setPresetStatus(`预设已保存：${response.name}`);
}

async function deleteSelectedVoicePreset() {
  const preset = voicePresets.find((item) => item.id === field("preset-select").value);
  if (!preset) return;
  if (!window.confirm(`删除预设“${preset.name}”？参考音频文件会保留，已有文件项目不会受影响。`)) return;
  await fetchJson(apiUrl(`api/presets/${encodeURIComponent(preset.id)}`), { method: "DELETE" });
  field("preset-name").value = "";
  await loadVoicePresets();
  setPresetStatus(`已删除预设：${preset.name}`);
}

window.QwenTTSNative = {
  applyPreset: (preset) => applyVoicePreset(preset),
  refreshPresets: () => loadVoicePresets(),
};

function renderCloneVoiceList() {
  const list = field("clone-voices-list");
  list.innerHTML = "";
  const favoriteVoices = favoriteCloneVoices();
  const hiddenVoices = hiddenCloneVoices();
  const voices = activeCloneVoiceTab === "hidden" ? hiddenVoices : favoriteVoices;
  const favoritesTab = field("clone-tab-favorites");
  const hiddenTab = field("clone-tab-hidden");
  favoritesTab.textContent = `收藏 (${favoriteVoices.length})`;
  hiddenTab.textContent = `隐藏 (${hiddenVoices.length})`;
  favoritesTab.classList.toggle("active", activeCloneVoiceTab === "favorites");
  hiddenTab.classList.toggle("active", activeCloneVoiceTab === "hidden");
  favoritesTab.setAttribute("aria-selected", activeCloneVoiceTab === "favorites" ? "true" : "false");
  hiddenTab.setAttribute("aria-selected", activeCloneVoiceTab === "hidden" ? "true" : "false");
  if (!voices.length) {
    const empty = document.createElement("div");
    empty.className = "clone-voices-empty";
    empty.textContent = activeCloneVoiceTab === "hidden" ? "暂无隐藏音色" : "暂无收藏音色";
    list.appendChild(empty);
    return;
  }
  for (const voice of voices) {
    const item = document.createElement("div");
    item.className = "clone-voice-item";
    item.dataset.audioPath = voice.audio_path;
    const info = document.createElement("div");
    info.className = "clone-voice-info";
    const name = document.createElement("div");
    name.className = "clone-voice-name";
    name.textContent = voice.name;
    const description = document.createElement("div");
    description.className = "clone-voice-description";
    description.textContent = voice.description;
    info.append(name, description);
    if (String(voice.transcript || "").trim()) {
      const meta = document.createElement("div");
      meta.className = "clone-voice-meta";
      meta.textContent = `ICL 文案已就绪 · ${voice.language || "Chinese"}`;
      info.appendChild(meta);
    }
    const controls = document.createElement("div");
    controls.className = "clone-voice-controls";
    const actionButton = document.createElement("button");
    actionButton.type = "button";
    actionButton.className = "clone-voice-action";
    actionButton.textContent = activeCloneVoiceTab === "hidden" ? "收藏" : "隐藏";
    actionButton.onclick = (event) => {
      event.stopPropagation();
      if (activeCloneVoiceTab === "hidden") favoriteCloneVoice(voice);
      else hideCloneVoice(voice);
    };
    controls.appendChild(actionButton);
    if (activeCloneVoiceTab === "favorites") {
      const previewButton = document.createElement("button");
      previewButton.type = "button";
      previewButton.className = "clone-voice-preview";
      previewButton.textContent = "▶ 试听";
      previewButton.onclick = (event) => {
        event.stopPropagation();
        selectCloneVoice(voice.audio_path, true);
      };
      controls.appendChild(previewButton);
      item.onclick = () => selectCloneVoice(voice.audio_path, false);
    }
    item.append(info, controls);
    list.appendChild(item);
  }
  syncCloneVoiceListSelection();
}

function setupCloneVoices() {
  const select = field("clone-voice");
  const selectedPath = select.value;
  select.innerHTML = "";
  for (const voice of favoriteCloneVoices()) {
    const option = document.createElement("option");
    option.value = voice.audio_path;
    option.textContent = `${voice.name} — ${voice.description}`;
    select.appendChild(option);
  }
  if (favoriteCloneVoices().some((voice) => voice.audio_path === selectedPath)) select.value = selectedPath;
  select.onchange = () => selectCloneVoice(select.value, false);
}
function updateModeHint() {
  const continuationMode = selectedMode() === "continuation" || selectedMode() === "continuation_clone";
  field("reference-transcript-panel").classList.toggle("hidden", !continuationMode);
  if (!hasReference()) {
    field("mode-hint").innerHTML = "Current mode: <b>Direct Generation</b> (no reference audio uploaded)";
  } else if (selectedMode() === "voice_clone") {
    field("mode-hint").innerHTML = "Current mode: <b>Clone</b> (speaker timbre will be cloned from the reference audio)";
  } else {
    field("mode-hint").innerHTML = `Current mode: <b>${selectedModeName()}</b><br><span class="hint">${CONTINUATION_NOTICE}</span>`;
  }
  updateDurationControls();
}
function detectTextLanguage(text) {
  const zh = (text.match(/[\u4e00-\u9fff]/g) || []).length;
  const en = (text.match(/[A-Za-z]/g) || []).length;
  if (zh === 0 && en === 0) return "en";
  return zh >= en ? "zh" : "en";
}
function supportsDurationControl() {
  return selectedMode() === "voice_clone";
}
function updateDurationControls() {
  const checkbox = field("tokens-control");
  const wrap = field("tokens-wrap");
  if (!supportsDurationControl()) {
    checkbox.checked = false;
    checkbox.disabled = true;
    wrap.classList.add("hidden");
    field("duration-hint").textContent = "Duration control is disabled for Continuation modes.";
    return;
  }
  checkbox.disabled = false;
  if (!checkbox.checked) {
    wrap.classList.add("hidden");
    field("duration-hint").textContent = "Duration control is disabled.";
    return;
  }
  const text = field("text").value || "";
  const lang = detectTextLanguage(text);
  const factor = lang === "zh" ? 3.098411951313033 : 0.8673376262755219;
  const defaultTokens = Math.max(1, Math.round(Math.max(text.length, 1) * factor));
  const minTokens = Math.max(1, Math.round(defaultTokens * 0.5));
  const maxTokens = Math.max(minTokens, Math.round(defaultTokens * 1.5));
  const current = Math.max(minTokens, Math.min(maxTokens, Number(field("tokens").value || defaultTokens)));
  field("tokens").min = String(minTokens);
  field("tokens").max = String(maxTokens);
  field("tokens").value = String(current);
  wrap.classList.remove("hidden");
  field("duration-hint").textContent = `Duration control enabled | detected language: ${lang === "zh" ? "Chinese" : "English"} | default=${defaultTokens}, range=[${minTokens}, ${maxTokens}]`;
}
function renderExamples() {
  const tbody = field("examples-body");
  tbody.innerHTML = "";
  for (const [index, example] of EXAMPLES.entries()) {
    const tr = document.createElement("tr");
    const role = document.createElement("td");
    role.className = "role-cell";
    role.textContent = example.role;
    const text = document.createElement("td");
    text.textContent = example.text;
    tr.append(role, text);
    tr.onclick = () => {
      field("text").value = example.text;
      field("example-audio-path").value = example.audio_path;
      if (LANGUAGES.includes(example.language)) field("language").value = example.language;
      if (referenceRecordingActive) stopReferenceRecording(true);
      recordedReferenceFile = null;
      referenceSourceMode = "upload";
      field("prompt-audio").value = "";
      syncReferenceSourceControls();
      updateReferencePreview();
      updateReferenceLabel();
      updateDurationControls();
      setStatus(`Example selected: ${example.role}`);
    };
    tbody.appendChild(tr);
  }
}
function setupLanguages() {
  const select = field("language");
  select.innerHTML = "";
  for (const language of LANGUAGES) {
    const option = document.createElement("option");
    option.value = language === "Auto (omit)" ? "" : language;
    option.textContent = language;
    select.appendChild(option);
  }
  select.value = "Chinese";
}
function setupRangePair(id, integer = false) {
  const range = field(`${id}-range`);
  const input = field(id);
  range.oninput = () => { input.value = range.value; };
  input.oninput = () => {
    let value = Number(input.value);
    const min = Number(input.min || range.min || 0);
    const max = Number(input.max || range.max || value);
    if (!Number.isFinite(value)) value = min;
    value = Math.max(min, Math.min(max, value));
    if (integer) value = Math.round(value);
    input.value = String(value);
    range.value = String(value);
  };
}
function mergeUint8Arrays(a, b) {
  const merged = new Uint8Array(a.length + b.length);
  merged.set(a, 0);
  merged.set(b, a.length);
  return merged;
}

function documentSettingsSnapshot() {
  const voiceSelect = field("clone-voice");
  const params = activeGenerationParameters();
  return {
    model_profile: selectedModelProfile(),
    reference_audio_path: field("example-audio-path").value || voiceSelect.value,
    voice_name: currentVoiceName(),
    temperature: params.temperature,
    top_p: params.topP,
    top_k: params.topK,
    repetition_penalty: params.repetitionPenalty,
    max_new_tokens: params.maxNewTokens,
    codec_chunk_frames: params.chunkFrames,
    seed: params.seed,
    qwen_clone_mode: field("qwen-clone-mode").value,
    qwen_reference_text: field("qwen-reference-text").value,
    qwen_non_streaming_mode: field("qwen-non-streaming-mode").checked,
    qwen_append_silence: field("qwen-append-silence").checked,
    qwen_instruct: field("qwen-instruct").value,
    qwen_min_new_tokens: field("qwen-min-new-tokens").value,
  };
}

function documentProjectUrl(projectId, suffix = "") {
  return apiUrl(`api/document-projects/${encodeURIComponent(projectId)}${suffix}`);
}

function documentMediaUrl(projectId, path) {
  const url = new URL(documentProjectUrl(projectId, "/media"));
  url.searchParams.set("path", path);
  return url.toString();
}

function formatDocumentDuration(seconds) {
  if (seconds == null || !Number.isFinite(Number(seconds))) return "--:--";
  const value = Math.max(0, Math.round(Number(seconds)));
  const hours = Math.floor(value / 3600);
  const minutes = Math.floor((value % 3600) / 60);
  const rest = value % 60;
  return hours > 0
    ? `${hours}:${String(minutes).padStart(2, "0")}:${String(rest).padStart(2, "0")}`
    : `${minutes}:${String(rest).padStart(2, "0")}`;
}

async function loadDocumentProjects(preferredProjectId = "") {
  const response = await fetch(apiUrl("api/document-projects"));
  if (!response.ok) throw new Error(await response.text());
  const data = await response.json();
  const select = field("document-project-select");
  const remembered = preferredProjectId || select.value || localStorage.getItem(DOCUMENT_PROJECT_SELECTION_KEY) || "";
  select.innerHTML = "";
  if (!data.projects.length) {
    const empty = document.createElement("option");
    empty.value = "";
    empty.textContent = "暂无项目";
    select.appendChild(empty);
    currentDocumentProject = null;
    renderDocumentProject(null);
    return;
  }
  for (const project of data.projects) {
    const option = document.createElement("option");
    option.value = project.id;
    option.textContent = `${project.name} · ${project.state} · ${project.stats.completed_segments}/${project.stats.total_segments}`;
    select.appendChild(option);
  }
  if (data.projects.some((project) => project.id === remembered)) select.value = remembered;
  localStorage.setItem(DOCUMENT_PROJECT_SELECTION_KEY, select.value);
  await loadDocumentProject(select.value);
}

async function loadDocumentProject(projectId) {
  if (!projectId) {
    currentDocumentProject = null;
    renderDocumentProject(null);
    return;
  }
  const response = await fetch(documentProjectUrl(projectId));
  if (!response.ok) throw new Error(await response.text());
  currentDocumentProject = await response.json();
  renderDocumentProject(currentDocumentProject);
}

function renderDocumentProject(project) {
  const startButton = field("document-start");
  const stopButton = field("document-stop");
  const playButton = field("document-play-continue");
  const deleteButton = field("document-delete");
  const finalDownload = field("document-final-download");
  const segmentsContainer = field("document-segments");
  if (!project) {
    startButton.disabled = true;
    stopButton.disabled = true;
    playButton.disabled = true;
    deleteButton.disabled = true;
    field("document-progress-fill").style.width = "0%";
    field("document-stats").textContent = "尚未选择项目。";
    field("document-project-status").textContent = "idle";
    finalDownload.classList.add("hidden");
    segmentsContainer.innerHTML = '<div class="clone-voices-empty">项目段落将在这里显示</div>';
    return;
  }
  const stats = project.stats || {};
  const progress = Math.max(0, Math.min(1, Number(stats.progress || 0)));
  field("document-progress-fill").style.width = `${(progress * 100).toFixed(1)}%`;
  const eta = stats.eta_seconds == null ? "计算中" : formatDocumentDuration(stats.eta_seconds);
  const speed = Number(stats.speed_realtime || 0);
  field("document-stats").textContent =
    `${project.state} · ${stats.completed_segments || 0}/${stats.total_segments || 0} 段 · ` +
    `${stats.completed_chars || 0}/${stats.total_chars || 0} 字 · 已生成 ${formatDocumentDuration(stats.completed_audio_seconds || 0)} · ` +
    `速度 ${speed > 0 ? speed.toFixed(2) + "×实时" : "计算中"} · 预计剩余 ${eta}`;
  field("document-project-status").textContent =
    `${project.name}\n${project.message || project.state}\n模型：${project.settings.model_label || "Qwen3-TTS"}（项目已锁定）\n音色：${project.settings.voice_name || ""}\n` +
    `Seed：${formatSeed(project.settings.seed, project.settings.seed_mode)}\n参数指纹：${String(project.settings_fingerprint || "").slice(0, 12)}`;
  startButton.disabled = project.state === "running" || project.state === "stopping";
  stopButton.disabled = project.state !== "running";
  deleteButton.disabled = project.state === "running" || project.state === "stopping";
  const completed = project.segments.filter((segment) => segment.status === "completed" && segment.audio_file);
  playButton.disabled = completed.length === 0;
  if (project.final_audio) {
    finalDownload.href = documentMediaUrl(project.id, project.final_audio);
    finalDownload.classList.remove("hidden");
  } else {
    finalDownload.classList.add("hidden");
  }
  segmentsContainer.innerHTML = "";
  for (const segment of project.segments) {
    const row = document.createElement("div");
    row.className = "document-segment";
    if (documentPlaybackIndex === segment.index) row.classList.add("playing");
    const number = document.createElement("div");
    number.textContent = `#${segment.index + 1}`;
    const text = document.createElement("div");
    text.className = "document-segment-text";
    text.textContent = segment.text;
    text.title = segment.text;
    const status = document.createElement("div");
    status.className = "document-segment-status";
    status.textContent = segment.status === "completed"
      ? formatDocumentDuration(segment.duration_seconds)
      : segment.status === "failed" ? "失败" : segment.status === "generating" ? "生成中" : segment.status === "encoding" ? "转码中" : "等待";
    const action = document.createElement("div");
    if (segment.status === "completed" && segment.audio_file) {
      row.classList.add("playable");
      row.tabIndex = 0;
      row.setAttribute("role", "button");
      row.setAttribute("aria-label", `播放第 ${segment.index + 1} 段`);
      action.className = "document-segment-play";
      action.textContent = documentPlaybackIndex === segment.index ? "正在播放" : "▶ 播放";
      row.onclick = () => playDocumentSegment(segment.index, true, 0);
      row.onkeydown = (event) => {
        if (event.key !== "Enter" && event.key !== " ") return;
        event.preventDefault();
        playDocumentSegment(segment.index, true, 0);
      };
    }
    row.append(number, text, status, action);
    segmentsContainer.appendChild(row);
  }
}

async function createDocumentProject(file) {
  if (!file) return;
  const form = new FormData();
  form.append("document", file);
  form.append("name", field("document-project-name").value);
  form.append("max_chars", field("document-max-chars").value);
  form.append("settings_json", JSON.stringify(documentSettingsSnapshot()));
  field("document-project-status").textContent = "正在创建项目并解析文档...";
  const response = await fetch(apiUrl("api/document-projects"), {method: "POST", body: form});
  if (!response.ok) throw new Error(await response.text());
  const project = await response.json();
  localStorage.setItem(DOCUMENT_PROJECT_SELECTION_KEY, project.id);
  field("document-create-file").value = "";
  await loadDocumentProjects(project.id);
}

async function appendDocumentToProject(file) {
  if (!file || !currentDocumentProject) return;
  const form = new FormData();
  form.append("document", file);
  field("document-project-status").textContent = "正在追加文档...";
  const response = await fetch(documentProjectUrl(currentDocumentProject.id, "/append"), {method: "POST", body: form});
  field("document-append-file").value = "";
  if (!response.ok) throw new Error(await response.text());
  currentDocumentProject = await response.json();
  renderDocumentProject(currentDocumentProject);
  await loadDocumentProjects(currentDocumentProject.id);
}

async function startDocumentProject() {
  if (!currentDocumentProject) return;
  const response = await fetch(documentProjectUrl(currentDocumentProject.id, "/start"), {method: "POST"});
  if (!response.ok) throw new Error(await response.text());
  currentDocumentProject = await response.json();
  renderDocumentProject(currentDocumentProject);
}

async function stopDocumentProject() {
  if (!currentDocumentProject) return;
  const response = await fetch(documentProjectUrl(currentDocumentProject.id, "/stop"), {method: "POST"});
  if (!response.ok) throw new Error(await response.text());
  currentDocumentProject = await response.json();
  renderDocumentProject(currentDocumentProject);
}

async function deleteDocumentProject() {
  if (!currentDocumentProject) return;
  if (!window.confirm(`确定删除项目“${currentDocumentProject.name}”及其全部音频吗？`)) return;
  const response = await fetch(documentProjectUrl(currentDocumentProject.id), {method: "DELETE"});
  if (!response.ok) throw new Error(await response.text());
  currentDocumentProject = null;
  localStorage.removeItem(DOCUMENT_PROJECT_SELECTION_KEY);
  await loadDocumentProjects();
}

async function playDocumentSegment(segmentIndex, continuous = true, offsetSeconds = 0) {
  if (!currentDocumentProject) return;
  const segment = currentDocumentProject.segments.find(
    (item) => item.index === segmentIndex && item.status === "completed" && item.audio_file
  );
  if (!segment) return;
  documentPlaybackIndex = segmentIndex;
  documentPlaybackActive = continuous;
  renderDocumentProject(currentDocumentProject);
  const audio = field("document-audio");
  audio.src = documentMediaUrl(currentDocumentProject.id, segment.audio_file);
  audio.onloadedmetadata = () => {
    if (offsetSeconds > 0 && offsetSeconds < audio.duration) audio.currentTime = offsetSeconds;
  };
  await audio.play();
}

async function playDocumentFromSavedPosition() {
  if (!currentDocumentProject) return;
  const saved = currentDocumentProject.playback || {segment_index: 0, offset_seconds: 0};
  const completed = currentDocumentProject.segments.filter(
    (segment) => segment.status === "completed" && segment.audio_file
  );
  const target = completed.find((segment) => segment.index >= Number(saved.segment_index || 0)) || completed[0];
  if (target) await playDocumentSegment(target.index, true, target.index === saved.segment_index ? saved.offset_seconds : 0);
}

async function saveDocumentPlaybackPosition() {
  if (!currentDocumentProject || documentPlaybackIndex == null) return;
  const audio = field("document-audio");
  const form = new FormData();
  form.append("segment_index", String(documentPlaybackIndex));
  form.append("offset_seconds", String(Number(audio.currentTime || 0).toFixed(3)));
  await fetch(documentProjectUrl(currentDocumentProject.id, "/playback"), {method: "POST", body: form});
}

async function advanceDocumentPlayback() {
  if (!documentPlaybackActive || !currentDocumentProject || documentPlaybackIndex == null) return;
  await loadDocumentProject(currentDocumentProject.id);
  const next = currentDocumentProject.segments.find(
    (segment) => segment.index > documentPlaybackIndex && segment.status === "completed" && segment.audio_file
  );
  if (next) await playDocumentSegment(next.index, true, 0);
  else documentPlaybackActive = false;
}

async function pollCurrentDocumentProject() {
  if (!currentDocumentProject) return;
  try {
    await loadDocumentProject(currentDocumentProject.id);
  } catch (err) {
    field("document-project-status").textContent = String(err);
  }
}
function clearPlaybackCompletionTimer() {
  if (playbackCompletionTimer) {
    window.clearTimeout(playbackCompletionTimer);
    playbackCompletionTimer = null;
  }
}
function updatePauseButtonState() {
  const pauseBtn = field("pause");
  if (audioContext) {
    pauseBtn.disabled = false;
    pauseBtn.textContent = playbackPaused ? "Resume Playback" : "Pause Playback";
    return;
  }
  pauseBtn.disabled = true;
  pauseBtn.textContent = "Pause Playback";
}
function resolveInitialPlaybackDelaySeconds() {
  const fallback = isQwenProfile() ? 0.8 : 1.5;
  const raw = Number(activeGenerationParameters().initialPlaybackDelay || fallback);
  return Number.isFinite(raw) ? Math.max(0.3, Math.min(60, raw)) : fallback;
}
function estimateRealtimeAudioDurationSeconds() {
  if (!isQwenProfile() && field("tokens-control").checked) {
    return Math.max(0.1, Number(field("tokens").value || 1) / 12.5);
  }
  const text = field("text").value || "";
  const qwen = isQwenProfile();
  const frameRate = qwen ? 12 : 12.5;
  const factor = qwen ? 3.2 : (detectTextLanguage(text) === "zh" ? 3.098411951313033 : 0.8673376262755219);
  const estimated = Math.max(0.1, text.length * factor / frameRate);
  const configuredMaximum = Math.max(1, Number(activeGenerationParameters().maxNewTokens || (qwen ? 2048 : 7500))) / frameRate;
  return Math.min(estimated, configuredMaximum);
}
function calculateAdaptiveRealtimeBufferTarget(minimumSeconds, expectedAudioSeconds, generationRate) {
  const rate = Number(generationRate || 0);
  const deficitBuffer = rate > 0 && rate < 1
    ? Number(expectedAudioSeconds || 0) * (1 - rate) + ADAPTIVE_PLAYBACK_BUFFER_SAFETY_SECONDS
    : 0;
  return Math.min(
    MAX_ADAPTIVE_PLAYBACK_BUFFER_SECONDS,
    Math.max(Number(minimumSeconds || 1.5), deficitBuffer)
  );
}
function updateAdaptiveRealtimeBufferTarget(status = null) {
  if (realtimePlaybackStarted) return;
  const measuredRate = Number(
    (status && status.post_first_generation_realtime_factor) ||
    (status && status.generation_realtime_factor) ||
    latestRealtimeGenerationRate || 0
  );
  if (Number.isFinite(measuredRate) && measuredRate > 0) latestRealtimeGenerationRate = measuredRate;
  adaptiveRealtimeBufferTargetSeconds = calculateAdaptiveRealtimeBufferTarget(
    currentInitialPlaybackDelaySeconds,
    estimatedRealtimeAudioSeconds,
    latestRealtimeGenerationRate
  );
  if (pendingRealtimePcmSeconds >= adaptiveRealtimeBufferTargetSeconds && pendingRealtimePcmChunks.length > 0) {
    flushPendingRealtimePcmChunks();
  }
}
function setGenerationActive(active) {
  generationActive = Boolean(active);
  field("start").disabled = generationActive;
  field("stop").disabled = !generationActive;
}
function resetRealtimePlaybackBuffer() {
  pendingRealtimePcmChunks = [];
  pendingRealtimePcmSeconds = 0;
  realtimePlaybackStarted = false;
  latestRealtimeGenerationRate = 0;
  adaptiveRealtimeBufferTargetSeconds = currentInitialPlaybackDelaySeconds;
}
function pcmChunkDurationSeconds(bytes, sampleRate, channels) {
  const bytesPerFrame = Math.max(1, Number(channels || 2) * 2);
  const frames = Math.floor(bytes.byteLength / bytesPerFrame);
  const resolvedSampleRate = Math.max(1, Number(sampleRate || 48000));
  return frames / resolvedSampleRate;
}
function flushPendingRealtimePcmChunks() {
  if (pendingRealtimePcmChunks.length === 0) return;
  const chunks = pendingRealtimePcmChunks;
  pendingRealtimePcmChunks = [];
  pendingRealtimePcmSeconds = 0;
  realtimePlaybackStarted = true;
  for (const chunk of chunks) {
    schedulePcmChunk(chunk.bytes, chunk.sampleRate, chunk.channels);
  }
}
function enqueueRealtimePcmChunk(bytes, sampleRate, channels) {
  if (realtimePlaybackStarted) {
    schedulePcmChunk(bytes, sampleRate, channels);
    return;
  }
  pendingRealtimePcmChunks.push({ bytes, sampleRate, channels });
  pendingRealtimePcmSeconds += pcmChunkDurationSeconds(bytes, sampleRate, channels);
  if (pendingRealtimePcmSeconds >= adaptiveRealtimeBufferTargetSeconds) {
    flushPendingRealtimePcmChunks();
  }
}
function pcm16ToAudioBuffer(bytes, sampleRate, channels) {
  const bytesPerFrame = channels * 2;
  const frames = Math.floor(bytes.byteLength / bytesPerFrame);
  const buffer = audioContext.createBuffer(channels, frames, sampleRate);
  const view = new DataView(bytes.buffer, bytes.byteOffset, frames * bytesPerFrame);
  for (let channelIndex = 0; channelIndex < channels; channelIndex += 1) {
    const channelData = buffer.getChannelData(channelIndex);
    for (let frameIndex = 0; frameIndex < frames; frameIndex += 1) {
      const byteOffset = (frameIndex * channels + channelIndex) * 2;
      channelData[frameIndex] = view.getInt16(byteOffset, true) / 32768.0;
    }
  }
  return buffer;
}
function schedulePcmChunk(bytes, sampleRate, channels) {
  if (!audioContext) {
    const AudioContextCtor = window.AudioContext || window.webkitAudioContext;
    if (!AudioContextCtor) throw new Error("This browser does not support Web Audio streaming playback.");
    audioContext = new AudioContextCtor({ sampleRate });
    nextPlaybackTime = 0;
    playbackPaused = false;
    updatePauseButtonState();
  }
  const buffer = pcm16ToAudioBuffer(bytes, sampleRate, channels);
  if (buffer.length === 0) return;
  const source = audioContext.createBufferSource();
  source.buffer = buffer;
  source.connect(audioContext.destination);
  const startAt = Math.max(nextPlaybackTime || (audioContext.currentTime + 0.04), audioContext.currentTime + 0.02);
  source.start(startAt);
  nextPlaybackTime = startAt + buffer.duration;
}
async function prepareRealtimePlayback(sampleRate) {
  const AudioContextCtor = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextCtor) throw new Error("This browser does not support Web Audio streaming playback.");
  audioContext = new AudioContextCtor({ sampleRate });
  await audioContext.resume();
  currentInitialPlaybackDelaySeconds = resolveInitialPlaybackDelaySeconds();
  estimatedRealtimeAudioSeconds = estimateRealtimeAudioDurationSeconds();
  adaptiveRealtimeBufferTargetSeconds = currentInitialPlaybackDelaySeconds;
  latestRealtimeGenerationRate = 0;
  nextPlaybackTime = 0;
  resetRealtimePlaybackBuffer();
  playbackPaused = false;
  updatePauseButtonState();
}
async function closeRealtimeStream(stopTrackedJob = currentJobOwned) {
  clearPlaybackCompletionTimer();
  if (statusTimer) {
    window.clearInterval(statusTimer);
    statusTimer = null;
  }
  if (currentStreamAbortController) {
    currentStreamAbortController.abort();
    currentStreamAbortController = null;
  }
  if (currentJob && stopTrackedJob) {
    fetch(apiUrl(`api/generate-stream/${currentJob}/close`), { method: "POST" }).catch(() => {});
  }
  currentJob = null;
  currentJobOwned = false;
  if (audioContext) {
    try { await audioContext.close(); } catch (err) {}
    audioContext = null;
  }
  playbackPaused = false;
  nextPlaybackTime = 0;
  resetRealtimePlaybackBuffer();
  updatePauseButtonState();
  setGenerationActive(false);
}
function monitorPlaybackCompletion() {
  clearPlaybackCompletionTimer();
  if (!audioContext) return;
  const poll = async () => {
    if (!audioContext) return;
    if (playbackPaused || nextPlaybackTime - audioContext.currentTime > 0.05) {
      playbackCompletionTimer = window.setTimeout(() => poll().catch(() => {}), 120);
      return;
    }
    try { await audioContext.close(); } catch (err) {}
    audioContext = null;
    playbackPaused = false;
    nextPlaybackTime = 0;
    updatePauseButtonState();
  };
  playbackCompletionTimer = window.setTimeout(() => poll().catch(() => {}), 120);
}
async function streamAudio(jobId, sampleRate, channels) {
  const response = await fetch(apiUrl(`api/generate-stream/${jobId}/audio`), {
    signal: currentStreamAbortController ? currentStreamAbortController.signal : undefined,
  });
  if (!response.ok) throw new Error(await response.text());
  if (!response.body) throw new Error("ReadableStream is not available on this response.");
  const reader = response.body.getReader();
  const resolvedChannels = Number(channels || response.headers.get("X-Audio-Channels") || 2);
  const resolvedSampleRate = Number(sampleRate || response.headers.get("X-Audio-Sample-Rate") || 48000);
  const bytesPerFrame = resolvedChannels * 2;
  let remainder = new Uint8Array(0);
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    if (value && value.byteLength > 0) {
      const merged = mergeUint8Arrays(remainder, value);
      const alignedLength = Math.floor(merged.length / bytesPerFrame) * bytesPerFrame;
      if (alignedLength <= 0) {
        remainder = merged;
        continue;
      }
      enqueueRealtimePcmChunk(merged.subarray(0, alignedLength), resolvedSampleRate, resolvedChannels);
      remainder = merged.subarray(alignedLength);
    }
  }
  flushPendingRealtimePcmChunks();
  monitorPlaybackCompletion();
}
async function pollStatus(jobId) {
  const status = await fetchJson(apiUrl(`api/generate-stream/${jobId}/status`));
  updateAdaptiveRealtimeBufferTarget(status);
  setStatus(status);
  const bufferedSeconds = realtimePlaybackStarted && audioContext
    ? Math.max(0, nextPlaybackTime - audioContext.currentTime)
    : pendingRealtimePcmSeconds;
  field("summary").textContent = `${status.state} | mode=${status.mode || selectedModeName()} | seed=${formatSeed(status.seed, status.seed_mode)} | frames=${status.generated_frames || 0} | emitted=${Number(status.emitted_audio_seconds || 0).toFixed(2)}s | generation=${Number(latestRealtimeGenerationRate || 0).toFixed(2)}× | buffer=${bufferedSeconds.toFixed(2)}/${adaptiveRealtimeBufferTargetSeconds.toFixed(2)}s`;
  if (status.state === "finished") {
    clearInterval(statusTimer);
    statusTimer = null;
    const result = await fetchJson(apiUrl(`api/generate-stream/${jobId}/result`));
    field("download").href = apiUrl(`api/generate-stream/${jobId}/result-audio`);
    field("download").style.display = "inline";
    const outputAudio = field("audio-output");
    outputAudio.src = apiUrl(`api/generate-stream/${jobId}/result-audio`);
    outputAudio.removeAttribute("disabled");
    outputAudio.load();
    setStatus({ ...status, result });
    if (!currentStreamingGenerationEnabled) {
      outputAudio.play().catch(err => {
        setStatus({ ...status, result, autoplay_error: String(err) });
      });
    }
    setGenerationActive(false);
  }
  if (status.state === "error" || status.state === "closed") {
    clearInterval(statusTimer);
    statusTimer = null;
    setGenerationActive(false);
  }
}
field("start").onclick = async () => {
  await closeRealtimeStream();
  setGenerationActive(true);
  field("download").style.display = "none";
  field("bar").style.width = "0%";
  field("summary").textContent = "";
  field("audio-output").pause();
  field("audio-output").setAttribute("disabled", "disabled");
  field("audio-output").removeAttribute("src");
  field("audio-output").load();
  const form = new FormData();
  form.append("mode", "voice_clone");
  form.append("language", "Chinese");
  form.append("text", field("text").value);
  form.append("prompt_text", field("prompt-text").value);
  const params = activeGenerationParameters();
  form.append("max_new_tokens", params.maxNewTokens);
  form.append("codec_chunk_frames", params.chunkFrames);
  form.append("seed", params.seed);
  form.append("tokens_control", field("tokens-control").checked ? "1" : "0");
  form.append("tokens", field("tokens").value);
  form.append("temperature", params.temperature);
  form.append("top_p", params.topP);
  form.append("top_k", params.topK);
  form.append("repetition_penalty", params.repetitionPenalty);
  form.append("model_profile", selectedModelProfile());
  form.append("voice_name", currentVoiceName());
  form.append("qwen_clone_mode", field("qwen-clone-mode").value);
  form.append("qwen_reference_text", field("qwen-reference-text").value);
  form.append("qwen_non_streaming_mode", field("qwen-non-streaming-mode").checked ? "1" : "0");
  form.append("qwen_append_silence", field("qwen-append-silence").checked ? "1" : "0");
  form.append("qwen_instruct", field("qwen-instruct").value);
  form.append("qwen_min_new_tokens", field("qwen-min-new-tokens").value);
  currentStreamingGenerationEnabled = params.streaming;
  form.append("streaming_generation", currentStreamingGenerationEnabled ? "1" : "0");
  form.append("example_audio_path", field("example-audio-path").value);
  const file = field("prompt-audio").files[0] || recordedReferenceFile;
  if (file) form.append("prompt_audio", file);
  try {
    if ((selectedMode() === "continuation" || selectedMode() === "continuation_clone") && hasReference() && !field("prompt-text").value.trim()) {
      throw new Error("Reference Audio Transcript is required for Continuation modes.");
    }
    if (isQwenProfile() && field("qwen-clone-mode").value === "icl" && !field("qwen-reference-text").value.trim()) {
      throw new Error("Qwen ICL克隆模式必须填写与参考音频逐字对应的文字。");
    }
    const configuredSeed = Number(params.seed);
    setStatus({
      state: "starting",
      seed: configuredSeed >= 0 ? configuredSeed : null,
      configured_seed: configuredSeed,
      seed_mode: configuredSeed < 0 ? "random" : "fixed",
    });
    const response = await fetch(apiUrl("api/generate-stream/start"), { method: "POST", body: form });
    if (!response.ok) throw new Error(await response.text());
    const start = await response.json();
    setStatus({
      state: "queued",
      seed: start.seed,
      configured_seed: start.configured_seed,
      seed_mode: start.seed_mode,
    });
    currentJob = start.job_id;
    currentJobOwned = true;
    currentInitialPlaybackDelaySeconds = resolveInitialPlaybackDelaySeconds();
    if (currentStreamingGenerationEnabled) {
      currentStreamAbortController = new AbortController();
      await prepareRealtimePlayback(start.sample_rate || 48000);
      streamAudio(currentJob, start.sample_rate || 48000, start.channels || 2).catch(err => {
        if (!(err && String(err).includes("AbortError"))) setStatus(String(err));
      });
    } else {
      currentStreamAbortController = null;
      resetRealtimePlaybackBuffer();
      updatePauseButtonState();
    }
    if (statusTimer) clearInterval(statusTimer);
    statusTimer = setInterval(() => pollStatus(currentJob), 500);
    pollStatus(currentJob);
    pollServiceTasks();
  } catch (err) {
    setGenerationActive(false);
    setStatus(String(err));
  }
};
field("stop").onclick = async () => {
  if (!generationActive) return;
  await closeRealtimeStream(true);
  field("summary").textContent = "已停止当前生成";
  setStatus("当前生成已停止，未播放的流式音频缓冲已清空。已完成的文档项目内容不受影响。");
};
field("pause").onclick = async () => {
  if (!audioContext) return;
  if (playbackPaused) {
    await audioContext.resume();
    playbackPaused = false;
  } else {
    await audioContext.suspend();
    playbackPaused = true;
  }
  updatePauseButtonState();
};

field("text").value = DEFAULT_TEXT;
setupLanguages();
setupCloneVoices();
renderCloneVoiceList();
field("workspace-tab-text").onclick = () => setWorkspaceTab("text");
field("workspace-tab-document").onclick = () => setWorkspaceTab("document");
field("service-task-refresh").onclick = () => pollServiceTasks();
field("clone-tab-favorites").onclick = () => setCloneVoiceTab("favorites");
field("clone-tab-hidden").onclick = () => setCloneVoiceTab("hidden");
const initialVisibleVoices = favoriteCloneVoices();
const initialCloneVoice = initialVisibleVoices.find((voice) => voice.name === "龙嫱") || initialVisibleVoices[0];
if (initialCloneVoice) selectCloneVoice(initialCloneVoice.audio_path, false);
renderRuntime(INITIAL_RUNTIME);
for (const id of ["temperature", "top-p", "top-k", "repetition-penalty", "max-new-tokens", "codec-chunk-frames", "seed"]) {
  setupRangePair(id, ["top-k", "max-new-tokens", "codec-chunk-frames", "seed"].includes(id));
}
for (const id of ["qwen-temperature", "qwen-top-p", "qwen-top-k", "qwen-repetition-penalty", "qwen-max-new-tokens", "qwen-chunk-size", "qwen-min-new-tokens", "qwen-seed"]) {
  setupRangePair(id, ["qwen-top-k", "qwen-max-new-tokens", "qwen-chunk-size", "qwen-min-new-tokens", "qwen-seed"].includes(id));
}
field("tokens-control").onchange = updateDurationControls;
field("text").oninput = updateDurationControls;
field("prompt-audio").onchange = () => {
  if (field("prompt-audio").files[0]) {
    if (referenceRecordingActive) stopReferenceRecording(true);
    recordedReferenceFile = null;
    referenceSourceMode = "upload";
    field("example-audio-path").value = "";
    field("clone-voice").value = "";
    activePresetVoiceName = "";
    syncCloneVoiceListSelection();
    clearIclTranscriptForCustomReference();
    syncReferenceSourceControls();
  }
  updateReferencePreview();
  updateReferenceLabel();
};
field("reference-source-upload").onclick = () => setReferenceSourceMode("upload");
field("reference-source-record").onclick = () => setReferenceSourceMode("record");
field("reference-record-button").onclick = () => toggleReferenceRecording();
field("clear-reference").onclick = clearReferenceAudio;
field("qwen-transcript-reset").onclick = resetCurrentIclTranscript;
field("preset-select").onchange = () => {
  const preset = voicePresets.find((item) => item.id === field("preset-select").value);
  field("preset-update").disabled = !preset;
  field("preset-delete").disabled = !preset;
  if (preset) field("preset-name").value = preset.name;
};
field("preset-apply").onclick = () => {
  const preset = voicePresets.find((item) => item.id === field("preset-select").value);
  if (!preset) return setPresetStatus("请先选择一个预设");
  try { applyVoicePreset(preset); } catch (err) { setPresetStatus(String(err)); }
};
field("preset-save").onclick = () => {
  saveVoicePreset(false).catch((err) => setPresetStatus(String(err)));
};
field("preset-update").onclick = () => {
  saveVoicePreset(true).catch((err) => setPresetStatus(String(err)));
};
field("preset-delete").onclick = () => {
  deleteSelectedVoicePreset().catch((err) => setPresetStatus(String(err)));
};
field("document-create-file").onchange = async () => {
  try { await createDocumentProject(field("document-create-file").files[0]); }
  catch (err) { field("document-project-status").textContent = String(err); }
};
field("document-append-file").onchange = async () => {
  try { await appendDocumentToProject(field("document-append-file").files[0]); }
  catch (err) { field("document-project-status").textContent = String(err); }
};
field("document-project-select").onchange = async () => {
  try {
    localStorage.setItem(DOCUMENT_PROJECT_SELECTION_KEY, field("document-project-select").value);
    await loadDocumentProject(field("document-project-select").value);
  } catch (err) { field("document-project-status").textContent = String(err); }
};
field("document-refresh").onclick = async () => {
  try { await loadDocumentProjects(); }
  catch (err) { field("document-project-status").textContent = String(err); }
};
field("document-start").onclick = async () => {
  try { await startDocumentProject(); }
  catch (err) { field("document-project-status").textContent = String(err); }
};
field("document-stop").onclick = async () => {
  try { await stopDocumentProject(); }
  catch (err) { field("document-project-status").textContent = String(err); }
};
field("document-delete").onclick = async () => {
  try { await deleteDocumentProject(); }
  catch (err) { field("document-project-status").textContent = String(err); }
};
field("document-play-continue").onclick = async () => {
  try { await playDocumentFromSavedPosition(); }
  catch (err) { field("document-project-status").textContent = String(err); }
};
field("document-audio").onended = () => advanceDocumentPlayback();
field("document-audio").ontimeupdate = () => {
  const now = Date.now();
  if (now - lastPlaybackSaveAt < 2000) return;
  lastPlaybackSaveAt = now;
  saveDocumentPlaybackPosition().catch(() => {});
};
for (const radio of document.querySelectorAll("input[name='mode']")) {
  radio.onchange = updateModeHint;
}
syncReferenceSourceControls();
updateRecordButtonState();
setupUiStatePersistence();
restoreUiState();
updateReferenceLabel();
loadVoicePresets().catch((err) => setPresetStatus(`预设服务不可用：${err}`));
loadDocumentProjects().catch((err) => { field("document-project-status").textContent = String(err); });
documentProjectPollTimer = setInterval(pollCurrentDocumentProject, 1000);
serviceTasksPollTimer = setInterval(pollServiceTasks, 1000);
pollServiceTasks();
setInterval(pollRuntime, 1500);
pollRuntime();
</script>
</body>
</html>
"""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Qwen3-TTS Apple Silicon service.")
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "7861")))
    parser.add_argument("--qwen-python", default=os.environ.get("QWEN_TTS_PYTHON", str(DEFAULT_QWEN_PYTHON)))
    parser.add_argument(
        "--qwen-worker-script",
        default=os.environ.get("QWEN_TTS_WORKER_SCRIPT", str(DEFAULT_QWEN_WORKER_SCRIPT)),
    )
    parser.add_argument(
        "--qwen-0-6b-model-dir",
        default=os.environ.get("QWEN_TTS_0_6B_MODEL_DIR", str(DEFAULT_QWEN_0_6B_MODEL_DIR)),
    )
    parser.add_argument(
        "--qwen-1-7b-model-dir",
        default=os.environ.get("QWEN_TTS_1_7B_MODEL_DIR", str(DEFAULT_QWEN_1_7B_MODEL_DIR)),
    )
    parser.add_argument(
        "--qwen-0-6b-lanes",
        type=int,
        default=int(os.environ.get("QWEN_TTS_0_6B_LANES", "1")),
    )
    parser.add_argument(
        "--qwen-1-7b-lanes",
        type=int,
        default=int(os.environ.get("QWEN_TTS_1_7B_LANES", "1")),
    )
    parser.add_argument(
        "--qwen-backend",
        choices=["ggml", "torch"],
        default=os.environ.get("QWEN_TTS_BACKEND", DEFAULT_QWEN_BACKEND),
    )
    parser.add_argument(
        "--qwen-quant",
        choices=["BF16", "Q8_0", "Q4_K_M"],
        default=os.environ.get("QWEN_TTS_QUANT", DEFAULT_QWEN_QUANT),
    )
    parser.add_argument(
        "--qwentts-library",
        default=os.environ.get("QWENTTS_CPP_LIBRARY", str(DEFAULT_QWENTTS_LIBRARY)),
    )
    parser.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR)))
    parser.add_argument("--upload-dir", default=os.environ.get("UPLOAD_DIR", str(DEFAULT_UPLOAD_DIR)))
    parser.add_argument("--preset-dir", default=os.environ.get("QWEN_TTS_PRESET_DIR", str(DEFAULT_PRESET_DIR)))
    parser.add_argument("--no-preload", action="store_true")
    parser.add_argument(
        "--max-parallel-generations",
        type=int,
        default=int(os.environ.get("QWEN_TTS_MAX_PARALLEL_GENERATIONS", "1")),
    )
    parser.add_argument(
        "--document-parallel-generations",
        type=int,
        default=int(os.environ.get("QWEN_TTS_DOCUMENT_PARALLEL_GENERATIONS", "2")),
    )
    parser.add_argument(
        "--whisper-server",
        default=os.environ.get("QWEN_STT_SERVER", str(DEFAULT_WHISPER_SERVER)),
    )
    parser.add_argument(
        "--whisper-model",
        default=os.environ.get("QWEN_STT_MODEL", str(DEFAULT_WHISPER_MODEL)),
    )
    parser.add_argument(
        "--whisper-port",
        type=int,
        default=int(os.environ.get("QWEN_STT_PORT", str(DEFAULT_WHISPER_PORT))),
    )
    parser.add_argument(
        "--whisper-threads",
        type=int,
        default=int(os.environ.get("QWEN_STT_THREADS", "8")),
    )
    parser.add_argument("--no-stt", action="store_true")
    parser.add_argument("--no-stt-preload", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    app = create_app(
        qwen_python=args.qwen_python,
        qwen_worker_script=args.qwen_worker_script,
        qwen_0_6b_model_dir=args.qwen_0_6b_model_dir,
        qwen_1_7b_model_dir=args.qwen_1_7b_model_dir,
        qwen_0_6b_lanes=max(1, int(args.qwen_0_6b_lanes)),
        qwen_1_7b_lanes=max(1, int(args.qwen_1_7b_lanes)),
        qwen_backend=args.qwen_backend,
        qwen_quant=args.qwen_quant,
        qwentts_library=args.qwentts_library,
        output_dir=args.output_dir,
        upload_dir=args.upload_dir,
        preset_dir=args.preset_dir,
        preload=not args.no_preload,
        max_parallel_generations=max(1, int(args.max_parallel_generations)),
        document_parallel_generations=max(1, int(args.document_parallel_generations)),
        access_password=os.environ.get("QWEN_TTS_ACCESS_PASSWORD", ""),
        stt_enabled=not args.no_stt,
        stt_preload=not args.no_stt_preload,
        whisper_server=args.whisper_server,
        whisper_model=args.whisper_model,
        whisper_port=args.whisper_port,
        whisper_threads=max(1, int(args.whisper_threads)),
    )
    uvicorn.run(app, host=args.host, port=int(args.port))


if __name__ == "__main__":
    main()
