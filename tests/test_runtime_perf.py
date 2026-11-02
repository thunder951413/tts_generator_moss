"""Regression tests for the 2026-10-14 runtime performance round."""

from __future__ import annotations

import base64
import io
import json
import logging
import logging.handlers
import os
import sys
import textwrap
import threading
import time
import types
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
SERVICE_DIR = ROOT / "qwen_tts_service"
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

import stt_runtime  # noqa: E402
from activity_control import PlaybackCoordinator  # noqa: E402
from qwen_protocol import StreamingRequest  # noqa: E402
from qwen_runtime import QwenWorkerRuntime  # noqa: E402
from storage_policy import StoragePolicy  # noqa: E402
from streaming_jobs import StreamingJobManager  # noqa: E402
from webapp.cli import QuietPollingAccessFilter, configure_logging  # noqa: E402
from webapp.util import _pcm16le_bytes  # noqa: E402
from test_macos_qwen_service import load_app_module, load_worker_module, make_worker_state  # noqa: E402


# --------------------------------------------------------------------------
# 1. Access log filter / logging configuration


def _access_record(method: str, path: str, status: int) -> logging.LogRecord:
    return logging.LogRecord(
        "uvicorn.access", logging.INFO, __file__, 1,
        '%s - "%s %s HTTP/%s" %d', ("127.0.0.1:5000", method, path, "1.1", status), None,
    )


@pytest.mark.parametrize(
    ("method", "path", "status", "kept"),
    [
        ("GET", "/api/health", 200, False),
        ("GET", "/api/health?x=1", 304, False),
        ("GET", "/api/playback/status", 200, False),
        ("GET", "/api/generate-stream/abc123/status", 200, False),
        ("GET", "/api/document-projects/p1/progress?since=4", 200, False),
        ("HEAD", "/reader", 200, False),
        ("GET", "/api/health", 500, True),
        ("GET", "/api/generate-stream/abc123/status", 404, True),
        ("GET", "/api/generate-stream/abc123/audio", 200, True),
        ("POST", "/api/generate-stream/start", 200, True),
        ("GET", "/api/document-projects", 200, True),
    ],
)
def test_access_filter_drops_successful_polling(method, path, status, kept) -> None:
    assert QuietPollingAccessFilter().filter(_access_record(method, path, status)) is kept


def test_configure_logging_writes_service_logs_to_rotating_file(tmp_path: Path) -> None:
    root = logging.getLogger()
    previous = (list(root.handlers), root.level)
    try:
        log_path = configure_logging(tmp_path / "logs")
        assert log_path == tmp_path / "logs" / "service.log"
        logging.getLogger("qwen-runtime").info("service info line")
        access = logging.getLogger("uvicorn.access")
        access.handle(_access_record("GET", "/api/health", 200))
        access.handle(_access_record("POST", "/api/generate-stream/start", 200))
        for handler in root.handlers:
            handler.flush()
        text = log_path.read_text(encoding="utf-8")
        assert "service info line" in text
        assert "/api/generate-stream/start" in text
        assert "GET /api/health" not in text
        rotating = [h for h in root.handlers if isinstance(h, logging.handlers.RotatingFileHandler)]
        assert rotating and rotating[0].maxBytes == 5 * 1024 * 1024 and rotating[0].backupCount == 3
    finally:
        for handler in list(root.handlers):
            if getattr(handler, "_qwen_service_handler", False):
                root.removeHandler(handler)
                handler.close()
        for handler in previous[0]:
            if handler not in root.handlers:
                root.addHandler(handler)
        root.setLevel(previous[1])


# --------------------------------------------------------------------------
# 2. Health never probes the network on the request path


class _AliveProcess:
    pid = 4242
    returncode = None

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = 0

    kill = terminate

    def wait(self, timeout=None):
        return self.returncode


def test_stt_status_uses_cached_readiness_without_sync_network(monkeypatch) -> None:
    runtime = stt_runtime.WhisperCppRuntime(binary="whisper-fake", model="model-fake")
    monkeypatch.setattr(runtime, "available", lambda: (True, None))
    probes: list[str] = []
    probe_started = threading.Event()
    release_probe = threading.Event()

    def slow_is_ready(*, timeout: float = 0.5) -> bool:
        probes.append(threading.current_thread().name)
        probe_started.set()
        release_probe.wait(2)
        return True

    monkeypatch.setattr(runtime, "_is_ready", slow_is_ready)
    process = _AliveProcess()
    runtime._process = process
    runtime._mark_ready(process, True)

    began = time.monotonic()
    status = runtime.status()
    assert time.monotonic() - began < 0.1
    assert status["ready"] is True and status["pid"] == 4242
    assert probes == []  # probed recently: no re-probe at all

    runtime._ready_probed_at -= stt_runtime.READY_REPROBE_SECONDS + 1
    began = time.monotonic()
    assert runtime.status()["ready"] is True
    assert time.monotonic() - began < 0.1  # re-probe is in the background
    assert probe_started.wait(1)
    assert probes and probes[0] == "whisper-ready-probe"
    assert runtime.status()["ready"] is True  # no second concurrent probe
    release_probe.set()
    assert len(probes) == 1

    runtime.close()
    assert runtime.status()["ready"] is False


def test_stt_available_check_is_cached(monkeypatch) -> None:
    runtime = stt_runtime.WhisperCppRuntime(binary="whisper-fake", model="model-fake")
    calls = []
    monkeypatch.setattr(runtime, "available", lambda: calls.append(1) or (False, "missing"))
    for _ in range(5):
        assert runtime.status()["state"] == "unavailable"
    assert len(calls) == 1


def test_health_endpoint_does_not_call_whisper_synchronously(monkeypatch, tmp_path: Path) -> None:
    module = load_app_module()
    app = module.create_app(
        qwen_python=sys.executable, qwentts_library="", output_dir=tmp_path / "out",
        upload_dir=tmp_path / "up", preset_dir=tmp_path / "presets",
        document_project_dir=tmp_path / "docs", service_job_dir=tmp_path / "jobs",
        performance_tuning_path=tmp_path / "perf.json",
        preload=False, stt_preload=False, access_password="",
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("health must not open a network connection")

    monkeypatch.setattr(stt_runtime.urllib.request, "urlopen", forbidden)
    with TestClient(app) as client:
        for _ in range(3):
            body = client.get("/api/health").json()
            assert body["stt"]["ready"] is False
            assert "reference_cache_entries" not in body
        assert client.get("/api/runtime").json()["stt"]["ready"] is False
        assert client.get("/api/stt/status").status_code == 200


def test_runtime_manager_file_checks_are_cached(monkeypatch, tmp_path: Path) -> None:
    import runtime_manager as runtime_manager_module

    manager = runtime_manager_module.RuntimeManager(
        qwen_python=sys.executable, qwen_worker_script=__file__,
        qwen_0_6b_model_dir=str(tmp_path), qwen_1_7b_model_dir=str(tmp_path),
        qwen_0_6b_lanes=1, qwen_1_7b_lanes=1, qwen_backend="ggml", qwen_quant="Q4_K_M",
        qwentts_library="",
    )
    calls = []
    original = runtime_manager_module.Path.is_file

    def counting_is_file(self):
        calls.append(str(self))
        return original(self)

    monkeypatch.setattr(runtime_manager_module.Path, "is_file", counting_is_file)
    for _ in range(4):
        assert all(item["available"] for item in manager.status()["profiles"])
    assert len(calls) == 2  # python + worker script, once


# --------------------------------------------------------------------------
# 3. Cancelling a stream reuses the worker when it goes idle


FAKE_WORKER = textwrap.dedent(
    r'''
    import argparse, base64, json, os, threading, time
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    parser = argparse.ArgumentParser()
    for name in ("--model", "--profile-id", "--backend", "--quant"):
        parser.add_argument(name)
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    TOKEN = os.environ["QWEN_WORKER_TOKEN"]
    LOCK = threading.Lock()
    PCM = base64.b64encode(b"\x01\x00" * 480).decode()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, payload):
            raw = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.headers.get("X-Qwen-Worker-Token") != TOKEN:
                self.send_response(401); self.end_headers(); return
            self._json({"ok": True, "busy": LOCK.locked(), "pid": os.getpid()})

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.send_response(200)
            self.end_headers()
            with LOCK:
                if payload["text"] == "stubborn":
                    time.sleep(5)  # ignores the disconnect entirely
                    return
                try:
                    self.wfile.write(json.dumps({"type": "metadata", "data": {}}).encode() + b"\n")
                    for _ in range(400):
                        event = {"type": "audio", "data": {"pcm16_base64": PCM, "samples": 480}}
                        self.wfile.write(json.dumps(event).encode() + b"\n")
                        self.wfile.flush()
                        time.sleep(0.02)
                    self.wfile.write(json.dumps({"type": "result", "data": {"metadata": {}}}).encode() + b"\n")
                except (BrokenPipeError, ConnectionResetError):
                    return

    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever(poll_interval=0.05)
    '''
)


@pytest.fixture
def fake_worker_runtime(tmp_path: Path):
    script = tmp_path / "fake_worker.py"
    script.write_text(FAKE_WORKER, encoding="utf-8")
    runtime = QwenWorkerRuntime(
        profile_id="qwen_0_6b", model_dir=str(tmp_path), python_executable=sys.executable,
        worker_script=script, lanes=1, base_port=0, log_dir=tmp_path / "logs",
    )
    try:
        yield runtime
    finally:
        runtime.close()


def _cancelled_stream(runtime: QwenWorkerRuntime, tmp_path: Path, text: str) -> None:
    cancelled = threading.Event()
    request = StreamingRequest(text=text, prompt_audio_path=str(tmp_path / "ref.wav"))
    request.cancelled = cancelled.is_set
    events = runtime.synthesize(request, output_dir=tmp_path / "out")
    if text != "stubborn":
        assert next(events).type == "metadata"
        audio = next(events)
        assert audio.type == "audio" and audio.data["pcm16"] == b"\x01\x00" * 480
        assert "waveform" not in audio.data
        cancelled.set()
        events.close()
    else:
        errors: list[BaseException] = []

        def drain() -> None:
            try:
                list(events)
            except RuntimeError as exc:  # the cancelled stream ends without a result
                errors.append(exc)

        thread = threading.Thread(target=drain, daemon=True)
        thread.start()
        time.sleep(0.3)
        cancelled.set()
        thread.join(5)
        assert not thread.is_alive()


def test_cancelled_stream_keeps_idle_worker_loaded(fake_worker_runtime, tmp_path: Path) -> None:
    runtime = fake_worker_runtime
    worker = runtime._workers[0]
    pid = worker.process.pid
    began = time.monotonic()
    _cancelled_stream(runtime, tmp_path, "hello")
    assert time.monotonic() - began < 3
    assert worker.process.poll() is None and worker.process.pid == pid
    assert runtime._available.qsize() == 1 and runtime._workers == [worker]
    assert runtime.pool_status()["state"] == "ready"
    assert worker.health()["busy"] is False


def test_cancelled_stream_retires_worker_that_stays_busy(fake_worker_runtime, tmp_path: Path) -> None:
    runtime = fake_worker_runtime
    worker = runtime._workers[0]
    worker.cancel_grace_seconds = 0.3
    _cancelled_stream(runtime, tmp_path, "stubborn")
    assert worker.process.poll() is not None
    deadline = time.monotonic() + 10
    while runtime.pool_status()["state"] != "ready" and time.monotonic() < deadline:
        time.sleep(0.05)
    assert runtime.pool_status()["state"] == "ready"
    assert runtime._workers[0] is not worker


def test_worker_health_reports_busy_and_detects_closed_client(tmp_path: Path) -> None:
    module = load_worker_module()

    class Model:
        def _resolve_clone_reference(self, **_kwargs):
            return {}, 1.0, {}

        def _stream_runtime(self, **_kwargs):
            for _ in range(50):
                yield np.zeros(240, dtype=np.float32), 24000, {}

    state = make_worker_state(module, Model())
    state.generation_lock = threading.Lock()
    assert state.health()["busy"] is False
    with state.generation_lock:
        assert state.health()["busy"] is True
    reference = tmp_path / "ref.wav"
    reference.write_bytes(b"x")
    checks = iter([False, False, True])
    events = state.generate(
        {"text": "测试", "ref_audio": str(reference), "output_dir": str(tmp_path), "seed": 1,
         "stream_audio": False},
        client_gone=lambda: next(checks),
    )
    with pytest.raises(ConnectionResetError):
        list(events)


# --------------------------------------------------------------------------
# 4. int16 PCM passthrough is bit-identical to the saved WAV


def test_streamed_pcm_is_bit_identical_to_saved_wav(tmp_path: Path) -> None:
    import soundfile as sf

    module = load_worker_module()
    rng = np.random.default_rng(7)
    chunks = [rng.uniform(-1.1, 1.1, 960).astype(np.float32) for _ in range(4)]

    class Model:
        def _resolve_clone_reference(self, **_kwargs):
            return {}, 1.0, {}

        def _stream_runtime(self, **_kwargs):
            for chunk in chunks:
                yield chunk, 24000, {}

    state = make_worker_state(module, Model())
    reference = tmp_path / "ref.wav"
    reference.write_bytes(b"x")
    events = list(state.generate({"text": "测试", "ref_audio": str(reference),
                                  "output_dir": str(tmp_path), "seed": 3}))
    streamed = b"".join(
        base64.b64decode(event["data"]["pcm16_base64"]) for event in events if event["type"] == "audio"
    )
    result = [event for event in events if event["type"] == "result"][0]
    saved, _rate = sf.read(result["data"]["audio_path"], dtype="int16")
    assert streamed == saved.astype("<i2").tobytes()


def test_pcm_helper_passthrough_and_channel_expansion() -> None:
    mono = np.array([1, -2, 32767, -32768], dtype="<i2").tobytes()
    assert _pcm16le_bytes(mono, 1) == mono
    stereo = np.frombuffer(_pcm16le_bytes(mono, 2), dtype="<i2")
    assert stereo.tolist() == [1, 1, -2, -2, 32767, 32767, -32768, -32768]
    floats = np.array([[0.5, -0.5, 1.5]], dtype=np.float32)
    assert np.frombuffer(_pcm16le_bytes(floats, 1), dtype="<i2").tolist() == [16384, -16384, 32767]


def test_main_process_does_not_import_torch() -> None:
    for name in ("qwen_runtime.py", "runtime_manager.py", "webapp/util.py"):
        source = (SERVICE_DIR / name).read_text(encoding="utf-8")
        assert "import torch" not in source, name
    assert "import torch" not in (ROOT / "clis" / "qwen_tts_app.py").read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# 5/7. Pure ASGI middleware, epochs, and streaming lease shortening


class _FakeRuntime:
    sample_rate = 24000
    n_vq = 16
    requests: list[StreamingRequest] = []

    def synthesize(self, request, *, output_dir):
        self.requests.append(request)
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        yield types.SimpleNamespace(type="metadata", data={"sample_rate": 24000, "channels": 1})
        yield types.SimpleNamespace(type="audio", data={
            "pcm16": b"\x02\x00" * 2400, "generated_frames": 1,
            "emitted_audio_seconds": 0.1, "generated_audio_seconds": 0.1, "sample_rate": 24000,
        })
        audio_path = output / "fake.wav"
        audio_path.write_bytes(b"RIFF")
        (output / "fake.json").write_text("{}", encoding="utf-8")
        yield types.SimpleNamespace(type="result", data={
            "audio_path": str(audio_path), "tokens_path": str(audio_path),
            "metadata_path": str(output / "fake.json"),
            "metadata": {"generated_frames": 1, "duration_seconds": 0.1},
        })


def _fake_app(module, tmp_path: Path, monkeypatch, password: str = ""):
    class FakeRuntimeManager:
        def __init__(self, **_kwargs):
            self.qwen_backend = "ggml"
            self.qwen_quant = "Q4_K_M"
            self.qwentts_library = ""
            self.device = "metal"
            self.dtype = "gguf"
            self.attn_implementation = "ggml_metal"
            self.active_profile = None
            self.profiles = {
                key: {"label": "t", "sample_rate": 24000, "channels": 1, "streaming": True, "backend": "qwen"}
                for key in ("qwen_0_6b", "qwen_1_7b")
            }

        @contextmanager
        def session(self, _profile, **_kwargs):
            yield _FakeRuntime()

        def status(self):
            return {"state": "ready", "device": "metal", "profiles": []}

        def close(self):
            return None

    monkeypatch.setattr(module, "RuntimeManager", FakeRuntimeManager)
    return module.create_app(
        qwen_python=sys.executable, qwentts_library="", output_dir=tmp_path / "out",
        upload_dir=tmp_path / "up", preset_dir=tmp_path / "presets",
        document_project_dir=tmp_path / "docs", service_job_dir=tmp_path / "jobs",
        performance_tuning_path=tmp_path / "perf.json",
        preload=False, stt_preload=False, access_password=password,
    )


def test_login_middleware_is_pure_asgi_and_preserves_auth(monkeypatch, tmp_path: Path) -> None:
    from starlette.middleware.base import BaseHTTPMiddleware

    module = load_app_module()
    app = _fake_app(module, tmp_path, monkeypatch, password="secret")
    assert not any(issubclass(item.cls, BaseHTTPMiddleware) for item in app.user_middleware
                   if isinstance(item.cls, type))
    with TestClient(app) as client:
        assert client.get("/api/health").status_code == 200
        assert client.get("/api/voices").status_code == 401
        assert client.get("/api/voices", headers={"Authorization": "Bearer secret"}).status_code == 200
        assert client.get("/api/voices", headers={"X-API-Key": "secret"}).status_code == 200
        redirect = client.get("/reader", follow_redirects=False)
        assert redirect.status_code == 303 and redirect.headers["location"] == "/login?next=/reader"


def test_stream_epoch_and_lease_is_shortened_after_completion(monkeypatch, tmp_path: Path) -> None:
    module = load_app_module()
    app = _fake_app(module, tmp_path, monkeypatch)
    _FakeRuntime.requests = []
    with TestClient(app) as client:
        epoch = client.get("/api/playback/status").json()["playback_epoch"]
        stale = client.post("/api/generate-stream/start", data={
            "text": "测试", "use_service_settings": "0", "expected_playback_epoch": str(epoch - 1),
        })
        assert stale.status_code == 409
        started = client.post("/api/generate-stream/start", data={
            "text": "Hello there, reader.", "use_service_settings": "0", "language": "",
        })
        assert started.status_code == 200
        job_id = started.json()["job_id"]
        with client.stream("GET", f"/api/generate-stream/{job_id}/audio") as response:
            assert response.headers["x-playback-epoch"] == str(epoch)
            payload = b"".join(response.iter_raw())
        assert payload == b"\x02\x00" * 2400
        playback = client.get("/api/playback/status").json()
        assert playback["active_job_id"] == job_id
        # 0.1 s of audio delivered: lease is now remaining audible time + 10 s,
        # not the 900 s generation lease.
        assert playback["lease_expires_in_seconds"] <= 10.2
        assert _FakeRuntime.requests[-1].language == "English"
        client.post("/api/generate-stream/start", data={
            "text": "你好", "use_service_settings": "0", "language": "japanese",
        })
        deadline = time.monotonic() + 2
        while len(_FakeRuntime.requests) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert _FakeRuntime.requests[-1].language == "Japanese"


def test_coordinator_shorten_lease_never_extends() -> None:
    coordinator = PlaybackCoordinator()
    assert coordinator.acquire("job", "internal", lease_timeout=900.0)
    assert coordinator.playback_epoch == coordinator.status()["playback_epoch"]
    assert coordinator.shorten_lease("job", 5.0)
    assert coordinator.status()["lease_expires_in_seconds"] <= 5.0
    assert coordinator.shorten_lease("job", 100.0)
    assert coordinator.status()["lease_expires_in_seconds"] <= 5.0
    assert not coordinator.shorten_lease("other", 1.0)
    assert coordinator.shorten_lease("job", 0.0)
    assert coordinator.status()["active_job_id"] is None


# --------------------------------------------------------------------------
# 9. Worker language passthrough and quality guards


def _language_model():
    calls = []

    class Model:
        def _resolve_clone_reference(self, **_kwargs):
            return {}, 1.0, {}

        def _stream_runtime(self, **kwargs):
            calls.append(kwargs)
            yield np.zeros(24000, dtype=np.float32), 24000, {}

    return Model(), calls


@pytest.mark.parametrize(
    ("requested", "text", "expected"),
    [
        ("English", "测试", "English"),
        ("korean", "测试", "Korean"),
        ("", "这是中文", "Chinese"),
        ("auto", "This is English", "English"),
        (None, "これはテストです", "Japanese"),
        ("klingon", "测试", "auto"),
    ],
)
def test_worker_honours_requested_language(tmp_path: Path, requested, text, expected) -> None:
    module = load_worker_module()
    model, calls = _language_model()
    state = make_worker_state(module, model)
    reference = tmp_path / "ref.wav"
    reference.write_bytes(b"x")
    payload = {"text": text, "ref_audio": str(reference), "output_dir": str(tmp_path), "seed": 5}
    if requested is not None:
        payload["language"] = requested
    events = list(state.generate(payload))
    assert calls[0]["lang"] == expected
    assert events[0]["data"]["language"] == expected


def test_runaway_budget_formula() -> None:
    module = load_worker_module()
    assert module.runaway_budget_seconds(1) == 6.0
    assert module.runaway_budget_seconds(4) == 6.0
    assert module.runaway_budget_seconds(10) == 9.0
    assert module.runaway_budget_seconds(100) == 54.0
    # An explicit per-request frame budget (document blocks) is never cut short by the guard.
    assert module.runaway_budget_seconds(10, 2048) == 9.0
    assert module.runaway_budget_seconds(10, 125) == 11.0
    assert module.runaway_budget_seconds(10, 250) == 21.0


def test_streaming_runaway_on_short_text_fails_instead_of_splicing(tmp_path: Path) -> None:
    module = load_worker_module()

    class Model:
        def _resolve_clone_reference(self, **_kwargs):
            return {}, 1.0, {}

        def _stream_runtime(self, **_kwargs):
            for _ in range(10):  # 10 s for a 2-char text: budget is 6 s
                yield np.zeros(24000, dtype=np.float32), 24000, {}

    state = make_worker_state(module, Model())
    reference = tmp_path / "ref.wav"
    reference.write_bytes(b"x")
    events = []
    with pytest.raises(RuntimeError, match="失控"):
        for event in state.generate({"text": "测试", "ref_audio": str(reference),
                                     "output_dir": str(tmp_path), "seed": 1}):
            events.append(event)
    assert not any(event["type"] == "result" for event in events)


def test_non_streaming_early_end_retries_once_with_new_seed(tmp_path: Path) -> None:
    module = load_worker_module()
    module._retry_seed = lambda _previous: 99
    seeds = []

    class Model:
        def _resolve_clone_reference(self, **_kwargs):
            return {}, 1.0, {}

        def _stream_runtime(self, **kwargs):
            seeds.append(kwargs["seed"])
            samples = 2400 if kwargs["seed"] == 1 else 24000 * 3  # 0.1 s, then 3 s
            yield np.zeros(samples, dtype=np.float32), 24000, {}

    state = make_worker_state(module, Model())
    reference = tmp_path / "ref.wav"
    reference.write_bytes(b"x")
    text = "这是一个用于检测提前结束的句子"
    events = list(state.generate({"text": text, "ref_audio": str(reference), "output_dir": str(tmp_path),
                                  "seed": 1, "stream_audio": False}))
    assert seeds == [1, 99]
    audio = [event for event in events if event["type"] == "audio"]
    assert len(audio) == 1 and audio[0]["data"]["samples"] == 24000 * 3
    metadata = [event for event in events if event["type"] == "result"][0]["data"]["metadata"]
    assert metadata["early_end_retries"] == 1 and metadata["effective_seed"] == 99

    seeds.clear()
    list(state.generate({"text": "短句", "ref_audio": str(reference), "output_dir": str(tmp_path),
                         "seed": 1, "stream_audio": False}))
    assert seeds == [1]  # short texts are never second-guessed


# --------------------------------------------------------------------------
# 8/14. Job manifests: lazy boot, persisted only when changed, atomic writes


def _write_manifest(root: Path, job_id: str, state: str, **extra) -> Path:
    path = root / f"{job_id}.json"
    status = {"job_id": job_id, "state": state, "updated_at": 1.0, **extra}
    result = {"audio_path": "/tmp/x.wav"} if state == "finished" else None
    path.write_text(json.dumps({"status": status, "result": result}), encoding="utf-8")
    os.utime(path, (1_000_000, 1_000_000))
    return path


def test_job_manager_boot_only_rewrites_changed_manifests(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "jobs"
    root.mkdir()
    finished = [_write_manifest(root, f"{index:032x}", "finished") for index in range(1, 6)]
    running = _write_manifest(root, "f" * 32, "running")
    monkeypatch.setenv("QWEN_TTS_JOB_CACHE_LIMIT", "2")
    manager = StreamingJobManager(root, playback_epoch=77)
    assert all(path.stat().st_mtime == 1_000_000 for path in finished)
    assert json.loads(running.read_text(encoding="utf-8"))["status"]["state"] == "interrupted"
    assert not list(root.glob("*.tmp")) and not list(root.glob(".*.tmp"))
    with manager._lock:
        loaded = len(manager._jobs)
    assert loaded <= 3
    lazily = manager.get(f"{1:032x}").snapshot()
    assert lazily["playback_epoch"] == 77 and lazily["stream_expired"] is True


def test_atomic_json_writer_uses_unique_temp_names(tmp_path: Path, monkeypatch) -> None:
    import presets

    target = tmp_path / "manifest.json"
    seen = []
    original_replace = os.replace

    def spy_replace(source, destination):
        seen.append(Path(source).name)
        return original_replace(source, destination)

    fsyncs = []
    monkeypatch.setattr(presets.os, "replace", spy_replace)
    monkeypatch.setattr(presets.os, "fsync", lambda fd: fsyncs.append(fd))
    presets.atomic_write_json(target, {"a": 1})
    presets.atomic_write_json(target, {"a": 2})
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 2}
    assert len(set(seen)) == 2 and all(name.endswith(".tmp") for name in seen)
    assert len(fsyncs) == 2
    assert [path.name for path in tmp_path.iterdir()] == ["manifest.json"]


# --------------------------------------------------------------------------
# 10/11/12. Performance profile cache, STT upload streaming, storage walk


def test_performance_profile_is_cached_by_mtime(tmp_path: Path, monkeypatch) -> None:
    from performance_tuning import PerformanceTuningStore

    store = PerformanceTuningStore(tmp_path / "perf.json")
    store.save_profile("qwen_0_6b", {"recommendation": {"block_parallel": 2}})
    reads = []
    original = Path.read_text

    def counting_read_text(self, *args, **kwargs):
        if self.name == "perf.json":
            reads.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", counting_read_text)
    for _ in range(5):
        assert store.recommendation("qwen_0_6b")["block_parallel"] == 2
    assert len(reads) <= 1
    store.profile("qwen_0_6b")["recommendation"]["block_parallel"] = 1  # copies are private
    assert store.recommendation("qwen_0_6b")["block_parallel"] == 2


def test_stt_transcribe_streams_file_uploads(monkeypatch) -> None:
    runtime = stt_runtime.WhisperCppRuntime(binary="whisper-fake", model="model-fake")
    monkeypatch.setattr(runtime, "start", lambda **_kwargs: None)
    audio = os.urandom(3 * 1024 * 1024 + 17)
    captured = {}

    class Response:
        headers = types.SimpleNamespace(get_content_type=lambda: "application/json")

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return json.dumps({"text": "ok"}).encode()

    def fake_urlopen(request, timeout):
        assert not isinstance(request.data, (bytes, bytearray))
        captured["body"] = b"".join(request.data)
        captured["length"] = int(request.get_header("Content-length"))
        return Response()

    monkeypatch.setattr(stt_runtime.urllib.request, "urlopen", fake_urlopen)
    payload, _ = runtime.transcribe(audio=io.BytesIO(audio), audio_size=len(audio), filename="a.wav")
    assert json.loads(payload) == {"text": "ok"}
    boundary = captured["body"][2:captured["body"].index(b"\r\n")].decode()
    expected, _ = runtime._multipart(audio=audio, filename="a.wav", language="auto", prompt="",
                                     response_format="json")
    expected = expected.replace(expected[2:expected.index(b"\r\n")], boundary.encode())
    assert captured["body"] == expected
    assert captured["length"] == len(expected)


def test_storage_policy_counts_files_and_reuses_settled_directories(tmp_path: Path) -> None:
    root = tmp_path / "data"
    nested = root / "a" / "b"
    nested.mkdir(parents=True)
    (root / "one.bin").write_bytes(b"x" * 100)
    (nested / "two.bin").write_bytes(b"x" * 50)
    (root / "link.bin").symlink_to(root / "one.bin")
    policy = StoragePolicy((root, nested))
    assert policy.status()["used_bytes"] == 150
    (nested / "three.bin").write_bytes(b"x" * 25)
    policy.refresh()
    assert policy.status()["used_bytes"] == 175
    old = time.time() - 3600
    for directory in (root, root / "a", nested):
        os.utime(directory, (old, old))
    policy.refresh()
    assert policy.status()["used_bytes"] == 175
