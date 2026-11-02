from __future__ import annotations

import json
import queue
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "qwen_tts_service"))
from qwen_protocol import StreamingRequest
from qwen_runtime import QwenWorkerRuntime
from worker_protocol import MAX_WORKER_REQUEST_BYTES, WORKER_TOKEN_HEADER
from test_macos_qwen_service import load_worker_module, make_worker_state


class Model:
    def _resolve_clone_reference(self, **_kwargs):
        return {"ref_spk_emb": np.zeros(4, dtype=np.float32)}, 1.0, {}

    def _stream_runtime(self, **_kwargs):
        yield np.zeros(240, dtype=np.float32), 24000, {}


def test_worker_requires_private_token_and_enforces_paths(tmp_path):
    module = load_worker_module()
    state = make_worker_state(module, Model())
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    reference = allowed / "reference.wav"
    reference.write_bytes(b"test")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "reference.wav").write_bytes(b"test")
    symlink = allowed / "link.wav"
    symlink.symlink_to(outside / "reference.wav")
    state.reference_roots = state.output_roots = (allowed,)
    state.generation_lock = threading.Lock()
    server = module.ThreadingHTTPServer(("127.0.0.1", 0), module.WorkerHandler)
    server.worker_state = state
    server.worker_token = "private-test-token"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def post(payload, *, token=None, content_type="application/json", length=None):
        body = json.dumps(payload).encode()
        headers = {"Content-Type": content_type}
        if token is not None:
            headers[WORKER_TOKEN_HEADER] = token
        if length is not None:
            headers["Content-Length"] = str(length)
        request = urllib.request.Request(
            f"http://127.0.0.1:{server.server_address[1]}/generate",
            data=body, headers=headers,
        )
        try:
            with urllib.request.urlopen(request, timeout=3) as response:
                return response.status, response.read().decode()
        except urllib.error.HTTPError as response:
            return response.code, response.read().decode()

    payload = {"text": "测试", "ref_audio": str(reference), "output_dir": str(allowed), "seed": 1}
    try:
        assert post(payload)[0] == 401
        assert post(payload, token="incorrect")[0] == 401
        assert post(payload, token=server.worker_token, content_type="text/plain")[0] == 415
        assert post(payload, token=server.worker_token, length=MAX_WORKER_REQUEST_BYTES + 1)[0] == 413
        for changes in ({"output_dir": str(outside)}, {"run_id": "../escaped"}, {"run_id": str(outside)}, {"ref_audio": str(symlink)}):
            code, raw = post(payload | changes, token=server.worker_token)
            assert code == 200
            assert json.loads(raw.splitlines()[-1])["type"] == "error"
            assert not list(outside.glob("**/generated.wav"))
        code, raw = post(payload, token=server.worker_token)
        assert code == 200
        assert json.loads(raw.splitlines()[-1])["type"] == "result"
        assert list(allowed.glob("*/generated.wav"))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.parametrize("stream_audio", [True, False])
def test_gradual_runaway_never_splices_attempts(tmp_path, stream_audio):
    module = load_worker_module()
    module.RUNAWAY_MIN_BUDGET_SECONDS = 2.0
    module.RUNAWAY_SECONDS_PER_CHAR = 0.0
    module.RUNAWAY_BASE_SECONDS = 0.0
    module._retry_seed = lambda _seed: 777777
    seeds = []

    class GradualModel(Model):
        def _stream_runtime(self, **kwargs):
            seeds.append(kwargs["seed"])
            if kwargs["seed"] == 424242:
                for _ in range(4):
                    yield np.zeros(24000, dtype=np.float32), 24000, {}
            else:
                yield np.zeros(19200, dtype=np.float32), 24000, {}

    state = make_worker_state(module, GradualModel())
    reference = tmp_path / "ref.wav"
    reference.write_bytes(b"test")
    events = []
    generator = state.generate({"text": "测试", "ref_audio": str(reference),
                                "output_dir": str(tmp_path), "seed": 424242, "stream_audio": stream_audio})
    if stream_audio:
        with pytest.raises(RuntimeError, match="失控"):
            events.extend(generator)
        assert seeds == [424242]
        assert not any(event["type"] == "result" for event in events)
    else:
        events.extend(generator)
        assert seeds == [424242, 777777]
        assert sum(e["data"]["samples"] for e in events if e["type"] == "audio") == 19200
        assert events[-1]["data"]["metadata"]["duration_seconds"] == 0.8


def make_pool(monkeypatch, tmp_path, *, lanes=1):
    class Worker:
        def __init__(self, lane):
            self.lane_index = lane
            self.closed = False
            self.process = SimpleNamespace(poll=lambda: None)

        def health(self):
            return {"ok": True}

        def generate(self, _payload, **_kwargs):
            yield {"type": "result", "data": {"metadata": {}}}

        def close(self):
            self.closed = True

    monkeypatch.setattr(QwenWorkerRuntime, "_create_worker", lambda self, lane: Worker(lane))
    runtime = QwenWorkerRuntime(profile_id="qwen_0_6b", model_dir="fake", python_executable=sys.executable,
                                worker_script=__file__, lanes=lanes, base_port=0, log_dir=tmp_path)
    return runtime


def test_all_failed_restarts_release_waiter_and_publish_error(monkeypatch, tmp_path):
    runtime = make_pool(monkeypatch, tmp_path)
    runtime._workers.clear()
    runtime._available = queue.Queue()
    attempts = []
    def fail(lane):
        attempts.append(lane)
        raise RuntimeError("failed test load")
    runtime._create_worker = fail
    import qwen_runtime
    monkeypatch.setattr(qwen_runtime, "time", SimpleNamespace(monotonic=time.monotonic, sleep=lambda _seconds: None))
    runtime._restart_lane(0)
    assert len(attempts) == 3
    assert runtime.pool_status()["state"] == "error"
    with pytest.raises(RuntimeError, match="recovery failed"):
        next(runtime.synthesize(StreamingRequest(text="test", prompt_audio_path=str(tmp_path / "ref.wav")), output_dir=tmp_path))
    runtime.close()


def test_partial_restart_failure_preserves_healthy_lane(monkeypatch, tmp_path):
    runtime = make_pool(monkeypatch, tmp_path, lanes=2)
    old = runtime._workers.pop()
    runtime._available = queue.Queue()
    runtime._available.put(runtime._workers[0])
    old.close()
    def fail(_lane):
        raise RuntimeError("failed second lane")
    runtime._create_worker = fail
    import qwen_runtime
    monkeypatch.setattr(qwen_runtime, "time", SimpleNamespace(monotonic=time.monotonic, sleep=lambda _seconds: None))
    runtime._restart_lane(1)
    assert runtime.pool_status()["state"] == "degraded"
    events = list(runtime.synthesize(StreamingRequest(text="test", prompt_audio_path=str(tmp_path / "ref.wav")), output_dir=tmp_path))
    assert events[-1].type == "result"
    runtime.close()


def test_worker_wait_can_be_cancelled(monkeypatch, tmp_path):
    runtime = make_pool(monkeypatch, tmp_path)
    runtime._available.get_nowait()  # Another job occupies the only worker.
    cancelled = threading.Event()
    errors = []
    def wait():
        try:
            next(runtime.synthesize(StreamingRequest(text="test", prompt_audio_path=str(tmp_path / "ref.wav"), cancelled=cancelled.is_set), output_dir=tmp_path))
        except RuntimeError as exc:
            errors.append(str(exc))
    thread = threading.Thread(target=wait)
    thread.start()
    cancelled.set()
    thread.join(timeout=1)
    assert not thread.is_alive()
    assert errors and "cancelled" in errors[0]
    runtime.close()


def test_cancel_wakes_blocked_worker_read_and_retires_decoder():
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from qwen_runtime import QwenWorkerClient
    held = threading.Event()
    release = threading.Event()
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            self.send_response(200);self.end_headers();self.wfile.flush()
            held.set();release.wait(3)
        def log_message(self,*_args):pass
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    server.daemon_threads=True
    threading.Thread(target=server.serve_forever,daemon=True).start()
    client=QwenWorkerClient.__new__(QwenWorkerClient)
    client.port=server.server_address[1];client._request_lock=threading.Lock();client._token='test-private'
    retired=threading.Event();client.close=retired.set
    # This fake worker never reports idle (no /health), so after the cancel
    # grace period the still-busy lane must be retired.
    client.cancel_grace_seconds=0.2
    cancelled=threading.Event();errors=[]
    def generate():
        try:list(client.generate({'text':'test'},cancelled=cancelled.is_set))
        except Exception as error:errors.append(error)
    thread=threading.Thread(target=generate)
    thread.start()
    try:
        assert held.wait(1)
        began=time.monotonic();cancelled.set();thread.join(1)
        assert not thread.is_alive()
        assert time.monotonic()-began < 1
        assert retired.is_set()
    finally:
        release.set();server.shutdown();server.server_close()


def test_empty_model_output_is_an_error_not_a_successful_silent_wav(tmp_path):
    module=load_worker_module()
    class EmptyModel(Model):
        def _stream_runtime(self,**_kwargs):
            if False:
                yield None
    state=make_worker_state(module,EmptyModel())
    state.output_roots=state.reference_roots=(tmp_path,)
    reference=tmp_path/'reference.wav';reference.write_bytes(b'audio')
    with pytest.raises(RuntimeError,match='未生成有效音频'):
        list(state.generate({'text':'test','seed':1234,'ref_audio':str(reference),'output_dir':str(tmp_path)}))
    assert not list(tmp_path.rglob('generated.wav'))
