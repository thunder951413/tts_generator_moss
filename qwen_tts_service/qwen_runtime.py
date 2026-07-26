# coding=utf-8
"""Main-service controller for isolated Faster Qwen3-TTS workers."""

from __future__ import annotations

import base64
import http.client
import json
import logging
import os
import queue
import subprocess
import sys
import threading
import time
import urllib.request
from collections import OrderedDict
from pathlib import Path
from typing import Any, Generator

import numpy as np
import torch

from qwen_protocol import StreamingEvent, StreamingRequest


LOG = logging.getLogger("qwen-runtime")


class QwenWorkerClient:
    def __init__(
        self,
        *,
        python_executable: str | Path,
        worker_script: str | Path,
        model_dir: str | Path,
        backend: str,
        quant: str,
        library_path: str | Path | None,
        profile_id: str,
        lane_index: int,
        port: int,
        log_dir: str | Path,
        startup_timeout: float = 300.0,
    ) -> None:
        self.profile_id = profile_id
        self.lane_index = int(lane_index)
        self.port = int(port)
        self._request_lock = threading.Lock()
        log_dir = Path(log_dir).resolve()
        log_dir.mkdir(parents=True, exist_ok=True)
        self.stdout_path = log_dir / f"{profile_id}_lane_{lane_index}.out.log"
        self.stderr_path = log_dir / f"{profile_id}_lane_{lane_index}.err.log"
        self._stdout = open(self.stdout_path, "ab", buffering=0)
        self._stderr = open(self.stderr_path, "ab", buffering=0)
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.process = subprocess.Popen(
            [
                str(Path(python_executable).resolve()),
                str(Path(worker_script).resolve()),
                "--model",
                str(model_dir),
                "--profile-id",
                profile_id,
                "--backend",
                backend,
                "--quant",
                quant,
                "--port",
                str(self.port),
            ],
            cwd=str(Path(worker_script).resolve().parent),
            stdin=subprocess.DEVNULL,
            stdout=self._stdout,
            stderr=self._stderr,
            creationflags=creationflags,
            env={
                **dict(os.environ),
                **(
                    {"QWENTTS_CPP_LIBRARY": str(Path(library_path).expanduser().resolve())}
                    if library_path
                    else {}
                ),
            },
        )
        self._wait_until_ready(float(startup_timeout))

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def _wait_until_ready(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                tail = ""
                try:
                    tail = self.stderr_path.read_text(encoding="utf-8", errors="replace")[-4000:]
                except OSError:
                    pass
                raise RuntimeError(
                    f"Qwen worker lane {self.lane_index} exited with {self.process.returncode}: {tail}"
                )
            try:
                with urllib.request.urlopen(f"{self.base_url}/health", timeout=1.0) as response:
                    data = json.loads(response.read().decode("utf-8"))
                if data.get("ok"):
                    return
            except Exception as exc:  # noqa: BLE001
                last_error = exc
            time.sleep(0.25)
        self.close()
        raise TimeoutError(f"Qwen worker lane {self.lane_index} did not become ready: {last_error}")

    def health(self) -> dict[str, Any]:
        with urllib.request.urlopen(f"{self.base_url}/health", timeout=2.0) as response:
            return json.loads(response.read().decode("utf-8"))

    def generate(self, payload: dict[str, Any]) -> Generator[dict[str, Any], None, None]:
        with self._request_lock:
            connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=600)
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            try:
                connection.request(
                    "POST",
                    "/generate",
                    body=body,
                    headers={"Content-Type": "application/json", "Content-Length": str(len(body))},
                )
                response = connection.getresponse()
                if response.status != 200:
                    raise RuntimeError(response.read().decode("utf-8", errors="replace"))
                while True:
                    line = response.readline()
                    if not line:
                        break
                    event = json.loads(line.decode("utf-8"))
                    if event.get("type") == "error":
                        raise RuntimeError(str((event.get("data") or {}).get("error") or "Qwen generation failed"))
                    yield event
            finally:
                connection.close()

    def close(self) -> None:
        process = getattr(self, "process", None)
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        for handle_name in ("_stdout", "_stderr"):
            handle = getattr(self, handle_name, None)
            if handle is not None:
                try:
                    handle.close()
                except OSError:
                    pass


class QwenWorkerRuntime:
    """A lane pool that presents the streaming surface used by the web app."""

    runtime_family = "faster_qwen3_tts"
    sample_rate = 24000
    frame_rate = 12.0
    n_vq = 16
    attn_implementation = "ggml_metal" if sys.platform == "darwin" else "ggml"
    codec_weight_dtype = "gguf"
    codec_compute_dtype = "ggml"

    def __init__(
        self,
        *,
        profile_id: str,
        model_dir: str | Path,
        backend: str = "ggml",
        quant: str = "Q4_K_M",
        library_path: str | Path | None = None,
        python_executable: str | Path,
        worker_script: str | Path,
        lanes: int,
        base_port: int,
        log_dir: str | Path,
    ) -> None:
        self.profile_id = profile_id
        self.model_dir = str(model_dir)
        self.backend = str(backend)
        self.quant = str(quant)
        self.library_path = str(library_path or "")
        self.codec_dir = self.model_dir
        runtime_device = "mps" if sys.platform == "darwin" else "cpu"
        self.device = torch.device(runtime_device)
        self.tts_device = torch.device(runtime_device)
        self.codec_device = torch.device(runtime_device)
        self.dtype = torch.float32
        self.reference_audio_cache: OrderedDict[str, bool] = OrderedDict()
        self.reference_audio_cache_hits = 0
        self.reference_audio_cache_misses = 0
        self.reference_audio_cache_lock = threading.RLock()
        self._workers: list[QwenWorkerClient] = []
        self._available: queue.Queue[QwenWorkerClient] = queue.Queue()
        self._closed = False
        try:
            for index in range(max(1, int(lanes))):
                worker = QwenWorkerClient(
                    python_executable=python_executable,
                    worker_script=worker_script,
                    model_dir=self.model_dir,
                    backend=self.backend,
                    quant=self.quant,
                    library_path=self.library_path or None,
                    profile_id=profile_id,
                    lane_index=index,
                    port=int(base_port) + index,
                    log_dir=log_dir,
                )
                self._workers.append(worker)
                self._available.put(worker)
        except Exception:
            self.close()
            raise

    @property
    def lanes(self) -> int:
        return len(self._workers)

    def status(self) -> dict[str, Any]:
        workers = []
        for worker in self._workers:
            try:
                workers.append(worker.health())
            except Exception as exc:  # noqa: BLE001
                workers.append({"ok": False, "lane": worker.lane_index, "error": str(exc)})
        return {"lanes": self.lanes, "workers": workers}

    def synthesize(
        self,
        request: StreamingRequest,
        *,
        output_dir: str | Path,
    ) -> Generator[StreamingEvent, None, None]:
        if self._closed:
            raise RuntimeError("Qwen runtime is closed")
        worker = self._available.get()
        reference_audio_path = str(Path(request.prompt_audio_path).resolve())
        reference_key = (
            f"{reference_audio_path}|{request.qwen_reference_text}|"
            f"{request.qwen_xvec_only}"
        )
        with self.reference_audio_cache_lock:
            if reference_key in self.reference_audio_cache:
                self.reference_audio_cache_hits += 1
            else:
                self.reference_audio_cache_misses += 1
        payload = {
            "text": request.text,
            "ref_audio": reference_audio_path,
            "ref_text": request.qwen_reference_text,
            "xvec_only": request.qwen_xvec_only,
            "non_streaming_mode": request.qwen_non_streaming_mode,
            "append_silence": request.qwen_append_silence,
            "instruct": request.qwen_instruct,
            "max_new_tokens": request.max_new_frames,
            "min_new_tokens": request.qwen_min_new_tokens,
            "temperature": request.temperature,
            "top_k": request.top_k,
            "top_p": request.top_p,
            "do_sample": request.do_sample,
            "repetition_penalty": request.repetition_penalty,
            "chunk_size": request.codec_chunk_frames,
            "seed": request.seed,
            "output_dir": str(Path(output_dir).resolve()),
        }
        try:
            for raw_event in worker.generate(payload):
                event_type = str(raw_event.get("type") or "")
                data = dict(raw_event.get("data") or {})
                if event_type == "audio":
                    pcm = base64.b64decode(str(data.pop("pcm16_base64")))
                    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32767.0
                    data["waveform"] = torch.from_numpy(samples.copy()).unsqueeze(0)
                elif event_type == "result":
                    with self.reference_audio_cache_lock:
                        self.reference_audio_cache[reference_key] = True
                        while len(self.reference_audio_cache) > 64:
                            self.reference_audio_cache.popitem(last=False)
                yield StreamingEvent(event_type, data)
        finally:
            self._available.put(worker)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for worker in self._workers:
            try:
                worker.close()
            except Exception:  # noqa: BLE001
                LOG.exception("failed to close Qwen worker lane %s", worker.lane_index)
        self._workers.clear()
