# coding=utf-8
"""Main-service controller for isolated Faster Qwen3-TTS workers."""

from __future__ import annotations

import base64
import http.client
import json
import logging
import os
import queue
import secrets
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Generator

from qwen_protocol import StreamingEvent, StreamingRequest
from worker_protocol import WORKER_TOKEN_HEADER


LOG = logging.getLogger("qwen-runtime")

# After a cancelled request the worker stops at its next audio chunk (broken
# pipe for streaming, socket EOF check for complete-block generation).  Give
# it this long to report ``busy: false`` before retiring the lane.
CANCEL_GRACE_SECONDS = float(os.environ.get("QWEN_TTS_CANCEL_GRACE_SECONDS", "3.0"))
_ROTATE_LOG_BYTES = 5 * 1024 * 1024


def rotate_log_if_large(path: Path, max_bytes: int = _ROTATE_LOG_BYTES) -> None:
    """Keep append-mode subprocess logs bounded (one ``.1`` backup)."""
    try:
        if path.stat().st_size > max_bytes:
            os.replace(path, path.with_name(path.name + ".1"))
    except OSError:
        pass


def wait_until_idle(worker: Any, timeout: float) -> bool:
    """Poll a worker's /health until it is alive and not generating."""
    deadline = time.monotonic() + max(0.0, float(timeout))
    while True:
        process = getattr(worker, "process", None)
        if process is not None and process.poll() is not None:
            return False
        try:
            data = worker.health()
            if data.get("ok") and not data.get("busy", False):
                return True
        except Exception:  # noqa: BLE001
            pass
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)


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
        output_roots: tuple[str, ...] = (),
        reference_roots: tuple[str, ...] = (),
        startup_cancelled=None,
    ) -> None:
        self.profile_id = profile_id
        self.lane_index = int(lane_index)
        if int(port) == 0:
            with socket.socket() as reservation:
                reservation.bind(("127.0.0.1", 0))
                port = reservation.getsockname()[1]
        self.port = int(port)
        self._startup_cancelled = startup_cancelled
        self._request_lock = threading.Lock()
        self._token = secrets.token_urlsafe(32)
        log_dir = Path(log_dir).resolve()
        log_dir.mkdir(parents=True, exist_ok=True)
        self.stdout_path = log_dir / f"{profile_id}_lane_{lane_index}.out.log"
        self.stderr_path = log_dir / f"{profile_id}_lane_{lane_index}.err.log"
        rotate_log_if_large(self.stdout_path)
        rotate_log_if_large(self.stderr_path)
        self.cancel_grace_seconds = CANCEL_GRACE_SECONDS
        self._stdout = open(self.stdout_path, "ab", buffering=0)
        self._stderr = open(self.stderr_path, "ab", buffering=0)
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        # Keep the virtual-environment launcher path intact.  On macOS it is a
        # symlink to the Homebrew interpreter; Path.resolve() would bypass the
        # venv and start a worker without any of the project dependencies.
        python_path = os.path.abspath(os.path.expanduser(os.fspath(python_executable)))
        try:
            self.process = subprocess.Popen(
                [
                    python_path,
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
                    "QWEN_WORKER_TOKEN": self._token,
                    "QWEN_WORKER_OUTPUT_ROOTS": json.dumps(output_roots),
                    "QWEN_WORKER_REFERENCE_ROOTS": json.dumps(reference_roots),
                    **(
                        {"QWENTTS_CPP_LIBRARY": str(Path(library_path).expanduser().resolve())}
                        if library_path
                        else {}
                    ),
                },
            )
        except BaseException:
            self.close()
            raise
        try:
            self._wait_until_ready(float(startup_timeout))
        except BaseException:
            self.close()
            raise

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def _wait_until_ready(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            if self._startup_cancelled and self._startup_cancelled():
                raise RuntimeError("Qwen worker startup cancelled")
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
                with urllib.request.urlopen(self._health_request(), timeout=1.0) as response:
                    data = json.loads(response.read().decode("utf-8"))
                if data.get("ok"):
                    return
            except Exception as exc:  # noqa: BLE001
                last_error = exc
            time.sleep(0.25)
        self.close()
        raise TimeoutError(f"Qwen worker lane {self.lane_index} did not become ready: {last_error}")

    def health(self) -> dict[str, Any]:
        with urllib.request.urlopen(self._health_request(), timeout=2.0) as response:
            return json.loads(response.read().decode("utf-8"))

    def wait_until_idle(self, timeout: float) -> bool:
        return wait_until_idle(self, timeout)

    def _health_request(self) -> urllib.request.Request:
        return urllib.request.Request(f"{self.base_url}/health", headers={WORKER_TOKEN_HEADER: self._token})

    def generate(self, payload: dict[str, Any], *, cancelled=None) -> Generator[dict[str, Any], None, None]:
        with self._request_lock:
            connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=600)
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            watcher_done = threading.Event()
            watcher = None
            active_socket = None
            response = None
            finished = False
            try:
                if cancelled and cancelled():
                    raise RuntimeError("generation cancelled")
                connection.request(
                    "POST",
                    "/generate",
                    body=body,
                    headers={"Content-Type": "application/json", "Content-Length": str(len(body)), WORKER_TOKEN_HEADER: self._token},
                )
                # Retain this exact socket: HTTP/1.0 responses detach it from
                # HTTPConnection while readline may still be blocked on it.
                active_socket = connection.sock
                def watch_cancel() -> None:
                    while not watcher_done.wait(0.1):
                        if cancelled and cancelled():
                            if active_socket is not None:
                                try:
                                    active_socket.shutdown(socket.SHUT_RDWR)
                                except OSError:
                                    pass
                            return
                if cancelled is not None:
                    watcher = threading.Thread(target=watch_cancel, daemon=True)
                    watcher.start()
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
                finished = True
            finally:
                watcher_done.set()
                if not finished and active_socket is not None:
                    # The response object keeps the socket referenced, so
                    # connection.close() alone may not send FIN.  Shut it down
                    # so the worker notices at its next chunk.
                    try:
                        active_socket.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                if response is not None:
                    response.close()
                connection.close()
                if watcher is not None:
                    watcher.join(timeout=0.5)
                if cancelled and cancelled():
                    # The closed socket makes the worker stop at its next chunk
                    # (it checks for EOF even when not streaming).  Reuse the
                    # loaded model when it goes idle promptly; otherwise retire
                    # the lane so the next job never queues behind cancelled
                    # inference holding generation_lock.
                    if not self.wait_until_idle(getattr(self, "cancel_grace_seconds", CANCEL_GRACE_SECONDS)):
                        self.close()
                        LOG.warning(
                            "Qwen worker lane %s still busy after cancel; retired it",
                            getattr(self, "lane_index", "?"),
                        )

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
        output_roots: tuple[str, ...] = (),
        reference_roots: tuple[str, ...] = (),
        cancelled=None,
    ) -> None:
        self.profile_id = profile_id
        self.model_dir = str(model_dir)
        self.backend = str(backend)
        self.quant = str(quant)
        self.library_path = str(library_path or "")
        self.codec_dir = self.model_dir
        # Informational only: inference runs in the isolated worker process.
        runtime_device = "metal" if sys.platform == "darwin" else "cpu"
        self.device = runtime_device
        self.tts_device = runtime_device
        self.codec_device = runtime_device
        self.dtype = "gguf" if self.backend == "ggml" else "bfloat16"
        self._worker_lock = threading.RLock()
        self._workers: list[QwenWorkerClient] = []
        self._available: queue.Queue[QwenWorkerClient] = queue.Queue()
        self._closed = False
        self._desired_lanes = max(1, int(lanes))
        self._restart_epoch = 0
        self._recovering: dict[int, int] = {}
        self._lane_errors: dict[int, str] = {}
        self._worker_wait_timeout = max(1.0, float(os.environ.get("QWEN_TTS_WORKER_WAIT_TIMEOUT", "330")))
        self._worker_config = {
            "python_executable": python_executable,
            "worker_script": worker_script,
            "model_dir": self.model_dir,
            "backend": self.backend,
            "quant": self.quant,
            "library_path": self.library_path or None,
            "profile_id": profile_id,
            "base_port": int(base_port),
            "log_dir": log_dir,
            "output_roots": output_roots,
            "reference_roots": reference_roots,
            "startup_cancelled": cancelled,
        }
        try:
            for index in range(self._desired_lanes):
                worker = self._create_worker(index)
                self._workers.append(worker)
                self._available.put(worker)
        except Exception:
            self.close()
            raise

    @property
    def lanes(self) -> int:
        with self._worker_lock:
            return len(self._workers)

    def _create_worker(self, lane_index: int) -> QwenWorkerClient:
        config = self._worker_config
        return QwenWorkerClient(
            python_executable=config["python_executable"],
            worker_script=config["worker_script"],
            model_dir=config["model_dir"],
            backend=str(config["backend"]),
            quant=str(config["quant"]),
            library_path=config["library_path"],
            profile_id=str(config["profile_id"]),
            lane_index=lane_index,
            port=0,
            log_dir=config["log_dir"],
            output_roots=config["output_roots"],
            reference_roots=config["reference_roots"],
            startup_cancelled=config["startup_cancelled"],
        )

    @staticmethod
    def _worker_is_healthy(worker: QwenWorkerClient) -> bool:
        """A lane is reusable once it is alive and has finished generating."""
        return wait_until_idle(worker, getattr(worker, "cancel_grace_seconds", CANCEL_GRACE_SECONDS))

    def _restart_lane(self, lane_index: int, epoch: int | None = None) -> None:
        with self._worker_lock:
            epoch = self._restart_epoch if epoch is None else epoch
            if self._closed or epoch != self._restart_epoch:
                return
            self._recovering[lane_index] = epoch
        for attempt in range(1, 4):
            with self._worker_lock:
                if self._closed or epoch != self._restart_epoch or lane_index >= self._desired_lanes:
                    return
            try:
                replacement = self._create_worker(lane_index)
                with self._worker_lock:
                    if self._closed or epoch != self._restart_epoch or lane_index >= self._desired_lanes:
                        replacement.close()
                        return
                    self._workers.append(replacement)
                    self._available.put(replacement)
                    self._recovering.pop(lane_index, None)
                    self._lane_errors.pop(lane_index, None)
                LOG.info("restarted Qwen worker lane %s", lane_index)
                return
            except Exception as exc:  # noqa: BLE001
                LOG.exception(
                    "failed to restart Qwen worker lane %s (attempt %s/3)",
                    lane_index,
                    attempt,
                )
                if attempt < 3:
                    time.sleep(float(attempt))
                else:
                    with self._worker_lock:
                        if epoch == self._restart_epoch and not self._closed:
                            self._recovering.pop(lane_index, None)
                            self._lane_errors[lane_index] = str(exc)

    def _retire_worker(self, worker: QwenWorkerClient) -> None:
        with self._worker_lock:
            was_registered = worker in self._workers
            if worker in self._workers:
                self._workers.remove(worker)
            epoch = self._restart_epoch
            if not self._closed and was_registered:
                self._recovering[worker.lane_index] = epoch
        try:
            worker.close()
        except Exception:  # noqa: BLE001
            LOG.exception("failed to close unhealthy Qwen worker lane %s", worker.lane_index)
        if not self._closed and was_registered:
            threading.Thread(
                target=self._restart_lane,
                args=(worker.lane_index, epoch),
                name=f"qwen-worker-restart-{self.profile_id}-{worker.lane_index}",
                daemon=True,
            ).start()

    def status(self) -> dict[str, Any]:
        workers = []
        with self._worker_lock:
            current_workers = list(self._workers)
        for worker in current_workers:
            try:
                workers.append(worker.health())
            except Exception as exc:  # noqa: BLE001
                workers.append({"ok": False, "lane": worker.lane_index, "error": str(exc)})
        return {**self.pool_status(), "workers": workers}

    def pool_status(self) -> dict[str, Any]:
        with self._worker_lock:
            lanes = len(self._workers)
            recovering = len(self._recovering)
            error = "; ".join(f"lane {lane}: {reason}" for lane, reason in self._lane_errors.items()) or None
            state = (
                "closed" if self._closed else "recovering" if recovering and not lanes
                else "error" if not lanes else "degraded" if lanes < self._desired_lanes else "ready"
            )
            return {"state": state, "lanes": lanes, "target_lanes": self._desired_lanes,
                    "recovering_lanes": recovering, "error": error}

    def resize_lanes(self, target: int) -> int:
        target = max(1, min(2, int(target)))
        with self._worker_lock:
            if self._recovering or self._lane_errors:
                raise RuntimeError("通道正在恢复或恢复失败，请重新加载模型后调整通道数")
            current = len(self._workers)
            epoch = self._restart_epoch
        if target > current:
            created: list[QwenWorkerClient] = []
            try:
                for index in range(current, target):
                    created.append(self._create_worker(index))
            except Exception:
                for worker in created:
                    worker.close()
                raise
            with self._worker_lock:
                if self._closed or epoch != self._restart_epoch:
                    for worker in created:
                        worker.close()
                    raise RuntimeError("Qwen runtime is closed")
                self._workers.extend(created)
                self._desired_lanes = target
                for worker in created:
                    self._available.put(worker)
            return self.lanes
        if target < current:
            available: list[QwenWorkerClient] = []
            while True:
                try:
                    available.append(self._available.get_nowait())
                except queue.Empty:
                    break
            with self._worker_lock:
                if len(available) != len(self._workers):
                    for worker in available:
                        self._available.put(worker)
                    raise RuntimeError("有生成任务正在运行，暂时不能调整通道数")
                keep = sorted(available, key=lambda worker: worker.lane_index)[:target]
                retire = [worker for worker in available if worker not in keep]
                self._workers = list(keep)
                self._desired_lanes = target
                for worker in keep:
                    self._available.put(worker)
            for worker in retire:
                worker.close()
        return self.lanes

    def interrupt_all(self) -> int:
        """Terminate every active lane and rebuild the configured pool."""
        with self._worker_lock:
            if self._closed:
                return 0
            self._restart_epoch += 1
            restart_epoch = self._restart_epoch
            workers = list(self._workers)
            self._workers.clear()
            target = self._desired_lanes
            self._recovering = {index: restart_epoch for index in range(target)}
            self._lane_errors.clear()
            while True:
                try:
                    self._available.get_nowait()
                except queue.Empty:
                    break
        for worker in workers:
            try:
                worker.close()
            except Exception:  # noqa: BLE001
                LOG.exception("failed to interrupt Qwen worker lane %s", worker.lane_index)

        def rebuild() -> None:
            for index in range(target):
                self._restart_lane(index, restart_epoch)

        threading.Thread(
            target=rebuild,
            name=f"qwen-worker-force-restart-{self.profile_id}",
            daemon=True,
        ).start()
        return len(workers)

    def synthesize(
        self,
        request: StreamingRequest,
        *,
        output_dir: str | Path,
    ) -> Generator[StreamingEvent, None, None]:
        if self._closed:
            raise RuntimeError("Qwen runtime is closed")
        # Validate before borrowing a lane so malformed direct requests cannot
        # permanently consume the pool's last available worker.
        reference_audio_path = str(Path(request.prompt_audio_path).resolve())
        resolved_output_dir = str(Path(output_dir).resolve())
        deadline = time.monotonic() + self._worker_wait_timeout
        cancelled = request.cancelled
        while True:
            if self._closed:
                raise RuntimeError("Qwen runtime is closed")
            if cancelled and cancelled():
                raise RuntimeError("generation cancelled while waiting for worker")
            pool = self.pool_status()
            if pool["state"] == "error":
                raise RuntimeError(f"Qwen worker recovery failed: {pool['error'] or 'no usable lanes'}")
            if time.monotonic() >= deadline:
                raise TimeoutError("等待 Qwen worker 超时，请重新启动 TTS")
            try:
                worker = self._available.get(timeout=0.1)
                break
            except queue.Empty:
                continue
        payload = {
            "text": request.text,
            "language": request.language,
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
            "output_dir": resolved_output_dir,
            "stream_audio": request.stream_audio,
        }
        stream = worker.generate(payload, cancelled=cancelled)
        try:
            completed = False
            for raw_event in stream:
                event_type = str(raw_event.get("type") or "")
                data = dict(raw_event.get("data") or {})
                if event_type == "audio":
                    # Mono little-endian int16 exactly as the worker quantized
                    # it; consumers forward these bytes without conversion.
                    data["pcm16"] = base64.b64decode(str(data.pop("pcm16_base64")))
                    data.setdefault("channels", 1)
                elif event_type == "result":
                    completed = True
                yield StreamingEvent(event_type, data)
            if not completed:
                raise RuntimeError("Qwen worker stream ended without a complete result")
        finally:
            # Close the worker request first (an abandoned consumer otherwise
            # keeps the HTTP stream open until garbage collection) so the
            # worker sees the disconnect before we judge whether it is idle.
            stream.close()
            if self._closed:
                worker.close()
            elif self._worker_is_healthy(worker):
                self._available.put(worker)
            else:
                LOG.error(
                    "Qwen worker lane %s became unhealthy and will be restarted",
                    worker.lane_index,
                )
                self._retire_worker(worker)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._restart_epoch += 1
        with self._worker_lock:
            workers = list(self._workers)
            self._workers.clear()
            self._recovering.clear()
            self._lane_errors.clear()
        for worker in workers:
            try:
                worker.close()
            except Exception:  # noqa: BLE001
                LOG.exception("failed to close Qwen worker lane %s", worker.lane_index)
