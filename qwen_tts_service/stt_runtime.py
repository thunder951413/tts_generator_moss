"""Persistent whisper.cpp Metal server used by the unified speech service."""

from __future__ import annotations

import json
import logging
import mimetypes
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Callable


class STTUnavailableError(RuntimeError):
    """Raised when the configured STT runtime cannot be used."""


class WhisperCppRuntime:
    def __init__(
        self,
        *,
        binary: str | Path,
        model: str | Path,
        host: str = "127.0.0.1",
        port: int = 7890,
        threads: int = 8,
        log_path: str | Path | None = None,
    ) -> None:
        self.binary = Path(binary).expanduser()
        self.model = Path(model).expanduser()
        self.host = str(host)
        self.port = int(port)
        self.threads = max(1, int(threads))
        self.log_path = Path(log_path) if log_path else None
        self._process: subprocess.Popen[bytes] | None = None
        self._log_handle: Any = None
        self._condition = threading.Condition()
        # A transcription owns the resident whisper.cpp HTTP server for the
        # duration of its request.  whisper-server is single-request oriented
        # in this deployment, so serializing here also keeps each response
        # paired with the upload that produced it.
        self._transcribe_lock = threading.Lock()
        self._starting = False
        self._retiring = False
        self._closing = False
        self._close_owner = False
        self._close_count = 0
        self._lifecycle_epoch = 0
        self._started_at: float | None = None
        self._ready_at: float | None = None
        self._error: str | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def _resolve_binary(self) -> str:
        if self.binary.is_file() and os.access(self.binary, os.X_OK):
            return str(self.binary)
        from shutil import which

        resolved = which(str(self.binary))
        if resolved:
            return resolved
        raise STTUnavailableError(
            f"whisper-server not found: {self.binary}. Install it with: brew install whisper-cpp"
        )

    def _subprocess_environment(self) -> dict[str, str]:
        """Keep whisper.cpp media conversion working from a sandboxed macOS app."""
        environment = os.environ.copy()
        existing = environment.get("PATH", "")
        candidates = [
            str(self.binary.expanduser().parent),
            "/opt/homebrew/bin",
            "/usr/local/bin",
            *existing.split(os.pathsep),
        ]
        environment["PATH"] = os.pathsep.join(dict.fromkeys(path for path in candidates if path))
        return environment

    def available(self) -> tuple[bool, str | None]:
        try:
            self._resolve_binary()
        except STTUnavailableError as exc:
            return False, str(exc)
        if not self.model.is_file():
            return False, f"Whisper model not found: {self.model}"
        return True, None

    def _is_ready(self, *, timeout: float = 0.5) -> bool:
        try:
            with urllib.request.urlopen(f"{self.base_url}/", timeout=timeout) as response:
                return 200 <= int(response.status) < 500
        except (OSError, urllib.error.URLError):
            return False

    @staticmethod
    def _terminate_process(process: subprocess.Popen[bytes]) -> None:
        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)

    @staticmethod
    def _close_log_handle(handle: Any) -> None:
        if handle is not None:
            handle.close()

    def _fail_start(
        self,
        process: subprocess.Popen[bytes],
        epoch: int,
        error: str,
    ) -> None:
        """Retire and clear only the startup attempt that reported the failure."""
        log_handle = None
        with self._condition:
            if self._lifecycle_epoch == epoch and self._process is process:
                self._starting = False
                self._retiring = True
                self._error = error
                log_handle = self._log_handle
            else:
                return
        retired = False
        try:
            self._terminate_process(process)
            retired = True
        finally:
            try:
                self._close_log_handle(log_handle)
            finally:
                with self._condition:
                    if (
                        retired
                        and self._lifecycle_epoch == epoch
                        and self._process is process
                    ):
                        self._process = None
                        self._log_handle = None
                    self._retiring = False
                    self._condition.notify_all()

    def start(self, *, timeout: float = 30.0) -> None:
        """Start one worker, sharing a single startup attempt among callers."""
        process: subprocess.Popen[bytes]
        epoch: int
        with self._condition:
            request_epoch = self._lifecycle_epoch
        while True:
            stale_process = None
            stale_log_handle = None
            with self._condition:
                if self._lifecycle_epoch != request_epoch:
                    raise STTUnavailableError("STT runtime was stopped during startup")
                while self._closing and self._lifecycle_epoch == request_epoch:
                    self._condition.wait()
                if self._lifecycle_epoch != request_epoch:
                    raise STTUnavailableError("STT runtime was stopped during startup")
                current = self._process
                if current is not None and current.poll() is None and self._is_ready():
                    return
                if self._starting or self._retiring:
                    waited_for_start = self._starting
                    waiting_epoch = self._lifecycle_epoch
                    while (
                        (self._starting or self._retiring)
                        and self._lifecycle_epoch == waiting_epoch
                    ):
                        self._condition.wait()
                    if self._lifecycle_epoch != waiting_epoch:
                        raise STTUnavailableError("STT runtime was stopped during startup")
                    current = self._process
                    if current is not None and current.poll() is None and self._is_ready():
                        return
                    if waited_for_start:
                        raise STTUnavailableError(self._error or "STT runtime failed to start")
                    continue
                if current is not None:
                    # A previous, completed startup left an unhealthy server.
                    # Retire it before binding a replacement to the same port.
                    self._retiring = True
                    stale_process = current
                    stale_log_handle = self._log_handle
                else:
                    available, error = self.available()
                    if not available:
                        self._error = error
                        raise STTUnavailableError(error or "STT runtime is unavailable")
                    binary = self._resolve_binary()
                    log_handle = None
                    try:
                        if self.log_path:
                            self.log_path.parent.mkdir(parents=True, exist_ok=True)
                            log_handle = self.log_path.open("ab")
                        self._starting = True
                        epoch = self._lifecycle_epoch
                        self._started_at = time.time()
                        self._ready_at = None
                        self._error = None
                        process = subprocess.Popen(
                            [
                                binary, "-m", str(self.model.resolve()), "--host", self.host,
                                "--port", str(self.port), "--convert", "--language", "auto",
                                "--threads", str(self.threads), "--flash-attn",
                            ],
                            stdin=subprocess.DEVNULL,
                            stdout=log_handle or subprocess.DEVNULL,
                            stderr=subprocess.STDOUT,
                            cwd=str(self.model.resolve().parent),
                            env=self._subprocess_environment(),
                        )
                    except Exception:
                        self._starting = False
                        self._error = "failed to launch whisper-server"
                        self._condition.notify_all()
                        self._close_log_handle(log_handle)
                        raise
                    self._process = process
                    self._log_handle = log_handle
                    break
            if stale_process is not None:
                retired = False
                try:
                    self._terminate_process(stale_process)
                    retired = True
                finally:
                    try:
                        self._close_log_handle(stale_log_handle)
                    finally:
                        with self._condition:
                            if retired and self._process is stale_process:
                                self._process = None
                                self._log_handle = None
                            self._retiring = False
                            self._condition.notify_all()

        deadline = time.monotonic() + max(1.0, float(timeout))
        while time.monotonic() < deadline:
            with self._condition:
                if self._lifecycle_epoch != epoch or self._process is not process:
                    raise STTUnavailableError("STT runtime was stopped during startup")
            code = process.poll()
            if code is not None:
                error = f"whisper-server exited during startup with code {code}"
                self._fail_start(process, epoch, error)
                raise STTUnavailableError(error)
            if self._is_ready():
                with self._condition:
                    if self._lifecycle_epoch != epoch or self._process is not process:
                        raise STTUnavailableError("STT runtime was stopped during startup")
                    self._ready_at = time.time()
                    self._starting = False
                    self._condition.notify_all()
                return
            time.sleep(0.1)
        error = f"whisper-server did not become ready within {timeout:.0f}s"
        self._fail_start(process, epoch, error)
        raise STTUnavailableError(error)

    def close(self) -> None:
        with self._condition:
            self._lifecycle_epoch += 1
            self._close_count += 1
            self._closing = True
            self._condition.notify_all()
            while self._close_owner:
                self._condition.wait()
            self._close_owner = True
            while self._retiring:
                self._condition.wait()
            process = self._process
            log_handle = self._log_handle
            self._starting = False
            self._retiring = process is not None
            self._ready_at = None
        retired = process is None
        try:
            if process is not None:
                self._terminate_process(process)
                retired = True
        finally:
            try:
                self._close_log_handle(log_handle)
            finally:
                with self._condition:
                    if retired and self._process is process:
                        self._process = None
                        self._log_handle = None
                    self._retiring = False
                    self._close_owner = False
                    self._close_count = max(0, self._close_count - 1)
                    self._closing = self._close_count > 0
                    self._condition.notify_all()

    def status(self) -> dict[str, Any]:
        available, availability_error = self.available()
        with self._condition:
            process = self._process
        running = process is not None and process.poll() is None
        ready = bool(running and self._is_ready())
        return {
            "state": "ready" if ready else ("stopped" if available else "unavailable"),
            "ready": ready,
            "available": available,
            "error": self._error or availability_error,
            "backend": "whisper.cpp",
            "device": "metal",
            "model": str(self.model),
            "model_size_bytes": self.model.stat().st_size if self.model.is_file() else None,
            "pid": process.pid if running else None,
            "host": self.host,
            "port": self.port,
            "started_at": self._started_at,
            "ready_at": self._ready_at,
        }

    @staticmethod
    def _multipart(
        *,
        audio: bytes,
        filename: str,
        language: str,
        prompt: str,
        response_format: str,
    ) -> tuple[bytes, str]:
        boundary = f"----qwen-speech-{uuid.uuid4().hex}"
        parts: list[bytes] = []

        def add_field(name: str, value: str) -> None:
            parts.extend(
                [
                    f"--{boundary}\r\n".encode(),
                    f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                    value.encode("utf-8"),
                    b"\r\n",
                ]
            )

        safe_filename = Path(filename or "audio.wav").name.replace('"', "")
        content_type = mimetypes.guess_type(safe_filename)[0] or "application/octet-stream"
        parts.extend(
            [
                f"--{boundary}\r\n".encode(),
                (
                    f'Content-Disposition: form-data; name="file"; filename="{safe_filename}"\r\n'
                ).encode(),
                f"Content-Type: {content_type}\r\n\r\n".encode(),
                audio,
                b"\r\n",
            ]
        )
        add_field("response_format", response_format)
        add_field("language", language or "auto")
        if prompt:
            add_field("prompt", prompt)
        parts.append(f"--{boundary}--\r\n".encode())
        return b"".join(parts), boundary

    def transcribe(
        self,
        *,
        audio: bytes,
        filename: str,
        language: str = "auto",
        prompt: str = "",
        response_format: str = "json",
        timeout: float = 600.0,
        cancelled: Callable[[], bool] | None = None,
    ) -> tuple[bytes, str]:
        if not audio:
            raise ValueError("audio file is empty")
        if cancelled and cancelled():
            raise STTUnavailableError("STT transcription was cancelled")
        with self._condition:
            requested_epoch = self._lifecycle_epoch
        # Do not let a disconnected/force-stopped request wait behind a long
        # transcription merely because ``Lock.acquire`` has no cancellation.
        while not self._transcribe_lock.acquire(timeout=0.1):
            if cancelled and cancelled():
                raise STTUnavailableError("STT transcription was cancelled")
        try:
            with self._condition:
                if self._lifecycle_epoch != requested_epoch:
                    raise STTUnavailableError("STT transcription was cancelled because the runtime stopped")
            if cancelled and cancelled():
                raise STTUnavailableError("STT transcription was cancelled")
            self.start()
            with self._condition:
                if self._lifecycle_epoch != requested_epoch:
                    raise STTUnavailableError("STT transcription was cancelled because the runtime stopped")
            if cancelled and cancelled():
                raise STTUnavailableError("STT transcription was cancelled")
            whisper_format = "verbose_json" if response_format == "verbose_json" else response_format
            body, boundary = self._multipart(
                audio=audio, filename=filename, language=language, prompt=prompt,
                response_format=whisper_format,
            )
            request = urllib.request.Request(
                f"{self.base_url}/inference", data=body, method="POST",
                headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            )
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    payload = response.read()
                    content_type = response.headers.get_content_type()
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"whisper.cpp rejected the audio: {detail}") from exc
            with self._condition:
                if self._lifecycle_epoch != requested_epoch:
                    raise STTUnavailableError("STT transcription was cancelled because the runtime stopped")
            if cancelled and cancelled():
                raise STTUnavailableError("STT transcription was cancelled")
            if response_format in {"json", "verbose_json"}:
                parsed = json.loads(payload)
                if response_format == "json":
                    payload = json.dumps(
                        {"text": str(parsed.get("text") or "").strip()},
                        ensure_ascii=False,
                    ).encode("utf-8")
                else:
                    payload = json.dumps(parsed, ensure_ascii=False).encode("utf-8")
                content_type = "application/json"
            return payload, content_type
        finally:
            self._transcribe_lock.release()


__all__ = ["STTUnavailableError", "WhisperCppRuntime"]
