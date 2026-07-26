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
from typing import Any


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
        self._lock = threading.Lock()
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

    def start(self, *, timeout: float = 30.0) -> None:
        with self._lock:
            if self._process is not None and self._process.poll() is None and self._is_ready():
                return
            available, error = self.available()
            if not available:
                self._error = error
                raise STTUnavailableError(error or "STT runtime is unavailable")
            binary = self._resolve_binary()
            if self.log_path:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                self._log_handle = self.log_path.open("ab")
            command = [
                binary,
                "-m",
                str(self.model.resolve()),
                "--host",
                self.host,
                "--port",
                str(self.port),
                "--convert",
                "--language",
                "auto",
                "--threads",
                str(self.threads),
                "--flash-attn",
            ]
            self._started_at = time.time()
            self._ready_at = None
            self._error = None
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=self._log_handle or subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
                cwd=str(self.model.resolve().parent),
            )

        deadline = time.monotonic() + max(1.0, float(timeout))
        while time.monotonic() < deadline:
            process = self._process
            if process is None:
                break
            code = process.poll()
            if code is not None:
                self._error = f"whisper-server exited during startup with code {code}"
                raise STTUnavailableError(self._error)
            if self._is_ready():
                self._ready_at = time.time()
                return
            time.sleep(0.1)
        self._error = f"whisper-server did not become ready within {timeout:.0f}s"
        self.close()
        raise STTUnavailableError(self._error)

    def close(self) -> None:
        with self._lock:
            process = self._process
            self._process = None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None

    def status(self) -> dict[str, Any]:
        available, availability_error = self.available()
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
    ) -> tuple[bytes, str]:
        if not audio:
            raise ValueError("audio file is empty")
        self.start()
        whisper_format = "verbose_json" if response_format == "verbose_json" else response_format
        body, boundary = self._multipart(
            audio=audio,
            filename=filename,
            language=language,
            prompt=prompt,
            response_format=whisper_format,
        )
        request = urllib.request.Request(
            f"{self.base_url}/inference",
            data=body,
            method="POST",
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read()
                content_type = response.headers.get_content_type()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"whisper.cpp rejected the audio: {detail}") from exc
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


__all__ = ["STTUnavailableError", "WhisperCppRuntime"]
