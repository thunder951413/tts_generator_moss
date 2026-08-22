# coding=utf-8
"""Persistent streaming generation job records for the local Qwen3-TTS service."""

from __future__ import annotations

import json
import logging
import os
import queue
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from fastapi import HTTPException

DEFAULT_MAX_NEW_TOKENS = 7500


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
            "sample_rate": 24000,
            "channels": 1,
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
    def __init__(
        self,
        root_dir: str | Path,
        *,
        max_active_jobs: int = 64,
        close_callback: Callable[[str], None] | None = None,
    ) -> None:
        self._jobs: dict[str, StreamingJob] = {}
        self._lock = threading.Lock()
        self.max_active_jobs = max(1, int(max_active_jobs))
        self._close_callback = close_callback
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
            active_count = sum(
                item.snapshot().get("state")
                in {"queued", "loading_runtime", "running"}
                for item in self._jobs.values()
            )
            if active_count >= self.max_active_jobs:
                raise HTTPException(status_code=429, detail="audio generation queue is full")
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
        if self._close_callback is not None:
            self._close_callback(job.job_id)
        return job

    def close_all(self) -> list[str]:
        with self._lock:
            job_ids = [
                job.job_id
                for job in self._jobs.values()
                if job.snapshot().get("state")
                in {"queued", "loading_runtime", "running"}
            ]
        for job_id in job_ids:
            self.close(job_id)
        return job_ids
