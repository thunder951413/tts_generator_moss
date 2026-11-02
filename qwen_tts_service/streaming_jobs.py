# coding=utf-8
"""Persistent streaming generation job records for the local Qwen3-TTS service."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import re
import threading
import time
import uuid
import tempfile
from collections import deque
from pathlib import Path
from typing import Any, Callable

from fastapi import HTTPException

from presets import atomic_write_json

DEFAULT_MAX_NEW_TOKENS = 7500


class LosslessAudioQueue:
    """Spool slow-consumer PCM to disk instead of dropping audible chunks."""

    def __init__(self, *, max_bytes: int = 64 * 1024 * 1024) -> None:
        self._condition = threading.Condition()
        self._file = tempfile.SpooledTemporaryFile(max_size=512 * 1024, mode="w+b")
        self._lengths: deque[int] = deque()
        self._written = 0
        self._read = 0
        self._finished = False
        self._closed = False
        self._max_bytes = max_bytes
        self._async_waiters: list[tuple[asyncio.AbstractEventLoop, asyncio.Future]] = []

    def _wake_async_locked(self) -> None:
        waiters, self._async_waiters = self._async_waiters, []
        for loop, future in waiters:
            try:
                loop.call_soon_threadsafe(_resolve_future, future)
            except RuntimeError:  # loop already closed
                pass

    def _ready_locked(self) -> bool:
        return bool(self._lengths) or self._closed or self._finished

    def _take_locked(self) -> bytes | None:
        if self._closed:
            return None
        if not self._lengths:
            self.close()
            return None
        size = self._lengths.popleft()
        self._file.seek(self._read)
        chunk = self._file.read(size)
        self._read += size
        return chunk

    async def get_async(self, timeout: float | None = None) -> bytes | None:
        """Event-loop native ``get``: no executor thread hop per chunk."""
        loop = asyncio.get_running_loop()
        while True:
            with self._condition:
                if self._ready_locked():
                    return self._take_locked()
                future = loop.create_future()
                self._async_waiters.append((loop, future))
            try:
                if timeout is None:
                    await future
                else:
                    await asyncio.wait_for(future, timeout)
            except asyncio.TimeoutError:
                with self._condition:
                    self._async_waiters = [item for item in self._async_waiters if item[1] is not future]
                raise queue.Empty from None

    def put_nowait(self, item: bytes | None) -> None:
        with self._condition:
            if self._closed:
                return
            if item is None:
                self._finished = True
                self._condition.notify_all()
                self._wake_async_locked()
                return
            if self._finished:
                return
            if self._written + len(item) > self._max_bytes:
                raise queue.Full("流式音频缓存超出限制，请缩短单次生成文本")
            self._file.seek(self._written)
            self._file.write(item)
            self._written += len(item)
            self._lengths.append(len(item))
            self._condition.notify_all()
            self._wake_async_locked()

    def get(self, block: bool = True, timeout: float | None = None) -> bytes | None:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while not self._lengths and not self._closed and not self._finished:
                remaining = None if deadline is None else deadline - time.monotonic()
                if not block or (remaining is not None and remaining <= 0):
                    raise queue.Empty
                self._condition.wait(remaining)
            return self._take_locked()

    def get_nowait(self) -> bytes | None:
        return self.get(block=False)

    def close(self) -> None:
        with self._condition:
            if not self._closed:
                self._closed = True
                self._file.close()
                self._lengths.clear()
                self._condition.notify_all()
                self._wake_async_locked()


def _resolve_future(future: asyncio.Future) -> None:
    if not future.done():
        future.set_result(None)


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
        self.audio_queue = LosslessAudioQueue()
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
        self.active_audio_readers = 0
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
        playback_epoch: int | None = None,
    ) -> None:
        self._jobs: dict[str, StreamingJob] = {}
        self._ephemeral_pending: set[str] = set()
        self._lock = threading.Lock()
        self.max_active_jobs = max(1, int(max_active_jobs))
        self._close_callback = close_callback
        self._playback_epoch = playback_epoch
        self.max_cached_jobs = max(1, int(os.environ.get("QWEN_TTS_JOB_CACHE_LIMIT", "1000")))
        self.max_saved_jobs = max(self.max_cached_jobs, int(os.environ.get("QWEN_TTS_JOB_HISTORY_LIMIT", "20000")))
        self.pcm_ttl = max(1.0, float(os.environ.get("QWEN_TTS_PCM_TTL_SECONDS", "300")))
        self.ephemeral_ttl = max(60.0, float(os.environ.get("QWEN_TTS_EPHEMERAL_TTL_SECONDS", "3600")))
        self.cleanup_callback: Callable[[StreamingJob], None] | None = None
        self.root_dir = Path(root_dir).resolve()
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._load_existing()
        self._saved_jobs = sum(1 for _ in self.root_dir.glob("*.json"))
        self.maintenance()

    def _manifest_path(self, job_id: str) -> Path:
        if not re.fullmatch(r"[a-f0-9]{32}", job_id or ""):
            raise ValueError("invalid job id")
        return self.root_dir / f"{job_id}.json"

    def maintenance(self, *, now: float | None = None) -> None:
        now = time.time() if now is None else now
        with self._lock:
            self._saved_jobs = sum(1 for _ in self.root_dir.glob("*.json"))
            jobs = list(self._jobs.values())
            pending = list(self._ephemeral_pending)
        if self.cleanup_callback:
            cached = {job.job_id for job in jobs}
            for job_id in pending:
                if job_id not in cached:
                    try:
                        jobs.append(self.get(job_id))
                    except HTTPException:
                        with self._lock:
                            self._ephemeral_pending.discard(job_id)
        terminal = []
        for job in jobs:
            status = job.snapshot()
            if status.get("state") in {"queued", "loading_runtime", "running"} or (job.thread and job.thread.is_alive()):
                continue
            if job.active_audio_readers:
                continue
            age = now - float(status.get("updated_at") or now)
            if age >= self.pcm_ttl and not job.active_audio_readers:
                with job.status_lock:
                    if not job.active_audio_readers and now - float(job.status.get("updated_at") or now) >= self.pcm_ttl:
                        job.audio_queue.close()
                        job.status["stream_expired"] = True
            if (age >= self.ephemeral_ttl and self.cleanup_callback
                    and status.get("ephemeral_audio") and not status.get("ephemeral_cleaned")):
                self.cleanup_callback(job)
                with self._lock:
                    self._ephemeral_pending.discard(job.job_id)
            terminal.append(job)
        # Evict only terminal objects from RAM; saved manifests and user audio
        # stay available through get() and are never removed by this policy.
        terminal.sort(key=lambda job: float(job.snapshot().get("updated_at") or 0))
        with self._lock:
            excess = max(0, len(self._jobs) - self.max_cached_jobs)
            for job in terminal[:excess]:
                job.audio_queue.close()
                self._jobs.pop(job.job_id, None)

    def _restored_status(self, manifest: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        """Normalize a manifest from an earlier process; report real changes."""
        status = dict(manifest.get("status") or {})
        changed = False
        if status.get("state") in {"queued", "loading_runtime", "running"}:
            status.update(
                state="interrupted",
                error="服务重启时任务仍未完成，请重新提交",
                closed=True,
                updated_at=time.time(),
            )
            changed = True
        elif status.get("state") == "finished" and manifest.get("result") and self._playback_epoch is not None:
            # A stored result is replayable in a new service process. Live
            # requests from this process still retain their epoch.  This is
            # applied on every load, so it never needs to be written back.
            status["playback_epoch"] = self._playback_epoch
            status["playback_state"] = "idle"
        status["stream_expired"] = True
        return status, changed

    def _restore_job(self, job_id: str, manifest: dict[str, Any]) -> tuple[StreamingJob, bool]:
        status, changed = self._restored_status(manifest)
        job = StreamingJob(job_id, status=status, result=manifest.get("result"), persist_callback=self._persist)
        job.audio_queue.close()  # Live PCM is not restorable across service processes.
        return job, changed

    def _load_existing(self) -> None:
        """Scan saved manifests at boot without rewriting them all.

        Only the newest ``max_cached_jobs`` are kept in RAM (older ones load
        lazily through ``get``), and a manifest is persisted only when its
        status actually changed (an unfinished job marked interrupted).
        """
        entries: list[tuple[float, Path]] = []
        for path in self.root_dir.glob("*.json"):
            try:
                entries.append((path.stat().st_mtime, path))
            except OSError:
                continue
        entries.sort(key=lambda item: item[0], reverse=True)
        for index, (_mtime, path) in enumerate(entries):
            try:
                manifest = json.loads(path.read_text(encoding="utf-8"))
                job_id = str((manifest.get("status") or {}).get("job_id") or path.stem)
                state = (manifest.get("status") or {}).get("state")
                status = manifest.get("status") or {}
                if status.get("ephemeral_audio") and not status.get("ephemeral_cleaned"):
                    self._ephemeral_pending.add(job_id)
                needs_rewrite = state in {"queued", "loading_runtime", "running"}
                if index >= self.max_cached_jobs and not needs_rewrite:
                    continue
                job, changed = self._restore_job(job_id, manifest)
                self._jobs[job_id] = job
                if changed:
                    job.persist()
            except Exception:
                logging.exception("failed to restore service job manifest: %s", path)

    def _persist(self, job: StreamingJob) -> None:
        atomic_write_json(self._manifest_path(job.job_id), job.manifest())

    def create(self, *, summary: dict[str, Any] | None = None) -> StreamingJob:
        job = StreamingJob(uuid.uuid4().hex, persist_callback=self._persist)
        if summary:
            with job.status_lock:
                job.status.update(summary)
        with self._lock:
            if self._saved_jobs >= self.max_saved_jobs:
                job.audio_queue.close()
                raise HTTPException(507, "历史任务已达到数量上限，请归档历史记录后重试")
            active_count = sum(
                item.snapshot().get("state")
                in {"queued", "loading_runtime", "running"}
                for item in self._jobs.values()
            )
            if active_count >= self.max_active_jobs:
                job.audio_queue.close()
                raise HTTPException(status_code=429, detail="audio generation queue is full")
            self._jobs[job.job_id] = job
            if job.status.get("ephemeral_audio"):
                self._ephemeral_pending.add(job.job_id)
            self._saved_jobs += 1
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
            try:
                manifest = json.loads(self._manifest_path(job_id).read_text(encoding="utf-8"))
                restored, _changed = self._restore_job(job_id, manifest)
                with self._lock:
                    job = self._jobs.setdefault(job_id, restored)
                if job is not restored:
                    restored.audio_queue.close()
            except (OSError, ValueError, TypeError):
                raise HTTPException(status_code=404, detail=f"stream job not found: {job_id}") from None
        return job

    def reference_usage(self, path: str | Path) -> list[str]:
        normalized = str(Path(path).resolve())
        with self._lock:
            jobs = list(self._jobs.values())
        return [
            f"生成任务“{job.snapshot().get('title') or job.job_id[:8]}”"
            for job in jobs
            if (
                job.snapshot().get("state") in {"queued", "loading_runtime", "running"}
                or (job.thread is not None and job.thread.is_alive())
            )
            and job.snapshot().get("reference_audio_path")
            and str(Path(job.snapshot()["reference_audio_path"]).resolve()) == normalized
        ]

    def close(self, job_id: str) -> StreamingJob:
        job = self.get(job_id)
        with job.status_lock:
            job.is_closed = True
            job.status["closed"] = True
            job.status["close_revision"] = int(job.status.get("close_revision", 0)) + 1
            if job.status.get("state") not in {"finished", "error"}:
                job.status["state"] = "closed"
            job.audio_queue.close()
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

    def shutdown(self) -> None:
        self.close_all()
        with self._lock:
            jobs = list(self._jobs.values())
        for job in jobs:
            job.audio_queue.close()
