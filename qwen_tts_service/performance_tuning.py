from __future__ import annotations

import copy
import json
import threading
import time
from pathlib import Path
from typing import Any

from presets import atomic_write_json


DEFAULT_RECOMMENDATION = {
    "stream_chunk_frames": 8,
    "stream_parallel": 1,
    "block_parallel": 1,
    "document_workers": 1,
}


class PerformanceTuningStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._cache: tuple[tuple[int, int] | None, dict[str, Any]] | None = None

    def _load(self) -> dict[str, Any]:
        """Parsed store, re-read only when the file's mtime/size changes.

        Every synthesis applies the performance profile, so this sits on the
        hot path; callers receive a deep copy they may mutate.
        """
        try:
            stat = self.path.stat()
            key: tuple[int, int] | None = (stat.st_mtime_ns, stat.st_size)
        except OSError:
            key = None
        cached = self._cache
        if cached is not None and cached[0] == key:
            return copy.deepcopy(cached[1])
        payload: Any = {"version": 1, "profiles": {}}
        if key is not None:
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = {"version": 1, "profiles": {}}
        if not isinstance(payload, dict):
            payload = {"version": 1, "profiles": {}}
        payload.setdefault("version", 1)
        payload.setdefault("profiles", {})
        self._cache = (key, payload)
        return copy.deepcopy(payload)

    def profile(self, profile_id: str) -> dict[str, Any] | None:
        with self._lock:
            profile = self._load().get("profiles", {}).get(profile_id)
            return dict(profile) if isinstance(profile, dict) else None

    def recommendation(self, profile_id: str) -> dict[str, int]:
        profile = self.profile(profile_id) or {}
        recommendation = dict(DEFAULT_RECOMMENDATION)
        recommendation.update(profile.get("recommendation") or {})
        return {
            "stream_chunk_frames": max(1, min(24, int(recommendation["stream_chunk_frames"]))),
            "stream_parallel": max(1, min(2, int(recommendation["stream_parallel"]))),
            "block_parallel": max(1, min(2, int(recommendation["block_parallel"]))),
            "document_workers": max(1, min(2, int(recommendation["document_workers"]))),
        }

    def save_profile(self, profile_id: str, result: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            payload = self._load()
            profiles = payload.setdefault("profiles", {})
            stored = dict(result)
            stored["profile_id"] = profile_id
            stored["updated_at"] = time.time()
            profiles[profile_id] = stored
            atomic_write_json(self.path, payload)
            self._cache = None
            return dict(stored)

    def public_payload(self) -> dict[str, Any]:
        with self._lock:
            return self._load()


def choose_recommendation(
    *,
    stream_measurements: list[dict[str, Any]],
    single_block_seconds: float,
    parallel_block_seconds: float | None,
) -> dict[str, int]:
    usable_streams = [
        item
        for item in stream_measurements
        if float(item.get("elapsed_seconds") or 0) > 0
        and float(item.get("first_audio_seconds") or 0) > 0
    ]
    if usable_streams:
        best_stream = min(
            usable_streams,
            key=lambda item: (
                float(item["first_audio_seconds"])
                + 0.30 / max(
                    0.01,
                    float(item.get("generation_realtime_factor") or 1.0),
                ),
                int(item["chunk_frames"]),
            ),
        )
        stream_chunk_frames = int(best_stream["chunk_frames"])
    else:
        stream_chunk_frames = 8

    throughput_gain = 1.0
    if parallel_block_seconds and parallel_block_seconds > 0 and single_block_seconds > 0:
        throughput_gain = (2.0 * single_block_seconds) / parallel_block_seconds
    block_parallel = 2 if throughput_gain >= 1.20 else 1
    return {
        "stream_chunk_frames": max(1, min(24, stream_chunk_frames)),
        "stream_parallel": block_parallel,
        "block_parallel": block_parallel,
        "document_workers": block_parallel,
    }
