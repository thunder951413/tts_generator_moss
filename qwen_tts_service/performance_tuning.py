from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any


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

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"version": 1, "profiles": {}}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"version": 1, "profiles": {}}
        if not isinstance(payload, dict):
            return {"version": 1, "profiles": {}}
        payload.setdefault("version", 1)
        payload.setdefault("profiles", {})
        return payload

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
            temporary = self.path.with_suffix(f"{self.path.suffix}.tmp")
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(temporary, self.path)
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
