# coding=utf-8
"""Small pure helpers shared by the web application modules."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import numpy as np

def _safe_int(value: Any, *, default: int, minimum: int, maximum: int | None = None) -> int:
    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        parsed = int(default)
    parsed = max(int(minimum), parsed)
    if maximum is not None:
        parsed = min(int(maximum), parsed)
    return parsed


def _safe_aac_bitrate(value: Any, *, default: str = "80k") -> str:
    bitrate = str(value or default).strip().lower()
    return bitrate if bitrate in {"48k", "64k", "80k", "96k", "128k", "192k"} else default


def _resolve_ffmpeg_path() -> str:
    """Resolve ffmpeg even when a macOS app launches with a minimal PATH."""
    candidates = [
        os.environ.get("QWEN_TTS_FFMPEG", ""),
        shutil.which("ffmpeg") or "",
        "/opt/homebrew/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
    ]
    for value in candidates:
        if not value:
            continue
        candidate = Path(value).expanduser()
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate.resolve())
    return "ffmpeg"


def _safe_float(value: Any, *, default: float, minimum: float, maximum: float | None = None) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = float(default)
    parsed = max(float(minimum), parsed)
    if maximum is not None:
        parsed = min(float(maximum), parsed)
    return parsed


def _decode_reference_path(path: str) -> str:
    decoded = str(path or "")
    for _ in range(2):
        next_decoded = unquote(decoded)
        if next_decoded == decoded:
            break
        decoded = next_decoded
    return decoded


def _resolve_allowed_reference_audio_path(path: str, *roots: Path) -> Path:
    """Resolve a reference file without allowing an arbitrary local-file read."""
    try:
        candidate = Path(_decode_reference_path(path)).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise FileNotFoundError(path) from exc
    for root in roots:
        try:
            resolved_root = root.resolve(strict=True)
        except (OSError, RuntimeError):
            continue
        if candidate.is_file() and (candidate == resolved_root or resolved_root in candidate.parents):
            return candidate
    raise PermissionError("reference audio path is not allowed")


def _pcm16le_bytes(audio: Any, channels: int = 2) -> bytes:
    """Return interleaved little-endian int16 PCM for ``channels`` (1 or 2).

    ``audio`` is either raw mono int16 bytes as produced by the worker (the
    fast path: returned unchanged for mono, sample-duplicated for stereo), an
    int16 numpy array, or a float waveform shaped ``(samples,)`` or
    ``(channels, samples)`` in [-1, 1].
    """
    target_channels = 1 if int(channels) == 1 else 2
    if isinstance(audio, (bytes, bytearray, memoryview)):
        if target_channels == 1:
            return bytes(audio)
        mono = np.frombuffer(audio, dtype="<i2")
        return np.repeat(mono, 2).tobytes()
    array = np.asarray(audio)
    if array.ndim == 1:
        array = array[np.newaxis, :]
    if array.dtype != np.int16:
        array = (np.clip(array.astype(np.float32, copy=False), -1.0, 1.0) * 32767.0).round().astype(np.int16)
    if target_channels == 1:
        if array.shape[0] > 1:
            array = array.astype(np.int32).mean(axis=0, keepdims=True).round().astype(np.int16)
        else:
            array = array[:1]
    elif array.shape[0] == 1:
        array = np.repeat(array, 2, axis=0)
    elif array.shape[0] > 2:
        array = array[:2]
    return np.ascontiguousarray(array.T).astype("<i2", copy=False).tobytes()
