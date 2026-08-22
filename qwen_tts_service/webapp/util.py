# coding=utf-8
"""Small pure helpers shared by the web application modules."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import torch

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


def _pcm16le_bytes(waveform: torch.Tensor, channels: int = 2) -> bytes:
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    target_channels = 1 if int(channels) == 1 else 2
    if target_channels == 1:
        waveform = waveform.mean(dim=0, keepdim=True) if waveform.shape[0] > 1 else waveform[:1]
    elif waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    elif waveform.shape[0] > 2:
        waveform = waveform[:2]
    pcm = waveform.detach().cpu().to(torch.float32).clamp(-1.0, 1.0)
    pcm = (pcm * 32767.0).round().to(torch.int16)
    return pcm.transpose(0, 1).contiguous().numpy().tobytes()
