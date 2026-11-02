"""Post-processing for document (audiobook) segments before AAC encoding.

All functions take and return mono float32 numpy arrays in [-1, 1].
"""

from __future__ import annotations

import numpy as np

SILENCE_THRESHOLD_DBFS = -45.0
FRAME_SECONDS = 0.010
LEAD_SECONDS = 0.060
TAIL_SECONDS = 0.100
FADE_IN_SECONDS = 0.005
FADE_OUT_SECONDS = 0.010
TARGET_SPEECH_RMS_DBFS = -20.0
MAX_GAIN_DB = 10.0
PEAK_LIMIT_DBFS = -1.5
PAUSE_SECONDS = {"sentence": 0.30, "paragraph": 0.60, "chapter": 1.0}


def _db_to_amp(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def to_mono_float(samples: np.ndarray) -> np.ndarray:
    audio = np.asarray(samples, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1, dtype=np.float32)
    return np.ascontiguousarray(audio.reshape(-1), dtype=np.float32)


def remove_dc(audio: np.ndarray) -> np.ndarray:
    audio = to_mono_float(audio)
    if audio.size == 0:
        return audio
    return (audio - np.float32(audio.mean(dtype=np.float64))).astype(np.float32)


def frame_rms(audio: np.ndarray, sample_rate: int, frame_seconds: float = FRAME_SECONDS) -> np.ndarray:
    frame = max(1, int(round(sample_rate * frame_seconds)))
    count = audio.size // frame
    if count == 0:
        return np.sqrt(np.mean(np.square(audio, dtype=np.float64))).reshape(1) if audio.size else np.zeros(0)
    framed = audio[: count * frame].astype(np.float64).reshape(count, frame)
    rms = np.sqrt(np.mean(framed * framed, axis=1))
    if audio.size > count * frame:
        tail = audio[count * frame :].astype(np.float64)
        rms = np.append(rms, np.sqrt(np.mean(tail * tail)))
    return rms


def trim_silence(
    audio: np.ndarray,
    sample_rate: int,
    *,
    threshold_dbfs: float = SILENCE_THRESHOLD_DBFS,
    lead_seconds: float = LEAD_SECONDS,
    tail_seconds: float = TAIL_SECONDS,
) -> np.ndarray:
    """Trim leading/trailing silence, keeping a short natural lead and tail."""
    audio = to_mono_float(audio)
    if audio.size == 0:
        return audio
    frame = max(1, int(round(sample_rate * FRAME_SECONDS)))
    rms = frame_rms(audio, sample_rate)
    voiced = np.flatnonzero(rms > _db_to_amp(threshold_dbfs))
    if voiced.size == 0:
        return audio
    start = max(0, int(voiced[0]) * frame - int(round(lead_seconds * sample_rate)))
    end = min(audio.size, (int(voiced[-1]) + 1) * frame + int(round(tail_seconds * sample_rate)))
    return audio[start:end].copy()


def apply_fades(
    audio: np.ndarray,
    sample_rate: int,
    *,
    fade_in_seconds: float = FADE_IN_SECONDS,
    fade_out_seconds: float = FADE_OUT_SECONDS,
) -> np.ndarray:
    audio = to_mono_float(audio).copy()
    fade_in = min(audio.size, int(round(fade_in_seconds * sample_rate)))
    fade_out = min(audio.size, int(round(fade_out_seconds * sample_rate)))
    if fade_in > 0:
        audio[:fade_in] *= np.linspace(0.0, 1.0, fade_in, dtype=np.float32)
    if fade_out > 0:
        audio[audio.size - fade_out :] *= np.linspace(1.0, 0.0, fade_out, dtype=np.float32)
    return audio


def speech_rms(audio: np.ndarray, sample_rate: int, *, threshold_dbfs: float = SILENCE_THRESHOLD_DBFS) -> float:
    """RMS measured over non-silent 10 ms frames only (0.0 if all silent)."""
    audio = to_mono_float(audio)
    if audio.size == 0:
        return 0.0
    frame = max(1, int(round(sample_rate * FRAME_SECONDS)))
    count = audio.size // frame
    if count == 0:
        return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    framed = audio[: count * frame].astype(np.float64).reshape(count, frame)
    energies = np.mean(framed * framed, axis=1)
    voiced = energies > _db_to_amp(threshold_dbfs) ** 2
    if not np.any(voiced):
        return 0.0
    return float(np.sqrt(np.mean(energies[voiced])))


def normalize_loudness(
    audio: np.ndarray,
    sample_rate: int,
    *,
    target_dbfs: float = TARGET_SPEECH_RMS_DBFS,
    max_gain_db: float = MAX_GAIN_DB,
) -> np.ndarray:
    audio = to_mono_float(audio)
    rms = speech_rms(audio, sample_rate)
    if rms <= 0.0:
        return audio.copy()
    gain_db = target_dbfs - 20.0 * float(np.log10(rms))
    gain_db = max(-max_gain_db, min(max_gain_db, gain_db))
    return (audio * np.float32(_db_to_amp(gain_db))).astype(np.float32)


def limit_peak(audio: np.ndarray, *, ceiling_dbfs: float = PEAK_LIMIT_DBFS) -> np.ndarray:
    """Scale the whole block down if its peak exceeds the ceiling (no clipping)."""
    audio = to_mono_float(audio)
    if audio.size == 0:
        return audio
    peak = float(np.max(np.abs(audio)))
    ceiling = _db_to_amp(ceiling_dbfs)
    if peak <= ceiling or peak <= 0.0:
        return audio.copy()
    return (audio * np.float32(ceiling / peak)).astype(np.float32)


def pause_seconds(pause_after: str | None) -> float:
    return PAUSE_SECONDS.get(str(pause_after or "sentence"), PAUSE_SECONDS["sentence"])


def append_pause(audio: np.ndarray, sample_rate: int, pause_after: str | None) -> np.ndarray:
    pad = int(round(pause_seconds(pause_after) * sample_rate))
    return np.concatenate([to_mono_float(audio), np.zeros(pad, dtype=np.float32)])


def process_segment_audio(audio: np.ndarray, sample_rate: int, pause_after: str | None = "sentence") -> np.ndarray:
    """DC removal, trim, fades, speech-RMS normalisation, peak limit, then pause."""
    processed = remove_dc(audio)
    processed = trim_silence(processed, sample_rate)
    processed = apply_fades(processed, sample_rate)
    processed = normalize_loudness(processed, sample_rate)
    processed = limit_peak(processed)
    return append_pause(processed, sample_rate, pause_after)


def process_wav_file(source: str, destination: str, pause_after: str | None = "sentence") -> tuple[int, int]:
    """Read a WAV, post-process it, write PCM16 WAV. Returns (samples, sample_rate)."""
    import soundfile as sf

    audio, sample_rate = sf.read(str(source), dtype="float32", always_2d=False)
    processed = process_segment_audio(audio, int(sample_rate), pause_after)
    sf.write(str(destination), processed, int(sample_rate), subtype="PCM_16")
    return int(processed.size), int(sample_rate)


__all__ = [
    "PAUSE_SECONDS",
    "append_pause",
    "apply_fades",
    "limit_peak",
    "normalize_loudness",
    "pause_seconds",
    "process_segment_audio",
    "process_wav_file",
    "remove_dc",
    "speech_rms",
    "trim_silence",
]
