# coding=utf-8
"""Persistent Qwen3-TTS web service for Apple Silicon (thin CLI entry).

The implementation lives in ``qwen_tts_service/webapp``; this script keeps the
historical launch path (``start-macos.sh``, QwenTTS.app) working and acts as
the compatibility facade: names patched here (tests override paths and inject
fake runtimes) are synced into the application factory before every call.
"""

from __future__ import annotations

import queue  # noqa: F401  (re-exported for tests)
import subprocess  # noqa: F401  (re-exported for tests)
import sys
from pathlib import Path

import torch  # noqa: F401  (re-exported for tests)

REPO_ROOT = Path(__file__).resolve().parent.parent
STREAMING_MODULE_DIR = REPO_ROOT / "qwen_tts_service"
if str(STREAMING_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(STREAMING_MODULE_DIR))

from activity_control import PlaybackCoordinator  # noqa: E402,F401
from document_projects import DocumentProjectManager  # noqa: E402,F401
from performance_tuning import PerformanceTuningStore, choose_recommendation  # noqa: E402,F401
from presets import VoicePresetStore  # noqa: E402,F401
from qwen_protocol import StreamingRequest  # noqa: E402,F401
from stt_runtime import STTUnavailableError, WhisperCppRuntime  # noqa: E402,F401
from runtime_manager import DEFAULT_MODEL_PROFILE, MODEL_PROFILE_LABELS, RuntimeManager  # noqa: E402,F401
from generation_scheduler import GpuGenerationScheduler  # noqa: E402,F401
from streaming_jobs import DEFAULT_MAX_NEW_TOKENS, StreamingJob, StreamingJobManager  # noqa: E402,F401
from webapp.app_factory import create_app as _create_app  # noqa: E402
from webapp.util import _pcm16le_bytes, _resolve_ffmpeg_path, _safe_int  # noqa: E402,F401
from webapp.cli import (  # noqa: E402
    _is_loopback_bind_host,
    _parse_args,
    _validate_bind_security,
    main,
)
from webapp.config import (  # noqa: E402,F401
    DEFAULT_DOCUMENT_PROJECT_DIR,
    DEFAULT_PERFORMANCE_TUNING_PATH,
    DEFAULT_PRESET_DIR,
    DEFAULT_SERVICE_JOB_DIR,
)

_SYNCED_NAMES = (
    "PlaybackCoordinator",
    "DocumentProjectManager",
    "PerformanceTuningStore",
    "choose_recommendation",
    "VoicePresetStore",
    "StreamingRequest",
    "STTUnavailableError",
    "WhisperCppRuntime",
    "DEFAULT_MODEL_PROFILE",
    "MODEL_PROFILE_LABELS",
    "RuntimeManager",
    "GpuGenerationScheduler",
    "DEFAULT_MAX_NEW_TOKENS",
    "StreamingJob",
    "StreamingJobManager",
    "DEFAULT_DOCUMENT_PROJECT_DIR",
    "DEFAULT_PERFORMANCE_TUNING_PATH",
    "DEFAULT_PRESET_DIR",
    "DEFAULT_SERVICE_JOB_DIR",
)


def create_app(**kwargs):
    """Build the FastAPI app; names patched on this module stay authoritative."""
    import webapp.app_factory as _factory

    for name in _SYNCED_NAMES:
        if name in globals():
            setattr(_factory, name, globals()[name])
    return _create_app(**kwargs)


if __name__ == "__main__":
    main()
