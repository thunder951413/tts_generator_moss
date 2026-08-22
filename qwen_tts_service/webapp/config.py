# coding=utf-8
"""Path and default configuration for the local Qwen3-TTS web service."""

from __future__ import annotations

import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
STREAMING_MODULE_DIR = REPO_ROOT / "qwen_tts_service"

DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "qwen_tts_streaming"
DEFAULT_UPLOAD_DIR = REPO_ROOT / "outputs" / "qwen_tts_uploads"
DEFAULT_DOCUMENT_PROJECT_DIR = REPO_ROOT / "outputs" / "qwen_tts_document_projects"
DEFAULT_SERVICE_JOB_DIR = REPO_ROOT / "outputs" / "qwen_tts_service_jobs"
DEFAULT_PRESET_DIR = REPO_ROOT / "outputs" / "qwen_tts_presets"
DEFAULT_PERFORMANCE_TUNING_PATH = REPO_ROOT / "outputs" / "qwen_tts_performance.json"
NOVEL_READER_WEB_DIR = REPO_ROOT / "web" / "novel_reader"
SERVICE_AUTH_COOKIE = "qwen_tts_service_session"
DEFAULT_QWEN_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
DEFAULT_QWEN_WORKER_SCRIPT = STREAMING_MODULE_DIR / "qwen_worker.py"
DEFAULT_QWEN_0_6B_MODEL_DIR = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
DEFAULT_QWEN_1_7B_MODEL_DIR = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
DEFAULT_QWEN_BACKEND = "ggml"
DEFAULT_QWEN_QUANT = "Q4_K_M"
DEFAULT_QWENTTS_LIBRARY = REPO_ROOT / ".runtime" / "qwentts.cpp" / "build-metal" / "libqwen.dylib"
DEFAULT_WHISPER_SERVER = Path("/opt/homebrew/bin/whisper-server")
DEFAULT_WHISPER_MODEL = REPO_ROOT / ".runtime" / "whisper.cpp" / "models" / "ggml-small.bin"
DEFAULT_WHISPER_PORT = 7890
MODE_CLONE = "Clone"
REFERENCE_AUDIO_DIR = REPO_ROOT / "assets" / "audio"
BAILIAN_VOICES_TSV_PATH = REFERENCE_AUDIO_DIR / "bailian" / "voices.tsv"


def build_bailian_voice_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not BAILIAN_VOICES_TSV_PATH.exists():
        return rows
    with open(BAILIAN_VOICES_TSV_PATH, "r", encoding="utf-8-sig", newline="") as f:
        for item in csv.DictReader(f, delimiter="\t"):
            name = str(item.get("name") or "").strip()
            description = str(item.get("description") or "").strip()
            filename = str(item.get("filename") or "").strip()
            if not name or not filename:
                continue
            audio_path = BAILIAN_VOICES_TSV_PATH.parent / filename
            if audio_path.exists():
                rows.append(
                    {
                        "name": name,
                        "description": description,
                        "audio_path": str(audio_path),
                        "language": str(item.get("language") or "Chinese").strip(),
                        "transcript": str(item.get("transcript") or "").strip(),
                        "transcript_source": str(item.get("transcript_source") or "").strip(),
                    }
                )
    return rows


BAILIAN_VOICE_ROWS = build_bailian_voice_rows()
DEFAULT_CLONE_AUDIO_PATH = next(
    (row["audio_path"] for row in BAILIAN_VOICE_ROWS if row["name"] == "龙嫱"),
    BAILIAN_VOICE_ROWS[0]["audio_path"] if BAILIAN_VOICE_ROWS else "",
)
