# coding=utf-8
"""Command-line launcher for the local Qwen3-TTS web service."""

from __future__ import annotations

import argparse
import ipaddress
import os

import uvicorn

from webapp.app_factory import create_app
from webapp.config import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PRESET_DIR,
    DEFAULT_QWEN_0_6B_MODEL_DIR,
    DEFAULT_QWEN_1_7B_MODEL_DIR,
    DEFAULT_QWEN_BACKEND,
    DEFAULT_QWEN_PYTHON,
    DEFAULT_QWEN_QUANT,
    DEFAULT_QWEN_WORKER_SCRIPT,
    DEFAULT_QWENTTS_LIBRARY,
    DEFAULT_UPLOAD_DIR,
    DEFAULT_WHISPER_MODEL,
    DEFAULT_WHISPER_PORT,
    DEFAULT_WHISPER_SERVER,
)


def _is_loopback_bind_host(host: str) -> bool:
    normalized = str(host or "").strip().lower().strip("[]")
    if normalized == "localhost" or normalized.startswith("127."):
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _validate_bind_security(host: str, password: str) -> None:
    selected_password = str(password or "")
    if (
        not _is_loopback_bind_host(host)
        and (len(selected_password) < 4 or selected_password == "change-me")
    ):
        raise ValueError(
            "Refusing non-loopback exposure with an empty/weak password. "
            "Set --access-password or QWEN_TTS_ACCESS_PASSWORD to at least 4 characters."
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Qwen3-TTS Apple Silicon service.")
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "7861")))
    parser.add_argument(
        "--access-password",
        default=os.environ.get("QWEN_TTS_ACCESS_PASSWORD", ""),
        help="Password accepted as a login, Bearer token, or X-API-Key.",
    )
    parser.add_argument("--qwen-python", default=os.environ.get("QWEN_TTS_PYTHON", str(DEFAULT_QWEN_PYTHON)))
    parser.add_argument(
        "--qwen-worker-script",
        default=os.environ.get("QWEN_TTS_WORKER_SCRIPT", str(DEFAULT_QWEN_WORKER_SCRIPT)),
    )
    parser.add_argument(
        "--qwen-0-6b-model-dir",
        default=os.environ.get("QWEN_TTS_0_6B_MODEL_DIR", str(DEFAULT_QWEN_0_6B_MODEL_DIR)),
    )
    parser.add_argument(
        "--qwen-1-7b-model-dir",
        default=os.environ.get("QWEN_TTS_1_7B_MODEL_DIR", str(DEFAULT_QWEN_1_7B_MODEL_DIR)),
    )
    parser.add_argument(
        "--qwen-0-6b-lanes",
        type=int,
        default=int(os.environ.get("QWEN_TTS_0_6B_LANES", "1")),
    )
    parser.add_argument(
        "--qwen-1-7b-lanes",
        type=int,
        default=int(os.environ.get("QWEN_TTS_1_7B_LANES", "1")),
    )
    parser.add_argument(
        "--qwen-backend",
        choices=["ggml", "torch"],
        default=os.environ.get("QWEN_TTS_BACKEND", DEFAULT_QWEN_BACKEND),
    )
    parser.add_argument(
        "--qwen-quant",
        choices=["BF16", "Q8_0", "Q4_K_M"],
        default=os.environ.get("QWEN_TTS_QUANT", DEFAULT_QWEN_QUANT),
    )
    parser.add_argument(
        "--qwentts-library",
        default=os.environ.get("QWENTTS_CPP_LIBRARY", str(DEFAULT_QWENTTS_LIBRARY)),
    )
    parser.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR)))
    parser.add_argument("--upload-dir", default=os.environ.get("UPLOAD_DIR", str(DEFAULT_UPLOAD_DIR)))
    parser.add_argument("--preset-dir", default=os.environ.get("QWEN_TTS_PRESET_DIR", str(DEFAULT_PRESET_DIR)))
    parser.add_argument("--no-preload", action="store_true")
    parser.add_argument(
        "--max-parallel-generations",
        type=int,
        default=int(os.environ.get("QWEN_TTS_MAX_PARALLEL_GENERATIONS", "1")),
    )
    parser.add_argument(
        "--document-parallel-generations",
        type=int,
        default=int(os.environ.get("QWEN_TTS_DOCUMENT_PARALLEL_GENERATIONS", "2")),
    )
    parser.add_argument(
        "--whisper-server",
        default=os.environ.get("QWEN_STT_SERVER", str(DEFAULT_WHISPER_SERVER)),
    )
    parser.add_argument(
        "--whisper-model",
        default=os.environ.get("QWEN_STT_MODEL", str(DEFAULT_WHISPER_MODEL)),
    )
    parser.add_argument(
        "--whisper-port",
        type=int,
        default=int(os.environ.get("QWEN_STT_PORT", str(DEFAULT_WHISPER_PORT))),
    )
    parser.add_argument(
        "--whisper-threads",
        type=int,
        default=int(os.environ.get("QWEN_STT_THREADS", "8")),
    )
    parser.add_argument("--no-stt", action="store_true")
    parser.add_argument("--no-stt-preload", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        _validate_bind_security(args.host, args.access_password)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    app = create_app(
        qwen_python=args.qwen_python,
        qwen_worker_script=args.qwen_worker_script,
        qwen_0_6b_model_dir=args.qwen_0_6b_model_dir,
        qwen_1_7b_model_dir=args.qwen_1_7b_model_dir,
        qwen_0_6b_lanes=max(1, int(args.qwen_0_6b_lanes)),
        qwen_1_7b_lanes=max(1, int(args.qwen_1_7b_lanes)),
        qwen_backend=args.qwen_backend,
        qwen_quant=args.qwen_quant,
        qwentts_library=args.qwentts_library,
        output_dir=args.output_dir,
        upload_dir=args.upload_dir,
        preset_dir=args.preset_dir,
        preload=not args.no_preload,
        max_parallel_generations=max(1, int(args.max_parallel_generations)),
        document_parallel_generations=max(1, int(args.document_parallel_generations)),
        access_password=args.access_password,
        stt_enabled=not args.no_stt,
        stt_preload=not args.no_stt_preload,
        whisper_server=args.whisper_server,
        whisper_model=args.whisper_model,
        whisper_port=args.whisper_port,
        whisper_threads=max(1, int(args.whisper_threads)),
    )
    uvicorn.run(app, host=args.host, port=int(args.port))
