# coding=utf-8
"""FastAPI application factory for the local Qwen3-TTS web service."""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from document_projects import DocumentProjectManager
from activity_control import PlaybackCoordinator
from performance_tuning import PerformanceTuningStore
from presets import VoicePresetStore
from qwen_protocol import StreamingRequest
from stt_runtime import STTUnavailableError, WhisperCppRuntime
from runtime_manager import DEFAULT_MODEL_PROFILE, RuntimeManager
from generation_scheduler import GpuGenerationScheduler
from streaming_jobs import StreamingJobManager
from webapp.config import (
    BAILIAN_VOICE_ROWS,
    DEFAULT_CLONE_AUDIO_PATH,
    DEFAULT_DOCUMENT_PROJECT_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PERFORMANCE_TUNING_PATH,
    DEFAULT_PRESET_DIR,
    DEFAULT_QWEN_0_6B_MODEL_DIR,
    DEFAULT_QWEN_1_7B_MODEL_DIR,
    DEFAULT_QWEN_BACKEND,
    DEFAULT_QWENTTS_LIBRARY,
    DEFAULT_QWEN_PYTHON,
    DEFAULT_QWEN_QUANT,
    DEFAULT_QWEN_WORKER_SCRIPT,
    DEFAULT_SERVICE_JOB_DIR,
    DEFAULT_UPLOAD_DIR,
    DEFAULT_WHISPER_MODEL,
    DEFAULT_WHISPER_PORT,
    DEFAULT_WHISPER_SERVER,
    NOVEL_READER_WEB_DIR,
    REFERENCE_AUDIO_DIR,
    REPO_ROOT,
    SERVICE_AUTH_COOKIE,
)
from webapp.context import ServiceContext
from webapp.routers.auth_pages import register_auth_pages_routes
from webapp.routers.documents import register_documents_routes
from webapp.routers.generation import register_generation_routes
from webapp.routers.playback import register_playback_routes
from webapp.routers.settings_performance import register_settings_performance_routes
from webapp.routers.system_control import register_system_control_routes
from webapp.routers.voices_presets import register_voices_presets_routes
from webapp.util import _resolve_allowed_reference_audio_path, _resolve_ffmpeg_path


def create_app(
    *,
    qwen_python: str | Path = DEFAULT_QWEN_PYTHON,
    qwen_worker_script: str | Path = DEFAULT_QWEN_WORKER_SCRIPT,
    qwen_0_6b_model_dir: str | Path = DEFAULT_QWEN_0_6B_MODEL_DIR,
    qwen_1_7b_model_dir: str | Path = DEFAULT_QWEN_1_7B_MODEL_DIR,
    qwen_0_6b_lanes: int = 1,
    qwen_1_7b_lanes: int = 1,
    qwen_backend: str = DEFAULT_QWEN_BACKEND,
    qwen_quant: str = DEFAULT_QWEN_QUANT,
    qwentts_library: str | Path = DEFAULT_QWENTTS_LIBRARY,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    upload_dir: str | Path = DEFAULT_UPLOAD_DIR,
    preset_dir: str | Path = DEFAULT_PRESET_DIR,
    document_project_dir: str | Path | None = None,
    service_job_dir: str | Path | None = None,
    performance_tuning_path: str | Path | None = None,
    preload: bool = True,
    max_parallel_generations: int = 1,
    document_parallel_generations: int = 2,
    access_password: str = "",
    stt_enabled: bool = True,
    stt_preload: bool | None = None,
    whisper_server: str | Path = DEFAULT_WHISPER_SERVER,
    whisper_model: str | Path = DEFAULT_WHISPER_MODEL,
    whisper_port: int = DEFAULT_WHISPER_PORT,
    whisper_threads: int = 8,
) -> FastAPI:
    document_project_dir = Path(document_project_dir or DEFAULT_DOCUMENT_PROJECT_DIR)
    service_job_dir = Path(service_job_dir or DEFAULT_SERVICE_JOB_DIR)
    performance_tuning_path = Path(performance_tuning_path or DEFAULT_PERFORMANCE_TUNING_PATH)
    runtime_manager = RuntimeManager(
        qwen_python=str(qwen_python),
        qwen_worker_script=str(qwen_worker_script),
        qwen_0_6b_model_dir=str(qwen_0_6b_model_dir),
        qwen_1_7b_model_dir=str(qwen_1_7b_model_dir),
        qwen_0_6b_lanes=max(1, int(qwen_0_6b_lanes)),
        qwen_1_7b_lanes=max(1, int(qwen_1_7b_lanes)),
        qwen_backend=str(qwen_backend),
        qwen_quant=str(qwen_quant),
        qwentts_library=str(qwentts_library),
    )
    performance_tuning = PerformanceTuningStore(performance_tuning_path)
    for profile_id, profile in runtime_manager.profiles.items():
        if performance_tuning.profile(profile_id) is not None:
            profile["lanes"] = performance_tuning.recommendation(profile_id)["block_parallel"]
    playback_coordinator = PlaybackCoordinator(
        max_waiters=64, internal_burst_limit=4, initial_epoch=time.time_ns() // 1_000_000,
    )
    jobs = StreamingJobManager(
        service_job_dir,
        max_active_jobs=64,
        close_callback=playback_coordinator.release,
    )
    output_dir = Path(output_dir)
    upload_dir = Path(upload_dir)
    reader_temp_dir = output_dir / "reader-temporary-audio"
    output_dir.mkdir(parents=True, exist_ok=True)
    upload_dir.mkdir(parents=True, exist_ok=True)
    reader_temp_dir.mkdir(parents=True, exist_ok=True)
    preset_store = VoicePresetStore(preset_dir)
    initial_recommendation = performance_tuning.recommendation(DEFAULT_MODEL_PROFILE)
    initial_parallel = (
        initial_recommendation["block_parallel"]
        if performance_tuning.profile(DEFAULT_MODEL_PROFILE) is not None
        else max(1, int(max_parallel_generations))
    )
    generation_scheduler = GpuGenerationScheduler(
        max_parallel=initial_parallel,
        external_blocked=playback_coordinator.internal_playback_active,
    )
    # Keep the HTTP service available while allowing the two resident model
    # runtimes to be controlled independently from the native workbench.
    ffmpeg_path = _resolve_ffmpeg_path()
    stt_runtime = WhisperCppRuntime(
        binary=whisper_server,
        model=whisper_model,
        port=int(whisper_port),
        threads=max(1, int(whisper_threads)),
        log_path=REPO_ROOT / "logs" / "whisper-server.log",
    )
    should_preload_stt = bool(preload if stt_preload is None else stt_preload)
    def synthesize_for_profile_runtime(runtime: Any, request: StreamingRequest, *, output_dir: str | Path):
        # A queued request may target a different model from the one currently
        # resident. Apply shared scheduling only once its session is admitted.
        profile_id = getattr(runtime, "profile_id", None)
        if profile_id in runtime_manager.profiles and not generation_scheduler.status()["performance_test_active"]:
            apply_performance_profile(profile_id)
        yield from runtime.synthesize(request, output_dir=output_dir)

    document_projects = DocumentProjectManager(
        root_dir=document_project_dir,
        runtime_session=runtime_manager.session,
        synthesize_fn=synthesize_for_profile_runtime,
        request_cls=StreamingRequest,
        generation_lock=generation_scheduler,
        ffmpeg_path=ffmpeg_path,
        synthesis_workers=(
            initial_recommendation["document_workers"]
            if performance_tuning.profile(DEFAULT_MODEL_PROFILE) is not None
            else min(
                max(1, int(document_parallel_generations)),
                max(1, int(max_parallel_generations)),
            )
        ),
    )

    def apply_performance_profile(profile_id: str) -> dict[str, int]:
        recommendation = performance_tuning.recommendation(profile_id)
        if performance_tuning.profile(profile_id) is None:
            recommendation["block_parallel"] = min(2, max(1, int(runtime_manager.profiles[profile_id].get("lanes", 1))))
            recommendation["document_workers"] = min(recommendation["block_parallel"], max(1, int(document_parallel_generations)))
        else:
            runtime_manager.profiles[profile_id]["lanes"] = recommendation["block_parallel"]
        if runtime_manager.status().get("active_profile") in {None, profile_id}:
            generation_scheduler.configure_max_parallel(recommendation["block_parallel"])
            document_projects.configure_synthesis_workers(recommendation["document_workers"])
        return recommendation

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if preload:
            with runtime_manager.session(DEFAULT_MODEL_PROFILE):
                pass
        if stt_enabled and should_preload_stt:
            try:
                stt_runtime.start()
            except STTUnavailableError:
                logging.exception("STT preload failed; TTS service will remain available")
        try:
            yield
        finally:
            stt_runtime.close()
            runtime_manager.close()

    app = FastAPI(title="Qwen3-TTS Apple Silicon Service", lifespan=lifespan)
    app.state.stt_runtime = stt_runtime
    app.mount(
        "/reader-assets",
        StaticFiles(directory=str(NOVEL_READER_WEB_DIR)),
        name="novel-reader-assets",
    )
    configured_cors = [
        origin.strip()
        for origin in os.environ.get("QWEN_TTS_CORS_ORIGINS", "").split(",")
        if origin.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        # The reader and native app are same-origin/local clients. Cross-origin
        # browser access is opt-in so a random website cannot probe a LAN-bound
        # service with a leaked API key.
        allow_origins=configured_cors,
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-API-Key"],
        expose_headers=[
            "X-Audio-Codec",
            "X-Audio-Sample-Rate",
            "X-Audio-Channels",
            "X-Stream-Id",
            "X-Playback-Epoch",
            "X-Playback-Lease",
        ],
    )
    resolved_access_password = str(access_password or "")
    expected_session = hmac.new(
        resolved_access_password.encode("utf-8"), b"qwen-tts-service-session", hashlib.sha256
    ).hexdigest()

    @app.middleware("http")
    async def require_service_login(request: Request, call_next):
        request.state.caller_kind = "internal" if request.client and request.client.host in {
            "127.0.0.1", "::1", "localhost", "testclient"
        } else "external"
        if (
            not resolved_access_password
            or request.method == "OPTIONS"
            or request.url.path in {"/login", "/api/health"}
        ):
            return await call_next(request)
        supplied = request.cookies.get(SERVICE_AUTH_COOKIE, "")
        authorization = request.headers.get("authorization", "")
        bearer = authorization[7:].strip() if authorization.lower().startswith("bearer ") else ""
        api_key = request.headers.get("x-api-key", "")
        if (
            hmac.compare_digest(supplied, expected_session)
            or hmac.compare_digest(bearer, resolved_access_password)
            or hmac.compare_digest(api_key, resolved_access_password)
        ):
            request.state.caller_kind = (
                "internal"
                if hmac.compare_digest(supplied, expected_session)
                else "external"
            )
            return await call_next(request)
        if request.url.path.startswith("/api/"):
            return JSONResponse({"detail": "authentication required"}, status_code=401)
        next_path = request.url.path if request.url.path.startswith("/") else "/"
        return RedirectResponse(url=f"/login?next={next_path}", status_code=303)

    def _remove_generated_result_files(result: dict[str, Any] | None) -> None:
        if not result:
            return
        resolved_output = output_dir.resolve()
        for key in ("audio_path", "tokens_path", "metadata_path"):
            raw_path = result.get(key)
            if not raw_path:
                continue
            try:
                candidate = Path(str(raw_path)).resolve()
                candidate.relative_to(resolved_output)
            except (OSError, RuntimeError, ValueError):
                continue
            if candidate.is_file():
                candidate.unlink(missing_ok=True)


    def preset_payload(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        name = str(payload.get("name") or "").strip()
        settings = payload.get("settings")
        if not isinstance(settings, dict):
            raise HTTPException(status_code=400, detail="预设设置必须是对象")
        copied = dict(settings)
        model_profile = str(copied.get("model_profile") or DEFAULT_MODEL_PROFILE)
        if model_profile not in runtime_manager.profiles:
            raise HTTPException(status_code=400, detail="无效的模型预设")
        copied["model_profile"] = model_profile
        reference_path = str(copied.get("reference_audio_path") or "").strip()
        if reference_path:
            try:
                copied["reference_audio_path"] = str(
                    _resolve_allowed_reference_audio_path(
                        reference_path,
                        REFERENCE_AUDIO_DIR,
                        preset_store.audio_dir,
                    )
                )
            except FileNotFoundError as exc:
                raise HTTPException(status_code=400, detail="预设参考音频不存在") from exc
            except PermissionError as exc:
                raise HTTPException(status_code=400, detail="预设参考音频不在允许目录") from exc
        return name, copied


    def active_service_settings() -> dict[str, Any]:
        configuration = preset_store.service_configuration()
        if configuration is not None:
            return {
                "active_preset_id": str(configuration.get("preset_id") or ""),
                "name": str(configuration.get("name") or "服务设置"),
                "source": "preset" if configuration.get("preset_id") else "studio",
                "settings": dict(configuration.get("settings") or {}),
                "updated_at": configuration.get("updated_at"),
            }
        active = preset_store.active()
        if active is not None:
            return {
                "active_preset_id": active["id"],
                "name": active["name"],
                "source": "preset",
                "settings": dict(active.get("settings") or {}),
            }
        default_voice = next(
            (voice for voice in BAILIAN_VOICE_ROWS if voice.get("audio_path") == DEFAULT_CLONE_AUDIO_PATH),
            BAILIAN_VOICE_ROWS[0] if BAILIAN_VOICE_ROWS else {},
        )
        return {
            "active_preset_id": "",
            "name": str(default_voice.get("name") or "服务默认音色"),
            "source": "default",
            "settings": {
                "model_profile": DEFAULT_MODEL_PROFILE,
                "voice_name": str(default_voice.get("name") or "服务默认音色"),
                "reference_audio_path": str(default_voice.get("audio_path") or DEFAULT_CLONE_AUDIO_PATH),
                "qwen_clone_mode": "xvec",
                "qwen_reference_text": "",
                "qwen_temperature": 0.9,
                "qwen_top_p": 1.0,
                "qwen_top_k": 50,
                "qwen_repetition_penalty": 1.05,
                "qwen_max_new_tokens": 2048,
                "qwen_chunk_size": 8,
                "qwen_min_new_tokens": 2,
                "qwen_seed": 1234,
                "qwen_append_silence": True,
                "qwen_aac_bitrate": "80k",
            },
        }



    ctx = ServiceContext(
        runtime_manager=runtime_manager,
        performance_tuning=performance_tuning,
        playback_coordinator=playback_coordinator,
        jobs=jobs,
        preset_store=preset_store,
        generation_scheduler=generation_scheduler,
        stt_runtime=stt_runtime,
        document_projects=document_projects,
        output_dir=output_dir,
        upload_dir=upload_dir,
        reader_temp_dir=reader_temp_dir,
        ffmpeg_path=ffmpeg_path,
        stt_enabled=stt_enabled,
        expected_session=expected_session,
        access_password=resolved_access_password,
        synthesize_for_profile_runtime=synthesize_for_profile_runtime,
        apply_performance_profile=apply_performance_profile,
        active_service_settings=active_service_settings,
        preset_payload=preset_payload,
        remove_generated_result_files=_remove_generated_result_files,
    )
    register_auth_pages_routes(app, ctx)
    register_generation_routes(app, ctx)
    register_voices_presets_routes(app, ctx)
    register_settings_performance_routes(app, ctx)
    register_playback_routes(app, ctx)
    register_system_control_routes(app, ctx)
    register_documents_routes(app, ctx)
    return app
