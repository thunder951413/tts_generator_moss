# coding=utf-8
"""Document/novel project pipeline."""

from __future__ import annotations

import asyncio
import json
import re
import secrets
import threading

from typing import Any

from fastapi import (
    Body,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)

from fastapi.responses import (
    FileResponse,
    JSONResponse,
)

from runtime_manager import (
    DEFAULT_MODEL_PROFILE,
    MODEL_PROFILE_LABELS,
)

from streaming_jobs import DEFAULT_MAX_NEW_TOKENS

from webapp.config import (
    DEFAULT_CLONE_AUDIO_PATH,
    REFERENCE_AUDIO_DIR,
)

from webapp.util import (
    _decode_reference_path,
    _resolve_allowed_reference_audio_path,
    _safe_aac_bitrate,
    _safe_float,
    _safe_int,
)

def register_documents_routes(app, ctx):
    runtime_manager = ctx.runtime_manager
    performance_tuning = ctx.performance_tuning
    playback_coordinator = ctx.playback_coordinator
    jobs = ctx.jobs
    preset_store = ctx.preset_store
    generation_scheduler = ctx.generation_scheduler
    stt_runtime = ctx.stt_runtime
    document_projects = ctx.document_projects
    output_dir = ctx.output_dir
    upload_dir = ctx.upload_dir
    reader_temp_dir = ctx.reader_temp_dir
    ffmpeg_path = ctx.ffmpeg_path
    stt_enabled = ctx.stt_enabled
    expected_session = ctx.expected_session
    resolved_access_password = ctx.access_password
    synthesize_for_profile_runtime = ctx.synthesize_for_profile_runtime
    apply_performance_profile = ctx.apply_performance_profile
    active_service_settings = ctx.active_service_settings
    preset_payload = ctx.preset_payload
    _remove_generated_result_files = ctx.remove_generated_result_files

    def require_tts_admission() -> None:
        if not ctx.tts_enabled or bool(getattr(ctx, "stopping", False)):
            raise HTTPException(status_code=503, detail="TTS service is stopping or disabled")

    async def _acquire_media_playback(
        request: Request,
        session_id: str,
    ) -> tuple[str, dict[str, object]]:
        session_id = str(session_id or "").strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]{12,128}", session_id):
            raise HTTPException(status_code=400, detail="valid playback session_id is required")
        lease_id = f"media:{session_id}"
        disconnected = threading.Event()
        acquire_task = asyncio.create_task(
            asyncio.to_thread(
                playback_coordinator.acquire,
                lease_id,
                str(getattr(request.state, "caller_kind", "external")),
                cancelled=disconnected.is_set,
                timeout=600.0,
                allow_reentrant=True,
                lease_timeout=900.0,
            )
        )
        while not acquire_task.done():
            await asyncio.wait({acquire_task}, timeout=0.25)
            if await request.is_disconnected():
                disconnected.set()
                playback_coordinator.release(lease_id)
                break
        acquired = await acquire_task
        if acquired and await request.is_disconnected():
            playback_coordinator.release(lease_id)
            acquired = False
        if not acquired:
            raise HTTPException(status_code=409, detail="playback was cancelled or queue is full")
        return lease_id, playback_coordinator.status()

    def document_settings(raw: str) -> dict[str, Any]:
        try:
            incoming = json.loads(raw or "{}")
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="invalid document settings") from exc
        service_settings = dict(active_service_settings().get("settings") or {})
        inherited = {
            "model_profile": service_settings.get("model_profile"),
            "reference_audio_path": service_settings.get("reference_audio_path"),
            "voice_name": service_settings.get("voice_name"),
            "qwen_clone_mode": service_settings.get("qwen_clone_mode"),
            "qwen_reference_text": service_settings.get("qwen_reference_text"),
            "temperature": service_settings.get("qwen_temperature"),
            "top_p": service_settings.get("qwen_top_p"),
            "top_k": service_settings.get("qwen_top_k"),
            "repetition_penalty": service_settings.get("qwen_repetition_penalty"),
            "max_new_tokens": service_settings.get("qwen_max_new_tokens"),
            "codec_chunk_frames": service_settings.get("qwen_chunk_size"),
            "qwen_min_new_tokens": service_settings.get("qwen_min_new_tokens"),
            "seed": service_settings.get("qwen_seed"),
            "qwen_append_silence": service_settings.get("qwen_append_silence"),
            "aac_bitrate": service_settings.get("qwen_aac_bitrate"),
        }
        incoming = {key: value for key, value in inherited.items() if value is not None} | incoming
        reference_audio_path = str(incoming.get("reference_audio_path") or DEFAULT_CLONE_AUDIO_PATH)
        try:
            reference_candidate = _resolve_allowed_reference_audio_path(
                reference_audio_path,
                REFERENCE_AUDIO_DIR,
                preset_store.audio_dir,
            )
        except (FileNotFoundError, PermissionError) as exc:
            raise HTTPException(status_code=400, detail="invalid clone reference audio") from exc
        model_profile = str(incoming.get("model_profile") or DEFAULT_MODEL_PROFILE)
        if model_profile not in runtime_manager.profiles:
            raise HTTPException(status_code=400, detail="invalid model profile")
        qwen_profile = runtime_manager.profiles[model_profile].get("backend") == "qwen"
        qwen_clone_mode = "icl" if str(incoming.get("qwen_clone_mode") or "xvec") == "icl" else "xvec"
        qwen_reference_text = str(incoming.get("qwen_reference_text") or "").strip()
        if qwen_profile and qwen_clone_mode == "icl" and not qwen_reference_text:
            raise HTTPException(status_code=400, detail="Qwen ICL克隆模式必须填写参考音频文字")
        configured_seed = _safe_int(incoming.get("seed"), default=1234, minimum=-1, maximum=999999)
        seed_mode = "random" if configured_seed < 0 else "fixed"
        resolved_seed = secrets.randbelow(1_000_000) if configured_seed < 0 else configured_seed
        return {
            "model_profile": model_profile,
            "model_label": MODEL_PROFILE_LABELS[model_profile],
            "runtime_backend": str(runtime_manager.profiles[model_profile]["runtime_backend"]),
            "quant": str(runtime_manager.profiles[model_profile]["quant"]),
            "reference_audio_path": str(reference_candidate),
            "voice_name": str(incoming.get("voice_name") or "本地克隆音色")[:120],
            "language": "Chinese",
            "temperature": _safe_float(incoming.get("temperature"), default=1.7, minimum=0.1, maximum=3.0),
            "top_p": _safe_float(incoming.get("top_p"), default=0.8, minimum=0.1, maximum=1.0),
            "top_k": _safe_int(incoming.get("top_k"), default=25, minimum=1, maximum=200),
            "repetition_penalty": _safe_float(
                incoming.get("repetition_penalty"), default=1.0, minimum=0.8, maximum=2.0
            ),
            "max_new_tokens": _safe_int(
                incoming.get("max_new_tokens"),
                default=2048 if qwen_profile else DEFAULT_MAX_NEW_TOKENS,
                minimum=24 if qwen_profile else 80,
                maximum=2048 if qwen_profile else DEFAULT_MAX_NEW_TOKENS,
            ),
            "codec_chunk_frames": _safe_int(
                incoming.get("codec_chunk_frames"),
                default=8 if qwen_profile else 16,
                minimum=1,
                maximum=24 if qwen_profile else 32,
            ),
            "seed": resolved_seed,
            "configured_seed": configured_seed,
            "seed_mode": seed_mode,
            "qwen_clone_mode": qwen_clone_mode,
            "qwen_reference_text": qwen_reference_text,
            "qwen_non_streaming_mode": bool(incoming.get("qwen_non_streaming_mode", False)),
            "qwen_append_silence": bool(incoming.get("qwen_append_silence", True)),
            "qwen_instruct": "",
            "qwen_min_new_tokens": _safe_int(
                incoming.get("qwen_min_new_tokens"), default=2, minimum=2, maximum=256
            ),
            "aac_bitrate": _safe_aac_bitrate(incoming.get("aac_bitrate"), default="80k"),
            "sample_rate": 48000,
            "channels": 1,
        }

    @app.get("/api/document-projects")
    async def list_document_projects() -> JSONResponse:
        return JSONResponse({"projects": document_projects.list_projects()})

    @app.get("/api/document-projects/{project_id}")
    async def get_document_project(project_id: str) -> JSONResponse:
        try:
            return JSONResponse(document_projects.get_project(project_id))
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="project not found") from exc

    @app.post("/api/document-projects")
    async def create_document_project(
        document: UploadFile = File(...),
        name: str = Form(""),
        settings_json: str = Form("{}"),
        max_chars: int = Form(200),
    ) -> JSONResponse:
        try:
            project = document_projects.create_project(
                name=name,
                filename=document.filename or "document.txt",
                data=await document.read(),
                settings=document_settings(settings_json),
                max_chars=_safe_int(max_chars, default=200, minimum=40, maximum=500),
            )
            return JSONResponse(project)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/document-projects/{project_id}/append")
    async def append_document_project(project_id: str, document: UploadFile = File(...)) -> JSONResponse:
        try:
            return JSONResponse(
                document_projects.append_document(
                    project_id, filename=document.filename or "document.txt", data=await document.read()
                )
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="project not found") from exc
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/document-projects/{project_id}/start")
    async def start_document_project(project_id: str) -> JSONResponse:
        try:
            require_tts_admission()
            project = document_projects.get_project(project_id)
            apply_performance_profile(
                str((project.get("settings") or {}).get("model_profile") or DEFAULT_MODEL_PROFILE)
            )
            # A project is immutable with respect to voice/model parameters.
            # Resume always uses its saved settings, never the page's current controls.
            return JSONResponse(document_projects.start(project_id, settings=None))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="project not found") from exc
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/document-projects/{project_id}/start-selection")
    async def start_document_project_selection(
        project_id: str,
        segment_start: int = Form(0),
        segment_end: int = Form(0),
        settings_json: str = Form(""),
    ) -> JSONResponse:
        try:
            require_tts_admission()
            project = document_projects.get_project(project_id)
            requested_settings = document_settings(settings_json) if settings_json.strip() else None
            apply_performance_profile(
                str(
                    (requested_settings or project.get("settings") or {}).get("model_profile")
                    or DEFAULT_MODEL_PROFILE
                )
            )
            maximum = max(0, len(project.get("segments", [])) - 1)
            start_index = _safe_int(segment_start, default=0, minimum=0, maximum=maximum)
            end_index = _safe_int(
                segment_end,
                default=start_index,
                minimum=start_index,
                maximum=maximum,
            )
            return JSONResponse(
                document_projects.start(
                    project_id,
                    settings=requested_settings,
                    segment_indices=list(range(start_index, end_index + 1)),
                )
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="project not found") from exc
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/document-projects/{project_id}/stop")
    async def stop_document_project(project_id: str) -> JSONResponse:
        try:
            return JSONResponse(document_projects.stop(project_id))
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="project not found") from exc

    @app.delete("/api/document-projects/{project_id}")
    async def delete_document_project(project_id: str) -> JSONResponse:
        try:
            document_projects.delete(project_id)
            return JSONResponse({"ok": True})
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="project not found") from exc
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.put("/api/document-projects/{project_id}")
    async def rename_document_project(
        project_id: str,
        payload: dict[str, Any] = Body(...),
    ) -> JSONResponse:
        try:
            return JSONResponse(
                document_projects.rename(project_id, name=str(payload.get("name") or ""))
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="project not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/document-projects/{project_id}/playback")
    async def save_document_playback(
        project_id: str,
        segment_index: int = Form(0),
        offset_seconds: float = Form(0.0),
    ) -> JSONResponse:
        try:
            return JSONResponse(
                document_projects.update_playback(
                    project_id, segment_index=segment_index, offset_seconds=offset_seconds
                )
            )
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="project not found") from exc

    @app.put("/api/document-projects/{project_id}/segments/{segment_index}")
    async def update_document_segment(
        project_id: str,
        segment_index: int,
        payload: dict[str, Any] = Body(...),
    ) -> JSONResponse:
        try:
            return JSONResponse(
                document_projects.update_segment_text(
                    project_id,
                    segment_index=segment_index,
                    text=str(payload.get("text") or ""),
                )
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="project not found") from exc
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/document-projects/{project_id}/media")
    async def document_project_media(
        request: Request,
        project_id: str,
        path: str,
        playback: int = 0,
        session_id: str = "",
    ) -> FileResponse:
        try:
            media_path = document_projects.media_path(project_id, _decode_reference_path(path))
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="media not found") from exc
        media_type = "audio/mp4" if media_path.suffix.lower() == ".m4a" else "application/octet-stream"
        headers = None
        if bool(_safe_int(playback, default=0, minimum=0, maximum=1)):
            lease_id, playback_status = await _acquire_media_playback(request, session_id)
            headers = {
                "X-Playback-Epoch": str(playback_status["playback_epoch"]),
                "X-Playback-Lease": lease_id,
            }
        return FileResponse(
            str(media_path),
            media_type=media_type,
            filename=media_path.name,
            headers=headers,
        )
