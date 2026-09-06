# coding=utf-8
"""Runtime control, health, tasks and transcription endpoints."""

from __future__ import annotations

import asyncio
import time
import threading

from typing import Any

from fastapi import (
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)

from fastapi.responses import (
    JSONResponse,
    Response,
)

from runtime_manager import (
    DEFAULT_MODEL_PROFILE,
    MODEL_PROFILE_LABELS,
)

from stt_runtime import STTUnavailableError

from streaming_jobs import DEFAULT_MAX_NEW_TOKENS

def register_system_control_routes(app, ctx):
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

    @app.post("/api/service/stop-all")
    async def service_stop_all(request: Request) -> JSONResponse:
        if str(getattr(request.state, "caller_kind", "external")) != "internal":
            raise HTTPException(status_code=403, detail="global stop is restricted to internal clients")
        async with ctx.control_lock:
            ctx.stopping = True
            ctx.tts_epoch += 1
            ctx.stt_epoch += 1
            generation_scheduler.cancel_pending()
            try:
                playback = playback_coordinator.force_stop()
                stopped_jobs = jobs.close_all()
                stopped_projects = document_projects.stop_all()
                interrupted_workers, stt_stopped = await asyncio.gather(
                    asyncio.to_thread(runtime_manager.interrupt_active),
                    asyncio.to_thread(stt_runtime.close),
                )
                await asyncio.to_thread(runtime_manager.close)
            finally:
                ctx.stopping = False
        return JSONResponse(
            {
                "ok": True,
                **playback,
                "stopped_jobs": stopped_jobs,
                "stopped_projects": stopped_projects,
                "interrupted_workers": interrupted_workers,
                "stt_stopped": stt_stopped is None,
            }
        )

    def _internal_service_control(request: Request) -> None:
        if str(getattr(request.state, "caller_kind", "external")) != "internal":
            raise HTTPException(status_code=403, detail="service controls are restricted to internal clients")

    @app.post("/api/tts/stop")
    async def stop_tts_runtime(request: Request) -> JSONResponse:
        _internal_service_control(request)
        async with ctx.control_lock:
            ctx.stopping = True
            ctx.tts_enabled = False
            ctx.tts_epoch += 1
            try:
                playback = playback_coordinator.force_stop()
                stopped_jobs = jobs.close_all()
                stopped_projects = document_projects.stop_all()
                # Job cancellation handles TTS waiters; STT requests remain valid.
                interrupted = await asyncio.to_thread(runtime_manager.interrupt_active)
                await asyncio.to_thread(runtime_manager.close)
            finally:
                ctx.stopping = False
        return JSONResponse(
            {
                "ok": True,
                "tts_enabled": False,
                "stopped_jobs": stopped_jobs,
                "stopped_projects": stopped_projects,
                "interrupted_workers": interrupted,
                **playback,
            }
        )

    @app.post("/api/tts/start")
    async def start_tts_runtime(request: Request) -> JSONResponse:
        _internal_service_control(request)
        if ctx.stopping:
            raise HTTPException(status_code=409, detail="服务正在停止，请稍后启动")
        epoch = ctx.tts_epoch
        ctx.tts_enabled = True
        configured = str(
            (active_service_settings().get("settings") or {}).get(
                "model_profile", DEFAULT_MODEL_PROFILE
            )
        )
        profile_id = configured if configured in runtime_manager.profiles else DEFAULT_MODEL_PROFILE

        def warm_runtime() -> None:
            with generation_scheduler.api_slot("internal", cancelled=lambda: epoch != ctx.tts_epoch):
                if epoch != ctx.tts_epoch:
                    raise RuntimeError("TTS 启动已取消")
                with runtime_manager.session(profile_id):
                    pass

        try:
            await asyncio.to_thread(warm_runtime)
        except Exception as exc:  # noqa: BLE001
            if epoch == ctx.tts_epoch:
                ctx.tts_enabled = False
            raise HTTPException(status_code=503, detail=f"TTS startup failed: {exc}") from exc
        if epoch != ctx.tts_epoch:
            raise HTTPException(status_code=409, detail="TTS 启动已被停止操作取消")
        return JSONResponse({"ok": True, "tts_enabled": True, "runtime": runtime_manager.status()})

    @app.post("/api/stt/stop")
    async def stop_stt_runtime(request: Request) -> JSONResponse:
        _internal_service_control(request)
        if not stt_enabled:
            raise HTTPException(status_code=503, detail="STT is disabled")
        ctx.stt_epoch += 1
        await asyncio.to_thread(stt_runtime.close)
        return JSONResponse({"ok": True, "stt": stt_runtime.status()})

    @app.post("/api/stt/start")
    async def start_stt_runtime(request: Request) -> JSONResponse:
        _internal_service_control(request)
        if not stt_enabled:
            raise HTTPException(status_code=503, detail="STT is disabled")
        if ctx.stopping:
            raise HTTPException(status_code=409, detail="服务正在停止，请稍后启动")
        epoch = ctx.stt_epoch

        def warm_stt() -> None:
            with generation_scheduler.exclusive_slot(
                kind="stt", wait_for_playback=True,
                cancelled=lambda: epoch != ctx.stt_epoch,
            ):
                stt_runtime.start()
        try:
            await asyncio.to_thread(warm_stt)
        except (STTUnavailableError, RuntimeError, TimeoutError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return JSONResponse({"ok": True, "stt": stt_runtime.status()})

    @app.get("/api/runtime")
    async def runtime_info() -> JSONResponse:
        return JSONResponse(
            {
                "output_dir": str(output_dir),
                "upload_dir": str(upload_dir),
                "backend": runtime_manager.qwen_backend,
                "quant": runtime_manager.qwen_quant,
                "qwentts_library": runtime_manager.qwentts_library,
                "device": runtime_manager.device,
                "dtype": runtime_manager.dtype,
                "attn_implementation": runtime_manager.attn_implementation,
                "runtime": runtime_manager.status(),
                "generation_scheduler": {
                    **generation_scheduler.status(),
                    "document_parallel": document_projects.synthesis_workers,
                },
                "playback": playback_coordinator.status(),
                "stt": stt_runtime.status() if stt_enabled else {
                    "state": "disabled",
                    "ready": False,
                    "available": False,
                },
            }
        )

    @app.get("/api/stt/status")
    async def stt_status() -> JSONResponse:
        if not stt_enabled:
            return JSONResponse({"state": "disabled", "ready": False, "available": False})
        return JSONResponse(stt_runtime.status())

    @app.post("/v1/audio/transcriptions")
    async def transcribe_audio(
        request: Request,
        file: UploadFile = File(...),
        model: str = Form("whisper-small"),
        language: str = Form("auto"),
        prompt: str = Form(""),
        response_format: str = Form("json"),
    ) -> Response:
        del model  # OpenAI-compatible field; this service has one resident model.
        if not stt_enabled:
            raise HTTPException(status_code=503, detail="STT is disabled")
        if ctx.stopping:
            raise HTTPException(status_code=503, detail="服务正在停止，请稍后重试")
        if response_format not in {"json", "verbose_json", "text", "srt", "vtt"}:
            raise HTTPException(status_code=400, detail="unsupported response_format")
        if ctx.stt_requests >= 16:
            raise HTTPException(status_code=429, detail="STT 等待队列已满，请稍后重试")
        ctx.stt_requests += 1
        epoch = ctx.stt_epoch
        disconnected = threading.Event()
        work = None
        try:
            # Multipart uploads are already disk-spooled by Starlette. Keep
            # waiters on that spool rather than retaining 16 x 500 MB in RAM.
            if file.size is not None and file.size > 500 * 1024 * 1024:
                raise HTTPException(status_code=413, detail="audio file exceeds 500 MB")
            if file.size == 0:
                raise HTTPException(status_code=400, detail="audio file is empty")

            def cancelled() -> bool:
                return disconnected.is_set() or epoch != ctx.stt_epoch

            def transcribe_when_idle():
                with generation_scheduler.exclusive_slot(
                    kind="stt", cancelled=cancelled, wait_for_playback=True, timeout=600,
                ):
                    if cancelled():
                        raise RuntimeError("STT 请求已取消")
                    audio = bytearray()
                    file.file.seek(0)
                    while chunk := file.file.read(1024 * 1024):
                        if cancelled():
                            raise RuntimeError("STT 请求已取消")
                        audio.extend(chunk)
                        if len(audio) > 500 * 1024 * 1024:
                            raise HTTPException(status_code=413, detail="audio file exceeds 500 MB")
                    if not audio:
                        raise HTTPException(status_code=400, detail="audio file is empty")
                    result = stt_runtime.transcribe(
                        audio=audio, filename=file.filename or "audio.wav",
                        language=(language or "auto").strip(),
                        prompt=(prompt or "").strip(), response_format=response_format,
                        cancelled=cancelled,
                    )
                    if cancelled():
                        raise RuntimeError("STT 请求已取消")
                    return result

            work = asyncio.create_task(asyncio.to_thread(transcribe_when_idle))
            while not work.done():
                await asyncio.wait({work}, timeout=0.2)
                if await request.is_disconnected():
                    disconnected.set()
            payload, media_type = await work
        except TimeoutError as exc:
            raise HTTPException(status_code=504, detail=str(exc)) from exc
        except (STTUnavailableError, RuntimeError, ValueError, OSError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        finally:
            disconnected.set()
            ctx.stt_requests -= 1
            # Cancelling an HTTP coroutine does not cancel an already-running
            # thread. Consume its eventual outcome; queued work sees the event.
            if work is not None and not work.done():
                work.add_done_callback(lambda task: task.exception() if not task.cancelled() else None)
        return Response(content=payload, media_type=media_type)

    @app.get("/api/health")
    async def health() -> JSONResponse:
        return JSONResponse(
            {
                **runtime_manager.status(),
                "tts_enabled": ctx.tts_enabled,
                "generation_scheduler": {
                    **generation_scheduler.status(),
                    "document_parallel": document_projects.synthesis_workers,
                },
                "playback": playback_coordinator.status(),
                "stt": stt_runtime.status() if stt_enabled else {
                    "state": "disabled",
                    "ready": False,
                    "available": False,
                },
            }
        )

    @app.get("/api/service/tasks")
    async def service_tasks() -> JSONResponse:
        tasks: list[dict[str, Any]] = []
        for item in jobs.list(limit=100):
            state = str(item.get("state") or "unknown")
            tasks.append(
                {
                    "type": "text",
                    "id": item["job_id"],
                    "title": item.get("title") or item.get("text_preview") or "文本生成",
                    "state": state,
                    "active": state in {"queued", "loading_runtime", "running"},
                    "created_at": item.get("created_at"),
                    "updated_at": item.get("updated_at"),
                    "model_profile": item.get("model_profile") or DEFAULT_MODEL_PROFILE,
                    "model_label": item.get("model_label") or MODEL_PROFILE_LABELS.get(
                        str(item.get("model_profile") or DEFAULT_MODEL_PROFILE), ""
                    ),
                    "voice_name": item.get("voice_name") or "",
                    "seed": item.get("seed"),
                    "configured_seed": item.get("configured_seed"),
                    "seed_mode": item.get("seed_mode"),
                    "generated_frames": item.get("generated_frames", 0),
                    "max_new_tokens": item.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS),
                    "duration_seconds": item.get("emitted_audio_seconds", 0.0),
                    "result_ready": bool(item.get("result_ready")),
                    "error": item.get("error"),
                }
            )
        for project in document_projects.list_projects():
            state = str(project.get("state") or "unknown")
            stats = project.get("stats") or {}
            tasks.append(
                {
                    "type": "document",
                    "id": project["id"],
                    "title": project.get("name") or "文档项目",
                    "state": state,
                    "active": state in {"running", "stopping"},
                    "created_at": project.get("created_at"),
                    "updated_at": project.get("updated_at"),
                    "model_profile": (project.get("settings") or {}).get("model_profile", DEFAULT_MODEL_PROFILE),
                    "model_label": (project.get("settings") or {}).get("model_label") or MODEL_PROFILE_LABELS[
                        (project.get("settings") or {}).get("model_profile", DEFAULT_MODEL_PROFILE)
                    ],
                    "voice_name": (project.get("settings") or {}).get("voice_name", ""),
                    "seed": (project.get("settings") or {}).get("seed"),
                    "configured_seed": (project.get("settings") or {}).get("configured_seed"),
                    "seed_mode": (project.get("settings") or {}).get("seed_mode"),
                    "completed_segments": stats.get("completed_segments", 0),
                    "total_segments": stats.get("total_segments", 0),
                    "progress": stats.get("progress", 0.0),
                    "eta_seconds": stats.get("eta_seconds"),
                    "error": project.get("message") if state == "error" else None,
                }
            )
        tasks.sort(
            key=lambda item: (bool(item.get("active")), float(item.get("updated_at") or 0)),
            reverse=True,
        )
        return JSONResponse(
            {
                "tasks": tasks[:100],
                "active_count": sum(1 for item in tasks if item.get("active")),
                "server_time": time.time(),
            }
        )
