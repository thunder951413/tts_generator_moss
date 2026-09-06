# coding=utf-8
"""Playback coordination status and release."""

from __future__ import annotations

import re

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

def register_playback_routes(app, ctx):
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

    def require_internal_listening(request: Request, session_id: str) -> None:
        if str(getattr(request.state, "caller_kind", "external")) != "internal":
            raise HTTPException(status_code=403, detail="listening reservations are restricted to internal clients")
        if not re.fullmatch(r"[A-Za-z0-9_-]{12,128}", str(session_id or "")):
            raise HTTPException(status_code=400, detail="valid listening session_id is required")

    @app.get("/api/playback/status")
    async def playback_status() -> JSONResponse:
        return JSONResponse(playback_coordinator.status())

    @app.post("/api/playback/{lease_id}/release")
    async def release_playback(lease_id: str) -> JSONResponse:
        return JSONResponse(
            {"ok": True, "lease_id": lease_id, "released": playback_coordinator.release(lease_id)}
        )

    @app.post("/api/listening/{session_id}/heartbeat")
    async def heartbeat_listening(request: Request, session_id: str, playback_epoch: int | None = None) -> JSONResponse:
        require_internal_listening(request, session_id)
        if playback_epoch is None:
            return JSONResponse(playback_coordinator.reserve_listening(session_id, ttl=12.0))
        reservation = playback_coordinator.heartbeat_listening(
            session_id, expected_playback_epoch=playback_epoch, ttl=12.0
        )
        if reservation is None:
            raise HTTPException(status_code=409, detail="listening reservation is stale or stopped")
        return JSONResponse(reservation)

    @app.post("/api/listening/{session_id}/release")
    async def release_listening(request: Request, session_id: str) -> JSONResponse:
        require_internal_listening(request, session_id)
        return JSONResponse({"ok": True, "session_id": session_id, "released": playback_coordinator.release_listening(session_id)})
