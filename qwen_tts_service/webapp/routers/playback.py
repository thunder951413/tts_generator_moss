# coding=utf-8
"""Playback coordination status and release."""

from __future__ import annotations

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

    @app.get("/api/playback/status")
    async def playback_status() -> JSONResponse:
        return JSONResponse(playback_coordinator.status())

    @app.post("/api/playback/{lease_id}/release")
    async def release_playback(lease_id: str) -> JSONResponse:
        return JSONResponse(
            {"ok": True, "lease_id": lease_id, "released": playback_coordinator.release(lease_id)}
        )
