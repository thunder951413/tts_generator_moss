# coding=utf-8
"""Login, logout and page-level routes."""

from __future__ import annotations

import hmac

from fastapi import (
    Form,
    Request,
)

from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    RedirectResponse,
)

from webapp.config import (
    NOVEL_READER_WEB_DIR,
    SERVICE_AUTH_COOKIE,
)

from webapp.pages import _login_html

def register_auth_pages_routes(app, ctx):
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

    @app.get("/login", response_class=HTMLResponse)
    async def login_page(request: Request):
        if not resolved_access_password:
            return RedirectResponse(url="/", status_code=303)
        supplied = request.cookies.get(SERVICE_AUTH_COOKIE, "")
        if hmac.compare_digest(supplied, expected_session):
            return RedirectResponse(url="/", status_code=303)
        next_path = request.query_params.get("next", "/")
        if not next_path.startswith("/") or next_path.startswith("//"):
            next_path = "/"
        return HTMLResponse(_login_html(next_path=next_path, error=""))

    @app.post("/login", response_class=HTMLResponse)
    async def login_submit(password: str = Form(""), next_path: str = Form("/")):
        if not resolved_access_password:
            return RedirectResponse(url="/", status_code=303)
        if not hmac.compare_digest(str(password), resolved_access_password):
            return HTMLResponse(_login_html(next_path=next_path, error="密码不正确"), status_code=401)
        if not next_path.startswith("/") or next_path.startswith("//"):
            next_path = "/"
        response = RedirectResponse(url=next_path, status_code=303)
        response.set_cookie(
            SERVICE_AUTH_COOKIE,
            expected_session,
            max_age=30 * 24 * 60 * 60,
            httponly=True,
            samesite="strict",
        )
        return response

    @app.post("/logout")
    async def logout() -> RedirectResponse:
        response = RedirectResponse(url="/login", status_code=303)
        response.delete_cookie(SERVICE_AUTH_COOKIE)
        return response

    @app.get("/")
    async def index() -> RedirectResponse:
        return RedirectResponse(url="/reader", status_code=307)

    @app.get("/reader")
    async def novel_reader() -> FileResponse:
        return FileResponse(
            str(NOVEL_READER_WEB_DIR / "index.html"),
            media_type="text/html",
        )
