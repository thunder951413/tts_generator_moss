# coding=utf-8
"""Voice presets and reference audio library."""

from __future__ import annotations

import asyncio
import hashlib
import mimetypes
import subprocess
import uuid

from pathlib import Path

from typing import Any

from urllib.parse import quote

from fastapi import (
    Body,
    File,
    HTTPException,
    UploadFile,
)

from fastapi.responses import (
    FileResponse,
    JSONResponse,
)

from runtime_manager import MODEL_PROFILE_LABELS
from media_process import run_media_process

from webapp.config import (
    BAILIAN_VOICE_ROWS,
    DEFAULT_CLONE_AUDIO_PATH,
    REFERENCE_AUDIO_DIR,
)

from webapp.util import _resolve_allowed_reference_audio_path

def register_voices_presets_routes(app, ctx):
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

    @app.get("/api/reference-audio")
    async def reference_audio(path: str) -> FileResponse:
        try:
            candidate = _resolve_allowed_reference_audio_path(
                path,
                REFERENCE_AUDIO_DIR,
                preset_store.audio_dir,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="reference audio not found") from exc
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail="reference audio path is not allowed") from exc
        media_type = mimetypes.guess_type(str(candidate))[0] or "application/octet-stream"
        return FileResponse(str(candidate), media_type=media_type, filename=candidate.name)

    @app.get("/api/presets")
    async def list_presets() -> JSONResponse:
        active = preset_store.active()
        return JSONResponse(
            {
                "presets": preset_store.list(),
                "active_preset_id": active["id"] if active else "",
            }
        )

    def builtin_reference_id(path: str) -> str:
        return "builtin-" + hashlib.sha256(str(Path(path).resolve()).encode()).hexdigest()[:24]

    def reference_audio_library(*, include_hidden: bool) -> list[dict[str, Any]]:
        book_usages = document_projects.reference_usage_index()
        usage_cache: dict[str, list[str]] = {}

        def reference_usages(path: str) -> list[str]:
            if path not in usage_cache:
                usage_cache[path] = (preset_store.reference_usage(path)
                    + book_usages.get(str(Path(path).resolve()), []) + jobs.reference_usage(path))
            return usage_cache[path]

        hidden_builtin = preset_store.hidden_builtin_references()
        rows: list[dict[str, Any]] = []
        for voice in BAILIAN_VOICE_ROWS:
            path = str(Path(str(voice.get("audio_path") or "")).resolve())
            hidden = path in hidden_builtin
            if hidden and not include_hidden:
                continue
            rows.append(
                {
                    "id": builtin_reference_id(path),
                    "kind": "builtin",
                    "name": str(voice.get("name") or Path(path).stem),
                    "description": str(voice.get("description") or ""),
                    "path": path,
                    "audio_path": path,
                    "language": str(voice.get("language") or "Chinese"),
                    "transcript": str(voice.get("transcript") or ""),
                    "transcript_source": str(voice.get("transcript_source") or ""),
                    "hidden": hidden,
                    "in_use": bool(reference_usages(path)),
                    "usages": reference_usages(path),
                }
            )
        for item in preset_store.list_reference_audio(include_hidden=include_hidden):
            path = str(item.get("path") or "")
            rows.append(
                {
                    **item,
                    "kind": "custom",
                    "audio_path": path,
                    "description": "用户参考音频",
                    "language": "Chinese",
                    "transcript": "",
                    "transcript_source": "",
                    "in_use": bool(reference_usages(path)),
                    "usages": reference_usages(path),
                }
            )
        return rows

    @app.get("/api/voices")
    async def list_native_voices() -> JSONResponse:
        visible_references = reference_audio_library(include_hidden=False)
        return JSONResponse(
            {
                "voices": visible_references,
                "default_reference_audio_path": DEFAULT_CLONE_AUDIO_PATH,
                "models": [
                    {
                        "id": profile_id,
                        "label": MODEL_PROFILE_LABELS[profile_id],
                    }
                    for profile_id in ("qwen_0_6b", "qwen_1_7b")
                ],
            }
        )

    @app.get("/api/reference-audio-library")
    async def list_reference_audio_library(include_hidden: bool = False) -> JSONResponse:
        return JSONResponse(
            {"references": reference_audio_library(include_hidden=include_hidden)}
        )

    @app.put("/api/reference-audio-library/{reference_id}/visibility")
    async def update_reference_audio_visibility(
        reference_id: str,
        payload: dict[str, Any] = Body(...),
    ) -> JSONResponse:
        hidden = bool(payload.get("hidden"))
        builtin = next(
            (
                voice
                for voice in BAILIAN_VOICE_ROWS
                if builtin_reference_id(str(voice.get("audio_path") or "")) == reference_id
            ),
            None,
        )
        if builtin is not None:
            preset_store.set_builtin_hidden(str(builtin["audio_path"]), hidden)
        else:
            try:
                preset_store.set_reference_hidden(reference_id, hidden)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail="reference audio not found") from exc
        return JSONResponse({"ok": True, "hidden": hidden})

    @app.put("/api/reference-audio-library/{reference_id}")
    async def rename_reference_audio(
        reference_id: str,
        payload: dict[str, Any] = Body(...),
    ) -> JSONResponse:
        if any(
            builtin_reference_id(str(voice.get("audio_path") or "")) == reference_id
            for voice in BAILIAN_VOICE_ROWS
        ):
            raise HTTPException(status_code=400, detail="内置参考音频不能改名")
        try:
            reference = preset_store.rename_reference_audio(reference_id, payload.get("name"))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="reference audio not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return JSONResponse({"ok": True, "reference": reference})

    @app.delete("/api/reference-audio-library/{reference_id}")
    async def delete_reference_audio(
        reference_id: str,
        replace_usages: bool = False,
    ) -> JSONResponse:
        if any(
            builtin_reference_id(str(voice.get("audio_path") or "")) == reference_id
            for voice in BAILIAN_VOICE_ROWS
        ):
            raise HTTPException(status_code=400, detail="内置参考音频只能隐藏，不能删除")
        default_voice = next(
            (
                voice
                for voice in BAILIAN_VOICE_ROWS
                if voice.get("audio_path") == DEFAULT_CLONE_AUDIO_PATH
            ),
            BAILIAN_VOICE_ROWS[0] if BAILIAN_VOICE_ROWS else {},
        )
        try:
            reference = preset_store.reference_audio_record(reference_id)
            protected_usages = document_projects.reference_usage(reference["path"]) + jobs.reference_usage(reference["path"])
            if protected_usages:
                raise ValueError("该音频仍被" + "、".join(protected_usages) + "引用，请先移除书籍或等待生成任务结束")
            replaced_usages = preset_store.delete_reference_audio(
                reference_id,
                replacement_audio_path=(
                    str(default_voice.get("audio_path") or DEFAULT_CLONE_AUDIO_PATH)
                    if replace_usages
                    else ""
                ),
                replacement_voice_name=(
                    str(default_voice.get("name") or "服务默认音色")
                    if replace_usages
                    else ""
                ),
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="reference audio not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return JSONResponse(
            {
                "ok": True,
                "replaced_usages": replaced_usages,
                "replacement": (
                    {
                        "name": str(default_voice.get("name") or "服务默认音色"),
                        "reference_audio_path": str(
                            default_voice.get("audio_path") or DEFAULT_CLONE_AUDIO_PATH
                        ),
                    }
                    if replaced_usages
                    else None
                ),
            }
        )

    @app.post("/api/presets")
    async def create_preset(payload: dict[str, Any] = Body(...)) -> JSONResponse:
        try:
            name, settings = preset_payload(payload)
            return JSONResponse(preset_store.create(name=name, settings=settings), status_code=201)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.put("/api/presets/{preset_id}")
    async def update_preset(preset_id: str, payload: dict[str, Any] = Body(...)) -> JSONResponse:
        try:
            name, settings = preset_payload(payload)
            return JSONResponse(preset_store.update(preset_id, name=name, settings=settings))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="preset not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.delete("/api/presets/{preset_id}")
    async def delete_preset(preset_id: str) -> JSONResponse:
        try:
            preset_store.delete(preset_id)
            return JSONResponse({"ok": True})
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="preset not found") from exc

    @app.post("/api/presets/reference-audio")
    async def import_preset_reference_audio(audio: UploadFile = File(...)) -> JSONResponse:
        filename = audio.filename or "reference.wav"
        temporary = upload_dir / f"preset-{uuid.uuid4().hex}{Path(filename).suffix or '.media'}"
        normalized = upload_dir / f"preset-normalized-{uuid.uuid4().hex}.wav"
        try:
            total = 0
            with temporary.open("wb") as output:
                while chunk := await audio.read(1024 * 1024):
                    total += len(chunk)
                    if total > 1024 * 1024 * 1024:
                        raise HTTPException(
                            status_code=413,
                            detail="导入的音频或视频不能超过 1 GB",
                        )
                    output.write(chunk)
            if total == 0:
                raise HTTPException(status_code=400, detail="导入的音频或视频为空")

            conversion = await asyncio.to_thread(
                run_media_process,
                [
                    ffmpeg_path,
                    "-y",
                    "-v",
                    "error",
                    "-i",
                    str(temporary),
                    "-map",
                    "0:a:0",
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    "24000",
                    "-c:a",
                    "pcm_s16le",
                    str(normalized),
                ],
                cancelled=lambda: False,
                timeout=300,
            )
            if conversion.returncode != 0 or not normalized.is_file() or normalized.stat().st_size <= 44:
                detail = (conversion.stderr or "").strip().splitlines()
                reason = detail[-1] if detail else "无法识别媒体编码或文件中没有音轨"
                raise ValueError(f"无法从音频或视频读取声音：{reason[:240]}")
            imported = preset_store.import_reference_audio(
                filename=f"{Path(filename).stem}.wav",
                temporary_path=normalized,
            )
        except subprocess.TimeoutExpired as exc:
            raise HTTPException(status_code=400, detail="音视频处理超时") from exc
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=503,
                detail="缺少音视频处理组件 ffmpeg",
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            await audio.close()
            temporary.unlink(missing_ok=True)
            normalized.unlink(missing_ok=True)
        return JSONResponse(
            {
                "reference_audio_path": str(imported),
                "audio_url": f"/api/reference-audio?path={quote(str(imported), safe='')}",
                "reference": next(
                    (
                        item
                        for item in preset_store.list_reference_audio(include_hidden=True)
                        if item.get("path") == str(imported)
                    ),
                    None,
                ),
            }
        )
