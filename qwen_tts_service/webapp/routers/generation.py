# coding=utf-8
"""Streaming text-generation job lifecycle."""

from __future__ import annotations

import asyncio
import os
import queue
import secrets
import subprocess
import threading
import time
import uuid

from pathlib import Path

from typing import Any

from fastapi import (
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)

from fastapi.responses import (
    FileResponse,
    JSONResponse,
    StreamingResponse,
)

from qwen_protocol import StreamingRequest
from media_process import run_media_process

from runtime_manager import (
    DEFAULT_MODEL_PROFILE,
    MODEL_PROFILE_LABELS,
)

from streaming_jobs import (
    DEFAULT_MAX_NEW_TOKENS,
    StreamingJob,
)

from webapp.config import (
    DEFAULT_CLONE_AUDIO_PATH,
    MODE_CLONE,
    REFERENCE_AUDIO_DIR,
)

from webapp.util import (
    _pcm16le_bytes,
    _resolve_allowed_reference_audio_path,
    _safe_aac_bitrate,
    _safe_float,
    _safe_int,
)

def register_generation_routes(app, ctx):
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
    aac_flights: dict[str, asyncio.Task[Path]] = {}
    aac_flights_lock = asyncio.Lock()

    def _tts_admission_open(epoch: int) -> bool:
        return (
            bool(ctx.tts_enabled)
            and not bool(getattr(ctx, "stopping", False))
            and int(getattr(ctx, "tts_epoch", 0)) == epoch
        )

    def _remove_owned_prompt_audio(job: StreamingJob) -> None:
        path = str(job.snapshot().get("owned_prompt_audio_path") or "")
        if not path:
            return
        try:
            candidate = Path(path).resolve()
            candidate.relative_to(upload_dir.resolve())
            candidate.unlink(missing_ok=True)
        except (OSError, RuntimeError, ValueError):
            return

    def _put_stream_audio(job: StreamingJob, pcm_bytes: bytes) -> None:
        with job.status_lock:
            if job.is_closed:
                return
        try:
            job.audio_queue.put_nowait(pcm_bytes)
        except queue.Full:
            raise RuntimeError("lossless audio queue capacity exceeded")

    def _finish_stream_audio(job: StreamingJob) -> None:
        job.audio_queue.put_nowait(None)

    def _run_job(
        job: StreamingJob, request: StreamingRequest, mode_name: str,
        streaming_generation: bool, model_profile: str, caller_kind: str,
    ) -> None:
        admitted_epoch = int(job.snapshot().get("tts_epoch", 0))

        def cancelled() -> bool:
            return (
                job.is_closed
                or bool(getattr(ctx, "stopping", False))
                or int(getattr(ctx, "tts_epoch", 0)) != admitted_epoch
            )

        try:
            if cancelled():
                return
            job.update(
                state="loading_runtime",
                started_at=time.time(),
                max_new_tokens=int(request.max_new_frames),
                mode=mode_name,
                model_profile=model_profile,
                streaming_generation=streaming_generation,
            )
            with generation_scheduler.api_slot(
                caller_kind,
                cancelled=cancelled,
            ):
              if cancelled():
                return
              with runtime_manager.session(model_profile) as runtime:
                if cancelled():
                    return
                channels = int(runtime_manager.profiles[model_profile]["channels"])
                job.update(state="running", sample_rate=runtime.sample_rate, channels=channels, n_vq=runtime.n_vq)
                for event in synthesize_for_profile_runtime(runtime, request, output_dir=output_dir):
                    with job.status_lock:
                        if cancelled():
                            if event.type == "result" and job.status.get("ephemeral_audio"):
                                _remove_generated_result_files(event.data)
                            break
                    if event.type == "metadata":
                        job.update(**event.data)
                    elif event.type == "progress":
                        job.update(**event.data)
                    elif event.type == "audio":
                        waveform = event.data["waveform"]
                        channels = 1 if waveform.ndim == 1 else int(min(2, waveform.shape[0]))
                        with job.status_lock:
                            if job.status.get("first_audio_at") is None:
                                job.status["first_audio_at"] = time.time()
                        if streaming_generation:
                            _put_stream_audio(job, _pcm16le_bytes(waveform, channels))
                        job.update(
                            generated_frames=event.data.get("generated_frames", job.snapshot().get("generated_frames", 0)),
                            emitted_audio_seconds=event.data.get("emitted_audio_seconds", 0.0),
                            generated_audio_seconds=event.data.get("generated_audio_seconds", 0.0),
                            sample_rate=event.data.get("sample_rate", runtime.sample_rate),
                            channels=channels,
                            lead_seconds=event.data.get("lead_seconds", 0.0),
                            generation_lead_seconds=event.data.get("generation_lead_seconds", 0.0),
                            playback_lead_seconds=event.data.get("playback_lead_seconds"),
                            generation_realtime_factor=event.data.get("generation_realtime_factor", 0.0),
                            post_first_generation_realtime_factor=event.data.get(
                                "post_first_generation_realtime_factor"
                            ),
                            first_audio_latency_seconds=event.data.get("first_audio_latency_seconds"),
                            decode_chunks_submitted=event.data.get("decode_chunks_submitted", 0),
                            decode_queue_depth=event.data.get("decode_queue_depth", 0),
                            pending_decode_frames=event.data.get("pending_decode_frames", 0),
                            chunk_frames=event.data.get("chunk_frames", 0),
                        )
                    elif event.type == "result":
                        metadata = dict(event.data["metadata"])
                        task_seed_status = job.snapshot()
                        metadata["seed"] = int(request.seed) if request.seed is not None else None
                        # After a runaway-retry the worker regenerated with a
                        # different seed; report the seed that made the audio.
                        effective_seed = metadata.get("effective_seed")
                        if effective_seed is not None:
                            metadata["seed"] = int(effective_seed)
                        metadata["configured_seed"] = task_seed_status.get("configured_seed")
                        metadata["seed_mode"] = task_seed_status.get("seed_mode")
                        job.set_result({
                            "audio_path": event.data["audio_path"],
                            "tokens_path": event.data["tokens_path"],
                            "metadata_path": event.data["metadata_path"],
                            "metadata": metadata,
                        })
                        job.update(
                            state="finished",
                            generated_frames=metadata.get("generated_frames", 0),
                            emitted_audio_seconds=metadata.get("duration_seconds", 0.0),
                            audio_path=event.data["audio_path"],
                        )
            _finish_stream_audio(job)
        except Exception as exc:  # noqa: BLE001
            if job.is_closed:
                job.update(state="closed", error=None, closed=True)
            else:
                job.update(state="error", error=str(exc))
            _finish_stream_audio(job)
        finally:
            _remove_owned_prompt_audio(job)

    @app.post("/api/generate-stream/start")
    async def generate_stream_start(
        http_request: Request,
        mode: str = Form("voice_clone"),
        language: str = Form(""),
        text: str = Form(...),
        prompt_text: str = Form(""),
        max_new_tokens: int | None = Form(None),
        codec_chunk_frames: int | None = Form(None),
        seed: int | None = Form(None),
        tokens_control: int = Form(0),
        tokens: int = Form(0),
        temperature: float | None = Form(None),
        top_p: float | None = Form(None),
        top_k: int | None = Form(None),
        repetition_penalty: float | None = Form(None),
        model_profile: str = Form(""),
        voice_name: str = Form(""),
        streaming_generation: int | None = Form(None),
        qwen_clone_mode: str = Form(""),
        qwen_reference_text: str = Form(""),
        qwen_non_streaming_mode: int | None = Form(None),
        qwen_append_silence: int | None = Form(None),
        qwen_instruct: str = Form(""),
        qwen_min_new_tokens: int | None = Form(None),
        example_audio_path: str = Form(""),
        prompt_audio: UploadFile | None = File(None),
        use_service_settings: int = Form(1),
        ephemeral_audio: int = Form(0),
        expected_playback_epoch: int | None = Form(None),
    ) -> JSONResponse:
        admission_epoch = int(getattr(ctx, "tts_epoch", 0))
        if not _tts_admission_open(admission_epoch):
            raise HTTPException(status_code=503, detail="TTS service is stopped")
        admission_playback_epoch = int(playback_coordinator.status()["playback_epoch"])
        if (
            expected_playback_epoch is not None
            and int(expected_playback_epoch) != admission_playback_epoch
        ):
            raise HTTPException(status_code=409, detail="playback session is stale")
        text = (text or "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="text must not be empty")

        mode = "voice_clone"
        caller_kind = str(getattr(http_request.state, "caller_kind", "external"))
        language = "Chinese"
        mode_name = MODE_CLONE
        service_settings = dict(active_service_settings().get("settings") or {})
        use_applied_service_settings = bool(
            _safe_int(use_service_settings, default=1, minimum=0, maximum=1)
        )
        def service_value(explicit: Any, key: str, fallback: Any) -> Any:
            if use_applied_service_settings:
                return service_settings.get(key, fallback)
            return explicit if explicit is not None and explicit != "" else fallback

        max_new_tokens = service_value(max_new_tokens, "qwen_max_new_tokens", 2048)
        codec_chunk_frames = service_value(codec_chunk_frames, "qwen_chunk_size", 8)
        seed = service_value(seed, "qwen_seed", 1234)
        temperature = service_value(temperature, "qwen_temperature", 0.9)
        top_p = service_value(top_p, "qwen_top_p", 1.0)
        top_k = service_value(top_k, "qwen_top_k", 50)
        repetition_penalty = (
            service_value(repetition_penalty, "qwen_repetition_penalty", 1.05)
        )
        model_profile = str(service_value(model_profile, "model_profile", DEFAULT_MODEL_PROFILE))
        voice_name = str(service_value(voice_name, "voice_name", "本地克隆音色"))
        # Delivery mode belongs to the caller. The applied service setting owns
        # the voice and synthesis parameters, while a reader may still choose
        # realtime streaming or complete-block playback.
        streaming_generation = _safe_int(
            streaming_generation if streaming_generation is not None
            else service_settings.get("qwen_streaming_generation", 1),
            default=1,
            minimum=0,
            maximum=1,
        )
        qwen_clone_mode = str(service_value(qwen_clone_mode, "qwen_clone_mode", "xvec"))
        qwen_reference_text = str(service_value(
            qwen_reference_text, "qwen_reference_text", ""
        ))
        qwen_non_streaming_mode = _safe_int(
            qwen_non_streaming_mode if qwen_non_streaming_mode is not None
            else service_settings.get("qwen_non_streaming_mode", 0),
            default=0,
            minimum=0,
            maximum=1,
        )
        qwen_append_silence = int(bool(service_value(
            qwen_append_silence, "qwen_append_silence", True
        )))
        qwen_min_new_tokens = service_value(qwen_min_new_tokens, "qwen_min_new_tokens", 2)
        example_audio_path = str(service_value(
            example_audio_path, "reference_audio_path", ""
        ))
        if use_applied_service_settings:
            prompt_audio = None
        if model_profile not in runtime_manager.profiles:
            raise HTTPException(status_code=400, detail="invalid model profile")

        prompt_audio_path = ""
        owned_prompt_path: Path | None = None
        if prompt_audio is not None and prompt_audio.filename:
            suffix = Path(prompt_audio.filename).suffix or ".wav"
            prompt_path = upload_dir / f"{uuid.uuid4().hex}{suffix}"
            total = 0
            try:
                with prompt_path.open("xb") as destination:
                    while chunk := await prompt_audio.read(1024 * 1024):
                        total += len(chunk)
                        if total > 50 * 1024 * 1024:
                            raise HTTPException(status_code=413, detail="prompt audio exceeds 50 MB")
                        destination.write(chunk)
                prompt_audio_path = str(prompt_path)
                owned_prompt_path = prompt_path
            except Exception:
                prompt_path.unlink(missing_ok=True)
                raise
        elif example_audio_path:
            try:
                prompt_audio_path = str(
                    _resolve_allowed_reference_audio_path(
                        example_audio_path,
                        REFERENCE_AUDIO_DIR,
                        preset_store.audio_dir,
                    )
                )
            except (FileNotFoundError, PermissionError):
                prompt_audio_path = ""

        if not prompt_audio_path and DEFAULT_CLONE_AUDIO_PATH:
            prompt_audio_path = DEFAULT_CLONE_AUDIO_PATH
        if not prompt_audio_path:
            raise HTTPException(status_code=500, detail="no clone reference audio is available")

        max_new_tokens = _safe_int(
            max_new_tokens,
            default=DEFAULT_MAX_NEW_TOKENS,
            minimum=1,
            maximum=DEFAULT_MAX_NEW_TOKENS,
        )
        codec_chunk_frames = _safe_int(codec_chunk_frames, default=16, minimum=0, maximum=32)
        streaming_generation_enabled = bool(_safe_int(streaming_generation, default=1, minimum=0, maximum=1))
        if runtime_manager.profiles[model_profile].get("backend") == "qwen":
            max_new_tokens = _safe_int(max_new_tokens, default=2048, minimum=2, maximum=2048)
            codec_chunk_frames = _safe_int(codec_chunk_frames, default=8, minimum=1, maximum=24)
        configured_seed = _safe_int(seed, default=1234, minimum=-1, maximum=999999)
        seed_mode = "random" if configured_seed < 0 else "fixed"
        resolved_seed = secrets.randbelow(1_000_000) if configured_seed < 0 else configured_seed
        if not runtime_manager.profiles[model_profile]["streaming"]:
            streaming_generation_enabled = False
        try:
            performance_recommendation = apply_performance_profile(model_profile)
        except Exception:
            if owned_prompt_path is not None:
                owned_prompt_path.unlink(missing_ok=True)
            raise
        if (
            streaming_generation_enabled
            and use_applied_service_settings
            and performance_tuning.profile(model_profile) is not None
        ):
            codec_chunk_frames = performance_recommendation["stream_chunk_frames"]
        request = StreamingRequest(
            text=text,
            mode="voice_clone",
            prompt_text="",
            prompt_audio_path=prompt_audio_path,
            language="Chinese",
            tokens_control=bool(int(tokens_control)),
            tokens=_safe_int(tokens, default=0, minimum=0),
            max_new_frames=max_new_tokens,
            do_sample=True,
            temperature=_safe_float(temperature, default=1.7, minimum=0.1, maximum=3.0),
            top_p=_safe_float(top_p, default=0.8, minimum=0.1, maximum=1.0),
            top_k=_safe_int(top_k, default=25, minimum=1, maximum=200),
            repetition_penalty=_safe_float(repetition_penalty, default=1.0, minimum=0.8, maximum=2.0),
            seed=None if resolved_seed < 0 else resolved_seed,
            codec_chunk_frames=codec_chunk_frames,
            qwen_xvec_only=str(qwen_clone_mode or "xvec") != "icl",
            qwen_reference_text=str(qwen_reference_text or "").strip(),
            qwen_non_streaming_mode=bool(
                _safe_int(qwen_non_streaming_mode, default=0, minimum=0, maximum=1)
            ),
            qwen_append_silence=bool(
                _safe_int(qwen_append_silence, default=1, minimum=0, maximum=1)
            ),
            qwen_instruct="",
            qwen_min_new_tokens=_safe_int(
                qwen_min_new_tokens, default=2, minimum=2, maximum=256
            ),
        )
        if not _tts_admission_open(admission_epoch):
            if owned_prompt_path is not None:
                owned_prompt_path.unlink(missing_ok=True)
            raise HTTPException(status_code=503, detail="TTS service stopped while receiving the request")
        if (
            expected_playback_epoch is not None
            and int(expected_playback_epoch)
            != int(playback_coordinator.status()["playback_epoch"])
        ):
            if owned_prompt_path is not None:
                owned_prompt_path.unlink(missing_ok=True)
            raise HTTPException(status_code=409, detail="playback session is stale")
        try:
            job = jobs.create(
                summary={
                    "task_type": "text_generation",
                    "title": text[:80],
                    "text_preview": text[:240],
                    "model_profile": model_profile,
                    "model_label": MODEL_PROFILE_LABELS[model_profile],
                    "voice_name": str(voice_name or "本地克隆音色")[:120],
                    "seed": resolved_seed,
                    "configured_seed": configured_seed,
                    "seed_mode": seed_mode,
                    "ephemeral_audio": bool(
                        _safe_int(ephemeral_audio, default=0, minimum=0, maximum=1)
                    ),
                    "caller_kind": caller_kind,
                    "playback_epoch": admission_playback_epoch,
                    "reference_audio_path": prompt_audio_path,
                    "owned_prompt_audio_path": str(owned_prompt_path) if owned_prompt_path else "",
                    "tts_epoch": admission_epoch,
                }
            )
        except Exception:
            if owned_prompt_path is not None:
                owned_prompt_path.unlink(missing_ok=True)
            raise
        thread = threading.Thread(
            target=_run_job,
            args=(
                job,
                request,
                mode_name,
                streaming_generation_enabled,
                model_profile,
                caller_kind,
            ),
            daemon=True,
        )
        job.thread = thread
        try:
            thread.start()
        except Exception:
            jobs.close(job.job_id)
            _remove_owned_prompt_audio(job)
            raise
        return JSONResponse(
            {
                "job_id": job.job_id,
                "audio_url": f"/api/generate-stream/{job.job_id}/audio",
                "status_url": f"/api/generate-stream/{job.job_id}/status",
                "result_url": f"/api/generate-stream/{job.job_id}/result",
                "sample_rate": runtime_manager.profiles[model_profile]["sample_rate"],
                "channels": runtime_manager.profiles[model_profile]["channels"],
                "model_profile": model_profile,
                "streaming_generation": streaming_generation_enabled,
                "performance_profile_applied": performance_tuning.profile(model_profile) is not None,
                "performance_recommendation": performance_recommendation,
                "seed": resolved_seed,
                "configured_seed": configured_seed,
                "seed_mode": seed_mode,
                "caller_kind": caller_kind,
                "playback_epoch": playback_coordinator.status()["playback_epoch"],
            }
        )

    def _job_playback_lease_seconds(job: StreamingJob) -> float:
        snapshot = job.snapshot()
        metadata = (job.result or {}).get("metadata") if job.result else {}
        duration = _safe_float(
            (metadata or {}).get("duration_seconds", snapshot.get("emitted_audio_seconds", 0)),
            default=0.0,
            minimum=0.0,
        )
        # Normal clients release immediately after playback. This deadline is
        # only a crash/disconnect safety net and allows slower playback rates.
        return min(3600.0, max(15.0, duration * 1.75 + 10.0))

    async def _acquire_playback(
        job: StreamingJob,
        request: Request,
        *,
        lease_timeout: float | None = None,
    ) -> dict[str, object]:
        job.update(playback_state="waiting")
        disconnected = threading.Event()
        expected_epoch = int(job.snapshot().get("playback_epoch", -1))
        current_epoch = int(playback_coordinator.status()["playback_epoch"])
        if expected_epoch != current_epoch:
            job.update(playback_state="cancelled")
            raise HTTPException(status_code=409, detail="playback session is stale")
        acquire_task = asyncio.create_task(
            asyncio.to_thread(
                playback_coordinator.acquire,
                job.job_id,
                str(job.snapshot().get("caller_kind") or "external"),
                cancelled=lambda: (
                    job.is_closed
                    or disconnected.is_set()
                    or int(playback_coordinator.status()["playback_epoch"]) != expected_epoch
                ),
                timeout=600.0,
                lease_timeout=lease_timeout,
            )
        )
        while not acquire_task.done():
            await asyncio.wait({acquire_task}, timeout=0.25)
            if await request.is_disconnected():
                disconnected.set()
                playback_coordinator.release(job.job_id)
                break
        acquired = await acquire_task
        if acquired and await request.is_disconnected():
            playback_coordinator.release(job.job_id)
            acquired = False
        if not acquired:
            job.update(playback_state="cancelled")
            raise HTTPException(status_code=409, detail="playback was cancelled or queue is full")
        status = playback_coordinator.status()
        if int(status["playback_epoch"]) != expected_epoch:
            playback_coordinator.release(job.job_id)
            job.update(playback_state="cancelled")
            raise HTTPException(status_code=409, detail="playback session is stale")
        job.update(
            playback_state="active",
            playback_epoch=status["playback_epoch"],
        )
        return status

    def _playback_requested(value: int | None, request: Request) -> bool:
        if value is not None:
            return bool(_safe_int(value, default=0, minimum=0, maximum=1))
        return str(getattr(request.state, "caller_kind", "external")) == "external"

    @app.get("/api/generate-stream/{job_id}/audio")
    async def generate_stream_audio(job_id: str, request: Request) -> StreamingResponse:
        job = jobs.get(job_id)
        playback_status = await _acquire_playback(job, request, lease_timeout=900.0)

        async def iterator():
            try:
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        item = await asyncio.to_thread(job.audio_queue.get, True, 0.25)
                    except queue.Empty:
                        continue
                    if item is None:
                        break
                    yield item
            finally:
                snapshot = job.snapshot()
                if snapshot.get("state") not in {"finished", "error", "closed"}:
                    jobs.close(job_id)

        snapshot = job.snapshot()
        # The PCM format comes from the model profile itself.  Reading it from
        # the job snapshot is a race: a job that is still queued reports the
        # placeholder defaults, and clients that trust the headers (the novel
        # reader) would then decode 24 kHz mono PCM as 48 kHz stereo.
        profile = runtime_manager.profiles.get(str(snapshot.get("model_profile") or ""))
        if profile:
            audio_sample_rate = int(profile["sample_rate"])
            audio_channels = int(profile["channels"])
        else:
            audio_sample_rate = int(snapshot.get("sample_rate") or 24000)
            audio_channels = int(snapshot.get("channels") or 1)
        return StreamingResponse(
            iterator(),
            media_type="application/octet-stream",
            headers={
                "X-Audio-Codec": "pcm_s16le",
                "X-Audio-Sample-Rate": str(audio_sample_rate),
                "X-Audio-Channels": str(audio_channels),
                "X-Stream-Id": job_id,
                "X-Playback-Epoch": str(playback_status["playback_epoch"]),
            },
        )

    @app.get("/api/generate-stream/{job_id}/status")
    async def generate_stream_status(job_id: str) -> JSONResponse:
        return JSONResponse(jobs.get(job_id).snapshot())

    @app.get("/api/generate-stream/{job_id}/result")
    async def generate_stream_result(job_id: str) -> JSONResponse:
        job = jobs.get(job_id)
        if job.result is None:
            raise HTTPException(status_code=404, detail="result is not ready")
        return JSONResponse(job.result)

    @app.get("/api/generate-stream/{job_id}/result-audio")
    async def generate_stream_result_audio(
        request: Request,
        job_id: str,
        playback: int | None = None,
    ) -> FileResponse:
        job = jobs.get(job_id)
        if job.result is None:
            raise HTTPException(status_code=404, detail="result is not ready")
        audio_path = Path(str(job.result.get("audio_path") or ""))
        if not audio_path.is_file():
            raise HTTPException(status_code=404, detail="generated audio is missing")
        headers = None
        if _playback_requested(playback, request):
            playback_status = await _acquire_playback(
                job,
                request,
                lease_timeout=_job_playback_lease_seconds(job),
            )
            headers = {"X-Playback-Epoch": str(playback_status["playback_epoch"])}
        return FileResponse(
            str(audio_path),
            media_type="audio/wav",
            filename="generated.wav",
            headers=headers,
        )

    @app.get("/api/generate-stream/{job_id}/result-audio-aac")
    async def generate_stream_result_audio_aac(
        request: Request,
        job_id: str,
        bitrate: str = "80k",
        playback: int | None = None,
    ) -> FileResponse:
        job = jobs.get(job_id)
        if job.result is None:
            raise HTTPException(status_code=404, detail="result is not ready")
        source = Path(str(job.result.get("audio_path") or ""))
        if not source.is_file():
            raise HTTPException(status_code=404, detail="generated audio is missing")
        selected_bitrate = _safe_aac_bitrate(bitrate)
        target = reader_temp_dir / f"{job_id}-{selected_bitrate}.m4a"
        conversion_epoch = getattr(ctx, "tts_epoch", 0)

        def conversion_cancelled() -> bool:
            return job.is_closed or conversion_epoch != getattr(ctx, "tts_epoch", 0)

        def encode_aac() -> Path:
            temporary = target.with_name(
                f".{target.stem}.{uuid.uuid4().hex}.tmp{target.suffix}"
            )
            try:
                try:
                    completed = run_media_process(
                        [
                            ffmpeg_path,
                            "-hide_banner",
                            "-loglevel",
                            "error",
                            "-y",
                            "-i",
                            str(source),
                            "-vn",
                            "-ac",
                            "1",
                            "-ar",
                            "48000",
                            "-c:a",
                            "aac",
                            "-b:a",
                            selected_bitrate,
                            "-movflags",
                            "+faststart",
                            str(temporary),
                        ],
                        timeout=120,
                        cancelled=conversion_cancelled,
                    )
                except FileNotFoundError as exc:
                    raise HTTPException(
                        status_code=503,
                        detail="AAC 编码器 ffmpeg 未找到，请运行 brew install ffmpeg",
                    ) from exc
                except subprocess.TimeoutExpired as exc:
                    raise HTTPException(status_code=504, detail="AAC encoding timed out") from exc
                except RuntimeError as exc:
                    raise HTTPException(status_code=409, detail=str(exc)) from exc
                if completed.returncode != 0 or not temporary.is_file():
                    raise HTTPException(
                        status_code=500,
                        detail=(completed.stderr or "AAC encoding failed").strip(),
                    )
                # ``jobs.close`` takes this lock before flagging the job closed.
                # The publication check and atomic replacement therefore cannot
                # race a close into making a newly encoded file visible afterwards.
                with job.status_lock:
                    if conversion_cancelled():
                        raise HTTPException(status_code=409, detail="generation job was closed")
                    os.replace(temporary, target)
                return target
            finally:
                temporary.unlink(missing_ok=True)

        async def ensure_aac() -> Path:
            key = f"{job_id}:{selected_bitrate}"
            async with aac_flights_lock:
                task = aac_flights.get(key)
                if task is None:
                    task = asyncio.create_task(asyncio.to_thread(encode_aac))
                    aac_flights[key] = task
                    def cleanup_completed_task(completed: asyncio.Task[Path]) -> None:
                        # Consume failures so an abandoned client cannot leave
                        # an unobserved-task warning, then retire this exact
                        # flight. Task callbacks run on this same event loop.
                        try:
                            completed.exception()
                        except asyncio.CancelledError:
                            pass
                        if aac_flights.get(key) is completed:
                            aac_flights.pop(key, None)
                    task.add_done_callback(cleanup_completed_task)
            try:
                return await asyncio.shield(task)
            finally:
                if task.done():
                    async with aac_flights_lock:
                        if aac_flights.get(key) is task:
                            aac_flights.pop(key, None)

        if not target.is_file():
            await ensure_aac()
        headers = None
        if _playback_requested(playback, request):
            playback_status = await _acquire_playback(
                job,
                request,
                lease_timeout=_job_playback_lease_seconds(job),
            )
            headers = {"X-Playback-Epoch": str(playback_status["playback_epoch"])}
        return FileResponse(
            str(target),
            media_type="audio/mp4",
            filename=f"reader-block-{job_id}.m4a",
            headers=headers,
        )

    def _remove_ephemeral_job_audio(job: StreamingJob) -> None:
        for target in reader_temp_dir.glob(f"{job.job_id}-*.m4a"):
            target.unlink(missing_ok=True)
        if not bool(job.snapshot().get("ephemeral_audio")) or job.result is None:
            return
        _remove_generated_result_files(job.result)
        with job.status_lock:
            job.result = None
            job.status["audio_path"] = None
            job.status["ephemeral_cleaned"] = True
            job.status["updated_at"] = time.time()
        job.persist()

    @app.delete("/api/generate-stream/{job_id}/ephemeral-audio")
    async def generate_stream_delete_ephemeral_audio(job_id: str) -> JSONResponse:
        job = jobs.close(job_id)
        _remove_ephemeral_job_audio(job)
        return JSONResponse({"ok": True, "job_id": job_id})

    @app.post("/api/generate-stream/{job_id}/ephemeral-audio/close")
    async def generate_stream_close_ephemeral_audio(job_id: str) -> JSONResponse:
        job = jobs.close(job_id)
        _remove_ephemeral_job_audio(job)
        return JSONResponse({"ok": True, "job_id": job_id})

    @app.post("/api/generate-stream/{job_id}/close")
    async def generate_stream_close(job_id: str) -> JSONResponse:
        jobs.close(job_id)
        return JSONResponse({"ok": True})
