# coding=utf-8
"""Service settings and Metal performance benchmarking."""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor

from typing import Any

from fastapi import (
    Body,
    HTTPException,
)

from fastapi.responses import JSONResponse

from performance_tuning import choose_recommendation

from qwen_protocol import StreamingRequest

from runtime_manager import (
    DEFAULT_MODEL_PROFILE,
    MODEL_PROFILE_LABELS,
)

from webapp.config import (
    DEFAULT_CLONE_AUDIO_PATH,
    REFERENCE_AUDIO_DIR,
)

from webapp.util import (
    _resolve_allowed_reference_audio_path,
    _safe_float,
    _safe_int,
)

def register_settings_performance_routes(app, ctx):
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
    active_service_settings = ctx.active_service_settings
    preset_payload = ctx.preset_payload
    _remove_generated_result_files = ctx.remove_generated_result_files

    @app.get("/api/service-settings")
    async def get_service_settings() -> JSONResponse:
        return JSONResponse(active_service_settings())

    def run_performance_benchmark(profile_id: str, epoch: int) -> dict[str, Any]:
        def check_cancelled() -> None:
            if ctx.stopping or not ctx.tts_enabled or epoch != ctx.tts_epoch:
                raise RuntimeError("性能测试已被停止操作取消")

        check_cancelled()
        if profile_id not in runtime_manager.profiles:
            raise ValueError("无效的模型")
        service_settings = dict(active_service_settings().get("settings") or {})
        reference_path = str(
            service_settings.get("reference_audio_path") or DEFAULT_CLONE_AUDIO_PATH
        )
        reference_path = str(
            _resolve_allowed_reference_audio_path(
                reference_path,
                REFERENCE_AUDIO_DIR,
                preset_store.audio_dir,
            )
        )
        clone_mode = str(service_settings.get("qwen_clone_mode") or "xvec")
        reference_text = str(service_settings.get("qwen_reference_text") or "").strip()
        benchmark_dir = output_dir / "performance-benchmark"
        benchmark_dir.mkdir(parents=True, exist_ok=True)
        benchmark_text = "这是一段性能测试语音，用于选择最适合这台 Mac 的生成策略。"

        def run_case(*, chunk_frames: int, non_streaming: bool, seed: int) -> dict[str, Any]:
            check_cancelled()
            request = StreamingRequest(
                text=benchmark_text,
                mode="voice_clone",
                prompt_audio_path=reference_path,
                language="Chinese",
                max_new_frames=72,
                do_sample=True,
                temperature=_safe_float(
                    service_settings.get("qwen_temperature"),
                    default=0.9,
                    minimum=0.1,
                    maximum=3.0,
                ),
                top_p=_safe_float(
                    service_settings.get("qwen_top_p"),
                    default=1.0,
                    minimum=0.1,
                    maximum=1.0,
                ),
                top_k=_safe_int(
                    service_settings.get("qwen_top_k"),
                    default=50,
                    minimum=1,
                    maximum=200,
                ),
                repetition_penalty=_safe_float(
                    service_settings.get("qwen_repetition_penalty"),
                    default=1.05,
                    minimum=0.8,
                    maximum=2.0,
                ),
                seed=seed,
                codec_chunk_frames=chunk_frames,
                qwen_xvec_only=clone_mode != "icl",
                qwen_reference_text=reference_text,
                qwen_non_streaming_mode=non_streaming,
                qwen_append_silence=False,
                qwen_min_new_tokens=2,
            )
            started = time.perf_counter()
            first_audio_seconds: float | None = None
            result: dict[str, Any] | None = None
            with runtime_manager.session(profile_id) as runtime:
                for event in synthesize_for_profile_runtime(
                    runtime,
                    request,
                    output_dir=benchmark_dir,
                ):
                    check_cancelled()
                    if event.type == "audio" and first_audio_seconds is None:
                        first_audio_seconds = time.perf_counter() - started
                    elif event.type == "result":
                        result = dict(event.data)
            elapsed = time.perf_counter() - started
            metadata = dict((result or {}).get("metadata") or {})
            measurement = {
                "chunk_frames": chunk_frames,
                "elapsed_seconds": round(elapsed, 4),
                "first_audio_seconds": round(
                    float(first_audio_seconds or metadata.get("first_audio_latency_seconds") or elapsed),
                    4,
                ),
                "audio_seconds": round(float(metadata.get("duration_seconds") or 0), 4),
                "generation_realtime_factor": round(
                    float(metadata.get("generation_realtime_factor") or 0),
                    4,
                ),
            }
            _remove_generated_result_files(result)
            return measurement

        with generation_scheduler.exclusive_slot(cancelled=lambda: epoch != ctx.tts_epoch):
            check_cancelled()
            with runtime_manager.session(profile_id) as runtime:
                runtime.resize_lanes(1)
            # Warm the selected reference and model before timed cases.
            run_case(chunk_frames=8, non_streaming=False, seed=880001)
            stream_measurements = [
                run_case(chunk_frames=chunk, non_streaming=False, seed=880010 + chunk)
                for chunk in (4, 8, 12)
            ]
            single_block = run_case(
                chunk_frames=8,
                non_streaming=True,
                seed=880101,
            )
            parallel_block_seconds: float | None = None
            parallel_error = ""
            try:
                with runtime_manager.session(profile_id) as runtime:
                    runtime.resize_lanes(2)
                parallel_started = time.perf_counter()
                with ThreadPoolExecutor(max_workers=2, thread_name_prefix="qwen-benchmark") as pool:
                    futures = [
                        pool.submit(
                            run_case,
                            chunk_frames=8,
                            non_streaming=True,
                            seed=880201 + index,
                        )
                        for index in range(2)
                    ]
                    parallel_measurements = [future.result() for future in futures]
                parallel_block_seconds = time.perf_counter() - parallel_started
            except Exception as exc:
                logging.exception("two-lane performance benchmark failed")
                parallel_measurements = []
                parallel_error = str(exc)

            check_cancelled()
            recommendation = choose_recommendation(
                stream_measurements=stream_measurements,
                single_block_seconds=float(single_block["elapsed_seconds"]),
                parallel_block_seconds=parallel_block_seconds,
            )
            with runtime_manager.session(profile_id) as runtime:
                runtime.resize_lanes(recommendation["block_parallel"])
            runtime_manager.profiles[profile_id]["lanes"] = recommendation["block_parallel"]
            generation_scheduler.configure_max_parallel(recommendation["block_parallel"])
            document_projects.configure_synthesis_workers(
                recommendation["document_workers"]
            )

        throughput_gain = (
            (2.0 * float(single_block["elapsed_seconds"])) / parallel_block_seconds
            if parallel_block_seconds
            else 1.0
        )
        result = {
            "model_profile": profile_id,
            "model_label": MODEL_PROFILE_LABELS[profile_id],
            "recommendation": recommendation,
            "stream_measurements": stream_measurements,
            "block_measurements": {
                "single": single_block,
                "parallel_elapsed_seconds": (
                    round(parallel_block_seconds, 4) if parallel_block_seconds else None
                ),
                "parallel_cases": parallel_measurements,
                "throughput_gain": round(throughput_gain, 3),
                "parallel_error": parallel_error,
            },
            "applied": True,
        }
        check_cancelled()
        return performance_tuning.save_profile(profile_id, result)

    @app.get("/api/performance")
    async def performance_profile() -> JSONResponse:
        active_profile = (
            runtime_manager.status().get("active_profile") or DEFAULT_MODEL_PROFILE
        )
        return JSONResponse(
            {
                **performance_tuning.public_payload(),
                "active_profile": active_profile,
                "active_recommendation": performance_tuning.recommendation(active_profile),
            }
        )

    @app.post("/api/performance/benchmark")
    async def performance_benchmark(
        payload: dict[str, Any] = Body(default={}),
    ) -> JSONResponse:
        if not ctx.tts_enabled or ctx.stopping:
            raise HTTPException(status_code=503, detail="TTS 已停止，请先启动服务")
        profile_id = str(
            payload.get("model_profile")
            or active_service_settings().get("settings", {}).get("model_profile")
            or DEFAULT_MODEL_PROFILE
        )
        try:
            result = await asyncio.to_thread(run_performance_benchmark, profile_id, ctx.tts_epoch)
            return JSONResponse(result)
        except (FileNotFoundError, PermissionError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.put("/api/service-settings/active-preset")
    async def activate_service_preset(payload: dict[str, Any] = Body(...)) -> JSONResponse:
        try:
            preset = preset_store.activate(str(payload.get("preset_id") or ""))
            return JSONResponse(
                {
                    "active_preset_id": preset["id"],
                    "name": preset["name"],
                    "source": "preset",
                    "settings": dict(preset.get("settings") or {}),
                }
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="preset not found") from exc

    @app.put("/api/service-settings")
    async def apply_service_settings(payload: dict[str, Any] = Body(...)) -> JSONResponse:
        try:
            name, settings = preset_payload(payload)
            configuration = preset_store.apply_configuration(name=name or "音频工作台设置", settings=settings)
            return JSONResponse(
                {
                    "active_preset_id": "",
                    "name": configuration["name"],
                    "source": "studio",
                    "settings": dict(configuration.get("settings") or {}),
                    "updated_at": configuration.get("updated_at"),
                }
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.delete("/api/service-settings")
    async def reset_service_settings() -> JSONResponse:
        preset_store.clear_configuration()
        return JSONResponse(active_service_settings())
