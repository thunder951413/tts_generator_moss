# coding=utf-8
"""Shared service singletons handed to every route group.

The context carries the instances created by ``create_app`` plus the small
number of closures that several route groups share. Route groups receive the
context and bind the fields they need as locals, keeping route bodies free of
attribute noise. ``tts_enabled`` is the only mutable runtime flag and must be
read through the context (``ctx.tts_enabled``) so updates are visible.
"""

from __future__ import annotations

from pathlib import Path
import asyncio
from typing import Any, Callable


class ServiceContext:
    def __init__(
        self,
        *,
        runtime_manager: Any,
        performance_tuning: Any,
        playback_coordinator: Any,
        jobs: Any,
        preset_store: Any,
        generation_scheduler: Any,
        stt_runtime: Any,
        document_projects: Any,
        output_dir: Path,
        upload_dir: Path,
        reader_temp_dir: Path,
        ffmpeg_path: str,
        stt_enabled: bool,
        expected_session: str,
        access_password: str,
        synthesize_for_profile_runtime: Callable[..., Any],
        apply_performance_profile: Callable[[str], dict[str, int]],
        active_service_settings: Callable[[], dict[str, Any]],
        preset_payload: Callable[[dict[str, Any]], tuple[str, dict[str, Any]]],
        remove_generated_result_files: Callable[[dict[str, Any] | None], None],
        tts_enabled: bool = True,
    ) -> None:
        self.runtime_manager = runtime_manager
        self.performance_tuning = performance_tuning
        self.playback_coordinator = playback_coordinator
        self.jobs = jobs
        self.preset_store = preset_store
        self.generation_scheduler = generation_scheduler
        self.stt_runtime = stt_runtime
        self.document_projects = document_projects
        self.output_dir = output_dir
        self.upload_dir = upload_dir
        self.reader_temp_dir = reader_temp_dir
        self.ffmpeg_path = ffmpeg_path
        self.stt_enabled = stt_enabled
        self.expected_session = expected_session
        self.access_password = access_password
        self.synthesize_for_profile_runtime = synthesize_for_profile_runtime
        self.apply_performance_profile = apply_performance_profile
        self.active_service_settings = active_service_settings
        self.preset_payload = preset_payload
        self.remove_generated_result_files = remove_generated_result_files
        self.tts_enabled = tts_enabled
        self.stopping = False
        self.tts_epoch = 0
        self.stt_epoch = 0
        self.stt_requests = 0
        self.control_lock = asyncio.Lock()
