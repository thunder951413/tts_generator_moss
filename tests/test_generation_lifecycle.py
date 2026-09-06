from __future__ import annotations

import shutil
import sys
import threading
import time
import wave
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest


ROOT = Path(__file__).resolve().parents[1]
SERVICE_DIR = ROOT / "qwen_tts_service"
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

from activity_control import PlaybackCoordinator
from streaming_jobs import StreamingJobManager
from webapp.routers.generation import register_generation_routes


def _app_with_completed_job(
    tmp_path: Path,
    *,
    ffmpeg_path: str = "fake-ffmpeg",
    valid_wav: bool = False,
):
    app = FastAPI()
    app.get("/health")(lambda: {"ok": True})
    jobs = StreamingJobManager(tmp_path / "jobs")
    source = tmp_path / "source.wav"
    if valid_wav:
        with wave.open(str(source), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(24000)
            audio.writeframes(b"\0\0" * 2400)
    else:
        source.write_bytes(b"RIFF-source")
    job = jobs.create()
    job.set_result({"audio_path": str(source), "metadata": {"duration_seconds": 1}})
    profiles = {
        "qwen_0_6b": {"sample_rate": 24000, "channels": 1, "streaming": True, "backend": "qwen"}
    }
    ctx = SimpleNamespace(
        runtime_manager=SimpleNamespace(profiles=profiles),
        performance_tuning=SimpleNamespace(profile=lambda _profile: None),
        playback_coordinator=PlaybackCoordinator(),
        jobs=jobs,
        preset_store=None,
        generation_scheduler=None,
        stt_runtime=None,
        document_projects=None,
        output_dir=tmp_path / "output",
        upload_dir=tmp_path / "upload",
        reader_temp_dir=tmp_path / "reader",
        ffmpeg_path=ffmpeg_path,
        stt_enabled=False,
        expected_session="",
        access_password="",
        synthesize_for_profile_runtime=None,
        apply_performance_profile=lambda _profile: {},
        active_service_settings=lambda: {"settings": {}},
        preset_payload=lambda _payload: ("", {}),
        remove_generated_result_files=lambda _result: None,
        tts_enabled=True,
    )
    ctx.output_dir.mkdir()
    ctx.upload_dir.mkdir()
    ctx.reader_temp_dir.mkdir()
    register_generation_routes(app, ctx)
    return app, job, ctx


def test_aac_encoding_does_not_block_health_and_is_singleflight(monkeypatch, tmp_path: Path) -> None:
    app, job, _ctx = _app_with_completed_job(tmp_path)
    import webapp.routers.generation as generation

    started = threading.Event()
    release = threading.Event()
    calls = 0
    calls_lock = threading.Lock()

    def fake_ffmpeg(command, **_kwargs):
        nonlocal calls
        with calls_lock:
            calls += 1
        started.set()
        assert release.wait(2)
        Path(command[-1]).write_bytes(b"aac")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(generation, "run_media_process", fake_ffmpeg)
    responses = []
    with TestClient(app) as client:
        def request_aac() -> None:
            responses.append(
                client.get(f"/api/generate-stream/{job.job_id}/result-audio-aac", params={"playback": 0})
            )

        first = threading.Thread(target=request_aac)
        second = threading.Thread(target=request_aac)
        first.start()
        assert started.wait(1)
        second.start()
        health_started = time.monotonic()
        health = client.get("/health")
        assert time.monotonic() - health_started < 0.5
        assert health.status_code == 200
        release.set()
        first.join(2)
        second.join(2)

    assert calls == 1
    assert len(responses) == 2
    assert all(response.status_code == 200 and response.content == b"aac" for response in responses), [
        (response.status_code, response.text) for response in responses
    ]


def test_aac_failure_removes_temporary_output(monkeypatch, tmp_path: Path) -> None:
    app, job, ctx = _app_with_completed_job(tmp_path)
    import webapp.routers.generation as generation

    def failing_ffmpeg(command, **_kwargs):
        Path(command[-1]).write_bytes(b"partial")
        return SimpleNamespace(returncode=1, stderr="encoder failed")

    monkeypatch.setattr(generation, "run_media_process", failing_ffmpeg)
    with TestClient(app) as client:
        response = client.get(f"/api/generate-stream/{job.job_id}/result-audio-aac")

    assert response.status_code == 500
    assert not list(ctx.reader_temp_dir.iterdir())


@pytest.mark.parametrize("stop_kind", ["job", "service_epoch"])
def test_aac_conversion_is_cancelled_before_publishing(monkeypatch, tmp_path: Path, stop_kind: str) -> None:
    app, job, ctx = _app_with_completed_job(tmp_path)
    import webapp.routers.generation as generation

    started = threading.Event()
    responses = []

    def waiting_encoder(command, *, cancelled, timeout):
        Path(command[-1]).write_bytes(b"partial")
        started.set()
        deadline = time.monotonic() + 2
        while not cancelled() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert cancelled(), "stop did not reach the media process"
        raise RuntimeError("media process was cancelled")

    monkeypatch.setattr(generation, "run_media_process", waiting_encoder)
    with TestClient(app) as client:
        worker = threading.Thread(target=lambda: responses.append(client.get(
            f"/api/generate-stream/{job.job_id}/result-audio-aac", params={"playback": 0}
        )))
        worker.start()
        assert started.wait(1)
        if stop_kind == "job":
            ctx.jobs.close(job.job_id)
        else:
            ctx.tts_epoch = 1
        worker.join(3)
        assert not worker.is_alive()
    assert responses[0].status_code == 409
    assert not list(ctx.reader_temp_dir.iterdir())


def test_aac_route_encodes_a_real_wav_with_ffmpeg(tmp_path: Path) -> None:
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        pytest.skip("ffmpeg is unavailable")
    app, job, ctx = _app_with_completed_job(
        tmp_path, ffmpeg_path=ffmpeg_path, valid_wav=True
    )

    with TestClient(app) as client:
        response = client.get(
            f"/api/generate-stream/{job.job_id}/result-audio-aac",
            params={"playback": 0},
        )

    target = ctx.reader_temp_dir / f"{job.job_id}-80k.m4a"
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("audio/mp4")
    assert target.is_file() and target.stat().st_size > 0
    assert not list(ctx.reader_temp_dir.glob(".*.tmp.m4a"))


def test_finished_result_cannot_reacquire_playback_after_global_stop(tmp_path: Path) -> None:
    app, job, ctx = _app_with_completed_job(tmp_path)
    job.update(playback_epoch=ctx.playback_coordinator.status()["playback_epoch"])
    ctx.playback_coordinator.force_stop()

    with TestClient(app) as client:
        response = client.get(f"/api/generate-stream/{job.job_id}/result-audio", params={"playback": 1})

    assert response.status_code == 409
    assert response.json()["detail"] == "playback session is stale"
