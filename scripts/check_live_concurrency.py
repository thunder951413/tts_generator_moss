"""Opt-in local Metal smoke test; creates only ephemeral jobs and never plays audio.

Run with the repository Python and --live. The test needs an idle local service,
temporarily loads both TTS profiles and STT, and restores STT's initial state.
Credentials are read locally and never printed.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import io
import threading
import time
import uuid
import wave
from pathlib import Path

import httpx


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", required=True)
    args = parser.parse_args()
    if not args.live:
        return
    root = Path(__file__).resolve().parents[1]
    config = dict(
        line.split("=", 1) for line in (root / ".env.macos").read_text().splitlines()
        if "=" in line and not line.startswith("#")
    )
    password = config.get("QWEN_TTS_ACCESS_PASSWORD", "").strip("\"'")
    cookie = hmac.new(password.encode(), b"qwen-tts-service-session", hashlib.sha256).hexdigest()
    base = "http://127.0.0.1:" + config.get("PORT", "7861")
    session_id = "verification-" + uuid.uuid4().hex
    jobs = []
    responses = []
    worker = None
    sample = io.BytesIO()
    with wave.open(sample, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\0\0" * 8000)

    with httpx.Client(base_url=base, cookies={"qwen_tts_service_session": cookie}, timeout=180) as internal:
        initial = internal.get("/api/health").json()
        assert initial["generation_scheduler"]["active"] == 0, "Service must be idle"
        assert not initial["playback"]["active_job_id"], "Playback must be idle"
        settings = internal.get("/api/service-settings").json()
        reference = settings["settings"]["reference_audio_path"]
        heartbeat_at = 0.0
        reservation = internal.post(f"/api/listening/{session_id}/heartbeat")
        reservation.raise_for_status()
        epoch = reservation.json()["playback_epoch"]

        def heartbeat() -> None:
            nonlocal heartbeat_at
            if time.monotonic() - heartbeat_at >= 2:
                result = internal.post(f"/api/listening/{session_id}/heartbeat", params={"playback_epoch": epoch})
                result.raise_for_status()
                heartbeat_at = time.monotonic()

        def transcribe() -> None:
            try:
                with httpx.Client(base_url=base, headers={"Authorization": "Bearer " + password}, timeout=180) as external:
                    responses.append(external.post(
                        "/v1/audio/transcriptions",
                        files={"file": ("verification.wav", sample.getvalue(), "audio/wav")},
                    ))
            except Exception as exc:
                responses.append(exc)

        try:
            worker = threading.Thread(target=transcribe, daemon=True)
            worker.start()
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                heartbeat()
                if internal.get("/api/health").json()["generation_scheduler"]["waiting_stt"] == 1:
                    break
                time.sleep(0.1)
            assert internal.get("/api/health").json()["generation_scheduler"]["waiting_stt"] == 1
            print("PASS: external STT waits for continuous reader", flush=True)

            for profile in ("qwen_0_6b", "qwen_1_7b"):
                result = internal.post("/api/generate-stream/start", data={
                    "text": "这是一段并发调度测试。", "model_profile": profile,
                    "use_service_settings": "0", "example_audio_path": reference,
                    "qwen_clone_mode": "xvec", "seed": "1234", "max_new_tokens": "128",
                    "streaming_generation": "0", "qwen_non_streaming_mode": "1",
                    "ephemeral_audio": "1", "expected_playback_epoch": str(epoch),
                })
                result.raise_for_status()
                jobs.append(result.json()["job_id"])
            pending = set(jobs)
            deadline = time.monotonic() + 150
            while pending and time.monotonic() < deadline:
                heartbeat()
                for job_id in list(pending):
                    status = internal.get(f"/api/generate-stream/{job_id}/status").json()
                    assert status["state"] not in {"error", "closed"}, status.get("error")
                    if status["state"] == "finished":
                        pending.remove(job_id)
                assert not responses, "STT must not start while continuous listening is reserved"
                time.sleep(0.1)
            assert not pending, "Generation timed out"
            print("PASS: both TTS profiles finish their own queued requests while STT waits", flush=True)
            for job_id in jobs:
                heartbeat()
                result = internal.get(f"/api/generate-stream/{job_id}/result-audio-aac", params={"bitrate": "80k", "playback": 0})
                result.raise_for_status()
                assert b"ftyp" in result.content[:32], "Expected M4A container"
            print("PASS: real AAC export for both models", flush=True)
            internal.post(f"/api/listening/{session_id}/release").raise_for_status()
            worker.join(90)
            assert not worker.is_alive(), "STT did not finish after release"
            assert responses and isinstance(responses[0], httpx.Response), responses
            responses[0].raise_for_status()
            assert "text" in responses[0].json()
            assert internal.get("/api/service-settings").json() == settings
            print("PASS: real STT runs after reader releases; applied settings unchanged", flush=True)
        finally:
            internal.post(f"/api/listening/{session_id}/release")
            if worker is not None and worker.is_alive():
                internal.post("/api/stt/stop")
                worker.join(10)
            for job_id in jobs:
                internal.delete(f"/api/generate-stream/{job_id}/ephemeral-audio").raise_for_status()
            if not initial.get("stt", {}).get("ready"):
                internal.post("/api/stt/stop").raise_for_status()
            print("Temporary verification audio cleaned; no audio playback requested", flush=True)


if __name__ == "__main__":
    main()
