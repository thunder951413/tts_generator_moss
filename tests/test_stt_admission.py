import threading
import time

from fastapi.testclient import TestClient

from test_macos_qwen_service import load_app_module, silent_wav_bytes


def test_reader_stt_queue_stop_and_retry(tmp_path, monkeypatch):
    module = load_app_module()
    app = module.create_app(
        output_dir=tmp_path / "output", upload_dir=tmp_path / "uploads",
        preset_dir=tmp_path / "presets", document_project_dir=tmp_path / "books",
        service_job_dir=tmp_path / "jobs", performance_tuning_path=tmp_path / "performance.json",
        preload=False, stt_preload=False, access_password="",
    )
    invoked = []

    def fake_transcribe(**kwargs):
        invoked.append(kwargs["filename"])
        return b'{"text":"transcribed"}', "application/json"

    monkeypatch.setattr(app.state.stt_runtime, "transcribe", fake_transcribe)
    with TestClient(app) as client:
        reserved = client.post("/api/listening/reader-test-session/heartbeat")
        assert reserved.status_code == 200
        outcomes = []

        def request():
            outcomes.append(client.post(
                "/v1/audio/transcriptions",
                files={"file": ("queued.wav", silent_wav_bytes(), "audio/wav")},
            ))

        thread = threading.Thread(target=request, daemon=True)
        thread.start()
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            health = client.get("/api/health").json()
            if health["generation_scheduler"]["waiting_stt"]:
                break
            time.sleep(0.01)
        assert health["generation_scheduler"]["waiting_stt"] == 1
        assert invoked == []
        # Stop STT only: reader reservation survives but queued transcription
        # is cancelled, not auto-started after the listener leaves.
        assert client.post("/api/stt/stop").status_code == 200
        thread.join(3)
        assert not thread.is_alive()
        assert outcomes[0].status_code == 503
        assert invoked == []
        assert client.post("/api/listening/reader-test-session/release").status_code == 200
        result = client.post(
            "/v1/audio/transcriptions", files={"file": ("new.wav", silent_wav_bytes(), "audio/wav")},
        )
        assert result.status_code == 200
        assert invoked == ["new.wav"]


def test_empty_stt_upload_does_not_wait_or_invoke_model(tmp_path, monkeypatch):
    # An empty upload is rejected before waiting for the GPU, even during listening.
    module = load_app_module()
    app = module.create_app(
        output_dir=tmp_path / "out", upload_dir=tmp_path / "upload",
        preset_dir=tmp_path / "presets", document_project_dir=tmp_path / "books",
        service_job_dir=tmp_path / "jobs", performance_tuning_path=tmp_path / "perf.json",
        preload=False, stt_preload=False, access_password="",
    )
    monkeypatch.setattr(app.state.stt_runtime, "transcribe", lambda **kwargs: (_ for _ in ()).throw(AssertionError("must not run")))
    with TestClient(app) as client:
        client.post("/api/listening/reader-test-session/heartbeat")
        response = client.post("/v1/audio/transcriptions", files={"file": ("empty.wav", b"", "audio/wav")})
        assert response.status_code == 400
        assert client.get("/api/health").json()["generation_scheduler"]["waiting_stt"] == 0
