from __future__ import annotations

import importlib.util
import io
import json
import os
import sys
import threading
import time
import types
import wave
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[1]


def silent_wav_bytes(*, sample_rate: int = 24000, duration: float = 0.1) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(b"\0\0" * max(1, int(sample_rate * duration)))
    return output.getvalue()


def load_app_module():
    app_path = ROOT / "clis" / "qwen_tts_app.py"
    spec = importlib.util.spec_from_file_location("qwen_tts_app_test", app_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ffmpeg_resolves_from_explicit_macos_app_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_app_module()
    ffmpeg = tmp_path / "ffmpeg"
    ffmpeg.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    ffmpeg.chmod(0o755)
    monkeypatch.setenv("QWEN_TTS_FFMPEG", str(ffmpeg))

    assert module._resolve_ffmpeg_path() == str(ffmpeg.resolve())


def test_stt_runtime_adds_homebrew_tools_to_app_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    from qwen_tts_service.stt_runtime import WhisperCppRuntime

    runtime = WhisperCppRuntime(binary="/opt/homebrew/bin/whisper-server", model="model.bin")
    paths = runtime._subprocess_environment()["PATH"].split(os.pathsep)

    assert paths[0] == "/opt/homebrew/bin"
    assert "/usr/bin" in paths


def test_mac_app_only_exposes_qwen_profiles(tmp_path: Path) -> None:
    module = load_app_module()
    module.DEFAULT_DOCUMENT_PROJECT_DIR = tmp_path / "documents"
    module.DEFAULT_PERFORMANCE_TUNING_PATH = tmp_path / "performance.json"
    app = module.create_app(
        qwen_python=sys.executable,
        qwentts_library="",
        output_dir=tmp_path / "output",
        upload_dir=tmp_path / "upload",
        preload=False,
        access_password="",
    )

    with TestClient(app) as client:
        page = client.get("/", follow_redirects=False)
        assert page.status_code == 307
        assert page.headers["location"] == "/reader"

        reader = client.get("/reader")
        assert reader.status_code == 200
        assert "Qwen 声阅" in reader.text
        assert 'id="reader-settings-dialog"' in reader.text
        assert 'id="listening-settings-card"' not in reader.text
        assert 'id="inspector"' not in reader.text
        assert 'id="listening-controls-button"' in reader.text
        assert 'id="listening-controls-dialog"' in reader.text
        assert 'id="start-listening"' in reader.text
        assert 'id="playback-rate"' in reader.text
        assert 'id="library-manage-button"' in reader.text
        assert 'id="library-manager-dialog"' in reader.text
        assert 'id="book-manager-tab-all"' in reader.text
        assert 'id="book-manager-tab-completed"' in reader.text
        assert "reader.css?v=" in reader.text
        assert 'rel="icon"' in reader.text
        assert "/reader-assets/assets/favicon-64.png" in reader.text
        assert 'rel="apple-touch-icon"' in reader.text
        assert 'id="preset-select"' not in reader.text
        assert 'id="voice-select"' not in reader.text
        assert 'id="stop-generation"' in reader.text
        assert 'id="generate-whole-book"' in reader.text
        assert 'id="download-whole-book"' in reader.text
        assert 'id="book-aac-bitrate"' in reader.text
        assert 'id="book-progress-detail-button"' in reader.text
        reader_script = client.get("/reader-assets/reader.js")
        assert reader_script.status_code == 200
        reader_icon = client.get("/reader-assets/assets/favicon-64.png")
        assert reader_icon.status_code == 200
        assert reader_icon.headers["content-type"] == "image/png"
        assert '"/api/service-settings"' in reader_script.text
        assert '"/api/presets"' not in reader_script.text
        assert 'form.set("qwen_non_streaming_mode", "1")' in reader_script.text
        assert 'ephemeral-audio' in reader_script.text
        assert "queueQualityBlock" in reader_script.text
        assert "startSelectedListening" in reader_script.text
        assert 'setListeningMode("quality")' in reader_script.text
        assert "qwen-reader-playback-rate" in reader_script.text
        assert "qualityJobs: new Set()" in reader_script.text
        assert "renderWholeBookProgress" in reader_script.text
        assert 'window.addEventListener("pagehide"' in reader_script.text
        assert 'document.addEventListener("visibilitychange"' in reader_script.text
        assert "if (document.hidden) return;" in reader_script.text
        assert "超过 60 秒没有数据" in reader_script.text
        assert "/api/playback/status" in reader_script.text
        assert "/api/playback/" in reader_script.text
        assert "playback=1" in reader_script.text
        assert "remainingListeningSegments" in reader_script.text
        assert "activatePlaybackChapter" in reader_script.text
        assert 'stopCurrentGeneration' in reader_script.text
        assert "renameManagedBook" in reader_script.text
        assert "deleteManagedBook" in reader_script.text
        assert 'method: "DELETE"' in reader_script.text
        performance = client.get("/api/performance")
        assert performance.status_code == 200
        assert performance.json()["active_recommendation"]["block_parallel"] == 1

        native_source = (ROOT / "macos" / "NativeStudio.swift").read_text(encoding="utf-8")
        app_source = (ROOT / "macos" / "QwenTTSApp.swift").read_text(encoding="utf-8")
        stt_source = (ROOT / "macos" / "STTWorkbench.swift").read_text(encoding="utf-8")
        reader_app_source = (ROOT / "macos" / "QwenReaderApp.swift").read_text(encoding="utf-8")
        assert 'action: #selector(closeStudio(_:))' in app_source
        assert 'keyEquivalent: "w"' in app_source
        assert 'title: "打开语音转文字工作台"' in app_source
        assert 'action: #selector(openSTTWorkbench(_:))' in app_source
        assert 'struct STTWorkbenchView: View' in stt_source
        assert 'Toggle("生成带时间戳的字幕"' in stt_source
        assert 'case srt' in stt_source and 'case vtt' in stt_source
        assert '"v1/audio/transcriptions"' in stt_source
        assert 'Label("保存", systemImage: "square.and.arrow.down")' in stt_source
        assert 'CommandLine.arguments.contains("--background")' in app_source
        assert 'NSAttributedString(string: "TTS"' in app_source
        assert 'title: "强制停止所有音频与运算"' in app_source
        assert 'title: "启动本地服务"' in app_source
        assert 'title: "停止本地服务"' in app_source
        assert 'keyEquivalentModifierMask = [.command, .option]' in app_source
        assert 'keyEquivalentModifierMask = [.command, .shift]' in app_source
        assert 'requestTTSRuntimeToggle' in app_source
        assert 'requestSTTRuntimeToggle' in app_source
        assert 'api/service/stop-all' in native_source
        assert 'api/tts/\\(enabled ? "start" : "stop")' in native_source
        assert 'api/stt/\\(enabled ? "start" : "stop")' in native_source
        assert 'onTTSServiceToggleRequested' in native_source
        assert 'onSTTServiceToggleRequested' in native_source
        assert '"应用当前设置"' in native_source
        assert '.disabled(model.referenceAudioPath.isEmpty || model.isApplyingServiceSettings)' in native_source
        assert '.disabled(model.outputAudioURL == nil || model.isApplyingServiceSettings)' not in native_source
        assert '?playback=1' in native_source
        assert 'voiceStatusItem?.title = "当前音色：' in app_source
        assert 'taskStatusItem?.title = ttsEnabled && active > 0' in app_source
        assert "WKWebView" in reader_app_source
        assert "WKUIDelegate" in reader_app_source
        assert "webView.uiDelegate = self" in reader_app_source
        assert "runOpenPanelWith parameters: WKOpenPanelParameters" in reader_app_source
        assert 'Window("Qwen 声阅", id: "reader")' in reader_app_source
        assert "QwenReaderServiceURL" in reader_app_source
        assert 'configuration.arguments = ["--background"]' in reader_app_source
        assert "final class NativePCMStreamPlayer" in native_source
        assert "final class SystemAudioRecorder" in native_source
        assert "SCStreamOutput" in native_source
        assert "struct SystemAudioTrimView" in native_source
        assert "prepareReferenceAudioForTrimming" in native_source
        assert "func trimReference(_ reference: NativeReferenceAudio)" in native_source
        assert 'Label("裁剪", systemImage: "scissors")' in native_source
        assert 'Text(model.audioTrimTitle)' in native_source
        assert '"裁剪并加入参考音频"' in native_source
        assert "if trimSourceIsTemporary, let systemAudioRecordingURL" in native_source
        assert "struct ReferenceAudioLibraryView" in native_source
        assert 'Text("可用音频 · \\(visibleReferences.count)")' in native_source
        assert 'Text("已隐藏 · \\(hiddenReferences.count)")' in native_source
        assert 'Label("试听", systemImage: "play.fill")' in native_source
        assert 'Label("删除", systemImage: "trash")' in native_source
        assert '"api/performance/benchmark"' in native_source
        assert '"测试性能"' in native_source
        assert 'api/generate-stream/\\(jobID)/audio' in native_source
        service_source = (ROOT / "clis" / "qwen_tts_app.py").read_text(encoding="utf-8")
        assert "await request.is_disconnected()" in service_source
        start_script = (ROOT / "start-macos.sh").read_text(encoding="utf-8")
        assert "Refusing LAN exposure with an empty/weak password" in start_script

        disallowed_cors = client.options(
            "/api/health",
            headers={
                "Origin": "https://untrusted.example",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert "access-control-allow-origin" not in disallowed_cors.headers

        service_settings = client.get("/api/service-settings")
        assert service_settings.status_code == 200
        assert service_settings.json()["settings"]["reference_audio_path"]

        created_book = client.post(
            "/api/document-projects",
            files={"document": ("测试小说.txt", "第一章 开始\n这是一段测试正文。", "text/plain")},
            data={"name": "测试小说", "max_chars": "180", "settings_json": "{}"},
        )
        assert created_book.status_code == 200
        book_id = created_book.json()["id"]
        renamed_book = client.put(
            f"/api/document-projects/{book_id}",
            json={"name": "重命名后的小说"},
        )
        assert renamed_book.status_code == 200
        assert renamed_book.json()["name"] == "重命名后的小说"
        deleted_book = client.delete(f"/api/document-projects/{book_id}")
        assert deleted_book.status_code == 200
        assert client.get(f"/api/document-projects/{book_id}").status_code == 404

        runtime = client.get("/api/runtime").json()
        assert runtime["backend"] == "ggml"
        assert runtime["quant"] == "Q4_K_M"
        assert [profile["id"] for profile in runtime["runtime"]["profiles"]] == [
            "qwen_0_6b",
            "qwen_1_7b",
        ]
        voices = client.get("/api/voices").json()
        assert voices["voices"]
        assert voices["default_reference_audio_path"]


def test_performance_recommendation_selects_stream_chunk_and_parallelism() -> None:
    module = load_app_module()
    recommendation = module.choose_recommendation(
        stream_measurements=[
            {
                "chunk_frames": 4,
                "elapsed_seconds": 3.0,
                "first_audio_seconds": 0.45,
                "generation_realtime_factor": 1.4,
            },
            {
                "chunk_frames": 8,
                "elapsed_seconds": 2.7,
                "first_audio_seconds": 0.65,
                "generation_realtime_factor": 1.8,
            },
        ],
        single_block_seconds=4.0,
        parallel_block_seconds=5.8,
    )
    assert recommendation["stream_chunk_frames"] == 4
    assert recommendation["block_parallel"] == 2
    conservative = module.choose_recommendation(
        stream_measurements=[],
        single_block_seconds=4.0,
        parallel_block_seconds=7.5,
    )
    assert conservative["stream_chunk_frames"] == 8
    assert conservative["block_parallel"] == 1


def test_random_seed_is_resolved_before_task_creation() -> None:
    module = load_app_module()
    assert module._safe_int(-1, default=1234, minimum=-1, maximum=999999) == -1


def test_direct_service_bind_defaults_are_safe() -> None:
    module = load_app_module()
    module._validate_bind_security("127.0.0.1", "")
    module._validate_bind_security("::1", "")
    module._validate_bind_security("0.0.0.0", "1234")
    with pytest.raises(ValueError, match="Refusing non-loopback exposure"):
        module._validate_bind_security("0.0.0.0", "")
    with pytest.raises(ValueError, match="Refusing non-loopback exposure"):
        module._validate_bind_security("::", "change-me")


def test_voice_presets_are_persistent_and_can_import_reference_audio(tmp_path: Path) -> None:
    module = load_app_module()
    app = module.create_app(
        qwen_python=sys.executable,
        qwentts_library="",
        output_dir=tmp_path / "output",
        upload_dir=tmp_path / "upload",
        preset_dir=tmp_path / "presets",
        preload=False,
        access_password="",
    )
    with TestClient(app) as client:
        imported = client.post(
            "/api/presets/reference-audio",
            files={"audio": ("my-voice.wav", silent_wav_bytes(), "audio/wav")},
        )
        assert imported.status_code == 200
        reference_path = imported.json()["reference_audio_path"]
        assert Path(reference_path).is_file()
        assert client.get("/api/reference-audio", params={"path": reference_path}).status_code == 200

        created = client.post(
            "/api/presets",
            json={
                "name": "旁白测试",
                "settings": {
                    "model_profile": "qwen_0_6b",
                    "reference_audio_path": reference_path,
                    "qwen_seed": "1234",
                    "qwen_temperature": "0.9",
                    "qwen_top_p": 0.95,
                    "qwen_top_k": 40,
                    "qwen_repetition_penalty": 1.08,
                    "qwen_max_new_tokens": 2048,
                    "qwen_chunk_size": 8,
                    "qwen_min_new_tokens": 2,
                    "qwen_append_silence": True,
                    "qwen_aac_bitrate": "80k",
                },
            },
        )
        assert created.status_code == 201
        preset = created.json()
        assert preset["name"] == "旁白测试"
        assert preset["settings"]["reference_audio_path"] == reference_path
        assert preset["settings"]["qwen_top_k"] == 40
        assert preset["settings"]["qwen_aac_bitrate"] == "80k"

        activated = client.put(
            "/api/service-settings/active-preset",
            json={"preset_id": preset["id"]},
        )
        assert activated.status_code == 200
        assert activated.json()["active_preset_id"] == preset["id"]
        assert client.get("/api/service-settings").json()["name"] == "旁白测试"

        listed = client.get("/api/presets")
        assert listed.status_code == 200
        assert [item["id"] for item in listed.json()["presets"]] == [preset["id"]]
        assert listed.json()["active_preset_id"] == preset["id"]

        updated = client.put(
            f"/api/presets/{preset['id']}",
            content=json.dumps(
                {
                    "name": "旁白测试（更新）",
                    "settings": {
                        "model_profile": "qwen_1_7b",
                        "reference_audio_path": reference_path,
                        "qwen_seed": "5678",
                    },
                },
                ensure_ascii=False,
            ),
            headers={"Content-Type": "application/json"},
        )
        assert updated.status_code == 200
        assert updated.json()["settings"]["model_profile"] == "qwen_1_7b"
        assert client.get("/api/service-settings").json()["settings"]["model_profile"] == "qwen_1_7b"

        applied = client.put(
            "/api/service-settings",
            json={
                "name": "工作台应用",
                "settings": {
                    "model_profile": "qwen_0_6b",
                    "voice_name": "工作台音色",
                    "reference_audio_path": reference_path,
                    "qwen_seed": -1,
                    "qwen_temperature": 0.85,
                },
            },
        )
        assert applied.status_code == 200
        assert applied.json()["source"] == "studio"
        assert client.get("/api/service-settings").json()["settings"]["qwen_seed"] == -1

        deleted = client.delete(f"/api/presets/{preset['id']}")
        assert deleted.status_code == 200
        assert client.get("/api/presets").json()["presets"] == []


def test_reference_audio_library_can_hide_restore_and_safely_delete(tmp_path: Path) -> None:
    module = load_app_module()
    app = module.create_app(
        qwen_python=sys.executable,
        qwentts_library="",
        output_dir=tmp_path / "output",
        upload_dir=tmp_path / "upload",
        preset_dir=tmp_path / "presets",
        preload=False,
        access_password="",
    )
    with TestClient(app) as client:
        imported = client.post(
            "/api/presets/reference-audio",
            files={"audio": ("narrator.wav", silent_wav_bytes(sample_rate=48000), "audio/wav")},
        )
        assert imported.status_code == 200
        reference_path = imported.json()["reference_audio_path"]
        custom = next(
            item
            for item in client.get(
                "/api/reference-audio-library",
                params={"include_hidden": True},
            ).json()["references"]
            if item["path"] == reference_path
        )
        assert custom["kind"] == "custom"
        assert custom["name"] == "narrator"
        assert any(item["audio_path"] == reference_path for item in client.get("/api/voices").json()["voices"])

        hidden = client.put(
            f"/api/reference-audio-library/{custom['id']}/visibility",
            json={"hidden": True},
        )
        assert hidden.status_code == 200
        assert all(item["audio_path"] != reference_path for item in client.get("/api/voices").json()["voices"])
        restored = client.put(
            f"/api/reference-audio-library/{custom['id']}/visibility",
            json={"hidden": False},
        )
        assert restored.status_code == 200

        preset = client.post(
            "/api/presets",
            json={
                "name": "引用预设",
                "settings": {
                    "model_profile": "qwen_0_6b",
                    "voice_name": "narrator",
                    "reference_audio_path": reference_path,
                },
            },
        )
        assert preset.status_code == 201
        applied = client.put(
            "/api/service-settings",
            json={
                "name": "引用保护",
                "settings": {
                    "model_profile": "qwen_0_6b",
                    "reference_audio_path": reference_path,
                },
            },
        )
        assert applied.status_code == 200
        blocked = client.delete(f"/api/reference-audio-library/{custom['id']}")
        assert blocked.status_code == 409
        assert Path(reference_path).is_file()

        deleted = client.delete(
            f"/api/reference-audio-library/{custom['id']}",
            params={"replace_usages": True},
        )
        assert deleted.status_code == 200
        assert deleted.json()["replaced_usages"] == ["预设“引用预设”", "当前服务设置"]
        assert deleted.json()["replacement"]["name"] == "龙嫱"
        service_settings = client.get("/api/service-settings").json()["settings"]
        assert service_settings["reference_audio_path"] != reference_path
        assert service_settings["voice_name"] == "龙嫱"
        saved_preset = client.get("/api/presets").json()["presets"][0]
        assert saved_preset["settings"]["reference_audio_path"] != reference_path
        assert saved_preset["settings"]["voice_name"] == "龙嫱"
        assert not Path(reference_path).exists()

        builtin = next(
            item
            for item in client.get(
                "/api/reference-audio-library",
                params={"include_hidden": True},
            ).json()["references"]
            if item["kind"] == "builtin"
        )
        assert client.delete(f"/api/reference-audio-library/{builtin['id']}").status_code == 400


def test_service_accepts_bearer_api_clients_and_stt_uploads(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = load_app_module()
    app = module.create_app(
        qwen_python=sys.executable,
        qwentts_library="",
        output_dir=tmp_path / "output",
        upload_dir=tmp_path / "upload",
        preload=False,
        stt_preload=False,
        access_password="1234",
    )
    captured: dict[str, object] = {}

    def fake_transcribe(**kwargs):
        captured.update(kwargs)
        return (
            json.dumps({"text": "这是流式服务的转写测试。"}, ensure_ascii=False).encode(),
            "application/json",
        )

    monkeypatch.setattr(app.state.stt_runtime, "transcribe", fake_transcribe)
    headers = {"Authorization": "Bearer 1234"}
    with TestClient(app) as client:
        assert client.get("/api/voices").status_code == 401
        assert client.get("/api/voices", headers=headers).status_code == 200
        response = client.post(
            "/v1/audio/transcriptions",
            headers=headers,
            files={"file": ("sample.wav", b"RIFF-test-audio", "audio/wav")},
            data={"language": "zh", "response_format": "json"},
        )
        assert response.status_code == 200
        assert response.json()["text"] == "这是流式服务的转写测试。"
        assert captured["filename"] == "sample.wav"
        assert captured["language"] == "zh"


def test_tts_audio_endpoint_streams_generated_pcm_chunks(monkeypatch, tmp_path: Path) -> None:
    module = load_app_module()
    module.DEFAULT_SERVICE_JOB_DIR = tmp_path / "jobs"
    chunks = [
        module.torch.tensor([[0.10, -0.10, 0.20, -0.20]], dtype=module.torch.float32),
        module.torch.tensor([[0.30, -0.30, 0.40, -0.40]], dtype=module.torch.float32),
    ]

    class FakeRuntime:
        sample_rate = 24000
        n_vq = 16

        def synthesize(self, _request, *, output_dir):
            output = Path(output_dir)
            output.mkdir(parents=True, exist_ok=True)
            yield types.SimpleNamespace(
                type="metadata",
                data={"sample_rate": 24000, "channels": 1},
            )
            elapsed = 0.0
            for index, waveform in enumerate(chunks, start=1):
                elapsed += waveform.shape[-1] / 24000
                yield types.SimpleNamespace(
                    type="audio",
                    data={
                        "waveform": waveform,
                        "generated_frames": index,
                        "emitted_audio_seconds": elapsed,
                        "generated_audio_seconds": elapsed,
                        "sample_rate": 24000,
                    },
                )
            audio_path = output / "fake.wav"
            tokens_path = output / "fake.npy"
            metadata_path = output / "fake.json"
            audio_path.write_bytes(b"RIFF-result")
            tokens_path.write_bytes(b"tokens")
            metadata_path.write_text("{}", encoding="utf-8")
            yield types.SimpleNamespace(
                type="result",
                data={
                    "audio_path": str(audio_path),
                    "tokens_path": str(tokens_path),
                    "metadata_path": str(metadata_path),
                    "metadata": {
                        "generated_frames": len(chunks),
                        "duration_seconds": elapsed,
                    },
                },
            )

    class FakeRuntimeManager:
        def __init__(self, **_kwargs):
            self.qwen_backend = "ggml"
            self.qwen_quant = "Q4_K_M"
            self.qwentts_library = ""
            self.device = "metal"
            self.dtype = "gguf"
            self.attn_implementation = "ggml_metal"
            self.profiles = {
                "qwen_0_6b": {
                    "label": "test",
                    "sample_rate": 24000,
                    "channels": 1,
                    "streaming": True,
                    "backend": "qwen",
                },
                "qwen_1_7b": {
                    "label": "test",
                    "sample_rate": 24000,
                    "channels": 1,
                    "streaming": True,
                    "backend": "qwen",
                },
            }

        @contextmanager
        def session(self, _profile):
            yield FakeRuntime()

        def status(self):
            return {"state": "ready", "device": "metal", "profiles": []}

        def close(self):
            return None

    monkeypatch.setattr(module, "RuntimeManager", FakeRuntimeManager)
    app = module.create_app(
        qwen_python=sys.executable,
        qwentts_library="",
        output_dir=tmp_path / "output",
        upload_dir=tmp_path / "upload",
        preset_dir=tmp_path / "presets",
        preload=False,
        stt_preload=False,
        access_password="",
    )
    with TestClient(app) as client:
        voice = client.get("/api/voices").json()["voices"][0]
        applied = client.put(
            "/api/service-settings",
            json={
                "name": "API 默认音色",
                "settings": {
                    "model_profile": "qwen_1_7b",
                    "voice_name": voice["name"],
                    "reference_audio_path": voice["audio_path"],
                    "qwen_clone_mode": "xvec",
                    "qwen_seed": 4321,
                },
            },
        )
        assert applied.status_code == 200
        started = client.post(
            "/api/generate-stream/start",
            data={
                "text": "测试真正的 PCM 流式输出。",
                "streaming_generation": "1",
                "model_profile": "qwen_0_6b",
            },
        )
        assert started.status_code == 200
        assert started.json()["model_profile"] == "qwen_1_7b"
        assert started.json()["seed"] == 4321
        job_id = started.json()["job_id"]
        with client.stream("GET", f"/api/generate-stream/{job_id}/audio") as streamed:
            payload = b"".join(streamed.iter_raw())
            assert streamed.headers["x-audio-codec"] == "pcm_s16le"
            assert streamed.headers["x-audio-sample-rate"] == "24000"
        expected = b"".join(module._pcm16le_bytes(chunk, 1) for chunk in chunks)
        assert payload == expected
        status = client.get(f"/api/generate-stream/{job_id}/status").json()
        assert status["state"] == "finished"
        assert status["generated_frames"] == 2
        completed_result = client.get(f"/api/generate-stream/{job_id}/result").json()
        Path(completed_result["audio_path"]).unlink()
        assert client.get(f"/api/generate-stream/{job_id}/result-audio").status_code == 404

        def fake_ffmpeg(command, **_kwargs):
            Path(command[-1]).write_bytes(b"fake-aac")
            return types.SimpleNamespace(returncode=0, stderr="")

        monkeypatch.setattr(module.subprocess, "run", fake_ffmpeg)
        nonstream = client.post(
            "/api/generate-stream/start",
            data={
                "text": "逐段非流式高质量听书。",
                "streaming_generation": "0",
                "qwen_non_streaming_mode": "1",
                "ephemeral_audio": "1",
            },
        )
        assert nonstream.status_code == 200
        assert nonstream.json()["streaming_generation"] is False
        ephemeral_job_id = nonstream.json()["job_id"]
        for _ in range(50):
            ephemeral_status = client.get(
                f"/api/generate-stream/{ephemeral_job_id}/status"
            ).json()
            if ephemeral_status["state"] == "finished":
                break
            time.sleep(0.01)
        assert ephemeral_status["state"] == "finished"
        result = client.get(f"/api/generate-stream/{ephemeral_job_id}/result").json()
        source_paths = [
            Path(result[key])
            for key in ("audio_path", "tokens_path", "metadata_path")
        ]
        encoded = client.get(
            f"/api/generate-stream/{ephemeral_job_id}/result-audio-aac",
            params={"bitrate": "64k"},
        )
        assert encoded.status_code == 200
        assert encoded.headers["content-type"].startswith("audio/mp4")
        assert encoded.content == b"fake-aac"
        cleaned = client.post(
            f"/api/generate-stream/{ephemeral_job_id}/ephemeral-audio/close"
        )
        assert cleaned.status_code == 200
        assert not any(path.exists() for path in source_paths)
        cleaned_status = client.get(
            f"/api/generate-stream/{ephemeral_job_id}/status"
        ).json()
        assert cleaned_status["ephemeral_cleaned"] is True
        assert cleaned_status["result_ready"] is False


def test_global_stop_rejects_api_keys_and_accepts_internal_session(tmp_path: Path) -> None:
    module = load_app_module()
    module.DEFAULT_SERVICE_JOB_DIR = tmp_path / "jobs"
    module.DEFAULT_DOCUMENT_PROJECT_DIR = tmp_path / "documents"
    module.DEFAULT_PERFORMANCE_TUNING_PATH = tmp_path / "performance.json"
    app = module.create_app(
        qwen_python=sys.executable,
        qwentts_library="",
        output_dir=tmp_path / "output",
        upload_dir=tmp_path / "upload",
        preset_dir=tmp_path / "presets",
        preload=False,
        stt_preload=False,
        access_password="1234",
    )

    with TestClient(app) as client:
        external = client.post(
            "/api/service/stop-all",
            headers={"Authorization": "Bearer 1234"},
        )
        assert external.status_code == 403

        login = client.post(
            "/login",
            data={"password": "1234", "next_path": "/reader"},
            follow_redirects=False,
        )
        assert login.status_code == 303
        stopped = client.post("/api/service/stop-all")
        assert stopped.status_code == 200
        assert stopped.json()["ok"] is True
        assert stopped.json()["playback_epoch"] == 1

        tts_stopped = client.post("/api/tts/stop")
        assert tts_stopped.status_code == 200
        assert tts_stopped.json()["tts_enabled"] is False
        assert client.get("/api/health").json()["tts_enabled"] is False
        rejected_tts = client.post("/api/generate-stream/start", data={"text": "服务已停止"})
        assert rejected_tts.status_code == 503


def test_api_scheduler_prioritizes_internal_work_and_keeps_external_fifo() -> None:
    module = load_app_module()
    scheduler = module.GpuGenerationScheduler(max_parallel=1, interactive_burst_limit=2)
    order: list[str] = []

    def run(label: str, caller_kind: str) -> None:
        with scheduler.api_slot(caller_kind):
            order.append(label)
            time.sleep(0.02)

    with scheduler.api_slot("internal"):
        threads = [
            threading.Thread(target=run, args=("external-1", "external")),
            threading.Thread(target=run, args=("external-2", "external")),
            threading.Thread(target=run, args=("internal-1", "internal")),
            threading.Thread(target=run, args=("internal-2", "internal")),
            threading.Thread(target=run, args=("internal-3", "internal")),
        ]
        for thread in threads:
            thread.start()
            time.sleep(0.01)

    for thread in threads:
        thread.join(timeout=3)
        assert not thread.is_alive()

    assert set(order[:2]).issubset({"internal-1", "internal-2", "internal-3"})
    assert order[2] == "external-1"
    assert order[3].startswith("internal-")
    assert order[4] == "external-2"


def test_dead_qwen_worker_is_retired_and_restarted(tmp_path: Path) -> None:
    module = load_app_module()

    class FakeProcess:
        def __init__(self, return_code):
            self.return_code = return_code

        def poll(self):
            return self.return_code

    class DeadWorker:
        lane_index = 0
        process = FakeProcess(9)
        closed = False

        def generate(self, _payload):
            if False:
                yield {}
            raise RuntimeError("worker crashed")

        def health(self):
            raise ConnectionError("worker is gone")

        def close(self):
            self.closed = True

    class ReplacementWorker:
        lane_index = 0
        process = FakeProcess(None)

        def health(self):
            return {"ok": True}

        def close(self):
            return None

    dead = DeadWorker()
    replacement = ReplacementWorker()
    runtime = object.__new__(module.QwenWorkerRuntime)
    runtime.profile_id = "qwen_0_6b"
    runtime._closed = False
    runtime._workers = [dead]
    runtime._available = module.queue.Queue()
    runtime._available.put(dead)
    runtime._worker_lock = threading.RLock()
    runtime._worker_config = {}
    runtime.reference_audio_cache = {}
    runtime.reference_audio_cache_hits = 0
    runtime.reference_audio_cache_misses = 0
    runtime.reference_audio_cache_lock = threading.RLock()
    runtime._create_worker = lambda _lane_index: replacement
    request = module.StreamingRequest(
        text="触发 worker 崩溃恢复",
        prompt_audio_path=str(tmp_path / "reference.wav"),
    )

    try:
        list(runtime.synthesize(request, output_dir=tmp_path / "output"))
    except RuntimeError as exc:
        assert "worker crashed" in str(exc)
    else:
        raise AssertionError("dead worker failure should reach the caller")

    for _ in range(100):
        if replacement in runtime._workers:
            break
        time.sleep(0.01)
    assert dead.closed is True
    assert dead not in runtime._workers
    assert replacement in runtime._workers
    assert runtime._available.get_nowait() is replacement


def test_ggml_worker_passes_effective_seed_to_native_stream(tmp_path: Path) -> None:
    fake_package = types.ModuleType("faster_qwen3_tts")
    fake_package.FasterQwen3TTS = object
    previous = sys.modules.get("faster_qwen3_tts")
    sys.modules["faster_qwen3_tts"] = fake_package
    try:
        worker_path = ROOT / "qwen_tts_service" / "qwen_worker.py"
        spec = importlib.util.spec_from_file_location("qwen_worker_test", worker_path)
        assert spec is not None and spec.loader is not None
        worker_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(worker_module)
    finally:
        if previous is None:
            sys.modules.pop("faster_qwen3_tts", None)
        else:
            sys.modules["faster_qwen3_tts"] = previous

    class FakeModel:
        def _resolve_clone_reference(self, **_kwargs):
            return {"ref_spk_emb": np.zeros(4, dtype=np.float32)}, 1.0, {"mode": "clone"}

        def _stream_runtime(self, **kwargs):
            self.stream_kwargs = kwargs
            yield np.zeros(240, dtype=np.float32), 24000, {"total_steps_so_far": 1}

    state = object.__new__(worker_module.WorkerState)
    state.profile_id = "qwen_0_6b"
    state.model_path = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    state.backend = "ggml"
    state.quant = "Q4_K_M"
    state.started_at = time.time()
    state.requests = 0
    state.reference_keys = set()
    state.model = FakeModel()
    state.sample_rate = 24000

    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"test")
    events = list(
        state.generate(
            {
                "text": "测试文本",
                "ref_audio": str(reference),
                "output_dir": str(tmp_path / "output"),
                "seed": 424242,
                "xvec_only": True,
            }
        )
    )

    assert state.model.stream_kwargs["seed"] == 424242
    assert events[0]["data"]["seed"] == 424242
    assert events[-1]["data"]["metadata"]["seed"] == 424242


def load_worker_module():
    fake_package = types.ModuleType("faster_qwen3_tts")
    fake_package.FasterQwen3TTS = object
    previous = sys.modules.get("faster_qwen3_tts")
    sys.modules["faster_qwen3_tts"] = fake_package
    try:
        worker_path = ROOT / "qwen_tts_service" / "qwen_worker.py"
        spec = importlib.util.spec_from_file_location("qwen_worker_test", worker_path)
        assert spec is not None and spec.loader is not None
        worker_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(worker_module)
        return worker_module
    finally:
        if previous is None:
            sys.modules.pop("faster_qwen3_tts", None)
        else:
            sys.modules["faster_qwen3_tts"] = previous


def make_worker_state(worker_module, model):
    state = object.__new__(worker_module.WorkerState)
    state.profile_id = "qwen_0_6b"
    state.model_path = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    state.backend = "ggml"
    state.quant = "Q4_K_M"
    state.started_at = time.time()
    state.requests = 0
    state.reference_keys = set()
    state.model = model
    state.sample_rate = 24000
    return state


def test_worker_retries_runaway_generation_with_fresh_seed(tmp_path: Path) -> None:
    worker_module = load_worker_module()
    worker_module.RUNAWAY_MIN_BUDGET_SECONDS = 2.0
    worker_module.RUNAWAY_SECONDS_PER_CHAR = 0.0
    worker_module._retry_seed = lambda previous: 777777

    class RunawayThenHealthyModel:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def _resolve_clone_reference(self, **_kwargs):
            return {"ref_spk_emb": np.zeros(4, dtype=np.float32)}, 1.0, {"mode": "clone"}

        def _stream_runtime(self, **kwargs):
            self.calls.append(kwargs)
            if kwargs["seed"] == 424242:
                # 失控：单块音频超过 2 秒预算（复读坍缩直到撞帧上限的表现）
                yield np.zeros(24000 * 3, dtype=np.float32), 24000, {}
            else:
                yield np.zeros(19200, dtype=np.float32), 24000, {}

    model = RunawayThenHealthyModel()
    state = make_worker_state(worker_module, model)
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"test")
    events = list(
        state.generate(
            {
                "text": "测试失控重试",
                "ref_audio": str(reference),
                "output_dir": str(tmp_path / "output"),
                "seed": 424242,
                "xvec_only": True,
                "repetition_penalty": 1.05,
            }
        )
    )

    assert [call["seed"] for call in model.calls] == [424242, 777777]
    # 重试时提高重复惩罚，帮助模型跳出复读吸引域
    assert model.calls[0]["repetition_penalty"] == 1.05
    assert model.calls[1]["repetition_penalty"] == 1.15

    progress = [event for event in events if event["type"] == "progress"]
    assert progress and progress[0]["data"]["runaway_retry"] == 1
    audio_events = [event for event in events if event["type"] == "audio"]
    # 只有重试那一轮的音频块被送出；19200 样本 = 10 帧
    assert len(audio_events) == 1
    assert audio_events[0]["data"]["samples"] == 19200
    assert audio_events[0]["data"]["generated_frames"] == 10

    result = [event for event in events if event["type"] == "result"][0]
    metadata = result["data"]["metadata"]
    assert metadata["runaway_retries"] == 1
    assert metadata["effective_seed"] == 777777
    assert metadata["seed"] == 424242
    assert metadata["generated_frames"] == 10

    import soundfile as sf

    waveform, sample_rate = sf.read(result["data"]["audio_path"], dtype="float32")
    assert sample_rate == 24000
    assert len(waveform) == 19200


def test_worker_aborts_when_runaway_persists_after_retry(tmp_path: Path) -> None:
    worker_module = load_worker_module()
    worker_module.RUNAWAY_MIN_BUDGET_SECONDS = 2.0
    worker_module.RUNAWAY_SECONDS_PER_CHAR = 0.0
    worker_module._retry_seed = lambda previous: 777777

    class AlwaysRunawayModel:
        def _resolve_clone_reference(self, **_kwargs):
            return {"ref_spk_emb": np.zeros(4, dtype=np.float32)}, 1.0, {"mode": "clone"}

        def _stream_runtime(self, **_kwargs):
            while True:
                yield np.zeros(24000 * 3, dtype=np.float32), 24000, {}

    state = make_worker_state(worker_module, AlwaysRunawayModel())
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"test")
    with pytest.raises(RuntimeError, match="失控"):
        for _ in state.generate(
            {
                "text": "测试持续失控",
                "ref_audio": str(reference),
                "output_dir": str(tmp_path / "output"),
                "seed": 424242,
                "xvec_only": True,
            }
        ):
            pass


def test_worker_reports_frames_from_emitted_samples(tmp_path: Path) -> None:
    worker_module = load_worker_module()

    class FakeModel:
        def _resolve_clone_reference(self, **_kwargs):
            return {"ref_spk_emb": np.zeros(4, dtype=np.float32)}, 1.0, {"mode": "clone"}

        def _stream_runtime(self, **_kwargs):
            # GGML 路径的 timing 不含 total_steps_so_far
            for _ in range(3):
                yield np.zeros(19200, dtype=np.float32), 24000, {"chunk_index": 0}

    state = make_worker_state(worker_module, FakeModel())
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"test")
    events = list(
        state.generate(
            {
                "text": "测试帧数统计",
                "ref_audio": str(reference),
                "output_dir": str(tmp_path / "output"),
                "seed": 424242,
                "xvec_only": True,
            }
        )
    )
    audio_events = [event for event in events if event["type"] == "audio"]
    # 修复前 generated_frames 永远卡在第一个块的值（8）
    assert [event["data"]["generated_frames"] for event in audio_events] == [10, 20, 30]
    result = [event for event in events if event["type"] == "result"][0]
    assert result["data"]["metadata"]["generated_frames"] == 30


def test_stream_audio_headers_report_profile_format_while_job_queued(
    monkeypatch, tmp_path: Path
) -> None:
    module = load_app_module()
    module.DEFAULT_SERVICE_JOB_DIR = tmp_path / "jobs"
    gate = threading.Event()

    class FakeRuntime:
        sample_rate = 24000
        n_vq = 16

        def synthesize(self, _request, *, output_dir):
            output = Path(output_dir)
            output.mkdir(parents=True, exist_ok=True)
            audio_path = output / "fake.wav"
            tokens_path = output / "fake.npy"
            metadata_path = output / "fake.json"
            audio_path.write_bytes(b"RIFF-result")
            tokens_path.write_bytes(b"tokens")
            metadata_path.write_text("{}", encoding="utf-8")
            yield types.SimpleNamespace(type="metadata", data={"sample_rate": 24000, "channels": 1})
            yield types.SimpleNamespace(
                type="result",
                data={
                    "audio_path": str(audio_path),
                    "tokens_path": str(tokens_path),
                    "metadata_path": str(metadata_path),
                    "metadata": {"generated_frames": 1, "duration_seconds": 0.1},
                },
            )

    class FakeRuntimeManager:
        def __init__(self, **_kwargs):
            self.qwen_backend = "ggml"
            self.qwen_quant = "Q4_K_M"
            self.qwentts_library = ""
            self.device = "metal"
            self.dtype = "gguf"
            self.attn_implementation = "ggml_metal"
            self.profiles = {
                "qwen_0_6b": {
                    "label": "test",
                    "sample_rate": 24000,
                    "channels": 1,
                    "streaming": True,
                    "backend": "qwen",
                },
                "qwen_1_7b": {
                    "label": "test",
                    "sample_rate": 24000,
                    "channels": 1,
                    "streaming": True,
                    "backend": "qwen",
                },
            }

        @contextmanager
        def session(self, _profile):
            # 让任务停在 loading_runtime：并发占满 lane 时客户端立刻拉流，
            # 修复前头部会泄露 48000/2 占位默认值
            gate.wait(timeout=10.0)
            yield FakeRuntime()

        def status(self):
            return {"state": "ready", "device": "metal", "profiles": []}

        def close(self):
            return None

    monkeypatch.setattr(module, "RuntimeManager", FakeRuntimeManager)
    app = module.create_app(
        qwen_python=sys.executable,
        qwentts_library="",
        output_dir=tmp_path / "output",
        upload_dir=tmp_path / "upload",
        preset_dir=tmp_path / "presets",
        preload=False,
        stt_preload=False,
        access_password="",
    )
    with TestClient(app) as client:
        voice = client.get("/api/voices").json()["voices"][0]
        applied = client.put(
            "/api/service-settings",
            json={
                "name": "API 默认音色",
                "settings": {
                    "model_profile": "qwen_1_7b",
                    "voice_name": voice["name"],
                    "reference_audio_path": voice["audio_path"],
                    "qwen_clone_mode": "xvec",
                    "qwen_seed": 4321,
                },
            },
        )
        assert applied.status_code == 200
        started = client.post(
            "/api/generate-stream/start",
            data={
                "text": "验证排队时头部信息正确。",
                "streaming_generation": "1",
                "model_profile": "qwen_0_6b",
            },
        )
        assert started.status_code == 200
        job_id = started.json()["job_id"]
        with client.stream("GET", f"/api/generate-stream/{job_id}/audio") as streamed:
            assert streamed.headers["x-audio-codec"] == "pcm_s16le"
            assert streamed.headers["x-audio-sample-rate"] == "24000"
            assert streamed.headers["x-audio-channels"] == "1"
        gate.set()
        deadline = time.time() + 5.0
        status = {}
        while time.time() < deadline:
            status = client.get(f"/api/generate-stream/{job_id}/status").json()
            if status.get("state") in {"finished", "closed", "error"}:
                break
            time.sleep(0.05)
        assert status.get("state") in {"finished", "closed"}


def test_worker_preserves_virtualenv_python_symlink(monkeypatch, tmp_path: Path) -> None:
    service_dir = ROOT / "qwen_tts_service"
    sys.path.insert(0, str(service_dir))
    try:
        import qwen_runtime
    finally:
        sys.path.remove(str(service_dir))

    real_python = tmp_path / "real-python"
    real_python.write_bytes(b"")
    venv_python = tmp_path / "venv-python"
    venv_python.symlink_to(real_python)
    captured: dict[str, object] = {}

    class FakeProcess:
        def __init__(self, command, **_kwargs):
            captured["command"] = command

        def poll(self):
            return None

    monkeypatch.setattr(qwen_runtime.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(qwen_runtime.QwenWorkerClient, "_wait_until_ready", lambda *_args: None)
    client = qwen_runtime.QwenWorkerClient(
        python_executable=venv_python,
        worker_script=ROOT / "qwen_tts_service" / "qwen_worker.py",
        model_dir="model",
        backend="ggml",
        quant="Q4_K_M",
        library_path=None,
        profile_id="qwen_0_6b",
        lane_index=0,
        port=7900,
        log_dir=tmp_path / "logs",
    )
    try:
        command = captured["command"]
        assert isinstance(command, list)
        assert command[0] == os.path.abspath(venv_python)
        assert command[0] != str(venv_python.resolve())
    finally:
        client._stdout.close()
        client._stderr.close()
