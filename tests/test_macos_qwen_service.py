from __future__ import annotations

import importlib.util
import sys
import time
import types
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[1]


def load_app_module():
    app_path = ROOT / "clis" / "qwen_tts_app.py"
    spec = importlib.util.spec_from_file_location("qwen_tts_app_test", app_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mac_app_only_exposes_qwen_profiles(tmp_path: Path) -> None:
    module = load_app_module()
    app = module.create_app(
        qwen_python=sys.executable,
        qwentts_library="",
        output_dir=tmp_path / "output",
        upload_dir=tmp_path / "upload",
        preload=False,
        access_password="",
    )

    with TestClient(app) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert "Qwen3-TTS · Apple Silicon Metal" in page.text
        assert "MOSS 高质量 4B" not in page.text

        runtime = client.get("/api/runtime").json()
        assert runtime["backend"] == "ggml"
        assert runtime["quant"] == "Q4_K_M"
        assert [profile["id"] for profile in runtime["runtime"]["profiles"]] == [
            "qwen_0_6b",
            "qwen_1_7b",
        ]


def test_random_seed_is_resolved_before_task_creation() -> None:
    module = load_app_module()
    assert module._safe_int(-1, default=1234, minimum=-1, maximum=999999) == -1


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
