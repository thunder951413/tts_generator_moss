"""Download both Qwen3-TTS Base checkpoints from ModelScope."""

from pathlib import Path

from modelscope import snapshot_download


ROOT = Path(__file__).resolve().parents[2] / "faster-qwen3-tts" / "models"
ROOT.mkdir(parents=True, exist_ok=True)

for model_name in ("Qwen3-TTS-12Hz-0.6B-Base", "Qwen3-TTS-12Hz-1.7B-Base"):
    snapshot_download(
        f"Qwen/{model_name}",
        local_dir=str(ROOT / model_name),
    )
