"""Download Whisper large-v3 from ModelScope for local ICL transcription."""

from pathlib import Path

from modelscope import snapshot_download


TARGET = Path(__file__).resolve().parents[1] / "models" / "Whisper-large-v3"
TARGET.mkdir(parents=True, exist_ok=True)

snapshot_download(
    "AI-ModelScope/whisper-large-v3",
    local_dir=str(TARGET),
    allow_patterns=[
        "model.safetensors",
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "normalizer.json",
        "vocab.json",
        "merges.txt",
    ],
)
