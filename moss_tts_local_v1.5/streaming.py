# coding=utf-8
"""Streaming inference helpers for MOSS-TTS Local Transformer v1.5.

The helper uses the Hugging Face local-transformer checkpoint together with
MOSS-Audio-Tokenizer-v2 and yields decoded audio chunks while generation is
still running.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import queue
import threading
import time
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator, Optional

import torch
import torchaudio
from transformers import AutoModel, AutoProcessor


DEFAULT_MODEL_DIR = "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5"
DEFAULT_CODEC_DIR = "OpenMOSS-Team/MOSS-Audio-Tokenizer-v2"
DEFAULT_OUTPUT_DIR = Path("outputs/moss_tts_local_v1_5_streaming")

TEXT_TO_AUDIO_TOKENS_PER_CHAR = {
    "zh": 3.098411951313033,
    "cmn": 3.098411951313033,
    "yue": 3.098411951313033,
    "chinese": 3.098411951313033,
    "english": 0.8673376262755219,
    "en": 0.8673376262755219,
    "french": 0.9,
    "fr": 0.9,
    "japanese": 2.2,
    "ja": 2.2,
    "korean": 1.8,
    "ko": 1.8,
}


@dataclass
class StreamingRuntime:
    model: Any
    processor: Any
    device: torch.device
    tts_device: torch.device
    codec_device: torch.device
    dtype: torch.dtype
    model_dir: str | Path
    codec_dir: str | Path
    sample_rate: int
    frame_rate: float
    n_vq: int
    attn_implementation: str
    codec_weight_dtype: str
    codec_compute_dtype: str
    reference_audio_cache: OrderedDict[tuple[str, int, int, int], torch.Tensor] = field(
        default_factory=OrderedDict, repr=False
    )
    reference_audio_cache_lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    codec_execution_lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    reference_audio_cache_hits: int = 0
    reference_audio_cache_misses: int = 0
    profile_id: str = "quality_4b"
    runtime_family: str = "local_v1_5"


@dataclass
class LegacyRuntime:
    """Runtime for the original 1.7B checkpoint and 24 kHz codec."""

    model: Any
    processor: Any
    device: torch.device
    tts_device: torch.device
    codec_device: torch.device
    dtype: torch.dtype
    model_dir: str | Path
    codec_dir: str | Path
    sample_rate: int
    frame_rate: float
    n_vq: int
    attn_implementation: str
    codec_weight_dtype: str
    codec_compute_dtype: str
    profile_id: str = "light_1_7b"
    runtime_family: str = "legacy_1_7b"
    reference_audio_cache: OrderedDict[tuple[str, int, int, int], torch.Tensor] = field(
        default_factory=OrderedDict, repr=False
    )
    reference_audio_cache_lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    codec_execution_lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    generation_execution_lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    reference_audio_cache_hits: int = 0
    reference_audio_cache_misses: int = 0


@dataclass
class StreamingRequest:
    text: str
    mode: str = "continuation"
    prompt_text: str = ""
    prompt_audio_path: Optional[str] = None
    language: str = ""
    tokens_control: bool = False
    tokens: int = 0
    max_new_frames: int = 7500
    do_sample: bool = True
    temperature: float = 1.2
    top_p: float = 1.0
    top_k: int = 25
    repetition_penalty: float = 1.0
    text_temperature: float = 1.0
    text_top_p: float = 1.0
    text_top_k: int = 50
    seed: Optional[int] = None
    codec_chunk_frames: int = 8
    qwen_xvec_only: bool = True
    qwen_reference_text: str = ""
    qwen_non_streaming_mode: bool = False
    qwen_append_silence: bool = True
    qwen_instruct: str = ""
    qwen_min_new_tokens: int = 2


@dataclass
class StreamingEvent:
    type: str
    data: dict[str, Any] = field(default_factory=dict)


def _resolve_dtype(dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    normalized = str(dtype or "bfloat16").strip().lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype!r}")


def resolve_attn_implementation(
    requested: str | None,
    device: torch.device,
    dtype: torch.dtype,
) -> str:
    requested_norm = str(requested or "flash_attention_2").strip().lower()
    if requested_norm in {"", "auto"}:
        requested_norm = "flash_attention_2"

    if requested_norm == "flash_attention_2":
        can_try_flash = (
            device.type == "cuda"
            and importlib.util.find_spec("flash_attn") is not None
            and dtype in {torch.float16, torch.bfloat16}
        )
        if can_try_flash:
            try:
                major, _ = torch.cuda.get_device_capability(device)
                if major >= 8:
                    return "flash_attention_2"
            except Exception:
                pass
        return "sdpa" if device.type == "cuda" else "eager"

    if requested_norm in {"sdpa", "eager"}:
        return requested_norm

    raise ValueError(
        "attn_implementation must be one of "
        "['auto', 'eager', 'flash_attention_2', 'sdpa'], "
        f"got {requested!r}."
    )


def _has_cjk(text: str) -> bool:
    return any("\u3400" <= ch <= "\u9fff" for ch in text or "")


def normalize_language_for_tokens(language: str, text: str) -> str:
    normalized = str(language or "").strip().lower().replace("_", "-")
    if normalized:
        if normalized.startswith("zh") or normalized in {"cmn", "yue", "chinese"}:
            return "zh"
        if normalized.startswith("en") or normalized == "english":
            return "en"
        if normalized.startswith("fr") or normalized == "french":
            return "fr"
        if normalized.startswith("ja") or normalized == "japanese":
            return "ja"
        if normalized.startswith("ko") or normalized == "korean":
            return "ko"
    return "zh" if _has_cjk(text) else "en"


def estimate_tokens(text: str, language: str = "") -> int:
    key = normalize_language_for_tokens(language, text)
    ratio = TEXT_TO_AUDIO_TOKENS_PER_CHAR.get(key, TEXT_TO_AUDIO_TOKENS_PER_CHAR["en"])
    return max(1, int(round(len(text or "") * float(ratio))))


def _move_batch_to_device(batch: Any, device: torch.device) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for key, value in dict(batch).items():
        if torch.is_tensor(value):
            result[key] = value.to(device)
        else:
            result[key] = value
    return result


def load_runtime(
    *,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    codec_dir: str | Path = DEFAULT_CODEC_DIR,
    device: str | torch.device = "cuda",
    tts_device: str | torch.device | None = None,
    codec_device: str | torch.device | None = None,
    dtype: str | torch.dtype = "bfloat16",
    attn_implementation: str = "flash_attention_2",
    codec_weight_dtype: str = "fp32",
    codec_compute_dtype: str = "bf16",
    warmup: bool = True,
) -> StreamingRuntime:
    model_ref = str(model_dir)
    codec_ref = str(codec_dir)
    local_model_path = Path(model_ref)
    local_files_only = local_model_path.exists()
    resolved_tts_device = torch.device(tts_device if tts_device is not None else device)
    resolved_codec_device = torch.device(codec_device if codec_device is not None else resolved_tts_device)
    if (resolved_tts_device.type == "cuda" or resolved_codec_device.type == "cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    resolved_dtype = _resolve_dtype(dtype)
    resolved_attn_implementation = resolve_attn_implementation(
        attn_implementation,
        resolved_tts_device,
        resolved_dtype,
    )

    processor = AutoProcessor.from_pretrained(
        model_ref,
        trust_remote_code=True,
        codec_path=codec_ref,
        codec_weight_dtype=codec_weight_dtype,
        codec_compute_dtype=codec_compute_dtype,
        codec_attention_implementation=resolved_attn_implementation,
    )
    model = AutoModel.from_pretrained(
        model_ref,
        trust_remote_code=True,
        dtype=resolved_dtype,
        local_files_only=local_files_only,
        attn_implementation=resolved_attn_implementation,
    )
    model.to(resolved_tts_device)
    model.eval()

    audio_tokenizer = processor.audio_tokenizer
    if hasattr(audio_tokenizer, "set_attention_implementation"):
        audio_tokenizer.set_attention_implementation(resolved_attn_implementation)
    if hasattr(audio_tokenizer, "set_compute_dtype"):
        audio_tokenizer.set_compute_dtype(codec_compute_dtype)
    if hasattr(audio_tokenizer, "to"):
        audio_tokenizer.to(resolved_codec_device)
    if hasattr(audio_tokenizer, "eval"):
        audio_tokenizer.eval()

    sample_rate = int(getattr(audio_tokenizer, "sampling_rate", getattr(model.config, "sampling_rate", 48000)))
    downsample_rate = int(getattr(audio_tokenizer, "downsample_rate", 3840))
    n_vq = int(getattr(model.config, "n_vq", 12))
    runtime = StreamingRuntime(
        model=model,
        processor=processor,
        device=resolved_tts_device,
        tts_device=resolved_tts_device,
        codec_device=resolved_codec_device,
        dtype=resolved_dtype,
        model_dir=model_ref,
        codec_dir=codec_ref,
        sample_rate=sample_rate,
        frame_rate=float(sample_rate) / float(downsample_rate),
        n_vq=n_vq,
        attn_implementation=resolved_attn_implementation,
        codec_weight_dtype=str(codec_weight_dtype),
        codec_compute_dtype=str(codec_compute_dtype),
    )
    if warmup:
        warmup_streaming_runtime(runtime)
    return runtime


def ensure_legacy_streaming_checkpoint_patch(model_dir: str | Path) -> None:
    """Patch the downloaded 1.7B remote-code module for local streaming.

    Model weights and remote-code files live under the ignored ``models``
    directory, so this small idempotent source transform makes the optimization
    reproducible after a fresh model download without committing model assets.
    """

    source_path = Path(model_dir) / "modeling_moss_tts.py"
    if not source_path.is_file():
        raise FileNotFoundError(f"1.7B model source not found: {source_path}")
    source = source_path.read_text(encoding="utf-8")
    original = source

    if "put_audio_frame = getattr(streamer, \"put_audio_frame\", None)" not in source:
        old = """            if streamer is not None:
                streamer.put(next_tokens[:, 0].cpu())
"""
        new = """            if streamer is not None:
                # Preserve the standard text-streamer contract, while allowing
                # local TTS runtimes to consume the complete RVQ row.
                put_audio_frame = getattr(streamer, \"put_audio_frame\", None)
                if callable(put_audio_frame):
                    put_audio_frame(next_tokens.detach().cpu())
                else:
                    streamer.put(next_tokens[:, 0].cpu())
"""
        if old not in source:
            raise RuntimeError("unsupported 1.7B checkpoint: streamer hook changed")
        source = source.replace(old, new, 1)

    if "sampling_generator = getattr(streamer, \"sampling_generator\", None)" not in source:
        old = """        # 提取配置参数
        # assert False
        speech_pad_idx = self.config.audio_pad_code
"""
        new = """        # 提取配置参数
        # assert False
        sampling_generator = getattr(streamer, \"sampling_generator\", None)
        speech_pad_idx = self.config.audio_pad_code
"""
        if old not in source:
            raise RuntimeError("unsupported 1.7B checkpoint: sampler prelude changed")
        source = source.replace(old, new, 1)
        old = """                    channel_ntk = torch.multinomial(nn.functional.softmax(next_token_score, dim=-1), num_samples=1).squeeze(1) # (B, )
"""
        new = """                    channel_ntk = torch.multinomial(
                        nn.functional.softmax(next_token_score, dim=-1),
                        num_samples=1,
                        generator=sampling_generator,
                    ).squeeze(1) # (B, )
"""
        if old not in source:
            raise RuntimeError("unsupported 1.7B checkpoint: multinomial call changed")
        source = source.replace(old, new, 1)

    if "local_past_key_values = None" not in source:
        old = """            local_trm_dim = self.local_transformer_config.hidden_size
            local_transformer_inputs = torch.zeros(batch_size, 0, local_trm_dim).to(device).to(dtype) # (B, 0 <= t <= Nq, D), 维护当前 local trm 的输入
            current_local_transformer_input = self.speech_embedding_to_local_mlp(global_trm_output_hidden_states) # (B, D) 维护当前 timestamp 的 local trm 的输入，
"""
        new = """            current_local_transformer_input = self.speech_embedding_to_local_mlp(global_trm_output_hidden_states) # (B, D) 维护当前 timestamp 的 local trm 的输入，
            local_past_key_values = None
"""
        if old not in source:
            raise RuntimeError("unsupported 1.7B checkpoint: local cache prelude changed")
        source = source.replace(old, new, 1)
        old = """                local_transformer_inputs = torch.cat([local_transformer_inputs, current_local_transformer_input.unsqueeze(1)], dim=1) # (B, t, D)
                local_transformer_outputs = self.local_transformer(
                    input_ids=None,
                    attention_mask=None,
                    inputs_embeds=local_transformer_inputs # (B, t=1+Nq, D)
                )[0] # (B, t=1+Nq, D)
                local_transformer_outputs = self.layer_norm_before_lm_heads[layer_index](
                    self.local_to_speech_embedding_mlps[layer_index](local_transformer_outputs) # (B, t=1+Nq, D)
                ) # (B, t=1+Nq, D)

                next_token_logit = self.lm_heads[layer_index](local_transformer_outputs[:, -1, :]) # (B, V)
"""
        new = """                local_transformer_outputs = self.local_transformer(
                    input_ids=None,
                    attention_mask=None,
                    past_key_values=local_past_key_values,
                    inputs_embeds=current_local_transformer_input.unsqueeze(1),
                    use_cache=True,
                    return_dict=True,
                )
                local_past_key_values = local_transformer_outputs.past_key_values
                local_hidden_state = self.layer_norm_before_lm_heads[layer_index](
                    self.local_to_speech_embedding_mlps[layer_index](
                        local_transformer_outputs.last_hidden_state[:, -1:, :]
                    )
                )

                next_token_logit = self.lm_heads[layer_index](local_hidden_state[:, -1, :]) # (B, V)
"""
        if old not in source:
            raise RuntimeError("unsupported 1.7B checkpoint: local transformer loop changed")
        source = source.replace(old, new, 1)

    if source != original:
        temporary = source_path.with_suffix(".py.tmp")
        temporary.write_text(source, encoding="utf-8")
        os.replace(temporary, source_path)


def load_legacy_runtime(
    *,
    model_dir: str | Path,
    codec_dir: str | Path,
    device: str | torch.device = "cuda",
    tts_device: str | torch.device | None = None,
    codec_device: str | torch.device | None = None,
    dtype: str | torch.dtype = "bfloat16",
    attn_implementation: str = "flash_attention_2",
) -> LegacyRuntime:
    """Load the official 1.7B release with its original audio tokenizer.

    The 1.7B repository exposes the regular ``model.generate`` API rather
    than the v1.5 frame iterator, so it is intentionally handled as a
    non-streaming backend.
    """

    model_ref = str(model_dir)
    codec_ref = str(codec_dir)
    ensure_legacy_streaming_checkpoint_patch(model_ref)
    resolved_tts_device = torch.device(tts_device if tts_device is not None else device)
    resolved_codec_device = torch.device(codec_device if codec_device is not None else resolved_tts_device)
    if (resolved_tts_device.type == "cuda" or resolved_codec_device.type == "cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    resolved_dtype = _resolve_dtype(dtype)
    resolved_attn = resolve_attn_implementation(attn_implementation, resolved_tts_device, resolved_dtype)
    local_files_only = Path(model_ref).exists() and Path(codec_ref).exists()

    processor = AutoProcessor.from_pretrained(
        model_ref,
        trust_remote_code=True,
        codec_path=codec_ref,
    )
    # The original codec checkpoint is stored as fp32 (~7 GB). Keeping it in
    # bf16/fp16 on GPU is necessary for 16 GB cards; its RVQ distance math
    # explicitly promotes codebook operations to fp32 internally.
    processor.audio_tokenizer = processor.audio_tokenizer.to(
        device=resolved_codec_device, dtype=resolved_dtype
    )
    processor.audio_tokenizer.eval()
    model = AutoModel.from_pretrained(
        model_ref,
        trust_remote_code=True,
        dtype=resolved_dtype,
        local_files_only=local_files_only,
        attn_implementation=resolved_attn,
    ).to(resolved_tts_device)
    model.eval()

    sample_rate = int(getattr(processor.model_config, "sampling_rate", 24000))
    n_vq = int(getattr(model.config, "n_vq", 32))
    runtime = LegacyRuntime(
        model=model,
        processor=processor,
        device=resolved_tts_device,
        tts_device=resolved_tts_device,
        codec_device=resolved_codec_device,
        dtype=resolved_dtype,
        model_dir=model_ref,
        codec_dir=codec_ref,
        sample_rate=sample_rate,
        frame_rate=12.5,
        n_vq=n_vq,
        attn_implementation=resolved_attn,
        codec_weight_dtype="fp32",
        codec_compute_dtype="fp32",
    )
    warmup_legacy_runtime(runtime)
    return runtime


@torch.inference_mode()
def synthesize_legacy_full(
    runtime: LegacyRuntime,
    request: StreamingRequest,
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> Generator[StreamingEvent, None, None]:
    """Compatibility fallback that generates a complete 1.7B waveform."""

    run_id = f"{int(time.time() * 1000)}_{os.getpid()}"
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    yield StreamingEvent(
        type="metadata",
        data={
            "run_id": run_id,
            "run_dir": str(run_dir),
            "sample_rate": runtime.sample_rate,
            "frame_rate": runtime.frame_rate,
            "n_vq": runtime.n_vq,
            "tts_device": str(runtime.tts_device),
            "codec_device": str(runtime.codec_device),
            "attn_implementation": runtime.attn_implementation,
            "processor_mode": "generation",
            "model_profile": runtime.profile_id,
        },
    )

    user_kwargs: dict[str, Any] = {"text": request.text, "language": request.language or "Chinese"}
    if request.prompt_audio_path:
        user_kwargs["reference"] = [request.prompt_audio_path]
    if request.tokens_control and request.tokens > 0:
        user_kwargs["tokens"] = int(request.tokens)
    conversations = [[runtime.processor.build_user_message(**user_kwargs)]]
    with (
        torch.autocast(device_type="cuda", dtype=runtime.dtype)
        if runtime.codec_device.type == "cuda" and runtime.dtype in {torch.float16, torch.bfloat16}
        else nullcontext()
    ):
        batch = runtime.processor(conversations, mode="generation", n_vq=runtime.n_vq)
    input_ids = batch["input_ids"].to(runtime.tts_device)
    attention_mask = batch["attention_mask"].to(runtime.tts_device)

    if request.seed is not None:
        torch.manual_seed(int(request.seed))
        if runtime.tts_device.type == "cuda":
            torch.cuda.manual_seed_all(int(request.seed))
    started_at = time.perf_counter()
    outputs = runtime.model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=int(request.max_new_frames),
        audio_temperature=float(request.temperature),
        audio_top_p=float(request.top_p),
        audio_top_k=int(request.top_k),
        audio_repetition_penalty=float(request.repetition_penalty),
    )
    with (
        torch.autocast(device_type="cuda", dtype=runtime.dtype)
        if runtime.codec_device.type == "cuda" and runtime.dtype in {torch.float16, torch.bfloat16}
        else nullcontext()
    ):
        messages = runtime.processor.decode(outputs)
    if not messages or messages[0] is None or not messages[0].audio_codes_list:
        raise RuntimeError("1.7B 模型没有返回可解码音频")
    waveform = messages[0].audio_codes_list[0]
    if not torch.is_tensor(waveform):
        waveform = torch.as_tensor(waveform)
    waveform = waveform.detach().cpu().to(torch.float32)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.ndim > 2:
        waveform = waveform.reshape(-1, waveform.shape[-1])
    audio_path = run_dir / "generated.wav"
    tokens_path = run_dir / "generated_tokens.pt"
    metadata_path = run_dir / "metadata.json"
    torchaudio.save(str(audio_path), waveform, runtime.sample_rate)
    torch.save(outputs.detach().cpu() if torch.is_tensor(outputs) else outputs, tokens_path)
    elapsed = time.perf_counter() - started_at
    duration = float(waveform.shape[-1]) / float(runtime.sample_rate)
    metadata = {
        "run_id": run_id,
        "model_profile": runtime.profile_id,
        "generated_frames": int(round(duration * runtime.frame_rate)),
        "duration_seconds": duration,
        "generation_seconds": elapsed,
        "sample_rate": runtime.sample_rate,
        "channels": int(waveform.shape[0]),
        "streaming_supported": False,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    yield StreamingEvent(
        type="result",
        data={
            "audio_path": str(audio_path),
            "tokens_path": str(tokens_path),
            "metadata_path": str(metadata_path),
            "metadata": metadata,
        },
    )


class LegacyAudioFrameStreamer:
    """Receive complete RVQ rows from the legacy model generation loop."""

    def __init__(
        self,
        *,
        sampling_generator: Optional[torch.Generator] = None,
    ) -> None:
        self.queue: queue.Queue[Optional[torch.Tensor]] = queue.Queue()
        self.sampling_generator = sampling_generator
        self._ended = False
        self._lock = threading.Lock()

    def put_audio_frame(self, value: torch.Tensor) -> None:
        self.queue.put(value.detach().cpu())

    def put(self, value: torch.Tensor) -> None:
        # GenerationMixin emits the complete input prompt once before entering
        # the checkpoint's custom sampling loop. It is not generated audio and
        # must not be decoded. Generated rows arrive through put_audio_frame().
        return

    def end(self) -> None:
        with self._lock:
            if self._ended:
                return
            self._ended = True
            self.queue.put(None)


def build_legacy_processor_inputs(
    runtime: LegacyRuntime,
    request: StreamingRequest,
) -> dict[str, torch.Tensor]:
    autocast_context = (
        torch.autocast(device_type="cuda", dtype=runtime.dtype)
        if runtime.codec_device.type == "cuda"
        and runtime.dtype in {torch.float16, torch.bfloat16}
        else nullcontext()
    )
    with autocast_context:
        user_kwargs: dict[str, Any] = {
            "text": request.text,
            "language": request.language or "Chinese",
        }
        if request.prompt_audio_path:
            user_kwargs["reference"] = [
                cached_reference_audio_codes(runtime, request.prompt_audio_path)
            ]
        if request.tokens_control and request.tokens > 0:
            user_kwargs["tokens"] = int(request.tokens)
        conversations = [[runtime.processor.build_user_message(**user_kwargs)]]
        batch = runtime.processor(conversations, mode="generation", n_vq=runtime.n_vq)
    return _move_batch_to_device(batch, runtime.tts_device)


@torch.inference_mode()
def warmup_legacy_runtime(runtime: LegacyRuntime, *, codec_frames: int = 4) -> None:
    """Warm the legacy codec; TTS is warmed naturally by the first real request."""
    try:
        frames = max(1, int(codec_frames))
        dummy_codes = torch.zeros((frames, runtime.n_vq), dtype=torch.long)
        with runtime.codec_execution_lock:
            with StatefulCodecDecoder(runtime.processor.audio_tokenizer, n_vq=runtime.n_vq) as decoder:
                with (
                    torch.autocast(device_type="cuda", dtype=runtime.dtype)
                    if runtime.codec_device.type == "cuda"
                    and runtime.dtype in {torch.float16, torch.bfloat16}
                    else nullcontext()
                ):
                    _ = decoder.decode_codes(
                        dummy_codes.to(device=decoder.device, dtype=torch.long)
                    )
        if runtime.codec_device.type == "cuda":
            torch.cuda.synchronize(runtime.codec_device)
    except Exception as exc:  # noqa: BLE001
        print(f"[moss_tts_local.v1.5] 1.7B codec warmup skipped: {exc}")


def _synthesize_legacy_stream_unlocked(
    runtime: LegacyRuntime,
    request: StreamingRequest,
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> Generator[StreamingEvent, None, None]:
    """Stream the original 1.7B model with cached prompts and chunked decoding."""

    batch = build_legacy_processor_inputs(runtime, request)
    output_dir = Path(output_dir)
    run_id = f"{int(time.time() * 1000)}_{abs(hash((request.text, request.mode, runtime.profile_id))) % 1000000:06d}"
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    yield StreamingEvent(
        type="metadata",
        data={
            "run_id": run_id,
            "run_dir": str(run_dir),
            "sample_rate": runtime.sample_rate,
            "frame_rate": runtime.frame_rate,
            "n_vq": runtime.n_vq,
            "tts_device": str(runtime.tts_device),
            "codec_device": str(runtime.codec_device),
            "attn_implementation": runtime.attn_implementation,
            "processor_mode": "generation",
            "model_profile": runtime.profile_id,
        },
    )

    sampling_generator: Optional[torch.Generator] = None
    if request.seed is not None and int(request.seed) >= 0:
        sampling_generator = torch.Generator(device=runtime.tts_device)
        sampling_generator.manual_seed(int(request.seed))

    frame_streamer = LegacyAudioFrameStreamer(
        sampling_generator=sampling_generator,
    )
    generation_result: dict[str, Any] = {}
    generation_error: list[BaseException] = []

    def _generation_worker() -> None:
        try:
            with torch.inference_mode():
                generation_result["outputs"] = runtime.model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    max_new_tokens=int(request.max_new_frames),
                    text_temperature=float(request.text_temperature),
                    text_top_p=float(request.text_top_p),
                    text_top_k=int(request.text_top_k),
                    audio_temperature=float(request.temperature),
                    audio_top_p=float(request.top_p),
                    audio_top_k=int(request.top_k),
                    audio_repetition_penalty=float(request.repetition_penalty),
                    n_vq_for_inference=runtime.n_vq,
                    streamer=frame_streamer,
                )
        except BaseException as exc:  # noqa: BLE001
            generation_error.append(exc)
        finally:
            frame_streamer.end()

    pending_frames: list[torch.Tensor] = []
    emitted_segments: list[torch.Tensor] = []
    generated_audio_tokens: list[torch.Tensor] = []
    generated_frames = 0
    emitted_audio_seconds = 0.0
    generation_start = time.perf_counter()
    first_audio_emitted_at: Optional[float] = None
    generated_frames_at_first_audio = 0
    decode_chunks_submitted = 0
    decode_input_queue: queue.Queue[Optional[dict[str, Any]]] = queue.Queue()
    decode_output_queue: queue.Queue[dict[str, Any]] = queue.Queue()
    speech_pad_idx = int(runtime.model.config.audio_pad_code)

    def _codec_worker() -> None:
        try:
            with runtime.codec_execution_lock:
                with StatefulCodecDecoder(runtime.processor.audio_tokenizer, n_vq=runtime.n_vq) as decoder:
                    while True:
                        item = decode_input_queue.get()
                        if item is None:
                            break
                        codes = item["codes"]
                        with (
                            torch.autocast(device_type="cuda", dtype=runtime.dtype)
                            if runtime.codec_device.type == "cuda"
                            and runtime.dtype in {torch.float16, torch.bfloat16}
                            else nullcontext()
                        ):
                            audio = decoder.decode_codes(
                                codes.to(device=decoder.device, dtype=torch.long)
                            )
                        decode_output_queue.put(
                            {
                                "type": "audio",
                                "waveform": audio.detach().cpu(),
                                "generated_frames": int(item["generated_frames"]),
                                "chunk_frames": int(codes.shape[0]),
                            }
                        )
        except BaseException as exc:  # noqa: BLE001
            decode_output_queue.put({"type": "error", "error": repr(exc)})
        finally:
            decode_output_queue.put({"type": "done"})

    def _metrics(*, now: Optional[float] = None) -> dict[str, Any]:
        current_time = time.perf_counter() if now is None else now
        elapsed = max(0.0, current_time - generation_start)
        generated_seconds = generated_frames / float(runtime.frame_rate)
        first_latency = (
            None
            if first_audio_emitted_at is None
            else max(0.0, first_audio_emitted_at - generation_start)
        )
        post_first_factor: Optional[float] = None
        playback_lead: Optional[float] = None
        if first_audio_emitted_at is not None:
            post_elapsed = max(0.0, current_time - first_audio_emitted_at)
            post_generated = max(
                0.0,
                (generated_frames - generated_frames_at_first_audio) / float(runtime.frame_rate),
            )
            post_first_factor = post_generated / post_elapsed if post_elapsed > 0 else 0.0
            playback_lead = emitted_audio_seconds - post_elapsed
        return {
            "generated_audio_seconds": generated_seconds,
            "generation_elapsed_seconds": elapsed,
            "generation_realtime_factor": generated_seconds / elapsed if elapsed > 0 else 0.0,
            "post_first_generation_realtime_factor": post_first_factor,
            "playback_lead_seconds": playback_lead,
            "first_audio_latency_seconds": first_latency,
            "decode_chunks_submitted": decode_chunks_submitted,
            "decode_queue_depth": decode_input_queue.qsize(),
            "pending_decode_frames": len(pending_frames),
        }

    decoder_thread = threading.Thread(
        target=_codec_worker,
        name="moss-tts-legacy-codec-worker",
        daemon=True,
    )
    generation_thread = threading.Thread(
        target=_generation_worker,
        name="moss-tts-legacy-generation-worker",
        daemon=True,
    )
    decoder_thread.start()
    generation_thread.start()

    decoder_done = False

    def _drain_decoder_outputs(*, block: bool = False) -> list[StreamingEvent]:
        nonlocal emitted_audio_seconds, first_audio_emitted_at
        nonlocal generated_frames_at_first_audio, decoder_done
        events: list[StreamingEvent] = []
        while True:
            try:
                item = decode_output_queue.get(
                    block=block and not events,
                    timeout=0.05 if block and not events else 0.0,
                )
            except queue.Empty:
                break
            if item.get("type") == "done":
                decoder_done = True
                continue
            if item.get("type") == "error":
                raise RuntimeError(f"1.7B codec worker failed: {item.get('error')}")
            audio = item["waveform"]
            if audio.numel() <= 0:
                continue
            emitted_segments.append(audio)
            now = time.perf_counter()
            if first_audio_emitted_at is None:
                first_audio_emitted_at = now
                generated_frames_at_first_audio = int(item["generated_frames"])
            emitted_audio_seconds += int(audio.shape[-1]) / float(runtime.sample_rate)
            events.append(
                StreamingEvent(
                    type="audio",
                    data={
                        "waveform": audio,
                        "sample_rate": runtime.sample_rate,
                        "generated_frames": int(item["generated_frames"]),
                        "emitted_audio_seconds": emitted_audio_seconds,
                        "chunk_frames": int(item["chunk_frames"]),
                        "tts_device": str(runtime.tts_device),
                        "codec_device": str(runtime.codec_device),
                        **_metrics(now=now),
                    },
                )
            )
        return events

    try:
        while True:
            for audio_event in _drain_decoder_outputs(block=False):
                yield audio_event
            try:
                row = frame_streamer.queue.get(timeout=0.05)
            except queue.Empty:
                if generation_error:
                    raise RuntimeError(f"1.7B generation failed: {generation_error[0]!r}")
                continue
            if row is None:
                break
            if row.ndim != 2 or int(row.shape[0]) != 1 or int(row.shape[1]) < runtime.n_vq + 1:
                raise RuntimeError(f"unexpected 1.7B streamed row shape: {tuple(row.shape)}")
            frame = row[0, 1 : runtime.n_vq + 1].contiguous().long()
            if bool(frame.eq(speech_pad_idx).all().item()):
                continue
            generated_audio_tokens.append(frame)
            pending_frames.append(frame)
            generated_frames += 1
            decode_budget = _decode_budget_from_stream_state(
                lead_seconds=(generated_frames / float(runtime.frame_rate))
                - (time.perf_counter() - generation_start)
                - emitted_audio_seconds,
                default_budget=int(request.codec_chunk_frames),
                first_decode_submitted=decode_chunks_submitted > 0,
                first_audio_emitted=first_audio_emitted_at is not None,
                decode_queue_depth=decode_input_queue.qsize(),
                decode_chunks_submitted=decode_chunks_submitted,
                n_vq=runtime.n_vq,
            )
            if len(pending_frames) >= decode_budget:
                codes = torch.stack(pending_frames, dim=0).contiguous()
                pending_frames.clear()
                decode_chunks_submitted += 1
                decode_input_queue.put(
                    {"codes": codes, "generated_frames": generated_frames}
                )
            for audio_event in _drain_decoder_outputs(block=False):
                yield audio_event
            yield StreamingEvent(
                type="progress",
                data={
                    "generated_frames": generated_frames,
                    "emitted_audio_seconds": emitted_audio_seconds,
                    "tts_device": str(runtime.tts_device),
                    "codec_device": str(runtime.codec_device),
                    **_metrics(),
                },
            )
        if generation_error:
            raise RuntimeError(f"1.7B generation failed: {generation_error[0]!r}")
        if pending_frames:
            codes = torch.stack(pending_frames, dim=0).contiguous()
            pending_frames.clear()
            decode_chunks_submitted += 1
            decode_input_queue.put({"codes": codes, "generated_frames": generated_frames})
    finally:
        decode_input_queue.put(None)

    while not decoder_done or not decode_output_queue.empty():
        for audio_event in _drain_decoder_outputs(block=True):
            yield audio_event
        if decoder_done:
            break
    generation_thread.join(timeout=5.0)
    decoder_thread.join(timeout=5.0)

    final_audio = (
        torch.cat(emitted_segments, dim=-1)
        if emitted_segments
        else torch.empty((1, 0), dtype=torch.float32)
    )
    final_tokens = (
        torch.stack(generated_audio_tokens, dim=0)
        if generated_audio_tokens
        else torch.empty((0, runtime.n_vq), dtype=torch.long)
    )
    if final_audio.numel() <= 0:
        raise RuntimeError("1.7B 模型没有返回可解码音频")
    audio_path = run_dir / "generated.wav"
    tokens_path = run_dir / "generated_tokens.pt"
    metadata_path = run_dir / "metadata.json"
    _save_waveform(audio_path, final_audio, runtime.sample_rate)
    torch.save(final_tokens, tokens_path)
    final_metrics = _metrics()
    metadata = {
        "run_id": run_id,
        "model_profile": runtime.profile_id,
        "generated_frames": int(final_tokens.shape[0]),
        "duration_seconds": final_audio.shape[-1] / float(runtime.sample_rate),
        "generation_seconds": time.perf_counter() - generation_start,
        "sample_rate": runtime.sample_rate,
        "channels": int(final_audio.shape[0]),
        "streaming_supported": True,
        "first_audio_latency_seconds": final_metrics["first_audio_latency_seconds"],
        "generation_realtime_factor": final_metrics["generation_realtime_factor"],
        "post_first_generation_realtime_factor": final_metrics[
            "post_first_generation_realtime_factor"
        ],
        "decode_chunks_submitted": final_metrics["decode_chunks_submitted"],
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    yield StreamingEvent(
        type="result",
        data={
            "waveform": final_audio,
            "sample_rate": runtime.sample_rate,
            "audio_path": str(audio_path),
            "tokens_path": str(tokens_path),
            "metadata_path": str(metadata_path),
            "metadata": metadata,
            "audio_token_ids": final_tokens,
        },
    )


def synthesize_legacy_stream(
    runtime: LegacyRuntime,
    request: StreamingRequest,
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> Generator[StreamingEvent, None, None]:
    # The official 1.7B checkpoint mutates generation/cache state on the model
    # instance and is not safe for concurrent generate() calls. Keeping one
    # resident instance and serializing complete jobs is both faster and much
    # less memory-intensive than loading a second ~13 GB model+codec runtime.
    with runtime.generation_execution_lock:
        yield from _synthesize_legacy_stream_unlocked(
            runtime,
            request,
            output_dir=output_dir,
        )


def synthesize_for_runtime(
    runtime: StreamingRuntime | LegacyRuntime,
    request: StreamingRequest,
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> Generator[StreamingEvent, None, None]:
    if isinstance(runtime, LegacyRuntime):
        yield from synthesize_legacy_stream(runtime, request, output_dir=output_dir)
    else:
        yield from synthesize_stream(runtime, request, output_dir=output_dir)


def _build_prompt_fields(request: StreamingRequest) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "instruction": None,
        "quality": None,
        "sound_event": None,
        "ambient_sound": None,
        "language": request.language.strip() or None,
    }
    if request.tokens_control:
        tokens = int(request.tokens)
        if tokens <= 0:
            tokens = estimate_tokens(request.text, request.language)
        fields["tokens"] = tokens
    return fields


def _apply_repetition_penalty_with_seen_mask(
    scores: torch.Tensor,
    seen_mask: Optional[torch.Tensor],
    penalty: float,
) -> torch.Tensor:
    if seen_mask is None or float(penalty) == 1.0:
        return scores
    penalized = torch.where(scores < 0, scores * float(penalty), scores / float(penalty))
    return torch.where(seen_mask, penalized, scores)


def _sample_next_token_topk_subspace(
    *,
    model: Any,
    logits: torch.Tensor,
    do_sample: bool,
    temperature: float,
    top_k: Optional[int],
    top_p: Optional[float],
    seen_mask: Optional[torch.Tensor] = None,
    repetition_penalty: float = 1.0,
    generator: Optional[torch.Generator] = None,
) -> torch.LongTensor:
    scores = logits.float()
    scores = _apply_repetition_penalty_with_seen_mask(scores, seen_mask, repetition_penalty)
    if not do_sample:
        return torch.argmax(scores, dim=-1)
    if float(temperature) <= 0:
        raise ValueError("temperature must be positive when do_sample=True.")

    vocab_size = int(scores.shape[-1])
    k = int(top_k) if top_k is not None else 0
    if k > 0 and k < vocab_size:
        top_values, top_indices = torch.topk(scores, k, dim=-1, sorted=True)
        top_values = top_values / float(temperature)
        if top_p is not None and 0.0 < float(top_p) < 1.0:
            top_probs = torch.softmax(top_values, dim=-1)
            cumulative_probs = top_probs.cumsum(dim=-1)
            remove_mask = cumulative_probs > float(top_p)
            remove_mask[..., 1:] = remove_mask[..., :-1].clone()
            remove_mask[..., 0] = False
            top_values = top_values.masked_fill(remove_mask, -torch.inf)
        probs = torch.softmax(top_values, dim=-1)
        sampled_offsets = torch.multinomial(probs, num_samples=1, generator=generator)
        return top_indices.gather(dim=-1, index=sampled_offsets).squeeze(-1)

    return _sample_next_token_with_generator(
        model=model,
        logits=scores,
        do_sample=do_sample,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        previous_token_ids=None,
        repetition_penalty=1.0,
        generator=generator,
    )


def _sample_next_token_with_generator(
    *,
    model: Any,
    logits: torch.Tensor,
    do_sample: bool,
    temperature: float,
    top_k: Optional[int],
    top_p: Optional[float],
    previous_token_ids: Optional[torch.LongTensor] = None,
    repetition_penalty: float = 1.0,
    generator: Optional[torch.Generator] = None,
) -> torch.LongTensor:
    scores = logits.float()
    scores = model._apply_repetition_penalty(scores, previous_token_ids, repetition_penalty)
    if not do_sample:
        return torch.argmax(scores, dim=-1)
    if float(temperature) <= 0:
        raise ValueError("temperature must be positive when do_sample=True.")
    scores = scores / float(temperature)
    scores = model._filter_logits(scores, top_k=top_k, top_p=top_p)
    probs = torch.softmax(scores, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)


def _sample_next_assistant_text_token_with_generator(
    *,
    model: Any,
    local_hidden_states: torch.Tensor,
    do_sample: bool,
    temperature: float,
    top_k: Optional[int],
    top_p: Optional[float],
    generator: Optional[torch.Generator],
) -> torch.LongTensor:
    if model._use_binary_local_text_head() and model.local_text_lm_head is not None:
        logits = model.local_text_lm_head(local_hidden_states)
    else:
        candidate_ids = model._local_text_candidate_ids(local_hidden_states.device)
        logits = model.text_lm_head(local_hidden_states).index_select(dim=-1, index=candidate_ids)
    sampled_indices = _sample_next_token_with_generator(
        model=model,
        logits=logits,
        do_sample=do_sample,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        generator=generator,
    )
    candidate_ids = model._local_text_candidate_ids(logits.device)
    return candidate_ids[sampled_indices]


def build_processor_inputs(
    runtime: StreamingRuntime,
    request: StreamingRequest,
) -> tuple[dict[str, torch.Tensor], Optional[torch.Tensor], str]:
    processor = runtime.processor
    mode = str(request.mode or "continuation").strip().lower()
    if mode not in {"continuation", "voice_clone"}:
        raise ValueError("mode must be continuation or voice_clone.")

    prompt_audio_path = (request.prompt_audio_path or "").strip() or None
    prompt_text = request.prompt_text or ""
    if mode == "voice_clone" and not prompt_audio_path:
        mode = "continuation"
    prompt_fields = _build_prompt_fields(request)
    user_kwargs = dict(
        text=request.text,
        instruction=prompt_fields.get("instruction"),
        tokens=prompt_fields.get("tokens"),
        quality=prompt_fields.get("quality"),
        sound_event=prompt_fields.get("sound_event"),
        ambient_sound=prompt_fields.get("ambient_sound"),
        language=prompt_fields.get("language"),
    )

    prompt_audio_codes: Optional[torch.Tensor] = None
    if mode == "voice_clone":
        reference_codes = cached_reference_audio_codes(runtime, prompt_audio_path)
        conversation = [
            processor.build_user_message(
                reference=[reference_codes],
                **user_kwargs,
            )
        ]
        processor_mode = "generation"
    elif prompt_audio_path:
        prompt_audio_codes = processor.encode_audios_from_path(prompt_audio_path, n_vq=runtime.n_vq)[0]
        continuation_text = prompt_text + request.text if prompt_text.strip() else request.text
        conversation = [
            processor.build_user_message(
                text=continuation_text,
                instruction=prompt_fields.get("instruction"),
                tokens=prompt_fields.get("tokens"),
                quality=prompt_fields.get("quality"),
                sound_event=prompt_fields.get("sound_event"),
                ambient_sound=prompt_fields.get("ambient_sound"),
                language=prompt_fields.get("language"),
            ),
            processor.build_assistant_message(audio_codes_list=[prompt_audio_codes]),
        ]
        processor_mode = "continuation"
    else:
        # No-prompt continuation degenerates to direct TTS generation. This is
        # trained for TACv5 and avoids forcing users to provide a reference.
        conversation = [processor.build_user_message(**user_kwargs)]
        processor_mode = "generation"

    batch = processor(conversation, mode=processor_mode, n_vq=runtime.n_vq)
    return _move_batch_to_device(batch, runtime.tts_device), prompt_audio_codes, processor_mode


def cached_reference_audio_codes(
    runtime: StreamingRuntime | LegacyRuntime,
    prompt_audio_path: str,
) -> torch.Tensor:
    """Encode a clone reference once and reuse its CPU audio codes across requests."""
    resolved = Path(prompt_audio_path).resolve(strict=True)
    stat = resolved.stat()
    key = (str(resolved), int(stat.st_mtime_ns), int(stat.st_size), int(runtime.n_vq))
    with runtime.reference_audio_cache_lock:
        cached = runtime.reference_audio_cache.get(key)
        if cached is not None:
            runtime.reference_audio_cache.move_to_end(key)
            runtime.reference_audio_cache_hits += 1
            return cached

        with runtime.codec_execution_lock:
            encoded = runtime.processor.encode_audios_from_path(str(resolved), n_vq=runtime.n_vq)[0]
        encoded = encoded.contiguous().cpu().long()
        for old_key in [item for item in runtime.reference_audio_cache if item[0] == str(resolved)]:
            runtime.reference_audio_cache.pop(old_key, None)
        runtime.reference_audio_cache[key] = encoded
        while len(runtime.reference_audio_cache) > 32:
            runtime.reference_audio_cache.popitem(last=False)
        runtime.reference_audio_cache_misses += 1
        return encoded


@torch.inference_mode()
def iter_generate_frames(
    model: Any,
    *,
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.Tensor],
    max_new_frames: int,
    do_sample: bool,
    text_temperature: float,
    text_top_p: float,
    text_top_k: int,
    audio_temperature: float,
    audio_top_p: float,
    audio_top_k: int,
    audio_repetition_penalty: float,
    use_kv_cache: bool = True,
    sampling_generator: Optional[torch.Generator] = None,
) -> Generator[StreamingEvent, None, None]:
    n_vq = int(model.config.n_vq)
    max_new_frames = int(max_new_frames)
    model._resolve_fixed_nq(n_vq_for_inference=n_vq)

    if input_ids.ndim == 2:
        input_ids = input_ids.unsqueeze(0)
    if input_ids.ndim != 3:
        raise ValueError(f"Expected input_ids with 3 dims, got {tuple(input_ids.shape)}.")
    if input_ids.shape[-1] != n_vq + 1:
        raise ValueError(
            f"Expected {n_vq + 1} channels from config.n_vq, got {input_ids.shape[-1]}."
        )
    if attention_mask is None:
        attention_mask = torch.ones(input_ids.shape[:2], dtype=torch.bool, device=input_ids.device)
    elif attention_mask.ndim == 1:
        attention_mask = attention_mask.unsqueeze(0)
    attention_mask = attention_mask.to(device=input_ids.device, dtype=torch.bool)

    batch_size = int(input_ids.shape[0])
    if batch_size != 1:
        raise ValueError("Streaming helper currently supports batch_size=1.")

    current_input_ids = input_ids
    current_attention_mask = attention_mask
    current_model_input_ids = current_input_ids
    generated_audio_history = torch.empty(
        (batch_size, max(0, max_new_frames), n_vq),
        dtype=torch.long,
        device=input_ids.device,
    )
    use_fast_audio_sampling = (
        n_vq >= 32
        and bool(do_sample)
        and audio_top_k is not None
        and int(audio_top_k) > 0
    )
    seen_audio_token_masks: Optional[list[torch.Tensor]] = None
    if use_fast_audio_sampling and float(audio_repetition_penalty) != 1.0:
        seen_audio_token_masks = [
            torch.zeros(
                (batch_size, int(model.config.audio_codebook_sizes[channel_index])),
                dtype=torch.bool,
                device=input_ids.device,
            )
            for channel_index in range(n_vq)
        ]
    generated_frame_count = 0
    finished = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
    past_key_values = None
    local_dtype = model.local_transformer.ln_f.weight.dtype

    for step in range(max_new_frames):
        global_inputs_embeds = model._build_inputs_embeds(current_model_input_ids)
        global_outputs = model.transformer(
            input_ids=None,
            past_key_values=past_key_values,
            attention_mask=current_attention_mask,
            position_ids=None,
            inputs_embeds=global_inputs_embeds,
            use_cache=use_kv_cache,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
            cu_seqlens=None,
            num_sequences=None,
        )
        global_hidden_states = global_outputs.last_hidden_state[:, -1, :]
        local_global_hidden_states = model._global_hidden_to_local(global_hidden_states).to(dtype=local_dtype)

        local_prefix_hidden_states, local_prefix_past_key_values = model._decode_local_hidden_states_with_cache(
            local_global_hidden_states.unsqueeze(1)
        )
        local_hidden_states = local_prefix_hidden_states[:, -1, :]
        next_text_tokens = _sample_next_assistant_text_token_with_generator(
            model=model,
            local_hidden_states=local_hidden_states,
            do_sample=do_sample,
            temperature=text_temperature,
            top_k=text_top_k,
            top_p=text_top_p,
            generator=sampling_generator,
        )
        should_continue = next_text_tokens.eq(int(model.config.audio_assistant_slot_token_id)) & ~finished
        finished = finished | next_text_tokens.eq(int(model.config.audio_end_token_id))
        if not bool(should_continue.any().item()):
            break

        next_frame_tokens = []
        for channel_index in range(n_vq):
            channel_logits = model.audio_lm_heads[channel_index](local_hidden_states)
            if use_fast_audio_sampling:
                channel_token = _sample_next_token_topk_subspace(
                    model=model,
                    logits=channel_logits,
                    do_sample=do_sample,
                    temperature=audio_temperature,
                    top_k=audio_top_k,
                    top_p=audio_top_p,
                    seen_mask=(
                        None
                        if seen_audio_token_masks is None
                        else seen_audio_token_masks[channel_index]
                    ),
                    repetition_penalty=audio_repetition_penalty,
                    generator=sampling_generator,
                )
            else:
                channel_token = _sample_next_token_with_generator(
                    model=model,
                    logits=channel_logits,
                    do_sample=do_sample,
                    temperature=audio_temperature,
                    top_k=audio_top_k,
                    top_p=audio_top_p,
                    previous_token_ids=(
                        None
                        if generated_frame_count <= 0
                        else generated_audio_history[:, :generated_frame_count, channel_index]
                    ),
                    repetition_penalty=audio_repetition_penalty,
                    generator=sampling_generator,
                )
            next_frame_tokens.append(channel_token)
            if seen_audio_token_masks is not None:
                seen_audio_token_masks[channel_index].scatter_(1, channel_token.unsqueeze(-1), True)
            if channel_index + 1 < n_vq:
                current_local_input = model.audio_embeddings[channel_index](channel_token).to(dtype=local_dtype)
                local_token_hidden_states, local_prefix_past_key_values = model._decode_local_hidden_states_with_cache(
                    current_local_input.unsqueeze(1),
                    past_key_values=local_prefix_past_key_values,
                )
                local_hidden_states = local_token_hidden_states[:, -1, :]

        next_frame = torch.stack(next_frame_tokens, dim=-1)
        next_frame = next_frame.masked_fill(~should_continue.unsqueeze(-1), int(model.config.audio_pad_token_id))
        generated_audio_history[:, generated_frame_count, :] = next_frame
        generated_frame_count += 1
        yield StreamingEvent(
            type="frame",
            data={
                "frame_index": step,
                "audio_token_ids": next_frame.detach().clone(),
                "finished": bool(finished.item()),
            },
        )

        next_row = model._build_generation_row(
            batch_size=batch_size,
            device=input_ids.device,
            audio_token_ids=next_frame,
        )
        if bool((~should_continue).any().item()):
            next_row[~should_continue, 0, 0] = int(model.config.pad_token_id)
            next_row[~should_continue, 0, 1:] = int(model.config.audio_pad_token_id)

        current_input_ids = torch.cat([current_input_ids, next_row], dim=1)
        current_attention_mask = torch.cat([current_attention_mask, should_continue.unsqueeze(1)], dim=1)
        if use_kv_cache:
            current_model_input_ids = next_row
            past_key_values = global_outputs.past_key_values
        else:
            current_model_input_ids = current_input_ids

    if generated_frame_count > 0:
        audio_token_ids = generated_audio_history[0, :generated_frame_count, :].detach().clone()
    else:
        audio_token_ids = torch.empty(
            (0, n_vq),
            dtype=torch.long,
            device=input_ids.device,
        )
    yield StreamingEvent(type="final_tokens", data={"audio_token_ids": audio_token_ids})


class StatefulCodecDecoder:
    def __init__(self, audio_tokenizer: Any, *, n_vq: int) -> None:
        self.audio_tokenizer = audio_tokenizer
        self.n_vq = int(n_vq)
        self._ctx = None

    @property
    def device(self) -> torch.device:
        try:
            return next(self.audio_tokenizer.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def __enter__(self) -> "StatefulCodecDecoder":
        self._ctx = self.audio_tokenizer.streaming(batch_size=1)
        self._ctx.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._ctx is not None:
            self._ctx.__exit__(exc_type, exc, tb)
            self._ctx = None

    @torch.inference_mode()
    def decode_codes(self, codes: torch.LongTensor) -> torch.Tensor:
        if codes.numel() == 0:
            return torch.empty((2, 0), dtype=torch.float32, device=self.device)
        if codes.ndim != 2 or int(codes.shape[1]) != self.n_vq:
            raise ValueError(f"Expected codes shape [T, {self.n_vq}], got {tuple(codes.shape)}.")
        codes_qbt = codes.transpose(0, 1).contiguous().unsqueeze(1).to(device=self.device, dtype=torch.long)
        codes_lengths = torch.tensor([codes_qbt.shape[-1]], device=self.device, dtype=torch.long)
        set_streaming_exec_mask = getattr(
            self.audio_tokenizer,
            "_set_streaming_exec_mask",
            None,
        )
        if callable(set_streaming_exec_mask):
            active_mask = torch.tensor(
                [codes_qbt.shape[-1] > 0],
                device=self.device,
                dtype=torch.bool,
            )
            set_streaming_exec_mask(active_mask)
        decoded = self.audio_tokenizer._decode_frame(codes_qbt, codes_lengths)
        if decoded.audio is None or decoded.audio_lengths is None:
            raise RuntimeError("audio tokenizer did not return audio/audio_lengths.")
        audio_length = int(decoded.audio_lengths[0].item())
        if audio_length <= 0:
            return torch.empty(
                (int(getattr(self.audio_tokenizer, "number_channels", 2)), 0),
                dtype=torch.float32,
                device=self.device,
            )
        return decoded.audio[0, :, :audio_length].detach().to(torch.float32)


@torch.inference_mode()
def warmup_streaming_runtime(runtime: StreamingRuntime, *, codec_frames: int = 4) -> None:
    """Pre-run small TTS and codec steps so the first user stream is hot."""
    try:
        request = StreamingRequest(
            text="这是一个流式预热。", language="Chinese", max_new_frames=2, do_sample=False
        )
        batch, _, _ = build_processor_inputs(runtime, request)
        for event in iter_generate_frames(
            runtime.model,
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            max_new_frames=2,
            do_sample=False,
            text_temperature=1.0,
            text_top_p=1.0,
            text_top_k=1,
            audio_temperature=1.0,
            audio_top_p=1.0,
            audio_top_k=1,
            audio_repetition_penalty=1.0,
            use_kv_cache=True,
        ):
            if event.type == "frame":
                break
        if runtime.tts_device.type == "cuda":
            torch.cuda.synchronize(runtime.tts_device)
    except Exception as exc:  # noqa: BLE001
        print(f"[moss_tts_local_v1.5] TTS warmup skipped: {exc}")

    try:
        frames = max(1, int(codec_frames))
        dummy_codes = torch.zeros((frames, runtime.n_vq), dtype=torch.long)
        with runtime.codec_execution_lock:
            with StatefulCodecDecoder(runtime.processor.audio_tokenizer, n_vq=runtime.n_vq) as decoder:
                _ = decoder.decode_codes(dummy_codes.to(device=decoder.device, dtype=torch.long))
        if runtime.codec_device.type == "cuda":
            torch.cuda.synchronize(runtime.codec_device)
    except Exception as exc:  # noqa: BLE001
        print(f"[moss_tts_local_v1.5] codec warmup skipped: {exc}")


def _decode_budget_from_stream_state(
    *,
    lead_seconds: float,
    default_budget: int = 0,
    first_decode_submitted: bool = False,
    first_audio_emitted: bool = False,
    decode_queue_depth: int = 0,
    decode_chunks_submitted: int = 0,
    n_vq: int = 12,
) -> int:
    if int(default_budget) > 0:
        return max(1, int(default_budget))

    if int(n_vq) >= 32:
        # RVQ32 TTS generation has 32 local autoregressive steps per frame.
        # Decoding 1-2 frames at a time creates too many codec calls and can
        # starve the generator thread. Keep automatic streaming chunks larger
        # for RVQ32 only; lower-depth models keep the lower-latency ladder below.
        if not first_decode_submitted:
            return 4
        if not first_audio_emitted:
            return 4
        if lead_seconds < 0.64:
            return 4
        if lead_seconds < 1.0:
            return 6
        return 8

    # Lower-depth models can decode sooner than RVQ32, but a one-frame
    # first packet is only 80 ms at 12.5 fps and tends to underrun browser
    # playback. Start with four frames so the first PCM packet carries about
    # 320 ms of audio; after audio is flowing, switch back to the lead ladder.
    if not first_decode_submitted:
        return 4

    budget = _smooth_decode_budget_from_lead(lead_seconds)

    if not first_audio_emitted:
        return 4

    # Once audio is flowing, use lead as a smooth latency/throughput knob:
    # low or negative lead decodes short chunks to avoid audible gaps; healthy
    # lead decodes longer chunks to reduce codec call overhead.
    if decode_queue_depth > 0:
        return min(8, max(4, budget))
    return budget


def _smooth_decode_budget_from_lead(lead_seconds: float) -> int:
    if lead_seconds <= 0.0:
        return 1
    if lead_seconds < 0.16:
        return 2
    if lead_seconds < 0.32:
        return 3
    if lead_seconds < 0.64:
        return 4
    if lead_seconds < 1.0:
        return 6
    return 8


def _decode_budget_from_lead(lead_seconds: float, default_budget: int = 0) -> int:
    return _decode_budget_from_stream_state(
        lead_seconds=lead_seconds,
        default_budget=default_budget,
        first_decode_submitted=True,
        first_audio_emitted=True,
    )


def _save_waveform(path: Path, waveform: torch.Tensor, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), waveform.detach().cpu().to(torch.float32), sample_rate)


def synthesize_stream(
    runtime: StreamingRuntime,
    request: StreamingRequest,
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> Generator[StreamingEvent, None, None]:
    sampling_generator: Optional[torch.Generator] = None
    if request.seed is not None and int(request.seed) >= 0:
        sampling_generator = torch.Generator(device=runtime.tts_device)
        sampling_generator.manual_seed(int(request.seed))

    batch, prompt_audio_codes, processor_mode = build_processor_inputs(runtime, request)
    output_dir = Path(output_dir)
    run_id = f"{int(time.time() * 1000)}_{abs(hash((request.text, request.mode))) % 1000000:06d}"
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    yield StreamingEvent(
        type="metadata",
        data={
            "run_id": run_id,
            "run_dir": str(run_dir),
            "sample_rate": runtime.sample_rate,
            "frame_rate": runtime.frame_rate,
            "n_vq": runtime.n_vq,
            "tts_device": str(runtime.tts_device),
            "codec_device": str(runtime.codec_device),
            "attn_implementation": runtime.attn_implementation,
            "processor_mode": processor_mode,
            "tokens": _build_prompt_fields(request).get("tokens"),
        },
    )

    pending_frames: list[torch.LongTensor] = []
    emitted_segments: list[torch.Tensor] = []
    generated_frames = 0
    emitted_audio_seconds = 0.0
    generation_start = time.perf_counter()
    first_decode_submitted = False
    decode_chunks_submitted = 0
    first_audio_emitted_at: Optional[float] = None
    generated_frames_at_first_audio = 0
    final_audio_token_ids: Optional[torch.Tensor] = None
    decode_input_queue: queue.Queue[Optional[dict[str, Any]]] = queue.Queue()
    decode_output_queue: queue.Queue[dict[str, Any]] = queue.Queue()

    def _codec_worker() -> None:
        try:
            with runtime.codec_execution_lock:
                with StatefulCodecDecoder(runtime.processor.audio_tokenizer, n_vq=runtime.n_vq) as decoder:
                    if prompt_audio_codes is not None and prompt_audio_codes.numel() > 0:
                        _ = decoder.decode_codes(prompt_audio_codes.to(device=decoder.device, dtype=torch.long))
                    while True:
                        item = decode_input_queue.get()
                        if item is None:
                            break
                        codes = item["codes"]
                        audio = decoder.decode_codes(codes.to(device=decoder.device, dtype=torch.long))
                        decode_output_queue.put(
                            {
                                "type": "audio",
                                "waveform": audio.detach().cpu(),
                                "generated_frames": int(item["generated_frames"]),
                                "lead_seconds": float(item.get("lead_seconds", 0.0)),
                                "chunk_frames": int(codes.shape[0]),
                            }
                        )
        except BaseException as exc:  # noqa: BLE001
            decode_output_queue.put({"type": "error", "error": repr(exc)})
        finally:
            decode_output_queue.put({"type": "done"})

    decoder_thread = threading.Thread(target=_codec_worker, name="moss-tts-local-v1.5-codec-worker", daemon=True)
    decoder_thread.start()
    decoder_done = False

    def _generation_lead_seconds(*, now: Optional[float] = None) -> float:
        current_time = time.perf_counter() if now is None else now
        if first_audio_emitted_at is not None:
            post_first_elapsed_seconds = max(0.0, current_time - first_audio_emitted_at)
            post_first_generated_audio_seconds = max(
                0.0,
                (generated_frames - generated_frames_at_first_audio) / float(runtime.frame_rate),
            )
            return post_first_generated_audio_seconds - post_first_elapsed_seconds
        elapsed_seconds = max(0.0, current_time - generation_start)
        generated_audio_seconds = generated_frames / float(runtime.frame_rate)
        return generated_audio_seconds - emitted_audio_seconds - elapsed_seconds

    def _stream_metrics(*, now: Optional[float] = None) -> dict[str, Any]:
        current_time = time.perf_counter() if now is None else now
        elapsed_seconds = max(0.0, current_time - generation_start)
        generated_audio_seconds = generated_frames / float(runtime.frame_rate)
        generation_realtime_factor = (
            generated_audio_seconds / elapsed_seconds if elapsed_seconds > 0.0 else 0.0
        )
        post_first_generation_realtime_factor: Optional[float] = None
        post_first_generated_audio_seconds: Optional[float] = None
        playback_lead_seconds: Optional[float] = None
        first_audio_latency_seconds: Optional[float] = None
        if first_audio_emitted_at is not None:
            first_audio_latency_seconds = max(0.0, first_audio_emitted_at - generation_start)
            post_first_elapsed_seconds = max(0.0, current_time - first_audio_emitted_at)
            post_first_generated_audio_seconds = max(
                0.0,
                (generated_frames - generated_frames_at_first_audio) / float(runtime.frame_rate),
            )
            post_first_generation_realtime_factor = (
                post_first_generated_audio_seconds / post_first_elapsed_seconds
                if post_first_elapsed_seconds > 0.0
                else 0.0
            )
            playback_lead_seconds = emitted_audio_seconds - post_first_elapsed_seconds
        return {
            "generated_audio_seconds": generated_audio_seconds,
            "generation_lead_seconds": _generation_lead_seconds(now=current_time),
            "generation_elapsed_seconds": elapsed_seconds,
            "generation_realtime_factor": generation_realtime_factor,
            "post_first_generated_audio_seconds": post_first_generated_audio_seconds,
            "post_first_generation_realtime_factor": post_first_generation_realtime_factor,
            "playback_lead_seconds": playback_lead_seconds,
            "first_audio_latency_seconds": first_audio_latency_seconds,
            "decode_chunks_submitted": decode_chunks_submitted,
            "decode_queue_depth": decode_input_queue.qsize(),
            "pending_decode_frames": len(pending_frames),
        }

    def _drain_decoder_outputs(*, block: bool = False) -> list[StreamingEvent]:
        nonlocal emitted_audio_seconds, decoder_done, first_audio_emitted_at, generated_frames_at_first_audio
        drained: list[StreamingEvent] = []
        while True:
            try:
                item = decode_output_queue.get(block=block and not drained, timeout=0.05 if block and not drained else 0.0)
            except queue.Empty:
                break
            if item.get("type") == "done":
                decoder_done = True
                continue
            if item.get("type") == "error":
                raise RuntimeError(f"codec worker failed: {item.get('error')}")
            audio = item["waveform"]
            emitted_segments.append(audio)
            now = time.perf_counter()
            if first_audio_emitted_at is None:
                first_audio_emitted_at = now
                generated_frames_at_first_audio = int(item.get("generated_frames", generated_frames))
            emitted_audio_seconds += int(audio.shape[-1]) / float(runtime.sample_rate)
            metrics = _stream_metrics(now=now)
            drained.append(
                StreamingEvent(
                    type="audio",
                    data={
                        "waveform": audio,
                        "sample_rate": runtime.sample_rate,
                        "generated_frames": int(item.get("generated_frames", generated_frames)),
                        "emitted_audio_seconds": emitted_audio_seconds,
                        "lead_seconds": float(item.get("lead_seconds", 0.0)),
                        "chunk_frames": int(item.get("chunk_frames", 0)),
                        "tts_device": str(runtime.tts_device),
                        "codec_device": str(runtime.codec_device),
                        **metrics,
                    },
                )
            )
        return drained

    try:
        for event in iter_generate_frames(
            runtime.model,
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            max_new_frames=int(request.max_new_frames),
            do_sample=bool(request.do_sample),
            text_temperature=float(request.text_temperature),
            text_top_p=float(request.text_top_p),
            text_top_k=int(request.text_top_k),
            audio_temperature=float(request.temperature),
            audio_top_p=float(request.top_p),
            audio_top_k=int(request.top_k),
            audio_repetition_penalty=float(request.repetition_penalty),
            use_kv_cache=True,
            sampling_generator=sampling_generator,
        ):
            for audio_event in _drain_decoder_outputs(block=False):
                yield audio_event
            if event.type == "frame":
                frame = event.data["audio_token_ids"][0].detach().cpu().clone()
                pending_frames.append(frame)
                generated_frames += 1
                now = time.perf_counter()
                lead_seconds = _generation_lead_seconds(now=now)
                decode_budget = _decode_budget_from_stream_state(
                    lead_seconds=lead_seconds,
                    default_budget=int(request.codec_chunk_frames),
                    first_decode_submitted=first_decode_submitted,
                    first_audio_emitted=first_audio_emitted_at is not None,
                    decode_queue_depth=decode_input_queue.qsize(),
                    decode_chunks_submitted=decode_chunks_submitted,
                    n_vq=runtime.n_vq,
                )
                if len(pending_frames) >= decode_budget:
                    codes = torch.stack(pending_frames, dim=0).contiguous()
                    pending_frames.clear()
                    first_decode_submitted = True
                    decode_chunks_submitted += 1
                    decode_input_queue.put(
                        {
                            "codes": codes,
                            "generated_frames": generated_frames,
                            "lead_seconds": lead_seconds,
                        }
                    )
                for audio_event in _drain_decoder_outputs(block=False):
                    yield audio_event
                yield StreamingEvent(
                    type="progress",
                    data={
                        "generated_frames": generated_frames,
                        "emitted_audio_seconds": emitted_audio_seconds,
                        "tts_device": str(runtime.tts_device),
                        "codec_device": str(runtime.codec_device),
                        **_stream_metrics(),
                    },
                )
            elif event.type == "final_tokens":
                final_audio_token_ids = event.data["audio_token_ids"].detach().cpu()

        if pending_frames:
            codes = torch.stack(pending_frames, dim=0).contiguous()
            pending_frames.clear()
            first_decode_submitted = True
            decode_chunks_submitted += 1
            decode_input_queue.put(
                {
                    "codes": codes,
                    "generated_frames": generated_frames,
                    "lead_seconds": 0.0,
                }
            )
    finally:
        decode_input_queue.put(None)

    while not decoder_done or not decode_output_queue.empty():
        for audio_event in _drain_decoder_outputs(block=True):
            yield audio_event
        if decoder_done:
            break
    decoder_thread.join(timeout=5.0)

    final_audio = (
        torch.cat(emitted_segments, dim=-1)
        if emitted_segments
        else torch.empty((2, 0), dtype=torch.float32)
    )
    audio_path = run_dir / "generated.wav"
    tokens_path = run_dir / "audio_tokens.pt"
    meta_path = run_dir / "metadata.json"
    _save_waveform(audio_path, final_audio, runtime.sample_rate)
    if final_audio_token_ids is None:
        final_audio_token_ids = torch.empty((0, runtime.n_vq), dtype=torch.long)
    torch.save(final_audio_token_ids, tokens_path)
    final_metrics = _stream_metrics()
    metadata = {
        "run_id": run_id,
        "mode": request.mode,
        "processor_mode": processor_mode,
        "text": request.text,
        "prompt_text": request.prompt_text,
        "prompt_audio_path": request.prompt_audio_path,
        "language": request.language,
        "tokens_control": request.tokens_control,
        "tokens": _build_prompt_fields(request).get("tokens"),
        "max_new_frames": request.max_new_frames,
        "generated_frames": int(final_audio_token_ids.shape[0]),
        "sample_rate": runtime.sample_rate,
        "duration_seconds": final_audio.shape[-1] / float(runtime.sample_rate),
        "first_audio_latency_seconds": final_metrics["first_audio_latency_seconds"],
        "generation_realtime_factor": final_metrics["generation_realtime_factor"],
        "post_first_generation_realtime_factor": final_metrics["post_first_generation_realtime_factor"],
        "generation_lead_seconds": final_metrics["generation_lead_seconds"],
        "decode_chunks_submitted": final_metrics["decode_chunks_submitted"],
        "audio_path": str(audio_path),
        "tokens_path": str(tokens_path),
    }
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    yield StreamingEvent(
        type="result",
        data={
            "waveform": final_audio,
            "sample_rate": runtime.sample_rate,
            "audio_path": str(audio_path),
            "tokens_path": str(tokens_path),
            "metadata_path": str(meta_path),
            "metadata": metadata,
            "audio_token_ids": final_audio_token_ids,
        },
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MOSS-TTS Local Transformer v1.5 streaming inference runner.")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--codec-dir", default=str(DEFAULT_CODEC_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--text", default="这是一个流式推理测试。")
    parser.add_argument("--mode", default="continuation", choices=["continuation", "voice_clone"])
    parser.add_argument("--prompt-text", default="")
    parser.add_argument("--prompt-audio-path", default="")
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--tokens-control", action="store_true")
    parser.add_argument("--tokens", type=int, default=0)
    parser.add_argument("--max-new-frames", type=int, default=8)
    parser.add_argument("--device", default="cuda", help="Legacy default device used when --tts-device is omitted.")
    parser.add_argument("--tts-device", default="", help="Device for the TTS model, e.g. cuda:0.")
    parser.add_argument("--codec-device", default="", help="Device for the codec, e.g. cuda:1. Defaults to TTS device.")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--attn-implementation",
        default="flash_attention_2",
        choices=["flash_attention_2", "sdpa", "eager"],
    )
    parser.add_argument(
        "--codec-weight-dtype",
        default="fp32",
        choices=["bf16", "bfloat16", "fp32", "float32"],
        help="Codec encoder/decoder parameter dtype. Defaults to fp32; pass bf16 to reduce memory. The quantizer stays fp32.",
    )
    parser.add_argument(
        "--codec-compute-dtype",
        default="bf16",
        choices=["bf16", "fp32"],
        help="Codec non-quantizer autocast compute dtype.",
    )
    parser.add_argument("--temperature", type=float, default=1.2)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--text-temperature", type=float, default=1.0)
    parser.add_argument("--text-top-p", type=float, default=1.0)
    parser.add_argument("--text-top-k", type=int, default=50)
    parser.add_argument(
        "--codec-chunk-frames",
        type=int,
        default=8,
        help="Codec decode chunk size. 0 uses adaptive streaming scheduling.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    runtime = load_runtime(
        model_dir=args.model_dir,
        codec_dir=args.codec_dir,
        device=args.device,
        tts_device=args.tts_device or args.device,
        codec_device=args.codec_device or args.tts_device or args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        codec_weight_dtype=args.codec_weight_dtype,
        codec_compute_dtype=args.codec_compute_dtype,
    )
    request = StreamingRequest(
        text=args.text,
        mode=args.mode,
        prompt_text=args.prompt_text,
        prompt_audio_path=args.prompt_audio_path or None,
        language=args.language,
        tokens_control=bool(args.tokens_control),
        tokens=int(args.tokens),
        max_new_frames=int(args.max_new_frames),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        top_k=int(args.top_k),
        repetition_penalty=float(args.repetition_penalty),
        text_temperature=float(args.text_temperature),
        text_top_p=float(args.text_top_p),
        text_top_k=int(args.text_top_k),
        codec_chunk_frames=int(args.codec_chunk_frames),
        seed=int(args.seed),
    )
    last_result = None
    for event in synthesize_stream(runtime, request, output_dir=args.output_dir):
        if event.type == "metadata":
            print(json.dumps(event.data, ensure_ascii=False))
        elif event.type == "progress":
            print(json.dumps(event.data, ensure_ascii=False))
        elif event.type == "audio":
            print(
                json.dumps(
                    {
                        "audio_chunk_samples": int(event.data["waveform"].shape[-1]),
                        "generated_frames": event.data["generated_frames"],
                        "chunk_frames": event.data.get("chunk_frames"),
                        "first_audio_latency_seconds": event.data.get("first_audio_latency_seconds"),
                        "generation_realtime_factor": event.data.get("generation_realtime_factor"),
                        "post_first_generation_realtime_factor": event.data.get(
                            "post_first_generation_realtime_factor"
                        ),
                    },
                    ensure_ascii=False,
                )
            )
        elif event.type == "result":
            last_result = event.data
    if last_result is not None:
        print(json.dumps(last_result["metadata"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
