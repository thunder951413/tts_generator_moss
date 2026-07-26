# coding=utf-8
"""Small, backend-neutral request/event types used by the Qwen-only app."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class StreamingRequest:
    text: str
    mode: str = "clone"
    prompt_text: str = ""
    prompt_audio_path: Optional[str] = None
    language: str = "Chinese"
    tokens_control: bool = False
    tokens: int = 0
    max_new_frames: int = 2048
    do_sample: bool = True
    temperature: float = 0.9
    top_p: float = 1.0
    top_k: int = 50
    repetition_penalty: float = 1.05
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
