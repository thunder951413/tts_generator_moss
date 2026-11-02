# coding=utf-8
"""Isolated HTTP worker for Faster Qwen3-TTS.

This file is executed with the Faster Qwen3-TTS virtual environment so its
native GGML/Metal dependencies stay isolated from the web service environment.
Only loopback HTTP is exposed and audio is returned as NDJSON PCM chunks.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import random
import select
import socket
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import soundfile as sf
import torch
from faster_qwen3_tts import FasterQwen3TTS
from text_processing import normalize_language
from worker_protocol import (
    MAX_TEXT_CHARS, MAX_WORKER_REQUEST_BYTES, WORKER_TOKEN_HEADER, allowed_path, secrets_equal,
)


LOG = logging.getLogger("qwen-worker")

# Runaway-generation watchdog.  When the talker collapses into a repetition
# loop it never emits EOS and keeps producing audio until max_new_tokens
# (2048 frames = 163.84 s).  Normal speech never exceeds ~0.38 s per
# character (observed P95 is 0.225), while recorded runaways were
# 0.97-2.28 s/char.  The budget is 4 s + 0.5 s/char with a 6 s floor: short
# texts no longer get a 30 s allowance, long texts keep ample headroom.
QWEN_CODEC_FRAMES_PER_SECOND = 12.5
RUNAWAY_BASE_SECONDS = 4.0
RUNAWAY_SECONDS_PER_CHAR = 0.5
RUNAWAY_MIN_BUDGET_SECONDS = 6.0
RUNAWAY_MAX_RETRIES = 1
RUNAWAY_RETRY_REPETITION_PENALTY = 1.15

# Early-end guard (complete-block generation only): a text longer than this
# many characters whose audio is shorter than EARLY_END_SECONDS_PER_CHAR per
# character most likely hit a premature EOS.  English packs ~3x more
# characters per second than CJK, so it uses a lower threshold.
EARLY_END_MIN_CHARS = 8
EARLY_END_SECONDS_PER_CHAR = 0.07
EARLY_END_SECONDS_PER_CHAR_ENGLISH = 0.025
EARLY_END_MAX_RETRIES = 1

WORKER_LANGUAGES = {"Chinese", "English", "Japanese", "Korean", "auto"}


class ClientDisconnected(ConnectionResetError):
    """The service closed the request; stop generating at the next chunk."""


def runaway_budget_seconds(text_chars: int, max_new_tokens: int | None = None) -> float:
    """Audio length after which a generation counts as a runaway loop.

    A caller that sends an explicit frame budget below the 2048 default (the
    document pipeline does, and doubles it on a truncation retry) has already
    bounded the generation, so the guard never fires below that budget; the
    1 s margin covers codec flush and appended silence.
    """
    budget = max(
        RUNAWAY_MIN_BUDGET_SECONDS,
        RUNAWAY_BASE_SECONDS + RUNAWAY_SECONDS_PER_CHAR * max(1, int(text_chars)),
    )
    if max_new_tokens is not None and 0 < int(max_new_tokens) < 2048:
        budget = max(budget, int(max_new_tokens) / QWEN_CODEC_FRAMES_PER_SECOND + 1.0)
    return budget


def resolve_language(value: Any, text: str) -> str:
    language = normalize_language(str(value or ""), text)
    return language if language in WORKER_LANGUAGES else "auto"


def early_end_detected(text: str, audio_seconds: float, language: str) -> bool:
    chars = len(text)
    if chars <= EARLY_END_MIN_CHARS:
        return False
    per_char = EARLY_END_SECONDS_PER_CHAR_ENGLISH if language == "English" else EARLY_END_SECONDS_PER_CHAR
    return audio_seconds < per_char * chars


def _retry_seed(previous: int | None) -> int:
    seed = random.randrange(1, 1_000_000)
    if previous is not None and seed == previous:
        seed = (seed + 1) % 1_000_000
    return seed


class WorkerState:
    def __init__(
        self,
        *,
        model_path: str,
        profile_id: str,
        max_seq_len: int,
        backend: str,
        quant: str,
        output_roots: tuple[Path, ...],
        reference_roots: tuple[Path, ...],
    ) -> None:
        self.profile_id = profile_id
        self.model_path = str(model_path)
        self.backend = str(backend)
        self.quant = str(quant)
        self.output_roots = output_roots
        self.reference_roots = reference_roots
        self.started_at = time.time()
        self.generation_lock = threading.Lock()
        self.requests = 0
        self.reference_keys: set[tuple[str, str, bool]] = set()
        LOG.info("loading Faster Qwen3-TTS model %s", self.model_path)
        load_kwargs: dict[str, Any] = {
            "backend": self.backend,
            "quant": self.quant,
            "max_seq_len": max_seq_len,
        }
        if self.backend == "torch":
            load_kwargs.update(
                device="cuda:0",
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
            )
        library_path = os.environ.get("QWENTTS_CPP_LIBRARY", "").strip()
        if library_path:
            load_kwargs["qwentts_library_path"] = library_path
        self.model = FasterQwen3TTS.from_pretrained(self.model_path, **load_kwargs)
        self.model.warmup(prefill_len=100)
        self.sample_rate = int(getattr(self.model, "sample_rate", 24000))
        LOG.info("worker ready: profile=%s sample_rate=%s", self.profile_id, self.sample_rate)

    def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "profile_id": self.profile_id,
            "model_path": self.model_path,
            "backend": self.backend,
            "quant": self.quant,
            "sample_rate": self.sample_rate,
            "requests": self.requests,
            # True while a generation holds the model.  The service polls this
            # after cancelling a stream to reuse the lane instead of killing it.
            "busy": bool(getattr(self, "generation_lock", None) is not None and self.generation_lock.locked()),
            "reference_cache_entries": len(self.reference_keys),
            "pid": os.getpid(),
            "uptime_seconds": time.time() - self.started_at,
        }

    def generate(self, payload: dict[str, Any], client_gone=None) -> Iterator[dict[str, Any]]:
        text = str(payload.get("text") or "").strip()
        if not text:
            raise ValueError("text must not be empty")
        if len(text) > MAX_TEXT_CHARS:
            raise ValueError(f"text exceeds {MAX_TEXT_CHARS} characters; use a document project")
        if "run_id" in payload:
            raise ValueError("run_id is generated by the worker")
        ref_audio = allowed_path(str(payload.get("ref_audio") or ""), self.reference_roots, must_exist=True)
        if not ref_audio.is_file():
            raise ValueError("reference audio is not a file")

        output_root = allowed_path(str(payload.get("output_dir") or ""), self.output_roots, must_exist=False)
        output_root.mkdir(parents=True, exist_ok=True)
        run_id = f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"
        run_dir = output_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        seed = payload.get("seed")
        if seed is not None and int(seed) >= 0:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))

        ref_text = str(payload.get("ref_text") or "").strip()
        xvec_only = bool(payload.get("xvec_only", True))
        if not xvec_only and not ref_text:
            raise ValueError("Qwen ICL clone mode requires an accurate reference transcript")
        cache_key = (str(ref_audio), ref_text, xvec_only)
        cache_hit = cache_key in self.reference_keys

        chunk_size = max(1, min(24, int(payload.get("chunk_size") or 8)))
        max_new_tokens = max(2, min(2048, int(payload.get("max_new_tokens") or 2048)))
        budget_seconds = runaway_budget_seconds(len(text), max_new_tokens)
        language = resolve_language(payload.get("language"), text)
        started = time.perf_counter()
        first_audio_latency: float | None = None
        generated_steps = 0
        audio_chunks: list[np.ndarray] = []
        emitted_samples = 0
        decode_chunks = 0
        runaway_retries = 0
        early_end_retries = 0
        best_attempt: dict[str, Any] | None = None
        stream_audio = bool(payload.get("stream_audio", True))
        effective_seed = int(seed) if seed is not None and int(seed) >= 0 else None

        yield {
            "type": "metadata",
            "data": {
                "run_id": run_id,
                "run_dir": str(run_dir),
                "model_profile": self.profile_id,
                "runtime_family": "faster_qwen3_tts",
                "sample_rate": self.sample_rate,
                "channels": 1,
                "n_vq": 16,
                "frame_rate": 12.0,
                "backend": self.backend,
                "quant": self.quant,
                "attn_implementation": "ggml_metal" if self.backend == "ggml" else "cuda_graph_sdpa",
                "codec_weight_dtype": "gguf" if self.backend == "ggml" else "bf16",
                "codec_compute_dtype": "ggml" if self.backend == "ggml" else "bf16",
                "processor_mode": "voice_clone",
                "reference_cache_hit": cache_hit,
                "seed": int(seed) if seed is not None else None,
                "language": language,
            },
        }

        generation_kwargs = {
            "text": text,
            "language": language,
            "ref_audio": str(ref_audio),
            "ref_text": ref_text,
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": max(2, min(int(payload.get("min_new_tokens") or 2), max_new_tokens)),
            "temperature": float(payload.get("temperature") or 0.9),
            "top_k": max(1, min(200, int(payload.get("top_k") or 50))),
            "top_p": max(0.01, min(1.0, float(payload.get("top_p") or 1.0))),
            "do_sample": bool(payload.get("do_sample", True)),
            "repetition_penalty": max(0.8, min(2.0, float(payload.get("repetition_penalty") or 1.05))),
            "chunk_size": chunk_size,
            "xvec_only": xvec_only,
            "non_streaming_mode": bool(payload.get("non_streaming_mode", False)),
            "append_silence": bool(payload.get("append_silence", True)),
            "instruct": str(payload.get("instruct") or "").strip() or None,
        }
        if self.backend == "ggml" and generation_kwargs["instruct"]:
            raise ValueError("GGML Base voice-clone backend does not support instruct")

        def build_generator(seed_value: int | None, repetition_penalty_value: float):
            # FasterQwen3TTS does not yet expose qwentts.cpp's seed argument.
            # Use its adapter preparation and native streaming helpers so the
            # effective seed saved by the service is also the seed actually used.
            if self.backend == "ggml" and seed_value is not None:
                ref_kwargs, adapter_prepare_ms, adapter_profile = self.model._resolve_clone_reference(
                    ref_audio=generation_kwargs["ref_audio"],
                    ref_text=ref_text,
                    xvec_only=xvec_only,
                    append_silence=generation_kwargs["append_silence"],
                    ref_spk=None,
                    ref_rvq=None,
                    ref_spk_emb=None,
                    ref_codes=None,
                )
                return self.model._stream_runtime(
                    text=text,
                    lang=language,
                    **ref_kwargs,
                    max_new_tokens=max_new_tokens,
                    do_sample=generation_kwargs["do_sample"],
                    temperature=generation_kwargs["temperature"],
                    top_k=generation_kwargs["top_k"],
                    top_p=generation_kwargs["top_p"],
                    repetition_penalty=repetition_penalty_value,
                    seed=int(seed_value),
                    chunk_size=chunk_size,
                    adapter_prepare_ms=adapter_prepare_ms,
                    adapter_profile=adapter_profile,
                )
            kwargs = dict(generation_kwargs)
            kwargs["repetition_penalty"] = repetition_penalty_value
            return self.model.generate_voice_clone_streaming(**kwargs)

        retry_reason = ""
        max_attempts = 1 + max(RUNAWAY_MAX_RETRIES, EARLY_END_MAX_RETRIES)
        for attempt in range(max_attempts):
            seed_value = int(seed) if seed is not None and int(seed) >= 0 else None
            repetition_penalty_value = generation_kwargs["repetition_penalty"]
            if attempt > 0 and retry_reason == "early_end":
                seed_value = _retry_seed(effective_seed)
                effective_seed = seed_value
                audio_chunks = []
                emitted_samples = 0
                decode_chunks = 0
                generated_steps = 0
                first_audio_latency = None
                yield {
                    "type": "progress",
                    "data": {
                        "early_end_retry": attempt,
                        "early_end_retry_seed": seed_value,
                        "early_end_retry_note": "音频明显短于文本，疑似提前结束，已更换种子重试",
                    },
                }
            elif attempt > 0:
                seed_value = _retry_seed(effective_seed)
                repetition_penalty_value = max(
                    repetition_penalty_value, RUNAWAY_RETRY_REPETITION_PENALTY
                )
                effective_seed = seed_value
                audio_chunks = []
                emitted_samples = 0
                decode_chunks = 0
                generated_steps = 0
                first_audio_latency = None
                yield {
                    "type": "progress",
                    "data": {
                        "runaway_retry": attempt,
                        "runaway_retry_seed": seed_value,
                        "runaway_retry_note": "检测到复读式失控生成，已更换种子重试",
                    },
                }
            if seed_value is not None:
                torch.manual_seed(int(seed_value))
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(int(seed_value))

            runaway = False
            generator = build_generator(seed_value, repetition_penalty_value)
            try:
                for audio, sample_rate, timing in generator:
                    if client_gone is not None and client_gone():
                        raise ClientDisconnected("service closed the generation request")
                    chunk = np.asarray(audio, dtype=np.float32).reshape(-1)
                    if chunk.size == 0:
                        continue
                    if int(sample_rate) != self.sample_rate:
                        self.sample_rate = int(sample_rate)
                    if first_audio_latency is None:
                        first_audio_latency = time.perf_counter() - started
                    emitted_seconds = (emitted_samples + chunk.size) / float(self.sample_rate)
                    if emitted_seconds > budget_seconds:
                        runaway = True
                        break
                    # Quantize once: the streamed PCM and the saved WAV are
                    # built from these exact int16 samples (bit-identical).
                    pcm = (np.clip(chunk, -1.0, 1.0) * 32767.0).round().astype("<i2")
                    audio_chunks.append(pcm)
                    emitted_samples += int(chunk.size)
                    steps = timing.get("total_steps_so_far")
                    if steps:
                        generated_steps = int(steps)
                    else:
                        generated_steps = round(emitted_samples * QWEN_CODEC_FRAMES_PER_SECOND / float(self.sample_rate))
                    decode_chunks += 1
                    elapsed = max(1e-6, time.perf_counter() - started)
                    audio_seconds = emitted_samples / float(self.sample_rate)
                    if stream_audio:
                        yield {
                            "type": "audio",
                            "data": {
                                "pcm16_base64": base64.b64encode(pcm.tobytes()).decode("ascii"),
                                "samples": int(chunk.size),
                                "sample_rate": self.sample_rate,
                                "channels": 1,
                                "generated_frames": generated_steps,
                                "generated_audio_seconds": audio_seconds,
                                "emitted_audio_seconds": audio_seconds,
                                "generation_realtime_factor": audio_seconds / elapsed,
                                "first_audio_latency_seconds": first_audio_latency,
                                "decode_chunks_submitted": decode_chunks,
                                "chunk_frames": chunk_size,
                            },
                        }
            finally:
                # Closing cancels the native stream promptly when we break out
                # on a runaway instead of waiting for garbage collection.
                generator.close()
            if not runaway:
                audio_seconds_total = emitted_samples / float(self.sample_rate)
                if (
                    not stream_audio
                    and early_end_retries < EARLY_END_MAX_RETRIES
                    and attempt + 1 < max_attempts
                    and early_end_detected(text, audio_seconds_total, language)
                ):
                    LOG.warning(
                        "early end suspected: %.2fs of audio for a %d-char text; retrying with a new seed",
                        audio_seconds_total,
                        len(text),
                    )
                    best_attempt = {
                        "audio_chunks": audio_chunks, "emitted_samples": emitted_samples,
                        "decode_chunks": decode_chunks, "generated_steps": generated_steps,
                        "first_audio_latency": first_audio_latency, "effective_seed": effective_seed,
                    }
                    early_end_retries += 1
                    retry_reason = "early_end"
                    continue
                if best_attempt is not None and best_attempt["emitted_samples"] > emitted_samples:
                    # The retry was even shorter; keep the original attempt.
                    audio_chunks = best_attempt["audio_chunks"]
                    emitted_samples = best_attempt["emitted_samples"]
                    decode_chunks = best_attempt["decode_chunks"]
                    generated_steps = best_attempt["generated_steps"]
                    first_audio_latency = best_attempt["first_audio_latency"]
                    effective_seed = best_attempt["effective_seed"]
                break
            LOG.warning(
                "runaway generation detected: attempt %d produced %.1fs of audio for a %d-char text (budget %.1fs)",
                attempt + 1,
                emitted_samples / float(self.sample_rate),
                len(text),
                budget_seconds,
            )
            if best_attempt is not None:
                # The early-end retry ran away; keep the short but complete take.
                audio_chunks = best_attempt["audio_chunks"]
                emitted_samples = best_attempt["emitted_samples"]
                decode_chunks = best_attempt["decode_chunks"]
                generated_steps = best_attempt["generated_steps"]
                first_audio_latency = best_attempt["first_audio_latency"]
                effective_seed = best_attempt["effective_seed"]
                break
            if (
                (stream_audio and emitted_samples > 0)
                or runaway_retries >= RUNAWAY_MAX_RETRIES
                or attempt + 1 >= max_attempts
            ):
                raise RuntimeError(
                    f"生成疑似复读失控：文本约 {len(text)} 字，已生成 "
                    f"{emitted_samples / float(self.sample_rate):.1f} 秒音频仍未结束，已中止。"
                    "请重试，或更换参考音频/种子。"
                )
            runaway_retries += 1
            retry_reason = "runaway"

        elapsed = max(1e-6, time.perf_counter() - started)
        if not audio_chunks:
            raise RuntimeError("Qwen 模型未生成有效音频，请重试或更换参考音频")
        waveform = np.concatenate(audio_chunks)
        duration_seconds = waveform.size / float(self.sample_rate)
        if not stream_audio:
            yield {"type": "audio", "data": {
                "pcm16_base64": base64.b64encode(waveform.tobytes()).decode("ascii"),
                "samples": int(waveform.size), "sample_rate": self.sample_rate, "channels": 1,
                "generated_frames": generated_steps, "generated_audio_seconds": duration_seconds,
                "emitted_audio_seconds": duration_seconds, "first_audio_latency_seconds": first_audio_latency,
            }}
        audio_path = run_dir / "generated.wav"
        tokens_path = run_dir / "generated_tokens.json"
        metadata_path = run_dir / "metadata.json"
        sf.write(str(audio_path), waveform, self.sample_rate, subtype="PCM_16")
        tokens_path.write_text(
            json.dumps({"generated_frames": generated_steps, "codec_tokens_saved": False}, indent=2),
            encoding="utf-8",
        )
        self.reference_keys.add(cache_key)
        self.requests += 1
        metadata = {
            "run_id": run_id,
            "model_profile": self.profile_id,
            "runtime_family": "faster_qwen3_tts",
            "generated_frames": generated_steps,
            "duration_seconds": duration_seconds,
            "generation_seconds": elapsed,
            "sample_rate": self.sample_rate,
            "channels": 1,
            "streaming_supported": True,
            "first_audio_latency_seconds": first_audio_latency,
            "generation_realtime_factor": duration_seconds / elapsed,
            "decode_chunks_submitted": decode_chunks,
            "chunk_frames": chunk_size,
            "reference_cache_hit": cache_hit,
            "reference_cache_entries": len(self.reference_keys),
            "xvec_only": xvec_only,
            "non_streaming_mode": bool(payload.get("non_streaming_mode", False)),
            "seed": int(seed) if seed is not None else None,
            "effective_seed": effective_seed,
            "runaway_retries": runaway_retries,
            "early_end_retries": early_end_retries,
            "language": language,
            "backend": self.backend,
            "quant": self.quant,
        }
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        yield {
            "type": "result",
            "data": {
                "audio_path": str(audio_path),
                "tokens_path": str(tokens_path),
                "metadata_path": str(metadata_path),
                "metadata": metadata,
            },
        }


class WorkerHandler(BaseHTTPRequestHandler):
    server_version = "FasterQwenWorker/1.0"

    @property
    def state(self) -> WorkerState:
        return self.server.worker_state  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:
        LOG.debug("%s - %s", self.address_string(), fmt % args)

    def _json_response(self, status: int, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:  # noqa: N802
        if not self._authorized():
            return
        if self.path != "/health":
            self._json_response(404, {"detail": "not found"})
            return
        self._json_response(200, self.state.health())

    def do_POST(self) -> None:  # noqa: N802
        if not self._authorized():
            return
        if self.path != "/generate":
            self._json_response(404, {"detail": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length") or "0")
            if length <= 0 or length > MAX_WORKER_REQUEST_BYTES:
                self._json_response(413, {"detail": "worker request size is invalid"})
                return
            if self.headers.get_content_type() != "application/json":
                self._json_response(415, {"detail": "application/json is required"})
                return
            self.connection.settimeout(30)
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("request must be an object")
        except Exception as exc:
            self._json_response(400, {"detail": f"invalid request: {exc}"})
            return

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        events = None
        try:
            with self.state.generation_lock:
                events = self.state.generate(payload, client_gone=self._client_gone)
                try:
                    for event in events:
                        self.wfile.write(json.dumps(event, ensure_ascii=False).encode("utf-8") + b"\n")
                        self.wfile.flush()
                finally:
                    # Release the native stream before generation_lock so a
                    # cancelled request frees the lane at the next chunk.
                    events.close()
        except (BrokenPipeError, ConnectionResetError):
            LOG.info("client disconnected; generation stream closed")
        except Exception as exc:
            LOG.exception("generation failed")
            try:
                event = {"type": "error", "data": {"error": str(exc)}}
                self.wfile.write(json.dumps(event, ensure_ascii=False).encode("utf-8") + b"\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

    def _client_gone(self) -> bool:
        """Detect a closed request socket without writing to it.

        The service sends nothing after the request body, so a readable socket
        means EOF/RST.  This lets complete-block (non-streaming) generations,
        which write nothing until the end, stop promptly on cancellation.
        """
        try:
            readable, _, _ = select.select([self.connection], [], [], 0)
            if not readable:
                return False
            return self.connection.recv(1, socket.MSG_PEEK) == b""
        except (OSError, ValueError):
            return True

    def _authorized(self) -> bool:
        expected = getattr(self.server, "worker_token", "")
        if not expected or not secrets_equal(self.headers.get(WORKER_TOKEN_HEADER, ""), expected):
            self._json_response(401, {"detail": "worker authentication required"})
            return False
        return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--backend", choices=["ggml", "torch"], default="ggml")
    parser.add_argument("--quant", choices=["BF16", "Q8_0", "Q4_K_M"], default="Q4_K_M")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    token = os.environ.get("QWEN_WORKER_TOKEN", "")
    output_roots = tuple(Path(p).resolve() for p in json.loads(os.environ.get("QWEN_WORKER_OUTPUT_ROOTS", "[]")))
    reference_roots = tuple(Path(p).resolve() for p in json.loads(os.environ.get("QWEN_WORKER_REFERENCE_ROOTS", "[]")))
    if not token or not output_roots or not reference_roots:
        parser.error("worker must be started by the service with private credentials and directory limits")
    if args.host != "127.0.0.1":
        parser.error("worker must bind to 127.0.0.1")
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    state = WorkerState(
        model_path=args.model,
        profile_id=args.profile_id,
        max_seq_len=args.max_seq_len,
        backend=args.backend,
        quant=args.quant,
        output_roots=output_roots,
        reference_roots=reference_roots,
    )
    server = ThreadingHTTPServer((args.host, args.port), WorkerHandler)
    server.worker_state = state  # type: ignore[attr-defined]
    server.worker_token = token  # type: ignore[attr-defined]
    LOG.info("listening on http://%s:%s", args.host, args.port)
    server.serve_forever(poll_interval=0.25)


if __name__ == "__main__":
    main()
