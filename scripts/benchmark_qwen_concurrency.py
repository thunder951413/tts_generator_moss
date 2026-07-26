# coding=utf-8
"""Benchmark Faster Qwen3-TTS lane counts on the local GPU."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
STREAMING_DIR = REPO_ROOT / "moss_tts_local_v1.5"
sys.path.insert(0, str(STREAMING_DIR))

from qwen_runtime import QwenWorkerRuntime  # noqa: E402
from streaming import StreamingRequest  # noqa: E402


def synthesize_once(
    runtime: QwenWorkerRuntime,
    *,
    reference_audio: Path,
    text: str,
    frames: int,
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    request = StreamingRequest(
        text=text,
        mode="voice_clone",
        prompt_audio_path=str(reference_audio),
        language="Chinese",
        max_new_frames=int(frames),
        temperature=0.9,
        top_p=1.0,
        top_k=50,
        repetition_penalty=1.05,
        seed=int(seed),
        codec_chunk_frames=8,
        qwen_xvec_only=True,
        qwen_min_new_tokens=2,
    )
    result: dict[str, Any] | None = None
    first_audio_wall: float | None = None
    started = time.perf_counter()
    for event in runtime.synthesize(request, output_dir=output_dir):
        if event.type == "audio" and first_audio_wall is None:
            first_audio_wall = time.perf_counter() - started
        elif event.type == "result":
            result = dict(event.data.get("metadata") or {})
    if result is None:
        raise RuntimeError("Qwen worker did not return a result event")
    result["client_first_audio_seconds"] = first_audio_wall
    result["client_wall_seconds"] = time.perf_counter() - started
    return result


def benchmark_lanes(
    *,
    profile_id: str,
    model_dir: Path,
    qwen_python: Path,
    worker_script: Path,
    reference_audio: Path,
    lanes: int,
    frames: int,
    output_dir: Path,
) -> dict[str, Any]:
    runtime = QwenWorkerRuntime(
        profile_id=profile_id,
        model_dir=model_dir,
        python_executable=qwen_python,
        worker_script=worker_script,
        lanes=lanes,
        base_port=7900 if profile_id == "qwen_0_6b" else 7920,
        log_dir=REPO_ROOT / "logs" / "qwen-benchmark",
    )
    try:
        warm_text = "这是Qwen音色缓存和CUDA图预热。"
        for index in range(lanes):
            synthesize_once(
                runtime,
                reference_audio=reference_audio,
                text=warm_text,
                frames=24,
                seed=9000 + index,
                output_dir=output_dir / "warmup",
            )

        barrier = threading.Barrier(lanes + 1)
        results: list[dict[str, Any] | None] = [None] * lanes
        errors: list[str | None] = [None] * lanes

        def run_lane(index: int) -> None:
            try:
                barrier.wait()
                results[index] = synthesize_once(
                    runtime,
                    reference_audio=reference_audio,
                    text=(
                        "这是Faster Qwen三语音模型的本地并发性能测试。"
                        "我们使用相同长度的中文内容测量总吞吐、首包延迟与稳定性。"
                    ),
                    frames=frames,
                    seed=1234 + index,
                    output_dir=output_dir / f"{lanes}_lanes",
                )
            except Exception as exc:  # noqa: BLE001
                errors[index] = str(exc)

        threads = [threading.Thread(target=run_lane, args=(index,), daemon=True) for index in range(lanes)]
        for thread in threads:
            thread.start()
        wall_started = time.perf_counter()
        barrier.wait()
        for thread in threads:
            thread.join()
        wall_seconds = time.perf_counter() - wall_started

        completed = [result for result in results if result is not None]
        total_audio_seconds = sum(float(result.get("duration_seconds") or 0.0) for result in completed)
        return {
            "profile_id": profile_id,
            "lanes": lanes,
            "successes": len(completed),
            "errors": [error for error in errors if error],
            "wall_seconds": wall_seconds,
            "total_audio_seconds": total_audio_seconds,
            "aggregate_realtime_factor": total_audio_seconds / max(wall_seconds, 1e-6),
            "mean_first_audio_seconds": (
                sum(float(result.get("client_first_audio_seconds") or 0.0) for result in completed)
                / max(len(completed), 1)
            ),
            "mean_job_wall_seconds": (
                sum(float(result.get("client_wall_seconds") or 0.0) for result in completed)
                / max(len(completed), 1)
            ),
            "jobs": completed,
        }
    finally:
        runtime.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-id", choices=["qwen_0_6b", "qwen_1_7b"], required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--qwen-python", type=Path, required=True)
    parser.add_argument("--worker-script", type=Path, default=STREAMING_DIR / "qwen_worker.py")
    parser.add_argument("--reference-audio", type=Path, required=True)
    parser.add_argument("--max-lanes", type=int, default=4)
    parser.add_argument("--frames", type=int, default=96)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    results = []
    for lanes in range(1, max(1, args.max_lanes) + 1):
        try:
            result = benchmark_lanes(
                profile_id=args.profile_id,
                model_dir=args.model_dir.resolve(),
                qwen_python=args.qwen_python.resolve(),
                worker_script=args.worker_script.resolve(),
                reference_audio=args.reference_audio.resolve(),
                lanes=lanes,
                frames=max(24, int(args.frames)),
                output_dir=REPO_ROOT / "outputs" / "qwen_concurrency_benchmark" / args.profile_id,
            )
        except Exception as exc:  # noqa: BLE001
            result = {
                "profile_id": args.profile_id,
                "lanes": lanes,
                "successes": 0,
                "errors": [str(exc)],
            }
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        if result.get("successes", 0) < lanes:
            break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
