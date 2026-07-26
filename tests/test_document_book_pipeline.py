from __future__ import annotations

import sys
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SERVICE_DIR = ROOT / "qwen_tts_service"
sys.path.insert(0, str(SERVICE_DIR))

from document_projects import DocumentProjectManager, split_novel_chapters  # noqa: E402
from qwen_protocol import StreamingRequest  # noqa: E402


def test_novel_chapters_are_detected_and_mapped_to_segments(tmp_path: Path) -> None:
    chapters = split_novel_chapters(
        "序章\n这是故事的开始。\n\n第一章 雨夜\n雨落在旧城。\n\n第二章 重逢\n他们终于重逢。"
    )
    assert [item["title"] for item in chapters] == ["序章", "第一章 雨夜", "第二章 重逢"]

    manager = DocumentProjectManager(
        root_dir=tmp_path / "projects",
        runtime_session=lambda _profile: nullcontext(object()),
        synthesize_fn=lambda *_args, **_kwargs: iter(()),
        request_cls=StreamingRequest,
        generation_lock=nullcontext(),
        ffmpeg_path="ffmpeg",
        synthesis_workers=2,
    )
    project = manager.create_project(
        name="章节测试",
        filename="novel.txt",
        data="序章\n这是故事的开始。\n第一章 雨夜\n雨落在旧城。".encode(),
        settings={"model_profile": "qwen_0_6b", "seed": 1234, "seed_mode": "fixed"},
        max_chars=40,
    )
    assert [item["title"] for item in project["chapters"]] == ["序章", "第一章 雨夜"]
    assert project["segments"][0]["chapter_index"] == 0
    assert project["segments"][-1]["chapter_index"] == 1

    revised = manager.update_segment_text(
        project["id"],
        segment_index=0,
        text="序章。这是编辑后保存的故事开端。",
    )
    assert revised["segments"][0]["text"] == "序章。这是编辑后保存的故事开端。"
    assert revised["segments"][0]["status"] == "pending"
    assert revised["final_audio"] is None


def test_book_pipeline_uses_two_lanes_and_preserves_segment_order(tmp_path: Path) -> None:
    counter_lock = threading.Lock()
    active = 0
    maximum_active = 0
    both_started = threading.Event()

    def synthesize(_runtime, request, *, output_dir):
        nonlocal active, maximum_active
        assert request.qwen_non_streaming_mode is True
        with counter_lock:
            active += 1
            maximum_active = max(maximum_active, active)
            if active == 2:
                both_started.set()
        assert both_started.wait(timeout=2)
        # Complete odd-numbered segments first to exercise ordered persistence.
        time.sleep(0.02 if "第二段" in request.text or "第四段" in request.text else 0.06)
        source = Path(output_dir) / f"{request.text}.wav"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b"wav")
        with counter_lock:
            active -= 1
        yield SimpleNamespace(
            type="result",
            data={
                "audio_path": str(source),
                "metadata": {"duration_seconds": 1.0},
            },
        )

    manager = DocumentProjectManager(
        root_dir=tmp_path / "projects",
        runtime_session=lambda _profile: nullcontext(object()),
        synthesize_fn=synthesize,
        request_cls=StreamingRequest,
        generation_lock=nullcontext(),
        ffmpeg_path="ffmpeg",
        synthesis_workers=2,
    )

    def encode(_project_id, index, _synthesis):
        time.sleep(0.01)
        return {
            "audio_file": f"segments/{index:06d}.m4a",
            "duration_seconds": 1.0,
            "generation_seconds": 0.1,
            "seed": 1234,
            "seed_mode": "fixed",
        }

    merged_order: list[str] = []

    def merge(manifest):
        merged_order.extend(segment["audio_file"] for segment in manifest["segments"])
        manifest["final_audio"] = "final/complete.m4a"

    manager._encode_segment_aac = encode  # type: ignore[method-assign]
    manager._merge_final_audio = merge  # type: ignore[method-assign]
    settings = {
        "model_profile": "qwen_0_6b",
        "reference_audio_path": str(tmp_path / "reference.wav"),
        "seed": 1234,
        "seed_mode": "fixed",
        "max_new_tokens": 2048,
        "qwen_non_streaming_mode": True,
    }
    project = manager.create_project(
        name="并行整本书",
        filename="book.txt",
        data=(
            "第一段用于验证整本书的双通道并行生成能力与顺序保持。\n"
            "第二段会更早完成，用于模拟真实生成中的乱序返回情况。\n"
            "第三段继续验证新的任务可以立即填满已经空闲的生成通道。\n"
            "第四段最终确认所有音频仍然严格按照原文顺序完成合并。"
        ).encode(),
        settings=settings,
        max_chars=40,
    )
    manager.start(
        project["id"],
        segment_indices=list(range(len(project["segments"]))),
    )
    thread = manager._threads[project["id"]]
    thread.join(timeout=5)

    assert not thread.is_alive()
    result = manager.get_project(project["id"])
    assert maximum_active == 2
    assert result["state"] == "completed"
    assert result["active_segments"] == []
    assert result["current_run_started_at"] is None
    assert result["stats"]["parallel_generations"] == 2
    assert result["stats"]["generation_elapsed_seconds"] > 0
    assert result["stats"]["speed_realtime"] > 0
    assert [segment["status"] for segment in result["segments"]] == ["completed"] * 4
    assert merged_order == [
        "segments/000000.m4a",
        "segments/000001.m4a",
        "segments/000002.m4a",
        "segments/000003.m4a",
    ]


def test_book_pipeline_stop_finishes_active_blocks_then_pauses(tmp_path: Path) -> None:
    started = threading.Event()
    release = threading.Event()
    active_lock = threading.Lock()
    active = 0

    def synthesize(_runtime, request, *, output_dir):
        nonlocal active
        with active_lock:
            active += 1
            if active == 2:
                started.set()
        assert release.wait(timeout=2)
        source = Path(output_dir) / f"{request.text[:8]}.wav"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b"wav")
        yield SimpleNamespace(
            type="result",
            data={
                "audio_path": str(source),
                "metadata": {"duration_seconds": 1.0},
            },
        )

    manager = DocumentProjectManager(
        root_dir=tmp_path / "projects",
        runtime_session=lambda _profile: nullcontext(object()),
        synthesize_fn=synthesize,
        request_cls=StreamingRequest,
        generation_lock=nullcontext(),
        ffmpeg_path="ffmpeg",
        synthesis_workers=2,
    )
    manager._encode_segment_aac = lambda _project_id, index, _synthesis: {  # type: ignore[method-assign]
        "audio_file": f"segments/{index:06d}.m4a",
        "duration_seconds": 1.0,
        "generation_seconds": 0.1,
        "seed": 1234,
        "seed_mode": "fixed",
    }
    project = manager.create_project(
        name="停止整书生成",
        filename="book.txt",
        data=(
            "第一段用于测试停止任务，保证这个文字块足够长且会独立生成。\n"
            "第二段用于测试停止任务，保证这个文字块足够长且会独立生成。\n"
            "第三段不应在停止后开始，仍然保持为等待生成的完整文字块。\n"
            "第四段不应在停止后开始，仍然保持为等待生成的完整文字块。"
        ).encode(),
        settings={
            "model_profile": "qwen_0_6b",
            "seed": 1234,
            "seed_mode": "fixed",
            "qwen_non_streaming_mode": True,
        },
        max_chars=18,
    )
    manager.start(project["id"])
    assert started.wait(timeout=2)
    stopping = manager.stop(project["id"])
    assert stopping["state"] == "stopping"
    release.set()
    manager._threads[project["id"]].join(timeout=3)

    result = manager.get_project(project["id"])
    assert result["state"] == "paused"
    assert result["message"] == "已暂停，可随时继续"
    assert sum(segment["status"] == "completed" for segment in result["segments"]) == 2
    assert any(segment["status"] == "pending" for segment in result["segments"])
