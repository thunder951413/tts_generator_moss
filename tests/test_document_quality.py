from __future__ import annotations

import json
import shutil
import subprocess
import sys
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SERVICE_DIR = ROOT / "qwen_tts_service"
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

import audio_post  # noqa: E402
from document_projects import (  # noqa: E402
    DocumentProjectManager,
    extract_document_text,
    qwen_frame_budget,
    segment_text,
    segment_text_blocks,
    split_sentences,
)
from qwen_protocol import StreamingRequest  # noqa: E402
from text_processing import prepare_tts_text  # noqa: E402

FFMPEG = shutil.which("ffmpeg")
SR = 24000


def make_manager(tmp_path: Path, synthesize=None, workers: int = 1) -> DocumentProjectManager:
    return DocumentProjectManager(
        root_dir=tmp_path / "projects",
        runtime_session=lambda *_a, **_k: nullcontext(object()),
        synthesize_fn=synthesize or (lambda *_a, **_k: iter(())),
        request_cls=StreamingRequest,
        generation_lock=nullcontext(),
        ffmpeg_path=FFMPEG or "ffmpeg",
        synthesis_workers=workers,
    )


# -- segmentation -------------------------------------------------------------


def test_title_is_never_glued_to_the_paragraph_without_a_boundary() -> None:
    blocks = segment_text_blocks("第一章 雨夜\n雨落在旧城。\n他走进了巷子。", max_chars=60)
    assert blocks[0]["text"].startswith("第一章 雨夜\n")
    assert "雨夜雨落" not in "".join(block["text"] for block in blocks)
    assert blocks[-1]["pause_after"] == "chapter"


def test_short_paragraphs_merge_with_newline_but_long_ones_flush() -> None:
    long_paragraph = "这是一个很长的段落，" * 12 + "结束。"
    blocks = segment_text_blocks(f"短段一。\n短段二。\n{long_paragraph}\n尾段收束了这一章节的内容。", max_chars=80)
    assert blocks[0]["text"] == "短段一。\n短段二。"
    assert blocks[0]["pause_after"] == "paragraph"
    assert all(len(block["text"]) <= 80 * 1.2 for block in blocks)
    assert blocks[-1]["pause_after"] == "chapter"


def test_english_sentences_split_and_quotes_stay_attached() -> None:
    text = 'He opened the door. "Is anyone here?" she asked. Mr. Smith smiled. Pi is 3.14 exactly.'
    pieces = [piece.strip() for piece in split_sentences(text)]
    assert pieces == [
        "He opened the door.",
        '"Is anyone here?"',
        "she asked.",
        "Mr. Smith smiled.",
        "Pi is 3.14 exactly.",
    ]
    chinese = [piece for piece in split_sentences("他说：“走吧。”她点头。")]
    assert chinese == ["他说：“走吧。”", "她点头。"]


def test_english_paragraph_is_split_near_max_chars_without_cutting_words() -> None:
    sentence = "The quick brown fox jumps over the lazy dog near the riverbank. "
    chunks = segment_text(sentence * 8, max_chars=100)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 120
        assert chunk.endswith(".")
    assert " ".join(chunks).split() == (sentence * 8).split()


def test_long_sentence_cut_prefers_commas_then_spaces_never_inside_words() -> None:
    clause = "这一句话非常非常长并且没有句号"
    text = "，".join([clause] * 10) + "。"
    chunks = segment_text(text, max_chars=50)
    assert all(len(chunk) <= 50 for chunk in chunks[:-1])
    assert all(chunk.endswith("，") for chunk in chunks[:-1])
    assert "".join(chunks) == text

    words = " ".join(["supercalifragilistic"] * 20)
    english = segment_text(words, max_chars=60)
    for chunk in english:
        assert all(word == "supercalifragilistic" for word in chunk.split())
    mixed = "版本号ABCDEF1234567890XYZ" * 6
    mixed_chunks = segment_text(mixed, max_chars=45)
    assert "".join(mixed_chunks) == mixed
    for left, right in zip(mixed_chunks, mixed_chunks[1:]):
        assert not (left[-1].isascii() and left[-1].isalnum() and right[0].isascii() and right[0].isalnum())


def test_tiny_tail_chunks_merge_into_previous() -> None:
    text = "这是第一句比较长的内容，用来填满文字块的大部分长度。" * 2 + "好。"
    chunks = segment_text(text, max_chars=55)
    assert chunks[-1].endswith("好。")
    assert all(len(chunk) >= 12 for chunk in chunks)


def test_markdown_import_is_stripped() -> None:
    text = extract_document_text(
        "book.md",
        "# 第一章 雨夜\n\n**雨**落在[旧城](https://example.com)。\n- 列表项\n```\n```\n![图](a.png)见 https://x.y/z".encode(),
    )
    assert text.splitlines()[0] == "第一章 雨夜"
    assert "**" not in text and "](" not in text and "http" not in text and "#" not in text
    assert "雨落在旧城。" in text
    assert "列表项" in text


def test_prepare_tts_text() -> None:
    assert prepare_tts_text("第一章 雨夜\n雨落在旧城。") == "第一章 雨夜。雨落在旧城。"
    assert prepare_tts_text("Chapter One\nIt rained.") == "Chapter One. It rained."
    assert prepare_tts_text("他走了。\n她留下。") == "他走了。她留下。"
    assert prepare_tts_text("Done.\nNext.") == "Done. Next."
    assert prepare_tts_text("等等……他说....好") == "等等…他说…好"
    assert prepare_tts_text("编号１２３ＡＢ，好") == "编号123AB，好"
    assert prepare_tts_text("**重点** 见[链接](http://a.b)") == "重点 见链接"


# -- audio post-processing --------------------------------------------------------


def _tone(seconds: float, amplitude: float, freq: float = 220.0) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_audio_post_dc_trim_normalize_and_pauses() -> None:
    speech = _tone(1.0, 0.05)
    audio = np.concatenate([np.zeros(int(0.5 * SR), np.float32), speech, np.zeros(int(0.8 * SR), np.float32)]) + 0.1
    assert abs(float(audio_post.remove_dc(audio).mean())) < 1e-4

    trimmed = audio_post.trim_silence(audio_post.remove_dc(audio), SR)
    assert abs(trimmed.size / SR - (1.0 + 0.06 + 0.10)) < 0.03

    processed = audio_post.process_segment_audio(audio, SR, "sentence")
    body = processed[: processed.size - int(0.30 * SR)]
    assert np.all(processed[processed.size - int(0.30 * SR) :] == 0)
    rms_db = 20 * np.log10(audio_post.speech_rms(body, SR))
    assert abs(rms_db - (-20.0)) <= 1.0
    assert abs(processed[0]) < 1e-6  # fade-in starts at zero

    for pause, seconds in (("sentence", 0.30), ("paragraph", 0.60), ("chapter", 1.0)):
        padded = audio_post.append_pause(np.ones(100, np.float32), SR, pause)
        assert padded.size == 100 + int(round(seconds * SR))


def test_audio_post_gain_clamp_and_peak_limit() -> None:
    quiet = _tone(1.0, 0.012)  # -41 dBFS RMS: above the silence gate, needs +21 dB
    boosted = audio_post.normalize_loudness(quiet, SR)
    assert abs(20 * np.log10(boosted.max() / quiet.max()) - 10.0) < 0.1  # clamped to +10 dB

    spiky = _tone(1.0, 0.05)
    spiky[1000] = 0.99
    limited = audio_post.limit_peak(audio_post.normalize_loudness(spiky, SR))
    assert np.max(np.abs(limited)) <= 10 ** (-1.5 / 20) + 1e-6
    loud = _tone(1.0, 0.95)
    out = audio_post.process_segment_audio(loud, SR, "paragraph")
    assert np.max(np.abs(out)) <= 10 ** (-1.5 / 20) + 1e-6


# -- manifest store ---------------------------------------------------------------


def _create(manager: DocumentProjectManager, text: str = "第一段。\n第二段内容比较长，需要单独成块，这样才能分成多个文字块来测试。\n第三段也很长，同样需要单独成块，这样才能分成多个文字块。") -> dict:
    return manager.create_project(
        name="测试", filename="book.txt", data=text.encode(),
        settings={"model_profile": "qwen_0_6b", "seed": 1234, "seed_mode": "fixed"}, max_chars=40,
    )


def test_revision_progress_and_since_semantics(tmp_path: Path) -> None:
    manager = make_manager(tmp_path)
    project = _create(manager)
    assert project["revision"] >= 1 and project["text_revision"] >= 1
    assert all(segment["rev"] == project["revision"] for segment in project["segments"])

    full = manager.get_progress(project["id"])
    assert len(full["segments"]) == len(project["segments"])
    assert all("text" not in segment for segment in full["segments"])
    assert "status" in full["segments"][0]

    base = full["revision"]
    assert manager.get_progress(project["id"], since=base)["segments"] == []
    with manager._lock:
        manifest = manager._load(project["id"])
        manifest["segments"][1]["status"] = "failed"
        manager._save(manifest, segments=(1,))
    delta = manager.get_progress(project["id"], since=base)
    assert delta["revision"] == base + 1
    assert [segment["index"] for segment in delta["segments"]] == [1]
    assert delta["text_revision"] == full["text_revision"]

    manager.rename(project["id"], name="新书名")
    assert manager.get_progress(project["id"], since=base + 1)["segments"] == []

    edited = manager.update_segment_text(project["id"], segment_index=0, text="第一段\n改过。")
    assert edited["text_revision"] == full["text_revision"] + 1
    assert edited["segments"][0]["text"] == "第一段\n改过。"


def test_progress_since_ahead_of_server_revision_resyncs(tmp_path: Path) -> None:
    # Coalesced writes lost in a crash can move the revision backwards; a
    # reader polling with its newer ``since`` must not miss every change.
    manager = make_manager(tmp_path)
    project = _create(manager)
    revision = manager.get_progress(project["id"])["revision"]
    progress = manager.get_progress(project["id"], since=revision + 50)
    assert len(progress["segments"]) == len(project["segments"])


def test_summaries_lack_segments_and_are_cached(tmp_path: Path, monkeypatch) -> None:
    manager = make_manager(tmp_path)
    project = _create(manager)
    calls = []
    original = json.loads
    monkeypatch.setattr(json, "loads", lambda *a, **k: calls.append(1) or original(*a, **k))
    summaries = manager.list_projects()
    usage = manager.reference_usage_index()
    monkeypatch.undo()
    assert calls == []
    assert usage == {}
    summary = summaries[0]
    assert summary["id"] == project["id"]
    assert "segments" not in summary and "chapters" not in summary
    assert summary["chapter_count"] == 1
    for key in ("stats", "playback", "state", "message", "name", "created_at", "updated_at",
                "settings", "final_audio", "revision", "text_revision", "active_segments"):
        assert key in summary
    assert summary["stats"]["total_segments"] == len(project["segments"])


def test_playback_is_a_separate_file_and_stale_updates_are_ignored(tmp_path: Path) -> None:
    manager = make_manager(tmp_path)
    project = _create(manager)
    manifest_path = tmp_path / "projects" / project["id"] / "manifest.json"
    before = manifest_path.stat().st_mtime_ns
    stamp = time.time() * 1000
    result = manager.update_playback(project["id"], segment_index=1, offset_seconds=3.5, captured_at=stamp, seed=42)
    assert result == {"ok": True, "playback": result["playback"]}
    assert result["playback"]["offset_seconds"] == 3.5
    assert manifest_path.stat().st_mtime_ns == before
    stale = manager.update_playback(project["id"], segment_index=0, offset_seconds=0, captured_at=stamp - 5)
    assert stale["ok"] is True and stale["playback"]["offset_seconds"] == 3.5
    stored = json.loads((tmp_path / "projects" / project["id"] / "playback.json").read_text())
    assert stored["segment_index"] == 1
    assert make_manager(tmp_path).get_project(project["id"])["playback"]["offset_seconds"] == 3.5


def test_playback_migrates_from_legacy_manifest(tmp_path: Path) -> None:
    manager = make_manager(tmp_path)
    project = _create(manager)
    path = tmp_path / "projects" / project["id"] / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["playback"] = {"segment_index": 2, "offset_seconds": 9.0, "captured_at": 1.0}
    path.write_text(json.dumps(manifest, indent=2))
    restored = make_manager(tmp_path).get_project(project["id"])
    assert restored["playback"]["offset_seconds"] == 9.0
    assert restored["revision"] >= 1


def test_manifest_is_compact_and_status_churn_is_coalesced_then_flushed(tmp_path: Path) -> None:
    manager = make_manager(tmp_path)
    manager.coalesce_seconds = 30.0
    project = _create(manager)
    path = tmp_path / "projects" / project["id"] / "manifest.json"
    assert "\n" not in path.read_text(encoding="utf-8").strip()
    with manager._lock:
        manifest = manager._load(project["id"])
        manifest["segments"][0]["status"] = "generating"
        manager._save(manifest, segments=(0,), coalesce=True)
    on_disk = json.loads(path.read_text())
    assert on_disk["segments"][0]["status"] == "pending"  # deferred
    assert manager.get_project(project["id"])["segments"][0]["status"] == "generating"
    manager.stop(project["id"])
    assert json.loads(path.read_text())["segments"][0]["status"] == "generating"

    with manager._lock:
        manifest["segments"][1]["status"] = "encoding"
        manager._save(manifest, segments=(1,), coalesce=True)
    manager.stop_all()
    assert json.loads(path.read_text())["segments"][1]["status"] == "encoding"
    assert not list(path.parent.glob("*.tmp"))


def test_restart_repairs_interrupted_run(tmp_path: Path) -> None:
    manager = make_manager(tmp_path)
    project = _create(manager)
    with manager._lock:
        manifest = manager._load(project["id"])
        manifest["state"] = "running"
        manifest["segments"][0]["status"] = "generating"
        manager._save(manifest, segments=(0,))
    repaired = make_manager(tmp_path).get_project(project["id"])
    assert repaired["state"] == "paused"
    assert repaired["segments"][0]["status"] == "pending"
    assert repaired["segments"][0]["rev"] == repaired["revision"]


# -- synthesis request -------------------------------------------------------------


def test_frame_budget_formula() -> None:
    assert qwen_frame_budget("字" * 100, 2048) == 675
    assert qwen_frame_budget("字" * 1000, 2048) == 2048
    assert qwen_frame_budget("", 2048) == 50


def test_synthesis_uses_language_defaults_budget_and_retries_truncation(tmp_path: Path) -> None:
    requests: list[StreamingRequest] = []

    def synthesize(_runtime, request, *, output_dir):
        requests.append(request)
        run_dir = Path(output_dir) / f"run{len(requests)}"
        run_dir.mkdir(parents=True, exist_ok=True)
        source = run_dir / "generated.wav"
        source.write_bytes(b"wav")
        frames = request.max_new_frames if len(requests) == 1 else 10
        yield SimpleNamespace(type="result", data={
            "audio_path": str(source),
            "metadata": {"duration_seconds": frames / 12.5, "generated_frames": frames},
        })

    manager = make_manager(tmp_path, synthesize)
    project = manager.create_project(
        name="English", filename="book.txt", data=b"Chapter One\nIt was a dark and stormy night.",
        settings={"model_profile": "qwen_0_6b", "seed": 1234, "seed_mode": "fixed",
                  "max_new_tokens": 2048, "language": "auto"},
        max_chars=160,
    )
    segment = dict(manager.get_project(project["id"])["segments"][0])
    segment["pause_after"] = "chapter"
    result = manager._synthesize_segment(project["id"], segment, manager.get_project(project["id"])["settings"])
    first, retry = requests
    assert first.language == "English"
    assert first.text == "Chapter One. It was a dark and stormy night."
    assert first.temperature == 0.9 and first.top_p == 1.0 and first.top_k == 50
    assert first.repetition_penalty == 1.05
    assert first.max_new_frames == qwen_frame_budget(first.text, 2048)
    assert retry.max_new_frames == min(2048, int(first.max_new_frames * 1.5))
    assert retry.seed != first.seed
    assert result["truncated_retry"] is True and result["truncated"] is False
    assert result["pause_after"] == "chapter"
    assert not (tmp_path / "projects" / project["id"] / "working" / "run1").exists()


def test_failed_truncation_retry_keeps_first_take(tmp_path: Path) -> None:
    # The worker's runaway guard (4 s + 0.5 s/char) equals the block's frame
    # budget, so the doubled-budget retry can be aborted with an error. That
    # must not fail the block (and the whole book) or delete the first take.
    calls: list[StreamingRequest] = []

    def synthesize(_runtime, request, *, output_dir):
        calls.append(request)
        if len(calls) > 1:
            raise RuntimeError("生成疑似复读失控")
        run_dir = Path(output_dir) / "run1"
        run_dir.mkdir(parents=True, exist_ok=True)
        source = run_dir / "generated.wav"
        source.write_bytes(b"wav")
        frames = request.max_new_frames
        yield SimpleNamespace(type="result", data={
            "audio_path": str(source),
            "metadata": {"duration_seconds": frames / 12.5, "generated_frames": frames},
        })

    manager = make_manager(tmp_path, synthesize)
    project = _create(manager)
    segment = dict(manager.get_project(project["id"])["segments"][0])
    result = manager._synthesize_segment(project["id"], segment, manager.get_project(project["id"])["settings"])
    assert len(calls) == 2
    assert Path(result["source_wav"]).exists()
    assert result["truncated_retry"] is True and result["truncated"] is True
    assert result["max_new_frames"] == calls[0].max_new_frames


# -- ffmpeg encode / merge -----------------------------------------------------------


def _probe_duration(path: Path) -> float:
    output = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    return float(output.strip())


@pytest.mark.skipif(FFMPEG is None or shutil.which("ffprobe") is None, reason="ffmpeg not available")
def test_encode_and_gapless_merge(tmp_path: Path) -> None:
    import soundfile as sf

    manager = make_manager(tmp_path)
    project = _create(manager)
    project_dir = tmp_path / "projects" / project["id"]
    expected = 0.0
    segments = []
    for index, pause in enumerate(("sentence", "paragraph", "chapter")):
        run_dir = project_dir / "working" / f"run{index}"
        run_dir.mkdir(parents=True)
        wav = run_dir / "generated.wav"
        audio = np.concatenate([np.zeros(SR // 2, np.float32), _tone(1.0 + 0.1 * index, 0.2), np.zeros(SR // 2, np.float32)])
        sf.write(str(wav), audio, SR, subtype="PCM_16")
        result = manager._encode_segment_aac(project["id"], index, {
            "source_wav": str(wav), "duration_seconds": audio.size / SR, "generation_seconds": 0.1,
            "seed": 1, "seed_mode": "fixed", "aac_bitrate": "96k", "pause_after": pause,
        })
        assert result["post_processed"] is True
        assert not run_dir.exists()
        pause_seconds = audio_post.PAUSE_SECONDS[pause]
        assert abs(result["duration_seconds"] - (1.0 + 0.1 * index + 0.16 + pause_seconds)) < 0.03
        expected += result["duration_seconds"]
        segments.append({"index": index, **result})
    snapshot = {"id": project["id"], "settings": {}, "segments": segments}
    manager._merge_final_audio(snapshot)
    final = project_dir / snapshot["final_audio"]
    assert final.exists()
    assert abs(_probe_duration(final) - expected) < 0.06, (_probe_duration(final), expected)
    assert not list((project_dir / "final").glob("merge-*"))
