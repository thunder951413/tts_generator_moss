"""Multi-part novels (volumes, series) and per-book voice settings."""

from __future__ import annotations

import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SERVICE_DIR = ROOT / "qwen_tts_service"
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

import document_projects  # noqa: E402
from document_projects import (  # noqa: E402
    DocumentProjectManager,
    SettingsResetRequired,
    split_novel_chapters,
    voice_settings_differ,
)
from qwen_protocol import StreamingRequest  # noqa: E402

SETTINGS = {
    "model_profile": "qwen_0_6b",
    "reference_audio_path": "/voices/a.wav",
    "voice_name": "甲",
    "language": "auto",
    "temperature": 0.9,
    "top_p": 1.0,
    "top_k": 50,
    "repetition_penalty": 1.05,
    "seed": 1234,
    "configured_seed": 1234,
    "seed_mode": "fixed",
    "qwen_clone_mode": "xvec",
    "qwen_reference_text": "",
    "qwen_append_silence": True,
    "aac_bitrate": "80k",
}


def make_manager(tmp_path: Path) -> DocumentProjectManager:
    return DocumentProjectManager(
        root_dir=tmp_path / "projects",
        runtime_session=lambda *_a, **_k: nullcontext(object()),
        synthesize_fn=lambda *_a, **_k: iter(()),
        request_cls=StreamingRequest,
        generation_lock=nullcontext(),
        ffmpeg_path="ffmpeg",
        synthesis_workers=1,
    )


def create(manager: DocumentProjectManager, text: str, **kwargs):
    return manager.create_project(
        name=kwargs.pop("name", "书"),
        filename=kwargs.pop("filename", "book.txt"),
        data=text.encode(),
        settings=dict(SETTINGS),
        max_chars=60,
        **kwargs,
    )


def test_volume_headings_group_chapters_and_are_spoken() -> None:
    chapters = split_novel_chapters(
        "楔子\n很久以前。\n第一卷 风起\n第一章 雨夜\n雨落下。\n第二章 重逢\n他们重逢。\n"
        "第二卷 云涌\n卷首语在这里。\n第三章 远行\n出发。"
    )
    assert [(item["title"], item.get("volume")) for item in chapters] == [
        ("楔子", None),
        ("第一章 雨夜", "第一卷 风起"),
        ("第二章 重逢", "第一卷 风起"),
        ("第二卷 云涌", "第二卷 云涌"),
        ("第三章 远行", "第二卷 云涌"),
    ]
    # The volume heading directly before a chapter is read with that chapter.
    assert chapters[1]["text"].startswith("第一卷 风起\n第一章 雨夜")


def test_project_records_volumes_with_chapter_ranges(tmp_path: Path) -> None:
    manager = make_manager(tmp_path)
    book = create(manager, "第一卷 上\n第一章 a\n甲。\n第二章 b\n乙。\n第二卷 下\n第三章 c\n丙。")
    assert [(v["title"], v["chapter_start"], v["chapter_end"]) for v in book["volumes"]] == [
        ("第一卷 上", 0, 1),
        ("第二卷 下", 2, 2),
    ]
    assert [chapter["volume_index"] for chapter in book["chapters"]] == [0, 0, 1]
    summary = manager.list_projects()[0]
    assert summary["volume_count"] == 2
    assert "volumes" not in summary and "chapters" not in summary


def test_appending_files_turns_a_single_file_book_into_volumes(tmp_path: Path) -> None:
    manager = make_manager(tmp_path)
    book = create(manager, "第一章 a\n甲。\n第二章 b\n乙。", filename="凡人 第一卷.txt")
    assert book["volumes"] == []
    book = manager.append_document(
        book["id"], filename="凡人 第二卷.txt", data="第三章 c\n丙。".encode(), volume_title="第二卷"
    )
    assert [volume["title"] for volume in book["volumes"]] == ["凡人 第一卷", "第二卷"]
    assert [chapter["volume_index"] for chapter in book["chapters"]] == [0, 0, 1]
    # A file that declares its own volumes keeps them.
    book = manager.append_document(
        book["id"], filename="part3.txt", data="第三卷 终\n第四章 d\n丁。".encode()
    )
    assert [volume["title"] for volume in book["volumes"]][-1] == "第三卷 终"
    assert book["chapters"][-1]["volume_index"] == 2
    assert book["segments"][-1]["chapter_index"] == 3


def test_series_and_volume_titles_are_editable(tmp_path: Path) -> None:
    manager = make_manager(tmp_path)
    book = create(manager, "第一卷 x\n第一章 a\n甲。", series="  我的  系列 ")
    assert book["series"] == "我的 系列"
    updated = manager.update_metadata(book["id"], series="新系列", volumes=[{"index": 0, "title": "卷一 改"}])
    assert updated["series"] == "新系列"
    assert updated["volumes"][0]["title"] == "卷一 改"
    assert updated["name"] == "书"
    with pytest.raises(ValueError):
        manager.update_metadata(book["id"], name="   ")


def test_summary_reports_reading_chapter(tmp_path: Path) -> None:
    manager = make_manager(tmp_path)
    book = create(manager, "第一章 a\n甲。\n第二章 b\n乙。")
    second = book["chapters"][1]["segment_start"]
    manager.update_playback(book["id"], segment_index=second, offset_seconds=1.0)
    summary = manager.get_summary(book["id"])
    assert summary["reading_chapter_title"] == "第二章 b"
    assert summary["last_read_at"] is not None


def _complete_all(manager: DocumentProjectManager, project_id: str) -> None:
    with manager._locked(project_id) as manifest:
        for segment in manifest["segments"]:
            segment.update(status="completed", audio_file="segments/x.m4a", duration_seconds=1.0)
        manifest["final_audio"] = "final/complete.m4a"
        manifest["state"] = "completed"
        manager._save(manifest, all_segments=True)


def test_voice_change_requires_confirmation_and_keeps_reading_position(tmp_path: Path) -> None:
    manager = make_manager(tmp_path)
    book = create(manager, "第一章 a\n甲。\n第二章 b\n乙。")
    _complete_all(manager, book["id"])
    manager.update_playback(book["id"], segment_index=1, offset_seconds=2.0)
    other_voice = dict(SETTINGS, reference_audio_path="/voices/b.wav", voice_name="乙")
    with pytest.raises(SettingsResetRequired) as excinfo:
        manager.update_settings(book["id"], other_voice)
    assert excinfo.value.completed == len(book["segments"])
    assert manager.get_project(book["id"])["final_audio"] == "final/complete.m4a"

    updated = manager.update_settings(book["id"], other_voice, confirm_reset=True, voice_label="乙音色")
    assert updated["audio_reset"] is True
    assert updated["voice_label"] == "乙音色"
    assert updated["settings"]["reference_audio_path"] == "/voices/b.wav"
    assert all(segment["status"] == "pending" for segment in updated["segments"])
    assert updated["playback"]["segment_index"] == 1


def test_bitrate_only_change_keeps_audio_and_asks_for_remerge(tmp_path: Path) -> None:
    manager = make_manager(tmp_path)
    book = create(manager, "第一章 a\n甲。")
    _complete_all(manager, book["id"])
    # A random seed resolves differently every time; only the configured seed matters.
    updated = manager.update_settings(book["id"], dict(SETTINGS, aac_bitrate="64k", seed=999))
    assert updated["audio_reset"] is False
    assert updated["settings"]["aac_bitrate"] == "64k"
    assert all(segment["status"] == "completed" for segment in updated["segments"])
    assert updated["final_audio"] is None
    assert updated["state"] == "paused"


def test_voice_settings_differ_ignores_bitrate_and_resolved_seed() -> None:
    assert not voice_settings_differ(SETTINGS, dict(SETTINGS, aac_bitrate="128k", seed=55))
    assert voice_settings_differ(SETTINGS, dict(SETTINGS, configured_seed=-1))
    assert voice_settings_differ(SETTINGS, dict(SETTINGS, model_profile="qwen_1_7b"))


def test_aac_encode_falls_back_when_audiotoolbox_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "out.m4a"
    calls: list[str] = []

    def fake_run(command, **_kwargs):
        encoder = command[command.index("-c:a") + 1]
        calls.append(encoder)
        if encoder == "aac":
            output.write_bytes(b"ok")
            return SimpleNamespace(returncode=0, stderr="")
        return SimpleNamespace(returncode=1, stderr="AudioToolbox init error")

    monkeypatch.setattr(document_projects, "run_media_process", fake_run)
    monkeypatch.setitem(document_projects._AAC_ENCODER_CACHE, "fake-ffmpeg", "aac_at")
    encoder = document_projects.run_aac_encode(
        "fake-ffmpeg",
        lambda enc: ["ffmpeg", "-c:a", enc, str(output)],
        output,
        timeout=5,
        cancelled=lambda: False,
        failure_message="失败",
    )
    assert encoder == "aac"
    assert calls == ["aac_at", "aac"]
    # Later encodes go straight to the working encoder.
    assert document_projects.preferred_aac_encoder("fake-ffmpeg") == "aac"
