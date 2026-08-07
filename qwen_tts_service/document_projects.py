from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import subprocess
import threading
import time
import uuid
import zipfile
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Callable
from xml.etree import ElementTree


PROJECT_ID_RE = re.compile(r"^[a-f0-9]{32}$")
SUPPORTED_DOCUMENT_SUFFIXES = {".txt", ".md", ".markdown", ".docx"}
CHAPTER_HEADING_RE = re.compile(
    r"^(?:"
    r"第[零〇一二三四五六七八九十百千万两0-9]{1,12}[章节卷回部篇]"
    r"|Chapter\s+\d+"
    r"|序章|序言|前言|楔子|引子|尾声|后记|番外(?:篇|章)?"
    r")(?:[\s：:、.．\-—].*)?$",
    re.IGNORECASE,
)


def _now() -> float:
    return time.time()


def _atomic_json_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _decode_text_document(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("无法识别文本编码，请转换为 UTF-8 后重试")


def _extract_docx_text(data: bytes) -> str:
    from io import BytesIO

    with zipfile.ZipFile(BytesIO(data)) as archive:
        xml_data = archive.read("word/document.xml")
    root = ElementTree.fromstring(xml_data)
    namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    paragraphs: list[str] = []
    for paragraph in root.iter(namespace + "p"):
        text = "".join(node.text or "" for node in paragraph.iter(namespace + "t")).strip()
        if text:
            paragraphs.append(text)
    return "\n".join(paragraphs)


def extract_document_text(filename: str, data: bytes) -> str:
    suffix = Path(filename or "document.txt").suffix.lower()
    if suffix not in SUPPORTED_DOCUMENT_SUFFIXES:
        raise ValueError("当前仅支持 TXT、Markdown 和 DOCX 文档")
    if len(data) > 30 * 1024 * 1024:
        raise ValueError("文档不能超过 30 MB")
    text = _extract_docx_text(data) if suffix == ".docx" else _decode_text_document(data)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\t\u00a0]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "\n".join(line.strip() for line in text.splitlines()).strip()
    if not text:
        raise ValueError("文档中没有可生成的文本")
    return text


def segment_text(text: str, max_chars: int = 200) -> list[str]:
    max_chars = max(40, min(500, int(max_chars)))
    normalized = re.sub(r"[ \t]+", " ", text or "").strip()
    paragraphs = [part.strip() for part in re.split(r"\n+", normalized) if part.strip()]
    sentences: list[str] = []
    for paragraph in paragraphs:
        pieces = [part.strip() for part in re.split(r"(?<=[。！？!?；;…])", paragraph) if part.strip()]
        sentences.extend(pieces or [paragraph])

    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        while len(sentence) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            cut = sentence.rfind("，", 0, max_chars + 1)
            if cut < max_chars // 2:
                cut = sentence.rfind(",", 0, max_chars + 1)
            if cut < max_chars // 2:
                cut = max_chars
            else:
                cut += 1
            chunks.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if not sentence:
            continue
        candidate = sentence if not current else current + sentence
        if len(candidate) <= max_chars:
            current = candidate
        else:
            chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)
    return [chunk for chunk in chunks if chunk]


def split_novel_chapters(text: str, *, fallback_title: str = "正文") -> list[dict[str, str]]:
    """Split common Chinese/English novel headings without losing spoken text."""
    chapters: list[dict[str, str]] = []
    current_title = fallback_title
    current_lines: list[str] = []
    found_heading = False
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if line and len(line) <= 80 and CHAPTER_HEADING_RE.fullmatch(line):
            if current_lines:
                body = "\n".join(current_lines).strip()
                if body:
                    chapters.append({"title": current_title, "text": body})
            current_title = line
            current_lines = [line]
            found_heading = True
        else:
            current_lines.append(raw_line)
    body = "\n".join(current_lines).strip()
    if body:
        chapters.append({"title": current_title, "text": body})
    if not chapters:
        return [{"title": fallback_title, "text": text.strip()}]
    if found_heading and chapters[0]["title"] == fallback_title:
        chapters[0]["title"] = "序章"
    return chapters


def settings_fingerprint(settings: dict[str, Any]) -> str:
    payload = json.dumps(settings, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class DocumentProjectManager:
    def __init__(
        self,
        *,
        root_dir: str | Path,
        runtime_session: Callable[[str], AbstractContextManager[Any]],
        synthesize_fn: Callable[..., Any],
        request_cls: type,
        generation_lock: threading.Lock,
        ffmpeg_path: str,
        synthesis_workers: int = 2,
    ) -> None:
        self.root_dir = Path(root_dir).resolve()
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.runtime_session = runtime_session
        self.synthesize_fn = synthesize_fn
        self.request_cls = request_cls
        self.generation_lock = generation_lock
        self.ffmpeg_path = ffmpeg_path
        self.synthesis_workers = max(1, int(synthesis_workers))
        self._lock = threading.RLock()
        self._threads: dict[str, threading.Thread] = {}
        self._stop_events: dict[str, threading.Event] = {}
        self._repair_interrupted_projects()

    def configure_synthesis_workers(self, workers: int) -> int:
        with self._lock:
            self.synthesis_workers = max(1, min(2, int(workers)))
            return self.synthesis_workers

    def _project_dir(self, project_id: str) -> Path:
        if not PROJECT_ID_RE.fullmatch(project_id or ""):
            raise ValueError("invalid project id")
        path = (self.root_dir / project_id).resolve()
        if self.root_dir not in path.parents:
            raise ValueError("invalid project path")
        return path

    def _manifest_path(self, project_id: str) -> Path:
        return self._project_dir(project_id) / "manifest.json"

    def _load(self, project_id: str) -> dict[str, Any]:
        path = self._manifest_path(project_id)
        if not path.exists():
            raise FileNotFoundError(project_id)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        self._migrate_model_profile(manifest)
        return manifest

    @staticmethod
    def _migrate_model_profile(manifest: dict[str, Any]) -> bool:
        """Bind old projects to the Qwen 0.6B profile on the macOS branch."""
        changed = False
        settings = manifest.setdefault("settings", {})
        if not settings.get("model_profile"):
            settings["model_profile"] = "qwen_0_6b"
            changed = True
        if settings.get("seed_mode") not in {"fixed", "random"}:
            try:
                configured_seed = int(settings.get("seed", 1234))
            except (TypeError, ValueError):
                configured_seed = 1234
            settings["configured_seed"] = configured_seed
            if configured_seed < 0:
                settings["seed"] = secrets.randbelow(1_000_000)
                settings["seed_mode"] = "random"
            else:
                settings["seed"] = configured_seed
                settings["seed_mode"] = "fixed"
            changed = True
        profile = str(settings["model_profile"])
        for segment in manifest.get("segments", []):
            if segment.get("status") == "completed" and not segment.get("model_profile"):
                segment["model_profile"] = profile
                changed = True
        if changed:
            # This is a metadata migration only. Existing audio is deliberately
            # retained and the new fingerprint becomes the project's baseline.
            manifest["settings_fingerprint"] = settings_fingerprint(settings)
        return changed

    def _save(self, manifest: dict[str, Any]) -> None:
        manifest["updated_at"] = _now()
        _atomic_json_write(self._manifest_path(manifest["id"]), manifest)

    @staticmethod
    def _stop_run_clock(manifest: dict[str, Any], *, stopped_at: float | None = None) -> None:
        started_at = manifest.get("current_run_started_at")
        if started_at is not None:
            end = _now() if stopped_at is None else float(stopped_at)
            manifest["generation_elapsed_seconds"] = float(
                manifest.get("generation_elapsed_seconds", 0.0)
            ) + max(0.0, end - float(started_at))
        manifest["current_run_started_at"] = None

    def _repair_interrupted_projects(self) -> None:
        for manifest_path in self.root_dir.glob("*/manifest.json"):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                migrated = self._migrate_model_profile(manifest)
                if manifest.get("state") in {"running", "stopping"}:
                    self._stop_run_clock(
                        manifest,
                        stopped_at=float(manifest.get("updated_at") or _now()),
                    )
                    manifest["state"] = "paused"
                    manifest["message"] = "服务重启后已暂停，可继续生成"
                    manifest["current_segment"] = None
                    manifest["active_segments"] = []
                    for segment in manifest.get("segments", []):
                        if segment.get("status") in {"generating", "encoding"}:
                            segment["status"] = "pending"
                    _atomic_json_write(manifest_path, manifest)
                elif migrated:
                    _atomic_json_write(manifest_path, manifest)
            except Exception:
                continue

    def list_projects(self) -> list[dict[str, Any]]:
        projects: list[dict[str, Any]] = []
        with self._lock:
            for path in self.root_dir.glob("*/manifest.json"):
                try:
                    projects.append(self._public_manifest(json.loads(path.read_text(encoding="utf-8"))))
                except Exception:
                    continue
        projects.sort(key=lambda item: item.get("updated_at", 0), reverse=True)
        return projects

    def get_project(self, project_id: str) -> dict[str, Any]:
        with self._lock:
            return self._public_manifest(self._load(project_id))

    def rename(self, project_id: str, *, name: str) -> dict[str, Any]:
        revised = re.sub(r"\s+", " ", name or "").strip()
        if not revised:
            raise ValueError("书名不能为空")
        if len(revised) > 120:
            raise ValueError("书名不能超过 120 个字符")
        with self._lock:
            manifest = self._load(project_id)
            manifest["name"] = revised
            self._save(manifest)
            return self._public_manifest(manifest)

    def create_project(
        self,
        *,
        name: str,
        filename: str,
        data: bytes,
        settings: dict[str, Any],
        max_chars: int = 200,
    ) -> dict[str, Any]:
        text = extract_document_text(filename, data)
        project_id = uuid.uuid4().hex
        project_dir = self._project_dir(project_id)
        (project_dir / "sources").mkdir(parents=True, exist_ok=True)
        (project_dir / "segments").mkdir(parents=True, exist_ok=True)
        safe_suffix = Path(filename).suffix.lower() or ".txt"
        source_name = f"{uuid.uuid4().hex}{safe_suffix}"
        (project_dir / "sources" / source_name).write_bytes(data)
        chapter_inputs = split_novel_chapters(
            text,
            fallback_title=Path(filename).stem or "正文",
        )
        segments: list[dict[str, Any]] = []
        chapters: list[dict[str, Any]] = []
        char_offset = 0
        for chapter_index, chapter in enumerate(chapter_inputs):
            chapter_chunks = segment_text(chapter["text"], max_chars=max_chars)
            segment_start = len(segments)
            segments.extend(
                self._new_segments(
                    chapter_chunks,
                    start_index=segment_start,
                    char_offset=char_offset,
                    chapter_index=chapter_index,
                )
            )
            char_count = sum(len(chunk) for chunk in chapter_chunks)
            chapters.append(
                {
                    "index": chapter_index,
                    "title": chapter["title"],
                    "segment_start": segment_start,
                    "segment_end": len(segments) - 1,
                    "char_start": char_offset,
                    "char_end": char_offset + char_count,
                }
            )
            char_offset += char_count
        now = _now()
        manifest: dict[str, Any] = {
            "id": project_id,
            "name": (name or Path(filename).stem or "未命名项目").strip()[:120],
            "state": "ready",
            "message": "项目已创建",
            "created_at": now,
            "updated_at": now,
            "settings": settings,
            "settings_fingerprint": settings_fingerprint(settings),
            "max_chars_per_segment": int(max_chars),
            "sources": [{"filename": filename, "stored_name": source_name, "added_at": now, "chars": len(text)}],
            "chapters": chapters,
            "segments": segments,
            "current_segment": None,
            "active_segments": [],
            "completed_audio_seconds": 0.0,
            "generation_wall_seconds": 0.0,
            "generation_elapsed_seconds": 0.0,
            "current_run_started_at": None,
            "final_audio": None,
            "playback": {"segment_index": 0, "offset_seconds": 0.0},
        }
        with self._lock:
            self._save(manifest)
        return self._public_manifest(manifest)

    def append_document(self, project_id: str, *, filename: str, data: bytes) -> dict[str, Any]:
        text = extract_document_text(filename, data)
        with self._lock:
            manifest = self._load(project_id)
            if manifest.get("state") in {"running", "stopping"}:
                raise RuntimeError("请先暂停项目，再追加文档")
            project_dir = self._project_dir(project_id)
            safe_suffix = Path(filename).suffix.lower() or ".txt"
            source_name = f"{uuid.uuid4().hex}{safe_suffix}"
            (project_dir / "sources" / source_name).write_bytes(data)
            existing_chars = sum(len(item["text"]) for item in manifest["segments"])
            chapter_inputs = split_novel_chapters(
                text,
                fallback_title=Path(filename).stem or "追加内容",
            )
            chapters = manifest.setdefault("chapters", [])
            added_chunks = 0
            for chapter_input in chapter_inputs:
                chapter_index = len(chapters)
                chunks = segment_text(
                    chapter_input["text"],
                    max_chars=manifest.get("max_chars_per_segment", 200),
                )
                segment_start = len(manifest["segments"])
                manifest["segments"].extend(
                    self._new_segments(
                        chunks,
                        start_index=segment_start,
                        char_offset=existing_chars,
                        chapter_index=chapter_index,
                    )
                )
                char_count = sum(len(chunk) for chunk in chunks)
                chapters.append(
                    {
                        "index": chapter_index,
                        "title": chapter_input["title"],
                        "segment_start": segment_start,
                        "segment_end": len(manifest["segments"]) - 1,
                        "char_start": existing_chars,
                        "char_end": existing_chars + char_count,
                    }
                )
                existing_chars += char_count
                added_chunks += len(chunks)
            manifest["sources"].append(
                {"filename": filename, "stored_name": source_name, "added_at": _now(), "chars": len(text)}
            )
            manifest["state"] = "paused" if any(s["status"] == "completed" for s in manifest["segments"]) else "ready"
            manifest["message"] = f"已追加 {added_chunks} 个段落"
            manifest["final_audio"] = None
            self._save(manifest)
            return self._public_manifest(manifest)

    def _new_segments(
        self,
        chunks: list[str],
        *,
        start_index: int,
        char_offset: int,
        chapter_index: int = 0,
    ) -> list[dict[str, Any]]:
        segments: list[dict[str, Any]] = []
        cursor = char_offset
        for offset, chunk in enumerate(chunks):
            index = start_index + offset
            segments.append(
                {
                    "index": index,
                    "text": chunk,
                    "char_start": cursor,
                    "char_end": cursor + len(chunk),
                    "chapter_index": chapter_index,
                    "status": "pending",
                    "attempts": 0,
                    "audio_file": None,
                    "duration_seconds": 0.0,
                    "generation_seconds": 0.0,
                    "error": None,
                    "model_profile": None,
                }
            )
            cursor += len(chunk)
        return segments

    def reset_for_settings(self, project_id: str, settings: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            manifest = self._load(project_id)
            if manifest.get("state") in {"running", "stopping"}:
                raise RuntimeError("请先停止项目，再修改生成参数")
            if settings_fingerprint(settings) == manifest.get("settings_fingerprint"):
                return self._public_manifest(manifest)
            segments_dir = self._project_dir(project_id) / "segments"
            if segments_dir.exists():
                for path in segments_dir.iterdir():
                    if path.is_file():
                        path.unlink()
            final_dir = self._project_dir(project_id) / "final"
            if final_dir.exists():
                shutil.rmtree(final_dir)
            for segment in manifest["segments"]:
                segment.update(
                    status="pending",
                    attempts=0,
                    audio_file=None,
                    duration_seconds=0.0,
                    generation_seconds=0.0,
                    error=None,
                )
            manifest.update(
                settings=settings,
                settings_fingerprint=settings_fingerprint(settings),
                state="ready",
                message="生成参数已改变，旧音频已清除",
                current_segment=None,
                active_segments=[],
                completed_audio_seconds=0.0,
                generation_wall_seconds=0.0,
                generation_elapsed_seconds=0.0,
                current_run_started_at=None,
                final_audio=None,
                playback={"segment_index": 0, "offset_seconds": 0.0},
            )
            self._save(manifest)
            return self._public_manifest(manifest)

    def start(
        self,
        project_id: str,
        *,
        settings: dict[str, Any] | None = None,
        segment_indices: list[int] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            manifest = self._load(project_id)
            settings_reset = False
            if settings is not None and settings_fingerprint(settings) != manifest.get("settings_fingerprint"):
                manifest = self._load(self.reset_for_settings(project_id, settings)["id"])
                settings_reset = True
            thread = self._threads.get(project_id)
            if thread is not None and thread.is_alive():
                return self._public_manifest(manifest)
            for segment in manifest["segments"]:
                if segment["status"] in {"failed", "generating", "encoding"}:
                    segment["status"] = "pending"
                    segment["error"] = None
            self._refresh_active_segments(manifest)
            if all(segment["status"] == "completed" for segment in manifest["segments"]) and manifest.get("final_audio"):
                manifest["state"] = "completed"
                self._save(manifest)
                return self._public_manifest(manifest)
            stop_event = threading.Event()
            self._stop_events[project_id] = stop_event
            valid_indices = sorted(
                {
                    int(index)
                    for index in (segment_indices or [])
                    if 0 <= int(index) < len(manifest["segments"])
                }
            )
            manifest["generation_scope"] = valid_indices or None
            manifest["state"] = "running"
            manifest["current_run_started_at"] = _now()
            manifest["message"] = (
                "参数已变化，旧音频已清除，正在双通道生成"
                if settings_reset
                else (
                    f"正在使用 {self.synthesis_workers} 个通道生成所选内容"
                    if valid_indices
                    else f"正在使用 {self.synthesis_workers} 个通道生成整本书"
                )
            )
            self._save(manifest)
            thread = threading.Thread(target=self._run_project, args=(project_id, stop_event), daemon=True)
            self._threads[project_id] = thread
            thread.start()
            return self._public_manifest(manifest)

    def stop(self, project_id: str) -> dict[str, Any]:
        with self._lock:
            manifest = self._load(project_id)
            event = self._stop_events.get(project_id)
            if event is not None:
                event.set()
            if manifest.get("state") == "running":
                manifest["state"] = "stopping"
                manifest["message"] = "将在当前并行段落完成并保存后停止"
                self._save(manifest)
            return self._public_manifest(manifest)

    def stop_all(self) -> list[str]:
        """Request an immediate pause for every active document project."""
        with self._lock:
            active_ids = list(self._stop_events)
            for project_id in active_ids:
                self._stop_events[project_id].set()
                try:
                    manifest = self._load(project_id)
                except FileNotFoundError:
                    continue
                if manifest.get("state") == "running":
                    manifest["state"] = "stopping"
                    manifest["message"] = "正在强制停止全部生成任务"
                    self._save(manifest)
            return active_ids

    def delete(self, project_id: str) -> None:
        with self._lock:
            manifest = self._load(project_id)
            thread = self._threads.get(project_id)
            if manifest.get("state") in {"running", "stopping"} or (thread is not None and thread.is_alive()):
                raise RuntimeError("请先停止项目，再删除")
            project_dir = self._project_dir(project_id)
            shutil.rmtree(project_dir)

    def update_playback(self, project_id: str, *, segment_index: int, offset_seconds: float) -> dict[str, Any]:
        with self._lock:
            manifest = self._load(project_id)
            manifest["playback"] = {
                "segment_index": max(0, int(segment_index)),
                "offset_seconds": max(0.0, float(offset_seconds)),
            }
            self._save(manifest)
            return self._public_manifest(manifest)

    def update_segment_text(self, project_id: str, *, segment_index: int, text: str) -> dict[str, Any]:
        revised = re.sub(r"\s+", " ", text or "").strip()
        if not revised:
            raise ValueError("文字块不能为空")
        if len(revised) > 5000:
            raise ValueError("单个文字块不能超过 5000 字")
        with self._lock:
            manifest = self._load(project_id)
            if manifest.get("state") in {"running", "stopping"}:
                raise RuntimeError("请先停止后台生成，再编辑正文")
            if not 0 <= int(segment_index) < len(manifest["segments"]):
                raise ValueError("文字块不存在")
            segment = manifest["segments"][int(segment_index)]
            if segment.get("text") == revised:
                return self._public_manifest(manifest)
            audio_file = segment.get("audio_file")
            if audio_file:
                try:
                    self.media_path(project_id, str(audio_file)).unlink()
                except (FileNotFoundError, ValueError):
                    pass
            segment.update(
                text=revised,
                status="pending",
                attempts=0,
                audio_file=None,
                duration_seconds=0.0,
                generation_seconds=0.0,
                error=None,
                model_profile=None,
            )
            cursor = 0
            for item in manifest["segments"]:
                item["char_start"] = cursor
                cursor += len(item.get("text", ""))
                item["char_end"] = cursor
            for chapter in manifest.get("chapters", []):
                start = max(0, int(chapter.get("segment_start", 0)))
                end = min(
                    len(manifest["segments"]) - 1,
                    int(chapter.get("segment_end", start)),
                )
                chapter["char_start"] = manifest["segments"][start]["char_start"]
                chapter["char_end"] = manifest["segments"][end]["char_end"]
            final_dir = self._project_dir(project_id) / "final"
            if final_dir.exists():
                shutil.rmtree(final_dir)
            manifest["final_audio"] = None
            manifest["completed_audio_seconds"] = sum(
                float(item.get("duration_seconds", 0.0))
                for item in manifest["segments"]
            )
            manifest["generation_wall_seconds"] = sum(
                float(item.get("generation_seconds", 0.0))
                for item in manifest["segments"]
            )
            manifest["state"] = (
                "paused"
                if any(item.get("status") == "completed" for item in manifest["segments"])
                else "ready"
            )
            manifest["message"] = f"第 {int(segment_index) + 1} 块文字已更新，旧音频已失效"
            self._save(manifest)
            return self._public_manifest(manifest)

    def _run_project(self, project_id: str, stop_event: threading.Event) -> None:
        synthesis_futures: dict[Future[dict[str, Any]], int] = {}
        encode_futures: dict[Future[dict[str, Any]], int] = {}
        fatal_error: tuple[int, BaseException] | None = None
        try:
            with (
                ThreadPoolExecutor(
                    max_workers=self.synthesis_workers,
                    thread_name_prefix="qwen-tts-book",
                ) as synthesizer,
                ThreadPoolExecutor(
                    max_workers=self.synthesis_workers,
                    thread_name_prefix="qwen-tts-aac",
                ) as encoder,
            ):
                while True:
                    for future in [item for item in encode_futures if item.done()]:
                        index = encode_futures.pop(future)
                        try:
                            self._finish_encode(project_id, index, future)
                        except Exception as exc:
                            fatal_error = fatal_error or (index, exc)

                    for future in [item for item in synthesis_futures if item.done()]:
                        index = synthesis_futures.pop(future)
                        try:
                            synthesis = future.result()
                        except Exception as exc:
                            if not stop_event.is_set():
                                fatal_error = fatal_error or (index, exc)
                            continue
                        if fatal_error is not None:
                            self._discard_synthesis(project_id, index, synthesis)
                            continue
                        with self._lock:
                            manifest = self._load(project_id)
                            segment = manifest["segments"][index]
                            segment["status"] = "encoding"
                            self._refresh_active_segments(manifest)
                            manifest["message"] = (
                                f"正在并行生成整本书；第 {index + 1} 段正在转为 AAC"
                            )
                            self._save(manifest)
                        future = encoder.submit(self._encode_segment_aac, project_id, index, synthesis)
                        encode_futures[future] = index

                    if fatal_error is None and not stop_event.is_set():
                        while len(synthesis_futures) < self.synthesis_workers:
                            claimed = self._claim_next_segment(project_id)
                            if claimed is None:
                                break
                            index, segment, settings = claimed
                            future = synthesizer.submit(
                                self._synthesize_segment,
                                project_id,
                                segment,
                                settings,
                            )
                            synthesis_futures[future] = index

                    if synthesis_futures or encode_futures:
                        wait(
                            set(synthesis_futures) | set(encode_futures),
                            timeout=0.25,
                            return_when=FIRST_COMPLETED,
                        )
                        continue

                    if fatal_error is not None:
                        self._fail_segment(project_id, *fatal_error)
                        with self._lock:
                            manifest = self._load(project_id)
                            failed_index = int(fatal_error[0])
                            for segment in manifest["segments"]:
                                if (
                                    int(segment["index"]) != failed_index
                                    and segment["status"] in {"generating", "encoding"}
                                ):
                                    segment["status"] = "pending"
                                    segment["started_at"] = None
                                    segment["error"] = None
                            self._refresh_active_segments(manifest)
                            self._save(manifest)
                        return

                    if stop_event.is_set():
                        with self._lock:
                            manifest = self._load(project_id)
                            for segment in manifest["segments"]:
                                if segment.get("status") in {"generating", "encoding"}:
                                    segment["status"] = "pending"
                                    segment["started_at"] = None
                                    segment["error"] = None
                            self._stop_run_clock(manifest)
                            manifest["state"] = "paused"
                            manifest["message"] = "已暂停，可随时继续"
                            self._refresh_active_segments(manifest)
                            self._save(manifest)
                        return

                    with self._lock:
                        manifest = self._load(project_id)
                        scope = manifest.get("generation_scope")
                        scoped_segments = (
                            [
                                manifest["segments"][int(index)]
                                for index in scope
                                if 0 <= int(index) < len(manifest["segments"])
                            ]
                            if scope
                            else manifest["segments"]
                        )
                        has_pending = any(
                            segment["status"] in {"pending", "failed", "generating", "encoding"}
                            for segment in scoped_segments
                        )
                        if has_pending:
                            continue
                        self._stop_run_clock(manifest)
                        all_completed = all(
                            segment["status"] == "completed"
                            for segment in manifest["segments"]
                        )
                        if all_completed:
                            self._merge_final_audio(manifest)
                            manifest["state"] = "completed"
                            manifest["message"] = "全部段落已完成，整书 M4A 已就绪"
                        else:
                            manifest["state"] = "paused"
                            manifest["message"] = "所选内容已生成高质量 AAC"
                        manifest["generation_scope"] = None
                        self._refresh_active_segments(manifest)
                        self._save(manifest)
                    return
        finally:
            with self._lock:
                self._threads.pop(project_id, None)
                self._stop_events.pop(project_id, None)

    def _refresh_active_segments(self, manifest: dict[str, Any]) -> None:
        active = sorted(
            int(segment["index"])
            for segment in manifest["segments"]
            if segment["status"] in {"generating", "encoding"}
        )
        manifest["active_segments"] = active
        manifest["current_segment"] = active[0] if active else None

    def _claim_next_segment(
        self, project_id: str
    ) -> tuple[int, dict[str, Any], dict[str, Any]] | None:
        with self._lock:
            manifest = self._load(project_id)
            next_segment = next(
                (
                    segment
                    for segment in manifest["segments"]
                    if segment["status"] in {"pending", "failed"}
                    and (
                        not manifest.get("generation_scope")
                        or int(segment["index"]) in manifest["generation_scope"]
                    )
                ),
                None,
            )
            if next_segment is None:
                return None
            index = int(next_segment["index"])
            next_segment["status"] = "generating"
            next_segment["attempts"] = int(next_segment.get("attempts", 0)) + 1
            next_segment["error"] = None
            next_segment["started_at"] = _now()
            self._refresh_active_segments(manifest)
            manifest["current_segment_started_at"] = min(
                float(manifest["segments"][active].get("started_at") or _now())
                for active in manifest["active_segments"]
            )
            manifest["message"] = (
                f"正在并行生成整本书：{len(manifest['active_segments'])}/"
                f"{self.synthesis_workers} 个生成通道工作中"
            )
            self._save(manifest)
            return index, dict(next_segment), dict(manifest["settings"])

    def _fail_segment(self, project_id: str, index: int, exc: BaseException) -> None:
        with self._lock:
            manifest = self._load(project_id)
            segment = manifest["segments"][index]
            segment["status"] = "failed"
            segment["error"] = str(exc)
            self._stop_run_clock(manifest)
            manifest["state"] = "error"
            manifest["message"] = f"第 {index + 1} 段生成失败，可继续重试"
            self._refresh_active_segments(manifest)
            self._save(manifest)

    def _discard_synthesis(self, project_id: str, index: int, synthesis: dict[str, Any]) -> None:
        source_wav = Path(str(synthesis.get("source_wav", "")))
        project_dir = self._project_dir(project_id)
        working_dir = project_dir / "working"
        if source_wav.parent.exists() and working_dir in source_wav.parent.parents:
            shutil.rmtree(source_wav.parent, ignore_errors=True)
        with self._lock:
            manifest = self._load(project_id)
            segment = manifest["segments"][index]
            segment["status"] = "pending"
            if manifest.get("current_segment") == index:
                manifest["current_segment"] = None
            self._save(manifest)

    def _finish_encode(
        self, project_id: str, index: int, future: Future[dict[str, Any]]
    ) -> None:
        try:
            result = future.result()
        except Exception as exc:
            self._fail_segment(project_id, index, exc)
            raise
        with self._lock:
            manifest = self._load(project_id)
            segment = manifest["segments"][index]
            segment.update(result)
            segment["status"] = "completed"
            segment["error"] = None
            manifest["completed_audio_seconds"] = sum(
                float(item.get("duration_seconds", 0.0)) for item in manifest["segments"]
            )
            manifest["generation_wall_seconds"] = sum(
                float(item.get("generation_seconds", 0.0)) for item in manifest["segments"]
            )
            manifest["message"] = f"已完成第 {index + 1}/{len(manifest['segments'])} 段"
            self._refresh_active_segments(manifest)
            self._save(manifest)

    def _synthesize_segment(
        self, project_id: str, segment: dict[str, Any], settings: dict[str, Any]
    ) -> dict[str, Any]:
        project_dir = self._project_dir(project_id)
        working_dir = project_dir / "working"
        working_dir.mkdir(parents=True, exist_ok=True)
        text = segment["text"]
        qwen_profile = str(settings.get("model_profile") or "").startswith("qwen_")
        estimated_frames = (
            max(24, min(int(settings.get("max_new_tokens", 2048)), int(len(text) * 3.2)))
            if qwen_profile
            else max(80, min(int(settings.get("max_new_tokens", 7500)), int(len(text) * 4.2)))
        )
        request = self.request_cls(
            text=text,
            mode="voice_clone",
            prompt_text="",
            prompt_audio_path=settings.get("reference_audio_path") or None,
            language="Chinese",
            tokens_control=False,
            tokens=0,
            max_new_frames=estimated_frames,
            do_sample=True,
            temperature=float(settings.get("temperature", 1.7)),
            top_p=float(settings.get("top_p", 0.8)),
            top_k=int(settings.get("top_k", 25)),
            repetition_penalty=float(settings.get("repetition_penalty", 1.0)),
            seed=None if int(settings.get("seed", 1234)) < 0 else int(settings.get("seed", 1234)),
            codec_chunk_frames=max(1, int(settings.get("codec_chunk_frames", 16))),
            qwen_xvec_only=str(settings.get("qwen_clone_mode") or "xvec") != "icl",
            qwen_reference_text=str(settings.get("qwen_reference_text") or ""),
            qwen_non_streaming_mode=bool(settings.get("qwen_non_streaming_mode", False)),
            qwen_append_silence=bool(settings.get("qwen_append_silence", True)),
            qwen_instruct=str(settings.get("qwen_instruct") or ""),
            qwen_min_new_tokens=max(2, int(settings.get("qwen_min_new_tokens", 2))),
        )
        result_event: dict[str, Any] | None = None
        with self.generation_lock:
            started_at = time.perf_counter()
            model_profile = str(settings.get("model_profile") or "qwen_0_6b")
            with self.runtime_session(model_profile) as runtime:
                for event in self.synthesize_fn(runtime, request, output_dir=working_dir):
                    if event.type == "result":
                        result_event = event.data
            generation_seconds = time.perf_counter() - started_at
        if result_event is None:
            raise RuntimeError("模型没有返回完整音频")
        source_wav = Path(result_event["audio_path"])
        return {
            "source_wav": str(source_wav),
            "duration_seconds": float(result_event.get("metadata", {}).get("duration_seconds", 0.0)),
            "generation_seconds": generation_seconds,
            "model_profile": str(settings.get("model_profile") or "qwen_0_6b"),
            "seed": int(settings.get("seed", 1234)),
            "seed_mode": str(settings.get("seed_mode") or "fixed"),
            "aac_bitrate": str(settings.get("aac_bitrate") or "80k"),
        }

    def _encode_segment_aac(
        self, project_id: str, index: int, synthesis: dict[str, Any]
    ) -> dict[str, Any]:
        project_dir = self._project_dir(project_id)
        source_wav = Path(synthesis["source_wav"])
        output_name = f"{int(index):06d}.m4a"
        output_path = project_dir / "segments" / output_name
        temporary_path = output_path.with_name(output_path.stem + ".tmp.m4a")
        command = [
            self.ffmpeg_path,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source_wav),
            "-c:a",
            "aac",
            "-profile:a",
            "aac_low",
            "-b:a",
            str(synthesis.get("aac_bitrate") or "80k"),
            "-ar",
            "48000",
            "-ac",
            "1",
            "-threads",
            "1",
            str(temporary_path),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=300)
        if completed.returncode != 0 or not temporary_path.exists():
            raise RuntimeError(completed.stderr.strip() or "AAC 转码失败")
        os.replace(temporary_path, output_path)
        run_dir = source_wav.parent
        working_dir = project_dir / "working"
        if run_dir.exists() and working_dir in run_dir.parents:
            shutil.rmtree(run_dir, ignore_errors=True)
        return {
            "audio_file": f"segments/{output_name}",
            "duration_seconds": float(synthesis.get("duration_seconds", 0.0)),
            "generation_seconds": float(synthesis.get("generation_seconds", 0.0)),
            "seed": int(synthesis.get("seed", 1234)),
            "seed_mode": str(synthesis.get("seed_mode") or "fixed"),
        }

    def _merge_final_audio(self, manifest: dict[str, Any]) -> None:
        project_dir = self._project_dir(manifest["id"])
        final_dir = project_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        concat_path = final_dir / "segments.txt"
        lines = []
        for segment in manifest["segments"]:
            audio_path = (project_dir / segment["audio_file"]).resolve()
            escaped = str(audio_path).replace("'", "'\\''")
            lines.append(f"file '{escaped}'")
        concat_path.write_text("\n".join(lines), encoding="utf-8")
        final_path = final_dir / "complete.m4a"
        temporary = final_dir / "complete.tmp.m4a"
        command = [
            self.ffmpeg_path,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_path),
            "-c",
            "copy",
            str(temporary),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=300)
        if completed.returncode != 0 or not temporary.exists():
            raise RuntimeError(completed.stderr.strip() or "最终 AAC 合并失败")
        os.replace(temporary, final_path)
        manifest["final_audio"] = "final/complete.m4a"

    def media_path(self, project_id: str, relative_path: str) -> Path:
        project_dir = self._project_dir(project_id)
        candidate = (project_dir / relative_path).resolve(strict=True)
        if project_dir not in candidate.parents or not candidate.is_file():
            raise ValueError("invalid media path")
        return candidate

    def _public_manifest(self, manifest: dict[str, Any]) -> dict[str, Any]:
        result = json.loads(json.dumps(manifest, ensure_ascii=False))
        result.setdefault("active_segments", [])
        segments = result.get("segments", [])
        if not result.get("chapters"):
            result["chapters"] = [
                {
                    "index": 0,
                    "title": result.get("name") or "正文",
                    "segment_start": 0,
                    "segment_end": max(0, len(segments) - 1),
                    "char_start": 0,
                    "char_end": sum(len(item.get("text", "")) for item in segments),
                }
            ]
            for segment in segments:
                segment.setdefault("chapter_index", 0)
        total_chars = sum(len(item.get("text", "")) for item in segments)
        completed_chars = sum(len(item.get("text", "")) for item in segments if item.get("status") == "completed")
        completed_audio = float(result.get("completed_audio_seconds", 0.0))
        work_seconds = float(result.get("generation_wall_seconds", 0.0))
        elapsed_seconds = float(result.get("generation_elapsed_seconds", 0.0))
        run_started_at = result.get("current_run_started_at")
        if result.get("state") in {"running", "stopping"} and run_started_at is not None:
            elapsed_seconds += max(0.0, _now() - float(run_started_at))
        speed = completed_audio / elapsed_seconds if elapsed_seconds > 0 else 0.0
        audio_per_char = completed_audio / completed_chars if completed_chars > 0 else 0.24
        active_indices = [
            int(index)
            for index in result.get("active_segments", [])
            if 0 <= int(index) < len(segments)
        ]
        in_progress_chars = 0.0
        if result.get("state") in {"running", "stopping"} and active_indices and speed > 0:
            active_lanes = max(1, len(active_indices))
            for index in active_indices:
                segment = segments[index]
                segment_chars = len(segment.get("text", ""))
                expected_wall = max(
                    1.0,
                    segment_chars * audio_per_char * active_lanes / speed,
                )
                segment_elapsed = max(
                    0.0,
                    _now() - float(segment.get("started_at") or _now()),
                )
                in_progress_chars += segment_chars * min(0.95, segment_elapsed / expected_wall)
        progress_chars = completed_chars + in_progress_chars
        progress = progress_chars / total_chars if total_chars > 0 else 0.0
        remaining_chars = max(0.0, total_chars - progress_chars)
        eta = remaining_chars * audio_per_char / speed if speed > 0 else None
        result["stats"] = {
            "total_segments": len(segments),
            "completed_segments": sum(1 for item in segments if item.get("status") == "completed"),
            "total_chars": total_chars,
            "completed_chars": completed_chars,
            "progress": min(1.0, progress),
            "speed_realtime": speed,
            "eta_seconds": eta,
            "completed_audio_seconds": completed_audio,
            "generation_elapsed_seconds": elapsed_seconds,
            "generation_work_seconds": work_seconds,
            "parallel_generations": self.synthesis_workers,
        }
        return result
