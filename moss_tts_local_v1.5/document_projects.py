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
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Callable
from xml.etree import ElementTree


PROJECT_ID_RE = re.compile(r"^[a-f0-9]{32}$")
SUPPORTED_DOCUMENT_SUFFIXES = {".txt", ".md", ".markdown", ".docx"}


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
    ) -> None:
        self.root_dir = Path(root_dir).resolve()
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.runtime_session = runtime_session
        self.synthesize_fn = synthesize_fn
        self.request_cls = request_cls
        self.generation_lock = generation_lock
        self.ffmpeg_path = ffmpeg_path
        self._lock = threading.RLock()
        self._threads: dict[str, threading.Thread] = {}
        self._stop_events: dict[str, threading.Event] = {}
        self._repair_interrupted_projects()

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
        """Bind projects created before multi-model support to the original 4B model."""
        changed = False
        settings = manifest.setdefault("settings", {})
        if not settings.get("model_profile"):
            settings["model_profile"] = "quality_4b"
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

    def _repair_interrupted_projects(self) -> None:
        for manifest_path in self.root_dir.glob("*/manifest.json"):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                migrated = self._migrate_model_profile(manifest)
                if manifest.get("state") in {"running", "stopping"}:
                    manifest["state"] = "paused"
                    manifest["message"] = "服务重启后已暂停，可继续生成"
                    manifest["current_segment"] = None
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
        chunks = segment_text(text, max_chars=max_chars)
        segments = self._new_segments(chunks, start_index=0, char_offset=0)
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
            "segments": segments,
            "current_segment": None,
            "completed_audio_seconds": 0.0,
            "generation_wall_seconds": 0.0,
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
            chunks = segment_text(text, max_chars=manifest.get("max_chars_per_segment", 200))
            manifest["segments"].extend(
                self._new_segments(chunks, start_index=len(manifest["segments"]), char_offset=existing_chars)
            )
            manifest["sources"].append(
                {"filename": filename, "stored_name": source_name, "added_at": _now(), "chars": len(text)}
            )
            manifest["state"] = "paused" if any(s["status"] == "completed" for s in manifest["segments"]) else "ready"
            manifest["message"] = f"已追加 {len(chunks)} 个段落"
            manifest["final_audio"] = None
            self._save(manifest)
            return self._public_manifest(manifest)

    def _new_segments(self, chunks: list[str], *, start_index: int, char_offset: int) -> list[dict[str, Any]]:
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
                completed_audio_seconds=0.0,
                generation_wall_seconds=0.0,
                final_audio=None,
                playback={"segment_index": 0, "offset_seconds": 0.0},
            )
            self._save(manifest)
            return self._public_manifest(manifest)

    def start(self, project_id: str, *, settings: dict[str, Any] | None = None) -> dict[str, Any]:
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
            if all(segment["status"] == "completed" for segment in manifest["segments"]) and manifest.get("final_audio"):
                manifest["state"] = "completed"
                self._save(manifest)
                return self._public_manifest(manifest)
            stop_event = threading.Event()
            self._stop_events[project_id] = stop_event
            manifest["state"] = "running"
            manifest["message"] = (
                "参数已变化，旧音频已清除，正在从头生成" if settings_reset else "正在生成"
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
                manifest["message"] = "将在当前段完成后停止"
                self._save(manifest)
            return self._public_manifest(manifest)

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

    def _run_project(self, project_id: str, stop_event: threading.Event) -> None:
        pending_encode: tuple[int, Future[dict[str, Any]]] | None = None
        try:
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix="moss-tts-aac") as encoder:
                while True:
                    if stop_event.is_set():
                        if pending_encode is not None:
                            self._finish_encode(project_id, *pending_encode)
                            pending_encode = None
                        with self._lock:
                            manifest = self._load(project_id)
                            manifest["state"] = "paused"
                            manifest["message"] = "已暂停，可随时继续"
                            manifest["current_segment"] = None
                            self._save(manifest)
                        return

                    with self._lock:
                        manifest = self._load(project_id)
                        next_segment = next(
                            (s for s in manifest["segments"] if s["status"] in {"pending", "failed", "generating"}),
                            None,
                        )
                        if next_segment is not None:
                            index = int(next_segment["index"])
                            next_segment["status"] = "generating"
                            next_segment["attempts"] = int(next_segment.get("attempts", 0)) + 1
                            next_segment["error"] = None
                            manifest["current_segment"] = index
                            manifest["current_segment_started_at"] = _now()
                            self._save(manifest)
                            settings = dict(manifest["settings"])

                    if next_segment is None:
                        if pending_encode is not None:
                            self._finish_encode(project_id, *pending_encode)
                            pending_encode = None
                            continue
                        with self._lock:
                            manifest = self._load(project_id)
                            self._merge_final_audio(manifest)
                            manifest["state"] = "completed"
                            manifest["message"] = "全部段落已完成"
                            manifest["current_segment"] = None
                            self._save(manifest)
                        return

                    try:
                        synthesis = self._synthesize_segment(project_id, next_segment, settings)
                    except Exception as exc:
                        self._fail_segment(project_id, index, exc)
                        return

                    if pending_encode is not None:
                        try:
                            self._finish_encode(project_id, *pending_encode)
                            pending_encode = None
                        except Exception:
                            self._discard_synthesis(project_id, index, synthesis)
                            return
                    try:
                        with self._lock:
                            manifest = self._load(project_id)
                            segment = manifest["segments"][index]
                            segment["status"] = "encoding"
                            manifest["message"] = f"第 {index + 1} 段正在转为 AAC"
                            self._save(manifest)
                        future = encoder.submit(self._encode_segment_aac, project_id, index, synthesis)
                        pending_encode = (index, future)
                    except Exception as exc:
                        self._fail_segment(project_id, index, exc)
                        return
        finally:
            with self._lock:
                self._threads.pop(project_id, None)
                self._stop_events.pop(project_id, None)

    def _fail_segment(self, project_id: str, index: int, exc: BaseException) -> None:
        with self._lock:
            manifest = self._load(project_id)
            segment = manifest["segments"][index]
            segment["status"] = "failed"
            segment["error"] = str(exc)
            manifest["state"] = "error"
            manifest["message"] = f"第 {index + 1} 段生成失败，可继续重试"
            manifest["current_segment"] = None
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
            if manifest.get("current_segment") == index:
                manifest["current_segment"] = None
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
            model_profile = str(settings.get("model_profile") or "quality_4b")
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
            "model_profile": str(settings.get("model_profile") or "quality_4b"),
            "seed": int(settings.get("seed", 1234)),
            "seed_mode": str(settings.get("seed_mode") or "fixed"),
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
            "192k",
            "-ar",
            "48000",
            "-ac",
            "2",
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
        segments = result.get("segments", [])
        total_chars = sum(len(item.get("text", "")) for item in segments)
        completed_chars = sum(len(item.get("text", "")) for item in segments if item.get("status") == "completed")
        completed_audio = float(result.get("completed_audio_seconds", 0.0))
        wall_seconds = float(result.get("generation_wall_seconds", 0.0))
        speed = completed_audio / wall_seconds if wall_seconds > 0 else 0.0
        current_fraction = 0.0
        current_index = result.get("current_segment")
        if result.get("state") in {"running", "stopping"} and current_index is not None and speed > 0:
            segment = segments[int(current_index)]
            audio_per_char = completed_audio / completed_chars if completed_chars > 0 else 0.24
            expected_wall = max(1.0, len(segment.get("text", "")) * audio_per_char / speed)
            elapsed = max(0.0, _now() - float(result.get("current_segment_started_at", _now())))
            current_fraction = min(0.95, elapsed / expected_wall)
        current_chars = 0
        if current_index is not None and int(current_index) < len(segments):
            current_chars = len(segments[int(current_index)].get("text", ""))
        progress_chars = completed_chars + current_chars * current_fraction
        progress = progress_chars / total_chars if total_chars > 0 else 0.0
        remaining_chars = max(0.0, total_chars - progress_chars)
        audio_per_char = completed_audio / completed_chars if completed_chars > 0 else 0.24
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
        }
        return result
