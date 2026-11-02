from __future__ import annotations

import atexit
import bisect
import copy
import hashlib
import json
import logging
import math
import os
import re
import secrets
import shutil
import threading
import time
import uuid
import weakref
import zipfile
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import AbstractContextManager, contextmanager
from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterator
from xml.etree import ElementTree

from media_process import run_media_process
from text_processing import detect_language, normalize_language, prepare_tts_text, strip_markdown

LOG = logging.getLogger(__name__)

PROJECT_ID_RE = re.compile(r"^[a-f0-9]{32}$")
SUPPORTED_DOCUMENT_SUFFIXES = {".txt", ".md", ".markdown", ".docx"}
MARKDOWN_SUFFIXES = {".md", ".markdown"}
MAX_DOCUMENT_BYTES = 30 * 1024 * 1024
MAX_DOCUMENT_CHARS = 5_000_000
DEFAULT_MAX_CHARS = 160
MIN_MAX_CHARS = 40
MAX_MAX_CHARS = 300
TINY_CHUNK_CHARS = 12
# Segment status churn while a run is active is persisted at most this often.
MANIFEST_COALESCE_SECONDS = 1.5
QWEN_FRAMES_PER_SECOND = 12.5
TRUNCATION_RATIO = 0.98
FINAL_SAMPLE_RATE = 24000
FINAL_AAC_BITRATE = "96k"
MERGE_BATCH_SIZE = 200
CHAPTER_HEADING_RE = re.compile(
    r"^(?:"
    r"第[零〇一二三四五六七八九十百千万两0-9]{1,12}[章节回篇]"
    r"|Chapter\s+\d+"
    r"|序章|序言|前言|楔子|引子|尾声|后记|番外(?:篇|章)?"
    r")(?:[\s：:、.．\-—].*)?$",
    re.IGNORECASE,
)
# Volume headings group chapters ("第一卷 风起", "卷二", "Book 2", "Part III").
VOLUME_HEADING_RE = re.compile(
    r"^(?:"
    r"第[零〇一二三四五六七八九十百千万两0-9]{1,8}[卷部集]"
    r"|[卷部][零〇一二三四五六七八九十百千两0-9]{1,6}"
    r"|(?:Volume|Book|Part)\s+(?:\d+|[IVXLC]+)"
    r")(?:[\s：:、.．\-—].*)?$",
    re.IGNORECASE,
)
MAX_SERIES_CHARS = 80


def _now() -> float:
    return time.time()


def _dump_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _atomic_write_bytes(path: Path, payload: bytes, *, fsync: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:12]}.tmp")
    try:
        with open(temporary, "wb") as handle:
            handle.write(payload)
            if fsync:
                handle.flush()
                os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _atomic_json_write(path: Path, value: dict[str, Any], *, fsync: bool = True) -> None:
    _atomic_write_bytes(path, _dump_json(value).encode("utf-8"), fsync=fsync)


# ---------------------------------------------------------------------------
# Document import
# ---------------------------------------------------------------------------


def _decode_text_document(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("无法识别文本编码，请转换为 UTF-8 后重试")


def _extract_docx_text(data: bytes) -> str:
    from io import BytesIO

    try:
        with zipfile.ZipFile(BytesIO(data)) as archive:
            entry = archive.getinfo("word/document.xml")
            if entry.file_size > MAX_DOCUMENT_BYTES:
                raise ValueError("DOCX 正文解压后不能超过 30 MB")
            with archive.open(entry) as source:
                xml_data = source.read(MAX_DOCUMENT_BYTES + 1)
            if len(xml_data) > MAX_DOCUMENT_BYTES:
                raise ValueError("DOCX 正文解压后不能超过 30 MB")
        # ElementTree expands internal entities; DOCX has no need for DTDs.
        declarations = xml_data.replace(b"\0", b"").upper()
        if b"<!DOCTYPE" in declarations or b"<!ENTITY" in declarations:
            raise ValueError("DOCX 正文不支持 DTD 或实体声明")
        root = ElementTree.fromstring(xml_data)
    except (zipfile.BadZipFile, KeyError, ElementTree.ParseError, RuntimeError, OSError) as exc:
        raise ValueError("DOCX 文件损坏或不包含有效正文") from exc
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
    if len(data) > MAX_DOCUMENT_BYTES:
        raise ValueError("文档不能超过 30 MB")
    text = _extract_docx_text(data) if suffix == ".docx" else _decode_text_document(data)
    if len(text) > MAX_DOCUMENT_CHARS:
        raise ValueError("单次导入正文不能超过 500 万字")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if suffix in MARKDOWN_SUFFIXES:
        text = strip_markdown(text)
    text = re.sub(r"[\t ]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "\n".join(line.strip() for line in text.splitlines()).strip()
    if not text:
        raise ValueError("文档中没有可生成的文本")
    return text


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

_CLOSERS = "”」』）)\"'’》】]"
_SENTENCE_END_RE = re.compile(r"([。！？!?；;…]+|\.+)([" + re.escape(_CLOSERS) + r"]*)")
_SOFT_BREAK_RE = re.compile(r"——|[，,、：:]")
_ABBREVIATIONS = {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "no", "mt", "fig", "e.g", "i.e"}
_TRAILING_WORD_RE = re.compile(r"([A-Za-z](?:[A-Za-z.]*[A-Za-z])?)$")


def clamp_max_chars(max_chars: Any) -> int:
    try:
        value = int(max_chars)
    except (TypeError, ValueError):
        value = DEFAULT_MAX_CHARS
    return max(MIN_MAX_CHARS, min(MAX_MAX_CHARS, value))


def _is_ascii_alnum(char: str) -> bool:
    return bool(char) and char.isascii() and char.isalnum()


def split_sentences(paragraph: str) -> list[str]:
    """Split one paragraph into sentences, keeping the original characters.

    Boundaries follow CJK/ASCII terminators (。！？!?；;…) and ASCII periods
    followed by whitespace. Closing quotes/brackets stay with the sentence.
    """
    pieces: list[str] = []
    start = 0
    for match in _SENTENCE_END_RE.finditer(paragraph):
        end = match.end()
        terminator = match.group(1)
        if terminator.startswith("."):
            following = paragraph[end : end + 1]
            if following and not following.isspace():
                continue
            if terminator == ".":
                word = _TRAILING_WORD_RE.search(paragraph[start : match.start()])
                if word is not None:
                    token = word.group(1)
                    if token.lower() in _ABBREVIATIONS or (len(token) == 1 and token.isupper()):
                        continue
        pieces.append(paragraph[start:end])
        start = end
    if start < len(paragraph):
        pieces.append(paragraph[start:])
    return [piece for piece in pieces if piece.strip()]


def _cut_position(text: str, limit: int) -> int:
    """Best cut index (exclusive) <= limit for a too-long sentence."""
    lowest = max(1, int(limit * 0.4))
    best = -1
    for match in _SOFT_BREAK_RE.finditer(text, 0, limit):
        if match.end() >= lowest:
            best = match.end()
    if best > 0:
        while best < len(text) and best < limit and text[best] in _CLOSERS:
            best += 1
        return best
    for index in range(min(limit, len(text) - 1), lowest - 1, -1):
        if text[index].isspace():
            return index
    for index in range(min(limit, len(text) - 1), 0, -1):
        if not (_is_ascii_alnum(text[index - 1]) and _is_ascii_alnum(text[index])):
            return index
    return limit


def _join_inline(left: str, right: str) -> str:
    if left and right and left[-1].isascii() and right[0].isascii() and not left[-1].isspace():
        return left + " " + right
    return left + right


def _pack_paragraph(paragraph: str, max_chars: int) -> list[str]:
    chunks: list[str] = []
    current = ""
    for piece in split_sentences(paragraph):
        if current and len((current + piece).strip()) <= max_chars:
            current += piece
            continue
        if current.strip():
            chunks.append(current.strip())
        current = piece.lstrip()
        while len(current.rstrip()) > max_chars:
            cut = _cut_position(current, max_chars)
            head = current[:cut].strip()
            if head:
                chunks.append(head)
            current = current[cut:].lstrip()
    if current.strip():
        chunks.append(current.strip())
    # Merge a tiny trailing chunk into the previous one if it stays reasonable.
    while (
        len(chunks) >= 2
        and len(chunks[-1]) < TINY_CHUNK_CHARS
        and len(_join_inline(chunks[-2], chunks[-1])) <= int(max_chars * 1.2)
    ):
        tail = chunks.pop()
        chunks[-1] = _join_inline(chunks[-1], tail)
    return chunks


def segment_text_blocks(text: str, max_chars: int = DEFAULT_MAX_CHARS) -> list[dict[str, str]]:
    """Segment one chapter into synthesis blocks.

    Each block is ``{"text", "pause_after"}`` where ``pause_after`` is
    ``"sentence"``, ``"paragraph"`` or ``"chapter"`` (last block). Paragraph
    boundaries are never erased: consecutive short paragraphs may share one
    block but are joined with "\\n" (those joiners count toward block length
    and toward the cumulative char offsets stored on segments).
    """
    max_chars = clamp_max_chars(max_chars)
    normalized = re.sub(r"[ \t]+", " ", text or "").strip()
    paragraphs = [part.strip() for part in normalized.split("\n") if part.strip()]
    blocks: list[dict[str, str]] = []
    for paragraph in paragraphs:
        chunks = _pack_paragraph(paragraph, max_chars)
        if not chunks:
            continue
        previous = blocks[-1] if blocks else None
        if (
            previous is not None
            and previous["pause_after"] == "paragraph"
            and len(previous["text"]) + 1 + len(chunks[0]) <= max_chars
            and (len(chunks) == 1 or len(previous["text"]) <= max_chars // 4)
        ):
            previous["text"] = previous["text"] + "\n" + chunks[0]
            previous["pause_after"] = "paragraph" if len(chunks) == 1 else "sentence"
            chunks = chunks[1:]
        for position, chunk in enumerate(chunks):
            blocks.append(
                {"text": chunk, "pause_after": "paragraph" if position == len(chunks) - 1 else "sentence"}
            )
    # A tiny paragraph that could not join (e.g. after a long paragraph).
    merged: list[dict[str, str]] = []
    for block in blocks:
        if (
            merged
            and len(block["text"]) < TINY_CHUNK_CHARS
            and len(merged[-1]["text"]) + 1 + len(block["text"]) <= int(max_chars * 1.2)
        ):
            joiner = "\n" if merged[-1]["pause_after"] == "paragraph" else ""
            merged[-1]["text"] = merged[-1]["text"] + joiner + block["text"]
            merged[-1]["pause_after"] = block["pause_after"]
            continue
        merged.append(block)
    if merged:
        merged[-1]["pause_after"] = "chapter"
    return merged


def segment_text(text: str, max_chars: int = DEFAULT_MAX_CHARS) -> list[str]:
    return [block["text"] for block in segment_text_blocks(text, max_chars)]


def split_novel_chapters(text: str, *, fallback_title: str = "正文") -> list[dict[str, str]]:
    """Split common Chinese/English novel headings without losing spoken text.

    Each chapter is ``{"title", "text"}`` plus ``"volume"`` (the enclosing
    volume heading) when the text has volume headings such as "第一卷". A
    volume heading directly followed by a chapter heading is read as part of
    that chapter; text between them becomes a chapter named after the volume.
    """
    chapters: list[dict[str, str]] = []
    current_title = fallback_title
    current_lines: list[str] = []
    current_volume = ""
    # True while current_lines holds only a volume heading (no body yet).
    volume_only = False
    found_heading = False

    def flush() -> None:
        body = "\n".join(current_lines).strip()
        if body:
            chapter = {"title": current_title, "text": body}
            if current_volume:
                chapter["volume"] = current_volume
            chapters.append(chapter)

    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        short = bool(line) and len(line) <= 80
        if short and VOLUME_HEADING_RE.fullmatch(line):
            flush()
            current_volume = line
            current_title = line
            current_lines = [line]
            volume_only = True
            found_heading = True
        elif short and CHAPTER_HEADING_RE.fullmatch(line):
            if volume_only:
                current_lines.append(line)
            else:
                flush()
                current_lines = [line]
            current_title = line
            volume_only = False
            found_heading = True
        else:
            if line:
                volume_only = False
            current_lines.append(raw_line)
    flush()
    if not chapters:
        return [{"title": fallback_title, "text": (text or "").strip()}]
    if found_heading and chapters[0]["title"] == fallback_title:
        chapters[0]["title"] = "序章"
    return chapters


def normalize_series(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:MAX_SERIES_CHARS]


# Settings that change how blocks sound; a change invalidates generated audio.
# ``aac_bitrate`` is deliberately absent: segments are decoded and re-encoded
# when the book is merged, so a new bitrate only needs a re-merge.
VOICE_SETTING_KEYS = (
    "model_profile",
    "reference_audio_path",
    "qwen_clone_mode",
    "qwen_reference_text",
    "configured_seed",
    "temperature",
    "top_p",
    "top_k",
    "repetition_penalty",
    "qwen_append_silence",
    "language",
)


def voice_settings_differ(old: dict[str, Any] | None, new: dict[str, Any]) -> bool:
    old = old or {}

    def key_value(settings: dict[str, Any], key: str) -> Any:
        if key == "configured_seed":
            return int(settings.get("configured_seed", settings.get("seed", 1234)) or 0)
        value = settings.get(key)
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return round(float(value), 6)
        return value

    return any(key_value(old, key) != key_value(new, key) for key in VOICE_SETTING_KEYS)


def settings_fingerprint(settings: dict[str, Any]) -> str:
    payload = json.dumps(settings, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def qwen_frame_budget(text: str, configured_max: int) -> int:
    """Codec frames for one block: 0.5 s per character plus 4 s, capped."""
    return max(24, min(int(configured_max), int(len(text) * 6.25) + 50))


_AAC_ENCODER_CACHE: dict[str, str] = {}
_AAC_ENCODER_LOCK = threading.Lock()


def preferred_aac_encoder(ffmpeg_path: str) -> str:
    """``aac_at`` (macOS AudioToolbox) when ffmpeg offers it, else ``aac``."""
    with _AAC_ENCODER_LOCK:
        cached = _AAC_ENCODER_CACHE.get(ffmpeg_path)
        if cached:
            return cached
    encoder = "aac"
    try:
        completed = run_media_process(
            [ffmpeg_path, "-hide_banner", "-encoders"], cancelled=lambda: False, timeout=10
        )
        if re.search(r"^\s*A\S*\s+aac_at\s", completed.stdout or "", re.MULTILINE):
            encoder = "aac_at"
    except Exception:  # pragma: no cover - ffmpeg missing is reported at encode time
        encoder = "aac"
    with _AAC_ENCODER_LOCK:
        _AAC_ENCODER_CACHE[ffmpeg_path] = encoder
    return encoder


def demote_aac_encoder(ffmpeg_path: str) -> None:
    """Fall back to the built-in ``aac`` encoder after ``aac_at`` failed to open.

    AudioToolbox can be listed by ``ffmpeg -encoders`` yet refuse to initialise
    (sandboxed processes, missing system services). One failure switches the
    process to the portable encoder instead of failing every block.
    """
    with _AAC_ENCODER_LOCK:
        _AAC_ENCODER_CACHE[ffmpeg_path] = "aac"


def run_aac_encode(
    ffmpeg_path: str,
    build_command: Callable[[str], list[str]],
    output_path: Path,
    *,
    timeout: float,
    cancelled: Callable[[], bool],
    failure_message: str,
) -> str:
    """Run an AAC encode, retrying once with ``aac`` if ``aac_at`` fails. Returns the encoder."""
    encoder = preferred_aac_encoder(ffmpeg_path)
    completed = run_media_process(build_command(encoder), timeout=timeout, cancelled=cancelled)
    if (completed.returncode != 0 or not output_path.exists()) and encoder == "aac_at" and not cancelled():
        LOG.warning("aac_at encoder failed (%s); falling back to aac", (completed.stderr or "").strip()[:200])
        demote_aac_encoder(ffmpeg_path)
        encoder = "aac"
        output_path.unlink(missing_ok=True)
        completed = run_media_process(build_command(encoder), timeout=timeout, cancelled=cancelled)
    if completed.returncode != 0 or not output_path.exists():
        raise RuntimeError((completed.stderr or "").strip() or failure_message)
    return encoder


def _aac_codec_args(encoder: str, bitrate: str) -> list[str]:
    args = ["-c:a", encoder, "-b:a", str(bitrate)]
    if encoder == "aac":
        args[2:2] = ["-profile:a", "aac_low"]
    return args


# ---------------------------------------------------------------------------
# Manifest store
# ---------------------------------------------------------------------------


def _attach_volumes(
    chapter_inputs: list[dict[str, Any]],
    volumes: list[dict[str, Any]],
    *,
    wrapper_title: str = "",
) -> None:
    """Give chapters a ``volume_index`` and append the new volumes in place.

    Chapters keep the volume heading found in their text; chapters without one
    go into ``wrapper_title`` (an appended file) or stay ungrouped.
    """
    previous_key: str | None = None
    for chapter in chapter_inputs:
        title = str(chapter.get("volume") or wrapper_title or "")
        if not title:
            previous_key = None
            continue
        if title != previous_key:
            volumes.append({"index": len(volumes), "title": title[:120]})
            previous_key = title
        chapter["volume_index"] = len(volumes) - 1


class SettingsResetRequired(RuntimeError):
    """Changing the voice would discard ``completed`` generated blocks."""

    def __init__(self, completed: int) -> None:
        super().__init__(f"更换朗读设置会清除已生成的 {completed} 个文字块音频")
        self.completed = completed


class _ProjectEntry:
    __slots__ = (
        "project_id",
        "manifest",
        "lock",
        "write_lock",
        "dirty",
        "last_write",
        "write_seq",
        "written_seq",
        "deleted",
        "claim_hint",
    )

    def __init__(self, project_id: str, manifest: dict[str, Any]) -> None:
        self.project_id = project_id
        self.manifest = manifest
        self.lock = threading.RLock()
        self.write_lock = threading.Lock()
        self.dirty = False
        self.last_write = 0.0
        self.write_seq = 0
        self.written_seq = 0
        self.deleted = False
        self.claim_hint = 0


def _flush_manager_at_exit(reference: "weakref.ReferenceType[DocumentProjectManager]") -> None:
    manager = reference()
    if manager is not None:
        try:
            manager.flush_all()
        except Exception:  # pragma: no cover
            pass


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
        # Global lock: thread/stop bookkeeping only. Lock order is always
        # ``self._lock`` -> entry.lock -> entry.write_lock (never reversed).
        self._lock = threading.RLock()
        self._cache_lock = threading.Lock()
        self._entries: dict[str, _ProjectEntry] = {}
        self._force_stop_epoch = 0
        self._threads: dict[str, threading.Thread] = {}
        self._stop_events: dict[str, threading.Event] = {}
        self._closed = False
        self._flush_wakeup = threading.Event()
        self._flusher: threading.Thread | None = None
        self.coalesce_seconds = MANIFEST_COALESCE_SECONDS
        self._repair_interrupted_projects()
        atexit.register(_flush_manager_at_exit, weakref.ref(self))

    def configure_synthesis_workers(self, workers: int) -> int:
        with self._lock:
            self.synthesis_workers = max(1, min(2, int(workers)))
            return self.synthesis_workers

    # -- paths --------------------------------------------------------------

    def _project_dir(self, project_id: str) -> Path:
        if not PROJECT_ID_RE.fullmatch(project_id or ""):
            raise ValueError("invalid project id")
        path = (self.root_dir / project_id).resolve()
        if self.root_dir not in path.parents:
            raise ValueError("invalid project path")
        return path

    def _manifest_path(self, project_id: str) -> Path:
        return self._project_dir(project_id) / "manifest.json"

    def _playback_path(self, project_id: str) -> Path:
        return self._project_dir(project_id) / "playback.json"

    # -- cache --------------------------------------------------------------

    def _read_manifest_from_disk(self, project_id: str) -> tuple[dict[str, Any], bool]:
        path = self._manifest_path(project_id)
        if not path.exists():
            raise FileNotFoundError(project_id)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        changed = self._migrate_model_profile(manifest)
        changed = self._migrate_revisions(manifest) or changed
        playback_path = self._playback_path(project_id)
        if playback_path.exists():
            try:
                playback = json.loads(playback_path.read_text(encoding="utf-8"))
                if isinstance(playback, dict):
                    manifest["playback"] = playback
            except (OSError, ValueError):
                pass
        return manifest, changed

    def _entry(self, project_id: str) -> _ProjectEntry:
        self._project_dir(project_id)  # validates the id
        with self._cache_lock:
            entry = self._entries.get(project_id)
        if entry is not None:
            if entry.deleted:
                raise FileNotFoundError(project_id)
            return entry
        manifest, changed = self._read_manifest_from_disk(project_id)
        with self._cache_lock:
            entry = self._entries.get(project_id)
            if entry is None:
                entry = _ProjectEntry(project_id, manifest)
                entry.dirty = changed
                self._entries[project_id] = entry
        if entry.deleted:
            raise FileNotFoundError(project_id)
        return entry

    @contextmanager
    def _locked(self, project_id: str) -> Iterator[dict[str, Any]]:
        entry = self._entry(project_id)
        with entry.lock:
            if entry.deleted:
                raise FileNotFoundError(project_id)
            yield entry.manifest

    def _load(self, project_id: str) -> dict[str, Any]:
        """Return the live cached manifest; mutate only while holding its lock."""
        return self._entry(project_id).manifest

    @staticmethod
    def _migrate_revisions(manifest: dict[str, Any]) -> bool:
        changed = False
        if not isinstance(manifest.get("revision"), int):
            manifest["revision"] = 1
            changed = True
        if not isinstance(manifest.get("text_revision"), int):
            manifest["text_revision"] = 1
            changed = True
        for segment in manifest.get("segments", []):
            if "rev" not in segment:
                segment["rev"] = manifest["revision"]
                changed = True
        return changed

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

    def _save(
        self,
        manifest: dict[str, Any],
        *,
        segments: Any = (),
        all_segments: bool = False,
        text_changed: bool = False,
        coalesce: bool = False,
    ) -> None:
        """Record a change (revision bump) and persist it.

        ``coalesce=True`` is for segment status churn during a run: the write
        is deferred so the manifest hits the disk at most every
        ``coalesce_seconds``. Every other save is written immediately (and
        carries any deferred changes with it).
        """
        project_id = manifest["id"]
        with self._cache_lock:
            entry = self._entries.get(project_id)
            if entry is None or entry.manifest is not manifest:
                if entry is None:
                    entry = _ProjectEntry(project_id, manifest)
                    self._entries[project_id] = entry
                else:
                    entry.manifest = manifest
        with entry.lock:
            revision = int(manifest.get("revision") or 0) + 1
            manifest["revision"] = revision
            manifest.setdefault("text_revision", 1)
            if text_changed:
                manifest["text_revision"] = int(manifest.get("text_revision") or 0) + 1
            items = manifest.get("segments", [])
            if all_segments or text_changed:
                for segment in items:
                    segment["rev"] = revision
            else:
                for index in segments:
                    if 0 <= int(index) < len(items):
                        items[int(index)]["rev"] = revision
            manifest["updated_at"] = _now()
            entry.dirty = True
            if coalesce and _now() - entry.last_write < self.coalesce_seconds:
                self._ensure_flusher()
                return
        self._write_entry(entry)

    def _write_entry(self, entry: _ProjectEntry) -> None:
        with entry.lock:
            if not entry.dirty or entry.deleted:
                return
            payload = _dump_json(entry.manifest).encode("utf-8")
            entry.write_seq += 1
            sequence = entry.write_seq
            entry.dirty = False
            entry.last_write = _now()
            path = self._manifest_path(entry.project_id)
        failure: BaseException | None = None
        with entry.write_lock:
            if entry.deleted or sequence <= entry.written_seq:
                return
            try:
                _atomic_write_bytes(path, payload)
                entry.written_seq = sequence
            except BaseException as exc:  # never take entry.lock under write_lock
                failure = exc
        if failure is not None:
            with entry.lock:
                entry.dirty = True
            raise failure

    def _ensure_flusher(self) -> None:
        if self._flusher is not None and self._flusher.is_alive():
            return
        with self._cache_lock:
            if self._flusher is not None and self._flusher.is_alive():
                return
            self._flusher = threading.Thread(
                target=self._flush_loop, name="document-manifest-flush", daemon=True
            )
            self._flusher.start()

    def _flush_loop(self) -> None:
        while not self._closed:
            self._flush_wakeup.wait(0.25)
            self._flush_wakeup.clear()
            self._flush_due()

    def _flush_due(self, *, force: bool = False) -> None:
        with self._cache_lock:
            entries = list(self._entries.values())
        now = _now()
        for entry in entries:
            if entry.dirty and (force or now - entry.last_write >= self.coalesce_seconds):
                try:
                    self._write_entry(entry)
                except Exception:  # pragma: no cover - disk errors retried next tick
                    LOG.exception("failed to persist document manifest %s", entry.project_id)

    def flush(self, project_id: str) -> None:
        with self._cache_lock:
            entry = self._entries.get(project_id)
        if entry is not None:
            self._write_entry(entry)

    def flush_all(self) -> None:
        self._flush_due(force=True)

    def close(self) -> None:
        """Stop background work and persist every pending manifest change."""
        self.stop_all()
        self._closed = True
        self._flush_wakeup.set()
        self.flush_all()

    def _store_playback(self, project_id: str, manifest: dict[str, Any], playback: dict[str, Any]) -> None:
        manifest["playback"] = playback
        _atomic_json_write(self._playback_path(project_id), playback, fsync=False)

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
        """Load every manifest into the cache once, pausing interrupted runs."""
        for manifest_path in self.root_dir.glob("*/manifest.json"):
            project_id = manifest_path.parent.name
            if not PROJECT_ID_RE.fullmatch(project_id):
                continue
            try:
                manifest, migrated = self._read_manifest_from_disk(project_id)
                interrupted = manifest.get("state") in {"running", "stopping"}
                if interrupted:
                    self._stop_run_clock(
                        manifest,
                        stopped_at=float(manifest.get("updated_at") or _now()),
                    )
                    manifest["state"] = "paused"
                    manifest["message"] = "服务重启后已暂停，可继续生成"
                    manifest["current_segment"] = None
                    manifest["active_segments"] = []
                    changed = []
                    for segment in manifest.get("segments", []):
                        if segment.get("status") in {"generating", "encoding"}:
                            segment["status"] = "pending"
                            changed.append(int(segment["index"]))
                entry = _ProjectEntry(project_id, manifest)
                with self._cache_lock:
                    self._entries[project_id] = entry
                if interrupted:
                    self._save(manifest, segments=changed)
                elif migrated:
                    entry.dirty = True
                    self._write_entry(entry)
            except Exception:
                LOG.exception("could not load document project %s", project_id)
                continue

    # -- public views -------------------------------------------------------

    def _all_entries(self) -> list[_ProjectEntry]:
        with self._cache_lock:
            return [entry for entry in self._entries.values() if not entry.deleted]

    def list_projects(self) -> list[dict[str, Any]]:
        """Cheap summaries (no segments/chapters) served from the cache."""
        projects: list[dict[str, Any]] = []
        for entry in self._all_entries():
            with entry.lock:
                if entry.deleted:
                    continue
                projects.append(self._summary(entry.manifest))
        projects.sort(key=lambda item: item.get("updated_at", 0), reverse=True)
        return projects

    def reference_usage(self, path: str | Path) -> list[str]:
        """Return persisted book projects that still depend on a reference file."""
        return self.reference_usage_index().get(str(Path(path).resolve()), [])

    def reference_usage_index(self) -> dict[str, list[str]]:
        usages: dict[str, list[str]] = {}
        for entry in self._all_entries():
            with entry.lock:
                manifest = entry.manifest
                settings = manifest.get("settings")
                raw = settings.get("reference_audio_path") if isinstance(settings, dict) else ""
                name = manifest.get("name") or "未命名项目"
            if not raw:
                continue
            try:
                reference_path = str(Path(str(raw)).resolve())
            except (OSError, ValueError):
                continue
            usages.setdefault(reference_path, []).append(f"书籍《{name}》")
        return {path: list(dict.fromkeys(labels)) for path, labels in usages.items()}

    def get_project(self, project_id: str) -> dict[str, Any]:
        with self._locked(project_id) as manifest:
            return self._public_manifest(manifest)

    def get_summary(self, project_id: str) -> dict[str, Any]:
        with self._locked(project_id) as manifest:
            return self._summary(manifest)

    def get_progress(self, project_id: str, since: int = 0) -> dict[str, Any]:
        """Incremental state: segments changed after ``since`` (without text)."""
        since = int(since or 0)
        with self._locked(project_id) as manifest:
            segments = manifest.get("segments", [])
            if since > int(manifest.get("revision") or 0):
                # The client saw revisions this process never persisted
                # (coalesced writes lost in a crash): resync everything.
                since = 0
            changed = [
                {key: value for key, value in segment.items() if key != "text"}
                for segment in segments
                if since <= 0 or int(segment.get("rev") or 0) > since
            ]
            return {
                "id": manifest["id"],
                "revision": int(manifest.get("revision") or 0),
                "text_revision": int(manifest.get("text_revision") or 0),
                "state": manifest.get("state"),
                "message": manifest.get("message"),
                "stats": self._stats(manifest),
                "active_segments": list(manifest.get("active_segments") or []),
                "current_segment": manifest.get("current_segment"),
                "playback": copy.deepcopy(manifest.get("playback")),
                "final_audio": manifest.get("final_audio"),
                "updated_at": manifest.get("updated_at"),
                "segments": changed,
            }

    def rename(self, project_id: str, *, name: str) -> dict[str, Any]:
        return self.update_metadata(project_id, name=name)

    def update_metadata(
        self,
        project_id: str,
        *,
        name: str | None = None,
        series: str | None = None,
        volumes: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Rename a book, file it under a series, or retitle its volumes."""
        revised = None
        if name is not None:
            revised = re.sub(r"\s+", " ", name or "").strip()
            if not revised:
                raise ValueError("书名不能为空")
            if len(revised) > 120:
                raise ValueError("书名不能超过 120 个字符")
        with self._locked(project_id) as manifest:
            if revised is not None:
                manifest["name"] = revised
            if series is not None:
                manifest["series"] = normalize_series(series)
            if volumes:
                existing = manifest.get("volumes") or []
                for item in volumes:
                    try:
                        index = int(item.get("index"))
                    except (TypeError, ValueError, AttributeError):
                        continue
                    title = re.sub(r"\s+", " ", str(item.get("title") or "")).strip()[:120]
                    if 0 <= index < len(existing) and title:
                        existing[index]["title"] = title
            self._save(manifest)
            return self._public_manifest(manifest)

    def update_settings(
        self,
        project_id: str,
        settings: dict[str, Any],
        *,
        confirm_reset: bool = False,
        voice_label: str = "",
    ) -> dict[str, Any]:
        """Change the book's own voice. Generated audio is only discarded when
        the voice really changes and the caller confirmed it."""
        with self._locked(project_id) as manifest:
            if manifest.get("state") in {"running", "stopping"}:
                raise RuntimeError("请先停止生成，再修改本书的朗读设置")
            current = manifest.get("settings") or {}
            completed = sum(1 for segment in manifest.get("segments", []) if segment.get("status") == "completed")
            label = str(voice_label or settings.get("voice_name") or "")[:120]
            if not voice_settings_differ(current, settings):
                changed = False
                if str(current.get("aac_bitrate") or "") != str(settings.get("aac_bitrate") or ""):
                    current["aac_bitrate"] = settings.get("aac_bitrate")
                    if manifest.get("final_audio"):
                        manifest["final_audio"] = None
                        manifest["state"] = "paused"
                        manifest["message"] = "码率已更改：点击“生成整本书”即可按新码率重新合并，无需重新合成"
                    changed = True
                for key in ("max_new_tokens", "codec_chunk_frames", "qwen_min_new_tokens", "voice_name"):
                    if key in settings and current.get(key) != settings[key]:
                        current[key] = settings[key]
                        changed = True
                if label and manifest.get("voice_label") != label:
                    manifest["voice_label"] = label
                    changed = True
                if changed:
                    manifest["settings"] = current
                    manifest["settings_fingerprint"] = settings_fingerprint(current)
                    self._save(manifest)
                return self._public_manifest(manifest) | {"audio_reset": False}
            if completed and not confirm_reset:
                raise SettingsResetRequired(completed)
            manifest["voice_label"] = label
        result = self.reset_for_settings(project_id, settings)
        return result | {"audio_reset": bool(completed)}

    def _build_segments(
        self,
        chapter_inputs: list[dict[str, str]],
        *,
        max_chars: int,
        start_index: int,
        start_chapter: int,
        char_offset: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
        segments: list[dict[str, Any]] = []
        chapters: list[dict[str, Any]] = []
        for offset, chapter in enumerate(chapter_inputs):
            chapter_index = start_chapter + offset
            blocks = segment_text_blocks(chapter["text"], max_chars=max_chars) or [
                {"text": chapter["text"], "pause_after": "chapter"}
            ]
            segment_start = start_index + len(segments)
            segments.extend(
                self._new_segments(
                    blocks,
                    start_index=segment_start,
                    char_offset=char_offset,
                    chapter_index=chapter_index,
                )
            )
            char_count = sum(len(block["text"]) for block in blocks)
            record = {
                "index": chapter_index,
                "title": chapter["title"],
                "segment_start": segment_start,
                "segment_end": start_index + len(segments) - 1,
                "char_start": char_offset,
                "char_end": char_offset + char_count,
            }
            if "volume_index" in chapter:
                record["volume_index"] = int(chapter["volume_index"])
            chapters.append(record)
            char_offset += char_count
        return segments, chapters, char_offset

    def create_project(
        self,
        *,
        name: str,
        filename: str,
        data: bytes,
        settings: dict[str, Any],
        max_chars: int = DEFAULT_MAX_CHARS,
        series: str = "",
        voice_label: str = "",
    ) -> dict[str, Any]:
        text = extract_document_text(filename, data)
        max_chars = clamp_max_chars(max_chars)
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
        volumes: list[dict[str, Any]] = []
        _attach_volumes(chapter_inputs, volumes)
        segments, chapters, _ = self._build_segments(
            chapter_inputs, max_chars=max_chars, start_index=0, start_chapter=0, char_offset=0
        )
        now = _now()
        manifest: dict[str, Any] = {
            "id": project_id,
            "name": (name or Path(filename).stem or "未命名项目").strip()[:120],
            "state": "ready",
            "message": "项目已创建",
            "created_at": now,
            "updated_at": now,
            "revision": 0,
            "text_revision": 0,
            "settings": settings,
            "settings_fingerprint": settings_fingerprint(settings),
            "max_chars_per_segment": max_chars,
            "detected_language": detect_language(text),
            "series": normalize_series(series),
            "voice_label": str(voice_label or settings.get("voice_name") or "")[:120],
            "sources": [{"filename": filename, "stored_name": source_name, "added_at": now, "chars": len(text)}],
            "volumes": volumes,
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
        self._save(manifest, text_changed=True)
        entry = self._entry(project_id)
        with entry.lock:
            return self._public_manifest(manifest)

    def append_document(
        self, project_id: str, *, filename: str, data: bytes, volume_title: str = ""
    ) -> dict[str, Any]:
        """Append a file as a new volume (or the volumes it declares itself)."""
        text = extract_document_text(filename, data)
        with self._locked(project_id) as manifest:
            if manifest.get("state") in {"running", "stopping"}:
                raise RuntimeError("请先暂停项目，再追加文档")
            project_dir = self._project_dir(project_id)
            safe_suffix = Path(filename).suffix.lower() or ".txt"
            source_name = f"{uuid.uuid4().hex}{safe_suffix}"
            (project_dir / "sources" / source_name).write_bytes(data)
            existing_chars = sum(len(item["text"]) for item in manifest["segments"])
            stem = Path(filename).stem or "追加内容"
            chapter_inputs = split_novel_chapters(text, fallback_title=stem)
            chapters = manifest.setdefault("chapters", [])
            volumes = manifest.setdefault("volumes", [])
            if not volumes and chapters:
                # Turn the existing single-file book into its first volume so
                # every file of a multi-part novel is grouped the same way.
                first_source = (manifest.get("sources") or [{}])[0]
                first_title = Path(str(first_source.get("filename") or "")).stem or manifest.get("name") or "第一部分"
                volumes.append({"index": 0, "title": str(first_title)[:120]})
                for chapter in chapters:
                    chapter["volume_index"] = 0
            _attach_volumes(chapter_inputs, volumes, wrapper_title=(volume_title or "").strip() or stem)
            new_segments, new_chapters, _ = self._build_segments(
                chapter_inputs,
                max_chars=clamp_max_chars(manifest.get("max_chars_per_segment", DEFAULT_MAX_CHARS)),
                start_index=len(manifest["segments"]),
                start_chapter=len(chapters),
                char_offset=existing_chars,
            )
            first_new = len(manifest["segments"])
            manifest["segments"].extend(new_segments)
            chapters.extend(new_chapters)
            manifest["sources"].append(
                {"filename": filename, "stored_name": source_name, "added_at": _now(), "chars": len(text)}
            )
            manifest["state"] = "paused" if any(s["status"] == "completed" for s in manifest["segments"]) else "ready"
            manifest["message"] = f"已追加《{stem}》：{len(new_chapters)} 章 · {len(new_segments)} 个文字块"
            manifest["final_audio"] = None
            manifest["revision"] = int(manifest.get("revision") or 0)
            manifest["text_revision"] = int(manifest.get("text_revision") or 0) + 1
            self._save(manifest, segments=range(first_new, len(manifest["segments"])))
            return self._public_manifest(manifest)

    def _new_segments(
        self,
        chunks: list[Any],
        *,
        start_index: int,
        char_offset: int,
        chapter_index: int = 0,
    ) -> list[dict[str, Any]]:
        segments: list[dict[str, Any]] = []
        cursor = char_offset
        for offset, chunk in enumerate(chunks):
            if isinstance(chunk, dict):
                text = chunk["text"]
                pause_after = chunk.get("pause_after") or "sentence"
            else:
                text = str(chunk)
                pause_after = "chapter" if offset == len(chunks) - 1 else "sentence"
            index = start_index + offset
            segments.append(
                {
                    "index": index,
                    "text": text,
                    "char_start": cursor,
                    "char_end": cursor + len(text),
                    "chapter_index": chapter_index,
                    "pause_after": pause_after,
                    "status": "pending",
                    "attempts": 0,
                    "audio_file": None,
                    "duration_seconds": 0.0,
                    "generation_seconds": 0.0,
                    "error": None,
                    "model_profile": None,
                    "rev": 0,
                }
            )
            cursor += len(text)
        return segments

    def reset_for_settings(self, project_id: str, settings: dict[str, Any]) -> dict[str, Any]:
        with self._locked(project_id) as manifest:
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
            )
            # The reading position survives a voice change; only audio is new.
            self._save(manifest, text_changed=True)
            return self._public_manifest(manifest)

    def start(
        self,
        project_id: str,
        *,
        settings: dict[str, Any] | None = None,
        segment_indices: list[int] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            entry = self._entry(project_id)
            with entry.lock:
                manifest = entry.manifest
                settings_reset = False
                if settings is not None and settings_fingerprint(settings) != manifest.get("settings_fingerprint"):
                    thread = self._threads.get(project_id)
                    if thread is None or not thread.is_alive():
                        self.reset_for_settings(project_id, settings)
                        settings_reset = True
                thread = self._threads.get(project_id)
                if thread is not None and thread.is_alive():
                    return self._public_manifest(manifest)
                if sum(thread.is_alive() for thread in self._threads.values()) >= 8:
                    raise RuntimeError("后台书籍项目已达到并发上限，请等待现有项目完成")
                changed = []
                for segment in manifest["segments"]:
                    if segment["status"] in {"failed", "generating", "encoding"}:
                        segment["status"] = "pending"
                        segment["error"] = None
                        changed.append(int(segment["index"]))
                self._refresh_active_segments(manifest)
                if all(segment["status"] == "completed" for segment in manifest["segments"]) and manifest.get("final_audio"):
                    manifest["state"] = "completed"
                    self._save(manifest, segments=changed)
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
                entry.claim_hint = valid_indices[0] if valid_indices else 0
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
                self._save(manifest, segments=changed)
                thread = threading.Thread(target=self._run_project, args=(project_id, stop_event), daemon=True)
                self._threads[project_id] = thread
                thread.start()
                return self._public_manifest(manifest)

    def stop(self, project_id: str) -> dict[str, Any]:
        with self._lock:
            event = self._stop_events.get(project_id)
            if event is not None:
                event.set()
        with self._locked(project_id) as manifest:
            if manifest.get("state") == "running":
                manifest["state"] = "stopping"
                manifest["message"] = "将在当前并行段落完成并保存后停止"
                self._save(manifest)
            else:
                self.flush(project_id)
            return self._public_manifest(manifest)

    def stop_all(self) -> list[str]:
        """Request an immediate pause for every active document project."""
        with self._lock:
            self._force_stop_epoch += 1
            active_ids = list(self._stop_events)
            for project_id in active_ids:
                self._stop_events[project_id].set()
        for project_id in active_ids:
            try:
                with self._locked(project_id) as manifest:
                    if manifest.get("state") == "running":
                        manifest["state"] = "stopping"
                        manifest["message"] = "正在强制停止全部生成任务"
                        self._save(manifest)
            except (FileNotFoundError, ValueError):
                continue
        self.flush_all()
        return active_ids

    def delete(self, project_id: str) -> None:
        with self._lock:
            entry = self._entry(project_id)
            with entry.lock:
                thread = self._threads.get(project_id)
                if entry.manifest.get("state") in {"running", "stopping"} or (
                    thread is not None and thread.is_alive()
                ):
                    raise RuntimeError("请先停止项目，再删除")
                with entry.write_lock:
                    entry.deleted = True
                    entry.dirty = False
                    project_dir = self._project_dir(project_id)
                    try:
                        shutil.rmtree(project_dir)
                    except BaseException:
                        # Keep the project usable instead of hiding a half-deleted directory.
                        entry.deleted = False
                        raise
                with self._cache_lock:
                    self._entries.pop(project_id, None)

    def update_playback(self, project_id: str, *, segment_index: int, offset_seconds: float,
                        captured_at: float = 0.0, seed: int = -1) -> dict[str, Any]:
        """Persist the listening position in playback.json (not the manifest)."""
        if not math.isfinite(offset_seconds) or not math.isfinite(captured_at):
            raise ValueError("无效的播放位置")
        with self._locked(project_id) as manifest:
            previous = manifest.get("playback") or {}
            captured_at = min(time.time() * 1000 + 60_000, captured_at or time.time() * 1000)
            if captured_at < float(previous.get("captured_at", 0) or 0):
                return {"ok": True, "playback": dict(previous)}
            playback = {
                "segment_index": max(0, min(len(manifest["segments"]) - 1, int(segment_index))),
                "offset_seconds": max(0.0, min(3600, float(offset_seconds))),
                "captured_at": captured_at,
                "seed": seed if 0 <= seed <= 999999 else None,
            }
            self._store_playback(project_id, manifest, playback)
            return {"ok": True, "playback": dict(playback)}

    def update_segment_text(self, project_id: str, *, segment_index: int, text: str) -> dict[str, Any]:
        # Keep paragraph joiners ("\n") that merged short paragraphs into one block.
        revised = re.sub(r"[^\S\n]+", " ", text or "")
        revised = re.sub(r"\s*\n\s*", "\n", revised).strip()
        if not revised:
            raise ValueError("文字块不能为空")
        if len(revised) > 5000:
            raise ValueError("单个文字块不能超过 5000 字")
        with self._locked(project_id) as manifest:
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
            if (manifest.get("playback") or {}).get("segment_index") == int(segment_index):
                self._store_playback(
                    project_id,
                    manifest,
                    {"segment_index": int(segment_index), "offset_seconds": 0.0,
                     "seed": None, "captured_at": time.time() * 1000},
                )
            self._recompute_totals(manifest)
            manifest["state"] = (
                "paused"
                if any(item.get("status") == "completed" for item in manifest["segments"])
                else "ready"
            )
            manifest["message"] = f"第 {int(segment_index) + 1} 块文字已更新，旧音频已失效"
            self._save(manifest, text_changed=True)
            return self._public_manifest(manifest)

    @staticmethod
    def _recompute_totals(manifest: dict[str, Any]) -> None:
        manifest["completed_audio_seconds"] = sum(
            float(item.get("duration_seconds", 0.0)) for item in manifest["segments"]
        )
        manifest["generation_wall_seconds"] = sum(
            float(item.get("generation_seconds", 0.0)) for item in manifest["segments"]
        )

    # -- background run -----------------------------------------------------

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
                            if "generation cancelled while queued" in str(exc):
                                # Scheduler cancellation is issued for a service-wide
                                # stop. Preserve finished AAC blocks and let this
                                # project resume instead of presenting a false error.
                                stop_event.set()
                            elif not stop_event.is_set():
                                fatal_error = fatal_error or (index, exc)
                            continue
                        if fatal_error is not None:
                            self._discard_synthesis(project_id, index, synthesis)
                            continue
                        with self._locked(project_id) as manifest:
                            segment = manifest["segments"][index]
                            segment["status"] = "encoding"
                            self._refresh_active_segments(manifest)
                            manifest["message"] = (
                                f"正在并行生成整本书；第 {index + 1} 段正在转为 AAC"
                            )
                            self._save(manifest, segments=(index,), coalesce=True)
                        future = encoder.submit(self._encode_segment_aac, project_id, index, synthesis)
                        encode_futures[future] = index

                    if fatal_error is None and not stop_event.is_set():
                        while (len(synthesis_futures) < self.synthesis_workers
                               and len(synthesis_futures) + len(encode_futures) < self.synthesis_workers * 2):
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
                        with self._locked(project_id) as manifest:
                            failed_index = int(fatal_error[0])
                            changed = []
                            for segment in manifest["segments"]:
                                if (
                                    int(segment["index"]) != failed_index
                                    and segment["status"] in {"generating", "encoding"}
                                ):
                                    segment["status"] = "pending"
                                    segment["started_at"] = None
                                    segment["error"] = None
                                    changed.append(int(segment["index"]))
                            self._refresh_active_segments(manifest)
                            self._save(manifest, segments=changed)
                        return

                    if stop_event.is_set():
                        with self._locked(project_id) as manifest:
                            changed = []
                            for segment in manifest["segments"]:
                                if segment.get("status") in {"generating", "encoding"}:
                                    segment["status"] = "pending"
                                    segment["started_at"] = None
                                    segment["error"] = None
                                    changed.append(int(segment["index"]))
                            self._stop_run_clock(manifest)
                            manifest["state"] = "paused"
                            manifest["message"] = "已暂停，可随时继续"
                            self._refresh_active_segments(manifest)
                            self._save(manifest, segments=changed)
                        return

                    with self._locked(project_id) as manifest:
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
                        all_completed = all(
                            segment["status"] == "completed"
                            for segment in manifest["segments"]
                        )
                        if not all_completed:
                            self._stop_run_clock(manifest)
                            manifest["state"] = "paused"
                            manifest["message"] = "所选内容已生成高质量 AAC"
                            manifest["generation_scope"] = None
                            self._refresh_active_segments(manifest)
                            self._save(manifest)
                            return
                    self._finalize_completed_project(project_id, stop_event)
                    return
        finally:
            with self._lock:
                self._threads.pop(project_id, None)
                self._stop_events.pop(project_id, None)
            try:
                self.flush(project_id)
            except Exception:  # pragma: no cover
                LOG.exception("failed to flush document manifest %s", project_id)

    def _finalize_completed_project(self, project_id: str, stop_event: threading.Event) -> None:
        """Merge completed AAC blocks without blocking stop/edit/delete state checks."""
        with self._lock:
            force_stop_epoch = self._force_stop_epoch
        with self._locked(project_id) as manifest:
            if stop_event.is_set() or manifest.get("state") == "stopping":
                self._stop_run_clock(manifest)
                manifest["state"] = "paused"
                manifest["message"] = "已暂停，可随时继续"
                manifest["generation_scope"] = None
                self._refresh_active_segments(manifest)
                self._save(manifest)
                return
            manifest["message"] = "全部段落已完成，正在合并整书 M4A"
            self._refresh_active_segments(manifest)
            self._save(manifest)
            # Detached snapshot for the merge; the live manifest is never
            # touched outside its lock.
            snapshot: dict[str, Any] = {
                "id": manifest["id"],
                "settings": dict(manifest.get("settings") or {}),
                "segments": [dict(segment) for segment in manifest["segments"]],
                "final_audio": None,
                "_merge_force_stop_epoch": force_stop_epoch,
            }
        try:
            self._merge_final_audio(snapshot)
        except Exception as exc:
            with self._locked(project_id) as latest:
                self._stop_run_clock(latest)
                latest["generation_scope"] = None
                if stop_event.is_set() or latest.get("state") == "stopping":
                    latest["state"] = "paused"
                    latest["message"] = "已暂停；整书 M4A 尚未合并，可继续重试"
                else:
                    latest["state"] = "error"
                    latest["message"] = f"整书 AAC 合并失败，可继续重试：{exc}"
                self._refresh_active_segments(latest)
                self._save(latest)
            return

        with self._locked(project_id) as latest:
            latest["final_audio"] = snapshot.get("final_audio")
            self._stop_run_clock(latest)
            latest["generation_scope"] = None
            if stop_event.is_set() or latest.get("state") == "stopping":
                latest["state"] = "paused"
                latest["message"] = "已暂停；整书 M4A 已合并，可继续播放"
            else:
                latest["state"] = "completed"
                latest["message"] = "全部段落已完成，整书 M4A 已就绪"
            self._refresh_active_segments(latest)
            self._save(latest)

    def _refresh_active_segments(self, manifest: dict[str, Any]) -> None:
        active = sorted(
            int(segment["index"])
            for segment in manifest["segments"]
            if segment["status"] in {"generating", "encoding"}
        )
        manifest["active_segments"] = active
        manifest["current_segment"] = active[0] if active else None

    @staticmethod
    def _pause_for(manifest: dict[str, Any], segment: dict[str, Any]) -> str:
        pause = segment.get("pause_after")
        if pause in {"sentence", "paragraph", "chapter"}:
            return str(pause)
        index = int(segment["index"])
        segments = manifest["segments"]
        if index + 1 >= len(segments) or segments[index + 1].get("chapter_index") != segment.get("chapter_index"):
            return "chapter"
        return "sentence"

    def _claim_next_segment(
        self, project_id: str
    ) -> tuple[int, dict[str, Any], dict[str, Any]] | None:
        entry = self._entry(project_id)
        with entry.lock:
            manifest = entry.manifest
            segments = manifest["segments"]
            scope = manifest.get("generation_scope")
            scope_set = set(int(index) for index in scope) if scope else None

            def eligible(segment: dict[str, Any]) -> bool:
                return segment["status"] in {"pending", "failed"} and (
                    scope_set is None or int(segment["index"]) in scope_set
                )

            hint = max(0, min(entry.claim_hint, len(segments)))
            next_segment = next((segment for segment in segments[hint:] if eligible(segment)), None)
            if next_segment is None:
                next_segment = next((segment for segment in segments[:hint] if eligible(segment)), None)
            if next_segment is None:
                return None
            index = int(next_segment["index"])
            entry.claim_hint = index + 1
            next_segment["status"] = "generating"
            next_segment["attempts"] = int(next_segment.get("attempts", 0)) + 1
            next_segment["error"] = None
            next_segment["started_at"] = _now()
            self._refresh_active_segments(manifest)
            manifest["current_segment_started_at"] = min(
                float(segments[active].get("started_at") or _now())
                for active in manifest["active_segments"]
            )
            manifest["message"] = (
                f"正在并行生成整本书：{len(manifest['active_segments'])}/"
                f"{self.synthesis_workers} 个生成通道工作中"
            )
            claimed = dict(next_segment)
            claimed["pause_after"] = self._pause_for(manifest, next_segment)
            claimed["_default_language"] = manifest.get("detected_language") or "Chinese"
            self._save(manifest, segments=(index,), coalesce=True)
            return index, claimed, dict(manifest["settings"])

    def _fail_segment(self, project_id: str, index: int, exc: BaseException) -> None:
        with self._locked(project_id) as manifest:
            segment = manifest["segments"][index]
            segment["status"] = "failed"
            segment["error"] = str(exc)
            self._stop_run_clock(manifest)
            manifest["state"] = "error"
            manifest["message"] = f"第 {index + 1} 段生成失败，可继续重试"
            self._refresh_active_segments(manifest)
            self._save(manifest, segments=(index,))

    def _discard_synthesis(self, project_id: str, index: int, synthesis: dict[str, Any]) -> None:
        source_wav = Path(str(synthesis.get("source_wav", "")))
        project_dir = self._project_dir(project_id)
        working_dir = project_dir / "working"
        if source_wav.parent.exists() and working_dir in source_wav.parent.parents:
            shutil.rmtree(source_wav.parent, ignore_errors=True)
        entry = self._entry(project_id)
        with entry.lock:
            manifest = entry.manifest
            segment = manifest["segments"][index]
            segment["status"] = "pending"
            entry.claim_hint = min(entry.claim_hint, index)
            if manifest.get("current_segment") == index:
                manifest["current_segment"] = None
            self._save(manifest, segments=(index,))

    def _finish_encode(
        self, project_id: str, index: int, future: Future[dict[str, Any]]
    ) -> None:
        try:
            result = future.result()
        except Exception as exc:
            self._fail_segment(project_id, index, exc)
            raise
        with self._locked(project_id) as manifest:
            segment = manifest["segments"][index]
            old_duration = float(segment.get("duration_seconds", 0.0) or 0.0)
            old_generation = float(segment.get("generation_seconds", 0.0) or 0.0)
            segment.update(result)
            segment["status"] = "completed"
            segment["error"] = None
            # Incremental totals: O(1) per finished block instead of O(segments).
            manifest["completed_audio_seconds"] = float(manifest.get("completed_audio_seconds", 0.0)) + (
                float(segment.get("duration_seconds", 0.0) or 0.0) - old_duration
            )
            manifest["generation_wall_seconds"] = float(manifest.get("generation_wall_seconds", 0.0)) + (
                float(segment.get("generation_seconds", 0.0) or 0.0) - old_generation
            )
            manifest["message"] = f"已完成第 {index + 1}/{len(manifest['segments'])} 段"
            self._refresh_active_segments(manifest)
            self._save(manifest, segments=(index,), coalesce=True)

    def _build_request(self, **values: Any) -> Any:
        if is_dataclass(self.request_cls):
            accepted = {item.name for item in fields(self.request_cls)}
            values = {key: value for key, value in values.items() if key in accepted}
        return self.request_cls(**values)

    def _synthesize_segment(
        self, project_id: str, segment: dict[str, Any], settings: dict[str, Any]
    ) -> dict[str, Any]:
        with self._lock:
            stop_event = self._stop_events.get(project_id)
            force_stop_epoch = self._force_stop_epoch

        def check_cancelled() -> None:
            if stop_event is not None and stop_event.is_set():
                raise RuntimeError("generation cancelled while queued")

        check_cancelled()
        project_dir = self._project_dir(project_id)
        working_dir = project_dir / "working"
        working_dir.mkdir(parents=True, exist_ok=True)
        stored_text = segment["text"]
        text = prepare_tts_text(stored_text) or stored_text
        language = normalize_language(
            settings.get("language"), text, default=str(segment.get("_default_language") or "Chinese")
        )
        qwen_profile = str(settings.get("model_profile") or "").startswith("qwen_")
        configured_max = int(settings.get("max_new_tokens", 2048 if qwen_profile else 7500))
        frame_budget = (
            qwen_frame_budget(text, configured_max)
            if qwen_profile
            else max(80, min(configured_max, int(len(text) * 4.2)))
        )
        try:
            configured_seed = int(settings.get("seed", 1234))
        except (TypeError, ValueError):
            configured_seed = 1234
        model_profile = str(settings.get("model_profile") or "qwen_0_6b")

        def make_request(frames: int, seed: int | None) -> Any:
            return self._build_request(
                text=text,
                mode="voice_clone",
                prompt_text="",
                prompt_audio_path=settings.get("reference_audio_path") or None,
                language=language,
                tokens_control=False,
                tokens=0,
                max_new_frames=frames,
                do_sample=True,
                temperature=float(settings.get("temperature", 0.9)),
                top_p=float(settings.get("top_p", 1.0)),
                top_k=int(settings.get("top_k", 50)),
                repetition_penalty=float(settings.get("repetition_penalty", 1.05)),
                seed=seed,
                codec_chunk_frames=max(1, int(settings.get("codec_chunk_frames", 16))),
                qwen_xvec_only=str(settings.get("qwen_clone_mode") or "xvec") != "icl",
                qwen_reference_text=str(settings.get("qwen_reference_text") or ""),
                qwen_non_streaming_mode=bool(settings.get("qwen_non_streaming_mode", False)),
                qwen_append_silence=bool(settings.get("qwen_append_silence", True)),
                qwen_instruct=str(settings.get("qwen_instruct") or ""),
                qwen_min_new_tokens=max(2, int(settings.get("qwen_min_new_tokens", 2))),
                stream_audio=False,
                cancelled=lambda: force_stop_epoch != self._force_stop_epoch,
            )

        slot = (
            self.generation_lock.document_slot(cancelled=lambda: stop_event is not None and stop_event.is_set())
            if hasattr(self.generation_lock, "document_slot") else self.generation_lock
        )

        def run_once(request: Any) -> dict[str, Any] | None:
            result_event: dict[str, Any] | None = None
            with self.runtime_session(model_profile, cancelled=lambda: stop_event is not None and stop_event.is_set()) as runtime:
                check_cancelled()
                for event in self.synthesize_fn(runtime, request, output_dir=working_dir):
                    if event.type == "result":
                        result_event = event.data
            return result_event

        seed = None if configured_seed < 0 else configured_seed
        truncated_retry = False
        truncated = False
        with slot:
            check_cancelled()
            started_at = time.perf_counter()
            result_event = run_once(make_request(frame_budget, seed))
            if result_event is not None and qwen_profile and self._looks_truncated(result_event, frame_budget):
                # The block ran into its frame budget: the tail was probably cut.
                # Retry once with a new seed and a 1.5x (capped) budget.
                check_cancelled()
                truncated_retry = True
                truncated = True
                retry_budget = max(frame_budget, min(configured_max, int(frame_budget * 1.5)))
                retry_seed = ((seed if seed is not None else secrets.randbelow(1_000_000)) + 7919) % 1_000_000
                # Keep the first (usable, possibly clipped) take until the retry
                # succeeds: the worker's runaway guard can abort the longer
                # retry, which must not turn a clipped block into a failed book.
                try:
                    retry_event = run_once(make_request(retry_budget, retry_seed))
                except Exception:
                    if (stop_event is not None and stop_event.is_set()) or force_stop_epoch != self._force_stop_epoch:
                        self._remove_run_dir(project_dir, result_event.get("audio_path"))
                        raise
                    LOG.warning("truncation retry failed; keeping the first take", exc_info=True)
                    retry_event = None
                if retry_event is not None:
                    self._remove_run_dir(project_dir, result_event.get("audio_path"))
                    result_event = retry_event
                    truncated = self._looks_truncated(retry_event, retry_budget)
                    frame_budget = retry_budget
                    seed = retry_seed
            generation_seconds = time.perf_counter() - started_at
        if result_event is None:
            raise RuntimeError("模型没有返回完整音频")
        source_wav = Path(result_event["audio_path"])
        synthesis = {
            "source_wav": str(source_wav),
            "duration_seconds": float(result_event.get("metadata", {}).get("duration_seconds", 0.0)),
            "generation_seconds": generation_seconds,
            "model_profile": model_profile,
            "seed": int(settings.get("seed", 1234)),
            "seed_mode": str(settings.get("seed_mode") or "fixed"),
            "aac_bitrate": str(settings.get("aac_bitrate") or FINAL_AAC_BITRATE),
            "force_stop_epoch": force_stop_epoch,
            "pause_after": str(segment.get("pause_after") or "sentence"),
            "language": language,
            "max_new_frames": frame_budget,
        }
        if truncated_retry:
            synthesis["truncated_retry"] = True
            synthesis["truncated"] = truncated
            synthesis["retry_seed"] = seed
        return synthesis

    @staticmethod
    def _looks_truncated(result_event: dict[str, Any], frame_budget: int) -> bool:
        metadata = result_event.get("metadata") or {}
        frames = metadata.get("generated_frames")
        try:
            produced = float(frames) if frames is not None else float(metadata.get("duration_seconds", 0.0)) * QWEN_FRAMES_PER_SECOND
        except (TypeError, ValueError):
            return False
        return frame_budget > 0 and produced >= TRUNCATION_RATIO * frame_budget

    @staticmethod
    def _remove_run_dir(project_dir: Path, audio_path: Any) -> None:
        if not audio_path:
            return
        run_dir = Path(str(audio_path)).parent
        working_dir = project_dir / "working"
        if run_dir.exists() and working_dir in run_dir.parents:
            shutil.rmtree(run_dir, ignore_errors=True)

    def _encode_segment_aac(
        self, project_id: str, index: int, synthesis: dict[str, Any]
    ) -> dict[str, Any]:
        project_dir = self._project_dir(project_id)
        source_wav = Path(synthesis["source_wav"])
        output_name = f"{int(index):06d}.m4a"
        output_path = project_dir / "segments" / output_name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_name(output_path.stem + f".{uuid.uuid4().hex[:8]}.tmp.m4a")
        duration = float(synthesis.get("duration_seconds", 0.0))
        encode_source = source_wav
        audio_samples: int | None = None
        sample_rate: int | None = None
        post_processed = False
        try:
            from audio_post import process_wav_file

            processed_wav = source_wav.with_name(source_wav.stem + ".post.wav")
            audio_samples, sample_rate = process_wav_file(
                str(source_wav), str(processed_wav), synthesis.get("pause_after") or "sentence"
            )
            encode_source = processed_wav
            duration = audio_samples / float(sample_rate)
            post_processed = True
        except Exception:
            LOG.warning("audio post-processing failed for segment %s; encoding raw audio", index, exc_info=True)
        bitrate = str(synthesis.get("aac_bitrate") or FINAL_AAC_BITRATE)

        def segment_command(encoder: str) -> list[str]:
            return [
                self.ffmpeg_path,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(encode_source),
                *_aac_codec_args(encoder, bitrate),
                "-ac",
                "1",
                "-threads",
                "1",
                str(temporary_path),
            ]

        force_stop_epoch = synthesis.get("force_stop_epoch", self._force_stop_epoch)
        try:
            encoder = run_aac_encode(
                self.ffmpeg_path,
                segment_command,
                temporary_path,
                timeout=300,
                cancelled=lambda: force_stop_epoch != self._force_stop_epoch,
                failure_message="AAC 转码失败",
            )
            with self._lock:
                if force_stop_epoch != self._force_stop_epoch:
                    raise RuntimeError("AAC 转码已停止")
                os.replace(temporary_path, output_path)
        finally:
            temporary_path.unlink(missing_ok=True)
        self._remove_run_dir(project_dir, source_wav)
        result: dict[str, Any] = {
            "audio_file": f"segments/{output_name}",
            "duration_seconds": duration,
            "generation_seconds": float(synthesis.get("generation_seconds", 0.0)),
            "seed": int(synthesis.get("seed", 1234)),
            "seed_mode": str(synthesis.get("seed_mode") or "fixed"),
            "post_processed": post_processed,
            "pause_after": synthesis.get("pause_after") or "sentence",
            "aac_encoder": encoder,
        }
        if audio_samples is not None and sample_rate:
            result["audio_samples"] = int(audio_samples)
            result["audio_sample_rate"] = int(sample_rate)
        for key in ("language", "max_new_frames", "truncated_retry", "truncated", "retry_seed", "model_profile"):
            if key in synthesis:
                result[key] = synthesis[key]
        return result

    def _merge_final_audio(self, manifest: dict[str, Any]) -> None:
        """Decode every segment (trimmed to its exact length) and re-encode once.

        ``-c copy`` concatenation of AAC files keeps each file's priming and
        padding, which leaves ~25 ms gaps and timestamp drift. Segments are
        decoded in batches through the concat filter (``atrim`` drops the
        encoder padding), written as PCM WAV parts, then encoded once into a
        single gapless AAC with ``+faststart``.
        """
        force_stop_epoch = manifest.pop("_merge_force_stop_epoch", self._force_stop_epoch)
        cancelled = lambda: force_stop_epoch != self._force_stop_epoch  # noqa: E731
        project_dir = self._project_dir(manifest["id"])
        final_dir = project_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        work_dir = final_dir / f"merge-{uuid.uuid4().hex[:8]}"
        work_dir.mkdir(parents=True, exist_ok=True)
        final_path = final_dir / "complete.m4a"
        temporary = final_dir / f"complete.{uuid.uuid4().hex[:8]}.tmp.m4a"
        segments = manifest["segments"]
        total_seconds = sum(float(segment.get("duration_seconds") or 0.0) for segment in segments)
        try:
            parts: list[Path] = []
            for batch_start in range(0, len(segments), MERGE_BATCH_SIZE):
                batch = segments[batch_start : batch_start + MERGE_BATCH_SIZE]
                part = work_dir / f"part-{batch_start // MERGE_BATCH_SIZE:05d}.wav"
                command: list[str] = [self.ffmpeg_path, "-y", "-hide_banner", "-loglevel", "error"]
                filters: list[str] = []
                for position, segment in enumerate(batch):
                    audio_path = (project_dir / segment["audio_file"]).resolve()
                    command += ["-i", str(audio_path)]
                    samples = segment.get("audio_samples")
                    rate = segment.get("audio_sample_rate")
                    if samples and rate:
                        trim = f"atrim=end={int(samples) / float(rate):.6f}"
                    elif float(segment.get("duration_seconds") or 0.0) > 0:
                        trim = f"atrim=end={float(segment['duration_seconds']):.6f}"
                    else:
                        trim = "anull"
                    filters.append(
                        f"[{position}:a]{trim},aresample={FINAL_SAMPLE_RATE},"
                        f"aformat=sample_fmts=s16:channel_layouts=mono[a{position}]"
                    )
                labels = "".join(f"[a{position}]" for position in range(len(batch)))
                filters.append(f"{labels}concat=n={len(batch)}:v=0:a=1[out]")
                # Inline graph: ffmpeg 7+ removed -filter_complex_script and older
                # builds lack the "-/filter_complex file" form; a 200-input graph
                # is ~20 KB, far below the argument-size limit.
                command += ["-filter_complex", ";".join(filters), "-map", "[out]", "-c:a", "pcm_s16le", str(part)]
                batch_seconds = sum(float(segment.get("duration_seconds") or 0.0) for segment in batch)
                completed = run_media_process(command, timeout=120 + batch_seconds / 5, cancelled=cancelled)
                if completed.returncode != 0 or not part.exists():
                    raise RuntimeError(completed.stderr.strip() or "最终 AAC 合并失败")
                parts.append(part)
            concat_path = work_dir / "parts.txt"
            concat_path.write_text(
                "\n".join("file '" + str(part).replace("'", "'\\''") + "'" for part in parts),
                encoding="utf-8",
            )
            bitrate = str((manifest.get("settings") or {}).get("aac_bitrate") or FINAL_AAC_BITRATE)
            run_aac_encode(
                self.ffmpeg_path,
                lambda encoder: [
                    self.ffmpeg_path, "-y", "-hide_banner", "-loglevel", "error",
                    "-f", "concat", "-safe", "0", "-i", str(concat_path),
                    *_aac_codec_args(encoder, bitrate),
                    "-ar", str(FINAL_SAMPLE_RATE), "-ac", "1",
                    "-movflags", "+faststart",
                    "-f", "mp4",
                    str(temporary),
                ],
                temporary,
                timeout=300 + total_seconds / 10,
                cancelled=cancelled,
                failure_message="最终 AAC 合并失败",
            )
            with self._lock:
                if force_stop_epoch != self._force_stop_epoch:
                    raise RuntimeError("AAC 合并已停止")
                os.replace(temporary, final_path)
        finally:
            temporary.unlink(missing_ok=True)
            shutil.rmtree(work_dir, ignore_errors=True)
        manifest["final_audio"] = "final/complete.m4a"

    def media_path(self, project_id: str, relative_path: str) -> Path:
        project_dir = self._project_dir(project_id)
        candidate = (project_dir / relative_path).resolve(strict=True)
        if project_dir not in candidate.parents or not candidate.is_file():
            raise ValueError("invalid media path")
        return candidate

    # -- serialisation ------------------------------------------------------

    def _stats(self, manifest: dict[str, Any]) -> dict[str, Any]:
        """Progress statistics computed directly from the cached manifest."""
        segments = manifest.get("segments", [])
        total_chars = 0
        completed_chars = 0
        completed_segments = 0
        for item in segments:
            length = len(item.get("text", ""))
            total_chars += length
            if item.get("status") == "completed":
                completed_chars += length
                completed_segments += 1
        completed_audio = float(manifest.get("completed_audio_seconds", 0.0))
        work_seconds = float(manifest.get("generation_wall_seconds", 0.0))
        elapsed_seconds = float(manifest.get("generation_elapsed_seconds", 0.0))
        run_started_at = manifest.get("current_run_started_at")
        running = manifest.get("state") in {"running", "stopping"}
        if running and run_started_at is not None:
            elapsed_seconds += max(0.0, _now() - float(run_started_at))
        speed = completed_audio / elapsed_seconds if elapsed_seconds > 0 else 0.0
        audio_per_char = completed_audio / completed_chars if completed_chars > 0 else 0.24
        active_indices = [
            int(index)
            for index in manifest.get("active_segments", []) or []
            if 0 <= int(index) < len(segments)
        ]
        in_progress_chars = 0.0
        if running and active_indices and speed > 0:
            active_lanes = max(1, len(active_indices))
            now = _now()
            for index in active_indices:
                segment = segments[index]
                segment_chars = len(segment.get("text", ""))
                expected_wall = max(1.0, segment_chars * audio_per_char * active_lanes / speed)
                segment_elapsed = max(0.0, now - float(segment.get("started_at") or now))
                in_progress_chars += segment_chars * min(0.95, segment_elapsed / expected_wall)
        progress_chars = completed_chars + in_progress_chars
        progress = progress_chars / total_chars if total_chars > 0 else 0.0
        remaining_chars = max(0.0, total_chars - progress_chars)
        eta = remaining_chars * audio_per_char / speed if speed > 0 else None
        return {
            "total_segments": len(segments),
            "completed_segments": completed_segments,
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

    @staticmethod
    def _top_level_copy(manifest: dict[str, Any], *, exclude: set[str]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in manifest.items():
            if key in exclude or key.startswith("_"):
                continue
            result[key] = copy.deepcopy(value) if isinstance(value, (dict, list)) else value
        return result

    @staticmethod
    def _chapter_at(manifest: dict[str, Any], segment_index: int) -> dict[str, Any] | None:
        chapters = manifest.get("chapters") or []
        starts = [int(chapter.get("segment_start", 0)) for chapter in chapters]
        position = bisect.bisect_right(starts, int(segment_index)) - 1
        return chapters[position] if 0 <= position < len(chapters) else None

    def _summary(self, manifest: dict[str, Any]) -> dict[str, Any]:
        result = self._top_level_copy(manifest, exclude={"segments", "chapters", "volumes", "sources"})
        result.setdefault("active_segments", [])
        result.setdefault("revision", 0)
        result.setdefault("text_revision", 0)
        result.setdefault("series", "")
        result["chapter_count"] = len(manifest.get("chapters") or []) or (1 if manifest.get("segments") else 0)
        result["volume_count"] = len(manifest.get("volumes") or [])
        result["source_count"] = len(manifest.get("sources") or [])
        playback = manifest.get("playback") or {}
        reading = self._chapter_at(manifest, int(playback.get("segment_index") or 0))
        result["reading_chapter_index"] = int(reading.get("index", 0)) if reading else 0
        result["reading_chapter_title"] = str(reading.get("title") or "") if reading else ""
        captured = float(playback.get("captured_at") or 0)
        result["last_read_at"] = captured / 1000.0 if captured > 0 else None
        result["stats"] = self._stats(manifest)
        return result

    @staticmethod
    def _public_volumes(manifest: dict[str, Any], chapters: list[dict[str, Any]]) -> list[dict[str, Any]]:
        volumes = [dict(volume) for volume in manifest.get("volumes") or []]
        for volume in volumes:
            members = [int(chapter["index"]) for chapter in chapters if chapter.get("volume_index") == volume["index"]]
            volume["chapter_start"] = min(members) if members else None
            volume["chapter_end"] = max(members) if members else None
            volume["chapter_count"] = len(members)
        return volumes

    def _public_manifest(self, manifest: dict[str, Any]) -> dict[str, Any]:
        result = self._top_level_copy(manifest, exclude={"segments", "chapters"})
        result.setdefault("active_segments", [])
        segments = [dict(segment) for segment in manifest.get("segments", [])]
        result["segments"] = segments
        chapters = [dict(chapter) for chapter in manifest.get("chapters") or []]
        if not chapters:
            chapters = [
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
        result["chapters"] = chapters
        result["volumes"] = self._public_volumes(manifest, chapters)
        result.setdefault("series", "")
        result["stats"] = self._stats(manifest)
        return result
