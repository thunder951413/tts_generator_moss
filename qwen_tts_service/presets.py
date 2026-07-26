"""Persistent voice presets shared by the web UI and the macOS menu-bar app."""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any


PRESET_ID_LENGTH = 32


def _now() -> float:
    return time.time()


def _atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


class VoicePresetStore:
    """Keep named voice/generation settings and their imported reference audio."""

    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir).resolve()
        self.audio_dir = self.root_dir / "reference_audio"
        self.manifest_path = self.root_dir / "presets.json"
        self._lock = threading.RLock()
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        if not self.manifest_path.exists():
            _atomic_write(self.manifest_path, {"version": 1, "presets": [], "active_preset_id": ""})

    def _load(self) -> dict[str, Any]:
        try:
            value = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            value = {"version": 1, "presets": []}
        if not isinstance(value, dict):
            value = {"version": 1, "presets": []}
        if not isinstance(value.get("presets"), list):
            value["presets"] = []
        if not isinstance(value.get("active_preset_id"), str):
            value["active_preset_id"] = ""
        if not isinstance(value.get("active_service"), dict):
            value["active_service"] = {}
        value["version"] = 1
        return value

    @staticmethod
    def _copy_preset(preset: dict[str, Any]) -> dict[str, Any]:
        return json.loads(json.dumps(preset, ensure_ascii=False))

    @staticmethod
    def _normalize_name(value: Any) -> str:
        name = str(value or "").strip()
        if not name:
            raise ValueError("预设名称不能为空")
        return name[:120]

    @staticmethod
    def _normalize_settings(value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise ValueError("预设设置必须是对象")
        try:
            copied = json.loads(json.dumps(value, ensure_ascii=False))
        except (TypeError, ValueError) as exc:
            raise ValueError("预设设置无法保存为 JSON") from exc
        if not isinstance(copied, dict):
            raise ValueError("预设设置必须是对象")
        if len(copied) > 64:
            raise ValueError("预设设置字段过多")
        return copied

    @staticmethod
    def _find(manifest: dict[str, Any], preset_id: str) -> dict[str, Any]:
        for preset in manifest["presets"]:
            if isinstance(preset, dict) and preset.get("id") == preset_id:
                return preset
        raise KeyError(preset_id)

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            manifest = self._load()
            presets = [self._copy_preset(item) for item in manifest["presets"] if isinstance(item, dict)]
        return sorted(presets, key=lambda item: (-float(item.get("updated_at", 0)), str(item.get("name", ""))))

    def active(self) -> dict[str, Any] | None:
        with self._lock:
            manifest = self._load()
            preset_id = str(manifest.get("active_preset_id") or "")
            if not preset_id:
                return None
            try:
                return self._copy_preset(self._find(manifest, preset_id))
            except KeyError:
                manifest["active_preset_id"] = ""
                _atomic_write(self.manifest_path, manifest)
                return None

    def service_configuration(self) -> dict[str, Any] | None:
        with self._lock:
            manifest = self._load()
            active_service = manifest.get("active_service")
            if not isinstance(active_service, dict) or not isinstance(active_service.get("settings"), dict):
                return None
            return self._copy_preset(active_service)

    def apply_configuration(
        self,
        *,
        name: Any,
        settings: Any,
        preset_id: str = "",
    ) -> dict[str, Any]:
        with self._lock:
            manifest = self._load()
            if preset_id:
                self._find(manifest, preset_id)
            configuration = {
                "name": self._normalize_name(name),
                "preset_id": preset_id,
                "settings": self._normalize_settings(settings),
                "updated_at": _now(),
            }
            manifest["active_preset_id"] = preset_id
            manifest["active_service"] = configuration
            _atomic_write(self.manifest_path, manifest)
            return self._copy_preset(configuration)

    def clear_configuration(self) -> None:
        with self._lock:
            manifest = self._load()
            manifest["active_preset_id"] = ""
            manifest["active_service"] = {}
            _atomic_write(self.manifest_path, manifest)

    def activate(self, preset_id: str) -> dict[str, Any]:
        if len(preset_id) != PRESET_ID_LENGTH:
            raise KeyError(preset_id)
        with self._lock:
            manifest = self._load()
            preset = self._find(manifest, preset_id)
            manifest["active_preset_id"] = preset_id
            manifest["active_service"] = {
                "name": preset["name"],
                "preset_id": preset_id,
                "settings": self._copy_preset(preset.get("settings") or {}),
                "updated_at": _now(),
            }
            _atomic_write(self.manifest_path, manifest)
            return self._copy_preset(preset)

    def create(self, *, name: Any, settings: Any) -> dict[str, Any]:
        with self._lock:
            manifest = self._load()
            now = _now()
            preset = {
                "id": uuid.uuid4().hex,
                "name": self._normalize_name(name),
                "settings": self._normalize_settings(settings),
                "created_at": now,
                "updated_at": now,
            }
            manifest["presets"].append(preset)
            _atomic_write(self.manifest_path, manifest)
            return self._copy_preset(preset)

    def update(self, preset_id: str, *, name: Any, settings: Any) -> dict[str, Any]:
        if len(preset_id) != PRESET_ID_LENGTH:
            raise KeyError(preset_id)
        with self._lock:
            manifest = self._load()
            preset = self._find(manifest, preset_id)
            preset["name"] = self._normalize_name(name)
            preset["settings"] = self._normalize_settings(settings)
            preset["updated_at"] = _now()
            if manifest.get("active_preset_id") == preset_id:
                manifest["active_service"] = {
                    "name": preset["name"],
                    "preset_id": preset_id,
                    "settings": self._copy_preset(preset["settings"]),
                    "updated_at": preset["updated_at"],
                }
            _atomic_write(self.manifest_path, manifest)
            return self._copy_preset(preset)

    def delete(self, preset_id: str) -> None:
        if len(preset_id) != PRESET_ID_LENGTH:
            raise KeyError(preset_id)
        with self._lock:
            manifest = self._load()
            preset = self._find(manifest, preset_id)
            manifest["presets"].remove(preset)
            if manifest.get("active_preset_id") == preset_id:
                manifest["active_preset_id"] = ""
                manifest["active_service"] = {}
            _atomic_write(self.manifest_path, manifest)

    def import_reference_audio(self, *, filename: str, temporary_path: str | Path) -> Path:
        source = Path(temporary_path)
        if not source.is_file():
            raise ValueError("参考音频不存在")
        suffix = Path(filename or source.name).suffix.lower()
        if suffix not in {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus", ".aac"}:
            raise ValueError("不支持的参考音频格式")
        destination = self.audio_dir / f"{uuid.uuid4().hex}{suffix}"
        with self._lock:
            shutil.copyfile(source, destination)
        return destination.resolve()

    def is_managed_reference(self, path: str | Path) -> bool:
        try:
            candidate = Path(path).resolve(strict=True)
        except (OSError, RuntimeError):
            return False
        return candidate.is_file() and (candidate == self.audio_dir or self.audio_dir in candidate.parents)
