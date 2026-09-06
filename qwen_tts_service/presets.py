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
            _atomic_write(
                self.manifest_path,
                {
                    "version": 2,
                    "presets": [],
                    "active_preset_id": "",
                    "reference_audio": [],
                    "hidden_builtin_references": [],
                },
            )

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
        if not isinstance(value.get("reference_audio"), list):
            value["reference_audio"] = []
        if not isinstance(value.get("hidden_builtin_references"), list):
            value["hidden_builtin_references"] = []
        value["version"] = 2
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
            manifest = self._load()
            shutil.copyfile(source, destination)
            now = _now()
            manifest["reference_audio"].append(
                {
                    "id": uuid.uuid4().hex,
                    "name": (Path(filename or source.name).stem.strip() or "参考音频")[:120],
                    "filename": Path(filename or source.name).name[:255],
                    "path": str(destination.resolve()),
                    "hidden": False,
                    "created_at": now,
                    "updated_at": now,
                }
            )
            _atomic_write(self.manifest_path, manifest)
        return destination.resolve()

    def _discover_reference_audio(self, manifest: dict[str, Any]) -> bool:
        known_paths = {
            str(item.get("path") or "")
            for item in manifest["reference_audio"]
            if isinstance(item, dict)
        }
        changed = False
        for path in sorted(self.audio_dir.iterdir()):
            if not path.is_file() or str(path.resolve()) in known_paths:
                continue
            now = path.stat().st_mtime
            manifest["reference_audio"].append(
                {
                    "id": uuid.uuid4().hex,
                    "name": path.stem[:120],
                    "filename": path.name[:255],
                    "path": str(path.resolve()),
                    "hidden": False,
                    "created_at": now,
                    "updated_at": now,
                }
            )
            changed = True
        return changed

    def list_reference_audio(self, *, include_hidden: bool = True) -> list[dict[str, Any]]:
        with self._lock:
            manifest = self._load()
            changed = self._discover_reference_audio(manifest)
            existing: list[dict[str, Any]] = []
            for item in manifest["reference_audio"]:
                if not isinstance(item, dict):
                    changed = True
                    continue
                path = Path(str(item.get("path") or ""))
                if not path.is_file() or not self.is_managed_reference(path):
                    changed = True
                    continue
                if include_hidden or not bool(item.get("hidden")):
                    existing.append(self._copy_preset(item))
            if changed:
                existing_paths = {
                    str(Path(str(item.get("path") or "")).resolve())
                    for item in manifest["reference_audio"]
                    if isinstance(item, dict)
                    and Path(str(item.get("path") or "")).is_file()
                    and self.is_managed_reference(str(item.get("path") or ""))
                }
                manifest["reference_audio"] = [
                    item
                    for item in manifest["reference_audio"]
                    if isinstance(item, dict)
                    and str(Path(str(item.get("path") or "")).resolve()) in existing_paths
                ]
                _atomic_write(self.manifest_path, manifest)
        return sorted(
            existing,
            key=lambda item: (-float(item.get("created_at", 0)), str(item.get("name", ""))),
        )

    def reference_audio_record(self, reference_id: str) -> dict[str, Any]:
        with self._lock:
            manifest = self._load()
            self._discover_reference_audio(manifest)
            for item in manifest["reference_audio"]:
                if isinstance(item, dict) and item.get("id") == reference_id:
                    return self._copy_preset(item)
        raise KeyError(reference_id)

    def set_reference_hidden(self, reference_id: str, hidden: bool) -> dict[str, Any]:
        with self._lock:
            manifest = self._load()
            self._discover_reference_audio(manifest)
            for item in manifest["reference_audio"]:
                if isinstance(item, dict) and item.get("id") == reference_id:
                    item["hidden"] = bool(hidden)
                    item["updated_at"] = _now()
                    _atomic_write(self.manifest_path, manifest)
                    return self._copy_preset(item)
        raise KeyError(reference_id)

    def rename_reference_audio(self, reference_id: str, name: Any) -> dict[str, Any]:
        """Rename a managed reference and keep settings that use it readable."""
        normalized_name = str(name or "").strip()
        if not normalized_name:
            raise ValueError("参考音频名称不能为空")
        normalized_name = normalized_name[:120]
        with self._lock:
            manifest = self._load()
            self._discover_reference_audio(manifest)
            target = next(
                (
                    item
                    for item in manifest["reference_audio"]
                    if isinstance(item, dict) and item.get("id") == reference_id
                ),
                None,
            )
            if target is None:
                raise KeyError(reference_id)
            path = Path(str(target.get("path") or ""))
            if not self.is_managed_reference(path):
                raise ValueError("只能重命名参考音频库中的文件")

            target["name"] = normalized_name
            target["updated_at"] = _now()
            resolved_path = str(path.resolve())
            for preset in manifest["presets"]:
                settings = preset.get("settings") if isinstance(preset, dict) else None
                if (
                    isinstance(settings, dict)
                    and str(Path(str(settings.get("reference_audio_path") or "")).resolve())
                    == resolved_path
                ):
                    settings["voice_name"] = normalized_name
                    preset["updated_at"] = _now()
            active_service = manifest.get("active_service")
            settings = active_service.get("settings") if isinstance(active_service, dict) else None
            if (
                isinstance(settings, dict)
                and str(Path(str(settings.get("reference_audio_path") or "")).resolve())
                == resolved_path
            ):
                settings["voice_name"] = normalized_name
                active_service["updated_at"] = _now()
            _atomic_write(self.manifest_path, manifest)
            return self._copy_preset(target)

    def hidden_builtin_references(self) -> set[str]:
        with self._lock:
            manifest = self._load()
            return {
                str(path)
                for path in manifest.get("hidden_builtin_references", [])
                if isinstance(path, str)
            }

    def set_builtin_hidden(self, path: str | Path, hidden: bool) -> None:
        normalized = str(Path(path).resolve())
        with self._lock:
            manifest = self._load()
            hidden_paths = self.hidden_builtin_references()
            if hidden:
                hidden_paths.add(normalized)
            else:
                hidden_paths.discard(normalized)
            manifest["hidden_builtin_references"] = sorted(hidden_paths)
            _atomic_write(self.manifest_path, manifest)

    def reference_usage(self, path: str | Path) -> list[str]:
        normalized = str(Path(path).resolve())
        with self._lock:
            manifest = self._load()
            usages: list[str] = []
            for preset in manifest["presets"]:
                if not isinstance(preset, dict):
                    continue
                settings = preset.get("settings")
                if (
                    isinstance(settings, dict)
                    and str(Path(str(settings.get("reference_audio_path") or "")).resolve()) == normalized
                ):
                    usages.append(f"预设“{preset.get('name') or '未命名'}”")
            active_service = manifest.get("active_service")
            if isinstance(active_service, dict):
                settings = active_service.get("settings")
                if (
                    isinstance(settings, dict)
                    and str(Path(str(settings.get("reference_audio_path") or "")).resolve()) == normalized
                ):
                    usages.append("当前服务设置")
            return list(dict.fromkeys(usages))

    def delete_reference_audio(
        self,
        reference_id: str,
        *,
        replacement_audio_path: str = "",
        replacement_voice_name: str = "",
    ) -> list[str]:
        with self._lock:
            manifest = self._load()
            self._discover_reference_audio(manifest)
            target = next(
                (
                    item
                    for item in manifest["reference_audio"]
                    if isinstance(item, dict) and item.get("id") == reference_id
                ),
                None,
            )
            if target is None:
                raise KeyError(reference_id)
            path = Path(str(target.get("path") or ""))
            usages = self.reference_usage(path)
            if usages and not replacement_audio_path:
                raise ValueError("该参考音频仍被" + "、".join(usages) + "使用，请先更换音色")
            if usages:
                normalized = str(path.resolve())
                for preset in manifest["presets"]:
                    if not isinstance(preset, dict):
                        continue
                    settings = preset.get("settings")
                    if (
                        isinstance(settings, dict)
                        and str(Path(str(settings.get("reference_audio_path") or "")).resolve())
                        == normalized
                    ):
                        settings["reference_audio_path"] = replacement_audio_path
                        settings["voice_name"] = replacement_voice_name
                        preset["updated_at"] = _now()
                active_service = manifest.get("active_service")
                if isinstance(active_service, dict):
                    settings = active_service.get("settings")
                    if (
                        isinstance(settings, dict)
                        and str(Path(str(settings.get("reference_audio_path") or "")).resolve())
                        == normalized
                    ):
                        settings["reference_audio_path"] = replacement_audio_path
                        settings["voice_name"] = replacement_voice_name
                        active_service["updated_at"] = _now()
            if not self.is_managed_reference(path):
                raise ValueError("只能删除导入到参考音频库的文件")
            path.unlink(missing_ok=True)
            manifest["reference_audio"].remove(target)
            _atomic_write(self.manifest_path, manifest)
            return usages

    def is_managed_reference(self, path: str | Path) -> bool:
        try:
            candidate = Path(path).resolve(strict=True)
        except (OSError, RuntimeError):
            return False
        return candidate.is_file() and (candidate == self.audio_dir or self.audio_dir in candidate.parents)
