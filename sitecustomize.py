"""Windows DLL search paths for the project-local MOSS-TTS runtime."""

from __future__ import annotations

import os
from pathlib import Path


_DLL_DIRECTORY_HANDLES = []

if os.name == "nt" and hasattr(os, "add_dll_directory"):
    _repo_dir = Path(__file__).resolve().parent
    _candidate_dirs = (
        _repo_dir / ".ffmpeg-runtime" / "Library" / "bin",
        _repo_dir / ".venv" / "Lib" / "site-packages" / "torch" / "lib",
    )
    for _directory in _candidate_dirs:
        if _directory.is_dir():
            _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(_directory)))
