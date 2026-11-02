"""Bound service-owned caches without deleting saved books or normal results."""
import os
import shutil
import threading
import time
from pathlib import Path
from fastapi import HTTPException

# A directory whose mtime is older than this is assumed settled: its direct
# file sizes are reused from the previous scan instead of stat'ing every file
# again.  Files are written via temp+rename (which bumps the directory mtime),
# so this only risks missing in-place appends, which a periodic full rescan
# picks up.
_SETTLED_DIRECTORY_SECONDS = 300.0
_FULL_RESCAN_EVERY = 10


class StoragePolicy:
    def __init__(self, roots: tuple[Path, ...]):
        self.roots = tuple(dict.fromkeys(path.resolve() for path in roots))
        self.max_bytes = max(1, int(os.environ.get("QWEN_TTS_STORAGE_MAX_BYTES", 10 * 1024**3)))
        self.min_free_bytes = max(0, int(os.environ.get("QWEN_TTS_STORAGE_MIN_FREE_BYTES", 1024**3)))
        self._lock = threading.Lock()
        self._used_bytes = 0
        # directory path -> (mtime_ns, total size of its direct regular files)
        self._directory_cache: dict[str, tuple[int, int]] = {}
        self._refreshes = 0
        self.refresh()

    def refresh(self):
        """Recount used bytes (same semantics as a full walk of every root).

        Uses ``os.scandir`` (one lstat per file, no pathlib overhead) and
        reuses per-directory totals for settled directories between periodic
        full rescans, so the 60 s maintenance loop stays cheap on large trees.
        """
        self._refreshes += 1
        full_rescan = self._refreshes % _FULL_RESCAN_EVERY == 1
        settled_before = time.time() - _SETTLED_DIRECTORY_SECONDS
        previous = self._directory_cache
        cache: dict[str, tuple[int, int]] = {}
        used = 0
        stack = [str(root) for root in self.roots]
        while stack:
            directory = stack.pop()
            if directory in cache:
                continue  # overlapping roots
            try:
                directory_mtime = os.stat(directory).st_mtime_ns
            except OSError:
                continue
            cached = previous.get(directory)
            reuse = (
                not full_rescan
                and cached is not None
                and cached[0] == directory_mtime
                and directory_mtime / 1e9 < settled_before
            )
            files_total = 0
            try:
                with os.scandir(directory) as entries:
                    for entry in entries:
                        try:
                            if entry.is_symlink():
                                continue
                            if entry.is_dir(follow_symlinks=False):
                                stack.append(entry.path)
                            elif not reuse and entry.is_file(follow_symlinks=False):
                                files_total += entry.stat(follow_symlinks=False).st_size
                        except OSError:
                            continue
            except OSError:
                continue
            if reuse:
                files_total = cached[1]
            cache[directory] = (directory_mtime, files_total)
            used += files_total
        self._directory_cache = cache
        with self._lock:
            self._used_bytes = used

    def status(self):
        with self._lock:
            return {"used_bytes": self._used_bytes, "max_bytes": self.max_bytes,
                    "min_free_bytes": self.min_free_bytes,
                    "saved_results_auto_deleted": False}

    def check_admission(self):
        if self.status()["used_bytes"] >= self.max_bytes:
            raise HTTPException(507, "语音数据已达到容量上限，请导出或删除不再需要的任务/书籍")
        existing = next((root for root in self.roots if root.exists()), Path.cwd())
        if shutil.disk_usage(existing).free < self.min_free_bytes:
            raise HTTPException(507, "磁盘剩余空间不足，请释放空间后重试")
