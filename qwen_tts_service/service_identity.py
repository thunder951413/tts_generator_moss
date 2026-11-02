"""Version and source identity shared by local clients and the service."""
import hashlib
from pathlib import Path

API_VERSION = 2
CAPABILITIES = ["worker-auth", "bounded-worker-recovery", "resumable-reader", "storage-policy"]

def source_build_id(root: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted(
        path for directory in (root / "qwen_tts_service", root / "web/novel_reader")
        for path in directory.rglob("*")
        if path.is_file() and path.suffix in {".py", ".js", ".html", ".css"}
    )
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()

BUILD_ID = source_build_id(Path(__file__).resolve().parent.parent)
