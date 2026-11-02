"""Private worker transport and filesystem boundary helpers."""

from __future__ import annotations

import hmac
from pathlib import Path

WORKER_TOKEN_HEADER = "X-Qwen-Worker-Token"
MAX_WORKER_REQUEST_BYTES = 2 * 1024 * 1024
MAX_TEXT_CHARS = 32000


def secrets_equal(left: str, right: str) -> bool:
    return hmac.compare_digest(left.encode("utf-8"), right.encode("utf-8"))


def decode_header_secret(value: str) -> str:
    # ASGI decodes raw header bytes as Latin-1. Local UTF-8 clients can send
    # the same Unicode password used by the login form without mojibake.
    try:
        return value.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return value


def allowed_path(value: str, roots: tuple[Path, ...], *, must_exist: bool) -> Path:
    candidate = Path(value).expanduser().resolve(strict=must_exist)
    if not any(candidate.is_relative_to(root.resolve()) for root in roots):
        raise ValueError("worker path is outside the configured directories")
    return candidate
