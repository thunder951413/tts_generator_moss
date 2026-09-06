"""Cancellable subprocess runner for bounded media conversion work."""

from __future__ import annotations

import subprocess
import time
from collections.abc import Callable, Sequence
from os import PathLike


_POLL_INTERVAL_SECONDS = 0.1
_TERMINATE_GRACE_SECONDS = 1.0
_KILL_GRACE_SECONDS = 1.0


def _stop_and_collect(process: subprocess.Popen[str]) -> tuple[str, str]:
    """Stop and reap one child while continuing to drain both output pipes."""
    if process.poll() is None:
        try:
            process.terminate()
        except ProcessLookupError:
            pass
    try:
        return process.communicate(timeout=_TERMINATE_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        if process.poll() is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
        try:
            return process.communicate(timeout=_KILL_GRACE_SECONDS)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("media process could not be reaped after kill") from exc


def run_media_process(
    command: Sequence[str | PathLike[str]],
    *,
    cancelled: Callable[[], bool],
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    """Run one media command with captured text output and bounded cancellation."""
    if cancelled():
        raise RuntimeError("media process was cancelled")

    timeout_seconds = max(0.0, float(timeout))
    process = subprocess.Popen(
        command,
        shell=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    deadline = time.monotonic() + timeout_seconds

    while True:
        try:
            cancel_requested = cancelled()
        except BaseException:
            _stop_and_collect(process)
            raise
        if cancel_requested:
            _stop_and_collect(process)
            raise RuntimeError("media process was cancelled")

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            stdout, stderr = _stop_and_collect(process)
            raise subprocess.TimeoutExpired(
                process.args,
                timeout_seconds,
                output=stdout,
                stderr=stderr,
            )
        try:
            stdout, stderr = process.communicate(
                timeout=min(_POLL_INTERVAL_SECONDS, remaining)
            )
        except subprocess.TimeoutExpired:
            continue
        return subprocess.CompletedProcess(
            process.args,
            int(process.returncode or 0),
            stdout,
            stderr,
        )


__all__ = ["run_media_process"]
