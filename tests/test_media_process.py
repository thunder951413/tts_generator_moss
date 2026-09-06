from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SERVICE_DIR = ROOT / "qwen_tts_service"
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

from media_process import run_media_process


def _long_running_command(pid_path: Path) -> list[str]:
    script = (
        "import os, pathlib, signal, sys, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); "
        "print('started', flush=True); "
        "print('diagnostic', file=sys.stderr, flush=True); "
        "time.sleep(30)"
    )
    return [sys.executable, "-c", script, str(pid_path)]


def _wait_for_file(path: Path, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() >= deadline:
            raise AssertionError("child did not publish its pid before timeout")
        time.sleep(0.01)


def _assert_process_gone(pid: int) -> None:
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_run_media_process_returns_completed_process_and_drains_output() -> None:
    size = 256 * 1024
    completed = run_media_process(
        [
            sys.executable,
            "-c",
            f"import sys; sys.stdout.write('o'*{size}); sys.stderr.write('e'*{size})",
        ],
        cancelled=lambda: False,
        timeout=3,
    )

    assert completed.returncode == 0
    assert completed.args[0] == sys.executable
    assert completed.stdout == "o" * size
    assert completed.stderr == "e" * size


def test_run_media_process_cancellation_kills_and_reaps_exact_child(
    tmp_path: Path,
) -> None:
    pid_path = tmp_path / "cancelled.pid"
    cancel = threading.Event()
    errors: list[BaseException] = []

    def run() -> None:
        try:
            run_media_process(
                _long_running_command(pid_path),
                cancelled=cancel.is_set,
                timeout=10,
            )
        except BaseException as exc:
            errors.append(exc)

    started_at = time.monotonic()
    worker = threading.Thread(target=run)
    worker.start()
    _wait_for_file(pid_path)
    pid = int(pid_path.read_text())
    cancel.set()
    worker.join(timeout=3)

    assert not worker.is_alive()
    assert time.monotonic() - started_at < 3
    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert "cancelled" in str(errors[0]).lower()
    _assert_process_gone(pid)


def test_run_media_process_timeout_kills_and_reaps_exact_child(
    tmp_path: Path,
) -> None:
    pid_path = tmp_path / "timed-out.pid"
    started_at = time.monotonic()

    with pytest.raises(subprocess.TimeoutExpired) as raised:
        run_media_process(
            _long_running_command(pid_path),
            cancelled=lambda: False,
            timeout=0.2,
        )

    _wait_for_file(pid_path)
    pid = int(pid_path.read_text())
    assert time.monotonic() - started_at < 3
    assert raised.value.timeout == 0.2
    assert "started" in str(raised.value.output)
    assert "diagnostic" in str(raised.value.stderr)
    _assert_process_gone(pid)
