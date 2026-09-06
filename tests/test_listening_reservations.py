from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "qwen_tts_service"))

from activity_control import PlaybackCoordinator  # noqa: E402


def test_listening_reservation_blocks_external_work_until_released() -> None:
    coordinator = PlaybackCoordinator()
    reservation = coordinator.reserve_listening("reader-session-123", ttl=12)

    assert reservation["playback_epoch"] == 0
    assert coordinator.internal_playback_active()
    assert coordinator.status()["listening_reservations"] == 1
    assert coordinator.release_listening("reader-session-123")
    assert not coordinator.internal_playback_active()


def test_stale_listening_heartbeat_cannot_restore_force_stopped_session() -> None:
    coordinator = PlaybackCoordinator()
    reservation = coordinator.reserve_listening("reader-session-123", ttl=12)
    stopped = coordinator.force_stop()

    assert stopped["playback_epoch"] == reservation["playback_epoch"] + 1
    assert coordinator.heartbeat_listening(
        "reader-session-123", expected_playback_epoch=int(reservation["playback_epoch"]), ttl=12
    ) is None
    assert coordinator.status()["listening_reservations"] == 0
    assert not coordinator.internal_playback_active()
