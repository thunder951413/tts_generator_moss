import sys
import threading
from pathlib import Path
import queue

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "qwen_tts_service"))
from streaming_jobs import LosslessAudioQueue


def test_slow_reader_keeps_every_chunk_in_order_past_old_queue_limit():
    audio = LosslessAudioQueue()
    expected = [bytes([index % 256]) * 8192 for index in range(130)]
    for chunk in expected:
        audio.put_nowait(chunk)
    audio.put_nowait(None)
    assert [audio.get() for _ in expected] == expected
    assert audio.get() is None


def test_close_wakes_waiter_and_overflow_is_explicit():
    audio = LosslessAudioQueue(max_bytes=8)
    audio.put_nowait(b"12345678")
    with pytest.raises(queue.Full):
        audio.put_nowait(b"9")
    assert audio.get() == b"12345678"
    results = []
    waiter = threading.Thread(target=lambda: results.append(audio.get()))
    waiter.start()
    audio.close()
    waiter.join(1)
    assert results == [None]
