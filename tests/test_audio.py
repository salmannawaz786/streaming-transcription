"""Tests for the two pieces with fiddly index arithmetic.

Run without the ASR stack — these need only numpy:

    python -m tests.test_audio
"""

from __future__ import annotations

import numpy as np

from app.audio.ringbuffer import RingBuffer
from app.audio.vad import EnergyVad
from app.config import FRAME_SAMPLES, SAMPLE_RATE


def test_wrap_preserves_order() -> None:
    rb = RingBuffer(10)
    rb.write(np.arange(8, dtype=np.float32))
    rb.write(np.arange(8, 14, dtype=np.float32))  # wraps
    assert np.array_equal(rb.read_last(10), np.arange(4, 14, dtype=np.float32))
    assert rb.dropped_samples == 4


def test_read_last_shorter_than_size() -> None:
    rb = RingBuffer(100)
    rb.write(np.arange(50, dtype=np.float32))
    assert np.array_equal(rb.read_last(5), np.arange(45, 50, dtype=np.float32))


def test_oversized_write_keeps_tail() -> None:
    rb = RingBuffer(4)
    rb.write(np.arange(10, dtype=np.float32))
    assert np.array_equal(rb.read_last(4), np.arange(6, 10, dtype=np.float32))
    assert rb.dropped_samples == 6


def test_read_last_empty() -> None:
    assert RingBuffer(8).read_last(4).size == 0


def _tone(seconds: float, amp: float) -> np.ndarray:
    n = int(SAMPLE_RATE * seconds)
    t = np.arange(n) / SAMPLE_RATE
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def _quiet(seconds: float) -> np.ndarray:
    n = int(SAMPLE_RATE * seconds)
    rng = np.random.default_rng(0)
    return (rng.normal(0, 0.001, n)).astype(np.float32)


def test_vad_detects_one_utterance() -> None:
    vad = EnergyVad()
    events = []
    events += vad.push(_quiet(1.0))
    events += vad.push(_tone(1.0, 0.3))
    events += vad.push(_quiet(1.5))

    kinds = [e.kind for e in events]
    assert kinds == ["speech_start", "speech_end"], kinds

    # Start is backdated past the onset debounce, so the first syllable is not
    # clipped: it must land at or before the true speech onset at t=1.0s.
    assert events[0].at_sample <= SAMPLE_RATE


def test_vad_ignores_steady_silence() -> None:
    vad = EnergyVad()
    assert vad.push(_quiet(3.0)) == []
    assert not vad.active


def test_vad_ignores_quiet_room_transients() -> None:
    """The 'Thank you.' bug.

    In a quiet room the adaptive floor sinks until a keystroke or a breath sits
    several times above it. The VAD opened, half a second of near-silence went
    to Whisper, and Whisper — whose training data is full of video captions —
    confidently returned "Thank you." over and over. The absolute floor is what
    stops that.
    """
    vad = EnergyVad()
    events = vad.push(_quiet(2.0))          # floor adapts down to room tone
    for _ in range(6):
        events += vad.push(_tone(0.05, 0.004))  # faint transient: click, breath
        events += vad.push(_quiet(0.4))
    assert events == [], f"false triggers on room noise: {[e.kind for e in events]}"


def test_vad_still_hears_real_speech_after_quiet() -> None:
    """The gate above must not deafen it to someone actually talking."""
    vad = EnergyVad()
    vad.push(_quiet(2.0))
    events = vad.push(_tone(1.0, 0.25)) + vad.push(_quiet(1.0))
    assert [e.kind for e in events] == ["speech_start", "speech_end"]


def test_vad_frame_alignment_across_ragged_writes() -> None:
    """Odd-sized writes must not desynchronise framing."""
    vad = EnergyVad()
    audio = np.concatenate([_quiet(0.5), _tone(0.8, 0.3), _quiet(1.2)])
    events = []
    i = 0
    for step in (37, FRAME_SAMPLES + 1, 5, 999):
        while i < audio.size:
            events += vad.push(audio[i:i + step])
            i += step
    assert [e.kind for e in events] == ["speech_start", "speech_end"]


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  ok   {name}")
            except AssertionError as exc:
                failures += 1
                print(f"  FAIL {name}: {exc}")
    raise SystemExit(failures)
