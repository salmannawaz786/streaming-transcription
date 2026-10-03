"""Measure decode cost vs. model size and CPU thread count.

Written because a guess was wrong: pinning CTranslate2 to the 2 *physical*
cores of an i7-7500U was assumed to beat its default of 4, on the theory that
oversubscription makes threads fight. The server said otherwise, so this
measures it directly instead of reasoning about it.

Also checks the claim the whole design rests on — that Whisper pads to a 30s
mel window, so decode cost barely depends on clip length.

    .venv/Scripts/python.exe -m bench.threads
"""

from __future__ import annotations


import time

import numpy as np
from faster_whisper import WhisperModel

RATE = 16_000
REPEATS = 3


def clip(seconds: float) -> np.ndarray:
    """Speech-like noise. Content does not matter; decode cost is what we time."""
    rng = np.random.default_rng(0)
    n = int(RATE * seconds)
    t = np.arange(n) / RATE
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)  # syllable-rate wobble
    tone = 0.3 * np.sin(2 * np.pi * 140 * t) * envelope
    return (tone + rng.normal(0, 0.01, n)).astype(np.float32)


def _once(model: WhisperModel, audio: np.ndarray, beam: int) -> float:
    t0 = time.perf_counter()
    segments, _ = model.transcribe(
        audio, language="en", beam_size=beam,
        vad_filter=False, without_timestamps=True,
        condition_on_previous_text=False,
    )
    list(segments)  # generator: force the work
    return (time.perf_counter() - t0) * 1000


def time_decode(model: WhisperModel, audio: np.ndarray, beam: int) -> float:
    """Minimum of N after a warm-up.

    The first decode on a freshly constructed model pays one-off costs and can
    run ~2x the steady-state figure. An earlier version of this script timed it,
    and produced the impossible result that beam=5 was faster than beam=1 —
    which is how the contamination was caught. Minimum, not median: this is a
    thermally throttled laptop, so the fast runs are the honest ones and the
    slow ones are the OS scheduling something else.
    """
    _once(model, audio, beam)
    return min(_once(model, audio, beam) for _ in range(REPEATS))


def main() -> None:
    three = clip(3.0)

    print("decode ms (median of 3), int8 on cpu\n")
    print(f"{'model':<10} {'threads':>7} {'beam':>5} {'ms':>8}")
    print("-" * 34)

    results: dict[tuple[str, int, int], float] = {}
    for name in ("tiny.en", "base.en"):
        for threads in (2, 4):
            model = WhisperModel(
                name, device="cpu", compute_type="int8",
                cpu_threads=threads, num_workers=1,
            )
            for beam in (1, 5):
                ms = time_decode(model, three, beam)
                results[(name, threads, beam)] = ms
                print(f"{name:<10} {threads:>7} {beam:>5} {ms:>8.0f}")
            del model

    print("\nlength independence (tiny.en, threads=4, beam=1)")
    print(f"{'audio_s':>8} {'ms':>8}  {'ratio vs 0.5s':>14}")
    print("-" * 34)
    model = WhisperModel(
        "tiny.en", device="cpu", compute_type="int8", cpu_threads=4, num_workers=1
    )
    base = None
    for secs in (0.5, 1.0, 2.0, 4.0):
        ms = time_decode(model, clip(secs), 1)
        base = base or ms
        print(f"{secs:>8.1f} {ms:>8.0f}  {ms / base:>13.2f}x")

    print("\nverdict")
    for name in ("tiny.en", "base.en"):
        t2, t4 = results[(name, 2, 1)], results[(name, 4, 1)]
        faster = "4 threads" if t4 < t2 else "2 threads"
        print(f"  {name}: {faster} wins ({t2:.0f}ms @2 vs {t4:.0f}ms @4)")


if __name__ == "__main__":
    main()
