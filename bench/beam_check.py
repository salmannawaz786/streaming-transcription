"""Verify the Beam GPU tier and measure warm round-trip latency.

Cold start (container boot + model load) is reported separately from warm
calls, because they are different numbers with different meanings: cold start
is a demo risk you mitigate with keep-warm, warm latency is what the tier
actually costs per utterance.

    set -a; . ./.env.beam; set +a
    .venv/Scripts/python.exe -m bench.beam_check
"""

from __future__ import annotations

import statistics
import sys
import time
import wave
from pathlib import Path

import numpy as np

from app.asr import remote

FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "speech.wav"
WARM_CALLS = 3


def main() -> int:
    if not remote.configured():
        print("SD_BEAM_URL / SD_BEAM_TOKEN not set — load .env.beam first")
        return 1

    with wave.open(str(FIXTURE), "rb") as w:
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    audio = audio.astype(np.float32) / 32768.0
    duration = audio.size / 16_000
    print(f"fixture: {duration:.1f}s\n")

    t0 = time.perf_counter()
    first = remote.transcribe(audio)
    cold_ms = (time.perf_counter() - t0) * 1000
    print(f"first call (may include cold start): {cold_ms:.0f}ms")
    print(f"  text: {first['text']}")
    print(f"  logprob={first['logprob']:.2f} no_speech={first['no_speech']:.2f}\n")

    warm = []
    for i in range(WARM_CALLS):
        t0 = time.perf_counter()
        remote.transcribe(audio)
        ms = (time.perf_counter() - t0) * 1000
        warm.append(ms)
        print(f"  warm call {i + 1}: {ms:.0f}ms")

    median = statistics.median(warm)
    rtf = (median / 1000) / duration
    print(f"\nwarm median: {median:.0f}ms for {duration:.1f}s audio  (RTF {rtf:.2f})")
    print("RTF well under 1.0 means the GPU tier keeps up with real time,")
    print("including network round-trip from this machine.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
