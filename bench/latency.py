"""Latency benchmark — the numbers that go in the README.

Streams a fixture through the live service at real-time pace and measures what
a user actually experiences:

  onset -> first word    how long until anything appears
  onset -> final         how long until the line is committed
  RTF                    decode time / audio duration; under 1.0 or the
                         service falls behind real time
  arbitration            how often each engine won the committed line

Reports p50 and p95 over N passes, not a single run: this laptop throttles, and
one sample would be a coin flip. The first pass is discarded as warm-up — an
earlier version of bench/threads.py timed it and produced the impossible result
that beam=5 was faster than beam=1.

    .venv/Scripts/python.exe -m bench.latency --runs 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
import wave
from pathlib import Path

import websockets

URL = "ws://127.0.0.1:8100/ws"
FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "speech.wav"
RATE = 16_000


def pct(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round((p / 100) * (len(ordered) - 1))))
    return ordered[idx]


async def one_pass(gpu: bool = False) -> dict:
    with wave.open(str(FIXTURE), "rb") as w:
        pcm = w.readframes(w.getnframes())

    first_word: list[float] = []
    to_final: list[float] = []
    rtfs: list[float] = []
    sources: list[str] = []
    opened_at: float | None = None
    seen_first = False

    async with websockets.connect(URL, max_size=None) as ws:
        if gpu:
            await ws.send(json.dumps({"type": "config", "gpu": True}))

        async def reader() -> None:
            nonlocal opened_at, seen_first
            async for raw in ws:
                msg = json.loads(raw)
                now = time.perf_counter()
                if msg["type"] == "speech_start":
                    opened_at, seen_first = now, False
                elif msg["type"] == "partial" and opened_at and not seen_first:
                    seen_first = True
                    first_word.append((now - opened_at) * 1000)
                elif msg["type"] == "final" and opened_at:
                    to_final.append((now - opened_at) * 1000)
                    sources.append(msg.get("source", "?"))
                elif msg["type"] == "stats" and msg.get("kind") == "final":
                    rtfs.append(msg["rtf"])

        task = asyncio.create_task(reader())
        started = time.perf_counter()
        step = 1600 * 2  # 100ms
        sent = 0
        for i in range(0, len(pcm), step):
            await ws.send(pcm[i:i + step])
            sent += len(pcm[i:i + step])
            if (d := started + sent / 2 / RATE - time.perf_counter()) > 0:
                await asyncio.sleep(d)
        await asyncio.sleep(15.0)
        task.cancel()

    return {
        "first_word": first_word,
        "to_final": to_final,
        "rtf": rtfs,
        "sources": sources,
    }


async def main(runs: int, gpu: bool) -> int:
    health = json.loads(
        __import__("urllib.request", fromlist=["x"])
        .urlopen("http://127.0.0.1:8100/health").read()
    )
    print(f"profile={health['profile']}  fast={health['fast_model']}  "
          f"accurate={health['accurate_model']}  device={health['device']}  "
          f"stream={health['streaming_engine']}\n")

    if gpu and not health.get("gpu_tier_available"):
        print("GPU tier requested but the server has no SD_BEAM_URL/TOKEN set")
        return 1
    if gpu:
        print("GPU tier: ON (remote finals)\n")

    print("warm-up pass (discarded)...", flush=True)
    await one_pass(gpu)

    agg: dict[str, list] = {"first_word": [], "to_final": [], "rtf": [], "sources": []}
    for i in range(runs):
        print(f"pass {i + 1}/{runs}...", flush=True)
        for k, v in (await one_pass(gpu)).items():
            agg[k].extend(v)

    print(f"\n{'=' * 60}\nRESULTS over {runs} passes\n{'=' * 60}")
    for label, key, unit in (
        ("onset -> first word", "first_word", "ms"),
        ("onset -> committed final", "to_final", "ms"),
    ):
        vals = agg[key]
        if vals:
            print(f"  {label:<26} p50 {pct(vals, 50):>7.0f}{unit}   "
                  f"p95 {pct(vals, 95):>7.0f}{unit}   (n={len(vals)})")

    if agg["rtf"]:
        r = agg["rtf"]
        print(f"  {'final RTF':<26} p50 {pct(r, 50):>7.2f}     "
              f"p95 {pct(r, 95):>7.2f}     "
              f"{'OK' if pct(r, 95) < 1 else 'OVER REAL TIME'}")

    if agg["sources"]:
        total = len(agg["sources"])
        for engine in ("whisper", "vosk"):
            n = agg["sources"].count(engine)
            print(f"  committed by {engine:<13} {n:>3}/{total}  ({100 * n / total:.0f}%)")

    print(f"{'=' * 60}")
    print("\nCopy these into the README. Do not round them in your favour.")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--gpu", action="store_true", help="route finals to the GPU tier")
    args = ap.parse_args()
    sys.exit(asyncio.run(main(args.runs, args.gpu)))
