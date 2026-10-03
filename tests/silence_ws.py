"""Regression test for the "Thank you." bug.

Streams a quiet room — room tone plus the sort of faint transients a real mic
picks up (keystrokes, breaths, a chair creak) — and asserts the service stays
silent.

Before the fix this produced an endless column of "Thank you." finals: the
energy VAD latched onto transients, half a second of near-silence went to
Whisper, and Whisper filled it with caption boilerplate from its training data.

    .venv/Scripts/python.exe -m tests.silence_ws
"""

from __future__ import annotations

import asyncio
import json
import sys

import numpy as np
import websockets

URL = "ws://127.0.0.1:8100/ws"
RATE = 16_000
STEP = 1600  # 100ms


def quiet_room(seconds: float) -> np.ndarray:
    """Room tone with faint transients — never loud enough to be speech."""
    rng = np.random.default_rng(7)
    n = int(RATE * seconds)
    audio = rng.normal(0, 0.0008, n).astype(np.float32)   # room tone
    audio += 0.0004 * np.sin(2 * np.pi * 50 * np.arange(n) / RATE)  # mains hum

    for start in rng.integers(0, n - RATE // 4, size=8):   # clicks and bumps
        length = int(RATE * 0.04)
        env = np.hanning(length).astype(np.float32)
        audio[start:start + length] += 0.004 * env * rng.normal(0, 1, length)
    return audio.astype(np.float32)


async def main() -> int:
    audio = quiet_room(12.0)
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    print(f"streaming {audio.size / RATE:.0f}s of quiet room "
          f"(peak {np.abs(audio).max():.4f} rms {np.sqrt((audio**2).mean()):.5f})\n")

    got: list[dict] = []
    async with websockets.connect(URL, max_size=None) as ws:
        async def reader() -> None:
            async for raw in ws:
                msg = json.loads(raw)
                if msg["type"] in ("partial", "final"):
                    got.append(msg)
                    print(f"  {msg['type']}: {msg['text']!r}")
                elif msg["type"] == "speech_start":
                    print(f"  (vad opened at t={msg['t']:.2f}s)")

        task = asyncio.create_task(reader())
        for i in range(0, pcm.size, STEP):
            await ws.send(pcm[i:i + STEP].tobytes())
            await asyncio.sleep(STEP / RATE)
        await asyncio.sleep(10.0)  # let anything queued surface
        task.cancel()

    finals = [m for m in got if m["type"] == "final"]
    print(f"\ntranscript lines produced: {len(got)} (finals: {len(finals)})")
    if got:
        print("FAIL: silence produced transcript output —")
        for m in got:
            print(f"  {m['type']}: {m['text']!r}")
        print("\nFAILED")
        return 1
    print("\nOK — silence stayed silent")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
