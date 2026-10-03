"""End-to-end smoke test against a running server.

Streams synthetic audio (silence, then a loud tone, then silence) at real-time
pace and asserts the full chain fires: VAD boundary events out, the ASR worker
decodes, and stats come back with a real-time factor.

This does NOT check transcription accuracy — a sine tone is not speech, so the
text is expected to be empty or nonsense. It checks that audio ingested over
the socket reaches the decoder and that timings are reported. Accuracy is a
Phase 3 concern measured against LibriSpeech.

    .venv/Scripts/python.exe -m tests.smoke_ws
"""

from __future__ import annotations

import asyncio
import json
import sys

import numpy as np
import websockets

URL = "ws://127.0.0.1:8100/ws"
RATE = 16_000
CHUNK = 320  # 20ms


def _pcm(samples: np.ndarray) -> bytes:
    return (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()


def _clip() -> np.ndarray:
    rng = np.random.default_rng(0)
    quiet = lambda s: rng.normal(0, 0.001, int(RATE * s)).astype(np.float32)
    t = np.arange(int(RATE * 1.5)) / RATE
    tone = (0.35 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    return np.concatenate([quiet(0.8), tone, quiet(1.2)])


async def main() -> int:
    seen: list[dict] = []
    async with websockets.connect(URL, max_size=None) as ws:
        async def reader() -> None:
            async for raw in ws:
                msg = json.loads(raw)
                seen.append(msg)
                print("  <-", msg)

        task = asyncio.create_task(reader())

        audio = _clip()
        for i in range(0, audio.size, CHUNK):
            await ws.send(_pcm(audio[i:i + CHUNK]))
            await asyncio.sleep(CHUNK / RATE)  # real-time pace

        await asyncio.sleep(3.0)  # let the tail decode land
        task.cancel()

    kinds = {m["type"] for m in seen}
    print("\nevent kinds:", sorted(kinds))

    failures = []
    if "ready" not in kinds:
        failures.append("no ready event")
    if "speech_start" not in kinds:
        failures.append("VAD never fired on the tone")
    if "speech_end" not in kinds:
        failures.append("VAD never closed the segment")
    if "stats" not in kinds:
        failures.append("no stats — audio never reached the decoder")
    else:
        stats = [m for m in seen if m["type"] == "stats"]
        rtf = [m["rtf"] for m in stats]
        queue = [m["queue_ms"] for m in stats]
        print(f"rtf:      min={min(rtf):.2f} max={max(rtf):.2f}")
        print(f"queue_ms: min={min(queue)} max={max(queue)}")

        # Queue wait is the backlog signal. RTF above 1.0 on a slow box is
        # survivable — partials get skipped and the transcript stays current.
        # A queue that keeps growing is not: it means work is being created
        # faster than it drains, and the transcript falls permanently behind.
        if max(queue) > 2000:
            failures.append(
                f"decode queue backed up to {max(queue)}ms — work is being "
                "created faster than it drains"
            )
        if max(rtf) >= 1.0:
            print("  ! RTF >= 1.0 on at least one window — slow box, expected here")

        # The committed final must cover the whole utterance. A clock skew
        # between the VAD and the session once truncated this to ~20ms, which
        # a tone-based test cannot see in the text — only in the duration.
        finals = [m for m in stats if m.get("kind") == "final"]
        if not finals:
            failures.append("no final decode")
        elif finals[-1]["audio_s"] < 1.0:
            failures.append(
                f"final decoded only {finals[-1]['audio_s']}s of a ~2s "
                "utterance — window truncated"
            )

    for f in failures:
        print("FAIL:", f)
    print("\nOK" if not failures else "\nFAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
