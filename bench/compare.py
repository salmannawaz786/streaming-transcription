"""Run several models over the same recording and print them side by side.

Answers the question that cannot be answered by guessing: when transcription is
wrong, is it the microphone or the model?

  - If every model returns garbage, the audio is the problem — check the level
    readout below, the mic, and browser noise suppression.
  - If tiny.en is wrong and small.en is right, the audio is fine and the model
    was too small. That is a hardware/latency decision, not an audio one.

Capture your own voice first by running the server with SD_DUMP=1, speaking a
few sentences, then:

    .venv/Scripts/python.exe -m bench.compare debug/*.wav
"""

from __future__ import annotations

import glob
import sys
import wave
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel

MODELS = ["tiny.en", "base.en", "small.en"]


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        if w.getframerate() != 16_000 or w.getnchannels() != 1:
            raise SystemExit(f"{path.name}: need 16kHz mono")
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def describe(audio: np.ndarray) -> str:
    """Level diagnostics. Bad audio usually shows up here, not in the words."""
    rms = float(np.sqrt(np.mean(audio**2)))
    peak = float(np.abs(audio).max())
    clipped = int(np.sum(np.abs(audio) > 0.99))
    dbfs = 20 * np.log10(rms) if rms > 0 else -99

    notes = []
    if dbfs < -40:
        notes.append("VERY QUIET — raise mic gain or move closer")
    elif dbfs < -30:
        notes.append("quiet")
    if clipped > 10:
        notes.append(f"CLIPPING ({clipped} samples) — lower mic gain")
    if peak < 0.05:
        notes.append("almost no signal")

    tail = ("  <-- " + "; ".join(notes)) if notes else "  (levels look fine)"
    return (
        f"  {audio.size / 16_000:.1f}s  rms {rms:.4f} ({dbfs:+.0f} dBFS)  "
        f"peak {peak:.2f}{tail}"
    )


def main(argv: list[str]) -> int:
    patterns = argv or ["debug/*.wav"]
    paths = sorted({Path(p) for pat in patterns for p in glob.glob(pat)})
    if not paths:
        print("no wav files found.\n\nRecord some first:\n"
              "  SD_DUMP=1 .venv/Scripts/python.exe -m uvicorn app.server:app --port 8100\n"
              "then talk at http://localhost:8100 and re-run this.")
        return 1

    print(f"{len(paths)} clip(s), {len(MODELS)} models\n")

    clips = [(p, read_wav(p)) for p in paths]
    for path, audio in clips:
        print(f"{path.name}\n{describe(audio)}")
    print()

    # One model in memory at a time: holding all three at once made them
    # contend for RAM and cores on a 2-core laptop.
    #
    # This tool deliberately reports NO timings. Two attempts to include them
    # gave impossible orderings — base.en slower than small.en (38s vs 17s),
    # then tiny.en slower than both (28s vs 4.7s) — because whichever model
    # runs first pays warm-up and the laptop throttles under sustained load.
    # Measuring accuracy and latency in one pass does neither well.
    # bench/threads.py owns timing; this file owns words.
    results: dict[Path, list[tuple[str, str, float]]] = {p: [] for p, _ in clips}
    for name in MODELS:
        print(f"loading {name}...", flush=True)
        model = WhisperModel(
            name, device="cpu", compute_type="int8", cpu_threads=4, num_workers=1
        )
        for path, audio in clips:
            segs = list(model.transcribe(
                audio, language="en", beam_size=5, vad_filter=False,
                without_timestamps=True, condition_on_previous_text=False,
            )[0])
            text = "".join(s.text for s in segs).strip() or "(nothing)"
            conf = max((s.no_speech_prob for s in segs), default=0.0)
            results[path].append((name, text, conf))
        del model

    for path, rows in results.items():
        print(f"\n{'=' * 72}\n{path.name}\n")
        for name, text, conf in rows:
            print(f"  {name:<9} no_speech={conf:.2f}")
            print(f"    {text}")

    print(f"\n{'=' * 72}")
    print("If every model is wrong -> the audio is the problem (see levels).")
    print("If only tiny.en is wrong -> the audio is fine, the model is too small.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
