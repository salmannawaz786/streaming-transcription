"""Decide which engine's transcript to commit.

The original design assumed the offline model is always right and replaced the
streaming hypothesis with it unconditionally. On clearly-articulated English
that holds. On accented or non-native speech it does not: a small Whisper is
frequently *worse* than Vosk, so "correction" actively destroys words that were
already correct on screen — and does it after a visible delay, so the user
watches a right answer turn into a wrong one.

So the final is a candidate, not an authority. Whisper wins only when it is
confident. When it is unsure, or when it disagrees so wildly with the streaming
hypothesis that one of them must be badly wrong, the streaming text stands.

Thresholds here are starting points, not truth. They need tuning against real
recordings of the actual speaker — capture some with SD_DUMP=1 and compare.
"""

from __future__ import annotations

import difflib
import os
import re
from dataclasses import dataclass

# Below this average token log-probability, Whisper is guessing. Its own
# decoder treats roughly this range as low quality.
CONFIDENT = float(os.getenv("SD_ARB_CONFIDENT", "-0.55"))
# Below this it is guessing badly enough that the streaming text always wins.
DESPERATE = float(os.getenv("SD_ARB_DESPERATE", "-0.90"))
# Similarity under which the two engines are telling different stories.
DIVERGENT = float(os.getenv("SD_ARB_DIVERGENT", "0.45"))
# Whisper dropping this fraction of the words Vosk heard means it lost content.
SHORT_RATIO = 0.6
# Clips shorter than this get extra scrutiny - see choose().
SHORT_CLIP_S = float(os.getenv("SD_ARB_SHORT_CLIP", "1.5"))

_WORD = re.compile(r"[a-z0-9']+")


@dataclass(frozen=True)
class Choice:
    text: str
    source: str   # "whisper" | "vosk"
    reason: str
    similarity: float


def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def similarity(a: str, b: str) -> float:
    """Word-level agreement between two transcripts, 0..1."""
    wa, wb = _words(a), _words(b)
    if not wa and not wb:
        return 1.0
    if not wa or not wb:
        return 0.0
    return difflib.SequenceMatcher(None, wa, wb).ratio()


def choose(
    stream_text: str,
    whisper_text: str,
    logprob: float,
    duration_s: float = 99.0,
    tier: str = "local",
) -> Choice:
    stream_text = (stream_text or "").strip()
    whisper_text = (whisper_text or "").strip()

    if not whisper_text:
        return Choice(stream_text, "vosk", "whisper returned nothing", 0.0)
    if not stream_text:
        # No streaming hypothesis to defend, so nothing to arbitrate.
        return Choice(whisper_text, "whisper", "no streaming text", 0.0)

    sim = similarity(stream_text, whisper_text)
    is_gpu = (tier == "gpu" or tier.startswith("beam"))

    # GPU tier (large-v3-turbo): has 800M params vs 40MB Kaldi. Divergence is
    # almost always Whisper fixing Kaldi's phonetic errors, so trust GPU tier
    # unless logprob indicates total gibberish.
    gpu_desperate = float(os.getenv("SD_ARB_GPU_DESPERATE", "-1.30"))
    if is_gpu:
        if logprob < gpu_desperate:
            return Choice(stream_text, "vosk", f"gpu whisper unsure ({logprob:.2f})", sim)
        return Choice(whisper_text, "whisper", f"gpu tier ({logprob:.2f})", sim)

    if logprob < DESPERATE:
        return Choice(stream_text, "vosk", f"whisper unsure ({logprob:.2f})", sim)

    # Short audio is where Whisper invents rather than declines: handed half a
    # second it emits a fluent, confident continuation of whatever it thinks it
    # heard. Demand real confidence on short clips.
    if duration_s < SHORT_CLIP_S and logprob < CONFIDENT:
        return Choice(
            stream_text, "vosk",
            f"short clip ({duration_s:.1f}s) and unsure ({logprob:.2f})", sim,
        )

    # Wild disagreement plus low confidence on local tiny model: one of them is
    # badly wrong, and the streaming engine is the one that was listening incrementally.
    if sim < DIVERGENT and logprob < CONFIDENT:
        return Choice(
            stream_text, "vosk", f"divergent ({sim:.2f}) and unsure ({logprob:.2f})", sim
        )

    # Whisper sometimes collapses a sentence into a fragment. If it kept far
    # fewer words than Vosk heard, it lost content rather than cleaned it up.
    nw, ns = len(_words(whisper_text)), len(_words(stream_text))
    if ns >= 4 and nw / ns < SHORT_RATIO and logprob < CONFIDENT:
        return Choice(
            stream_text, "vosk", f"whisper dropped words ({nw}/{ns})", sim
        )

    return Choice(whisper_text, "whisper", f"confident ({logprob:.2f})", sim)
