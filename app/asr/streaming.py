"""Streaming-native ASR for the live layer (Vosk / Kaldi).

Why this exists
---------------
Whisper is an offline model with a fixed 30-second window. Pumping it
repeatedly to fake streaming is what made live text arrive in ~1.5s lumps
instead of word by word — the decode cost is dominated by that fixed window, so
asking for a shorter update does not make it cheaper.

Vosk is built the other way round: a streaming recogniser that consumes audio
incrementally and grows its hypothesis word by word, designed to run real-time
on weak CPUs. It is worse than Whisper at uncommon words, which is exactly why
it is used only for the disposable layer:

    partials -> Vosk      (instant, word-by-word, revised constantly)
    finals   -> Whisper   (accurate, committed, runs after speech ends)

Each model does the job it is actually good at.

Runs inline on the event loop rather than in a worker thread: the small model
processes a 20ms chunk in single-digit milliseconds, and a queue hop would add
more latency than the decode costs. If that stops being true on a given box,
`feed_ms` in the stats will show it.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "models"


def model_path() -> Path | None:
    """Locate the Vosk model, or None if it is not installed.

    Absence is not an error — the service falls back to Whisper partials and
    still works, just chunkier. Keeping the fallback means the repo runs for
    someone who has not downloaded a 40MB model yet.
    """
    if env := os.getenv("SD_VOSK_MODEL"):
        p = Path(env)
        return p if p.is_dir() else None
    if not DEFAULT_MODEL_DIR.is_dir():
        return None
    for child in sorted(DEFAULT_MODEL_DIR.iterdir()):
        if child.is_dir() and (child / "am").is_dir():
            return child
    return None


class StreamingAsr:
    """Wraps one Vosk recogniser. One instance per session."""

    _model = None  # class-level: loading is expensive, the model is stateless

    def __init__(self, sample_rate: int = 16_000) -> None:
        from vosk import KaldiRecognizer

        self._rec = KaldiRecognizer(self._model, float(sample_rate))
        self._rec.SetWords(False)
        self._last = ""

    @classmethod
    def available(cls) -> bool:
        return cls._model is not None

    @classmethod
    def load(cls) -> bool:
        """Load the shared model. Returns False if unavailable."""
        path = model_path()
        if path is None:
            log.info("no Vosk model found — falling back to Whisper for partials")
            return False
        try:
            from vosk import Model, SetLogLevel

            SetLogLevel(-1)  # Kaldi is extremely chatty on stderr otherwise
            log.info("loading Vosk model from %s", path.name)
            cls._model = Model(str(path))
            return True
        except Exception:
            log.exception("Vosk failed to load — falling back to Whisper partials")
            return False

    def feed(self, pcm_bytes: bytes) -> tuple[str | None, bool]:
        """Feed Int16 PCM. Returns (changed_text_or_None, endpoint_reached).

        `None` on no-change matters: Vosk emits a partial for every chunk, and
        pushing an identical string 50 times a second is pure noise on the wire
        and pointless re-renders in the browser.

        `endpoint_reached` is the important half. Kaldi's endpointer decides an
        utterance is over using speech structure — whether the last word is
        complete, how long the trailing silence is relative to what was said —
        rather than a fixed silence timer. An energy VAD cannot do that, and
        the difference showed up directly in output quality: a threshold-based
        boundary cut sentences mid-thought, and Whisper handed a fragment does
        not transcribe it, it completes it from a template. That is where
        "Where is your computer?" / "Where is your water?" / "There's my pen."
        came from — one template, filled differently each time.
        """
        endpoint = bool(self._rec.AcceptWaveform(pcm_bytes))
        if endpoint:
            text = json.loads(self._rec.Result()).get("text", "")
        else:
            text = json.loads(self._rec.PartialResult()).get("partial", "")

        text = text.strip()
        if not text or text == self._last:
            return None, endpoint
        self._last = text
        return text, endpoint

    def reset(self) -> None:
        """Drop state at an utterance boundary so the next line starts clean."""
        self._rec.Reset()
        self._last = ""
