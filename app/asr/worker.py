"""ASR worker thread.

One model, one worker thread, shared by every session. On a 2-core box there is
nothing to gain from parallel decoders — they would just fight for the same
cores and make every stream worse. Serializing keeps latency predictable, and
predictable is what the benchmark measures.

The thread matters: faster-whisper spends its time in CTranslate2 C++ with the
GIL released, so decoding a 3-second window does not stall the event loop and
block the WebSocket reads.

Partial jobs are droppable — if the queue backs up, stale partials are
discarded rather than decoded late. Final jobs are never dropped; they are the
committed transcript.
"""

from __future__ import annotations

import asyncio
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from ..config import Profile, hotwords
from . import remote

log = logging.getLogger(__name__)
_HOTWORDS = hotwords()
_PROMPT_CARRY = os.getenv("SD_PROMPT_CARRY", "") == "1"

# Phrases Whisper reaches for when there is nothing to transcribe. Matched only
# against the WHOLE output — "thank you" as a complete utterance on quiet audio
# is an artifact; "thank you" inside a longer sentence is somebody talking.
_FILLER = {
    "thank you", "thank you.", "thanks for watching", "thanks for watching!",
    "you", "bye", "bye.", "okay", "ok", ".", "so", "uh", "um",
    "thank you very much", "thank you very much.", "please subscribe",
    "subtitles by the amara.org community",
}

# Above this, Whisper itself believes the audio is not speech.
NO_SPEECH_MAX = 0.6
# Average token log-probability. Below this the decode is a guess.
LOGPROB_MIN = -1.0


def _is_hallucination(text: str, no_speech: float, logprob: float) -> bool:
    if no_speech > NO_SPEECH_MAX:
        return True
    normalised = text.lower().strip()
    if normalised in _FILLER:
        # Filler is only suspicious when the model was also unsure. A confident
        # short "Okay" from someone actually saying it should survive.
        return no_speech > 0.2 or logprob < LOGPROB_MIN
    return logprob < LOGPROB_MIN and len(normalised.split()) <= 3


@dataclass
class Job:
    session_id: str
    kind: str            # "partial" | "final"
    audio: np.ndarray
    seq: int             # client-facing line identity; constant across the
                         # partials of one utterance so the UI can replace in place
    gen: int             # monotonic per session; used ONLY for staleness
    t0: float            # stream-relative start time of this audio, seconds
    # Text of the previous committed final, used as decoder context only when
    # SD_PROMPT_CARRY=1. Off by default — see _decode for why it backfires.
    prompt: str = ""
    # Route this final to the GPU endpoint. Per-job rather than global so the
    # browser can switch tiers mid-session without a restart, and so an idle
    # page never spends GPU credits.
    remote: bool = False
    queued_at: float = field(default_factory=time.perf_counter)


@dataclass
class Result:
    session_id: str
    kind: str
    seq: int
    text: str
    t0: float
    t1: float
    decode_ms: float
    queue_ms: float
    rtf: float           # decode time / audio duration; must stay under 1.0
    model: str = ""
    # "gpu" | "local". Reported to the browser so the GPU tier's state is
    # observable instead of guessed at.
    tier: str = "local"
    fell_back: bool = False
    logprob: float = 0.0   # average token log-probability; how sure Whisper is
    no_speech: float = 0.0
    dropped: bool = False  # superseded before it was decoded


def _remote_result(job: Job, payload: dict, started: float, queue_ms: float) -> Result:
    """Shape a GPU response like a local decode so nothing downstream cares."""
    decode_ms = (time.perf_counter() - started) * 1000
    duration_s = job.audio.size / 16_000
    text = payload.get("text", "").strip()
    logprob = float(payload.get("logprob", 0.0))
    no_speech = float(payload.get("no_speech", 0.0))
    if text and _is_hallucination(text, no_speech, logprob):
        text = ""
    return Result(
        session_id=job.session_id,
        kind=job.kind,
        seq=job.seq,
        text=text,
        t0=job.t0,
        t1=job.t0 + duration_s,
        decode_ms=decode_ms,
        queue_ms=queue_ms,
        rtf=(decode_ms / 1000) / max(duration_s, 1e-6),
        model=f"beam:{payload.get('model', 'large-v3-turbo')}",
        tier="gpu",
        logprob=logprob,
        no_speech=no_speech,
    )


class AsrWorker:
    def __init__(self, profile: Profile, loop: asyncio.AbstractEventLoop) -> None:
        self.profile = profile
        self._loop = loop
        self._jobs: queue.Queue[Job | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._fast = None      # partials
        self._accurate = None  # finals
        self.ready = threading.Event()
        self.results: asyncio.Queue[Result] = asyncio.Queue()
        # Latest partial generation per session; anything older is stale.
        self._latest_partial: dict[str, int] = {}

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="asr", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._jobs.put(None)

    def submit(self, job: Job) -> None:
        if job.kind == "partial":
            self._latest_partial[job.session_id] = job.gen
        self._jobs.put(job)

    def forget(self, session_id: str) -> None:
        self._latest_partial.pop(session_id, None)

    def _run(self) -> None:
        from faster_whisper import WhisperModel

        p = self.profile
        kw = {"device": p.device, "compute_type": p.compute_type}
        if p.cpu_threads:
            kw["cpu_threads"] = p.cpu_threads
            kw["num_workers"] = 1

        t0 = time.perf_counter()
        log.info("loading %s (partials) on %s", p.fast_model, p.device)
        self._fast = WhisperModel(p.fast_model, **kw)

        if p.accurate_model == p.fast_model:
            self._accurate = self._fast
        else:
            log.info("loading %s (finals) on %s", p.accurate_model, p.device)
            self._accurate = WhisperModel(p.accurate_model, **kw)

        log.info("models ready in %.1fs", time.perf_counter() - t0)
        self.ready.set()

        while True:
            job = self._jobs.get()
            if job is None:
                return

            # Drop superseded partials instead of decoding them late. Still
            # emit a result, otherwise the session's in-flight gate never
            # clears and it stops asking for partials entirely.
            if (
                job.kind == "partial"
                and self._latest_partial.get(job.session_id, job.gen) != job.gen
            ):
                self._emit(
                    Result(
                        session_id=job.session_id,
                        kind="partial",
                        seq=job.seq,
                        text="",
                        t0=job.t0,
                        t1=job.t0,
                        decode_ms=0.0,
                        queue_ms=(time.perf_counter() - job.queued_at) * 1000,
                        rtf=0.0,
                        dropped=True,
                    )
                )
                continue

            try:
                self._decode(job)
            except Exception:
                log.exception("decode failed for %s seq=%d", job.session_id, job.seq)

    def _decode(self, job: Job) -> None:
        queue_ms = (time.perf_counter() - job.queued_at) * 1000
        started = time.perf_counter()
        fell_back = False
        fallback_reason = ""

        is_partial = job.kind == "partial"
        model = self._fast if is_partial else self._accurate
        model_name = (
            self.profile.fast_model if is_partial else self.profile.accurate_model
        )

        opts: dict = {
            "language": "en",
            # Beam search costs real time for a hypothesis that is about to be
            # thrown away. Spend it only on the committed line.
            "beam_size": 1 if is_partial else 5,
            "vad_filter": False,       # already gated on VAD upstream
            "condition_on_previous_text": False,
            "without_timestamps": True,
        }
        prompt_val = None
        if not is_partial:
            # Only hotwords by default. Carrying the previous committed line
            # forward as context sounds obviously good and backfires badly:
            # on unclear audio Whisper REGURGITATES its prompt rather than
            # transcribing, which produced the same sentence three times in a
            # row on live mic input. Opt in with SD_PROMPT_CARRY=1.
            parts = [_HOTWORDS]
            if _PROMPT_CARRY:
                parts.insert(0, job.prompt)
            if prompt := " ".join(x for x in parts if x).strip():
                opts["initial_prompt"] = prompt
                prompt_val = prompt
            # Whisper can also fall into a repetition loop within one decode.
            opts["repetition_penalty"] = 1.1

        if job.remote and not is_partial:
            try:
                payload = remote.transcribe(job.audio, prompt=prompt_val)
                self._emit(_remote_result(job, payload, started, queue_ms))
                return
            except remote.RemoteError as exc:
                # Never lose an utterance to a cold start or a network blip.
                log.warning("remote tier failed (%s) - falling back to %s",
                            exc, model_name)
                fell_back = True
                fallback_reason = str(exc)[:120]

        segs = list(model.transcribe(job.audio, **opts)[0])
        text = "".join(s.text for s in segs).strip()

        # Whisper hallucinates on silence. Fed near-silence it emits confident
        # filler — "Thank you.", "Thanks for watching!", "you" — because its
        # training data is full of video captions that end that way. It is not
        # a bug in the audio path; the model genuinely believes it.
        no_speech = max((s.no_speech_prob for s in segs), default=0.0)
        logprob = min((s.avg_logprob for s in segs), default=0.0)
        if text and _is_hallucination(text, no_speech, logprob):
            log.debug(
                "dropped hallucination %r (no_speech=%.2f logprob=%.2f)",
                text, no_speech, logprob,
            )
            text = ""

        decode_ms = (time.perf_counter() - started) * 1000
        duration_s = job.audio.size / 16_000

        self._emit(
            Result(
                session_id=job.session_id,
                kind=job.kind,
                seq=job.seq,
                text=text,
                t0=job.t0,
                t1=job.t0 + duration_s,
                decode_ms=decode_ms,
                queue_ms=queue_ms,
                rtf=(decode_ms / 1000) / max(duration_s, 1e-6),
                model=model_name,
                tier="local",
                fell_back=fell_back,
                logprob=logprob,
                no_speech=no_speech,
            )
        )

    def _emit(self, result: Result) -> None:
        self._loop.call_soon_threadsafe(self.results.put_nowait, result)
