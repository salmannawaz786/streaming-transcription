"""Per-connection stream state.

Owns the sample clock. Every timestamp the client sees is derived from the
count of samples received, not from wall time, so timings stay correct even
when the decoder falls behind or the network stutters.

Phase 1 emits `partial` and `final` events. `speaker` is always 0 — Phase 2
fills it in from online clustering, and the wire format already carries the
field so the client does not change.
"""

from __future__ import annotations

import logging
import os
import wave
from pathlib import Path
import time
from dataclasses import dataclass, field

import numpy as np

from .asr.streaming import StreamingAsr
from .asr.worker import AsrWorker, Job
from .audio.ringbuffer import RingBuffer
from .audio.vad import EnergyVad
from .config import SAMPLE_RATE, Profile

# A final decodes the whole utterance up to Whisper's own 30s window, plus
# whatever arrived while it was being submitted — so the buffer has to hold a
# bit more than 30s or the front of a very long utterance is lost again.
BUFFER_SECONDS = 35

log = logging.getLogger(__name__)

# Below this, an "utterance" is a breath or quick noise.
# Lowered to 0.35s so single-word responses ("yes", "no", "okay") are decoded.
MIN_UTTERANCE_S = 0.35

# Audio included from BEFORE the VAD fired. Increased to 0.40s to ensure
# leading plosives/fricatives and quiet first consonants are never clipped.
PREROLL_S = 0.40

# SD_DUMP=1 writes every committed utterance to debug/ as a wav. This is the
# only way to answer "is it the mic or the model?" — the same audio can then be
# run through several models offline with bench/compare.py.
DUMP_DIR = (
    Path(__file__).resolve().parents[1] / "debug"
    if os.getenv("SD_DUMP") == "1"
    else None
)


def _dump(audio: np.ndarray, session_id: str) -> None:
    if DUMP_DIR is None:
        return
    DUMP_DIR.mkdir(exist_ok=True)
    path = DUMP_DIR / f"{session_id}-{int(time.time() * 1000)}.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
    log.info("dumped %s (%.1fs)", path.name, audio.size / SAMPLE_RATE)


@dataclass
class Utterance:
    start_sample: int
    seq: int
    last_partial_at: float = 0.0
    # Wall-clock instant speech was first detected, for onset->first-partial
    # latency. This is the headline number in the benchmark.
    detected_at: float = field(default_factory=time.perf_counter)
    first_partial_sent: bool = False
    # Did the streaming engine hear any actual words during this utterance?
    heard_words: bool = False
    # Last streaming hypothesis, kept so the final can be arbitrated against
    # it rather than blindly replacing it.
    stream_text: str = ""


class Session:
    def __init__(self, session_id: str, profile: Profile, worker: AsrWorker) -> None:
        self.id = session_id
        self.profile = profile
        self.worker = worker

        self.buffer = RingBuffer(SAMPLE_RATE * BUFFER_SECONDS)
        # ~480ms of quiet closes an utterance (24 frames @ 20ms).
        # Fast enough to feel responsive while preserving mid-sentence pauses.
        self.vad = EnergyVad(hangover_frames=24)
        # Previous committed line, fed to the next final as decoder context.
        self.last_final = ""

        # Live layer. When Vosk is available it owns partials entirely and
        # Whisper is left to do only what it is good at: the committed final.
        self.stream: StreamingAsr | None = (
            StreamingAsr(SAMPLE_RATE) if StreamingAsr.available() else None
        )
        self.samples_seen = 0
        self.utterance: Utterance | None = None
        self._seq = 0
        self._gen = 0
        # At most one partial in flight per session. The interval alone is not
        # enough: if a decode takes longer than the interval, work is created
        # faster than it drains and the queue grows without bound. Measured on
        # a 2-core i7-7500U, tiny.en decodes in ~870ms against a 600ms
        # interval — exactly that situation.
        self.partial_inflight = False
        # seq -> the Utterance it belongs to, so latency can be attributed when
        # the result comes back off the worker thread.
        self.pending: dict[int, Utterance] = {}
        # How often each engine won arbitration. Surfaced on /health so the
        # tiering can be judged on numbers rather than impressions.
        self.arbitration: dict[str, int] = {"whisper": 0, "vosk": 0}
        # GPU tier is opt-in per session so an open page never burns credits.
        self.use_remote = False

    def next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def feed(self, pcm: np.ndarray, pcm_bytes: bytes | None = None) -> list[dict]:
        """Feed float32 mono 16k audio. Returns events to send to the client.

        `pcm_bytes` is the original Int16 payload, passed through untouched for
        the streaming recogniser — converting back from float would be lossy
        and pointless.
        """
        self.buffer.write(pcm)
        # Advance the sample clock BEFORE draining VAD events. The VAD counts
        # the frames it just consumed, so if this lags the two clocks disagree
        # and `_submit` computes a negative `age` — which silently truncates
        # every final to a few hundred samples.
        self.samples_seen += pcm.size
        events: list[dict] = []

        for ev in self.vad.push(pcm):
            if ev.kind == "speech_start":
                self.utterance = Utterance(
                    start_sample=ev.at_sample, seq=self.next_seq()
                )
                events.append({"type": "speech_start", "t": ev.at_sample / SAMPLE_RATE})
            elif ev.kind == "speech_end" and self.utterance:
                # Backstop only. Kaldi's endpointer normally closes the
                # utterance first; this catches the case where the streaming
                # engine is unavailable, or missed the boundary entirely.
                self._close_utterance(ev.at_sample, events)

        # Live layer: Vosk streams words as they are recognised.
        if self.stream is not None and pcm_bytes:
            t0 = time.perf_counter()
            text, endpoint = self.stream.feed(pcm_bytes)
            feed_ms = (time.perf_counter() - t0) * 1000
            if text and self.utterance is not None:
                self.utterance.heard_words = True
                self.utterance.stream_text = text
                events.append(
                    {
                        "type": "partial",
                        "seq": self.utterance.seq,
                        "text": text,
                        "t0": round(self.utterance.start_sample / SAMPLE_RATE, 2),
                        "t1": round(self.samples_seen / SAMPLE_RATE, 2),
                        "speaker": 0,
                        "engine": "vosk",
                        "feed_ms": round(feed_ms, 1),
                    }
                )
                if not self.utterance.first_partial_sent:
                    self.utterance.first_partial_sent = True
                    events.append(
                        {
                            "type": "stats",
                            "kind": "partial",
                            "model": "vosk",
                            "audio_s": 0.0,
                            "rtf": 0.0,
                            "decode_ms": round(feed_ms),
                            "queue_ms": 0,
                            "first_partial_ms": round(
                                (time.perf_counter() - self.utterance.detected_at) * 1000
                            ),
                        }
                    )
            if endpoint and self.utterance is not None:
                # Kaldi decided the utterance is complete. Trust it over the
                # energy VAD: the endpointer reasons about speech structure and
                # word completion, while a loudness threshold cannot tell a
                # mid-sentence thinking pause from the end of a sentence.
                self._close_utterance(self.samples_seen, events)

            return events + self._drain_dropped()

        # Fallback: no streaming model, so Whisper does speculative partials.
        if self.utterance is not None and not self.partial_inflight:
            now = time.perf_counter()
            due = now - self.utterance.last_partial_at >= self.profile.partial_interval_s
            if due:
                self.utterance.last_partial_at = now
                self.partial_inflight = True
                self._submit(self.utterance, self.samples_seen, kind="partial")

        return events + self._drain_dropped()

    def _close_utterance(self, end_sample: int, events: list[dict]) -> None:
        """Commit the current utterance. Both boundary sources funnel here."""
        utt = self.utterance
        if utt is None:
            return

        # Cross-engine gate. Kaldi returns nothing on silence; it does not
        # invent words. So if the streaming engine heard nothing across the
        # whole utterance, the boundary came from noise and any Whisper output
        # would be pure hallucination. Skip the decode entirely — cheaper than
        # running it and discarding the result.
        if self.stream is not None and not utt.heard_words:
            log.debug("utterance %d: no words from stream, skipping final", utt.seq)
        else:
            self._submit(utt, end_sample, kind="final")

        events.append({"type": "speech_end", "t": end_sample / SAMPLE_RATE})
        self.utterance = None
        if self.stream:
            self.stream.reset()

    def _drain_dropped(self) -> list[dict]:
        if not self.buffer.dropped_samples:
            return []
        seconds = round(self.buffer.dropped_samples / SAMPLE_RATE, 2)
        self.buffer.dropped_samples = 0
        return [{"type": "dropped", "seconds": seconds}]

    def _submit(self, utt: Utterance, end_sample: int, kind: str) -> None:
        # `context_s` bounds a speculative partial, which is re-decoded
        # constantly and only needs recent context. A final must cover the WHOLE
        # utterance: capping it silently truncated the front of any sentence
        # longer than the window — a 4.1s "Streaming transcription with speaker
        # diarization on a laptop" came back as "Production with speaker
        # diarization on a laptop", having lost its first second.
        #
        # Whisper's own window is 30s, and cost grows only ~1.6x for 8x the
        # audio, so decoding the full utterance is close to free.
        max_samples = int(
            (self.profile.context_s if kind == "partial" else 30.0) * SAMPLE_RATE
        )
        start_sample = utt.start_sample
        if kind == "final":
            start_sample = max(0, start_sample - int(PREROLL_S * SAMPLE_RATE))
        length = min(end_sample - start_sample, max_samples)
        floor = int(MIN_UTTERANCE_S * SAMPLE_RATE) if kind == "final" else SAMPLE_RATE // 5
        if length < floor:
            if kind == "partial":
                self.partial_inflight = False
            return

        # How far the stream has advanced past the window we want to decode.
        age = max(0, self.samples_seen - end_sample)
        audio = self.buffer.read_last(length + age)
        if age:
            audio = audio[:-age]
        if audio.size == 0:
            if kind == "partial":
                self.partial_inflight = False
            return

        if kind == "final":
            _dump(audio, self.id)

        seq = utt.seq if kind == "partial" else self.next_seq()
        self._gen += 1
        self.pending[seq] = utt
        self.worker.submit(
            Job(
                session_id=self.id,
                kind=kind,
                audio=audio,
                seq=seq,
                gen=self._gen,
                prompt=self.last_final if kind == "final" else "",
                remote=self.use_remote and kind == "final",
                t0=(end_sample - audio.size) / SAMPLE_RATE,
            )
        )

    def close(self) -> None:
        self.worker.forget(self.id)
        self.buffer.clear()
