"""Voice activity detection.

VAD gating is not an optimization here, it is the reason CPU-only works at all:
we never pay for decoding silence. On two physical cores that is the difference
between keeping up with real time and falling behind it.

Phase 1 ships an energy VAD with an adaptive noise floor and hysteresis. Zero
dependencies, no model download, works everywhere. It is genuinely worse than
silero-vad in noisy rooms — it keys on loudness, so a slammed door reads as
speech. The interface is a single `push()` returning events, so swapping in a
silero ONNX backend later is a drop-in change and a fair A/B for the write-up.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import FRAME_SAMPLES, SAMPLE_RATE


@dataclass
class VadEvent:
    kind: str  # "speech_start" | "speech_end"
    at_sample: int


class EnergyVad:
    def __init__(
        self,
        onset_frames: int = 3,        # ~60ms of speech before we believe it
        hangover_frames: int = 25,    # ~500ms of quiet before we close a segment
        trigger_ratio: float = 4.0,   # speech is this much above the noise floor
        floor_alpha: float = 0.995,   # noise floor adaptation (slow)
        abs_floor: float = 0.006,     # absolute RMS gate, see below
    ) -> None:
        self.onset_frames = onset_frames
        self.hangover_frames = hangover_frames
        self.trigger_ratio = trigger_ratio
        self.floor_alpha = floor_alpha
        # A purely relative threshold is scale-free, which sounds elegant and
        # behaves badly: in a quiet room the adaptive floor drops until a
        # keystroke or a breath is 4x above it, the VAD opens, and Whisper
        # hallucinates "Thank you." over half a second of silence. The absolute
        # gate says a frame must also be audible in real terms.
        self.abs_floor = abs_floor

        self._floor: float | None = None
        self._active = False
        self._speech_run = 0
        self._silence_run = 0
        self._samples_seen = 0
        self._pending: list[np.ndarray] = []
        self._pending_len = 0

    @property
    def active(self) -> bool:
        return self._active

    def push(self, samples: np.ndarray) -> list[VadEvent]:
        """Feed arbitrary-length audio; get back segment boundary events."""
        self._pending.append(samples)
        self._pending_len += samples.size
        events: list[VadEvent] = []

        while self._pending_len >= FRAME_SAMPLES:
            block = np.concatenate(self._pending)
            frame, rest = block[:FRAME_SAMPLES], block[FRAME_SAMPLES:]
            self._pending = [rest] if rest.size else []
            self._pending_len = rest.size
            events.extend(self._push_frame(frame))

        return events

    def _push_frame(self, frame: np.ndarray) -> list[VadEvent]:
        self._samples_seen += frame.size
        rms = float(np.sqrt(np.mean(frame * frame)) + 1e-9)

        if self._floor is None:
            self._floor = rms
        is_speech = rms > self._floor * self.trigger_ratio and rms > self.abs_floor

        # Only adapt the floor on non-speech, otherwise a long utterance drags
        # the threshold up and the tail of the sentence gets cut.
        if not is_speech:
            self._floor = self.floor_alpha * self._floor + (1 - self.floor_alpha) * rms

        events: list[VadEvent] = []
        if is_speech:
            self._speech_run += 1
            self._silence_run = 0
            if not self._active and self._speech_run >= self.onset_frames:
                self._active = True
                # Backdate the start so the onset frames are not clipped off.
                events.append(
                    VadEvent("speech_start", self._samples_seen - self._speech_run * FRAME_SAMPLES)
                )
        else:
            self._silence_run += 1
            self._speech_run = 0
            if self._active and self._silence_run >= self.hangover_frames:
                self._active = False
                events.append(VadEvent("speech_end", self._samples_seen))

        return events


def seconds(n_samples: int) -> float:
    return n_samples / SAMPLE_RATE
