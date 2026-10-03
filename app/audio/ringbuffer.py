"""Bounded ring buffer — this is where backpressure lives.

Audio arrives in real time whether or not the decoder keeps up. The buffer is
deliberately fixed-size: when it overflows we drop the OLDEST audio and count
it, rather than growing a queue until the process dies.

Dropping oldest (not newest) is the right call for live captioning: a listener
would rather lose a stale second than have the transcript fall permanently
behind the speaker. `dropped_samples` is surfaced to the client so the demo can
show honestly when it is losing audio.
"""

from __future__ import annotations

import numpy as np


class RingBuffer:
    def __init__(self, capacity_samples: int) -> None:
        self._buf = np.zeros(capacity_samples, dtype=np.float32)
        self._capacity = capacity_samples
        self._write = 0
        self._size = 0
        self.dropped_samples = 0

    def __len__(self) -> int:
        return self._size

    @property
    def capacity(self) -> int:
        return self._capacity

    def write(self, samples: np.ndarray) -> None:
        n = samples.size
        if n == 0:
            return
        if n >= self._capacity:
            # A single write larger than the whole buffer: keep only the tail.
            self.dropped_samples += n - self._capacity
            samples = samples[-self._capacity:]
            n = self._capacity

        end = self._write + n
        if end <= self._capacity:
            self._buf[self._write:end] = samples
        else:
            split = self._capacity - self._write
            self._buf[self._write:] = samples[:split]
            self._buf[: end - self._capacity] = samples[split:]

        self._write = end % self._capacity
        overflow = self._size + n - self._capacity
        if overflow > 0:
            self.dropped_samples += overflow
            self._size = self._capacity
        else:
            self._size += n

    def read_last(self, n: int) -> np.ndarray:
        """Most recent `n` samples, oldest first. Does not consume."""
        n = min(n, self._size)
        if n == 0:
            return np.zeros(0, dtype=np.float32)
        start = (self._write - n) % self._capacity
        if start + n <= self._capacity:
            return self._buf[start:start + n].copy()
        split = self._capacity - start
        return np.concatenate([self._buf[start:], self._buf[: n - split]])

    def clear(self) -> None:
        self._size = 0
        self._write = 0
