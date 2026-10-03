"""Tests for two-engine arbitration.

    python -m tests.test_arbitrate
"""

from __future__ import annotations

from app.arbitrate import choose, similarity


def test_confident_whisper_corrects_the_stream() -> None:
    """The case the whole tiered design exists for."""
    c = choose(
        "streaming transcription with speaker die rise asian on a laptop",
        "Streaming transcription with speaker diarization on a laptop.",
        logprob=-0.25,
    )
    assert c.source == "whisper", c
    assert "diarization" in c.text


def test_unsure_whisper_does_not_overwrite() -> None:
    """The reported bug: a right answer replaced by a worse guess."""
    c = choose("i can move", "Never see your house.", logprob=-1.4)
    assert c.source == "vosk", c
    assert c.text == "i can move"


def test_divergent_and_middling_keeps_stream() -> None:
    c = choose(
        "why are you acting like this",
        "Babe is your house.",
        logprob=-0.75,
    )
    assert c.source == "vosk", c


def test_divergent_but_confident_trusts_whisper() -> None:
    """Low similarity alone is not enough — Vosk mangling every word looks
    divergent precisely when Whisper is most useful."""
    c = choose("die rise asian", "diarization", logprob=-0.20)
    assert c.source == "whisper", c


def test_whisper_dropping_words_keeps_stream() -> None:
    c = choose(
        "the quick brown fox jumps over the lazy dog",
        "The dog.",
        logprob=-0.70,
    )
    assert c.source == "vosk", c


def test_short_confident_whisper_still_wins() -> None:
    """A genuinely short utterance must not trip the dropped-words rule."""
    c = choose("okay", "Okay.", logprob=-0.2)
    assert c.source == "whisper", c


def test_short_clip_gets_extra_scrutiny() -> None:
    """Observed live: a fragment of "I am from Pakistan" came back as
    "I am from India", then "Korea", then "China" — Whisper completing a
    half-second scrap rather than transcribing it."""
    c = choose("i'm from", "I am from India.", logprob=-0.62, duration_s=0.8)
    assert c.source == "vosk", c


def test_short_clip_with_confident_whisper_still_commits() -> None:
    c = choose("pakistan", "Pakistan.", logprob=-0.18, duration_s=0.8)
    assert c.source == "whisper", c


def test_long_clip_unaffected_by_short_rule() -> None:
    c = choose(
        "i am from pakistan and i live in lahore",
        "I am from Pakistan and I live in Lahore.",
        logprob=-0.62, duration_s=4.0,
    )
    assert c.source == "whisper", c


def test_empty_whisper_falls_back() -> None:
    c = choose("hello there", "", logprob=-0.1)
    assert c.source == "vosk" and c.text == "hello there"


def test_empty_stream_uses_whisper() -> None:
    c = choose("", "Hello there.", logprob=-0.9)
    assert c.source == "whisper"


def test_similarity_ignores_case_and_punctuation() -> None:
    assert similarity("hello there", "Hello, there!") == 1.0
    assert similarity("", "") == 1.0
    assert similarity("abc", "") == 0.0


def test_gpu_tier_wins_over_mangled_stream() -> None:
    """GPU tier with large-v3-turbo should correct mangled Vosk output."""
    c = choose(
        "speaker die rise asian",
        "Speaker diarization.",
        logprob=-0.65,
        tier="gpu",
    )
    assert c.source == "whisper", c
    assert c.text == "Speaker diarization."


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  ok   {name}")
            except AssertionError as exc:
                failures += 1
                print(f"  FAIL {name}: {exc}")
    raise SystemExit(failures)
