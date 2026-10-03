# Streaming Transcription + Diarization

*Plan written 2026-08-23. Portfolio piece. Not started.*

**What it is:** live audio over a WebSocket → text streaming back as you speak,
with stable speaker labels. "Speaker 1: ... / Speaker 2: ..." updating in
real time.

**Why this one:** it demos in ten seconds with a microphone, it is scarce
(online diarization is fiddly and few people ship it), and it is the backend
proof of the existing portfolio thesis — *"I decide where the work runs."*

---

## 0. Hardware reality

**Dev machine: i7-7500U, 2 cores / 4 threads, 16GB.** 2016 dual-core mobile.

- Good enough to **develop** against with `faster-whisper` int8 on `tiny.en` /
  `base.en`, single stream.
- **Not** good enough to claim concurrency numbers. Diarization competes with
  transcription for the same 2 physical cores.

**Therefore:** develop locally, benchmark on a rented box. Make device + model
size a **config flag** and publish the CPU-vs-GPU tradeoff table as a headline
result rather than an apology.

| Profile | Model | Target | Purpose |
|---|---|---|---|
| `laptop` | `tiny.en` int8 | 1 stream, real-time | local dev |
| `cpu-server` | `base.en` / `small.en` int8 | N streams, measured | honest CPU story |
| `gpu` | `small` / `medium` fp16 | N streams, measured | the ceiling |

Do **not** guess these numbers. Measure them. Whatever they are is what ships.

---

## 1. Architecture

```
browser mic --WebSocket(PCM 16k mono)--> ingest
                                            |
                                    +-------+--------+
                                    |  ring buffer   |  (backpressure lives here)
                                    +-------+--------+
                                            v
                                        [ VAD ]  silero / webrtcvad
                                            |  speech segments only
                              +-------------+-------------+
                              v                           v
                      [ ASR worker ]              [ speaker embed ]
                   faster-whisper int8            ECAPA / resemblyzer
                     overlapping windows              per segment
                              |                           |
                              v                           v
                      partial + final              online clustering
                       hypotheses                  (stable label map)
                              +-------------+-------------+
                                            v
                                        merge + emit
                                            |
                              --WebSocket(JSON events)--> browser
```

### Event protocol

```json
{"type":"partial","text":"so what i was say","t0":12.4,"t1":13.9}
{"type":"final","text":"So what I was saying","speaker":1,"t0":12.4,"t1":14.1}
{"type":"speaker_merge","from":3,"to":1}
```

`partial` may be revised. `final` is committed. `speaker_merge` lets the client
retroactively relabel when clustering resolves two clusters into one — this is
what keeps labels from flip-flopping on screen.

---

## 2. The five hard parts

These are the engineering, and each one is a paragraph in the write-up.

**1. Chunking with overlap.** Fixed windows cut words in half. Use overlapping
windows and stitch on the overlap, or condition on the previous segment's
trailing context. Naive chunking is the number one reason toy demos read badly.

**2. VAD gating.** Never run ASR on silence — on 2 cores that is the difference
between real-time and not. Also gives natural segment boundaries for the
speaker embedder.

**3. Partial vs final hypotheses.** Emit fast, revise later. Text that
self-corrects as context arrives *feels* like a real product; waiting for
finals feels broken.

**4. Online diarization — the genuinely hard one.** Clustering speaker
embeddings incrementally, without knowing the speaker count up front, keeping
labels stable across time. Approach: embed each VAD segment, maintain running
centroids, assign by cosine distance with a threshold, spawn a new speaker when
nothing is close enough, and merge centroids that converge (emit
`speaker_merge`). Constrain to **2-4 speakers**.

**5. Backpressure.** Audio arrives in real time whether or not the CPU keeps
up. Bounded ring buffer plus an explicit drop-or-degrade policy (drop oldest?
downshift model?). State the policy. An unbounded queue that grows until OOM is
the classic failure, and naming it shows you know it.

---

## 3. Scope guard

**In:** 2-4 speakers, English, one WebSocket endpoint, a minimal browser client,
config-flag device/model profiles, benchmarks.

**Out — hard no:**

- No model training or fine-tuning
- No multi-language
- No speaker *identification* (naming real people), only separation
- No meeting-notes / summarization layer
- No auth, no accounts, no storage of audio

Online diarization can eat a month if accuracy is chased. Ship a
**good-enough, honestly-measured** version.

---

## 4. Measurement plan — build `bench/` alongside, not after

Three numbers, and they are the deliverable:

1. **Latency** — speech onset → first partial, and → committed final. p50 / p95.
   This is the number people care about most.
2. **Accuracy** — WER on a public set (LibriSpeech test-clean subset) and
   **DER** (diarization error rate) on a public 2-speaker set (AMI or CALLHOME
   subset). Report the exact subset used.
3. **Capacity** — concurrent streams sustained without falling behind real-time,
   per profile from section 0.

Plus a **real-time factor** (RTF) chart: processing time / audio duration,
across model sizes and devices. RTF < 1 is the survival line.

> Honesty rule: publish what is measured, including bad results. If DER is 18%,
> say 18% and say why. Anyone senior spots an inflated number instantly and it
> costs the whole piece.

---

## 5. Stack

- **FastAPI** WebSocket endpoint, `asyncio`
- **faster-whisper** (CTranslate2) — int8 on CPU, fp16 on GPU
- **silero-vad** (ONNX, tiny, CPU-cheap) or `webrtcvad`
- **speechbrain ECAPA-TDNN** or `resemblyzer` for speaker embeddings
- **numpy** ring buffer; ASR in a worker thread/process so the event loop never
  blocks
- **Vanilla JS client** — `AudioWorklet` → 16k mono PCM → WebSocket. No
  framework (on-brand, and keeps the demo page instant)
- Docker compose, one command

**Windows note:** develop in WSL2 or Docker if native Windows audio/threading
gets awkward. `faster-whisper` does work natively on Windows CPU.

---

## 6. Deliverables — the repo is not the deliverable

1. **Live deployed demo** — public URL, works with a laptop mic. Highest-value
   item by far.
2. **README opening with the measured numbers**, not a description.
3. **`docker compose up`** — works, no archaeology.
4. **60-90s demo video** — two people talking, labels resolving live.
5. **Case study page** on the portfolio site, same treatment as TENEBRA /
   Maison Noir.

---

## 7. Build order

**Phase 1 — make it stream. DONE 2026-08-24.** WebSocket in, PCM ring buffer,
VAD, ASR worker, partial/final events out, minimal browser client.

Built as planned except for one thing the plan got wrong: it assumed Whisper
alone could serve both partials and finals. It cannot — Whisper is an offline
30s-window model, so faked streaming arrives in ~1.8s lumps, and the only
Whisper small enough to keep up on this box is also the one that mangles
uncommon words. The live layer now runs **Vosk** (streaming-native, word-by-
word) with Whisper committing the accurate final. See the README.

Measured on the i7-7500U: first live word ~1.0s after speech starts, committed
final ~2.1s after it ends.

**Phase 2 — make it separate.** Speaker embeddings per VAD segment, online
clustering, stable label map, `speaker_merge` events, client relabeling.

**Phase 3 — make it provable.** `bench/` harness, the three numbers, the
CPU-vs-GPU profile table, deploy, video, case study.

Harness is built (`bench/latency.py`, `bench/threads.py`, `bench/compare.py`)
and the laptop column is measured — see the README table. **The remaining work
is a second machine**, not more code: this box cannot hold RTF < 1 at p95, so
the CPU-vs-GPU comparison needs a rented box before deploy, video or case
study. Everything else in Phase 3 is blocked on that.

Working before provable. A shipped Phase 1 demo beats a half-built Phase 2.

---

## 8. Open questions

- [ ] Where to rent the benchmark box (GPU hour vs. a fat CPU instance)
- [ ] Clustering threshold — needs a sweep, same precision/recall treatment as
      any threshold decision
- [ ] Drop-vs-degrade policy under backpressure: which one, and why
- [ ] Deploy target — the demo needs a persistent WebSocket, so serverless is
      likely out
- [ ] Does the demo page need a canned audio file fallback for people with no
      mic permission? (probably yes)
