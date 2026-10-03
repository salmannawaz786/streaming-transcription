# Streaming Transcription

Live audio over a WebSocket, transcript streaming back as you speak.

> **Status: Phase 1 (live transcription) is complete and measured. Speaker
> diarization is *not built yet* (Phase 2)** — the wire format already carries a
> `speaker` field, currently always `0`. Nothing in this repo claims otherwise.
> See [PLAN.md](PLAN.md).

## Measured

`bench/latency.py`, 4 passes over a 7.8s fixture, i7-7500U (2 physical /
4 logical cores, 2016 dual-core ULV), warm-up pass discarded:

| | `laptop`<br>tiny/tiny | `laptop-tiered`<br>tiny/base | **+ GPU tier**<br>tiny / turbo on A10G |
|---|---|---|---|
| onset → **first word** p50 | 632ms | 525ms | **585ms** |
| onset → first word p95 | 814ms | 929ms | 675ms |
| onset → committed final p50 | 4513ms | 7140ms | **1594ms** |
| onset → committed final p95 | 7027ms | 7866ms | 5872ms |
| final RTF p50 | 0.78 | 1.20 | **0.47** |
| final RTF p95 | 1.74 | 1.45 | **0.60** |
| keeps up with real time? | no | no | **yes** |

Three things this table says, in order of importance:

**1. The live layer never changed.** ~0.5–0.6s to first word in all three
columns, because it is always Vosk on the local CPU. The GPU does not make
words appear faster; it makes the committed line correct.

**2. Only the GPU column holds RTF < 1.0.** That is the line between a service
that keeps up and one that falls progressively behind under continuous speech.
`base.en` exceeds real time outright (1.20 at p50); `tiny.en` passes at p50 and
fails at p95. The in-flight gate stops the queue exploding, but it cannot
manufacture CPU.

**3. Commit latency drops ~2.8x** with the GPU tier — 4513ms → 1594ms — while
also being far more accurate. Faster *and* better, because the bottleneck was
never the algorithm.

Caveats, stated rather than buried:

- **p95 on 6–8 samples is effectively "worst observed"**, not a real 95th
  percentile. Treat the p95 column as a warning flag, not a measurement. The
  GPU p95 of 5872ms is one slow round-trip, not typical behaviour.
- Round-trip includes network from a home connection to `us-east-1`. Most of
  the GPU tier's 1.6s is network, not inference: the endpoint itself does 7.8s
  of audio in ~1.6s wall including transfer (RTF 0.21, `bench/beam_check.py`).
- Cold start is ~25s. Irrelevant to these numbers, relevant to a live demo —
  warm it first.
- Whisper won 100% of committed lines on this fixture. Correct on clean TTS,
  but it means **arbitration is untested against real accented speech**.
  Capture some with `SD_DUMP=1` and tune before trusting the thresholds.

## Run it

```bash
py -3.11 -m venv .venv && .venv/Scripts/python.exe -m pip install -r requirements.txt
```

Then fetch the Vosk model for the live layer (~40MB, optional — without it the
service falls back to Whisper partials):

```bash
.venv/Scripts/python.exe -m bench.get_model
```

```bash
SD_PROFILE=laptop .venv/Scripts/python.exe -m uvicorn app.server:app --port 8100
```

Open <http://localhost:8100>, press start, talk. First run downloads the
Whisper weights (~75MB for `tiny.en`), so give it a moment.

## Profiles

Device and model size are configuration, not code — the same service runs on a
2-core laptop and on a GPU box, and the difference between them is the headline
result of the write-up.

| `SD_PROFILE` | Partials | Finals | Intended for |
|---|---|---|---|
| `laptop` (default) | `tiny.en` int8 | `tiny.en` int8 | 2-core dev machine, real-time |
| `laptop-tiered` | `tiny.en` int8 | `base.en` int8 | same box, accuracy over latency |
| `cpu-server` | `base.en` int8 | `small.en` int8 | honest CPU capacity numbers |
| `gpu` | `small` fp16 | `medium` fp16 | the ceiling |

Bias decoding toward vocabulary that actually occurs — the cheapest accuracy
win available for proper nouns and jargon:

```bash
SD_HOTWORDS="Kubernetes, SalluLabs, diarization, CTranslate2"
```

## Why two engines

Whisper is an **offline** model with a fixed 30-second window. Pumping it
repeatedly to fake streaming is what made live text arrive in ~1.8s lumps
rather than word by word — and the small model fast enough to keep up is also
the one that is bad at uncommon words. Both complaints trace to that one
mismatch.

So the live layer uses a model actually designed for streaming:

| | engine | job |
|---|---|---|
| **partials** | Vosk (Kaldi) | instant, word-by-word, revised constantly, disposable |
| **finals** | Whisper | accurate, committed, runs after speech ends |

Each does what it is good at, and a run over the TTS fixture shows the shape:

```
 0.5s  speech_start
 1.4s  partial  the
 2.2s  partial  the quick brown
 2.8s  partial  the quick brown fox
 ...
 3.5s  speech_end          <- Kaldi endpoint, not a silence timer
 5.1s  FINAL    The quick brown fox jumps over the lazy dog.
```

Roughly 15 partial updates over 7.8s of speech; see the table above for p50/p95.

The division of labour is visible in one line of the fixture. Vosk hears
*"speaker die rise asian"*; Whisper's final commits *"speaker diarization"*.
The fast engine is wrong in exactly the way the slow one fixes.

Vosk is optional. Without the model the service falls back to Whisper partials
and still works — just chunkier.

### Kaldi decides where sentences end

The streaming engine does more than produce words. Vosk carries a trained
endpointer that reasons about *speech structure* — whether the last word is
complete, how long the trailing silence is relative to what was said — and it
is far better at this than a loudness threshold can ever be.

That matters more than it sounds, because Whisper handed a fragment does not
transcribe it, it **completes** it. Boundaries in the wrong place produce
fluent, confident, wrong output rather than obviously broken output. From a
real session on an energy-VAD boundary:

```
Where is your computer?     Where is your water?     Where is the big?
There's my pen.             Can I help you phone?    Grant, I kiss you.
```

One template, filled differently each time — the signature of a model given
half a sentence. Meanwhile every *long* utterance in the same session came back
correct ("Can you please leave my room and lock the door and go away for one
year?"). The failures correlated with length, not with accent or microphone.

So Kaldi's endpoint closes the utterance and the energy VAD is demoted to a
backstop for when the streaming engine is unavailable. Committing also got
*faster*: an endpointer fires when the sentence is structurally done, instead
of waiting out a fixed 650ms of silence.

## The accurate tier is pluggable, and optional

The commit tier has two backends. Which one runs is a per-session choice, made
from the browser:

| backend | model | where | first-word | notes |
|---|---|---|---|---|
| local (default) | `tiny.en` / `base.en` | any CPU | ~0.6s | works offline, no account, no cost |
| remote (opt-in) | `large-v3-turbo` | Beam A10G | ~0.5s | far better on accented speech; ~99 languages |

### Knowing whether the GPU is actually running

A toggle states an *intent*; it does not tell you the tier is working. Flipping
it on therefore probes the endpoint and reports back, and each committed line
carries the tier that produced it. The indicator has four states:

| shown | meaning |
|---|---|
| *(hidden)* | tier not configured, or toggle off |
| `waiting for first line…` | requested, nothing has come back yet |
| `ready (large-v3-turbo)` | endpoint answered; ` · warmed in 11.6s` appears if it was cold |
| `unavailable — using local` | probe failed, or a final silently fell back |

The cold-start figure matters: it is the difference between "broken" and "the
first sentence will be slow". Without it a ~25s cold container and a dead
endpoint look identical from the browser.

**The service never depends on the GPU.** It is off by default, the toggle only
appears when `SD_BEAM_URL`/`SD_BEAM_TOKEN` are configured, and every remote
failure — cold start, timeout, network — falls back to the local model rather
than losing the utterance. A worse transcript beats a missing one.

That is the point rather than a caveat. The live layer is a 40MB model on a
2-core CPU; the accurate layer is whatever the deployment can afford. Same
protocol, same arbitration, same code path — only the placement changes, and
you can flip it mid-sentence and watch the same audio decoded both ways.

Why `large-v3-turbo` and not `large-v3`: turbo is a distilled decoder, ~800MB
against ~3GB and 6-8x faster, giving up quality mainly on *translation*, which
this service does not do. On serverless that compounds into fewer GPU seconds
per utterance and a much faster cold start.

Deploy. The UTF-8 environment matters — Beam's CLI uses `rich`, which crashes
on Windows cp1252.

**PowerShell** (note: `VAR=value cmd` is bash syntax and is a parse error here):

```powershell
$env:PYTHONIOENCODING='utf-8'; $env:PYTHONUTF8='1'; beam deploy whisper_beam.py:web
```

**bash / WSL:**

```bash
PYTHONIOENCODING=utf-8 PYTHONUTF8=1 beam deploy whisper_beam.py:web
```

Run it from **inside `beam_app/`**. Beam resolves the handler as a module name,
so `beam_app/whisper_beam.py:web` fails with `No module named
'beam_app/whisper_beam'`.

The folder is `beam_app`, not `beam`, on purpose: a local folder named `beam`
shadows the Beam SDK, so `from beam import Image, asgi` would import the
directory itself.

Then point the service at it:

```bash
SD_BEAM_URL=https://<your-endpoint>.app.beam.cloud SD_BEAM_TOKEN=<token> .venv/Scripts/python.exe -m uvicorn app.server:app --port 8100
```

## Arbitration — the final is a candidate, not an authority

The obvious tiered design has the offline model overwrite the streaming
hypothesis every time. That is wrong, and it fails in the most visible way
possible: on accented or non-native speech a small Whisper is often *worse*
than Vosk, so the user watches a correct word sit on screen and then get
replaced by a worse one, after a delay. The "correction" destroys information.

So every final is arbitrated against what the streaming engine already had
([`arbitrate.py`](app/arbitrate.py)). Whisper wins only when it earns it:

| condition | winner |
|---|---|
| `avg_logprob` < −0.90 (guessing) | stream |
| word agreement < 0.45 **and** `avg_logprob` < −0.55 | stream |
| Whisper kept < 60% of the words Vosk heard, and is unsure | stream |
| otherwise | whisper |

The compound conditions matter. Low agreement *alone* means the stream wins
exactly when Whisper is most useful — Vosk mangling every word looks maximally
divergent, and that is the case tiering exists for. Only *unsure disagreement*
is suspicious.

Every committed line carries `source` on the wire and a badge in the UI, and
`/health` reports the running tally, so the tiering is judged on counts rather
than impressions. Thresholds are `SD_ARB_CONFIDENT`, `SD_ARB_DESPERATE`,
`SD_ARB_DIVERGENT` — starting points, not truth. Tune them against recordings
of the actual speaker (see below); the defaults were set against clean TTS and
one accented sample, which is not enough data to be confident.

## Diagnosing bad transcription

When words come back wrong, the question is whether the microphone or the model
is at fault — and that cannot be answered by listening or guessing. Record your
own voice through the real pipeline, then run every model over the same clip:

```bash
SD_DUMP=1 .venv/Scripts/python.exe -m uvicorn app.server:app --port 8100
```

Talk at <http://localhost:8100> — every committed utterance is written to
`debug/` as a wav. Then:

```bash
.venv/Scripts/python.exe -m bench.compare debug/*.wav
```

It prints level diagnostics (RMS in dBFS, peak, clipped samples, with warnings
for a mic that is too quiet or clipping) and each model's transcript of the
same audio.

- **every model wrong** → the audio is the problem; check the levels
- **only `tiny.en` wrong** → the audio is fine, the model is too small

It reports no timings, on purpose: two attempts to include them gave impossible
orderings, because whichever model runs first pays warm-up and the laptop
throttles under load. `bench/threads.py` owns timing; this owns words.

## Prompt carry-over is off by default

Feeding the previous committed line to the next decode as `initial_prompt`
sounds obviously good — more context, better words. On unclear audio Whisper
instead **regurgitates the prompt**, which on live mic input produced the same
sentence three times in a row. Hotwords are still passed; line carry-over is
opt-in via `SD_PROMPT_CARRY=1`.

## Hallucination on silence

Left alone, this service produced an endless column of:

```
Thank you.
Thank you.
Thank you very much.
```

Whisper does this when handed near-silence. Its training data is full of video
captions, so with no speech to transcribe it emits the boilerplate those
captions end with — confidently, with a good `avg_logprob`. It is not an audio
bug; the model genuinely believes it.

Four defences, cheapest first:

1. **Absolute VAD floor** ([`vad.py`](app/audio/vad.py)). A purely *relative*
   threshold is scale-free, which sounds elegant and behaves badly: in a quiet
   room the adaptive floor sinks until a keystroke sits 4x above it. A frame now
   has to be audible in absolute terms too.
2. **Minimum utterance length.** Under 400ms is a door slam, not a sentence.
3. **Cross-engine gate** ([`session.py`](app/session.py)). Kaldi returns nothing
   on silence — it does not invent. So if Vosk heard no words across the whole
   utterance, the VAD misfired and Whisper is never asked. This one costs a
   decode rather than merely discarding one.
4. **Confidence filter** ([`worker.py`](app/asr/worker.py)). Drop output when
   `no_speech_prob` is high, or when a known filler phrase coincides with an
   unsure decode. Matched against the *whole* output only — "thank you" mid-
   sentence is a person talking; "Thank you." alone on quiet audio is not. A
   confident "Okay" still survives.

Regression tests: `tests/silence_ws.py` streams 12s of room tone with faint
transients and asserts **zero** transcript lines;
`test_vad_ignores_quiet_room_transients` covers the VAD in isolation.

### Pre-roll

Raising the VAD threshold cost the leading "The" of "The quick brown fox" —
any threshold detector opens a beat late, and a word's first consonant is
quieter than its vowel, so it may never cross at all. Finals therefore include
250ms of audio from *before* the VAD fired.

Worth noting how nearly this hid: the word-presence check still scored 9/9,
because "the" recurs later in the sentence. The test now asserts on the opening
words specifically.

## Why two Whisper sizes

**Whisper pads every input to a 30-second mel spectrogram**, so decode cost is
dominated by fixed overhead rather than by how much audio you hand it. Measured
on an i7-7500U with `tiny.en`, beam 1:

| audio | decode | vs 0.5s |
|---|---|---|
| 0.5s | 1530ms | 1.00x |
| 1.0s | 1690ms | 1.10x |
| 2.0s | 1730ms | 1.13x |
| 4.0s | 2429ms | 1.59x |

Eight times the audio costs 1.6x the time. Shrinking windows therefore buys far
less than it looks like it should, and the two things that actually matter are
**how many decodes you run** and **how big the model is**.

Which sets up a split, because the two kinds of decode have opposite
requirements:

|  | Latency-critical? | Accuracy-critical? |
|---|---|---|
| **Partial** | yes — it is the live feel | no, it gets thrown away |
| **Final** | no — the speaker already stopped | yes, it is the transcript |

So partials run on the small model with `beam_size=1`, and finals run on a
larger model with `beam_size=5`, the previous committed line as
`initial_prompt`, and any configured hotwords. Live text stays instant and the
committed line reads correctly. Spending beam search on a hypothesis that is
about to be discarded is the version of this that does not work.

### What it costs, honestly

On the i7-7500U the accurate tier does not pay for itself. `base.en` takes
~3.5-4s to decode a 3s utterance, so finals commit about four seconds after you
stop speaking — better words, worse demo. `laptop` therefore defaults to
`tiny.en` for both tiers, and `laptop-tiered` is there to switch when accuracy
matters more than latency.

The design is right; this machine cannot pay for it. That is precisely the
CPU-vs-GPU tradeoff the benchmark table is meant to show, and it is a more
interesting result than a single number would have been.

### Thread count

`bench/threads.py` measures it rather than assuming. On this box 4 threads beat
2 for `tiny.en` (1781ms vs 2775ms), repeatably — the intuition that pinning to
the 2 *physical* cores avoids contention was simply wrong here. For `base.en`
the two configurations swapped places between runs, so that comparison is noise
on a thermally throttled laptop and no winner is claimed.

Run it yourself before trusting any of these numbers on different hardware:

```bash
.venv/Scripts/python.exe -m bench.threads
```

## How it works

```
mic ──ws──► ring buffer ──► VAD ──► ASR worker thread ──► partial/final ──ws──► browser
```

**Ring buffer** ([`app/audio/ringbuffer.py`](app/audio/ringbuffer.py)) is fixed
size and drops the *oldest* audio on overflow, counting what it lost. For live
captioning a listener would rather lose a stale second than have the transcript
fall permanently behind. An unbounded queue here is the classic way this class
of service dies.

**VAD** ([`app/audio/vad.py`](app/audio/vad.py)) is why CPU-only is viable at
all: silence is never decoded. Phase 1 uses an energy VAD with an adaptive
noise floor and hysteresis — zero dependencies, no model download, and
genuinely worse than silero in a noisy room. The interface is one `push()`
returning events, so a silero ONNX backend drops in later as a fair A/B.

**ASR worker** ([`app/asr/worker.py`](app/asr/worker.py)) is a single thread
shared by all sessions. On two cores, parallel decoders would just fight over
the same cores and make every stream worse; serializing keeps latency
predictable. faster-whisper releases the GIL inside CTranslate2, so decoding
never stalls the event loop. Superseded partials are dropped rather than
decoded late; finals are never dropped.

**At most one partial is in flight per session**, and this is the load-bearing
part. A fixed emit interval is not enough on its own: measured on a 2-core
i7-7500U, `tiny.en` takes ~1000ms to decode a 3s window against a 900ms
interval, so work would be created faster than it drains and the decode queue
would grow without bound — the transcript falls permanently behind the speaker
while the process slowly eats memory. Gating on in-flight makes the system
*skip* partials under load instead of *queueing* them, which is the correct
degradation: the listener sees fewer intermediate updates, never a stale
transcript. `stats.queue_ms` is the number to watch; if it climbs, this gate
has broken.

**Partial vs final.** Partials are emitted every ~0.6s while you are still
talking and are speculative — the client renders them greyed out and replaces
them when the committed final arrives at end of utterance. Emitting fast and
revising is what makes it feel live.

## Protocol

Client sends binary Int16 LE mono PCM @ 16kHz. Server sends JSON:

```json
{"type":"partial","seq":7,"text":"so what i was say","t0":12.4,"t1":13.9,"speaker":0}
{"type":"final","seq":8,"text":"So what I was saying.","t0":12.4,"t1":14.1,"speaker":0}
{"type":"stats","rtf":0.42,"decode_ms":310,"queue_ms":12,"first_partial_ms":540}
```

`rtf` is real-time factor — decode time over audio duration. **Under 1.0 is the
survival line.** The demo page shows it live and turns it red when it is not.

## Not in scope

No training, no multi-language, no speaker *identification* (Phase 2 separates
speakers, it does not name them), no summarization, no auth, no audio storage.
