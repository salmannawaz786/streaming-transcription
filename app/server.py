"""FastAPI WebSocket server.

Wire protocol
-------------
Client -> server:
  binary   Int16 little-endian mono PCM @ 16kHz
  text     {"type": "reset"}

Server -> client:
  {"type":"ready","profile":...,"model":...}
  {"type":"speech_start","t":12.4}
  {"type":"partial","seq":7,"text":"so what i was say","t0":12.4,"t1":13.9,"speaker":0}
  {"type":"final","seq":8,"text":"So what I was saying.","t0":12.4,"t1":14.1,"speaker":0}
  {"type":"speech_end","t":14.1}
  {"type":"dropped","seconds":0.4}
  {"type":"stats","rtf":0.42,"decode_ms":310,"queue_ms":12,"first_partial_ms":540}

`partial` is speculative and will be revised. `final` is committed. A client
should render the latest partial for a seq and replace it when the final for
that utterance arrives.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import uuid
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from dataclasses import replace

from . import arbitrate
from .asr.streaming import StreamingAsr
from .asr import remote
from .asr.worker import AsrWorker, Result
from .config import active_profile
from .session import Session

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s"
)
log = logging.getLogger("server")

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

profile = active_profile()

sessions: dict[str, tuple[Session, WebSocket]] = {}
worker: AsrWorker | None = None


@contextlib.asynccontextmanager
async def lifespan(_: FastAPI):
    global worker
    # Vosk owns the live layer when present; without it Whisper does chunky
    # partials instead. Loading is cheap (~40MB) so it happens inline.
    StreamingAsr.load()
    worker = AsrWorker(profile, asyncio.get_running_loop())
    worker.start()
    pump = asyncio.create_task(_drain_results())
    log.info("profile=%s fast=%s accurate=%s device=%s", profile.name,
             profile.fast_model, profile.accurate_model, profile.device)
    try:
        yield
    finally:
        pump.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await pump
        worker.stop()


app = FastAPI(title="Streaming Transcription", lifespan=lifespan)


async def _drain_results() -> None:
    """Route decoded results from the worker thread back to their sockets.

    Every result is handled inside the try. A bare asyncio task that raises
    dies silently and takes all transcript output with it — the service then
    looks alive (audio still ingests, VAD events still flow) while producing
    nothing. Never let one bad result kill the pump.
    """
    assert worker is not None
    while True:
        result = await worker.results.get()
        try:
            await _handle_result(result)
        except Exception:
            log.exception("failed handling result seq=%s", result.seq)


async def _handle_result(result: Result) -> None:
    entry = sessions.get(result.session_id)
    if entry is None:
        return
    session, ws = entry

    if result.kind == "partial":
        session.partial_inflight = False
        # A partial that lands after its utterance already ended would
        # redraw a speculative line on top of the committed final.
        if session.utterance is None or session.utterance.seq != result.seq:
            return
    if result.dropped:
        return

    # Arbitrate: the final is a candidate, not an authority. Overwriting a
    # correct streaming hypothesis with an unsure Whisper guess is worse than
    # not correcting at all — the user watches a right answer become a wrong
    # one, after a delay.
    source, reason, sim = "whisper", "", 0.0
    if result.kind == "final":
        utt = session.pending.get(result.seq)
        stream_text = utt.stream_text if utt else ""
        pick = arbitrate.choose(
            stream_text, result.text, result.logprob, result.t1 - result.t0, tier=result.tier
        )
        result = replace(result, text=pick.text)
        source, reason, sim = pick.source, pick.reason, pick.similarity
        session.arbitration[source] += 1
        log.debug("seq=%s -> %s (%s, sim=%.2f)", result.seq, source, reason, sim)

    if result.kind == "final" and result.text:
        # Keep only the tail: Whisper's prompt window is limited and a long
        # history crowds out the audio.
        session.last_final = " ".join(
            (session.last_final + " " + result.text).split()[-40:]
        )

    if result.text:
        await _send(
            ws,
            {
                "type": result.kind,
                "seq": result.seq,
                "text": result.text,
                "t0": round(result.t0, 2),
                "t1": round(result.t1, 2),
                "speaker": 0,  # Phase 2 fills this in
                "source": source,
                "agreement": round(sim, 2),
            },
        )

    stats = {
        "type": "stats",
        "kind": result.kind,
        "model": result.model,
        "audio_s": round(result.t1 - result.t0, 2),
        "rtf": round(result.rtf, 3),
        "decode_ms": round(result.decode_ms),
        "queue_ms": round(result.queue_ms),
    }
    utt = session.pending.pop(result.seq, None)
    if utt and result.kind == "partial" and not utt.first_partial_sent:
        utt.first_partial_sent = True
        import time

        stats["first_partial_ms"] = round(
            (time.perf_counter() - utt.detected_at) * 1000
        )
    await _send(ws, stats)


async def _control(text: str, sid: str, session: Session, ws: WebSocket) -> None:
    """Client control messages: {"type":"reset"} / {"type":"config","gpu":true}."""
    try:
        msg = json.loads(text)
    except ValueError:
        return

    if msg.get("type") == "reset":
        fresh = Session(sid, profile, worker)
        fresh.use_remote = session.use_remote
        session.close()
        sessions[sid] = (fresh, ws)
        return

    if msg.get("type") == "config" and "gpu" in msg:
        wanted = bool(msg["gpu"])
        if wanted and not remote.configured():
            await _send(ws, {
                "type": "config",
                "gpu": False,
                "error": "GPU tier not configured - set SD_BEAM_URL and SD_BEAM_TOKEN",
            })
            return
        session.use_remote = wanted
        log.info("session %s gpu tier %s", sid, "on" if wanted else "off")
        await _send(ws, {"type": "config", "gpu": wanted})

        if wanted:
            # Probe in an executor: a cold container can take ~25s to answer
            # and the event loop still has audio to ingest meanwhile.
            probe = await asyncio.get_running_loop().run_in_executor(
                None, remote.health
            )
            await _send(ws, {
                "type": "gpu_status",
                "ok": bool(probe.get("ok")),
                "model": probe.get("model", ""),
                "latency_ms": probe.get("latency_ms"),
                "error": probe.get("error", ""),
            })


async def _send(ws: WebSocket, payload: dict) -> None:
    with contextlib.suppress(RuntimeError, WebSocketDisconnect):
        await ws.send_json(payload)


@app.get("/health")
async def health() -> dict:
    return {
        "ok": worker is not None and worker.ready.is_set(),
        "profile": profile.name,
        "fast_model": profile.fast_model,
        "accurate_model": profile.accurate_model,
        "device": profile.device,
        "sessions": len(sessions),
        "streaming_engine": "vosk" if StreamingAsr.available() else None,
        "gpu_tier_available": remote.configured(),
        "arbitration": {
            k: sum(s.arbitration[k] for s, _ in sessions.values())
            for k in ("whisper", "vosk")
        },
    }


@app.websocket("/ws")
async def stream(ws: WebSocket) -> None:
    assert worker is not None
    await ws.accept()

    # The first model load takes a while; tell the client rather than sitting
    # silent while it happens.
    if not worker.ready.is_set():
        await _send(ws, {"type": "loading", "model": profile.accurate_model})
        await asyncio.get_running_loop().run_in_executor(None, worker.ready.wait)

    sid = uuid.uuid4().hex[:8]
    session = Session(sid, profile, worker)
    sessions[sid] = (session, ws)
    await _send(
        ws,
        {
            "type": "ready",
            "session": sid,
            "profile": profile.name,
            "model": f"{profile.fast_model} / {profile.accurate_model}",
            "device": profile.device,
            "gpu_available": remote.configured(),
        },
    )
    log.info("session %s open (%d active)", sid, len(sessions))

    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break

            if (data := msg.get("bytes")) is not None:
                pcm = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
                for event in session.feed(pcm, data):
                    await _send(ws, event)
            elif (text := msg.get("text")) is not None:
                await _control(text, sid, session, ws)
    except WebSocketDisconnect:
        pass
    finally:
        sessions.pop(sid, None)
        session.close()
        log.info("session %s closed (%d active)", sid, len(sessions))


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")
