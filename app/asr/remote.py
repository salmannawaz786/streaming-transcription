"""Remote accurate tier (Beam A10G, whisper large-v3).

Returns the same shape as a local decode, so arbitration and the wire protocol
do not care which backend produced a line.

Two properties matter more than speed here:

**It must never take the service down.** A GPU endpoint can cold-start, rate
limit, or simply be unreachable. Every failure falls back to the local model
rather than losing the utterance — a worse transcript beats a missing one.

**It is opt-in per session.** GPU seconds cost money, so the browser toggles it
per connection instead of every stream burning credits by default.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import time
import httpx
import numpy as np

log = logging.getLogger(__name__)

# Auto-load .env.beam if present in repo root
ENV_BEAM_PATH = Path(__file__).resolve().parents[2] / ".env.beam"
if ENV_BEAM_PATH.is_file():
    for line in ENV_BEAM_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            if k not in os.environ:
                os.environ[k] = v.strip().strip("'\"")

URL = os.getenv("SD_BEAM_URL", "").rstrip("/")
TOKEN = os.getenv("SD_BEAM_TOKEN", "")
# Generous: a cold container loads ~800MB of weights before it can answer.
TIMEOUT_S = float(os.getenv("SD_BEAM_TIMEOUT", "60"))

# Persistent HTTP client with connection pooling.
# Keeps TCP + TLS sockets open across utterances, eliminating 500-1000ms of
# connection setup overhead on every utterance from home internet to US cloud.
_client: httpx.Client | None = None


def _get_client() -> httpx.Client:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.Client(
            timeout=httpx.Timeout(TIMEOUT_S, connect=10.0),
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=20, keepalive_expiry=120.0),
            headers={"Authorization": f"Bearer {TOKEN}"},
            http2=False,
        )
    return _client


def configured() -> bool:
    return bool(URL and TOKEN)


class RemoteError(RuntimeError):
    pass


def health(timeout_s: float = 12.0) -> dict:
    """Probe the endpoint so the UI can report GPU state before anyone speaks."""
    if not configured():
        return {"ok": False, "error": "not configured"}
    started = time.perf_counter()
    try:
        client = _get_client()
        resp = client.get(f"{URL}/health", timeout=timeout_s)
        resp.raise_for_status()
        payload = resp.json()
        payload["latency_ms"] = round((time.perf_counter() - started) * 1000)
        return payload
    except Exception as exc:  # noqa: BLE001 - any failure is "tier unavailable"
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:160]}


def transcribe(audio: np.ndarray, language: str | None = "en", prompt: str | None = None) -> dict:
    """Send Int16 PCM to the GPU endpoint. Raises RemoteError on any failure."""
    if not configured():
        raise RemoteError("SD_BEAM_URL / SD_BEAM_TOKEN not set")

    body = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
    params: dict[str, str] = {}
    if language:
        params["lang"] = language
    if prompt:
        params["prompt"] = prompt

    try:
        client = _get_client()
        resp = client.post(
            f"{URL}/transcribe",
            params=params,
            content=body,
            headers={"Content-Type": "application/octet-stream"},
        )
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:
        raise RemoteError(str(exc)) from exc

    if "text" not in payload:
        raise RemoteError(f"unexpected response: {payload!r}")
    return payload
