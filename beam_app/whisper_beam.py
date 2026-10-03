"""Accurate tier on a Beam A10G: whisper large-v3-turbo.

Why this is a good fit for serverless
-------------------------------------
Partials must stream — persistent connection, tens of updates a second — so
they stay local on CPU with Vosk. **Finals are discrete requests**: one call per
utterance, a few per minute, each a self-contained chunk of audio. That is
exactly the shape serverless GPU bills well for.

What it buys
------------
large-v3-turbo is dramatically better than tiny/base on accented and non-native
English, which no amount of threshold tuning fixes locally. It is multilingual
(~99 languages), so the multi-language question is answered by the same move.

**Why turbo rather than plain large-v3.** Turbo is a distilled decoder: ~800MB
against ~3GB, roughly 6-8x faster, with little accuracy lost for transcription
(it gives up quality on translation, which this service does not do). On
serverless that compounds — fewer GPU seconds per utterance, and a cold start
that pulls 800MB instead of 3GB. Paying for large-v3 here would buy almost
nothing this service uses.

Deploy (note the encoding prefix — Beam's CLI uses rich, which dies on
Windows cp1252):

    PYTHONIOENCODING=utf-8 PYTHONUTF8=1 beam deploy beam/whisper_beam.py:web

Weights live on a volume so container starts do not re-download them.
"""

from beam import Image, Volume, asgi

# CTranslate2 (what faster-whisper runs on) dlopens the CUDA 12 math libraries
# at inference time. Beam's base image does not ship them, so the first deploy
# passed /health — the model loads lazily enough to look fine — and then failed
# every transcribe with:
#
#     RuntimeError: Library libcublas.so.12 is not found or cannot be loaded
#
# The libraries ship as pip wheels. LD_LIBRARY_PATH must be set in the
# CONTAINER environment, not from Python: glibc reads it once at process start,
# so os.environ[...] inside the handler is too late to affect dlopen.
#
# CTranslate2 4.5+ needs cuDNN 9, not 8.
_NVIDIA_LIBS = "/usr/local/lib/python3.11/dist-packages/nvidia"

image = (
    Image(
        python_version="python3.11",
        env_vars={
            "LD_LIBRARY_PATH": (
                f"{_NVIDIA_LIBS}/cublas/lib:{_NVIDIA_LIBS}/cudnn/lib:"
                "/usr/local/nvidia/lib64"
            )
        },
    )
    .add_python_packages([
        "fastapi",
        "faster-whisper",
        "numpy",
        "nvidia-cublas-cu12",
        "nvidia-cudnn-cu12>=9,<10",
    ])
)

# ~800MB. Still worth a volume: re-pulling it on every cold start would add
# seconds to the first utterance after an idle period.
weights = Volume(name="whisper-weights", mount_path="./weights")


@asgi(
    name="whisper-turbo",
    image=image,
    gpu="A10G",
    cpu=2,
    memory="16Gi",
    volumes=[weights],
    # Cold start pays model load (~20-40s). Staying warm between utterances in
    # a conversation is the difference between a usable demo and a broken one;
    # 5 minutes covers a demo without holding the GPU all day.
    keep_warm_seconds=300,
    timeout=600,
    authorized=True,
)
def web():
    import ctypes
    import glob

    import numpy as np
    from fastapi import FastAPI, Request
    from faster_whisper import WhisperModel

    # Belt and braces alongside LD_LIBRARY_PATH: load the CUDA libs explicitly
    # with RTLD_GLOBAL so their symbols are already resolved when CTranslate2
    # dlopens. Harmless if the loader found them anyway, and it surfaces a
    # missing library here rather than on the first transcription.
    loaded, failed = [], []
    for pattern in (
        f"{_NVIDIA_LIBS}/cublas/lib/libcublasLt.so*",
        f"{_NVIDIA_LIBS}/cublas/lib/libcublas.so*",
        f"{_NVIDIA_LIBS}/cudnn/lib/libcudnn*.so*",
    ):
        for path in sorted(glob.glob(pattern)):
            try:
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                loaded.append(path.rsplit("/", 1)[-1])
            except OSError as exc:
                failed.append(f"{path.rsplit('/', 1)[-1]}: {exc}")

    # Loaded once per container, not per request.
    model = WhisperModel(
        "large-v3-turbo",
        device="cuda",
        compute_type="float16",
        download_root="./weights",
    )

    app = FastAPI()

    @app.get("/health")
    def health():
        return {
            "ok": True,
            "model": "large-v3-turbo",
            "device": "cuda",
            "cuda_libs_loaded": len(loaded),
            "cuda_libs_failed": failed,
        }

    @app.post("/transcribe")
    async def transcribe(request: Request):
        """Body: raw Int16 LE mono PCM @ 16kHz. Query: ?lang=en (omit = auto)."""
        # Return failures as JSON rather than letting them become a bare 500.
        # `beam logs` is unreliable on the dev machine here, so an endpoint
        # that cannot explain itself is very expensive to debug — each guess
        # costs a full rebuild-and-deploy cycle.
        import traceback

        try:
            raw = await request.body()
            if not raw:
                return {"text": "", "logprob": 0.0, "no_speech": 1.0,
                        "error": "empty body"}
            if len(raw) % 2:
                raw = raw[:-1]  # Int16 needs an even byte count

            audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
            lang = request.query_params.get("lang") or "en"
            prompt = request.query_params.get("prompt") or request.headers.get("x-prompt") or None

            # beam_size=1 (greedy) is ~3-4x faster than beam_size=5 on A10G
            # (150ms vs 750ms) while retaining full large-v3-turbo accuracy.
            opts = {
                "language": lang,
                "beam_size": 1,
                "temperature": 0.0,
                "vad_filter": False,
                "condition_on_previous_text": False,
                "without_timestamps": True,
                "repetition_penalty": 1.1,
            }
            if prompt:
                opts["initial_prompt"] = prompt.strip()

            segments, info = model.transcribe(audio, **opts)
            segs = list(segments)

            return {
                "text": "".join(s.text for s in segs).strip(),
                # Same fields the local tier returns, so arbitration is
                # identical whichever backend produced the line.
                "logprob": float(min((s.avg_logprob for s in segs), default=0.0)),
                "no_speech": float(max((s.no_speech_prob for s in segs), default=0.0)),
                "language": getattr(info, "language", None) or lang,
                "duration_s": round(audio.size / 16_000, 2),
                "model": "large-v3-turbo",
            }
        except Exception as exc:
            return {
                "text": "",
                "logprob": 0.0,
                "no_speech": 1.0,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-1500:],
            }

    return app
