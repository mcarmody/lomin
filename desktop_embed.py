#!/usr/bin/env python3
"""Desktop-side speaker-embedding server for the LÓMIN voice gateway.

Runs on a desktop host with a GPU, bound to 127.0.0.1:9096 by default —
mirrors desktop_tts.py's pattern exactly. A host that cannot reach this
desktop directly (NAT-only, e.g. WSL2) reaches it through a second `-R`
forward on the same ssh reverse tunnel that carries the TTS hop
(`-R 127.0.0.1:9096:127.0.0.1:9096 <gateway-host>`). See README.md,
"Speaker ID" section.

Why this exists: the speaker-ID contract (a plain HTTP POST of raw WAV bytes
returning a fixed-dimension embedding) is served by any model that produces
stable, comparable vectors. speechbrain's ECAPA-TDNN voice-embedding model
needs a GPU to embed inside the hub's 5s timeout and the ~500ms it waits
after STT before giving up on a speaker label; the gateway host has no GPU,
and the desktop's idle RTX 4090 already carries the same job for TTS
(Chatterbox, desktop_tts.py) over the same tunnel pattern.

Endpoints:
    POST /embed   raw WAV body (any sample rate/channel count — downmixed
                  and resampled to 16kHz mono defensively) ->
                  {"embedding": [192 floats], "model": "speechbrain/spkrec-ecapa-voxceleb"}
                  400 on an empty, undecodable, or too-short (<0.5s) body.
    GET  /health  -> {"status": "ok"|"error", "model": ..., "device": ...,
                       "loaded": bool, "load_error": str|None}

The model (`speechbrain/spkrec-ecapa-voxceleb`) is loaded ONCE at process
start, synchronously, before the server starts accepting connections —
unlike desktop_tts.py's Chatterbox hop, there is no fallback engine to serve
requests with while this loads, so there is nothing gained by loading it in
the background. A single warmup encode runs right after load so the first
real request doesn't pay for CUDA kernel compilation / lazy allocator
warmup, which would otherwise risk missing the hub's 500ms post-STT budget.
With MDE_IDLE_UNLOAD_MIN > 0 the model is dropped after that many idle minutes
and reloaded by the next /embed.

Honors HF_HUB_OFFLINE=1 once the model is cached locally (same as
desktop_tts.py's Chatterbox hop) — the first run on a fresh cache needs it
unset so speechbrain can download the model from Hugging Face.

Inference is synchronized with a lock (GPU inference is not documented as
thread-safe for concurrent calls) and the encode call itself runs in
FastAPI/Starlette's threadpool (`starlette.concurrency.run_in_threadpool`,
called from the `async def` endpoint after `await request.body()`) so the
asyncio event loop isn't blocked while a call is in flight.

Runs under the same shared venv as desktop_tts.py (Python 3.11) — needs
speechbrain in addition to what's already there (torch/torchaudio, CUDA
build). Never the system Python.
"""

from __future__ import annotations

import gc
import io
import os
import sys
import threading
import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

PORT = int(os.environ.get("MDE_PORT", "9096"))
BIND = os.environ.get("MDE_BIND", "127.0.0.1")

MODEL_SOURCE = os.environ.get("MDE_MODEL_SOURCE", "speechbrain/spkrec-ecapa-voxceleb")
MODEL_ID = MODEL_SOURCE  # the model string the hub's speakers.json must match exactly
DEVICE = os.environ.get("MDE_DEVICE", "cuda:0")
SAVEDIR = Path(
    os.environ.get("MDE_SAVEDIR")
    or (Path(os.environ.get("MVG_HOME") or (Path.home() / ".mvg")) / "models" / "speechbrain-ecapa")
)
TARGET_SAMPLE_RATE = 16000
MIN_DURATION_SEC = 0.5
WARMUP_SEC = 1.0  # 1s of near-silence, just enough to exercise the full path
IDLE_UNLOAD_SEC = float(os.environ.get("MDE_IDLE_UNLOAD_MIN", "0") or 0) * 60.0

_model = None
# RLock: the idle reload runs under the lock, and _load_model() takes it again
# for the warmup encode.
_model_lock = threading.RLock()
_load_error: str | None = None
_load_ms: float | None = None
_warmup_ms: float | None = None
_active_device: str | None = None  # the device _load_model() actually used
_last_used: float = 0.0  # time.monotonic() of the last finished encode (or load)
_idle_unloaded: bool = False  # True once the idle watcher dropped the model


def _load_model() -> None:
    """Loads the ECAPA encoder and runs a warmup encode. Sets _load_error
    instead of raising, so a bad venv or missing GPU produces a 503 from
    /health rather than crashing the whole process on import. Also the reload
    path after an idle unload."""
    global _model, _load_error, _load_ms, _warmup_ms, _active_device, _last_used, _idle_unloaded
    _load_error = None
    try:
        import torch  # noqa: E402 (deliberately lazy: keeps this module
        from speechbrain.inference.speaker import EncoderClassifier  # noqa: E402
    except Exception as exc:  # pragma: no cover - only hit with a bad venv
        _load_error = f"speechbrain/torch not importable: {exc}"
        print(f"[desktop-embed] {_load_error}", file=sys.stderr, flush=True)
        return

    device = DEVICE if (not DEVICE.startswith("cuda") or torch.cuda.is_available()) else "cpu"
    if device != DEVICE:
        print("[desktop-embed] CUDA not available, falling back to cpu", flush=True)
    _active_device = device

    try:
        t0 = time.time()
        SAVEDIR.mkdir(parents=True, exist_ok=True)
        model = EncoderClassifier.from_hparams(
            source=MODEL_SOURCE,
            savedir=str(SAVEDIR),
            run_opts={"device": device},
        )
        _load_ms = (time.time() - t0) * 1000
        print(f"[desktop-embed] loaded {MODEL_SOURCE} on {device} in {_load_ms:.0f}ms", flush=True)

        t0 = time.time()
        warmup_samples = torch.zeros((1, int(TARGET_SAMPLE_RATE * WARMUP_SEC)))
        with _model_lock:
            model.encode_batch(warmup_samples)
        _warmup_ms = (time.time() - t0) * 1000
        print(f"[desktop-embed] warmup encode in {_warmup_ms:.0f}ms", flush=True)
        _model = model
        _last_used = time.monotonic()
        _idle_unloaded = False
    except Exception as exc:
        _load_error = f"model load/warmup failed: {exc}"
        print(f"[desktop-embed] {_load_error}", file=sys.stderr, flush=True)


def _unload_if_idle(now: float, idle_sec: float) -> bool:
    """Drops the model once no encode has finished for idle_sec seconds.
    Returns True when it unloaded. idle_sec <= 0 means disabled."""
    global _model, _idle_unloaded
    if idle_sec <= 0:
        return False
    with _model_lock:
        if _model is None:
            return False
        if now - _last_used < idle_sec:
            return False
        _model = None
        _idle_unloaded = True
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        print("[desktop-embed] unloaded after idle", flush=True)
        return True


def _idle_watch() -> None:
    while True:
        time.sleep(15)
        _unload_if_idle(time.monotonic(), IDLE_UNLOAD_SEC)


def _decode_and_resample(body: bytes):
    """Returns a 1-D float32 numpy array at TARGET_SAMPLE_RATE, mono.
    Raises ValueError on anything not decodable as audio."""
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio

    try:
        data, sample_rate = sf.read(io.BytesIO(body), dtype="float32", always_2d=True)
    except Exception as exc:
        raise ValueError(f"could not decode audio: {exc}") from exc

    if data.size == 0:
        raise ValueError("empty audio")

    mono = data.mean(axis=1)  # downmix any channel count to 1

    duration = len(mono) / float(sample_rate)
    if duration < MIN_DURATION_SEC:
        raise ValueError(f"audio too short: {duration:.2f}s < {MIN_DURATION_SEC}s")

    if sample_rate != TARGET_SAMPLE_RATE:
        tensor = torch.from_numpy(mono).unsqueeze(0)  # (1, N)
        tensor = torchaudio.functional.resample(tensor, sample_rate, TARGET_SAMPLE_RATE)
        mono = tensor.squeeze(0).numpy()

    return mono.astype(np.float32)


def _embed_sync(body: bytes) -> dict:
    """Runs entirely off the event loop (called via run_in_threadpool)."""
    global _last_used
    import torch

    # Decode before the lock and before any reload: a bad body is a 400 and
    # must not pay for a model load.
    samples = _decode_and_resample(body)
    tensor = torch.from_numpy(samples).unsqueeze(0)  # (1, N)

    with _model_lock:
        if _model is None and _idle_unloaded:
            _load_model()
        model = _model
        if model is None:
            raise RuntimeError(_load_error or "model not loaded")
        t0 = time.time()
        emb = model.encode_batch(tensor)
        _last_used = time.monotonic()
    ms = (time.time() - t0) * 1000

    vector = emb.squeeze(0).squeeze(0).detach().cpu().tolist()
    print(f"[desktop-embed] /embed ok in {ms:.0f}ms, {len(vector)} dims", flush=True)
    return {"embedding": vector, "model": MODEL_ID}


app = FastAPI()


@app.get("/health")
async def health():
    ok = _model is not None
    idle = _model is None and _idle_unloaded and _load_error is None
    if idle:
        return JSONResponse(
            {
                "status": "idle",
                "model": MODEL_ID,
                "device": _active_device or DEVICE,
                "loaded": False,
                "load_error": None,
                "load_ms": _load_ms,
                "warmup_ms": _warmup_ms,
            },
            status_code=200,
        )
    return JSONResponse(
        {
            "status": "ok" if ok else "error",
            "model": MODEL_ID,
            "device": _active_device or DEVICE,
            "loaded": ok,
            "load_error": _load_error,
            "load_ms": _load_ms,
            "warmup_ms": _warmup_ms,
        },
        status_code=200 if ok else 503,
    )


@app.post("/embed")
async def embed(request: Request):
    body = await request.body()
    if not body:
        return JSONResponse({"error": "empty body"}, status_code=400)
    if _model is None and not _idle_unloaded:
        return JSONResponse({"error": _load_error or "model not loaded"}, status_code=503)
    try:
        result = await run_in_threadpool(_embed_sync, body)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except Exception as exc:
        return JSONResponse({"error": f"embed failed: {exc}"}, status_code=500)
    return JSONResponse(result, status_code=200)


def main():
    import uvicorn

    _load_model()
    if IDLE_UNLOAD_SEC > 0:
        threading.Thread(target=_idle_watch, daemon=True).start()
    print(f"[desktop-embed] listening on {BIND}:{PORT}", flush=True)
    uvicorn.run(app, host=BIND, port=PORT, workers=1, log_level="warning")


if __name__ == "__main__":
    main()
