"""Demucs music source separation service.

Separates a music mix into stems (vocals, drums, bass, other) with Meta's
Demucs (htdemucs by default). Built for the "clean vocals before Whisper"
use case: upload a mix, get back one stem as WAV.

Endpoints mirror the house style of the whisper service: one POST with a
multipart file upload, plus /health.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import numpy as np
import soundfile
import torch
from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse, Response

# NOTE: no torchaudio here on purpose. demucs >= 4.1.0 dropped it as a runtime
# dependency (train extra only), so it isn't in the image. Decode goes through
# soundfile (bundled libsndfile: wav/flac/ogg/mp3), resampling through julius
# (a demucs dependency and its own resampler), WAV encode through the stdlib.

DEMUCS_MODEL = os.getenv("DEMUCS_MODEL", "htdemucs")
DEMUCS_DEVICE = os.getenv("DEMUCS_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
# Optional: shorter per-inference segments bound VRAM (the GPU is shared with
# whisper/ollama/SDXL). Unset = the model's own default (7.8s for htdemucs,
# which is also its maximum).
_seg = os.getenv("DEMUCS_SEGMENT", "").strip()
DEMUCS_SEGMENT: float | None = float(_seg) if _seg else None
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
_ENV_KEYS = ("DEMUCS_MODEL", "DEMUCS_DEVICE", "DEMUCS_SEGMENT", "TORCH_HOME", "LOG_LEVEL")
_SENSITIVE_ENV_MARKERS = ("PASSWORD", "TOKEN", "KEY", "SECRET")

logger = logging.getLogger("demucs")
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)


def _mask_env_value(key: str, value: str | None) -> str:
    if not value:
        return "<unset>"
    if any(marker in key.upper() for marker in _SENSITIVE_ENV_MARKERS):
        return "<set>"
    if len(value) > 120:
        return f"{value[:117]}..."
    return value


def _log_startup_env(service_name: str, keys: tuple[str, ...]) -> None:
    rendered = ", ".join(f"{key}={_mask_env_value(key, os.getenv(key))}" for key in keys)
    logger.info("%s startup env: %s", service_name, rendered)


def _load_separator():
    # Imported here so a broken install fails at startup with a clear log line.
    from demucs.api import Separator

    logger.info("Loading Demucs model: %s (device=%s)", DEMUCS_MODEL, DEMUCS_DEVICE)
    kwargs: dict = {"model": DEMUCS_MODEL, "device": DEMUCS_DEVICE, "overlap": 0.25}
    if DEMUCS_SEGMENT is not None:
        kwargs["segment"] = DEMUCS_SEGMENT
    separator = Separator(**kwargs)
    logger.info(
        "Demucs model loaded (samplerate=%d, sources=%s)",
        separator.samplerate,
        separator.model.sources,
    )
    return separator


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    _log_startup_env("demucs", _ENV_KEYS)
    app.state.separator = _load_separator()
    # The GPU is shared with the rest of the stack; run one separation at a
    # time so concurrent requests queue instead of stacking VRAM.
    app.state.gpu_lock = asyncio.Lock()
    yield


app = FastAPI(title="Demucs Source Separation", version="0.1.0", lifespan=lifespan)


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": "invalid_request_error"}},
    )


def _encode_wav(tensor: "torch.Tensor", sr: int) -> bytes:
    """(channels, time) float tensor -> 16-bit PCM WAV bytes, stdlib only."""
    import wave

    samples = tensor.numpy()
    pcm = np.clip(samples * 32768.0, -32768, 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(pcm.shape[0])
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(pcm.T.tobytes())  # interleave channels
    return buf.getvalue()


def _pick_stem(separated: dict, stem: str) -> "torch.Tensor":
    """One named stem, or the sum of everything-but-vocals for accompaniment."""
    if stem in ("no_vocals", "accompaniment", "instrumental"):
        others = [v for k, v in separated.items() if k != "vocals"]
        return sum(others[1:], start=others[0])
    return separated[stem]


@app.post("/v1/audio/separations")
async def create_separation(
    file: UploadFile,
    stem: str = "vocals",
    sample_rate: int | None = None,
    mono: bool = False,
    model: str | None = None,  # accepted for API symmetry; server model is used
) -> Response:
    separator = app.state.separator

    audio_bytes = await file.read()
    if not audio_bytes:
        return _error(400, "uploaded file is empty")

    valid_stems = list(separator.model.sources) + ["no_vocals", "accompaniment", "instrumental"]
    if stem not in valid_stems:
        return _error(400, f"unknown stem {stem!r}; expected one of {valid_stems}")
    if sample_rate is not None and not 4000 <= sample_rate <= 192000:
        return _error(400, "sample_rate must be between 4000 and 192000")

    try:
        data, sr = soundfile.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
    except Exception as exc:  # noqa: BLE001 - any decode failure is a client error
        return _error(400, f"could not decode audio: {exc}")
    wav = torch.from_numpy(data.T.copy())  # (channels, time)

    def _run() -> bytes:
        import julius

        # separate_tensor resamples/upmixes the input to the model's format.
        _origin, separated = separator.separate_tensor(wav, sr)
        out = _pick_stem(separated, stem)
        out_sr = separator.samplerate
        if mono:
            out = out.mean(dim=0, keepdim=True)
        if sample_rate is not None and sample_rate != out_sr:
            out = julius.resample_frac(out, out_sr, sample_rate)
            out_sr = sample_rate
        return _encode_wav(out.cpu(), out_sr)

    async with app.state.gpu_lock:
        result = await asyncio.to_thread(_run)

    logger.info(
        "separated %.1fs of audio -> stem=%s mono=%s sample_rate=%s (%d bytes out)",
        wav.shape[-1] / sr,
        stem,
        mono,
        sample_rate or separator.samplerate,
        len(result),
    )
    return Response(content=result, media_type="audio/wav")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
