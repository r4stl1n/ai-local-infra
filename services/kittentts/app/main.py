from __future__ import annotations

import asyncio
import importlib
import logging
import os
import struct
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import lameenc
import numpy as np
from fastapi import FastAPI
from fastapi.responses import Response, StreamingResponse
from huggingface_hub.errors import LocalEntryNotFoundError
from kittentts import KittenTTS
from pydantic import BaseModel, Field

# kittentts/__init__.py re-exports a function named get_model that shadows the
# submodule attribute, so resolve the module through importlib.
_kittentts_get_model = importlib.import_module("kittentts.get_model")

# KittenTTS doesn't expose local_files_only, so wrap its downloader to serve
# from the HF cache first — startup must work without network once cached.
_hf_hub_download = _kittentts_get_model.hf_hub_download


def _cache_first_hf_hub_download(*args, **kwargs):
    try:
        return _hf_hub_download(*args, local_files_only=True, **kwargs)
    except LocalEntryNotFoundError:
        return _hf_hub_download(*args, **kwargs)


_kittentts_get_model.hf_hub_download = _cache_first_hf_hub_download

TTS_MODEL = os.getenv("TTS_MODEL", "KittenML/kitten-tts-nano-0.8")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
_ENV_KEYS = ("TTS_MODEL", "HF_HOME", "LOG_LEVEL")
_SENSITIVE_ENV_MARKERS = ("PASSWORD", "TOKEN", "KEY", "SECRET")

SAMPLE_RATE = 24000
NUM_CHANNELS = 1
BITS_PER_SAMPLE = 16
BYTES_PER_SAMPLE = BITS_PER_SAMPLE // 8

logger = logging.getLogger("kittentts")
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


def _load_model(model_name: str) -> KittenTTS:
    logger.info("Loading TTS model: %s", model_name)
    model = KittenTTS(model_name)
    logger.info("TTS model loaded successfully")
    return model


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    _log_startup_env("kittentts", _ENV_KEYS)
    app.state.tts_model = _load_model(TTS_MODEL)
    yield


app = FastAPI(title="KittenTTS", version="0.1.0", lifespan=lifespan)


# -- audio helpers -----------------------------------------------------------

def _build_wav_header(data_size: int) -> bytes:
    """Build a 44-byte RIFF/WAVE header for 16-bit mono PCM at 24 kHz."""
    byte_rate = SAMPLE_RATE * NUM_CHANNELS * BYTES_PER_SAMPLE
    block_align = NUM_CHANNELS * BYTES_PER_SAMPLE
    riff_size = 36 + data_size
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        riff_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        NUM_CHANNELS,
        SAMPLE_RATE,
        byte_rate,
        block_align,
        BITS_PER_SAMPLE,
        b"data",
        data_size,
    )


def _audio_to_pcm(audio: np.ndarray) -> bytes:
    """Convert a float numpy audio array to 16-bit PCM bytes."""
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767).astype(np.int16)
    return pcm.tobytes()


def _audio_to_mp3(audio: np.ndarray) -> bytes:
    """Encode a float numpy audio array as MP3."""
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
    encoder = lameenc.Encoder()
    encoder.set_channels(NUM_CHANNELS)
    encoder.set_in_sample_rate(SAMPLE_RATE)
    encoder.set_bit_rate(64)
    encoder.set_quality(2)
    return bytes(encoder.encode(pcm) + encoder.flush())


MAX_CHUNK_CHARS = 250


def _safe_chunks(text: str) -> list[str]:
    """Split text into chunks of at most MAX_CHUNK_CHARS, preferring sentence boundaries."""
    chunks: list[str] = []
    while len(text) > MAX_CHUNK_CHARS:
        period = text.rfind(".", 0, MAX_CHUNK_CHARS)
        if period > 0:
            chunks.append(text[: period + 1].strip())
            text = text[period + 1 :].strip()
        else:
            space = text.rfind(" ", 0, MAX_CHUNK_CHARS)
            if space <= 0:
                space = MAX_CHUNK_CHARS
            chunks.append(text[:space].strip())
            text = text[space:].strip()
    if text.strip():
        chunks.append(text.strip())
    return chunks


async def _generate_chunk(model: KittenTTS, text: str, voice: str, speed: float) -> np.ndarray:
    """Preprocess a single text chunk and return raw audio samples."""
    preprocessed = await asyncio.to_thread(model.model.preprocessor, text)
    audio = await asyncio.to_thread(
        model.model.generate_single_chunk, preprocessed, voice, speed
    )
    return audio.flatten()


CROSSFADE_SAMPLES = 2400  # 100ms at 24 kHz


def _crossfade_chunks(parts: list[np.ndarray]) -> np.ndarray:
    """Concatenate audio arrays with a short crossfade to eliminate boundary clicks."""
    if not parts:
        return np.array([], dtype=np.float32)
    if len(parts) == 1:
        return parts[0]

    result = parts[0]
    for nxt in parts[1:]:
        fade = min(CROSSFADE_SAMPLES, len(result), len(nxt))
        if fade > 0:
            ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
            result[-fade:] = result[-fade:] * (1.0 - ramp) + nxt[:fade] * ramp
            result = np.concatenate([result, nxt[fade:]])
        else:
            result = np.concatenate([result, nxt])
    return result


async def _generate_audio(model: KittenTTS, text: str, voice: str, speed: float) -> bytes:
    """Generate a complete MP3 file from text."""
    chunks = _safe_chunks(text)
    if not chunks:
        return _audio_to_mp3(np.array([], dtype=np.float32))

    audio_parts = [await _generate_chunk(model, c, voice, speed) for c in chunks]
    return _audio_to_mp3(_crossfade_chunks(audio_parts))


async def _stream_audio(model: KittenTTS, text: str, voice: str, speed: float) -> AsyncIterator[bytes]:
    """Stream WAV audio chunk-by-chunk as each text segment is synthesized."""
    chunks = _safe_chunks(text)
    if not chunks:
        yield _build_wav_header(0)
        return

    yield _build_wav_header(0xFFFFFFFF - 36)

    for chunk in chunks:
        audio = await _generate_chunk(model, chunk, voice, speed)
        yield _audio_to_pcm(audio)


# -- routes ------------------------------------------------------------------

class SpeechRequest(BaseModel):
    input: str
    voice: str = "Bella"
    model: str = ""
    speed: float = Field(default=1.0, gt=0.0, le=5.0)
    stream: bool = False


@app.post("/v1/audio/speech", response_model=None)
async def create_speech(body: SpeechRequest) -> Response:
    tts_model = app.state.tts_model

    if not body.input.strip():
        return Response(status_code=400, content="input text must not be empty")

    if body.stream:
        return StreamingResponse(
            _stream_audio(tts_model, body.input, voice=body.voice, speed=body.speed),
            media_type="audio/wav",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "Transfer-Encoding": "chunked",
            },
        )

    mp3_bytes = await _generate_audio(
        tts_model, body.input, voice=body.voice, speed=body.speed
    )
    return Response(content=mp3_bytes, media_type="audio/mpeg")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
