from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3")
WHISPER_CACHE = os.getenv("WHISPER_CACHE", "/data/whisper")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
WHISPER_SAMPLE_RATE = 16000
# Interim hypotheses on the WS stream: re-decode the open segment once at
# least this much new audio arrived (self-paced; 0 disables partials).
WHISPER_PARTIAL_MS = int(os.getenv("WHISPER_PARTIAL_MS", "600"))
# LocalAgreement-2 partial stability: emit only words two consecutive decodes
# agree on, so the partial line never rewrites itself. 0 = raw hypotheses.
WHISPER_PARTIAL_STABLE = os.getenv("WHISPER_PARTIAL_STABLE", "1").strip().lower() not in ("0", "false", "no", "off")
_ENV_KEYS = ("WHISPER_MODEL", "WHISPER_CACHE", "WHISPER_PARTIAL_MS", "WHISPER_PARTIAL_STABLE", "LOG_LEVEL")
_SENSITIVE_ENV_MARKERS = ("PASSWORD", "TOKEN", "KEY", "SECRET")

logger = logging.getLogger("whisper")
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


def _load_model(model_name: str) -> WhisperModel:
    logger.info("Loading Whisper model: %s", model_name)
    kwargs = {
        "device": "cuda",
        "compute_type": "float16",
        "download_root": WHISPER_CACHE,
    }
    # Cache-first so startup works without network once the model is downloaded.
    try:
        model = WhisperModel(model_name, local_files_only=True, **kwargs)
    except Exception as exc:
        logger.info(
            "Cache-only load failed (%s: %s); retrying with download enabled",
            type(exc).__name__,
            exc,
        )
        model = WhisperModel(model_name, **kwargs)
    logger.info("Whisper model loaded successfully")
    return model


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    _log_startup_env("whisper", _ENV_KEYS)
    app.state.whisper_model = _load_model(WHISPER_MODEL)
    yield


app = FastAPI(title="Whisper STT", version="0.1.0", lifespan=lifespan)


# -- transcription helpers ---------------------------------------------------

async def _transcribe_audio(
    model: WhisperModel,
    audio_bytes: bytes,
    language: str | None = None,
) -> dict:
    """Transcribe uploaded audio bytes and return the full text."""

    def _run() -> str:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            tmp.write(audio_bytes)
            tmp.flush()
            kwargs: dict = {"beam_size": 5, "vad_filter": True}
            if language:
                kwargs["language"] = language
            segments, _info = model.transcribe(tmp.name, **kwargs)
            return " ".join(seg.text.strip() for seg in segments)

    text = await asyncio.to_thread(_run)
    return {"text": text}


def _pcm16_to_float32(pcm_bytes: bytes) -> np.ndarray:
    """Convert raw 16-bit signed little-endian PCM to float32 in [-1, 1]."""
    samples = np.frombuffer(pcm_bytes, dtype=np.int16)
    return samples.astype(np.float32) / 32768.0


def _decode_final(model: WhisperModel, pcm: bytes, vad_filter: bool = True) -> dict:
    """Full-quality decode used when a segment ends.

    Returns text plus the decoder's own confidence signals so callers can do
    principled hallucination filtering (high no_speech_prob + low avg_logprob
    is Whisper's signature for invented text over speechless audio) and
    language routing without a second model:
      language / language_probability — detected input language
      avg_logprob      — duration-weighted mean segment log-probability
      no_speech_prob   — duration-weighted mean no-speech probability
    """
    audio = _pcm16_to_float32(pcm)
    segments, info = model.transcribe(audio, beam_size=5, vad_filter=vad_filter)
    segs = list(segments)  # generator: decoding happens here
    text = " ".join(seg.text.strip() for seg in segs)
    if segs:
        weights = [max(seg.end - seg.start, 1e-3) for seg in segs]
        total = sum(weights)
        avg_logprob = sum(s.avg_logprob * w for s, w in zip(segs, weights)) / total
        no_speech = sum(s.no_speech_prob * w for s, w in zip(segs, weights)) / total
    else:
        avg_logprob, no_speech = 0.0, 1.0
    return {
        "type": "final",
        "text": text,
        "language": info.language,
        "language_probability": round(float(info.language_probability), 3),
        "avg_logprob": round(float(avg_logprob), 3),
        "no_speech_prob": round(float(no_speech), 3),
    }


def _decode_partial(model: WhisperModel, pcm: bytes) -> str:
    """Fast greedy decode for interim hypotheses of a still-open segment.

    condition_on_previous_text=False keeps repeated re-decodes of the same
    growing buffer from feeding their own output back in (hallucination loops).
    """
    audio = _pcm16_to_float32(pcm)
    segments, _info = model.transcribe(
        audio, beam_size=1, vad_filter=True, condition_on_previous_text=False
    )
    return " ".join(seg.text.strip() for seg in segments)


# -- routes ------------------------------------------------------------------

@app.post("/v1/audio/transcriptions")
async def create_transcription(
    file: UploadFile,
    language: str | None = None,
    model: str | None = None,
) -> JSONResponse:
    whisper_model = app.state.whisper_model

    audio_bytes = await file.read()
    if not audio_bytes:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "uploaded file is empty", "type": "invalid_request_error"}},
        )

    result = await _transcribe_audio(whisper_model, audio_bytes, language=language)
    return JSONResponse(content=result)


@app.websocket("/v1/audio/transcriptions/stream")
async def stream_transcription(ws: WebSocket) -> None:
    """Streaming transcription with interim hypotheses.

    Protocol (client -> server): binary PCM16 chunks accumulate a segment;
    `{"type":"end"}` decodes it and emits `{"type":"final","text":...}`;
    `{"type":"drop"}` discards the buffer silently (used by callers that
    re-send a cleaned-up version of the same audio before ending).

    Server -> client: while a segment is open, the growing buffer is
    re-decoded (greedy, self-paced: a new decode starts only when the previous
    one finished AND >= WHISPER_PARTIAL_MS of new audio arrived) and emitted
    as `{"type":"partial","text":...}`. Whisper cannot stream token-by-token,
    so re-decoding the buffer is the standard way to fake it. A generation
    counter guards against a slow partial landing after its segment ended.
    Set WHISPER_PARTIAL_MS=0 to disable partials entirely.

    Partial stability (WHISPER_PARTIAL_STABLE, default on): LocalAgreement-2 —
    a word is emitted only once two CONSECUTIVE re-decodes agree on it as a
    prefix, so the partial line only ever grows and never rewrites itself
    (~one extra decode cycle of latency per word). Off = raw hypotheses,
    which may revise earlier words. The final always supersedes partials.
    """
    await ws.accept()
    whisper_model = ws.app.state.whisper_model

    buf = bytearray()
    generation = 0
    decoded_upto = 0          # buffer size at the last started partial decode
    partial_busy = False
    prev_words: list[str] = []       # last hypothesis (for LocalAgreement)
    committed: list[str] = []        # agreed prefix; only ever extends
    last_sent = ""                   # dedupe: identical partials are not resent
    send_lock = asyncio.Lock()  # partial task and handler both send frames
    min_new = WHISPER_SAMPLE_RATE * 2 * WHISPER_PARTIAL_MS // 1000
    min_total = WHISPER_SAMPLE_RATE * 2 // 2  # >= 0.5s before first partial

    def _reset_partial_state() -> None:
        nonlocal decoded_upto, prev_words, committed, last_sent
        decoded_upto = 0
        prev_words = []
        committed = []
        last_sent = ""

    async def _emit_partial() -> None:
        nonlocal partial_busy, decoded_upto, prev_words, committed, last_sent
        my_generation = generation
        snapshot = bytes(buf)
        decoded_upto = len(snapshot)
        try:
            text = await asyncio.to_thread(_decode_partial, whisper_model, snapshot)
        except Exception:  # noqa: BLE001 - a failed partial must never kill the stream
            logger.exception("partial decode failed")
            return
        finally:
            partial_busy = False
        if my_generation != generation:
            return  # segment ended while decoding; discard
        if WHISPER_PARTIAL_STABLE:
            words = text.split()
            agree = 0
            for a, b in zip(prev_words, words):
                if a != b:
                    break
                agree += 1
            prev_words = words
            if agree > len(committed):
                committed = words[:agree]
            text = " ".join(committed)
        if text and text != last_sent:
            last_sent = text
            async with send_lock:
                await ws.send_json({"type": "partial", "text": text})

    try:
        while True:
            message = await ws.receive()

            if message.get("type") == "websocket.disconnect":
                break

            if "bytes" in message and message["bytes"]:
                buf.extend(message["bytes"])
                if (
                    WHISPER_PARTIAL_MS > 0
                    and not partial_busy
                    and len(buf) >= min_total
                    and len(buf) - decoded_upto >= min_new
                ):
                    partial_busy = True
                    asyncio.create_task(_emit_partial())
            elif "text" in message:
                try:
                    payload = json.loads(message["text"])
                except json.JSONDecodeError:
                    async with send_lock:
                        await ws.send_json({"type": "error", "message": "invalid JSON"})
                    continue

                kind = payload.get("type")
                if kind == "end":
                    generation += 1  # invalidate any in-flight partial
                    pcm = bytes(buf)
                    buf.clear()
                    _reset_partial_state()
                    # vad_filter=false lets callers decode audio Silero would
                    # reject as non-speech — sung vocals score ~0 on it, so
                    # the filter can strip a whole melodic segment to silence.
                    use_vad = payload.get("vad_filter", True) is not False
                    result = (
                        await asyncio.to_thread(_decode_final, whisper_model, pcm, use_vad)
                        if pcm
                        else {"type": "final", "text": ""}
                    )
                    async with send_lock:
                        await ws.send_json(result)
                elif kind == "drop":
                    generation += 1
                    buf.clear()
                    _reset_partial_state()

    except WebSocketDisconnect:
        pass


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
