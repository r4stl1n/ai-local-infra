from __future__ import annotations

import asyncio
import hmac
import time
import uuid

import httpx
import websockets
from fastapi import APIRouter, Query, Request, UploadFile, WebSocket, WebSocketDisconnect, status
from fastapi.responses import JSONResponse

from app.config import settings
from app.logging import get_logger, log_error, log_request, log_response

logger = get_logger("backend.routes.stt")

router = APIRouter()


@router.post("/v1/audio/transcriptions")
async def create_transcription(
    request: Request,
    file: UploadFile,
    language: str | None = None,
    model: str | None = None,
) -> JSONResponse:
    client: httpx.AsyncClient = request.state.http_client
    request_id = str(uuid.uuid4())
    url = f"{settings.whisper_url.rstrip('/')}/v1/audio/transcriptions"

    log_request(
        logger,
        request_id=request_id,
        method="POST",
        path="/v1/audio/transcriptions",
        model=model,
    )

    audio_bytes = await file.read()
    if not audio_bytes:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "uploaded file is empty", "type": "invalid_request_error"}},
        )

    start = time.monotonic()

    try:
        params = {}
        if language:
            params["language"] = language
        if model:
            params["model"] = model

        response = await client.post(
            url,
            files={"file": (file.filename or "audio.wav", audio_bytes, file.content_type or "audio/wav")},
            params=params,
        )
        latency_ms = (time.monotonic() - start) * 1000

        log_response(
            logger,
            request_id=request_id,
            method="POST",
            path="/v1/audio/transcriptions",
            status_code=response.status_code,
            latency_ms=latency_ms,
        )

        return JSONResponse(content=response.json(), status_code=response.status_code)
    except httpx.HTTPError as exc:
        latency_ms = (time.monotonic() - start) * 1000
        log_error(
            logger,
            request_id=request_id,
            method="POST",
            path="/v1/audio/transcriptions",
            error=str(exc),
            latency_ms=latency_ms,
        )
        raise


@router.websocket("/v1/audio/transcriptions/stream")
async def stream_transcription(ws: WebSocket, token: str = Query("")) -> None:
    if not token or not hmac.compare_digest(token, settings.api_key):
        await ws.close(code=status.WS_1008_POLICY_VIOLATION, reason="Invalid or missing API key")
        return

    await ws.accept()

    whisper_ws_url = (
        settings.whisper_url
        .replace("http://", "ws://")
        .replace("https://", "wss://")
        .rstrip("/")
        + "/v1/audio/transcriptions/stream"
    )

    try:
        async with websockets.connect(whisper_ws_url) as upstream_ws:
            async def client_to_upstream():
                try:
                    while True:
                        message = await ws.receive()
                        if message.get("type") == "websocket.disconnect":
                            break
                        if "bytes" in message and message["bytes"]:
                            await upstream_ws.send(message["bytes"])
                        elif "text" in message and message["text"]:
                            await upstream_ws.send(message["text"])
                except WebSocketDisconnect:
                    pass

            async def upstream_to_client():
                try:
                    async for message in upstream_ws:
                        if isinstance(message, bytes):
                            await ws.send_bytes(message)
                        else:
                            await ws.send_text(message)
                except websockets.ConnectionClosed:
                    pass

            await asyncio.gather(
                client_to_upstream(),
                upstream_to_client(),
                return_exceptions=True,
            )
    except Exception:
        logger.exception("WebSocket proxy error")
    finally:
        try:
            await ws.close()
        except Exception:
            pass
