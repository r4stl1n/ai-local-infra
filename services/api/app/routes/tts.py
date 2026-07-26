from __future__ import annotations

import time
import uuid
from typing import Any

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import Response, StreamingResponse

from app.config import settings
from app.logging import get_logger, log_error, log_request, log_response

logger = get_logger("backend.routes.tts")

router = APIRouter()


@router.post("/v1/audio/speech", response_model=None)
async def create_speech(request: Request) -> Response:
    body: dict[str, Any] = await request.json()
    client: httpx.AsyncClient = request.state.http_client
    request_id = str(uuid.uuid4())
    url = f"{settings.tts_url.rstrip('/')}/v1/audio/speech"

    log_request(
        logger,
        request_id=request_id,
        method="POST",
        path="/v1/audio/speech",
        model=body.get("model"),
        stream=body.get("stream", False),
    )

    start = time.monotonic()

    if body.get("stream", False):
        async def _stream():
            try:
                async with client.stream("POST", url, json=body) as resp:
                    latency_ms = (time.monotonic() - start) * 1000
                    log_response(
                        logger,
                        request_id=request_id,
                        method="POST",
                        path="/v1/audio/speech",
                        status_code=resp.status_code,
                        latency_ms=latency_ms,
                    )
                    async for chunk in resp.aiter_bytes():
                        yield chunk
            except httpx.HTTPError as exc:
                latency_ms = (time.monotonic() - start) * 1000
                log_error(
                    logger,
                    request_id=request_id,
                    method="POST",
                    path="/v1/audio/speech",
                    error=str(exc),
                    latency_ms=latency_ms,
                )
                raise

        return StreamingResponse(
            _stream(),
            media_type="audio/wav",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "Transfer-Encoding": "chunked",
            },
        )

    try:
        response = await client.post(url, json=body)
        latency_ms = (time.monotonic() - start) * 1000
        log_response(
            logger,
            request_id=request_id,
            method="POST",
            path="/v1/audio/speech",
            status_code=response.status_code,
            latency_ms=latency_ms,
        )
        return Response(
            content=response.content,
            status_code=response.status_code,
            media_type=response.headers.get("content-type", "audio/wav"),
        )
    except httpx.HTTPError as exc:
        latency_ms = (time.monotonic() - start) * 1000
        log_error(
            logger,
            request_id=request_id,
            method="POST",
            path="/v1/audio/speech",
            error=str(exc),
            latency_ms=latency_ms,
        )
        raise
