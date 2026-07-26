from __future__ import annotations

import time
import uuid

import httpx
from fastapi import APIRouter, Request, UploadFile
from fastapi.responses import JSONResponse, Response

from app.config import settings
from app.logging import get_logger, log_error, log_request, log_response

logger = get_logger("backend.routes.separation")

router = APIRouter()


@router.post("/v1/audio/separations", response_model=None)
async def create_separation(
    request: Request,
    file: UploadFile,
    stem: str = "vocals",
    sample_rate: int | None = None,
    mono: bool = False,
    model: str | None = None,
) -> Response:
    client: httpx.AsyncClient = request.state.http_client
    request_id = str(uuid.uuid4())
    url = f"{settings.demucs_url.rstrip('/')}/v1/audio/separations"

    log_request(
        logger,
        request_id=request_id,
        method="POST",
        path="/v1/audio/separations",
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
        params: dict = {"stem": stem, "mono": mono}
        if sample_rate is not None:
            params["sample_rate"] = sample_rate
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
            path="/v1/audio/separations",
            status_code=response.status_code,
            latency_ms=latency_ms,
        )

        if response.status_code != 200:
            # Upstream errors are JSON in the OpenAI shape; pass them through.
            return Response(
                content=response.content,
                status_code=response.status_code,
                media_type=response.headers.get("content-type", "application/json"),
            )
        return Response(
            content=response.content,
            media_type=response.headers.get("content-type", "audio/wav"),
        )
    except httpx.HTTPError as exc:
        latency_ms = (time.monotonic() - start) * 1000
        log_error(
            logger,
            request_id=request_id,
            method="POST",
            path="/v1/audio/separations",
            error=str(exc),
            latency_ms=latency_ms,
        )
        raise
