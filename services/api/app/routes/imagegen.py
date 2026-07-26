from __future__ import annotations

import time
import uuid

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import Response

from app.config import settings
from app.logging import get_logger, log_error, log_request, log_response

logger = get_logger("backend.routes.imagegen")

router = APIRouter()


async def _proxy(request: Request, method: str, path: str) -> Response:
    """Forward a request to the imagegen service, preserving status and body."""
    client: httpx.AsyncClient = request.state.http_client
    request_id = str(uuid.uuid4())
    url = f"{settings.imagegen_url.rstrip('/')}{path}"
    body = await request.json() if method == "POST" else None

    log_request(
        logger,
        request_id=request_id,
        method=method,
        path=path,
        model=(body or {}).get("model") if isinstance(body, dict) else None,
        stream=False,
    )
    start = time.monotonic()

    try:
        if method == "POST":
            response = await client.post(url, json=body)
        else:
            response = await client.get(url)
        latency_ms = (time.monotonic() - start) * 1000
        log_response(
            logger,
            request_id=request_id,
            method=method,
            path=path,
            status_code=response.status_code,
            latency_ms=latency_ms,
        )
        return Response(
            content=response.content,
            status_code=response.status_code,
            media_type=response.headers.get("content-type", "application/json"),
        )
    except httpx.HTTPError as exc:
        latency_ms = (time.monotonic() - start) * 1000
        log_error(
            logger,
            request_id=request_id,
            method=method,
            path=path,
            error=str(exc),
            latency_ms=latency_ms,
        )
        raise


@router.post("/v1/images/generations", response_model=None)
async def create_image(request: Request) -> Response:
    return await _proxy(request, "POST", "/v1/images/generations")


@router.get("/v1/images/models", response_model=None)
async def list_image_models(request: Request) -> Response:
    return await _proxy(request, "GET", "/v1/images/models")


@router.post("/v1/images/models/load", response_model=None)
async def load_image_model(request: Request) -> Response:
    return await _proxy(request, "POST", "/v1/images/models/load")


@router.post("/v1/images/models/unload", response_model=None)
async def unload_image_model(request: Request) -> Response:
    return await _proxy(request, "POST", "/v1/images/models/unload")
