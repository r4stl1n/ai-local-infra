from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.config import settings
from app.errors import error_response, normalized_error_content
from app.logging import get_logger
from app.proxy import proxy_request
from app.upstream import resolve_remote_target

router = APIRouter()
logger = get_logger("backend.models")


@router.get("/v1/models")
async def list_models(request: Request) -> JSONResponse:
    client: httpx.AsyncClient = request.state.http_client

    if settings.llm_provider == "local":
        upstream_base = settings.ollama_v1_url
        model_paths = ["/models"]
    else:
        upstream_base, model_paths = resolve_remote_target(settings.llm_upstream_url, "models")

    response: httpx.Response | None = None
    for idx, path in enumerate(model_paths):
        response = await proxy_request(
            client,
            upstream_url=upstream_base,
            method="GET",
            path=path,
            headers=settings.llm_upstream_headers,
        )
        if response.status_code != 404:
            break
        if idx < len(model_paths) - 1:
            logger.warning(
                "Models path returned 404 (%s%s); retrying with %s",
                upstream_base,
                path,
                model_paths[idx + 1],
            )

    if response is None:
        return error_response("No models path configured.", 500)

    try:
        payload = response.json()
    except ValueError:
        return JSONResponse(
            content=normalized_error_content(response.text, response.status_code),
            status_code=response.status_code if response.status_code >= 400 else 502,
        )

    if response.status_code >= 400:
        return JSONResponse(
            content=normalized_error_content(payload, response.status_code),
            status_code=response.status_code,
        )

    return JSONResponse(content=payload, status_code=response.status_code)


# How long /v1/models/unload waits for Ollama to actually release a runner
# (its unload call returns before the VRAM is freed).
_UNLOAD_WAIT_SECONDS = 30.0


async def _ollama(client: httpx.AsyncClient, method: str, path: str,
                  body: dict[str, Any] | None = None) -> tuple[Any, JSONResponse | None]:
    """Call the bundled Ollama's native API; returns (payload, error_response)."""
    response = await proxy_request(
        client, upstream_url=settings.ollama_url, method=method, path=path, body=body
    )
    try:
        payload = response.json()
    except ValueError:
        payload = response.text
    if response.status_code >= 400:
        return None, JSONResponse(
            content=normalized_error_content(payload, response.status_code),
            status_code=response.status_code,
        )
    return payload, None


def _running(payload: Any) -> list[dict[str, Any]]:
    return payload.get("models") or [] if isinstance(payload, dict) else []


def _with_tag(name: str) -> str:
    """Ollama reports untagged models as `<name>:latest`."""
    return name if ":" in name.rsplit("/", 1)[-1] else f"{name}:latest"


@router.get("/v1/models/loaded")
async def list_loaded_models(request: Request) -> JSONResponse:
    """Models currently resident in the bundled Ollama (LLMs and embeddings)."""
    payload, error = await _ollama(request.state.http_client, "GET", "/api/ps")
    if error is not None:
        return error
    return JSONResponse(content={
        "data": [
            {
                "id": m.get("name") or m.get("model"),
                "size_vram": m.get("size_vram"),
                "expires_at": m.get("expires_at"),
            }
            for m in _running(payload)
        ],
    })


@router.post("/v1/models/unload")
async def unload_models(request: Request) -> JSONResponse:
    """Unload one model (`{"model": "<id>"}`) or, with no model, every model
    resident in the bundled Ollama, freeing their VRAM. Ollama reloads a model
    on its next request."""
    client: httpx.AsyncClient = request.state.http_client
    raw = await request.body()
    try:
        body = json.loads(raw) if raw.strip() else {}
    except ValueError:
        return error_response("body must be JSON", 400)
    model = body.get("model") if isinstance(body, dict) else None
    if model is not None and (not isinstance(model, str) or not model.strip()):
        return error_response("model must be a non-empty string", 400, param="model")

    if model:
        targets = [model.strip()]
    else:
        payload, error = await _ollama(client, "GET", "/api/ps")
        if error is not None:
            return error
        targets = [m.get("name") or m.get("model") for m in _running(payload)]

    for name in targets:
        # keep_alive=0 with no prompt is Ollama's unload request.
        _, error = await _ollama(client, "POST", "/api/generate", {"model": name, "keep_alive": 0})
        if error is not None:
            return error

    # Wait until the runners are gone, so the VRAM is free when we answer.
    deadline = time.monotonic() + _UNLOAD_WAIT_SECONDS
    while targets:
        payload, error = await _ollama(client, "GET", "/api/ps")
        if error is not None:
            return error
        resident = {_with_tag(m.get("name") or m.get("model")) for m in _running(payload)}
        if not resident.intersection(map(_with_tag, targets)) or time.monotonic() >= deadline:
            break
        await asyncio.sleep(0.25)

    logger.info("Unloaded LLM models: %s", targets)
    return JSONResponse(content={"unloaded": targets})
