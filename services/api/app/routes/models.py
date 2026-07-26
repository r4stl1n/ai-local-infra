from __future__ import annotations

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
