"""Jev-style decision models (Ollama >= 0.35 `POST /v1/systemone`).

A request carries a `state` plus named, typed `questions` (`choice`, `noul`,
`score`); the model answers all of them in one call with calibrated
probabilities. The bundled Ollama serves it regardless of LLM_PROVIDER (like
embeddings), so the TypeSafe SDK works against the gateway with
TYPESAFE_BASE_URL=<gateway> and TYPESAFE_API_KEY=<API_KEY>.
"""

from __future__ import annotations

from typing import Any

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

from app.config import settings
from app.errors import error_response, normalized_error_content
from app.proxy import proxy_request

router = APIRouter()


@router.post("/v1/systemone", response_model=None)
async def systemone(request: Request) -> Response:
    try:
        body: Any = await request.json()
    except ValueError:
        return error_response("body must be JSON", 400)
    if not isinstance(body, dict):
        return error_response("body must be a JSON object", 400)

    client: httpx.AsyncClient = request.state.http_client
    response = await proxy_request(
        client,
        upstream_url=settings.ollama_url,
        method="POST",
        path="/v1/systemone",
        body=body,
    )
    if response.status_code >= 400:
        try:
            detail: Any = response.json()
        except ValueError:
            detail = response.text
            # A model error is JSON; a plain-text 404 means the route itself is missing.
            if response.status_code == 404:
                detail = "Ollama has no /v1/systemone endpoint; decision models need Ollama >= 0.35"
        return JSONResponse(
            content=normalized_error_content(detail, response.status_code),
            status_code=response.status_code,
        )
    # Pass the answers through untouched: SDKs parse the type-specific fields.
    return Response(
        content=response.content,
        status_code=response.status_code,
        media_type=response.headers.get("content-type", "application/json"),
    )
