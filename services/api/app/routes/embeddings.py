from __future__ import annotations

from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from app.config import settings
from app.proxy import proxy_request

router = APIRouter()


@router.post("/v1/embeddings")
async def embeddings(request: Request) -> JSONResponse:
    body: dict[str, Any] = await request.json()
    model = str(body.get("model", "")).strip()
    raw_input = body.get("input")

    if not model:
        raise HTTPException(status_code=400, detail="Field 'model' is required")
    if raw_input is None:
        raise HTTPException(status_code=400, detail="Field 'input' is required")

    client: httpx.AsyncClient = request.state.http_client
    upstream_payload = {
        "model": model,
        "input": raw_input,
    }

    # Ollama endpoint differs by version:
    # - newer: /api/embed
    # - older: /api/embeddings
    response = await proxy_request(
        client,
        upstream_url=settings.ollama_url,
        method="POST",
        path="/api/embed",
        body=upstream_payload,
    )
    if response.status_code == 404:
        response = await proxy_request(
            client,
            upstream_url=settings.ollama_url,
            method="POST",
            path="/api/embeddings",
            body=upstream_payload,
        )

    if response.status_code >= 400:
        # Preserve upstream status/error instead of raising internal 500.
        detail: Any
        try:
            detail = response.json()
        except ValueError:
            detail = response.text
        raise HTTPException(status_code=response.status_code, detail=detail)

    try:
        upstream = response.json()
    except ValueError as exc:
        raise HTTPException(status_code=502, detail="Invalid JSON response from Ollama embeddings endpoint") from exc

    embeddings_payload = upstream.get("embeddings")
    if embeddings_payload is None:
        # Some Ollama versions may return a singular embedding for single input.
        maybe_single = upstream.get("embedding")
        embeddings_payload = [maybe_single] if maybe_single is not None else []

    if not isinstance(embeddings_payload, list):
        raise HTTPException(status_code=502, detail="Unexpected embeddings response from Ollama")
    if embeddings_payload and isinstance(embeddings_payload[0], (float, int)):
        embeddings_payload = [embeddings_payload]

    data: list[dict[str, Any]] = []
    for idx, emb in enumerate(embeddings_payload):
        if not isinstance(emb, list):
            raise HTTPException(status_code=502, detail="Unexpected embedding vector format from Ollama")
        data.append({
            "object": "embedding",
            "index": idx,
            "embedding": emb,
        })

    prompt_tokens = upstream.get("prompt_eval_count", 0) or 0
    payload = {
        "object": "list",
        "data": data,
        "model": model,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "total_tokens": prompt_tokens,
        },
    }
    return JSONResponse(content=payload, status_code=200)
