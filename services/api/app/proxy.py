from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

import httpx

from app.cost import estimate_cost
from app.logging import get_logger, log_error, log_request, log_response
from app.ollama import strip_think_tags_stream_chunk

logger = get_logger("backend.proxy")


async def proxy_request(
    client: httpx.AsyncClient,
    *,
    upstream_url: str,
    method: str,
    path: str,
    body: dict[str, Any] | None = None,
    query_params: dict[str, str] | None = None,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    """Forward a non-streaming request to an upstream service."""
    url = f"{upstream_url.rstrip('/')}{path}"
    request_id = str(uuid.uuid4())

    log_request(
        logger,
        request_id=request_id,
        method=method,
        path=path,
        model=body.get("model") if body else None,
        stream=False,
        message_count=len(body["messages"]) if body and "messages" in body else None,
    )

    start = time.monotonic()
    try:
        response = await client.request(
            method=method,
            url=url,
            json=body,
            params=query_params,
            headers=headers,
        )
        latency_ms = (time.monotonic() - start) * 1000

        response_data = None
        if response.headers.get("content-type", "").startswith("application/json"):
            response_data = response.json()

        model = response_data.get("model") if response_data else None
        usage = response_data.get("usage") if response_data else None

        log_response(
            logger,
            request_id=request_id,
            method=method,
            path=path,
            status_code=response.status_code,
            latency_ms=latency_ms,
            model=model,
            usage=usage,
            cost=estimate_cost(model, usage),
        )

        return response
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


async def relay_sse_stream(
    response: httpx.Response,
    *,
    path: str,
    model: str | None = None,
    sanitize_content: bool = False,
    request_id: str | None = None,
    start: float | None = None,
) -> AsyncIterator[bytes]:
    """Relay an already-open upstream SSE stream, closing it when done.

    The caller opens the stream (see ``app.ollama.open_stream``) and checks
    the upstream status first, so error statuses reach the client instead
    of a committed 200.
    """
    request_id = request_id or str(uuid.uuid4())
    start = start if start is not None else time.monotonic()
    usage: dict[str, Any] | None = None
    strip_state: dict[str, Any] = {"in_think": False, "partial": ""}

    try:
        try:
            async for line in response.aiter_lines():
                if not line:
                    continue

                if sanitize_content and line.startswith("data: ") and line != "data: [DONE]":
                    payload = line[6:]
                    try:
                        chunk = json.loads(payload)
                    except json.JSONDecodeError:
                        chunk = None

                    if isinstance(chunk, dict):
                        choices = chunk.get("choices")
                        if isinstance(choices, list):
                            for choice in choices:
                                if not isinstance(choice, dict):
                                    continue
                                delta = choice.get("delta")
                                if not isinstance(delta, dict):
                                    continue
                                content = delta.get("content")
                                if isinstance(content, str):
                                    delta["content"] = strip_think_tags_stream_chunk(content, strip_state)
                        line = f"data: {json.dumps(chunk)}"

                yield f"{line}\n\n".encode()

                if line.startswith("data: ") and line != "data: [DONE]":
                    try:
                        chunk = json.loads(line[6:])
                        if "usage" in chunk and chunk["usage"]:
                            usage = chunk["usage"]
                    except (json.JSONDecodeError, KeyError):
                        pass
        finally:
            await response.aclose()

        latency_ms = (time.monotonic() - start) * 1000
        log_response(
            logger,
            request_id=request_id,
            method="POST",
            path=path,
            status_code=response.status_code,
            latency_ms=latency_ms,
            model=model,
            usage=usage,
            cost=estimate_cost(model, usage),
        )
    except httpx.HTTPError as exc:
        latency_ms = (time.monotonic() - start) * 1000
        log_error(
            logger,
            request_id=request_id,
            method="POST",
            path=path,
            error=str(exc),
            latency_ms=latency_ms,
        )
        raise
