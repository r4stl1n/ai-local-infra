from __future__ import annotations

import json
import time
import uuid
from typing import Any

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from app.config import settings
from app.cost import estimate_cost
from app.errors import error_response, normalized_error_content
from app.logging import get_logger, log_error, log_request, log_response
from app.ollama import (
    build_ollama_body,
    is_thinking_enabled,
    ollama_stream_to_sse,
    ollama_to_openai_response,
    open_stream,
    strip_think_tags,
)
from app.proxy import proxy_request, relay_sse_stream
from app.upstream import resolve_remote_target

router = APIRouter()
logger = get_logger("backend.chat")

_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}

# Gateway-internal fields never forwarded verbatim to a remote
# OpenAI-compatible upstream.
_GATEWAY_ONLY_FIELDS = frozenset({"think", "options", "extra_body"})


@router.post("/v1/chat/completions", response_model=None)
async def chat_completions(request: Request) -> Response:
    try:
        body: Any = await request.json()
    except json.JSONDecodeError:
        return error_response("Request body must be valid JSON.", 400)
    if not isinstance(body, dict):
        return error_response("Request body must be a JSON object.", 400)

    model = body.get("model")
    if not isinstance(model, str) or not model.strip():
        return error_response("you must provide a model parameter", 400, param="model")
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return error_response("'messages' must be a non-empty array", 400, param="messages")

    client: httpx.AsyncClient = request.state.http_client

    if settings.llm_provider == "local":
        return await _handle_local(body, client)
    return await _handle_remote(body, client)


async def _handle_local(body: dict[str, Any], client: httpx.AsyncClient) -> Response:
    """Route through Ollama's native /api/chat (honours think: false)."""
    thinking_enabled = is_thinking_enabled(body, settings.llm_thinking)
    ollama_body = build_ollama_body(body, settings)
    url = f"{settings.ollama_url.rstrip('/')}/api/chat"

    request_id = str(uuid.uuid4())
    model = ollama_body.get("model", "")
    log_request(
        logger,
        request_id=request_id,
        method="POST",
        path="/api/chat",
        model=model,
        stream=bool(ollama_body.get("stream")),
        message_count=len(ollama_body.get("messages", [])),
    )
    start = time.monotonic()

    if ollama_body.get("stream"):
        try:
            upstream = await open_stream(client, url=url, body=ollama_body)
        except httpx.HTTPError as exc:
            log_error(
                logger,
                request_id=request_id,
                method="POST",
                path="/api/chat",
                error=str(exc),
                latency_ms=(time.monotonic() - start) * 1000,
            )
            return error_response(f"Upstream LLM request failed: {exc}", 502)

        if upstream.status_code != 200:
            error_body = await upstream.aread()
            await upstream.aclose()
            log_error(
                logger,
                request_id=request_id,
                method="POST",
                path="/api/chat",
                error=error_body.decode("utf-8", errors="replace"),
                status_code=upstream.status_code,
                latency_ms=(time.monotonic() - start) * 1000,
            )
            return JSONResponse(
                content=normalized_error_content(error_body, upstream.status_code),
                status_code=upstream.status_code,
            )

        return StreamingResponse(
            ollama_stream_to_sse(
                upstream,
                model=model,
                strip_think=not thinking_enabled,
                request_id=request_id,
                start=start,
            ),
            media_type="text/event-stream",
            headers=_SSE_HEADERS,
        )

    try:
        resp = await client.post(url, json=ollama_body)
        latency_ms = (time.monotonic() - start) * 1000
        if resp.status_code != 200:
            log_error(
                logger,
                request_id=request_id,
                method="POST",
                path="/api/chat",
                error=resp.text,
                status_code=resp.status_code,
                latency_ms=latency_ms,
            )
            return JSONResponse(
                content=normalized_error_content(resp.text, resp.status_code),
                status_code=resp.status_code,
            )

        openai_resp = ollama_to_openai_response(
            resp.json(),
            strip_think=not thinking_enabled,
        )
        usage = openai_resp.get("usage")
        log_response(
            logger,
            request_id=request_id,
            method="POST",
            path="/api/chat",
            status_code=resp.status_code,
            latency_ms=latency_ms,
            model=model,
            usage=usage,
            cost=estimate_cost(model, usage),
        )
        return JSONResponse(content=openai_resp)
    except httpx.HTTPError as exc:
        latency_ms = (time.monotonic() - start) * 1000
        log_error(
            logger,
            request_id=request_id,
            method="POST",
            path="/api/chat",
            error=str(exc),
            latency_ms=latency_ms,
        )
        raise


def _build_remote_body(body: dict[str, Any], thinking_enabled: bool) -> dict[str, Any]:
    """Build a spec-clean body for an OpenAI-compatible upstream.

    Gateway-internal fields are dropped; ``extra_body`` is flattened into
    the payload the way the OpenAI SDK does client-side.  The vLLM-style
    ``chat_template_kwargs.enable_thinking`` toggle is only injected when
    the caller opted into the extension (explicit ``think`` or their own
    ``chat_template_kwargs``) — strict OpenAI upstreams reject unknown
    fields, and think-tag stripping still sanitises the response.
    """
    upstream_body = {k: v for k, v in body.items() if k not in _GATEWAY_ONLY_FIELDS}

    extra_body = body.get("extra_body")
    if isinstance(extra_body, dict):
        upstream_body.update(extra_body)

    wants_extension = "think" in body or isinstance(upstream_body.get("chat_template_kwargs"), dict)
    if wants_extension:
        kwargs = upstream_body.get("chat_template_kwargs")
        kwargs = dict(kwargs) if isinstance(kwargs, dict) else {}
        kwargs["enable_thinking"] = thinking_enabled
        upstream_body["chat_template_kwargs"] = kwargs

    return upstream_body


async def _handle_remote(body: dict[str, Any], client: httpx.AsyncClient) -> Response:
    """Proxy to a remote OpenAI-compatible endpoint."""
    upstream_base, chat_paths = resolve_remote_target(settings.llm_upstream_url, "chat/completions")
    upstream_headers = settings.llm_upstream_headers
    thinking_enabled = is_thinking_enabled(body, settings.llm_thinking)
    upstream_body = _build_remote_body(body, thinking_enabled)

    if upstream_body.get("stream", False):
        return await _stream_remote(
            client,
            upstream_base=upstream_base,
            chat_paths=chat_paths,
            body=upstream_body,
            headers=upstream_headers,
            sanitize_content=not thinking_enabled,
        )

    response: httpx.Response | None = None
    for idx, chat_path in enumerate(chat_paths):
        response = await proxy_request(
            client,
            upstream_url=upstream_base,
            method="POST",
            path=chat_path,
            body=upstream_body,
            headers=upstream_headers,
        )
        if response.status_code != 404:
            break
        if idx < len(chat_paths) - 1:
            logger.warning(
                "Remote chat path returned 404 (%s%s); retrying with %s",
                upstream_base,
                chat_path,
                chat_paths[idx + 1],
            )

    if response is None:
        return error_response("No remote chat path configured.", 500)

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

    if not thinking_enabled:
        _strip_non_stream_content(payload)
    return JSONResponse(content=payload, status_code=response.status_code)


async def _stream_remote(
    client: httpx.AsyncClient,
    *,
    upstream_base: str,
    chat_paths: list[str],
    body: dict[str, Any],
    headers: dict[str, str] | None,
    sanitize_content: bool,
) -> Response:
    request_id = str(uuid.uuid4())
    model = body.get("model")
    start = time.monotonic()

    upstream: httpx.Response | None = None
    used_path = chat_paths[0]
    for idx, chat_path in enumerate(chat_paths):
        used_path = chat_path
        url = f"{upstream_base.rstrip('/')}{chat_path}"
        log_request(
            logger,
            request_id=request_id,
            method="POST",
            path=chat_path,
            model=model,
            stream=True,
            message_count=len(body.get("messages", [])),
        )
        try:
            upstream = await open_stream(client, url=url, body=body, headers=headers)
        except httpx.HTTPError as exc:
            log_error(
                logger,
                request_id=request_id,
                method="POST",
                path=chat_path,
                error=str(exc),
                latency_ms=(time.monotonic() - start) * 1000,
            )
            return error_response(f"Upstream LLM request failed: {exc}", 502)

        if upstream.status_code == 404 and idx < len(chat_paths) - 1:
            await upstream.aclose()
            logger.warning(
                "Remote chat path returned 404 (%s%s); retrying with %s",
                upstream_base,
                chat_path,
                chat_paths[idx + 1],
            )
            continue
        break

    assert upstream is not None
    if upstream.status_code != 200:
        error_body = await upstream.aread()
        await upstream.aclose()
        log_error(
            logger,
            request_id=request_id,
            method="POST",
            path=used_path,
            error=error_body.decode("utf-8", errors="replace"),
            status_code=upstream.status_code,
            latency_ms=(time.monotonic() - start) * 1000,
        )
        return JSONResponse(
            content=normalized_error_content(error_body, upstream.status_code),
            status_code=upstream.status_code,
        )

    return StreamingResponse(
        relay_sse_stream(
            upstream,
            path=used_path,
            model=model,
            sanitize_content=sanitize_content,
            request_id=request_id,
            start=start,
        ),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


def _strip_non_stream_content(payload: dict[str, Any]) -> None:
    """Strip think tags from non-streaming OpenAI-style response payload."""
    choices = payload.get("choices")
    if not isinstance(choices, list):
        return

    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            message["content"] = strip_think_tags(content)
