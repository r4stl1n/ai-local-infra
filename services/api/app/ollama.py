"""Ollama native /api/chat translation layer.

Translates between OpenAI chat-completion format (used by the agent) and
Ollama's native /api/chat request/response format, which properly honours
the ``think`` parameter.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

import httpx

from app.config import Settings
from app.cost import estimate_cost
from app.logging import get_logger, log_error, log_response

logger = get_logger("backend.ollama")

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", flags=re.IGNORECASE | re.DOTALL)
_THINK_TAG_RE = re.compile(r"</?think>", flags=re.IGNORECASE)
_OPEN_THINK_TAG = "<think>"
_CLOSE_THINK_TAG = "</think>"
_MAX_THINK_TAG_LEN = len(_CLOSE_THINK_TAG)


def is_thinking_enabled(body: dict[str, Any], default: bool) -> bool:
    """Resolve request-level thinking setting."""
    raw = body.get("think")
    if isinstance(raw, bool):
        return raw
    return default


def strip_think_tags(text: str) -> str:
    """Remove think blocks/tags from a full text payload."""
    if not text:
        return text
    without_blocks = _THINK_BLOCK_RE.sub("", text)
    return _THINK_TAG_RE.sub("", without_blocks)


def _split_partial_tag_suffix(text: str) -> tuple[str, str]:
    """Split off an incomplete trailing think tag fragment."""
    max_check = min(len(text), _MAX_THINK_TAG_LEN - 1)
    for n in range(max_check, 0, -1):
        suffix = text[-n:]
        if _OPEN_THINK_TAG.startswith(suffix.lower()) or _CLOSE_THINK_TAG.startswith(suffix.lower()):
            return text[:-n], suffix
    return text, ""


def strip_think_tags_stream_chunk(text: str, state: dict[str, Any]) -> str:
    """Remove think content from a streaming delta while preserving state."""
    if not text:
        return ""

    combined = f"{state.get('partial', '')}{text}"
    process_text, partial = _split_partial_tag_suffix(combined)
    state["partial"] = partial

    out: list[str] = []
    i = 0
    lower = process_text.lower()
    in_think = bool(state.get("in_think", False))

    while i < len(process_text):
        if not in_think:
            next_open = lower.find(_OPEN_THINK_TAG, i)
            next_close = lower.find(_CLOSE_THINK_TAG, i)
            tag_positions = [pos for pos in (next_open, next_close) if pos != -1]
            if not tag_positions:
                out.append(process_text[i:])
                break

            next_tag = min(tag_positions)
            out.append(process_text[i:next_tag])
            if next_tag == next_open:
                in_think = True
                i = next_open + len(_OPEN_THINK_TAG)
            else:
                i = next_close + len(_CLOSE_THINK_TAG)
        else:
            close_idx = lower.find(_CLOSE_THINK_TAG, i)
            if close_idx == -1:
                i = len(process_text)
                break
            in_think = False
            i = close_idx + len(_CLOSE_THINK_TAG)

    state["in_think"] = in_think
    return "".join(out)


def _flatten_content_parts(content: list[Any]) -> tuple[str, list[str]]:
    """Flatten an OpenAI content-part list into (text, base64_images).

    Ollama's native /api/chat wants ``content`` as a plain string with
    images in a separate base64 ``images`` array.
    """
    texts: list[str] = []
    images: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        part_type = part.get("type")
        if part_type == "text":
            text = part.get("text")
            if isinstance(text, str) and text:
                texts.append(text)
        elif part_type == "image_url":
            image_url = part.get("image_url")
            url = image_url.get("url", "") if isinstance(image_url, dict) else ""
            if url.startswith("data:") and "," in url:
                images.append(url.split(",", 1)[1])
            elif url:
                logger.warning("Dropping non-data-URI image_url part (Ollama needs base64): %s", url[:80])
    return "\n".join(texts), images


def _openai_messages_to_ollama(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalise OpenAI-format messages for Ollama's native /api/chat.

    * Assistant ``tool_calls``: Ollama expects ``arguments`` as a dict,
      OpenAI sends it as a JSON string.
    * Tool-result messages: Ollama doesn't use ``tool_call_id``; strip it.
    * Content-part lists: flatten to a string, extracting base64 images.
    """
    out: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role", "")

        if role == "assistant" and msg.get("tool_calls"):
            converted_calls: list[dict[str, Any]] = []
            for tc in msg["tool_calls"]:
                func = tc.get("function", {})
                raw_args = func.get("arguments", "{}")
                if isinstance(raw_args, str):
                    try:
                        parsed_args = json.loads(raw_args)
                    except (json.JSONDecodeError, TypeError):
                        parsed_args = {}
                else:
                    parsed_args = raw_args
                converted_calls.append({
                    "function": {
                        "name": func.get("name", ""),
                        "arguments": parsed_args,
                    },
                })
            converted: dict[str, Any] = {"role": "assistant"}
            if msg.get("content"):
                converted["content"] = msg["content"]
            converted["tool_calls"] = converted_calls
            out.append(converted)

        elif role == "tool":
            out.append({
                "role": "tool",
                "content": msg.get("content", ""),
            })

        elif isinstance(msg.get("content"), list):
            text, images = _flatten_content_parts(msg["content"])
            converted = {**msg, "content": text}
            if images:
                converted["images"] = images
            out.append(converted)

        else:
            out.append(msg)

    return out


# OpenAI sampling params that map 1:1 onto Ollama option names.
_OPENAI_SAMPLING_TO_OLLAMA_OPTION: dict[str, str] = {
    "temperature": "temperature",
    "top_p": "top_p",
    "seed": "seed",
    "presence_penalty": "presence_penalty",
    "frequency_penalty": "frequency_penalty",
}


def build_ollama_body(body: dict[str, Any], settings: Settings) -> dict[str, Any]:
    """Convert an OpenAI-format request body to Ollama native format.

    Copies fields that Ollama understands, maps OpenAI sampling params
    onto Ollama ``options``, and drops OpenAI/vLLM-specific fields
    (``extra_body``, ``chat_template_kwargs``).
    """
    ollama_body: dict[str, Any] = {
        "model": body["model"],
        "messages": _openai_messages_to_ollama(body["messages"]),
        "stream": body.get("stream", False),
    }

    if "think" in body:
        ollama_body["think"] = body["think"]
    else:
        ollama_body["think"] = settings.llm_thinking

    if body.get("tools"):
        ollama_body["tools"] = body["tools"]

    response_format = body.get("response_format")
    if isinstance(response_format, dict) and response_format.get("type") == "json_object":
        ollama_body["format"] = "json"
    elif isinstance(body.get("format"), str):
        ollama_body["format"] = body["format"]

    options: dict[str, Any] = {}
    for openai_key, ollama_key in _OPENAI_SAMPLING_TO_OLLAMA_OPTION.items():
        value = body.get(openai_key)
        if value is not None:
            options[ollama_key] = value

    max_tokens = body.get("max_completion_tokens", body.get("max_tokens"))
    if max_tokens is not None:
        options["num_predict"] = max_tokens

    stop = body.get("stop")
    if isinstance(stop, str):
        options["stop"] = [stop]
    elif isinstance(stop, list) and stop:
        options["stop"] = stop

    # Ollama-native `options` passthrough wins over mapped OpenAI params.
    src_options = body.get("options")
    if isinstance(src_options, dict):
        options.update(src_options)
    if "num_ctx" not in options:
        options["num_ctx"] = settings.llm_num_ctx
    ollama_body["options"] = options

    return ollama_body


def _normalize_tool_calls(raw_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Ollama-native tool_calls to OpenAI format.

    Ollama may return ``arguments`` as a parsed dict and may place ``index``
    inside ``function``.  OpenAI expects ``arguments`` as a JSON string with
    ``id``, ``type``, and ``index`` at the top level.
    """
    result: list[dict[str, Any]] = []
    for i, tc in enumerate(raw_calls):
        func = tc.get("function", {})
        args = func.get("arguments", {})
        if isinstance(args, dict):
            args = json.dumps(args)
        idx = func.get("index", i)
        call_id = tc.get("id") or f"call_{uuid.uuid4().hex[:12]}"
        result.append({
            "id": call_id,
            "type": "function",
            "index": idx,
            "function": {
                "name": func.get("name", ""),
                "arguments": args,
            },
        })
    return result


def _map_finish_reason(done_reason: str | None) -> str:
    if done_reason == "tool_calls":
        return "tool_calls"
    if done_reason == "length":
        return "length"
    return "stop"


def ollama_to_openai_response(data: dict[str, Any], *, strip_think: bool = False) -> dict[str, Any]:
    """Translate a non-streaming Ollama response to OpenAI format."""
    message = data.get("message", {})
    content = message.get("content", "")
    if strip_think and isinstance(content, str):
        content = strip_think_tags(content)

    openai_message: dict[str, Any] = {
        "role": message.get("role", "assistant"),
        "content": content,
    }

    raw_tool_calls = message.get("tool_calls")
    if raw_tool_calls:
        openai_message["tool_calls"] = _normalize_tool_calls(raw_tool_calls)

    prompt_tokens = data.get("prompt_eval_count", 0) or 0
    completion_tokens = data.get("eval_count", 0) or 0

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": data.get("model", ""),
        "choices": [
            {
                "index": 0,
                "message": openai_message,
                "finish_reason": _map_finish_reason(data.get("done_reason")),
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


async def open_stream(
    client: httpx.AsyncClient,
    *,
    url: str,
    body: dict[str, Any],
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    """Open a streaming POST without consuming the body.

    Lets the caller inspect the upstream status *before* committing a
    response status to the client.  The caller must either consume the
    response via a relay generator or ``aclose()`` it.
    """
    request = client.build_request("POST", url, json=body, headers=headers)
    return await client.send(request, stream=True)


async def ollama_stream_to_sse(
    response: httpx.Response,
    *,
    model: str,
    strip_think: bool = False,
    request_id: str | None = None,
    start: float | None = None,
) -> AsyncIterator[bytes]:
    """Relay an open Ollama native ``/api/chat`` stream as OpenAI SSE chunks.

    Ollama sends newline-delimited JSON; this generator re-wraps each line
    as an SSE ``data:`` event in OpenAI chat-completion-chunk format and
    closes the upstream response when done.
    """
    request_id = request_id or str(uuid.uuid4())
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    first_chunk = True
    strip_state: dict[str, Any] = {"in_think": False, "partial": ""}

    start = start if start is not None else time.monotonic()
    usage: dict[str, Any] | None = None

    try:
        try:
            async for raw_line in response.aiter_lines():
                if not raw_line.strip():
                    continue

                try:
                    chunk = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue

                msg = chunk.get("message", {})
                done = chunk.get("done", False)

                # Ollama may send tool_calls in any chunk (typically
                # a non-final one with done=False).  Emit them as an
                # OpenAI tool_calls delta whenever they appear.
                tool_calls = msg.get("tool_calls")
                if tool_calls:
                    tc_delta = _normalize_tool_calls(tool_calls)
                    tc_chunk = {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"tool_calls": tc_delta},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(tc_chunk)}\n\n".encode()

                if done:
                    prompt_tokens = chunk.get("prompt_eval_count", 0) or 0
                    completion_tokens = chunk.get("eval_count", 0) or 0
                    usage = {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    }

                    finish_chunk = {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": _map_finish_reason(
                                    chunk.get("done_reason")
                                ),
                            }
                        ],
                        "usage": usage,
                    }
                    yield f"data: {json.dumps(finish_chunk)}\n\n".encode()
                    yield b"data: [DONE]\n\n"
                    break

                if not tool_calls:
                    delta: dict[str, Any] = {}
                    if first_chunk:
                        delta["role"] = "assistant"
                        first_chunk = False

                    content = msg.get("content")
                    if strip_think and isinstance(content, str):
                        content = strip_think_tags_stream_chunk(content, strip_state)
                    if content:
                        delta["content"] = content

                    if delta:
                        sse_chunk = {
                            "id": chat_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": delta,
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(sse_chunk)}\n\n".encode()
        finally:
            await response.aclose()

        latency_ms = (time.monotonic() - start) * 1000
        log_response(
            logger,
            request_id=request_id,
            method="POST",
            path="/api/chat",
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
            path="/api/chat",
            error=str(exc),
            latency_ms=latency_ms,
        )
        raise
