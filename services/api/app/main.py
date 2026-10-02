from __future__ import annotations

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

import hmac
import httpx
import os
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import settings
from app.errors import error_body, error_response, normalized_error_content
from app.logging import get_logger
from app.routes import chat, embeddings, imagegen, models, separation, stt, systemone, tts

logger = get_logger("backend")

_AUTH_EXEMPT_PATHS = frozenset({"/health", "/docs", "/openapi.json"})
_ENV_KEYS = (
    "OLLAMA_URL",
    "LLM_PROVIDER",
    "LLM_THINKING",
    "LLM_NUM_CTX",
    "LLM_URL",
    "LLM_API_TOKEN",
    "TTS_URL",
    "WHISPER_URL",
    "IMAGEGEN_URL",
    "API_KEY",
    "LOG_LEVEL",
)
_SENSITIVE_ENV_MARKERS = ("PASSWORD", "TOKEN", "KEY", "SECRET")


def _mask_env_value(key: str, value: str | None) -> str:
    if not value:
        return "<unset>"
    if any(marker in key.upper() for marker in _SENSITIVE_ENV_MARKERS):
        return "<set>"
    if len(value) > 120:
        return f"{value[:117]}..."
    return value


def _log_startup_env(service_name: str, keys: tuple[str, ...]) -> None:
    rendered = ", ".join(f"{key}={_mask_env_value(key, os.getenv(key))}" for key in keys)
    logger.info("%s startup env: %s", service_name, rendered)


class APIKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path in _AUTH_EXEMPT_PATHS:
            return await call_next(request)
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer ") or not hmac.compare_digest(auth[7:], settings.api_key):
            return error_response("Invalid or missing API key", 401, code="invalid_api_key")
        return await call_next(request)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    _log_startup_env("backend", _ENV_KEYS)
    logger.info(
        "Starting backend — LLM: %s  TTS: %s  STT: %s  ImageGen: %s  Separation: %s",
        settings.ollama_url,
        settings.tts_url,
        settings.whisper_url,
        settings.imagegen_url,
        settings.demucs_url,
    )
    app.state.http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(settings.backend_upstream_timeout_seconds),
        limits=httpx.Limits(max_connections=40, max_keepalive_connections=20),
    )
    yield
    await app.state.http_client.aclose()
    logger.info("Backend shut down")


app = FastAPI(
    title="AI Infra API",
    description="Unified API gateway for LLM, TTS, STT, and image generation services",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(APIKeyMiddleware)


@app.middleware("http")
async def inject_http_client(request: Request, call_next):
    request.state.http_client = request.app.state.http_client
    return await call_next(request)


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content=normalized_error_content(exc.detail, exc.status_code),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    first = exc.errors()[0] if exc.errors() else {}
    loc = ".".join(str(part) for part in first.get("loc", ()) if part != "body")
    message = first.get("msg", "invalid request")
    return JSONResponse(
        status_code=400,
        content=error_body(
            f"Invalid request: {loc}: {message}" if loc else f"Invalid request: {message}",
            err_type="invalid_request_error",
            param=loc or None,
        ),
    )


@app.exception_handler(httpx.HTTPError)
async def upstream_exception_handler(request: Request, exc: httpx.HTTPError) -> JSONResponse:
    logger.error("Upstream request failed: %s", exc)
    return JSONResponse(
        status_code=502,
        content=error_body(f"Upstream service error: {exc}", err_type="api_error"),
    )


app.include_router(chat.router)
app.include_router(embeddings.router)
app.include_router(systemone.router)
app.include_router(models.router)
app.include_router(tts.router)
app.include_router(stt.router)
app.include_router(imagegen.router)
app.include_router(separation.router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
