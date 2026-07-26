from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

from app.config import settings


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "data"):
            entry["data"] = record.data  # type: ignore[attr-defined]
        if record.exc_info and record.exc_info[1]:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
        logger.propagate = False
    return logger


def log_request(
    logger: logging.Logger,
    *,
    request_id: str,
    method: str,
    path: str,
    model: str | None = None,
    stream: bool = False,
    message_count: int | None = None,
) -> None:
    extra_data: dict[str, Any] = {
        "event": "request",
        "request_id": request_id,
        "method": method,
        "path": path,
        "model": model,
        "stream": stream,
    }
    if message_count is not None:
        extra_data["message_count"] = message_count
    record = logger.makeRecord(
        logger.name, logging.INFO, "", 0, "Incoming request", (), None
    )
    record.data = extra_data  # type: ignore[attr-defined]
    logger.handle(record)


def log_response(
    logger: logging.Logger,
    *,
    request_id: str,
    method: str,
    path: str,
    status_code: int,
    latency_ms: float,
    model: str | None = None,
    usage: dict[str, Any] | None = None,
    cost: dict[str, Any] | None = None,
) -> None:
    extra_data: dict[str, Any] = {
        "event": "response",
        "request_id": request_id,
        "method": method,
        "path": path,
        "status_code": status_code,
        "latency_ms": round(latency_ms, 2),
    }
    if model:
        extra_data["model"] = model
    if usage:
        extra_data["prompt_tokens"] = usage.get("prompt_tokens")
        extra_data["completion_tokens"] = usage.get("completion_tokens")
        extra_data["total_tokens"] = usage.get("total_tokens")
        extra_data["usage"] = usage
    if cost:
        extra_data["cost"] = cost
    record = logger.makeRecord(
        logger.name, logging.INFO, "", 0, "Upstream response", (), None
    )
    record.data = extra_data  # type: ignore[attr-defined]
    logger.handle(record)


def log_error(
    logger: logging.Logger,
    *,
    request_id: str,
    method: str,
    path: str,
    error: str,
    status_code: int | None = None,
    latency_ms: float | None = None,
) -> None:
    extra_data: dict[str, Any] = {
        "event": "error",
        "request_id": request_id,
        "method": method,
        "path": path,
        "error": error,
    }
    if status_code is not None:
        extra_data["status_code"] = status_code
    if latency_ms is not None:
        extra_data["latency_ms"] = round(latency_ms, 2)
    record = logger.makeRecord(
        logger.name, logging.ERROR, "", 0, "Request failed", (), None
    )
    record.data = extra_data  # type: ignore[attr-defined]
    logger.handle(record)
