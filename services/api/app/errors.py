"""OpenAI-style error payload helpers.

OpenAI clients expect errors shaped as
``{"error": {"message", "type", "param", "code"}}`` with a meaningful HTTP
status code; these helpers normalise upstream errors into that shape.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi.responses import JSONResponse


def _type_for_status(status_code: int) -> str:
    if status_code == 401:
        return "authentication_error"
    if status_code == 403:
        return "permission_error"
    if status_code == 404:
        return "not_found_error"
    if status_code == 429:
        return "rate_limit_error"
    if 400 <= status_code < 500:
        return "invalid_request_error"
    return "api_error"


def error_body(
    message: str,
    *,
    err_type: str = "api_error",
    param: str | None = None,
    code: str | None = None,
) -> dict[str, Any]:
    return {"error": {"message": message, "type": err_type, "param": param, "code": code}}


def error_response(
    message: str,
    status_code: int,
    *,
    param: str | None = None,
    code: str | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=error_body(message, err_type=_type_for_status(status_code), param=param, code=code),
    )


def extract_upstream_message(raw: Any) -> str:
    """Best-effort extraction of a human-readable message from an upstream error."""
    data = raw
    if isinstance(data, bytes):
        data = data.decode("utf-8", errors="replace")
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            return data
    if isinstance(data, dict):
        err = data.get("error")
        if isinstance(err, str):
            return err
        if isinstance(err, dict) and isinstance(err.get("message"), str):
            return err["message"]
        detail = data.get("detail")
        if isinstance(detail, str):
            return detail
    try:
        return json.dumps(data)
    except (TypeError, ValueError):
        return str(data)


def normalized_error_content(raw: Any, status_code: int) -> dict[str, Any]:
    """Return *raw* if already OpenAI-shaped, otherwise wrap it."""
    data = raw
    if isinstance(data, bytes):
        data = data.decode("utf-8", errors="replace")
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            data = None
    if (
        isinstance(data, dict)
        and isinstance(data.get("error"), dict)
        and isinstance(data["error"].get("message"), str)
    ):
        return data
    return error_body(extract_upstream_message(raw), err_type=_type_for_status(status_code))
