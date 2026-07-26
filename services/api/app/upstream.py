"""Remote upstream URL normalisation.

``LLM_URL`` may be configured as a bare host, a versioned base (``.../v1``),
or a full chat-completions URL.  Resolve any of those into a base URL plus
candidate paths for a given OpenAI resource (``chat/completions``,
``models``, ...), most likely path first.
"""

from __future__ import annotations

_CHAT_SUFFIX = "/chat/completions"


def resolve_remote_target(upstream_url: str, resource: str) -> tuple[str, list[str]]:
    """Return ``(base_url, candidate_paths)`` for *resource* on *upstream_url*."""
    normalized = upstream_url.rstrip("/")

    if normalized.endswith(f"/v1{_CHAT_SUFFIX}"):
        return normalized[: -len(f"/v1{_CHAT_SUFFIX}")], [f"/v1/{resource}"]
    if normalized.endswith(_CHAT_SUFFIX):
        return normalized[: -len(_CHAT_SUFFIX)], [f"/{resource}"]

    if normalized.endswith("/v1") or normalized.endswith("/api/v1"):
        return normalized, [f"/{resource}"]

    # Some providers expect /v1/<resource> from a root base URL; others
    # expose /<resource> from a versioned base.
    return normalized, [f"/v1/{resource}", f"/{resource}"]
