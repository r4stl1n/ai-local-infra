from __future__ import annotations

from typing import Any

from app.config import settings

# (input_per_million, output_per_million) in USD.
# Tiers roughly mirror hosted-API pricing for equivalent model sizes
# so local inference can be compared apples-to-apples with cloud costs.
_COST_TIERS: dict[str, tuple[float, float]] = {
    "small": (0.10, 0.30),
    "medium": (0.50, 1.50),
    "large": (2.00, 6.00),
}

# Case-insensitive substring -> tier mapping.  Order matters: first match wins,
# so more-specific patterns (e.g. "70b") should come before generic family names.
_MODEL_TIER: list[tuple[str, str]] = [
    # Large
    ("70b", "large"),
    ("72b", "large"),
    ("65b", "large"),
    ("deepseek-v3", "large"),
    ("deepseek-r1", "large"),
    ("command-r-plus", "large"),
    ("qwen2.5-coder-32b", "large"),
    # Small
    ("1b", "small"),
    ("2b", "small"),
    ("3b", "small"),
    ("phi", "small"),
    ("gemma-2b", "small"),
    ("tinyllama", "small"),
    # Medium (broad family names — matched last)
    ("llama", "medium"),
    ("mistral", "medium"),
    ("qwen", "medium"),
    ("gemma", "medium"),
    ("codellama", "medium"),
    ("command-r", "medium"),
    ("deepseek", "medium"),
    ("yi", "medium"),
]


def _resolve_rates(model: str) -> tuple[float, float]:
    """Return (input_per_million, output_per_million) for *model*."""
    name = model.lower()
    for pattern, tier in _MODEL_TIER:
        if pattern in name:
            return _COST_TIERS[tier]
    return (settings.llm_default_cost_input, settings.llm_default_cost_output)


def estimate_cost(model: str | None, usage: dict[str, Any] | None) -> dict[str, Any] | None:
    """Estimate the USD cost of a request given token usage.

    Returns ``None`` when *usage* is missing or contains no token counts.
    """
    if not usage or not model:
        return None

    prompt_tokens = usage.get("prompt_tokens", 0) or 0
    completion_tokens = usage.get("completion_tokens", 0) or 0

    if prompt_tokens == 0 and completion_tokens == 0:
        return None

    input_rate, output_rate = _resolve_rates(model)

    input_cost = prompt_tokens * input_rate / 1_000_000
    output_cost = completion_tokens * output_rate / 1_000_000
    total_cost = input_cost + output_cost

    return {
        "input_cost": round(input_cost, 8),
        "output_cost": round(output_cost, 8),
        "total_cost": round(total_cost, 8),
        "cost_per_million_input": input_rate,
        "cost_per_million_output": output_rate,
    }
