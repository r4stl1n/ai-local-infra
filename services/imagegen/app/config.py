"""Environment-derived settings shared by the server and the prefetch script."""

from __future__ import annotations

import os


def _is_truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in ("1", "true", "yes", "on")


def _optional(name: str, cast):
    """Env value cast to `cast`, or None when unset/empty (= use the pipeline default)."""
    raw = (os.getenv(name) or "").strip()
    return cast(raw) if raw else None


IMAGEGEN_MODEL = os.getenv("IMAGEGEN_MODEL", "krea/Krea-2-Turbo")
# Community pipeline that runs Krea 2 with optional reference-image conditioning;
# without references it is a plain Krea 2 text-to-image sampler.
IMAGEGEN_PIPELINE = os.getenv("IMAGEGEN_PIPELINE", "ostris/Krea2OstrisEdit")
# Unset = the pipeline's per-checkpoint defaults (Turbo: 8 / 0.0, Raw: 28 / 4.5),
# so swapping between the two at runtime needs no reconfiguration.
IMAGEGEN_STEPS: int | None = _optional("IMAGEGEN_STEPS", int)
IMAGEGEN_GUIDANCE: float | None = _optional("IMAGEGEN_GUIDANCE", float)
# Use ComfyUI's pre-scaled fp8 transformer (~13 GB instead of ~26 GB bf16) so
# Krea 2 fits a 24 GB GPU. Applies to the models listed in FP8_TRANSFORMERS;
# anything else loads its own bf16 transformer.
IMAGEGEN_FP8 = _is_truthy(os.getenv("IMAGEGEN_FP8", "1"))
FP8_TRANSFORMER_REPO = "Comfy-Org/Krea-2"
FP8_TRANSFORMERS = {
    "krea/Krea-2-Turbo": "diffusion_models/krea2_turbo_fp8_scaled.safetensors",
    "krea/Krea-2-Raw": "diffusion_models/krea2_raw_fp8_scaled.safetensors",
}
# LoRA used for reference_mode=style ("repo" or "repo:weight_file"); empty disables it.
IMAGEGEN_STYLE_LORA = os.getenv(
    "IMAGEGEN_STYLE_LORA", "ostris/krea2_turbo_style_reference:krea2_style_reference.safetensors"
).strip()
# When set, never touch the network: a cached model loads, a missing one errors
# immediately instead of hanging on an unreachable HuggingFace.
IMAGEGEN_OFFLINE = _is_truthy(os.getenv("IMAGEGEN_OFFLINE"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Processor the pipeline uses to turn reference images into Qwen3-VL vision
# tokens (Krea 2 repos ship only a tokenizer). Must match the pipeline's
# `vl_processor_id`; only its small config/tokenizer files are needed.
VL_PROCESSOR_REPO = "Qwen/Qwen3-VL-4B-Instruct"
VL_PROCESSOR_PATTERNS = ["*.json", "*.jinja", "*.txt"]


def fp8_transformer_file(model_name: str) -> str | None:
    """Comfy fp8_scaled transformer file to use for `model_name`, or None (bf16)."""
    return FP8_TRANSFORMERS.get(model_name) if IMAGEGEN_FP8 else None


def style_lora() -> tuple[str, str | None] | None:
    """(repo, weight_name) of the style LoRA, or None when disabled."""
    if not IMAGEGEN_STYLE_LORA:
        return None
    repo, _, weight = IMAGEGEN_STYLE_LORA.partition(":")
    return repo, (weight or None)
