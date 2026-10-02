"""Environment-derived settings shared by the server and the prefetch script."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _is_truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in ("1", "true", "yes", "on")


def _optional(name: str, cast):
    """Env value cast to `cast`, or None when unset/empty (= use the pipeline default)."""
    raw = (os.getenv(name) or "").strip()
    return cast(raw) if raw else None


@dataclass(frozen=True)
class Source:
    """Files to fetch from one HF repo. `required` must all exist locally for the
    cached copy to count as complete; `patterns` is what a download pulls."""

    repo: str
    patterns: tuple[str, ...]
    required: tuple[str, ...]


# Every Krea 2 component comes from an ungated repo, so no HF token is needed.
# Comfy-Org/Krea-2 redistributes the transformer (bf16 and pre-scaled fp8); the
# text encoder and VAE are the stock Qwen weights (byte-identical to the copies
# Comfy-Org ships), taken from Qwen's own repos in diffusers/transformers format.
TRANSFORMER_REPO = "Comfy-Org/Krea-2"
# Text encoder + tokenizer + the processor used to embed reference images.
QWEN_VL = Source(
    "Qwen/Qwen3-VL-4B-Instruct",
    patterns=("*.json", "*.jinja", "*.txt", "*.safetensors"),
    required=("config.json", "tokenizer.json", "preprocessor_config.json", "model.safetensors.index.json"),
)
VAE = Source(
    "Qwen/Qwen-Image",
    patterns=("vae/*",),
    required=("vae/config.json", "vae/diffusion_pytorch_model.safetensors"),
)
VAE_SUBFOLDER = "vae"


@dataclass(frozen=True)
class Variant:
    is_distilled: bool
    fp8_file: str
    bf16_file: str


VARIANTS = {
    # 8-step distillate: no CFG (guidance 0.0).
    "krea2-turbo": Variant(
        is_distilled=True,
        fp8_file="diffusion_models/krea2_turbo_fp8_scaled.safetensors",
        bf16_file="diffusion_models/krea2_turbo_bf16.safetensors",
    ),
    # Base model: 28 steps, guidance 4.5.
    "krea2-raw": Variant(
        is_distilled=False,
        fp8_file="diffusion_models/krea2_raw_fp8_scaled.safetensors",
        bf16_file="diffusion_models/krea2_raw_bf16.safetensors",
    ),
}

# Krea 2's resolution-aware exponential time shift (diffusers Krea2Pipeline docs);
# the Krea repos keep this config behind their gate, so it lives here.
SCHEDULER_CONFIG = {
    "num_train_timesteps": 1000,
    "use_dynamic_shifting": True,
    "time_shift_type": "exponential",
    "base_shift": 0.5,
    "max_shift": 1.15,
    "base_image_seq_len": 256,
    "max_image_seq_len": 6400,
}

IMAGEGEN_MODEL = os.getenv("IMAGEGEN_MODEL", "krea2-turbo")
# Community pipeline that runs Krea 2 with optional reference-image conditioning;
# without references it is a plain Krea 2 text-to-image sampler.
IMAGEGEN_PIPELINE = os.getenv("IMAGEGEN_PIPELINE", "ostris/Krea2OstrisEdit")
# Unset = the pipeline's per-checkpoint defaults (Turbo: 8 / 0.0, Raw: 28 / 4.5),
# so swapping between the two at runtime needs no reconfiguration.
IMAGEGEN_STEPS: int | None = _optional("IMAGEGEN_STEPS", int)
IMAGEGEN_GUIDANCE: float | None = _optional("IMAGEGEN_GUIDANCE", float)
# Pre-scaled fp8 transformer (~13 GB, fits a 24 GB GPU) instead of bf16 (~26 GB).
IMAGEGEN_FP8 = _is_truthy(os.getenv("IMAGEGEN_FP8", "1"))
# LoRA used for reference_mode=style ("repo" or "repo:weight_file"); empty disables it.
IMAGEGEN_STYLE_LORA = os.getenv(
    "IMAGEGEN_STYLE_LORA", "ostris/krea2_turbo_style_reference:krea2_style_reference.safetensors"
).strip()
# When set, never touch the network: cached files load, missing ones error
# immediately instead of hanging on an unreachable HuggingFace.
IMAGEGEN_OFFLINE = _is_truthy(os.getenv("IMAGEGEN_OFFLINE"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


def transformer_source(model_name: str) -> Source:
    """The Comfy-Org transformer file for a variant (KeyError if unknown)."""
    variant = VARIANTS[model_name]
    filename = variant.fp8_file if IMAGEGEN_FP8 else variant.bf16_file
    return Source(TRANSFORMER_REPO, patterns=(filename,), required=(filename,))


def pipeline_source() -> Source:
    return Source(IMAGEGEN_PIPELINE, patterns=("*.py",), required=("pipeline.py",))


def style_lora_source() -> Source | None:
    """Where the style LoRA lives, or None when disabled."""
    if not IMAGEGEN_STYLE_LORA:
        return None
    repo, _, weight = IMAGEGEN_STYLE_LORA.partition(":")
    if weight:
        return Source(repo, patterns=(weight,), required=(weight,))
    return Source(repo, patterns=("*.safetensors",), required=())
