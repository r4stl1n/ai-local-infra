"""Download everything the imagegen service needs into HF_HOME, so the first (or
an offline) start doesn't stall on a silent multi-GB download.

    python3 -m app.prefetch
"""

from __future__ import annotations

import json
import sys

from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.errors import GatedRepoError

from app.config import (
    FP8_TRANSFORMER_REPO,
    IMAGEGEN_MODEL,
    IMAGEGEN_PIPELINE,
    VL_PROCESSOR_PATTERNS,
    VL_PROCESSOR_REPO,
    fp8_transformer_file,
    style_lora,
)


def _model_patterns(repo: str, skip_transformer_weights: bool) -> list[str]:
    """Only the diffusers component folders listed in model_index.json — skips the
    redundant single-file checkpoint (e.g. turbo.safetensors) and sample images.
    With an fp8 transformer, only the transformer's config is needed."""
    with open(hf_hub_download(repo, "model_index.json")) as f:
        index = json.load(f)
    components = [k for k, v in index.items() if not k.startswith("_") and isinstance(v, list)]
    patterns = ["model_index.json"]
    for c in components:
        if c == "transformer" and skip_transformer_weights:
            patterns.append("transformer/config.json")
        else:
            patterns.append(f"{c}/*")
    return patterns


def main() -> int:
    try:
        fp8_file = fp8_transformer_file(IMAGEGEN_MODEL)
        print(f"Pulling image model: {IMAGEGEN_MODEL}", flush=True)
        snapshot_download(IMAGEGEN_MODEL, allow_patterns=_model_patterns(IMAGEGEN_MODEL, bool(fp8_file)))
    except GatedRepoError:
        print(
            f"{IMAGEGEN_MODEL} is gated: accept its license at "
            f"https://huggingface.co/{IMAGEGEN_MODEL} and set HF_TOKEN in .env",
            file=sys.stderr,
        )
        return 1

    if fp8_file:
        print(f"Pulling fp8 transformer: {FP8_TRANSFORMER_REPO}/{fp8_file}", flush=True)
        hf_hub_download(FP8_TRANSFORMER_REPO, fp8_file)

    print(f"Pulling pipeline: {IMAGEGEN_PIPELINE}", flush=True)
    snapshot_download(IMAGEGEN_PIPELINE, allow_patterns=["*.py"])

    print(f"Pulling reference-image processor: {VL_PROCESSOR_REPO}", flush=True)
    snapshot_download(VL_PROCESSOR_REPO, allow_patterns=VL_PROCESSOR_PATTERNS)

    lora = style_lora()
    if lora is not None:
        repo, weight = lora
        print(f"Pulling style LoRA: {repo}", flush=True)
        if weight:
            hf_hub_download(repo, weight)
        else:
            snapshot_download(repo, allow_patterns=["*.safetensors"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
