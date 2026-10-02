"""Download everything the imagegen service needs into HF_HOME, so the first (or
an offline) start doesn't stall on a silent multi-GB download.

    python3 -m app.prefetch
"""

from __future__ import annotations

import sys

from huggingface_hub import snapshot_download

from app.config import (
    IMAGEGEN_MODEL,
    QWEN_VL,
    VAE,
    VARIANTS,
    pipeline_source,
    reference_lora_sources,
    transformer_source,
)


def main() -> int:
    if IMAGEGEN_MODEL not in VARIANTS:
        print(f"Unknown IMAGEGEN_MODEL {IMAGEGEN_MODEL!r}; expected one of {sorted(VARIANTS)}", file=sys.stderr)
        return 1
    sources = [
        ("transformer", transformer_source(IMAGEGEN_MODEL)),
        ("text encoder + processor", QWEN_VL),
        ("VAE", VAE),
        ("pipeline", pipeline_source()),
        *((f"{mode} LoRA", source) for mode, source in reference_lora_sources().items()),
    ]
    for label, source in sources:
        print(f"Pulling {label}: {source.repo} {list(source.patterns)}", flush=True)
        snapshot_download(source.repo, allow_patterns=list(source.patterns))
    return 0


if __name__ == "__main__":
    sys.exit(main())
