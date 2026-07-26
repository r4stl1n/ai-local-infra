from __future__ import annotations

import asyncio
import base64
import gc
import io
import logging
import os
import sys
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from diffusers import StableDiffusionXLPipeline
from huggingface_hub.constants import HF_HUB_CACHE
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, Field

try:
    from compel import Compel, ReturnedEmbeddingsType

    _COMPEL_IMPORT_ERROR: str | None = None
except Exception as exc:  # optional dependency; degrade to plain (truncating) encoding
    Compel = None  # type: ignore[assignment,misc]
    ReturnedEmbeddingsType = None  # type: ignore[assignment,misc]
    _COMPEL_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

IMAGEGEN_MODEL = os.getenv("IMAGEGEN_MODEL", "RunDiffusion/Juggernaut-XL-v9")
# Defaults suit full SDXL checkpoints; turbo/lightning distillates
# (e.g. stabilityai/sdxl-turbo) want ~4 steps and guidance 0.0.
IMAGEGEN_STEPS = int(os.getenv("IMAGEGEN_STEPS", "30"))
IMAGEGEN_GUIDANCE = float(os.getenv("IMAGEGEN_GUIDANCE", "5.0"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


def _is_truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in ("1", "true", "yes", "on")


# When set, never touch the network: a cached model loads, a missing one errors
# immediately instead of hanging on an unreachable HuggingFace.
IMAGEGEN_OFFLINE = _is_truthy(os.getenv("IMAGEGEN_OFFLINE"))
_ENV_KEYS = (
    "IMAGEGEN_MODEL",
    "IMAGEGEN_STEPS",
    "IMAGEGEN_GUIDANCE",
    "IMAGEGEN_OFFLINE",
    "HF_HOME",
    "HF_HUB_OFFLINE",
    "LOG_LEVEL",
)
_SENSITIVE_ENV_MARKERS = ("PASSWORD", "TOKEN", "KEY", "SECRET")

logger = logging.getLogger("imagegen")
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)


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


def _cached_snapshot(model_name: str) -> str | None:
    """Path of the locally cached HF snapshot for a repo, or None if absent."""
    repo_dir = Path(HF_HUB_CACHE) / ("models--" + model_name.replace("/", "--"))
    ref = repo_dir / "refs" / "main"
    if not ref.is_file():
        return None
    snapshot = repo_dir / "snapshots" / ref.read_text().strip()
    return str(snapshot) if snapshot.is_dir() else None


def _installed_models() -> list[str]:
    """Repo ids of image models present in the local HF cache (with a snapshot)."""
    root = Path(HF_HUB_CACHE)
    if not root.is_dir():
        return []
    out: list[str] = []
    for entry in root.glob("models--*"):
        repo = entry.name[len("models--"):].replace("--", "/")
        if _cached_snapshot(repo):
            out.append(repo)
    return sorted(out)


def _load_pipeline_attempt(name_or_path: str, local_files_only: bool) -> StableDiffusionXLPipeline:
    try:
        return StableDiffusionXLPipeline.from_pretrained(
            name_or_path,
            torch_dtype=torch.float16,
            variant="fp16",
            local_files_only=local_files_only,
        )
    except (ValueError, OSError):
        # Not every checkpoint publishes fp16 variant files.
        logger.info("No fp16 variant for %s; loading default weights", name_or_path)
        return StableDiffusionXLPipeline.from_pretrained(
            name_or_path,
            torch_dtype=torch.float16,
            local_files_only=local_files_only,
        )


def _load_pipeline(model_name: str) -> StableDiffusionXLPipeline:
    logger.info("Loading image generation pipeline: %s", model_name)
    # A model that's present locally is ALWAYS loaded with the network disabled, so
    # a stray HEAD/etag check can never stall startup — and we never silently fall
    # back to a (hang-prone) download. Only a genuinely-missing model reaches out,
    # and only when offline mode isn't forced.
    cached = _cached_snapshot(model_name)
    if cached is not None:
        logger.info("Found in local cache; loading offline from %s", cached)
        pipe = _load_pipeline_attempt(cached, local_files_only=True)
    elif IMAGEGEN_OFFLINE:
        # Our resolver missed it but offline is forced — let HF's own cache lookup
        # try (network off): it either loads or raises at once, never hangs.
        logger.info("Not resolved locally; trying HF offline cache for %s", model_name)
        pipe = _load_pipeline_attempt(model_name, local_files_only=True)
    else:
        logger.info("Not cached; downloading from HuggingFace: %s", model_name)
        pipe = _load_pipeline_attempt(model_name, local_files_only=False)
    pipe.enable_model_cpu_offload()
    pipe.enable_attention_slicing()
    pipe.vae.enable_slicing()
    logger.info("Image generation pipeline loaded (model CPU offload + attention slicing)")
    return pipe


def _build_compel(pipe: StableDiffusionXLPipeline):
    """Build a Compel encoder for SDXL that accepts prompts longer than 77 tokens.

    `truncate_long_prompts=False` makes Compel split a long prompt into 75-token
    chunks and concatenate their CLIP embeddings (the "segment combination" the
    bare pipeline lacks), instead of hard-truncating at 77. Returns None if Compel
    is unavailable, so the service still runs with plain (truncating) encoding.
    """
    if Compel is None:
        logger.warning(
            "compel unavailable (%s); prompts over 77 tokens will be truncated",
            _COMPEL_IMPORT_ERROR,
        )
        return None
    try:
        # With enable_model_cpu_offload() the text encoders report device=cpu while
        # offloaded, so Compel would build its token-index tensors on CPU while the
        # accelerate hook runs the weights on cuda -> "tensors on different devices".
        # Pin Compel to the GPU so index and weights meet on the same device.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compel = Compel(
            tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
            text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True],
            truncate_long_prompts=False,
            device=device,
        )
        logger.info("Compel long-prompt encoder ready on %s (77-token limit lifted)", device)
        return compel
    except Exception as exc:
        logger.warning(
            "Failed to initialise compel (%s: %s); prompts over 77 tokens will be truncated",
            type(exc).__name__,
            exc,
        )
        return None


def _free_pipeline(pipe) -> None:
    """Drop a pipeline and actually reclaim its VRAM.

    enable_model_cpu_offload() installs accelerate hooks that pin the model's
    weights to the GPU execution device, so a plain `del` leaves that memory held
    and swapping models slowly exhausts VRAM (→ OOM → container restart, which
    reverts to the env-configured model). Strip the hooks and move everything back
    to CPU before releasing it.
    """
    if pipe is None:
        return
    try:
        from accelerate.hooks import remove_hook_from_module

        for name in ("unet", "vae", "text_encoder", "text_encoder_2", "image_encoder"):
            component = getattr(pipe, name, None)
            if component is not None:
                remove_hook_from_module(component, recurse=True)
        pipe.to("cpu")
    except Exception as exc:  # best-effort — never let cleanup crash a swap
        logger.warning("VRAM cleanup hit %s: %s", type(exc).__name__, exc)
    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _swap_pipeline(app: FastAPI, model_name: str) -> None:
    """Unload the current model and load `model_name` in its place. Holds the lock
    so it can never race an in-flight generation."""
    with app.state.lock:
        old = app.state.pipeline
        app.state.pipeline = None
        app.state.compel = None
        _free_pipeline(old)
        pipe = _load_pipeline(model_name)
        app.state.compel = _build_compel(pipe)
        app.state.pipeline = pipe
        app.state.model_name = model_name


def _unload_pipeline(app: FastAPI) -> None:
    """Unload the current model and free its VRAM, leaving nothing loaded."""
    with app.state.lock:
        old = app.state.pipeline
        app.state.pipeline = None
        app.state.compel = None
        app.state.model_name = None
        _free_pipeline(old)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    _log_startup_env("imagegen", _ENV_KEYS)
    # Serializes generation against model swaps (both run in worker threads).
    app.state.lock = threading.Lock()
    app.state.model_name = IMAGEGEN_MODEL
    app.state.pipeline = _load_pipeline(IMAGEGEN_MODEL)
    app.state.compel = _build_compel(app.state.pipeline)
    yield


app = FastAPI(title="Image Generation", version="0.1.0", lifespan=lifespan)


MAX_IMAGE_DIMENSION = 1024
MIN_IMAGE_DIMENSION = 64


def _parse_size(size: str) -> tuple[int, int]:
    """Parse a 'WxH' size string, clamped to MIN..MAX_IMAGE_DIMENSION."""
    try:
        w, h = size.lower().split("x")
        w, h = int(w), int(h)
    except (ValueError, AttributeError):
        return 512, 512
    w = max(MIN_IMAGE_DIMENSION, min(w, MAX_IMAGE_DIMENSION))
    h = max(MIN_IMAGE_DIMENSION, min(h, MAX_IMAGE_DIMENSION))
    return w, h


def _generate(
    app: FastAPI,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
) -> Image.Image:
    # Held for the whole render so a model swap can't pull the pipeline out from
    # under us mid-generation.
    with app.state.lock:
        pipe = app.state.pipeline
        compel = app.state.compel
        if pipe is None:
            raise RuntimeError("no image model is loaded")
        return _render(pipe, compel, prompt, negative_prompt, width, height)


def _pad_to_same_length(a: "torch.Tensor", b: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
    """Pad two [batch, seq, dim] conditioning tensors to equal sequence length.

    With truncate_long_prompts=False, positive and negative prompts can chunk into
    different multiples of 77 tokens; SDXL needs both the same length. Pad the
    shorter one with zeros (benign trailing padding, same as unused CLIP padding).
    """
    la, lb = a.shape[1], b.shape[1]
    if la == lb:
        return a, b
    target = max(la, lb)

    def pad(t: "torch.Tensor") -> "torch.Tensor":
        if t.shape[1] == target:
            return t
        filler = torch.zeros(
            t.shape[0], target - t.shape[1], t.shape[2], device=t.device, dtype=t.dtype
        )
        return torch.cat([t, filler], dim=1)

    return pad(a), pad(b)


def _render(
    pipe: StableDiffusionXLPipeline,
    compel,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
) -> Image.Image:
    base_kwargs = dict(
        num_inference_steps=IMAGEGEN_STEPS,
        guidance_scale=IMAGEGEN_GUIDANCE,
        width=width,
        height=height,
    )
    negative = (negative_prompt or "").strip()

    # Preferred path: Compel encodes both prompts (chunking anything over 77
    # tokens) and returns SDXL's dual embeddings + pooled vectors. Positive and
    # negative may end up different lengths, so pad them to match before use.
    if compel is not None:
        try:
            pos_embeds, pos_pooled = compel(prompt)
            neg_embeds, neg_pooled = compel(negative)
            # compel's own pad_conditioning_tensors_to_same_length is broken for the
            # multi-encoder (SDXL) provider in current versions, so pad here instead.
            pos_embeds, neg_embeds = _pad_to_same_length(pos_embeds, neg_embeds)
            result = pipe(
                prompt_embeds=pos_embeds,
                pooled_prompt_embeds=pos_pooled,
                negative_prompt_embeds=neg_embeds,
                negative_pooled_prompt_embeds=neg_pooled,
                **base_kwargs,
            )
            return result.images[0]
        except Exception as exc:
            logger.warning(
                "Long-prompt encoding failed (%s: %s); falling back to plain encoding",
                type(exc).__name__,
                exc,
            )

    # Fallback: plain diffusers encoding. Supports negative_prompt natively but
    # truncates either prompt at 77 tokens.
    result = pipe(
        prompt=prompt,
        negative_prompt=negative or None,
        **base_kwargs,
    )
    return result.images[0]


def _image_to_b64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class ImageRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    model: str = ""
    n: int = Field(default=1, ge=1, le=4)
    size: str = "512x512"
    response_format: str = "b64_json"


@app.post("/v1/images/generations")
async def create_image(body: ImageRequest) -> JSONResponse:
    if app.state.pipeline is None:
        return JSONResponse(
            status_code=503,
            content={"error": {"message": "no image model is loaded", "type": "api_error"}},
        )

    if not body.prompt.strip():
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "prompt must not be empty", "type": "invalid_request_error"}},
        )

    width, height = _parse_size(body.size)
    logger.info(
        "Generating %d image(s): %dx%d prompt=%r negative=%r",
        body.n,
        width,
        height,
        body.prompt[:120],
        body.negative_prompt[:120],
    )
    t0 = time.monotonic()

    data = []
    for i in range(body.n):
        image = await asyncio.to_thread(
            _generate, app, body.prompt, body.negative_prompt, width, height
        )
        b64 = _image_to_b64(image)
        data.append({"b64_json": b64})
        logger.info("Image %d/%d generated", i + 1, body.n)

    elapsed = time.monotonic() - t0
    logger.info("Generation complete in %.1fs", elapsed)

    return JSONResponse(content={
        "created": int(time.time()),
        "data": data,
    })


class LoadModelRequest(BaseModel):
    model: str


@app.get("/v1/images/models")
async def list_image_models() -> JSONResponse:
    """List image models present in the local cache and which one is loaded."""
    current = app.state.model_name
    ids = set(_installed_models())
    if current:
        ids.add(current)  # the loaded model is available even if the scan missed it
    return JSONResponse(content={
        "current": current,
        "data": [{"id": m, "loaded": m == current} for m in sorted(ids)],
    })


@app.post("/v1/images/models/load")
async def load_image_model(body: LoadModelRequest) -> JSONResponse:
    """Swap the active image model. Loading is slow (weights + VRAM); the request
    returns once the new model is ready."""
    model = body.model.strip()
    if not model:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "model is required", "type": "invalid_request_error"}},
        )
    if IMAGEGEN_OFFLINE and _cached_snapshot(model) is None:
        return JSONResponse(
            status_code=400,
            content={"error": {
                "message": f"model {model!r} is not in the local cache and IMAGEGEN_OFFLINE is set",
                "type": "invalid_request_error",
            }},
        )
    logger.info("Loading image model: %s", model)
    try:
        await asyncio.to_thread(_swap_pipeline, app, model)
    except Exception as exc:
        logger.exception("Failed to load image model %s", model)
        return JSONResponse(
            status_code=502,
            content={"error": {"message": f"failed to load {model!r}: {exc}", "type": "api_error"}},
        )
    return JSONResponse(content={"current": app.state.model_name})


@app.post("/v1/images/models/unload")
async def unload_image_model() -> JSONResponse:
    """Unload the active model and free its VRAM. Generation returns 503 until a
    model is loaded again."""
    await asyncio.to_thread(_unload_pipeline, app)
    logger.info("Image model unloaded")
    return JSONResponse(content={"current": None})


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
