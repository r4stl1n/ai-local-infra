from __future__ import annotations

import asyncio
import base64
import binascii
import gc
import io
import logging
import math
import os
import sys
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import torch
from diffusers import DiffusionPipeline
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import HF_HUB_CACHE
from huggingface_hub.errors import LocalEntryNotFoundError
from PIL import Image, ImageOps
from pydantic import BaseModel, Field, ValidationError
from starlette.datastructures import UploadFile

from app.config import (
    FP8_TRANSFORMER_REPO,
    IMAGEGEN_GUIDANCE,
    IMAGEGEN_MODEL,
    IMAGEGEN_OFFLINE,
    IMAGEGEN_PIPELINE,
    IMAGEGEN_STEPS,
    LOG_LEVEL,
    VL_PROCESSOR_REPO,
    fp8_transformer_file,
    style_lora,
)
from app.fp8 import load_scaled_fp8_transformer

_ENV_KEYS = (
    "IMAGEGEN_MODEL",
    "IMAGEGEN_PIPELINE",
    "IMAGEGEN_STEPS",
    "IMAGEGEN_GUIDANCE",
    "IMAGEGEN_STYLE_LORA",
    "IMAGEGEN_FP8",
    "IMAGEGEN_OFFLINE",
    "HF_HOME",
    "HF_HUB_OFFLINE",
    "HF_TOKEN",
    "LOG_LEVEL",
)
_SENSITIVE_ENV_MARKERS = ("PASSWORD", "TOKEN", "KEY", "SECRET")
STYLE_ADAPTER = "style"

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


def _local_or_repo(repo: str) -> str:
    """The cached snapshot path when present (loads with no network), else the repo id."""
    return _cached_snapshot(repo) or repo


def _installed_models() -> list[str]:
    """Repo ids of diffusers pipelines present in the local HF cache. Auxiliary repos
    (community pipeline code, LoRAs, the VL processor) have no model_index.json."""
    root = Path(HF_HUB_CACHE)
    if not root.is_dir():
        return []
    out: list[str] = []
    for entry in root.glob("models--*"):
        repo = entry.name[len("models--"):].replace("--", "/")
        snapshot = _cached_snapshot(repo)
        if snapshot and (Path(snapshot) / "model_index.json").is_file():
            out.append(repo)
    return sorted(out)


def _load_style_lora(pipe: DiffusionPipeline) -> bool:
    """Load the style-reference LoRA as a disabled adapter; requests enable it per
    render. Failure only disables reference_mode=style."""
    lora = style_lora()
    if lora is None:
        return False
    repo, weight = lora
    try:
        pipe.load_lora_weights(_local_or_repo(repo), weight_name=weight, adapter_name=STYLE_ADAPTER)
        pipe.transformer.disable_adapters()
        # PEFT creates adapters in fp32; the LoRA ships in bf16, so this is lossless.
        for name, param in pipe.transformer.named_parameters():
            if "lora_" in name:
                param.data = param.data.to(torch.bfloat16)
    except Exception as exc:
        logger.warning("Style LoRA %s unavailable (%s: %s); reference_mode=style disabled",
                       repo, type(exc).__name__, exc)
        return False
    logger.info("Style LoRA loaded: %s", repo)
    return True


def _prepare_reference_processor(pipe: DiffusionPipeline) -> None:
    """Point the pipeline's lazily-built Qwen3-VL processor at the local cache and
    build it now, so the first reference request never reaches the network."""
    repo = getattr(pipe, "vl_processor_id", VL_PROCESSOR_REPO)
    pipe.vl_processor_id = _local_or_repo(repo)
    try:
        pipe.vl_processor
    except Exception as exc:
        logger.warning(
            "Reference-image processor %s unavailable (%s: %s); /v1/images/edits will fail "
            "until it is fetched (./infra.sh pull-models)",
            repo, type(exc).__name__, exc,
        )


def _fp8_transformer_path(filename: str) -> str:
    """Local path of a Comfy fp8 transformer file: the cache when present (no
    network), else a download unless offline mode is forced."""
    try:
        return hf_hub_download(FP8_TRANSFORMER_REPO, filename, local_files_only=True)
    except LocalEntryNotFoundError:
        if IMAGEGEN_OFFLINE:
            raise
    logger.info("Downloading fp8 transformer %s/%s", FP8_TRANSFORMER_REPO, filename)
    return hf_hub_download(FP8_TRANSFORMER_REPO, filename)


def _install_fp8_transformer(pipe: DiffusionPipeline, source: str, local_only: bool, filename: str) -> None:
    """Build the pipeline's transformer from Comfy's pre-scaled fp8 file."""
    # The transformer class lives in the community pipeline module.
    transformer_cls = sys.modules[type(pipe).__module__].Krea2Transformer2DModel
    config = transformer_cls.load_config(source, subfolder="transformer", local_files_only=local_only)
    path = _fp8_transformer_path(filename)
    logger.info("Loading fp8 transformer from %s", path)
    pipe.register_modules(transformer=load_scaled_fp8_transformer(transformer_cls, config, path))


def _load_pipeline(model_name: str) -> tuple[DiffusionPipeline, bool]:
    """Load `model_name` through the reference-capable community pipeline.
    Returns (pipeline, style_lora_loaded)."""
    logger.info("Loading image generation pipeline: %s (pipeline %s)", model_name, IMAGEGEN_PIPELINE)
    # A model that's present locally is ALWAYS loaded with the network disabled, so
    # a stray HEAD/etag check can never stall startup — and we never silently fall
    # back to a (hang-prone) download. Only a genuinely-missing model reaches out,
    # and only when offline mode isn't forced.
    cached = _cached_snapshot(model_name)
    if cached is not None:
        logger.info("Found in local cache; loading offline from %s", cached)
        source, local_only = cached, True
    elif IMAGEGEN_OFFLINE:
        # Our resolver missed it but offline is forced — let HF's own cache lookup
        # try (network off): it either loads or raises at once, never hangs.
        logger.info("Not resolved locally; trying HF offline cache for %s", model_name)
        source, local_only = model_name, True
    else:
        logger.info("Not cached; downloading from HuggingFace: %s", model_name)
        source, local_only = model_name, False
    fp8_file = fp8_transformer_file(model_name)
    # With an fp8 file the repo's own bf16 transformer is skipped entirely.
    components = {"transformer": None} if fp8_file else {}
    pipe = DiffusionPipeline.from_pretrained(
        source,
        custom_pipeline=_local_or_repo(IMAGEGEN_PIPELINE),
        torch_dtype=torch.bfloat16,
        local_files_only=local_only,
        **components,
    )
    if fp8_file:
        _install_fp8_transformer(pipe, source, local_only, fp8_file)
    has_style = _load_style_lora(pipe)
    _prepare_reference_processor(pipe)
    # Transformer (~13 GB in fp8) and text encoder (~9 GB) don't fit together on
    # a 24 GB GPU: offload keeps only the active component there.
    pipe.enable_model_cpu_offload()
    logger.info("Image generation pipeline loaded (model CPU offload)")
    return pipe, has_style


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
        pipe.remove_all_hooks()
        pipe.to("cpu")
    except Exception as exc:  # best-effort — never let cleanup crash a swap
        logger.warning("VRAM cleanup hit %s: %s", type(exc).__name__, exc)
    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _install(app: FastAPI, model_name: str | None, pipe, has_style: bool) -> None:
    app.state.pipeline = pipe
    app.state.has_style = has_style
    app.state.model_name = model_name


def _swap_pipeline(app: FastAPI, model_name: str) -> None:
    """Unload the current model and load `model_name` in its place. Holds the lock
    so it can never race an in-flight generation."""
    with app.state.lock:
        old = app.state.pipeline
        _install(app, None, None, False)
        _free_pipeline(old)
        pipe, has_style = _load_pipeline(model_name)
        _install(app, model_name, pipe, has_style)


def _unload_pipeline(app: FastAPI) -> None:
    """Unload the current model and free its VRAM, leaving nothing loaded."""
    with app.state.lock:
        old = app.state.pipeline
        _install(app, None, None, False)
        _free_pipeline(old)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    _log_startup_env("imagegen", _ENV_KEYS)
    # Serializes generation against model swaps (both run in worker threads).
    app.state.lock = threading.Lock()
    pipe, has_style = _load_pipeline(IMAGEGEN_MODEL)
    _install(app, IMAGEGEN_MODEL, pipe, has_style)
    yield


app = FastAPI(title="Image Generation", version="0.2.0", lifespan=lifespan)


def _error(status: int, message: str, err_type: str = "invalid_request_error",
           param: str | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": err_type, "param": param, "code": None}},
    )


def _validation_message(exc: ValidationError | RequestValidationError) -> tuple[str, str | None]:
    first = exc.errors()[0] if exc.errors() else {}
    loc = ".".join(str(part) for part in first.get("loc", ()) if part != "body")
    message = first.get("msg", "invalid request")
    return (f"Invalid request: {loc}: {message}" if loc else f"Invalid request: {message}"), (loc or None)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    message, param = _validation_message(exc)
    return _error(400, message, param=param)


MAX_IMAGE_DIMENSION = 1536
MIN_IMAGE_DIMENSION = 256
# Krea 2 latents are packed 8x (VAE) * 2x (patch): sizes must be multiples of 16.
SIZE_MULTIPLE = 16
DEFAULT_SIZE = (1024, 1024)
# The pipeline was trained with 1-2 reference images.
MAX_REFERENCE_IMAGES = 2
MAX_REFERENCE_BYTES = 20 * 1024 * 1024


def _clamp_dimension(value: float) -> int:
    snapped = int(round(value / SIZE_MULTIPLE)) * SIZE_MULTIPLE
    return max(MIN_IMAGE_DIMENSION, min(snapped, MAX_IMAGE_DIMENSION))


def _parse_size(size: str, reference: Image.Image | None = None) -> tuple[int, int]:
    """Parse a 'WxH' size string, clamped to MIN..MAX_IMAGE_DIMENSION and snapped to
    multiples of 16. 'auto' (or anything unparseable) gives 1024x1024 — or, with a
    reference image, the reference's aspect ratio at the same ~1 MP area."""
    try:
        w, h = size.lower().split("x")
        return _clamp_dimension(int(w)), _clamp_dimension(int(h))
    except (ValueError, AttributeError):
        pass
    if reference is None:
        return DEFAULT_SIZE
    rw, rh = reference.size
    scale = math.sqrt(DEFAULT_SIZE[0] * DEFAULT_SIZE[1] / (rw * rh))
    return _clamp_dimension(rw * scale), _clamp_dimension(rh * scale)


def _set_style_adapter(pipe: DiffusionPipeline, enabled: bool) -> None:
    if enabled:
        pipe.transformer.enable_adapters()
        pipe.set_adapters(STYLE_ADAPTER)
    else:
        pipe.transformer.disable_adapters()


def _generate(
    app: FastAPI,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    references: list[Image.Image] | None = None,
    style: bool = False,
) -> Image.Image:
    # Held for the whole render so a model swap can't pull the pipeline out from
    # under us mid-generation.
    with app.state.lock:
        pipe = app.state.pipeline
        if pipe is None:
            raise RuntimeError("no image model is loaded")
        if style:
            _set_style_adapter(pipe, True)
        try:
            return _render(pipe, prompt, negative_prompt, width, height, references)
        finally:
            if style:
                _set_style_adapter(pipe, False)
            # Offload already moved the weights to CPU; hand the allocator's cached
            # blocks back too, so an idle imagegen leaves the VRAM to Ollama.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def _render(
    pipe: DiffusionPipeline,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    references: list[Image.Image] | None,
) -> Image.Image:
    kwargs: dict[str, Any] = dict(prompt=prompt, width=width, height=height)
    # Unset steps/guidance fall through to the pipeline's per-checkpoint defaults.
    if IMAGEGEN_STEPS is not None:
        kwargs["num_inference_steps"] = IMAGEGEN_STEPS
    if IMAGEGEN_GUIDANCE is not None:
        kwargs["guidance_scale"] = IMAGEGEN_GUIDANCE
    negative = (negative_prompt or "").strip()
    if negative:
        # Only used when guidance is on (Raw); the distilled Turbo runs without CFG.
        kwargs["negative_prompt"] = negative
    if references:
        kwargs["image"] = references
    return pipe(**kwargs).images[0]


def _image_to_b64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class ImageRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    model: str = ""
    n: int = Field(default=1, ge=1, le=4)
    size: str = "1024x1024"
    response_format: str = "b64_json"


class EditRequest(ImageRequest):
    size: str = "auto"
    # edit: generate with the references as context (subject/edit/composition).
    # style: render the prompt in the references' style (style-reference LoRA).
    reference_mode: Literal["edit", "style"] = "edit"


class _RequestError(Exception):
    def __init__(self, message: str, param: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.param = param


def _json_image_bytes(entry: Any, index: int) -> bytes:
    """Decode one JSON reference: a data: URI or bare base64 string, or an object
    {"image_url": ...} / {"image_url": {"url": ...}} / {"b64_json": ...}."""
    value = entry
    if isinstance(entry, dict):
        value = entry.get("image_url") or entry.get("b64_json")
        if isinstance(value, dict):
            value = value.get("url")
    param = f"images.{index}"
    if not isinstance(value, str) or not value:
        raise _RequestError(f"{param} must be a data: URI or base64 string", param)
    if value.startswith(("http://", "https://")):
        raise _RequestError(f"{param}: remote URLs are not fetched; send a data: URI or base64", param)
    if value.startswith("data:"):
        _, sep, value = value.partition(",")
        if not sep:
            raise _RequestError(f"{param}: malformed data: URI", param)
    if len(value) * 3 // 4 > MAX_REFERENCE_BYTES:
        raise _RequestError(f"{param} exceeds {MAX_REFERENCE_BYTES // (1024 * 1024)} MB", param)
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        raise _RequestError(f"{param} is not valid base64", param) from None


def _decode_reference(raw: bytes, index: int) -> Image.Image:
    param = f"image.{index}"
    if not raw:
        raise _RequestError(f"{param} is empty", param)
    if len(raw) > MAX_REFERENCE_BYTES:
        raise _RequestError(f"{param} exceeds {MAX_REFERENCE_BYTES // (1024 * 1024)} MB", param)
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except (OSError, Image.DecompressionBombError) as exc:
        raise _RequestError(f"{param} is not a readable image ({exc})", param) from None
    # Phone photos store rotation in EXIF; apply it so the model sees them upright.
    return ImageOps.exif_transpose(image).convert("RGB")


async def _read_edit_request(request: Request) -> tuple[EditRequest, list[bytes]]:
    """Accept OpenAI's multipart form (`image` / `image[]` files) or a JSON body
    with `images` (or `image`) as base64 / data: URIs."""
    content_type = request.headers.get("content-type", "")
    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        uploads = [*form.getlist("image"), *form.getlist("image[]")]
        if any(not isinstance(u, UploadFile) for u in uploads):
            raise _RequestError("image must be uploaded as a file", "image")
        raw = [await u.read() for u in uploads]
        has_mask = "mask" in form
        fields = {
            k: v for k, v in form.multi_items()
            if isinstance(v, str) and k not in ("image", "image[]", "mask")
        }
    else:
        try:
            payload = await request.json()
        except ValueError:
            raise _RequestError("body must be JSON or multipart/form-data") from None
        if not isinstance(payload, dict):
            raise _RequestError("body must be a JSON object")
        images = payload.pop("images", None)
        if images is None:
            images = payload.pop("image", None)
        if isinstance(images, (str, dict)):
            images = [images]
        if not isinstance(images, list):
            raise _RequestError("images is required", "images")
        raw = [_json_image_bytes(entry, i) for i, entry in enumerate(images)]
        has_mask = payload.pop("mask", None) is not None
        fields = payload

    if has_mask:
        raise _RequestError(
            "mask is not supported: Krea 2 conditions on whole reference images and does not inpaint",
            "mask",
        )
    if not raw:
        raise _RequestError("at least one reference image is required", "image")
    if len(raw) > MAX_REFERENCE_IMAGES:
        raise _RequestError(f"at most {MAX_REFERENCE_IMAGES} reference images are supported", "image")
    try:
        body = EditRequest.model_validate(fields)
    except ValidationError as exc:
        message, param = _validation_message(exc)
        raise _RequestError(message, param) from None
    return body, raw


async def _generate_response(
    body: ImageRequest,
    width: int,
    height: int,
    references: list[Image.Image] | None = None,
    style: bool = False,
) -> JSONResponse:
    if app.state.pipeline is None:
        return _error(503, "no image model is loaded", "api_error")
    if not body.prompt.strip():
        return _error(400, "prompt must not be empty", param="prompt")

    logger.info(
        "Generating %d image(s): %dx%d refs=%d%s prompt=%r negative=%r",
        body.n,
        width,
        height,
        len(references or ()),
        " (style)" if style else "",
        body.prompt[:120],
        body.negative_prompt[:120],
    )
    t0 = time.monotonic()

    data = []
    for i in range(body.n):
        image = await asyncio.to_thread(
            _generate, app, body.prompt, body.negative_prompt, width, height, references, style
        )
        data.append({"b64_json": _image_to_b64(image)})
        logger.info("Image %d/%d generated", i + 1, body.n)

    logger.info("Generation complete in %.1fs", time.monotonic() - t0)
    return JSONResponse(content={"created": int(time.time()), "data": data})


@app.post("/v1/images/generations")
async def create_image(body: ImageRequest) -> JSONResponse:
    width, height = _parse_size(body.size)
    return await _generate_response(body, width, height)


@app.post("/v1/images/edits")
async def edit_image(request: Request) -> JSONResponse:
    """Generate from a prompt plus 1-2 reference images (OpenAI images/edits shape)."""
    try:
        body, raw = await _read_edit_request(request)
        references = [_decode_reference(data, i) for i, data in enumerate(raw)]
    except _RequestError as exc:
        return _error(400, exc.message, param=exc.param)

    style = body.reference_mode == "style"
    if style and not app.state.has_style:
        return _error(
            400,
            "reference_mode=style needs the style LoRA, which is not loaded (see IMAGEGEN_STYLE_LORA)",
            param="reference_mode",
        )
    width, height = _parse_size(body.size, references[0])
    return await _generate_response(body, width, height, references, style)


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
        return _error(400, "model is required", param="model")
    if IMAGEGEN_OFFLINE and _cached_snapshot(model) is None:
        return _error(400, f"model {model!r} is not in the local cache and IMAGEGEN_OFFLINE is set",
                      param="model")
    logger.info("Loading image model: %s", model)
    try:
        await asyncio.to_thread(_swap_pipeline, app, model)
    except Exception as exc:
        logger.exception("Failed to load image model %s", model)
        return _error(502, f"failed to load {model!r}: {exc}", "api_error")
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
