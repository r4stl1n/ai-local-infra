from __future__ import annotations

import asyncio
import base64
import binascii
import functools
import gc
import importlib.util
import io
import json
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
from diffusers import AutoencoderKLQwenImage, DiffusionPipeline, FlowMatchEulerDiscreteScheduler
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from huggingface_hub import snapshot_download
from huggingface_hub.constants import HF_HUB_CACHE
from PIL import Image, ImageOps
from pydantic import BaseModel, Field, ValidationError
from starlette.datastructures import UploadFile
from transformers import AutoTokenizer, Qwen3VLModel

from app.config import (
    IMAGEGEN_GUIDANCE,
    IMAGEGEN_MODEL,
    IMAGEGEN_OFFLINE,
    IMAGEGEN_PIPELINE,
    IMAGEGEN_STEPS,
    LOG_LEVEL,
    QWEN_VL,
    SCHEDULER_CONFIG,
    TRANSFORMER_REPO,
    VAE,
    VAE_SUBFOLDER,
    VARIANTS,
    Source,
    pipeline_source,
    reference_lora_sources,
    transformer_source,
)
from app.fp8 import load_comfy_transformer

_ENV_KEYS = (
    "IMAGEGEN_MODEL",
    "IMAGEGEN_PIPELINE",
    "IMAGEGEN_STEPS",
    "IMAGEGEN_GUIDANCE",
    "IMAGEGEN_STYLE_LORA",
    "IMAGEGEN_EDIT_LORA",
    "IMAGEGEN_FP8",
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


def _cached_snapshot(repo: str) -> str | None:
    """Path of the locally cached HF snapshot for a repo, or None if absent."""
    repo_dir = Path(HF_HUB_CACHE) / ("models--" + repo.replace("/", "--"))
    ref = repo_dir / "refs" / "main"
    if not ref.is_file():
        return None
    snapshot = repo_dir / "snapshots" / ref.read_text().strip()
    return str(snapshot) if snapshot.is_dir() else None


def _missing_files(snapshot: Path, source: Source) -> list[str]:
    missing = [f for f in source.required if not (snapshot / f).is_file()]
    index = snapshot / "model.safetensors.index.json"
    if not missing and index.is_file():  # sharded weights: every shard must be there too
        shards = set(json.loads(index.read_text())["weight_map"].values())
        missing += [shard for shard in sorted(shards) if not (snapshot / shard).is_file()]
    return missing


def _snapshot(source: Source) -> Path:
    """Local snapshot directory holding `source`.

    A complete cached copy is ALWAYS used with the network untouched, so a stray
    HEAD/etag check can never stall startup. Only missing files are downloaded,
    and never when offline mode is forced (that errors at once instead of hanging).
    """
    cached = _cached_snapshot(source.repo)
    missing = _missing_files(Path(cached), source) if cached else list(source.required)
    if cached and not missing:
        return Path(cached)
    if IMAGEGEN_OFFLINE:
        raise FileNotFoundError(
            f"{source.repo} is missing {missing or 'its files'} locally and IMAGEGEN_OFFLINE is set "
            "(fetch with ./infra.sh pull-models)"
        )
    logger.info("Downloading %s %s", source.repo, list(source.patterns))
    return Path(snapshot_download(source.repo, allow_patterns=list(source.patterns)))


def _is_installed(model_name: str) -> bool:
    """Whether a variant's transformer file is already in the local cache."""
    cached = _cached_snapshot(TRANSFORMER_REPO)
    return bool(cached) and not _missing_files(Path(cached), transformer_source(model_name))


@functools.cache
def _community_classes():
    """(pipeline class, transformer class) from the community pipeline.py, imported
    straight from the local snapshot."""
    path = _snapshot(pipeline_source()) / "pipeline.py"
    spec = importlib.util.spec_from_file_location("krea2_community_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.Krea2OstrisEditPipeline, module.Krea2Transformer2DModel


def _load_reference_loras(pipe: DiffusionPipeline) -> frozenset[str]:
    """Load each configured reference-mode LoRA as a disabled adapter named after
    its mode; requests enable one per render. Returns the modes that loaded — a
    LoRA that fails only disables its own mode."""
    loaded = set()
    for mode, source in reference_lora_sources().items():
        try:
            weight = source.required[0] if source.required else None
            pipe.load_lora_weights(str(_snapshot(source)), weight_name=weight, adapter_name=mode)
        except Exception as exc:
            logger.warning("%s LoRA %s unavailable (%s: %s); reference_mode=%s disabled",
                           mode, source.repo, type(exc).__name__, exc, mode)
            continue
        loaded.add(mode)
        logger.info("%s LoRA loaded: %s", mode, source.repo)
    if loaded:
        pipe.transformer.disable_adapters()
        # PEFT creates adapters in fp32; the LoRAs ship in bf16, so this is lossless.
        for name, param in pipe.transformer.named_parameters():
            if "lora_" in name:
                param.data = param.data.to(torch.bfloat16)
    return frozenset(loaded)


def _prepare_reference_processor(pipe: DiffusionPipeline) -> None:
    """Build the pipeline's lazily-loaded Qwen3-VL processor now, so a broken
    processor shows up at startup instead of on the first reference request."""
    try:
        pipe.vl_processor
    except Exception as exc:
        logger.warning(
            "Reference-image processor unavailable (%s: %s); /v1/images/edits will fail",
            type(exc).__name__, exc,
        )


def _load_pipeline(model_name: str) -> tuple[DiffusionPipeline, frozenset[str]]:
    """Assemble Krea 2 `model_name` from its ungated sources.
    Returns (pipeline, reference modes whose LoRA loaded)."""
    if model_name not in VARIANTS:
        raise ValueError(f"unknown image model {model_name!r}; expected one of {sorted(VARIANTS)}")
    logger.info("Loading image generation pipeline: %s (pipeline %s)", model_name, IMAGEGEN_PIPELINE)
    pipeline_cls, transformer_cls = _community_classes()
    transformer = transformer_source(model_name)
    transformer_path = _snapshot(transformer) / transformer.required[0]
    logger.info("Loading transformer from %s", transformer_path)
    qwen = str(_snapshot(QWEN_VL))
    pipe = pipeline_cls(
        scheduler=FlowMatchEulerDiscreteScheduler(**SCHEDULER_CONFIG),
        vae=AutoencoderKLQwenImage.from_pretrained(
            str(_snapshot(VAE)), subfolder=VAE_SUBFOLDER, torch_dtype=torch.bfloat16
        ),
        text_encoder=Qwen3VLModel.from_pretrained(qwen, dtype=torch.bfloat16),
        tokenizer=AutoTokenizer.from_pretrained(qwen),
        transformer=load_comfy_transformer(transformer_cls, str(transformer_path)),
        is_distilled=VARIANTS[model_name].is_distilled,
    )
    # The reference-image processor comes from the same (local) Qwen3-VL snapshot.
    pipe.vl_processor_id = qwen
    reference_modes = _load_reference_loras(pipe)
    _prepare_reference_processor(pipe)
    # Transformer (~13 GB in fp8) and text encoder (~9 GB) don't fit together on
    # a 24 GB GPU: offload keeps only the active component there.
    pipe.enable_model_cpu_offload()
    logger.info("Image generation pipeline loaded (model CPU offload)")
    return pipe, reference_modes


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


def _install(app: FastAPI, model_name: str | None, pipe, reference_modes: frozenset[str]) -> None:
    app.state.pipeline = pipe
    app.state.reference_modes = reference_modes
    app.state.model_name = model_name


def _swap_pipeline(app: FastAPI, model_name: str) -> None:
    """Unload the current model and load `model_name` in its place. Holds the lock
    so it can never race an in-flight generation."""
    with app.state.lock:
        old = app.state.pipeline
        _install(app, None, None, frozenset())
        _free_pipeline(old)
        pipe, reference_modes = _load_pipeline(model_name)
        _install(app, model_name, pipe, reference_modes)


def _unload_pipeline(app: FastAPI) -> None:
    """Unload the current model and free its VRAM, leaving nothing loaded."""
    with app.state.lock:
        old = app.state.pipeline
        _install(app, None, None, frozenset())
        _free_pipeline(old)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    _log_startup_env("imagegen", _ENV_KEYS)
    # Serializes generation against model swaps (both run in worker threads).
    app.state.lock = threading.Lock()
    pipe, reference_modes = _load_pipeline(IMAGEGEN_MODEL)
    _install(app, IMAGEGEN_MODEL, pipe, reference_modes)
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


def _set_adapter(pipe: DiffusionPipeline, adapter: str | None) -> None:
    """Activate one reference-mode LoRA, or none."""
    if adapter:
        pipe.transformer.enable_adapters()
        pipe.set_adapters(adapter)
    else:
        pipe.transformer.disable_adapters()


def _reset_offload(pipe: DiffusionPipeline) -> None:
    """Put every model back on CPU and re-arm model CPU offload after a failed render.

    A render that dies mid-way (typically OOM while an offload hook is moving a
    model to the GPU) can leave that model split between GPU and CPU. accelerate's
    offload hook only checks the first parameter's device before skipping the
    move, so it never repairs the split, and every later render fails with
    "Expected all tensors to be on the same device". The pipeline's own cleanup
    (maybe_free_model_hooks) only runs on success, so run it here.
    """
    try:
        pipe.maybe_free_model_hooks()
    except Exception as exc:  # best-effort — never mask the original error
        logger.warning("Offload reset after failed render hit %s: %s", type(exc).__name__, exc)


def _generate(
    app: FastAPI,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    references: list[Image.Image] | None = None,
    adapter: str | None = None,
) -> Image.Image:
    # Held for the whole render so a model swap can't pull the pipeline out from
    # under us mid-generation.
    with app.state.lock:
        pipe = app.state.pipeline
        if pipe is None:
            raise RuntimeError("no image model is loaded")
        if adapter:
            _set_adapter(pipe, adapter)
        try:
            return _render(pipe, prompt, negative_prompt, width, height, references)
        except Exception:
            _reset_offload(pipe)
            raise
        finally:
            if adapter:
                _set_adapter(pipe, None)
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
    # style: render the prompt in the references' style (style-reference LoRA).
    # edit: generate with the references as context (subject/edit/composition);
    # only available when an edit LoRA is configured (IMAGEGEN_EDIT_LORA).
    reference_mode: Literal["style", "edit"] = "style"


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
    adapter: str | None = None,
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
        f" ({adapter})" if adapter else "",
        body.prompt[:120],
        body.negative_prompt[:120],
    )
    t0 = time.monotonic()

    data = []
    for i in range(body.n):
        try:
            image = await asyncio.to_thread(
                _generate, app, body.prompt, body.negative_prompt, width, height, references, adapter
            )
        except torch.cuda.OutOfMemoryError:
            logger.exception("GPU out of memory generating a %dx%d image", width, height)
            return _error(
                503,
                f"GPU out of memory generating a {width}x{height} image. Free VRAM (POST /v1/models/unload "
                "unloads the LLMs) or request a smaller size.",
                "api_error",
            )
        except Exception as exc:
            logger.exception("Image generation failed")
            return _error(500, f"image generation failed: {type(exc).__name__}: {exc}", "api_error")
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

    mode = body.reference_mode
    if mode not in app.state.reference_modes:
        env = "IMAGEGEN_EDIT_LORA" if mode == "edit" else "IMAGEGEN_STYLE_LORA"
        return _error(
            400,
            f"reference_mode={mode} is unavailable: its LoRA is not loaded (set {env}); "
            f"available: {sorted(app.state.reference_modes) or 'none'}",
            param="reference_mode",
        )
    width, height = _parse_size(body.size, references[0])
    return await _generate_response(body, width, height, references, mode)


class LoadModelRequest(BaseModel):
    model: str


@app.get("/v1/images/models")
async def list_image_models() -> JSONResponse:
    """List the Krea 2 variants, whether each is downloaded, and which is loaded,
    plus the /v1/images/edits reference modes the loaded model supports."""
    current = app.state.model_name
    return JSONResponse(content={
        "current": current,
        "reference_modes": sorted(app.state.reference_modes),
        "data": [
            {"id": m, "loaded": m == current, "downloaded": m == current or _is_installed(m)}
            for m in sorted(VARIANTS)
        ],
    })


@app.post("/v1/images/models/load")
async def load_image_model(body: LoadModelRequest) -> JSONResponse:
    """Swap the active image model. Loading is slow (weights + VRAM); the request
    returns once the new model is ready."""
    model = body.model.strip()
    if not model:
        return _error(400, "model is required", param="model")
    # Checked before the swap, which unloads the current model first.
    if model not in VARIANTS:
        return _error(400, f"unknown model {model!r}; expected one of {sorted(VARIANTS)}", param="model")
    if IMAGEGEN_OFFLINE and not _is_installed(model):
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
