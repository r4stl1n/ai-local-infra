"""Load a Comfy-Org Krea 2 transformer file into the diffusers model.

Comfy-Org/Krea-2 ships the transformer as one safetensors file in the original
Krea key layout, either in bf16 or "fp8_scaled": every block linear stored as
float8_e4m3fn plus a per-tensor `weight_scale` (dequantized weight = fp8 * scale),
which halves the ~26 GB bf16 transformer to ~13 GB.
"""

from __future__ import annotations

import re

import torch
import torch.nn.functional as F
from accelerate import init_empty_weights
from safetensors.torch import load_file
from torch import nn

# Original (Comfy) module names -> diffusers module names, outside the blocks.
_STANDALONE_MODULES = {
    "first": "img_in",
    "last.linear": "final_layer.linear",
    "tmlp.0": "time_embed.linear_1",
    "tmlp.2": "time_embed.linear_2",
    "tproj.1": "time_mod_proj",
    "txtmlp.1": "txt_in.linear_1",
    "txtmlp.3": "txt_in.linear_2",
    "txtfusion.projector": "text_fusion.projector",
}
_STANDALONE_TENSORS = {
    "last.modulation.lin": "final_layer.scale_shift_table",
    "last.norm.scale": "final_layer.norm.weight",
    "txtmlp.0.scale": "txt_in.norm.weight",
}
# Per-block tensors (blocks.N.* and txtfusion.{layerwise,refiner}_blocks.N.*).
# Norm `scale`s are zero-centered (multiplier = 1 + scale), as is Krea2RMSNorm's
# `weight`, so they copy over unchanged.
_BLOCK_TENSORS = {
    "attn.wq.weight": "attn.to_q.weight",
    "attn.wk.weight": "attn.to_k.weight",
    "attn.wv.weight": "attn.to_v.weight",
    "attn.wo.weight": "attn.to_out.0.weight",
    "attn.gate.weight": "attn.to_gate.weight",
    "attn.qknorm.qnorm.scale": "attn.norm_q.weight",
    "attn.qknorm.knorm.scale": "attn.norm_k.weight",
    "mlp.gate.weight": "ff.gate.weight",
    "mlp.up.weight": "ff.up.weight",
    "mlp.down.weight": "ff.down.weight",
    "prenorm.scale": "norm1.weight",
    "postnorm.scale": "norm2.weight",
    "mod.lin": "scale_shift_table",
}
_BLOCK_RE = re.compile(r"^(blocks|txtfusion\.(?:layerwise|refiner)_blocks)\.(\d+)\.(.+)$")
_SCALE_SUFFIX = ".weight_scale"


def convert_key(key: str) -> str | None:
    """Diffusers name for an original-layout tensor name, or None if unknown."""
    m = _BLOCK_RE.match(key)
    if m:
        group, idx, rest = m.groups()
        target = _BLOCK_TENSORS.get(rest)
        if target is None:
            return None
        prefix = "transformer_blocks" if group == "blocks" else group.replace("txtfusion", "text_fusion")
        return f"{prefix}.{idx}.{target}"
    if key in _STANDALONE_TENSORS:
        return _STANDALONE_TENSORS[key]
    module, _, param = key.rpartition(".")
    if module in _STANDALONE_MODULES and param in ("weight", "bias"):
        return f"{_STANDALONE_MODULES[module]}.{param}"
    return None


class ScaledFP8Linear(nn.Linear):
    """nn.Linear holding an fp8 weight plus a per-tensor scale, dequantized to the
    input dtype on each call.

    The fp8 bytes are stored as uint8: PEFT casts LoRA adapters to the base
    layer's dtype when that dtype is floating point, which for float8 would
    quantize the adapters too. As uint8 it leaves them alone.
    """

    weight_scale: torch.Tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight.view(torch.float8_e4m3fn).to(x.dtype)
        weight.mul_(self.weight_scale.to(x.dtype))
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, weight, bias)


def _to_scaled_fp8(linear: nn.Linear, scale: torch.Tensor) -> None:
    linear.weight = nn.Parameter(linear.weight.data.view(torch.uint8), requires_grad=False)
    linear.register_buffer("weight_scale", scale.to(torch.float32).reshape(()))
    linear.__class__ = ScaledFP8Linear


def load_comfy_transformer(transformer_cls, path: str) -> nn.Module:
    """Build `transformer_cls` with weights from a Comfy-Org bf16 or fp8_scaled file.
    The class defaults are the released Krea 2 config."""
    state = load_file(path)
    scales = {key[: -len(_SCALE_SUFFIX)]: state.pop(key) for key in list(state) if key.endswith(_SCALE_SUFFIX)}

    converted: dict[str, torch.Tensor] = {}
    for key, tensor in state.items():
        target = convert_key(key)
        if target is None:
            raise ValueError(f"unrecognised tensor in {path}: {key}")
        converted[target] = tensor

    # Parameters on meta (no allocation); buffers such as rotary tables stay real.
    with init_empty_weights(include_buffers=False):
        model = transformer_cls()
    expected = model.state_dict()
    missing = sorted(set(expected) - set(converted))
    unexpected = sorted(set(converted) - set(expected))
    if missing or unexpected:
        raise ValueError(f"transformer file does not match the model: missing={missing[:5]} unexpected={unexpected[:5]}")
    for key, tensor in converted.items():
        shape = expected[key].shape
        if tensor.shape != shape:
            if tensor.numel() != expected[key].numel():
                raise ValueError(f"{key}: shape {tuple(tensor.shape)} != {tuple(shape)}")
            converted[key] = tensor.reshape(shape)  # e.g. blocks.N.mod.lin is stored flat
    model.load_state_dict(converted, strict=True, assign=True)

    for original, scale in scales.items():
        module_name = convert_key(original + ".weight")
        if module_name is None:
            raise ValueError(f"unrecognised scaled tensor in {path}: {original}")
        module = model.get_submodule(module_name[: -len(".weight")])
        if not isinstance(module, nn.Linear) or module.weight.dtype != torch.float8_e4m3fn:
            raise ValueError(f"{module_name}: expected an fp8 nn.Linear weight")
        _to_scaled_fp8(module, scale)
    return model.eval()
