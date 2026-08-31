"""Native MXFP4 linear primitives for DeepSeek-V4 style checkpoints.

The checkpoint stores two FP4 E2M1 values in every byte and one E8M0 scale
for every 32 logical weight values.  DeepSeek-V4 quantizes activations to FP8
E4M3 with one E8M0 scale per 128 values.  Keeping those two group sizes
separate is important: MXFP8 kernels which quantize activations per 32 values
do not reproduce the checkpoint's inference recipe.
"""

from __future__ import annotations

import torch

from gllm.layers.quantization.fp8 import per_token_group_quant_fp8


ACTIVATION_GROUP_SIZE = 128
WEIGHT_GROUP_SIZE = 32


def e8m0_to_float32(scale: torch.Tensor) -> torch.Tensor:
    """Decode exponent-only E8M0 values without changing their bits."""
    if scale.dtype not in (torch.float8_e8m0fnu, torch.uint8):
        raise TypeError(
            "MXFP4 scales must be float8_e8m0fnu or uint8, got "
            f"{scale.dtype}"
        )
    return (scale.view(torch.uint8).to(torch.int32) << 23).view(torch.float32)


def prepare_mxfp4_scale(
    scale: torch.Tensor,
    *,
    output_size: int,
    input_size: int,
    num_groups: int | None = None,
) -> torch.Tensor:
    """Convert checkpoint E8M0 scales to DeepGEMM's packed TMA layout.

    The returned tensor uses an ``int32`` carrier holding four E8M0 bytes, so
    preprocessing does not increase the persistent scale memory footprint.
    """
    if input_size % WEIGHT_GROUP_SIZE != 0:
        raise ValueError(
            f"input_size={input_size} must be divisible by {WEIGHT_GROUP_SIZE}"
        )
    expected = (output_size, input_size // WEIGHT_GROUP_SIZE)
    if num_groups is not None:
        expected = (num_groups, *expected)
    if tuple(scale.shape) != expected:
        raise ValueError(
            f"MXFP4 scale shape must be {expected}, got {tuple(scale.shape)}"
        )

    import deep_gemm

    scale_fp32 = (
        e8m0_to_float32(scale)
        if scale.dtype in (torch.float8_e8m0fnu, torch.uint8)
        else scale
    )
    if scale_fp32.dtype != torch.float32:
        raise TypeError(
            "MXFP4 scales must be E8M0 bytes or float32, got "
            f"{scale.dtype}"
        )
    return deep_gemm.transform_sf_into_required_layout(
        scale_fp32,
        output_size,
        input_size,
        (1, WEIGHT_GROUP_SIZE),
        num_groups=num_groups,
    )


def quantize_mxfp4_activation(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the checkpoint's FP8-E4M3/E8M0 per-128 activation recipe."""
    if x.dtype != torch.bfloat16:
        raise TypeError(f"MXFP4 activation input must be bfloat16, got {x.dtype}")
    if x.ndim != 2:
        raise ValueError(f"MXFP4 activation input must be 2D, got {x.ndim}D")
    if x.shape[1] % ACTIVATION_GROUP_SIZE != 0:
        raise ValueError(
            f"input width={x.shape[1]} must be divisible by "
            f"{ACTIVATION_GROUP_SIZE}"
        )
    return per_token_group_quant_fp8(
        x,
        ACTIVATION_GROUP_SIZE,
        column_major_scales=False,
        round_scale=True,
    )


def deepgemm_mxfp4_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute ``x @ weight.T`` using native FP8-by-FP4 DeepGEMM.

    ``weight`` is the checkpoint's packed tensor with shape ``[N, K/2]``.
    ``weight_scale`` may be the raw ``[N, K/32]`` scale tensor or the packed
    tensor returned by :func:`prepare_mxfp4_scale`.
    """
    if x.ndim != 2 or weight.ndim != 2:
        raise ValueError("MXFP4 linear expects 2D activation and weight tensors")
    if weight.dtype not in (torch.int8, torch.uint8):
        raise TypeError(f"packed MXFP4 weight must be int8 or uint8, got {weight.dtype}")
    logical_k = weight.shape[1] * 2
    if x.shape[1] != logical_k:
        raise ValueError(
            f"activation width {x.shape[1]} does not match FP4 weight width "
            f"{logical_k}"
        )
    if not x.is_cuda or not weight.is_cuda or not weight_scale.is_cuda:
        raise ValueError("native MXFP4 linear requires CUDA tensors")

    import deep_gemm

    x_q, x_scale = quantize_mxfp4_activation(x.contiguous())
    n = weight.shape[0]
    if output is None:
        output = torch.empty(
            (x.shape[0], n), device=x.device, dtype=torch.bfloat16
        )
    elif tuple(output.shape) != (x.shape[0], n) or output.dtype != torch.bfloat16:
        raise ValueError(
            "output must be bfloat16 with shape "
            f"{(x.shape[0], n)}, got {tuple(output.shape)} {output.dtype}"
        )

    raw_scale_shape = (n, logical_k // WEIGHT_GROUP_SIZE)
    if tuple(weight_scale.shape) == raw_scale_shape:
        weight_scale = prepare_mxfp4_scale(
            weight_scale,
            output_size=n,
            input_size=logical_k,
        )
    elif weight_scale.dtype != torch.int32:
        raise ValueError(
            "weight_scale must be raw [N,K/32] scales or a packed int32 "
            "DeepGEMM scale tensor"
        )

    # DeepGEMM accepts row-major FP32 activation scales and performs their
    # inexpensive TMA packing as part of dispatch.  Weight scales are packed
    # once after loading because they are much larger and reused every call.
    deep_gemm.fp8_fp4_gemm_nt(
        (x_q, x_scale),
        (weight.view(torch.int8), weight_scale),
        output,
        recipe_a=(1, ACTIVATION_GROUP_SIZE),
        recipe_b=(1, WEIGHT_GROUP_SIZE),
    )
    return output


def deepgemm_mxfp4_expert(
    x: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    *,
    routing_weight: torch.Tensor | None = None,
    swiglu_limit: float = 10.0,
) -> torch.Tensor:
    """Run one DeepSeek-V4 MXFP4 expert in checkpoint inference order.

    ``w13`` is laid out as ``[w1; w3]`` (gate followed by up).  The routing
    weight is deliberately applied after the gated activation and before
    ``w2``.  Moving it after ``w2`` changes the input to the second dynamic
    FP8 quantizer and therefore does not reproduce the reference computation.
    """
    if w13.ndim != 2 or w13.shape[0] % 2:
        raise ValueError("w13 must be 2D with an even output dimension")
    intermediate_size = w13.shape[0] // 2
    if tuple(w2.shape) != (x.shape[1], intermediate_size // 2):
        raise ValueError(
            "w2 packed shape must be "
            f"{(x.shape[1], intermediate_size // 2)}, got {tuple(w2.shape)}"
        )

    gate_up = deepgemm_mxfp4_linear(x, w13, w13_scale).float()
    gate, up = gate_up.split(intermediate_size, dim=-1)
    if swiglu_limit > 0:
        gate = gate.clamp(max=swiglu_limit)
        up = up.clamp(min=-swiglu_limit, max=swiglu_limit)
    hidden = torch.nn.functional.silu(gate) * up
    if routing_weight is not None:
        if routing_weight.ndim == 1:
            routing_weight = routing_weight.unsqueeze(-1)
        if tuple(routing_weight.shape) != (x.shape[0], 1):
            raise ValueError(
                "routing_weight must have shape [M] or [M,1], got "
                f"{tuple(routing_weight.shape)}"
            )
        hidden = hidden * routing_weight.float()
    return deepgemm_mxfp4_linear(
        hidden.to(torch.bfloat16), w2, w2_scale
    )


__all__ = [
    "ACTIVATION_GROUP_SIZE",
    "WEIGHT_GROUP_SIZE",
    "deepgemm_mxfp4_linear",
    "deepgemm_mxfp4_expert",
    "e8m0_to_float32",
    "prepare_mxfp4_scale",
    "quantize_mxfp4_activation",
]
