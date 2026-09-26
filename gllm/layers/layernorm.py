from typing import Optional, Tuple, Union

import torch
from torch import nn


class GemmaRMSNorm(nn.Module):
    """RMSNorm with the Gemma convention: the stored weight is interpreted as
    ``(weight + 1)`` at runtime.

    Used by Qwen3.5 (and any checkpoint trained with Gemma-style
    normalization). The storage layout of ``weight`` matches the checkpoint
    exactly so existing weight loaders keep working.  The dedicated Gemma
    kernels apply ``+ 1`` in fp32 internally; precomputing it in bf16 would
    prematurely round the checkpoint's small learned offsets.

    Mirrors the ``forward(residual=...)`` contract of :class:`RMSNorm` (in-
    place residual fold + norm fused via ``ops.gemma_fused_add_rms_norm``) so it
    drops in wherever an RMSNorm is expected.
    """

    # Constant added to the stored weight to form the effective gain. Read by
    # kernels that apply the gain themselves (see
    # ``gllm.layers.fused_allreduce_norm``) so a norm's convention travels with
    # the class instead of being re-derived from its name.
    weight_bias = 1.0

    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.variance_epsilon = eps
        self.hidden_size = hidden_size
        # Init at zeros so an un-loaded ``GemmaRMSNorm`` is identity
        # (`weight + 1 == 1`).
        self.weight = nn.Parameter(torch.zeros(hidden_size, device="cuda"))

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        from gllm import _custom_ops as ops

        if residual is not None:
            ops.gemma_fused_add_rms_norm(
                x, residual, self.weight.data, self.variance_epsilon,
            )
            return x, residual
        out = torch.empty_like(x)
        ops.gemma_rms_norm(
            out, x, self.weight.data, self.variance_epsilon
        )
        return out


class RMSNorm(nn.Module):

    # The stored weight is the gain as-is; see ``GemmaRMSNorm.weight_bias``.
    weight_bias = 0.0

    def __init__(
        self,
        hidden_size: int,
        eps: float,
        params_dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.variance_epsilon = eps
        self.hidden_size = hidden_size
        # ``params_dtype`` lets a checkpoint whose norms are not the default
        # dtype say so at construction. Without it every caller has to reach
        # into ``.weight.data`` afterwards, which is easy to forget and easy to
        # get wrong on only some of a model's norms.
        self.weight = nn.Parameter(
            torch.ones(hidden_size, device="cuda", dtype=params_dtype)
        )
        self.has_weight = True

    def forward(
        self,
        x,
        residual=None,
    ):
        from gllm import _custom_ops as ops

        if residual is not None:
            ops.fused_add_rms_norm(
                x,
                residual,
                self.weight.data,
                self.variance_epsilon,
            )
            return x, residual
        out = torch.empty_like(x)
        ops.rms_norm(
            out,
            x,
            self.weight.data,
            self.variance_epsilon,
        )
        return out
