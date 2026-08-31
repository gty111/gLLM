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

    def __init__(
        self,
        hidden_size: int,
        eps: float,
        params_dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.variance_epsilon = eps
        self.variance_size_override = None
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

    def forward_native(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """PyTorch-native implementation equivalent to forward()."""
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        if residual is not None:
            x = x + residual.to(torch.float32)
            residual = x.to(orig_dtype)

        hidden_size = x.shape[-1]
        if hidden_size != self.hidden_size:
            raise ValueError(
                "Expected hidden_size to be "
                f"{self.hidden_size}, but found: {hidden_size}"
            )

        if self.variance_size_override is None:
            x_var = x
        else:
            if hidden_size < self.variance_size_override:
                raise ValueError(
                    "Expected hidden_size to be at least "
                    f"{self.variance_size_override}, but found: {hidden_size}"
                )

            x_var = x[:, :, : self.variance_size_override]

        variance = x_var.pow(2).mean(dim=-1, keepdim=True)

        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = x.to(orig_dtype)
        if self.has_weight:
            x = x * self.weight
        if residual is None:
            return x
        else:
            return x, residual
