from typing import Optional, Tuple, Union

import torch
from torch import nn


class GemmaRMSNorm(nn.Module):
    """RMSNorm with the Gemma convention: the stored weight is interpreted as
    ``(weight + 1)`` at runtime.

    Used by Qwen3.5 (and any checkpoint trained with Gemma-style
    normalization). The storage layout of ``weight`` matches the checkpoint
    exactly so existing weight loaders keep working; the ``+ 1`` is applied
    inside :meth:`forward` per call. ``ops.fused_add_rms_norm`` takes the
    scale tensor by value (the kernel just reads it), so passing
    ``weight + 1`` (a fresh tensor) is correct semantically — it does not
    mutate the parameter.

    Mirrors the ``forward(residual=...)`` contract of :class:`RMSNorm` (in-
    place residual fold + norm fused via ``ops.fused_add_rms_norm``) so it
    drops in wherever an RMSNorm is expected.
    """

    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.variance_epsilon = eps
        self.hidden_size = hidden_size
        # Init at zeros so an un-loaded ``GemmaRMSNorm`` is identity
        # (`weight + 1 == 1`).
        self.weight = nn.Parameter(torch.zeros(hidden_size))

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        from gllm import _custom_ops as ops

        scale = self.weight.data + 1.0
        if residual is not None:
            ops.fused_add_rms_norm(
                x, residual, scale, self.variance_epsilon,
            )
            return x, residual
        out = torch.empty_like(x)
        ops.rms_norm(out, x, scale, self.variance_epsilon)
        return out


class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float,
    ) -> None:
        super().__init__()
        self.variance_epsilon = eps
        self.variance_size_override = None
        self.hidden_size = hidden_size
        self.weight = nn.Parameter(torch.ones(hidden_size))
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
