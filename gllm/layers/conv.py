"""Conv layer utilities with workarounds for PyTorch 2.9.x CUDNN Conv3D regression.

PyTorch 2.9.0/2.9.1 disabled CUDNN's Conv3D, causing significant performance
regression. When kernel_size == stride (common in vision patch embedding), we
use unfold + F.linear as an equivalent but faster alternative.

Reference: https://github.com/pytorch/pytorch/issues/166122
"""

import math
from typing import Optional

import torch
import torch.nn.functional as F


def conv3d_patch_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    out_channels: int,
    input_size: int,
    kernel_size: tuple[int, int, int],
) -> torch.Tensor:
    """Unfold + linear forward: equivalent to Conv3d when kernel_size == stride.

    Use this as a workaround for PyTorch 2.9.x CUDNN Conv3D regression.

    Args:
        x: (batch, in_channels, T, H, W)
        weight: Conv3d weight tensor (out_channels, in_channels//groups, *kernel_size)
        bias: Optional bias tensor (out_channels,)
        out_channels: Number of output channels.
        input_size: in_channels * prod(kernel_size)
        kernel_size: (K1, K2, K3) tuple

    Returns:
        (batch, out_channels, T//K1, H//K2, W//K3)
    """
    B, C, T, H, W = x.shape
    K1, K2, K3 = kernel_size
    T_out, H_out, W_out = T // K1, H // K2, W // K3
    x = x.unfold(2, K1, K1).unfold(3, K2, K2).unfold(4, K3, K3)
    x = x.permute(0, 2, 3, 4, 1, 5, 6, 7).reshape(-1, input_size)
    x = F.linear(x, weight.view(out_channels, input_size), bias)
    x = x.view(B, T_out, H_out, W_out, out_channels).permute(0, 4, 1, 2, 3)
    return x
