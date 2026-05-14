"""Conv layers with workarounds for PyTorch 2.9.x CUDNN Conv3D regression.

PyTorch 2.9.0/2.9.1 disabled CUDNN's Conv3D, causing significant performance
regression. When kernel_size == stride (common in vision patch embedding), we
use unfold + F.linear as an equivalent but faster alternative.

Reference: https://github.com/pytorch/pytorch/issues/166122
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _is_torch_29x() -> bool:
    return torch.__version__.startswith("2.9.")


class Conv3dPatchEmbed(nn.Module):
    """Conv3d layer optimized for patch embedding (kernel_size == stride).

    Automatically uses unfold + linear on PyTorch 2.9.x to avoid the
    CUDNN Conv3D regression.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Tuple of (T, H, W) kernel sizes.
        bias: Whether to include bias.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int, int],
        bias: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.input_size = in_channels * math.prod(kernel_size)

        # stride == kernel_size for patch embedding
        self.proj = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=bias,
        )

        self._use_linear = _is_torch_29x()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, in_channels, T, H, W)

        Returns:
            (batch, out_channels, T//K1, H//K2, W//K3)
        """
        if self._use_linear:
            return self._forward_linear(x)
        return self.proj(x)

    def _forward_linear(self, x: torch.Tensor) -> torch.Tensor:
        """Unfold + linear: equivalent to conv3d when kernel_size == stride."""
        B, C, T, H, W = x.shape
        K1, K2, K3 = self.kernel_size
        T_out, H_out, W_out = T // K1, H // K2, W // K3
        x = x.unfold(2, K1, K1).unfold(3, K2, K2).unfold(4, K3, K3)
        x = x.permute(0, 2, 3, 4, 1, 5, 6, 7).reshape(-1, self.input_size)
        weight = self.proj.weight.view(self.out_channels, self.input_size)
        x = F.linear(x, weight, self.proj.bias)
        x = x.view(B, T_out, H_out, W_out, self.out_channels).permute(0, 4, 1, 2, 3)
        return x
