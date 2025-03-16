import torch
from torch import nn

class SiluAndMul(nn.Module):
    def forward(self, x) -> torch.Tensor:
        from gllm import _custom_ops as ops

        d = x.shape[-1] // 2
        output_shape = (x.shape[:-1] + (d, ))
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        ops.silu_and_mul(out, x)
        return out
