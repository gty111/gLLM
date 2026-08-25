import os
from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def _extension():
    from torch.utils.cpp_extension import load

    source = os.path.join(
        os.path.dirname(__file__), "csrc", "gemma_rmsnorm.cu"
    )
    return load(
        name="gllm_gemma_rmsnorm_cuda",
        sources=[source],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )


def gemma_rmsnorm_bf16(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    epsilon: float,
    has_residual: bool,
) -> None:
    _extension().gemma_rmsnorm_bf16(
        input, residual, weight, output, epsilon, has_residual
    )
