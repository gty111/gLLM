# SPDX-License-Identifier: Apache-2.0
"""FP8 weights with FP16/BF16 Tensor Core compute on SM80--SM88.

Packing and scale permutations are adapted from SGLang; see csrc/marlin/NOTICE.
The checkpoint layout is kept until model-specific loading has completed.
"""

from functools import lru_cache
import hashlib
import os
from pathlib import Path
from typing import Optional

import torch
from logger import logger

from gllm.utils import direct_register_custom_op


@lru_cache(maxsize=1)
def _log_enabled():
    logger.info("FP8 linear backend: Marlin W8A16 (JIT), FP16/BF16 activations")


@lru_cache(maxsize=None)
def _load_kernel(dtype: torch.dtype, capability: tuple[int, int]):
    from tvm_ffi.cpp import load_inline

    if dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("FP8 Marlin requires FP16 or BF16 activations")
    root = Path(__file__).parent / "csrc" / "marlin"
    # Include transitive headers in the cache identity, not just the wrapper.
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    scalar = "half" if dtype == torch.float16 else "nv_bfloat16"
    arch = f"{capability[0]}{capability[1]}"
    source = f'''
#include "gptq_marlin.cuh"
#include "gptq_marlin_repack.cuh"
using tvm::ffi::TensorView;
void fp8_gemm(TensorView a, TensorView w, TensorView s, TensorView workspace,
              TensorView out, TensorView scratch, TensorView empty_int,
              TensorView empty_float) {{
    sglang::gptq_marlin_gemm<{scalar}>(
        a, w, s, empty_float, empty_int, empty_int, empty_int,
        out, scratch, empty_float, workspace,
        sglang::host::kFE4M3fn.id(), true, false, true, false);
}}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(fp8_gemm, fp8_gemm);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(repack, sglang::gptq_marlin_repack);
'''
    # Conda CUDA toolkits place libcudart under targets rather than lib64.
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    ldflags = []
    if cuda_home:
        for path in (Path(cuda_home) / "targets").glob("*/lib"):
            if (path / "libcudart.so").exists():
                ldflags.append(f"-L{path}")
    return load_inline(
        name=f"gllm_fp8_marlin_{scalar}_sm{arch}_{digest.hexdigest()[:16]}",
        cuda_sources=source,
        extra_include_paths=[str(root), str(root / "include")],
        extra_cuda_cflags=[
            "-std=c++20", "-O3", "--expt-relaxed-constexpr",
            f"-DSGL_CUDA_ARCH={capability[0] * 100 + capability[1] * 10}",
            f"-gencode=arch=compute_{arch},code=sm_{arch}",
        ],
        extra_ldflags=ldflags,
    )


def _fake_marlin(input, weight, scales, workspace, size_n, bias=None):
    return input.new_empty((*input.shape[:-1], size_n))


def _marlin_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    scales: torch.Tensor,
    workspace: torch.Tensor,
    size_n: int,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if input.dtype != scales.dtype:
        raise ValueError("FP8 Marlin activation dtype must match the prepared scales")
    x = input.reshape(-1, input.shape[-1]).contiguous()
    output = torch.empty((x.shape[0], size_n), dtype=x.dtype, device=x.device)
    if x.shape[0]:
        kernel = _load_kernel(x.dtype, torch.cuda.get_device_capability(x.device))
        # FP32 split-K reduction avoids low-precision accumulation error.
        max_m_block = min((x.shape[0] + 15) // 16 * 16, 64)
        scratch = torch.empty(
            workspace.numel() * max_m_block * 256,
            dtype=torch.float32, device=x.device,
        )
        empty_int = torch.empty(0, dtype=torch.int32, device=x.device)
        empty_float = torch.empty(0, dtype=x.dtype, device=x.device)
        kernel.fp8_gemm(
            x, weight, scales, workspace, output, scratch, empty_int, empty_float
        )
    if bias is not None:
        output.add_(bias)
    return output.reshape(*input.shape[:-1], size_n)


direct_register_custom_op(
    "fp8_marlin_linear", _marlin_linear,
    mutates_args=["workspace"], fake_impl=_fake_marlin,
)


class FP8MarlinMethod:
    """One-time repack after TP slicing and model-specific QKV/GDN reordering."""

    def __init__(self):
        self.prepared = False

    def process_weights_after_loading(self, layer):
        if self.prepared:
            return
        weight = layer.weight
        n, k = weight.shape
        block_n, block_k = layer.weight_block_size
        if n % 64 or k % 128 or block_k != 128:
            raise ValueError(
                "FP8 Marlin requires N divisible by 64, K divisible by 128 "
                f"and block K=128; got N={n}, K={k}, block={layer.weight_block_size}"
            )
        dtype = layer.params_dtype
        kernel = _load_kernel(dtype, torch.cuda.get_device_capability(weight.device))
        packed = torch.empty((k // 16, n * 4), dtype=torch.int32, device=weight.device)
        empty = torch.empty(0, dtype=torch.int32, device=weight.device)
        kernel.repack(weight.contiguous().view(torch.int32).T.contiguous(),
                      empty, packed, k, n, 8)

        # [ceil(N/block_n), K/128] -> [K/128, N] group/channel scales.
        scales = layer.weight_scale_inv.T.repeat_interleave(block_n, dim=1)[:, :n]
        scales = scales.to(dtype)
        if block_k < k:
            perm = [i + 8 * j for i in range(8) for j in range(8)]
        else:
            perm = [2 * i + j for i in range(4) for j in (0, 1, 8, 9, 16, 17, 24, 25)]
        scales = scales.reshape(-1, len(perm))[:, perm].reshape(-1, n).contiguous()
        # Marlin's bitwise FP8 conversion has a different exponent bias.
        scales = scales * (2.0 ** (8 if dtype == torch.float16 else 120))
        layer.weight = torch.nn.Parameter(packed, requires_grad=False)
        del layer.weight_scale_inv
        layer.register_parameter("weight_scale", torch.nn.Parameter(scales, requires_grad=False))
        layer.register_buffer("marlin_workspace", torch.zeros(
            torch.cuda.get_device_properties(weight.device).multi_processor_count,
            dtype=torch.int32, device=weight.device,
        ), persistent=False)
        self.scales = layer.weight_scale
        self.workspace = layer.marlin_workspace
        self.size_n = n
        self.prepared = True
        _log_enabled()

    def __call__(self, input, weight, bias=None):
        if not self.prepared:
            raise RuntimeError("FP8 Marlin weights must be prepared after loading")
        return torch.ops.gllm.fp8_marlin_linear(
            input, weight, self.scales, self.workspace, self.size_n, bias
        )
