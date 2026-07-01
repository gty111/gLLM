import functools
from typing import Optional

import torch
import triton
import triton.language as tl

from gllm import _custom_ops as ops
from gllm.dist_utils import get_tp_size
from gllm.utils import cdiv
from logger import logger

@functools.lru_cache(maxsize=1)
def _sgl_group_quant_fp8():
    """Return the sgl-kernel fused per-token-group FP8 quant op, or ``None``.

    The Triton ``_per_token_group_quant_fp8`` launches one program per token and
    is dominated by launch overhead at decode batch sizes (thousands of tiny
    calls). sgl-kernel ships a compiled CUDA kernel (the same one vLLM/SGLang
    use) that is markedly cheaper per launch; resolved once, lazily, and falls
    back to Triton when sgl-kernel is unavailable.
    """
    try:
        from sgl_kernel import sgl_per_token_group_quant_fp8

        return sgl_per_token_group_quant_fp8
    except Exception as e:  # noqa: BLE001
        logger.warning(f"sgl-kernel FP8 quant unavailable, using Triton: {e}")
        return None


@functools.lru_cache(maxsize=1)
def deepgemm_available() -> bool:
    """Whether the DeepGEMM block-FP8 GEMM backend can be used on this device.

    DeepGEMM ships hand-tuned Hopper (SM90+) FP8 kernels that are markedly
    faster than the Triton ``_w8a8_block_fp8_matmul`` fallback. Availability is
    resolved once, lazily (on the first FP8 linear call, i.e. during warmup,
    never during CUDA-graph capture), and includes a tiny self-test GEMM so a
    broken JIT toolchain degrades gracefully to Triton instead of crashing on
    the first real matmul.
    """
    if not torch.cuda.is_available():
        return False
    if torch.cuda.get_device_capability()[0] < 9:
        return False
    try:
        import deep_gemm  # noqa: F401

        a = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        aq, as_ = per_token_group_quant_fp8(a, 128, column_major_scales=False)
        wq = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16).to(
            torch.float8_e4m3fn
        )
        ws = torch.ones(1, 1, device="cuda", dtype=torch.float32)
        c = torch.empty(128, 128, device="cuda", dtype=torch.bfloat16)
        deep_gemm.fp8_gemm_nt((aq, as_), (wq, ws), c)
        torch.cuda.synchronize()
    except Exception as e:  # noqa: BLE001
        logger.warning(f"DeepGEMM FP8 backend unavailable, falling back to Triton: {e}")
        return False
    logger.info("DeepGEMM FP8 GEMM backend enabled (Hopper block-scale kernels).")
    return True


def _deepgemm_shape_supported(
    N: int, K: int, block_n: int, block_k: int, output_dtype: torch.dtype
) -> bool:
    # DeepGEMM's block-scale nt kernel requires bf16 output and N/K aligned to
    # the quantization block; anything else stays on the Triton path.
    return output_dtype == torch.bfloat16 and N % block_n == 0 and K % block_k == 0


# Only small-M (decode) matmuls benefit from FlashInfer's swapAB kernel; vLLM
# gates on the same M<32 threshold, and there it is a *correctness* requirement
# too (the swapAB path loses accuracy at M>=32), not merely a perf choice.
_FLASHINFER_SWAPAB_MAX_M = 32


@functools.lru_cache(maxsize=1)
def _flashinfer_blockscale_gemm():
    """Return FlashInfer's ``fp8_blockscale_gemm_sm90`` op, or ``None``."""
    try:
        from flashinfer.gemm import fp8_blockscale_gemm_sm90

        return fp8_blockscale_gemm_sm90
    except Exception as e:  # noqa: BLE001
        logger.warning(f"FlashInfer swapAB FP8 GEMM unavailable: {e}")
        return None


@functools.lru_cache(maxsize=1)
def flashinfer_swapab_available() -> bool:
    """Whether FlashInfer's Hopper swapAB FP8 block-scale GEMM can be used.

    ``fp8_blockscale_gemm_sm90`` takes a *BF16* activation and fuses the
    per-token FP8 quantization with a swapAB GEMM that is ~2x faster than
    DeepGEMM's 1d2d kernel for skinny-M (decode) shapes. It is Hopper-only and
    JIT-compiled on first use (needs ninja + nvcc + CUDA_HOME), so availability
    is resolved once, lazily, with a self-test that degrades gracefully to the
    DeepGEMM/Triton path when the JIT toolchain is missing.
    """
    if not torch.cuda.is_available():
        return False
    if torch.cuda.get_device_capability() != (9, 0):
        return False
    fi_gemm = _flashinfer_blockscale_gemm()
    if fi_gemm is None:
        return False
    try:
        a = torch.randn(8, 256, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16).to(
            torch.float8_e4m3fn
        )
        ws = torch.ones(1, 2, device="cuda", dtype=torch.float32)
        fi_gemm(a, w, None, ws, out_dtype=torch.bfloat16)
        torch.cuda.synchronize()
    except Exception as e:  # noqa: BLE001
        logger.warning(
            f"FlashInfer swapAB self-test failed, using DeepGEMM/Triton: {e}"
        )
        return False
    logger.info("FlashInfer swapAB FP8 GEMM enabled for decode (M<32).")
    return True


def _flashinfer_shape_supported(
    N: int, K: int, block_n: int, block_k: int, weight_scale: torch.Tensor
) -> bool:
    # fp8_blockscale_gemm_sm90 hardcodes 128x128 block quantization: it quantizes
    # the BF16 activation at group=128 internally and expects per-block weight
    # scales laid out as (ceil(N/128), ceil(K/128)). Reject any other block
    # geometry or a mismatched scale tensor so we never feed the kernel a layout
    # it will silently misinterpret (wrong results) instead of the intended GEMM.
    if block_n != 128 or block_k != 128:
        return False
    if N % 64 != 0 or K % 128 != 0:
        return False
    return tuple(weight_scale.shape) == (cdiv(N, 128), cdiv(K, 128))


def validate_fp8_block_shape(
    layer: torch.nn.Module,
    input_size: int,
    output_size: int,
    input_size_per_partition: int,
    output_partition_sizes: list[int],
    block_size: list[int],
) -> None:
    """Validate block quantization shapes for tensor parallelism."""

    tp_size = getattr(layer, "tp_size", get_tp_size())
    block_n, block_k = block_size[0], block_size[1]

    # Required by row parallel
    if (
        tp_size > 1
        and input_size // input_size_per_partition == tp_size
        and input_size_per_partition % block_k != 0
    ):
        raise ValueError(
            f"Weight input_size_per_partition = {input_size_per_partition} "
            f"is not divisible by weight quantization block_k = {block_k}."
        )

    # Required by column parallel or enabling merged weights
    is_tp_split = tp_size > 1 and output_size // sum(output_partition_sizes) == tp_size
    is_merged_gemm = len(output_partition_sizes) > 1
    if is_tp_split or is_merged_gemm:
        sizes_to_check = output_partition_sizes
        if not is_tp_split and is_merged_gemm:
            # In case of merged matrices, we allow the last
            # matrix to not be a multiple of block size
            sizes_to_check = output_partition_sizes[:-1]
        for output_partition_size in sizes_to_check:
            if output_partition_size % block_n != 0:
                raise ValueError(
                    f"Weight output_partition_size = "
                    f"{output_partition_size} is not divisible by "
                    f"weight quantization block_n = {block_n}."
                )


def fp8LinearMethod(
    input: torch.Tensor,
    weight: torch.Tensor,
    block_size: list[int],
    weight_scale: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
):
    assert input_scale is None
    # View input as 2D matrix for fp8 methods
    input_2d = input.view(-1, input.shape[-1])
    output_shape = [*input.shape[:-1], weight.shape[0]]

    N, K = weight.shape
    block_n, block_k = block_size[0], block_size[1]
    M = input_2d.shape[0]

    # Decode (small M): FlashInfer's swapAB kernel takes the BF16 activation
    # directly and fuses quantization + GEMM, ~2x faster than DeepGEMM's 1d2d.
    # Restricted to M<32 (perf *and* accuracy: swapAB degrades at larger M).
    if (
        M < _FLASHINFER_SWAPAB_MAX_M
        and input.dtype == torch.bfloat16
        and _flashinfer_shape_supported(N, K, block_n, block_k, weight_scale)
        and flashinfer_swapab_available()
    ):
        output = _flashinfer_blockscale_gemm()(
            input_2d, weight, None, weight_scale, out_dtype=torch.bfloat16
        )
    else:
        q_input, x_scale = per_token_group_quant_fp8(
            input_2d, block_size[1], column_major_scales=False
        )
        if _deepgemm_shape_supported(
            N, K, block_n, block_k, input.dtype
        ) and deepgemm_available():
            output = w8a8_block_fp8_matmul_deepgemm(
                q_input, weight, x_scale, weight_scale, block_size, input.dtype
            )
        else:
            output = w8a8_block_fp8_matmul(
                q_input, weight, x_scale, weight_scale, block_size, input.dtype
            )

    if bias is not None:
        output = output + bias
    return output.to(dtype=input.dtype).view(*output_shape)


def w8a8_block_fp8_matmul_deepgemm(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Block-wise FP8 matmul via DeepGEMM's Hopper kernels (``C = A @ B^T``).

    Drop-in replacement for :func:`w8a8_block_fp8_matmul` with identical
    inputs: ``A`` (M, K) fp8 activations with per-token-group row-major scales
    ``As`` (M, K/block_k), and ``B`` (N, K) fp8 weights with per-block scales
    ``Bs`` (N/block_n, K/block_k). DeepGEMM re-lays-out the scales into its
    required TMA/UE8M0 form internally, so the raw scales are passed as-is.
    """
    import deep_gemm

    assert output_dtype == torch.bfloat16, "DeepGEMM only outputs bfloat16"
    N = B.shape[0]
    C = A.new_empty(A.shape[:-1] + (N,), dtype=output_dtype)
    deep_gemm.fp8_gemm_nt((A, As), (B, Bs), C.view(-1, N))
    return C


def w8a8_block_fp8_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """This function performs matrix multiplication with block-wise
    quantization.
    It takes two input tensors `A` and `B` with scales `As` and `Bs`.
    The output is returned in the specified `output_dtype`.
    Args:
        A: The input tensor, e.g., activation.
        B: The input tensor, e.g., weight.
        As: The per-token-group quantization scale for `A`.
        Bs: The per-block quantization scale for `B`.
        block_size: The block size for per-block quantization. It should
        be 2-dim, e.g., [128, 128].
        output_dytpe: The dtype of the returned tensor.
    Returns:
        torch.Tensor: The result of matmul.
    """
    assert len(block_size) == 2
    block_n, block_k = block_size[0], block_size[1]

    assert A.shape[-1] == B.shape[-1]
    assert A.shape[:-1] == As.shape[:-1] and A.is_contiguous()
    assert triton.cdiv(A.shape[-1], block_k) == As.shape[-1]
    M = A.numel() // A.shape[-1]

    assert B.ndim == 2 and Bs.ndim == 2
    N, K = B.shape
    assert triton.cdiv(N, block_n) == Bs.shape[0]
    assert triton.cdiv(K, block_k) == Bs.shape[1]

    C_shape = A.shape[:-1] + (N,)
    C = A.new_empty(C_shape, dtype=output_dtype)

    configs = None
    if configs:
        # Get the optimal config if there is one
        config = configs[min(configs.keys(), key=lambda x: abs(x - M))]
    else:
        # Default config
        # Block-wise quant: BLOCK_SIZE_N must be divisible by block_size[0]
        # BLOCK_SIZE_K must be divisible by block_size[1]
        config = {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": block_size[0],
            "BLOCK_SIZE_K": block_size[1],
            "GROUP_SIZE_M": 32,
            "num_warps": 4,
            "num_stages": 2,
        }

    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

    _w8a8_block_fp8_matmul[grid](
        A,
        B,
        C,
        As,
        Bs,
        M,
        N,
        K,
        block_n,
        block_k,
        A.stride(-2),
        A.stride(-1),
        B.stride(1),
        B.stride(0),
        C.stride(-2),
        C.stride(-1),
        As.stride(-2),
        As.stride(-1),
        Bs.stride(1),
        Bs.stride(0),
        **config,
    )

    return C


@triton.jit
def _w8a8_block_fp8_matmul(
    # Pointers to inputs and output
    A,
    B,
    C,
    As,
    Bs,
    # Shape for matmul
    M,
    N,
    K,
    # Block size for block-wise quantization
    group_n,
    group_k,
    # Stride for inputs and output
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_As_m,
    stride_As_k,
    stride_Bs_k,
    stride_Bs_n,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Triton-accelerated function used to perform linear operations (dot
    product) on input tensors `A` and `B` with block-wise quantization, and
    store the result in output tensor `C`.
    """

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    As_ptrs = As + offs_am * stride_As_m
    offs_bsn = offs_bn // group_n
    Bs_ptrs = Bs + offs_bsn * stride_Bs_n

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        k_start = k * BLOCK_SIZE_K
        offs_ks = k_start // group_k
        a_s = tl.load(As_ptrs + offs_ks * stride_As_k)
        b_s = tl.load(Bs_ptrs + offs_ks * stride_Bs_k)

        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if C.dtype.element_ty == tl.bfloat16:
        c = accumulator.to(tl.bfloat16)
    elif C.dtype.element_ty == tl.float16:
        c = accumulator.to(tl.float16)
    else:
        c = accumulator.to(tl.float32)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def input_to_float8(
    x: torch.Tensor, dtype: Optional[torch.dtype] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """This function quantizes input values to float8 values "
    "with tensor-wise quantization."""
    dtype = torch.float8_e4m3fn if dtype is None else dtype
    finfo = torch.finfo(dtype)
    min_val, max_val = x.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs()).clamp(min=1e-12)
    scale = finfo.max / amax
    x_scl_sat = (x * scale).clamp(min=finfo.min, max=finfo.max)
    return x_scl_sat.to(dtype).contiguous(), scale.float().reciprocal()


# Normalize the group_shape to the full extent for any dims that are -1
def _normalize_quant_group_shape(x: torch.Tensor, group_shape: tuple[int, int]):
    # -1 means full extent
    return (
        group_shape[0] if group_shape[0] > 0 else x.shape[-2],
        group_shape[1] if group_shape[1] > 0 else x.shape[-1],
    )


# Useful when treating N-dimensional group scaling as extended numpy-style
# broadcasting in numpy simply stretches dimensions with an extent of 1 to match
# the target shape by repeating the data along that dimension (broadcasting)
# , we extend these semantics to say if the extent of a dimension in the
# source shape is not 1 and does not match the target shape we repeat each
# element along that dimension src_shape[dim] // target_shape[dim] times
# example if we have:
#       a = [[1, 2], and target_shape = (2, 4)
#            [3, 4]]
# then we would expand a to:
#       a = [[1, 1, 2, 2],
#            [3, 3, 4, 4]]
# NOTE this function this function does not explicitly broadcast dimensions
# with an extent of 1, since this can be done implicitly by pytorch
def group_broadcast(t, shape):
    for i, s in enumerate(shape):
        if t.shape[i] != s and t.shape[i] != 1:
            assert s % t.shape[i] == 0
            t = (
                t.unsqueeze(i + 1)
                .expand(*t.shape[: i + 1], s // t.shape[i], *t.shape[i + 1 :])
                .flatten(i, i + 1)
            )
    return t


# inverses `scaled_quantize`
def scaled_dequantize(
    x_q: torch.Tensor,
    x_s: torch.Tensor,
    group_shape: Optional[tuple[int, int]] = None,
    out_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    if group_shape is not None:
        group_shape = _normalize_quant_group_shape(x_q, group_shape)

    if x_s.ndim == 0:  # scalar
        x_s = x_s.unsqueeze(-1).unsqueeze(-1)  # convert to (1, 1) tensor
    if x_s.ndim == 1:
        if group_shape is None:
            raise AssertionError(
                "if x_s is 1D tensor, group_shape must be provided otherwise "
                "its ambiguous which dimension to broadcast x_s to"
            )
        # unsqueeze the scales for the dimension where we want to broadcast
        # across the full extent
        if group_shape[0] == x_q.shape[-2]:
            x_s = x_s.unsqueeze(-2)
        elif group_shape[1] == x_q.shape[-1]:
            x_s = x_s.unsqueeze(-1)
        else:
            raise AssertionError(
                "if x_s is a vector we should be broadcasting it to the full "
                "extent of one of the dimensions"
            )

    if group_shape is not None:
        assert x_s.shape[-1] == x_q.shape[-1] // group_shape[1]
        assert x_s.shape[-2] == x_q.shape[-2] // group_shape[0]
    x_s = group_broadcast(x_s.to(torch.float32), x_q.shape)
    return (x_q.to(torch.float32) * x_s).to(out_dtype)


def block_quant_to_tensor_quant(
    x_q_block: torch.Tensor,
    x_s: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """This function converts block-wise quantization to tensor-wise
    quantization. The inputs are block-wise quantization tensor `x_q_block`,
    block-wise quantization scale and the block size.
    The outputs are tensor-wise quantization tensor and tensor-wise
    quantization scale. Note only float8 is supported for now.
    """
    x_dq_block = scaled_dequantize(x_q_block, x_s)
    x_q_tensor, scale = input_to_float8(x_dq_block, dtype=x_q_block.dtype)
    return x_q_tensor, scale


@triton.jit
def _per_token_group_quant_fp8(
    # Pointers to inputs and output
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    group_size,
    # Num columns of y
    y_num_columns,
    y_row_stride,
    # Avoid to divide zero
    eps,
    # Information for float8
    fp8_min,
    fp8_max,
    # Meta-parameters
    BLOCK: tl.constexpr,
):
    """A Triton-accelerated function to perform per-token-group
    quantization on a tensor.
    This function converts the tensor values into float8 values.
    """
    groups_per_row = y_num_columns // group_size

    # Map the program id to the row of X and Y it should compute.
    g_id = tl.program_id(0)
    row = g_id // groups_per_row
    row_g_id = g_id % groups_per_row

    # Ensure offset calculations use int64 to prevent overflow
    y_ptr_offset = (row.to(tl.int64) * y_row_stride) + (
        row_g_id.to(tl.int64) * group_size
    )
    y_ptr += y_ptr_offset

    y_q_ptr_offset = g_id.to(tl.int64) * group_size
    y_q_ptr += y_q_ptr_offset
    y_s_ptr += g_id

    cols = tl.arange(0, BLOCK)  # N <= BLOCK
    mask = cols < group_size

    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    # Quant
    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
    y_s = _absmax / fp8_max
    y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)


@triton.jit
def _per_token_group_quant_fp8_colmajor(
    # Pointers to inputs and output
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    group_size,
    # Num columns of y
    y_num_columns,
    y_row_stride,
    # Stride from one column to the next of y_s
    y_s_col_stride,
    # Avoid to divide zero
    eps,
    # Information for float8
    fp8_min,
    fp8_max,
    # Meta-parameters
    BLOCK: tl.constexpr,
):
    """A Triton-accelerated function to perform per-token-group
    quantization on a tensor.
    This function converts the tensor values into float8 values.
    """
    groups_per_row = y_num_columns // group_size

    # Map the program id to the row of X and Y it should compute.
    g_id = tl.program_id(0)
    row = g_id // groups_per_row
    row_g_id = g_id % groups_per_row

    # Ensure offset calculations use int64 to prevent overflow
    y_ptr_offset = (row.to(tl.int64) * y_row_stride) + (
        row_g_id.to(tl.int64) * group_size
    )
    y_ptr += y_ptr_offset

    y_q_ptr_offset = g_id.to(tl.int64) * group_size
    y_q_ptr += y_q_ptr_offset

    # Convert g_id the flattened block coordinate to 2D so we can index
    # into the output y_scales matrix
    blocks_per_row = y_num_columns // group_size
    scale_col = g_id % blocks_per_row
    scale_row = g_id // blocks_per_row
    # Ensure offset calculation uses int64 for y_s_ptr
    y_s_ptr_offset = (scale_col.to(tl.int64) * y_s_col_stride) + scale_row.to(tl.int64)
    y_s_ptr += y_s_ptr_offset

    cols = tl.arange(0, BLOCK)  # group_size <= BLOCK
    mask = cols < group_size

    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    # Quant
    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
    y_s = _absmax / fp8_max
    y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)


def per_token_group_quant_fp8(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dtype: Optional[torch.dtype] = None,
    column_major_scales: bool = False,
    out_q: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Function to perform per-token-group quantization on an input tensor `x`.
    It converts the tensor values into signed float8 values and returns the
    quantized tensor along with the scaling factor used for quantization.
    Args:
        x: The input tensor with ndim >= 2.
        group_size: The group size used for quantization.
        eps: The minimum to avoid dividing zero.
        dtype: The dype of output tensor. Note that only `torch.float8_e4m3fn`
        is supported for now.
        column_major_scales: Outputs scales in column major.
        out_q: Optional output tensor. If not provided, function will create.
    Returns:
        tuple[torch.Tensor, torch.Tensor]: The quantized tensor and the
        scaling factor for quantization.
    """
    dtype = torch.float8_e4m3fn if dtype is None else dtype
    assert x.shape[-1] % group_size == 0, (
        f"the last dimension of `x` {x.shape[-1]} must be divisible "
        f"by `group_size` {group_size}"
    )
    assert x.stride(-1) == 1, "`x` groups must be contiguous"

    finfo = torch.finfo(dtype)
    fp8_min = finfo.min
    fp8_max = finfo.max

    assert out_q is None or out_q.shape == x.shape
    x_q = out_q
    if x_q is None:
        x_q = torch.empty_like(x, device=x.device, dtype=dtype)

    # Fast path: fused CUDA kernel from sgl-kernel. Only the row-major scale
    # layout is a drop-in match for the Triton kernel below, which is exactly
    # what every gLLM caller uses; anything else falls through to Triton.
    sgl_quant = _sgl_group_quant_fp8()
    if (
        sgl_quant is not None
        and not column_major_scales
        and dtype == torch.float8_e4m3fn
        and x.is_contiguous()
    ):
        x_s = torch.empty(
            x.shape[:-1] + (x.shape[-1] // group_size,),
            device=x.device,
            dtype=torch.float32,
        )
        if x.shape[0] > 0:
            # enable_v2=False keeps the call self-contained: the sgl-kernel
            # wrapper otherwise imports sglang just to read an env var.
            sgl_quant(
                x, x_q, x_s, group_size, eps, fp8_min, fp8_max, enable_v2=False
            )
        return x_q, x_s

    M = x.numel() // group_size
    N = group_size
    if column_major_scales:
        shape = (x.shape[-1] // group_size,) + x.shape[:-1]
        x_s = torch.empty(shape, device=x.device, dtype=torch.float32).permute(-1, -2)
    else:
        shape = x.shape[:-1] + (x.shape[-1] // group_size,)
        x_s = torch.empty(shape, device=x.device, dtype=torch.float32)

    BLOCK = triton.next_power_of_2(N)
    # heuristics for number of warps
    num_warps = min(max(BLOCK // 256, 1), 8)
    num_stages = 1
    if column_major_scales:
        _per_token_group_quant_fp8_colmajor[(M,)](
            x,
            x_q,
            x_s,
            group_size,
            x.shape[1],
            x.stride(0),
            x_s.stride(1),
            eps,
            fp8_min=fp8_min,
            fp8_max=fp8_max,
            BLOCK=BLOCK,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        _per_token_group_quant_fp8[(M,)](
            x,
            x_q,
            x_s,
            group_size,
            x.shape[1],
            x.stride(0),
            eps,
            fp8_min=fp8_min,
            fp8_max=fp8_max,
            BLOCK=BLOCK,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return x_q, x_s


def _fp8_quantize(
    A: torch.Tensor,
    A_scale: torch.Tensor | None,
    per_act_token: bool,
    block_shape: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Perform fp8 quantization on the inputs.  If a block_shape
    is provided, the output will be blocked.
    """
    if block_shape is None:
        A, A_scale = ops.scaled_fp8_quant(
            A, A_scale, use_per_token_if_dynamic=per_act_token
        )
    else:
        assert not per_act_token
        assert len(block_shape) == 2
        _, block_k = block_shape[0], block_shape[1]
        A, A_scale = per_token_group_quant_fp8(A, block_k)
        assert cdiv(A.size(-1), block_k) == A_scale.size(-1)

    return A, A_scale
