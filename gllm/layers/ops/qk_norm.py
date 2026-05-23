"""Triton fused q/k RMS-norm kernel.

Fuses ``q_norm`` and ``k_norm`` (two independent per-head ``RMSNorm``
calls in vanilla Qwen3) into a single kernel pass. Each program instance
processes one ``(token, head)`` slice along the head_dim, eliminating the
host-side launch latency for one of the two norm kernels and the
intermediate ``aten::view``/``contiguous`` overhead between them.

Use case: VL/MoE Qwen3 paths where ``fused_qk_norm_rope`` is unavailable
(MRoPE / rope_scaling) so we still want q_norm + k_norm collapsed but
keep RoPE as a separate kernel. Mirrors SGLang's
``fused_inplace_qknorm`` JIT kernel (see
``sglang/jit_kernel/csrc/elementwise/qknorm.cuh``); profile shows the
sum of q_norm + k_norm drops from ~36 ms (19303 calls @ 1.9us) to ~14 ms
(9552 calls @ ~1.5us) on Qwen3-VL-30B-A3B-Instruct TP=4 on H20-3e.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_qk_norm_kernel(
    q_ptr,
    k_ptr,
    q_weight_ptr,
    k_weight_ptr,
    eps,
    n_q_heads,
    n_kv_heads,
    q_stride_token,
    k_stride_token,
    n_tokens,
    HEAD_DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """In-place RMS-norm on each (token, head) slice of q and k.

    Layout (head_dim must be the innermost contiguous axis, but tokens
    may have any stride, so we accept slices of QKVParallelLinear's
    output without a forced ``contiguous()``):
      * ``q``: shape ``[n_tokens, n_q_heads * head_dim]`` with
        ``q.stride(0) == q_stride_token``.
      * ``k``: shape ``[n_tokens, n_kv_heads * head_dim]`` with
        ``k.stride(0) == k_stride_token``.
      * ``*_weight``: shape ``[head_dim]`` (broadcast across heads).
      * Both q/k share the same ``eps`` (Qwen3 invariant).
    """
    pid = tl.program_id(0)

    n_total_heads = n_q_heads + n_kv_heads
    token_id = pid // n_total_heads
    head_id = pid % n_total_heads

    if token_id >= n_tokens:
        return

    is_q = head_id < n_q_heads
    if is_q:
        base = q_ptr + token_id * q_stride_token + head_id * HEAD_DIM
        weight_base = q_weight_ptr
    else:
        local_head = head_id - n_q_heads
        base = k_ptr + token_id * k_stride_token + local_head * HEAD_DIM
        weight_base = k_weight_ptr

    offsets = tl.arange(0, BLOCK)
    mask = offsets < HEAD_DIM

    x = tl.load(base + offsets, mask=mask, other=0.0).to(tl.float32)
    x_sq = x * x
    var = tl.sum(x_sq, axis=0) / HEAD_DIM
    rsqrt = 1.0 / tl.sqrt(var + eps)

    w = tl.load(weight_base + offsets, mask=mask, other=0.0).to(tl.float32)
    out = x * rsqrt * w
    tl.store(base + offsets, out.to(x.dtype), mask=mask)


def fused_qk_norm_inplace(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
) -> None:
    """In-place fused q/k RMS-norm.

    Args:
        q: ``[n_tokens, n_q_heads * head_dim]`` (or 3D
           ``[n_tokens, n_q_heads, head_dim]``). The ``head_dim`` axis
           must be the innermost (stride==1); ``q.stride(0)`` may be
           arbitrary so slices of ``qkv.split(..., dim=-1)`` are
           supported without an extra ``.contiguous()``.
        k: same shape constraints with ``n_kv_heads`` heads.
        q_weight / k_weight: ``[head_dim]``, same dtype as q/k.
        eps: variance epsilon shared by q_norm / k_norm (Qwen3 invariant).
    """
    assert q.dtype == k.dtype == q_weight.dtype == k_weight.dtype
    assert q_weight.numel() == k_weight.numel()
    head_dim = q_weight.numel()
    assert q.shape[-1] % head_dim == 0
    assert k.shape[-1] % head_dim == 0

    n_tokens = q.shape[0]
    if q.dim() == 2:
        n_q_heads = q.shape[-1] // head_dim
    else:
        n_q_heads = q.shape[-2]
    if k.dim() == 2:
        n_kv_heads = k.shape[-1] // head_dim
    else:
        n_kv_heads = k.shape[-2]

    # head_dim must be innermost (stride 1) so head-axis loads are coalesced.
    # token stride is whatever the caller passed -- supports qkv.split slices.
    assert q.stride(-1) == 1
    assert k.stride(-1) == 1
    q_stride_token = q.stride(0)
    k_stride_token = k.stride(0)
    n_total_heads = n_q_heads + n_kv_heads

    BLOCK = triton.next_power_of_2(head_dim)

    grid = (n_tokens * n_total_heads,)
    _fused_qk_norm_kernel[grid](
        q,
        k,
        q_weight,
        k_weight,
        eps,
        n_q_heads,
        n_kv_heads,
        q_stride_token,
        k_stride_token,
        n_tokens,
        HEAD_DIM=head_dim,
        BLOCK=BLOCK,
        num_warps=4 if head_dim <= 128 else 8,
    )
