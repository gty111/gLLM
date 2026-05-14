"""
Triton kernels for KV cache operations.

These kernels replace the vllm CUDA cache ops:
- reshape_and_cache_flash: Store K/V tensors into paged KV cache
- concat_and_cache_mla: Store MLA (kv_c + k_pe) into a single paged cache
- gather_and_maybe_dequant_cache: Load from paged cache with optional dequantization

Ported/adapted from sglang's Triton implementations.
"""

import torch
import triton
import triton.language as tl


# =============================================================================
# reshape_and_cache_flash
# =============================================================================


@triton.jit
def _reshape_and_cache_flash_kernel(
    key_ptr,
    value_ptr,
    key_cache_ptr,
    value_cache_ptr,
    slot_mapping_ptr,
    k_scale_ptr,
    v_scale_ptr,
    block_stride,
    key_stride,
    value_stride,
    num_heads,
    head_size,
    block_size,
    HEAD_BLOCK: tl.constexpr,
    BLOCK_D: tl.constexpr,
    USE_SCALE: tl.constexpr,
):
    """
    Triton kernel for reshaping per-token K/V tensors into paged KV cache layout.

    Source layout:
        key/value: [num_tokens, num_heads, head_size]

    Target cache layout:
        cache: [num_blocks, block_size, num_heads, head_size]

    Each Triton program instance handles:
        - one token (program_id(0))
        - one block of heads (program_id(1))
    """
    # program ids
    token_idx = tl.program_id(0)
    head_block_idx = tl.program_id(1)

    # slot mapping
    slot_idx = tl.load(slot_mapping_ptr + token_idx)

    if slot_idx < 0:
        return

    block_idx = slot_idx // block_size
    block_offset = slot_idx % block_size

    # head range
    head_idx = head_block_idx * HEAD_BLOCK + tl.arange(0, HEAD_BLOCK)
    head_mask = head_idx < num_heads

    dim_idx = tl.arange(0, BLOCK_D)

    # shape = [HEAD_BLOCK, BLOCK_D]
    offs = head_idx[:, None] * head_size + dim_idx[None, :]
    mask = head_mask[:, None] & (dim_idx[None, :] < head_size)

    # source load
    src_key = token_idx * key_stride + offs
    src_value = token_idx * value_stride + offs

    k = tl.load(key_ptr + src_key, mask=mask)
    v = tl.load(value_ptr + src_value, mask=mask)

    # optional scale (divide by scale for FP8 cache storage)
    if USE_SCALE:
        k_scale = tl.load(k_scale_ptr)
        v_scale = tl.load(v_scale_ptr)
        k = k / k_scale
        v = v / v_scale

    # target layout: [block_idx, block_offset, head, dim]
    tgt = block_idx * block_stride + block_offset * num_heads * head_size + offs

    tl.store(key_cache_ptr + tgt, k, mask=mask)
    tl.store(value_cache_ptr + tgt, v, mask=mask)


def reshape_and_cache_flash(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> None:
    """
    Store K/V tensors into paged KV cache.

    Args:
        key: [num_tokens, num_heads, head_size]
        value: [num_tokens, num_heads, head_size]
        key_cache: [num_blocks, block_size, num_heads, head_size]
        value_cache: [num_blocks, block_size, num_heads, head_size]
        slot_mapping: [num_tokens] - maps each token to a cache slot
        kv_cache_dtype: dtype string (e.g., "auto", "fp8", "fp8_e4m3")
        k_scale: scalar tensor for key scaling (used with FP8)
        v_scale: scalar tensor for value scaling (used with FP8)
    """
    num_tokens = key.shape[0]
    num_heads = key.shape[1]
    head_size = key.shape[2]

    use_scale = kv_cache_dtype != "auto"

    HEAD_BLOCK = 4
    BLOCK_D = triton.next_power_of_2(head_size)

    grid = (
        num_tokens,
        triton.cdiv(num_heads, HEAD_BLOCK),
    )

    _reshape_and_cache_flash_kernel[grid](
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        k_scale if use_scale else key,  # dummy pointer if not used
        v_scale if use_scale else key,
        key_cache.stride(0),
        key.stride(0),
        value.stride(0),
        num_heads,
        head_size,
        key_cache.shape[1],  # block_size
        HEAD_BLOCK=HEAD_BLOCK,
        BLOCK_D=BLOCK_D,
        USE_SCALE=use_scale,
    )


# =============================================================================
# concat_and_cache_mla
# =============================================================================


@triton.jit
def _concat_and_cache_mla_kernel(
    kv_c_ptr,
    k_pe_ptr,
    kv_cache_ptr,
    slot_mapping_ptr,
    scale_ptr,
    kv_c_stride,
    k_pe_stride,
    cache_stride_block,
    kv_c_dim,
    k_pe_dim,
    total_dim,
    block_size,
    BLOCK_D_C: tl.constexpr,
    BLOCK_D_PE: tl.constexpr,
    USE_SCALE: tl.constexpr,
):
    """
    Triton kernel for concatenating kv_c and k_pe into a single MLA cache entry.

    Source layout:
        kv_c: [num_tokens, kv_lora_rank]
        k_pe: [num_tokens, qk_rope_head_dim]

    Target cache layout:
        kv_cache: [num_blocks, block_size, kv_lora_rank + qk_rope_head_dim]

    Each Triton program instance handles one token.
    """
    token_idx = tl.program_id(0)

    # slot mapping
    slot_idx = tl.load(slot_mapping_ptr + token_idx)

    if slot_idx < 0:
        return

    block_idx = slot_idx // block_size
    block_offset = slot_idx % block_size

    # Load kv_c part
    c_offs = tl.arange(0, BLOCK_D_C)
    c_mask = c_offs < kv_c_dim
    kv_c = tl.load(kv_c_ptr + token_idx * kv_c_stride + c_offs, mask=c_mask)

    # Load k_pe part
    pe_offs = tl.arange(0, BLOCK_D_PE)
    pe_mask = pe_offs < k_pe_dim
    k_pe = tl.load(k_pe_ptr + token_idx * k_pe_stride + pe_offs, mask=pe_mask)

    # Optional scale (for FP8 quantized cache)
    if USE_SCALE:
        scale = tl.load(scale_ptr)
        kv_c = kv_c / scale
        k_pe = k_pe / scale

    # Store concatenated [kv_c | k_pe] to cache
    tgt_base = block_idx * cache_stride_block + block_offset * total_dim

    tl.store(kv_cache_ptr + tgt_base + c_offs, kv_c, mask=c_mask)
    tl.store(kv_cache_ptr + tgt_base + kv_c_dim + pe_offs, k_pe, mask=pe_mask)


def concat_and_cache_mla(
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    scale: torch.Tensor,
) -> None:
    """
    Store MLA kv_c and k_pe concatenated into paged cache.

    Args:
        kv_c: [num_tokens, kv_lora_rank]
        k_pe: [num_tokens, qk_rope_head_dim]
        kv_cache: [num_blocks, block_size, kv_lora_rank + qk_rope_head_dim]
        slot_mapping: [num_tokens]
        kv_cache_dtype: dtype string
        scale: scalar tensor for cache scaling
    """
    num_tokens = kv_c.shape[0]
    kv_c_dim = kv_c.shape[-1]
    k_pe_dim = k_pe.shape[-1]
    total_dim = kv_c_dim + k_pe_dim
    block_size = kv_cache.shape[1]

    use_scale = kv_cache_dtype != "auto"

    BLOCK_D_C = triton.next_power_of_2(kv_c_dim)
    BLOCK_D_PE = triton.next_power_of_2(k_pe_dim)

    grid = (num_tokens,)

    _concat_and_cache_mla_kernel[grid](
        kv_c,
        k_pe,
        kv_cache,
        slot_mapping,
        scale if use_scale else kv_c,  # dummy pointer if not used
        kv_c.stride(0),
        k_pe.stride(0),
        kv_cache.stride(0),
        kv_c_dim,
        k_pe_dim,
        total_dim,
        block_size,
        BLOCK_D_C=BLOCK_D_C,
        BLOCK_D_PE=BLOCK_D_PE,
        USE_SCALE=use_scale,
    )


# =============================================================================
# gather_and_maybe_dequant_cache
# =============================================================================


@triton.jit
def _gather_and_maybe_dequant_cache_kernel(
    src_cache_ptr,
    dst_ptr,
    block_table_ptr,
    cu_seq_lens_ptr,
    seq_starts_ptr,
    scale_ptr,
    cache_stride_block,
    dst_stride,
    block_table_stride,
    total_dim,
    block_size,
    BLOCK_D: tl.constexpr,
    USE_SCALE: tl.constexpr,
    HAS_SEQ_STARTS: tl.constexpr,
):
    """
    Triton kernel for gathering entries from paged MLA cache into a contiguous
    destination buffer. Optionally dequantizes (multiplies by scale).

    Source cache layout:
        src_cache: [num_blocks, block_size, total_dim]

    Destination layout:
        dst: [total_gathered_tokens, total_dim]

    Each Triton program instance handles:
        - one token position within a sequence (program_id(0))
        - one batch item (program_id(1))
    """
    pos_in_seq = tl.program_id(0)
    batch_idx = tl.program_id(1)

    # Determine the sequence range for this batch item
    seq_start = tl.load(cu_seq_lens_ptr + batch_idx)
    seq_end = tl.load(cu_seq_lens_ptr + batch_idx + 1)
    seq_len = seq_end - seq_start

    if pos_in_seq >= seq_len:
        return

    # Determine the actual position in the KV cache
    if HAS_SEQ_STARTS:
        actual_pos = tl.load(seq_starts_ptr + batch_idx) + pos_in_seq
    else:
        actual_pos = pos_in_seq

    # Look up the block in the block table
    block_idx_in_table = actual_pos // block_size
    block_offset = actual_pos % block_size

    physical_block = tl.load(
        block_table_ptr + batch_idx * block_table_stride + block_idx_in_table
    )

    # Load from cache
    dim_offs = tl.arange(0, BLOCK_D)
    dim_mask = dim_offs < total_dim

    src_offset = (
        physical_block * cache_stride_block + block_offset * total_dim + dim_offs
    )
    data = tl.load(src_cache_ptr + src_offset, mask=dim_mask)

    # Optional dequantization
    if USE_SCALE:
        scale = tl.load(scale_ptr)
        data = data * scale

    # Store to destination
    dst_offset = (seq_start + pos_in_seq) * dst_stride + dim_offs
    tl.store(dst_ptr + dst_offset, data, mask=dim_mask)


def gather_and_maybe_dequant_cache(
    src_cache: torch.Tensor,
    dst: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    batch_size: int,
    kv_cache_dtype: str,
    scale: torch.Tensor,
    seq_starts: torch.Tensor | None = None,
) -> None:
    """
    Gather entries from paged MLA cache into contiguous destination.

    Args:
        src_cache: [num_blocks, block_size, total_dim]
        dst: [total_tokens, total_dim] - output buffer
        block_table: [batch_size, max_blocks_per_seq]
        cu_seq_lens: [batch_size + 1] - cumulative sequence lengths
        batch_size: number of sequences
        kv_cache_dtype: dtype string ("auto" means no dequant)
        scale: scalar tensor for dequantization
        seq_starts: [batch_size] - optional starting positions within each sequence
    """
    total_dim = src_cache.shape[-1]
    block_size = src_cache.shape[1]

    use_scale = kv_cache_dtype != "auto"

    BLOCK_D = triton.next_power_of_2(total_dim)

    # Determine max sequence length for grid sizing
    max_seq_len = int((cu_seq_lens[1:] - cu_seq_lens[:-1]).max().item())

    grid = (max_seq_len, batch_size)

    _gather_and_maybe_dequant_cache_kernel[grid](
        src_cache,
        dst,
        block_table,
        cu_seq_lens,
        seq_starts if seq_starts is not None else cu_seq_lens,  # dummy
        scale if use_scale else src_cache,  # dummy pointer if not used
        src_cache.stride(0),
        dst.stride(0),
        block_table.stride(0),
        total_dim,
        block_size,
        BLOCK_D=BLOCK_D,
        USE_SCALE=use_scale,
        HAS_SEQ_STARTS=(seq_starts is not None),
    )
