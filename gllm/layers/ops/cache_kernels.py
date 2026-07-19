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
# concat_and_cache_mla_fp8  (native FP8 MLA cache, DeepSeek Sparse Attention)
# =============================================================================
#
# FlashMLA's sparse decode kernel reads a paged FP8 latent cache in a fixed
# 656-byte-per-token layout (for the standard MLA dims kv_lora_rank=512,
# qk_rope_head_dim=64):
#
#     bytes [0   : 512]  nope  -> FP8 e4m3            (512 vals x 1 byte)
#     bytes [512 : 528]  scales-> 4 x float32          (one per 128-wide tile)
#     bytes [528 : 656]  rope  -> bf16                 (64 vals x 2 bytes)
#
# i.e. the cache tensor is ``[num_blocks, block_size, 1, 656]`` viewed as
# float8_e4m3fn. This kernel writes that layout directly from the bf16 latent
# (kv_c) + rope (k_pe) at each token's slot -- a *native* FP8 store, no separate
# bf16 cache and no runtime dequant. It mirrors sglang's ``_quantize_k_cache_ref``
# (dsa/quant_k_cache.py): per-tile scale = max(|x|)/448, quantized = x/scale.


@triton.jit
def _concat_and_cache_mla_fp8_kernel(
    kv_c_ptr,          # [num_tokens, kv_lora_rank] bf16
    k_pe_ptr,          # [num_tokens, qk_rope_head_dim] bf16
    cache_ptr,         # [num_blocks, block_size, dim_q_bytes] float8_e4m3fn
    slot_mapping_ptr,  # [num_tokens] int
    kv_c_stride,
    k_pe_stride,
    cache_stride_block,
    kv_lora_rank: tl.constexpr,   # 512
    qk_rope_head_dim: tl.constexpr,  # 64
    tile_size: tl.constexpr,      # 128
    num_tiles: tl.constexpr,      # 4
    block_size: tl.constexpr,
):
    token_idx = tl.program_id(0)
    slot_idx = tl.load(slot_mapping_ptr + token_idx)
    if slot_idx < 0:
        return
    block_idx = slot_idx // block_size
    block_offset = slot_idx % block_size
    # Byte offset of this token's row within the fp8 cache (element == 1 byte).
    row = block_idx * cache_stride_block + block_offset * (
        kv_lora_rank + num_tiles * 4 + qk_rope_head_dim * 2
    )

    # ---- nope: quantize per 128-tile ----
    for t in tl.static_range(num_tiles):
        offs = t * tile_size + tl.arange(0, tile_size)
        x = tl.load(kv_c_ptr + token_idx * kv_c_stride + offs).to(tl.float32)
        amax = tl.max(tl.abs(x))
        scale = amax / 448.0
        # avoid div-by-zero: an all-zero tile yields scale 0 -> store zeros.
        inv = tl.where(scale > 0, 1.0 / scale, 0.0)
        q = (x * inv).to(cache_ptr.dtype.element_ty)
        tl.store(cache_ptr + row + offs, q)
        # scale (float32) goes into bytes [512 + t*4 : +4]; write via a
        # float32-typed view by computing the element index in a f32 grid.
        # cache is fp8 (1 byte); reinterpret the 4 scale bytes by storing to a
        # float32 pointer aliased at the right byte offset.
        scale_byte = kv_lora_rank + t * 4
        tl.store(
            (cache_ptr + row + scale_byte).cast(tl.pointer_type(tl.float32)),
            scale,
        )

    # ---- rope: store as bf16 (2 bytes each) ----
    rope_offs = tl.arange(0, qk_rope_head_dim)
    rope = tl.load(k_pe_ptr + token_idx * k_pe_stride + rope_offs).to(tl.bfloat16)
    rope_byte = kv_lora_rank + num_tiles * 4
    tl.store(
        (cache_ptr + row + rope_byte).cast(tl.pointer_type(tl.bfloat16)) + rope_offs,
        rope,
    )


def concat_and_cache_mla_fp8(
    kv_c: torch.Tensor,       # [num_tokens, kv_lora_rank] (bf16)
    k_pe: torch.Tensor,       # [num_tokens, qk_rope_head_dim] (bf16)
    fp8_cache: torch.Tensor,  # [num_blocks, block_size, 1, dim_q] float8_e4m3fn
    slot_mapping: torch.Tensor,
    tile_size: int = 128,
) -> None:
    """Native FP8 MLA cache store (DeepSeek Sparse Attention).

    Packs each token's latent into FlashMLA's FP8 sparse-decode layout
    (``kv_lora_rank`` FP8 + ``kv_lora_rank/tile_size`` fp32 scales + rope bf16)
    at its ``slot_mapping`` slot. See module comment for the byte layout.
    """
    num_tokens, kv_lora_rank = kv_c.shape
    qk_rope_head_dim = k_pe.shape[-1]
    assert kv_lora_rank % tile_size == 0
    num_tiles = kv_lora_rank // tile_size
    # fp8_cache flattened last dim is the per-token byte count.
    block_size = fp8_cache.shape[1]
    dim_q = fp8_cache.shape[-1]
    assert dim_q == kv_lora_rank + num_tiles * 4 + qk_rope_head_dim * 2, (
        f"fp8 MLA cache last dim {dim_q} != expected "
        f"{kv_lora_rank + num_tiles * 4 + qk_rope_head_dim * 2}"
    )
    cache_stride_block = block_size * dim_q  # bytes per block (fp8 = 1 byte)

    _concat_and_cache_mla_fp8_kernel[(num_tokens,)](
        kv_c,
        k_pe,
        fp8_cache,
        slot_mapping,
        kv_c.stride(0),
        k_pe.stride(0),
        cache_stride_block,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        tile_size=tile_size,
        num_tiles=num_tiles,
        block_size=block_size,
    )


# =============================================================================
# gather_and_dequant_mla_fp8  (read back native FP8 MLA cache -> bf16)
# =============================================================================
#
# Inverse of ``concat_and_cache_mla_fp8``: gathers a sequence's tokens from the
# 656-byte FP8-packed paged cache and *dequantizes* them back to a contiguous
# bf16 ``[gathered_tokens, kv_lora_rank + qk_rope_head_dim]`` buffer, which the
# MLA prefill context path (``_compute_prefill_context``) consumes exactly as it
# would the bf16 latent cache. Symmetric with the write kernel above:
#   nope[i] = fp8[i].to(f32) * scale[i // tile_size]   (scale = amax/448)
#   rope    = bf16 read verbatim


@triton.jit
def _gather_and_dequant_mla_fp8_kernel(
    src_cache_ptr,     # [num_blocks, block_size, dim_q_bytes] float8_e4m3fn
    dst_ptr,           # [gathered_tokens, kv_lora_rank + qk_rope_head_dim] bf16
    block_table_ptr,
    cu_seq_lens_ptr,
    seq_starts_ptr,
    cache_stride_block,
    dst_stride,
    block_table_stride,
    kv_lora_rank: tl.constexpr,      # 512
    qk_rope_head_dim: tl.constexpr,  # 64
    tile_size: tl.constexpr,         # 128
    num_tiles: tl.constexpr,         # 4
    block_size: tl.constexpr,
    HAS_SEQ_STARTS: tl.constexpr,
):
    pos_in_seq = tl.program_id(0)
    batch_idx = tl.program_id(1)

    seq_start = tl.load(cu_seq_lens_ptr + batch_idx)
    seq_end = tl.load(cu_seq_lens_ptr + batch_idx + 1)
    if pos_in_seq >= seq_end - seq_start:
        return

    if HAS_SEQ_STARTS:
        actual_pos = tl.load(seq_starts_ptr + batch_idx) + pos_in_seq
    else:
        actual_pos = pos_in_seq

    block_idx_in_table = actual_pos // block_size
    block_offset = actual_pos % block_size
    physical_block = tl.load(
        block_table_ptr + batch_idx * block_table_stride + block_idx_in_table
    )

    dim_q = kv_lora_rank + num_tiles * 4 + qk_rope_head_dim * 2
    row = physical_block * cache_stride_block + block_offset * dim_q
    dst_row = (seq_start + pos_in_seq) * dst_stride

    # ---- nope: per-128-tile dequant (fp8 * per-tile fp32 scale) ----
    for t in tl.static_range(num_tiles):
        offs = t * tile_size + tl.arange(0, tile_size)
        q = tl.load(src_cache_ptr + row + offs).to(tl.float32)
        scale_byte = kv_lora_rank + t * 4
        scale = tl.load(
            (src_cache_ptr + row + scale_byte).cast(tl.pointer_type(tl.float32))
        )
        tl.store(dst_ptr + dst_row + offs, (q * scale).to(dst_ptr.dtype.element_ty))

    # ---- rope: bf16 verbatim ----
    rope_offs = tl.arange(0, qk_rope_head_dim)
    rope_byte = kv_lora_rank + num_tiles * 4
    rope = tl.load(
        (src_cache_ptr + row + rope_byte).cast(tl.pointer_type(tl.bfloat16)) + rope_offs
    )
    tl.store(
        dst_ptr + dst_row + kv_lora_rank + rope_offs,
        rope.to(dst_ptr.dtype.element_ty),
    )


def gather_and_dequant_mla_fp8(
    src_cache: torch.Tensor,   # [num_blocks, block_size, 1, dim_q] float8_e4m3fn
    dst: torch.Tensor,         # [total_tokens, kv_lora_rank + qk_rope_head_dim] bf16
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    batch_size: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    seq_starts: torch.Tensor | None = None,
    tile_size: int = 128,
) -> None:
    """Gather + dequantize the native FP8 MLA cache into a bf16 buffer.

    Inverse of :func:`concat_and_cache_mla_fp8`. ``dst``'s last dim is
    ``kv_lora_rank + qk_rope_head_dim`` (the bf16 latent+rope layout the MLA
    prefill path expects); the per-tile scales embedded in the packed cache are
    applied here, so no external scale is needed.
    """
    assert kv_lora_rank % tile_size == 0
    num_tiles = kv_lora_rank // tile_size
    dim_q = src_cache.shape[-1]
    block_size = src_cache.shape[1]
    assert dim_q == kv_lora_rank + num_tiles * 4 + qk_rope_head_dim * 2, (
        f"fp8 MLA cache last dim {dim_q} != expected "
        f"{kv_lora_rank + num_tiles * 4 + qk_rope_head_dim * 2}"
    )
    assert dst.shape[-1] == kv_lora_rank + qk_rope_head_dim
    # Bytes per block: fp8 element == 1 byte, and the cache is
    # [num_blocks, block_size, 1, dim_q] so stride(0) == block_size * dim_q.
    cache_stride_block = src_cache.stride(0)

    max_seq_len = int((cu_seq_lens[1:] - cu_seq_lens[:-1]).max().item())
    grid = (max_seq_len, batch_size)

    _gather_and_dequant_mla_fp8_kernel[grid](
        src_cache,
        dst,
        block_table,
        cu_seq_lens,
        seq_starts if seq_starts is not None else cu_seq_lens,  # dummy
        cache_stride_block,
        dst.stride(0),
        block_table.stride(0),
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        tile_size=tile_size,
        num_tiles=num_tiles,
        block_size=block_size,
        HAS_SEQ_STARTS=seq_starts is not None,
    )


# =============================================================================
# dequant_mla_fp8_flat  (whole-cache FP8 -> bf16 latent, physical-slot indexed)
# =============================================================================
#
# Dequantizes the 656-byte FP8-packed MLA cache into a flat, physical-slot-indexed
# bf16 latent buffer ``[num_slots, kv_lora_rank + qk_rope_head_dim]``. Unlike
# ``gather_and_dequant_mla_fp8`` (which packs a batch's sequences contiguously by
# ``cu_seq_lens``), this preserves the physical slot ordering so absolute cache
# slots index the output directly -- exactly what DeepSeek Sparse Attention's
# prefill kernel needs, since its top-k ``indices`` are absolute physical slots
# (see ``MLAAttention._forward_prefill_sparse``). One program per slot.


@triton.jit
def _dequant_mla_fp8_flat_kernel(
    src_ptr,           # [num_slots, dim_q] float8_e4m3fn (packed 656B row)
    dst_ptr,           # [num_slots, kv_lora_rank + qk_rope_head_dim] bf16
    src_stride,
    dst_stride,
    kv_lora_rank: tl.constexpr,      # 512
    qk_rope_head_dim: tl.constexpr,  # 64
    tile_size: tl.constexpr,         # 128
    num_tiles: tl.constexpr,         # 4
):
    slot = tl.program_id(0)
    row = slot * src_stride
    dst_row = slot * dst_stride

    # ---- nope: per-128-tile dequant (fp8 * per-tile fp32 scale) ----
    for t in tl.static_range(num_tiles):
        offs = t * tile_size + tl.arange(0, tile_size)
        q = tl.load(src_ptr + row + offs).to(tl.float32)
        scale_byte = kv_lora_rank + t * 4
        scale = tl.load(
            (src_ptr + row + scale_byte).cast(tl.pointer_type(tl.float32))
        )
        tl.store(dst_ptr + dst_row + offs, (q * scale).to(dst_ptr.dtype.element_ty))

    # ---- rope: bf16 verbatim ----
    rope_offs = tl.arange(0, qk_rope_head_dim)
    rope_byte = kv_lora_rank + num_tiles * 4
    rope = tl.load(
        (src_ptr + row + rope_byte).cast(tl.pointer_type(tl.bfloat16)) + rope_offs
    )
    tl.store(
        dst_ptr + dst_row + kv_lora_rank + rope_offs,
        rope.to(dst_ptr.dtype.element_ty),
    )


def dequant_mla_fp8_flat(
    src_cache: torch.Tensor,   # [num_blocks, block_size, 1, dim_q] float8_e4m3fn
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    tile_size: int = 128,
) -> torch.Tensor:
    """Dequantize the whole FP8-packed MLA cache to a flat bf16 latent buffer.

    Returns ``[num_slots, kv_lora_rank + qk_rope_head_dim]`` bf16 where
    ``num_slots == num_blocks * block_size``, so an absolute physical cache slot
    indexes it directly (matching the bf16 latent cache's flat view). Used by
    the DSA sparse-prefill path when the MLA cache is stored FP8-packed.
    """
    assert kv_lora_rank % tile_size == 0
    num_tiles = kv_lora_rank // tile_size
    dim_q = src_cache.shape[-1]
    assert dim_q == kv_lora_rank + num_tiles * 4 + qk_rope_head_dim * 2, (
        f"fp8 MLA cache last dim {dim_q} != expected "
        f"{kv_lora_rank + num_tiles * 4 + qk_rope_head_dim * 2}"
    )
    src = src_cache.reshape(-1, dim_q)  # [num_slots, dim_q]
    num_slots = src.shape[0]
    dst = torch.empty(
        (num_slots, kv_lora_rank + qk_rope_head_dim),
        dtype=torch.bfloat16,
        device=src_cache.device,
    )
    _dequant_mla_fp8_flat_kernel[(num_slots,)](
        src,
        dst,
        src.stride(0),
        dst.stride(0),
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        tile_size=tile_size,
        num_tiles=num_tiles,
    )
    return dst


# =============================================================================
# dequant_mla_fp8_slots  (dequant only referenced slots -> full-width bf16 buf)
# =============================================================================
#
# Gather-only variant of ``dequant_mla_fp8_flat``: dequantizes ONLY the physical
# slots listed in ``slot_ids`` (the unique slots the DSA prefill top-k actually
# references) into a full-width ``[num_slots, dim]`` bf16 buffer. Unreferenced
# rows are left uninitialized -- the sparse kernel never reads them -- so absolute
# physical slots still index the output directly (NO index remapping needed),
# while the dequant compute scales with the referenced-slot count, not the whole
# cache. One program per referenced slot.


@triton.jit
def _dequant_mla_fp8_slots_kernel(
    src_ptr,           # [num_slots, dim_q] float8_e4m3fn
    dst_ptr,           # [num_slots, kv_lora_rank + qk_rope_head_dim] bf16
    slot_ids_ptr,      # [num_ref] int32 physical slots to dequant
    src_stride,
    dst_stride,
    kv_lora_rank: tl.constexpr,
    qk_rope_head_dim: tl.constexpr,
    tile_size: tl.constexpr,
    num_tiles: tl.constexpr,
):
    i = tl.program_id(0)
    slot = tl.load(slot_ids_ptr + i)
    if slot < 0:
        return
    row = slot * src_stride
    dst_row = slot * dst_stride

    for t in tl.static_range(num_tiles):
        offs = t * tile_size + tl.arange(0, tile_size)
        q = tl.load(src_ptr + row + offs).to(tl.float32)
        scale_byte = kv_lora_rank + t * 4
        scale = tl.load(
            (src_ptr + row + scale_byte).cast(tl.pointer_type(tl.float32))
        )
        tl.store(dst_ptr + dst_row + offs, (q * scale).to(dst_ptr.dtype.element_ty))

    rope_offs = tl.arange(0, qk_rope_head_dim)
    rope_byte = kv_lora_rank + num_tiles * 4
    rope = tl.load(
        (src_ptr + row + rope_byte).cast(tl.pointer_type(tl.bfloat16)) + rope_offs
    )
    tl.store(
        dst_ptr + dst_row + kv_lora_rank + rope_offs,
        rope.to(dst_ptr.dtype.element_ty),
    )


def dequant_mla_fp8_slots(
    src_cache: torch.Tensor,   # [num_blocks, block_size, 1, dim_q] float8_e4m3fn
    slot_ids: torch.Tensor,    # [num_ref] int32/int64 physical slots (>=0)
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    tile_size: int = 128,
) -> torch.Tensor:
    """Dequant only ``slot_ids`` of the FP8-packed MLA cache -> flat bf16 buffer.

    Like :func:`dequant_mla_fp8_flat` but only fills the referenced rows (the
    DSA prefill top-k's unique slots), so dequant cost scales with usage not
    cache size. Returns ``[num_slots, kv_lora_rank + qk_rope_head_dim]`` bf16;
    absolute physical slots index it directly (unreferenced rows are never read).
    """
    assert kv_lora_rank % tile_size == 0
    num_tiles = kv_lora_rank // tile_size
    dim_q = src_cache.shape[-1]
    assert dim_q == kv_lora_rank + num_tiles * 4 + qk_rope_head_dim * 2
    src = src_cache.reshape(-1, dim_q)  # [num_slots, dim_q]
    num_slots = src.shape[0]
    slot_ids = slot_ids.to(torch.int32).contiguous()
    dst = torch.empty(
        (num_slots, kv_lora_rank + qk_rope_head_dim),
        dtype=torch.bfloat16,
        device=src_cache.device,
    )
    if slot_ids.numel() > 0:
        _dequant_mla_fp8_slots_kernel[(slot_ids.numel(),)](
            src,
            dst,
            slot_ids,
            src.stride(0),
            dst.stride(0),
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            tile_size=tile_size,
            num_tiles=num_tiles,
        )
    return dst


# =============================================================================
# store_index_k_fp8  (write DSA indexer key into the paged FP8 index-K cache)
# =============================================================================
#
# Quantizes each token's index key ``[index_head_dim]`` (bf16) to e4m3 + one fp32
# scale (amax/448) and writes it into the block-contiguous paged FP8 index cache
# that ``deep_gemm.fp8_paged_mqa_logits`` reads. Per page (page_size tokens) the
# bytes are laid out as ``[page_size*D fp8]`` then ``[page_size*(D/128)*4 scale]``
# -- NOT per-token interleaved. One program per token, addressed by slot_mapping.


@triton.jit
def _store_index_k_fp8_kernel(
    idx_k_ptr,          # [num_tokens, D] bf16
    cache_ptr,          # [num_pages, page_size*(D + n_sf*4)] uint8
    slot_mapping_ptr,   # [num_tokens] int
    idx_k_stride,
    cache_page_stride,
    page_size: tl.constexpr,
    D: tl.constexpr,          # index_head_dim (128)
    n_sf: tl.constexpr,       # scales per token (D // 128 = 1)
):
    t = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + t)
    if slot < 0:
        return
    page = slot // page_size
    off = slot % page_size
    page_base = page * cache_page_stride

    offs = tl.arange(0, D)
    x = tl.load(idx_k_ptr + t * idx_k_stride + offs).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x)), 1e-4)
    scale = amax / 448.0
    # Quantize to e4m3 and store the fp8 BYTE PATTERN into the uint8 cache: cast
    # the write pointer to float8 so the fp8 bits are written verbatim (a plain
    # ``.to(uint8)`` would truncate the fp8 *value* to an integer instead).
    q = tl.clamp(x / scale, -448.0, 448.0).to(tl.float8e4nv)
    tl.store(
        (cache_ptr + page_base + off * D).cast(tl.pointer_type(tl.float8e4nv)) + offs,
        q,
    )
    # scale region: starts at page_size*D, one fp32 per token (n_sf==1).
    scale_byte = page_size * D + off * (n_sf * 4)
    tl.store(
        (cache_ptr + page_base + scale_byte).cast(tl.pointer_type(tl.float32)),
        scale,
    )


def store_index_k_fp8(
    idx_k: torch.Tensor,        # [num_tokens, index_head_dim] bf16
    cache: torch.Tensor,        # [num_pages, page_size*(D + n_sf*4)] uint8
    slot_mapping: torch.Tensor,
    page_size: int,
    index_head_dim: int,
    tile_size: int = 128,
) -> None:
    """Quantize + write the DSA indexer key into the paged FP8 index cache."""
    assert index_head_dim % tile_size == 0
    n_sf = index_head_dim // tile_size
    num_tokens = idx_k.shape[0]
    if num_tokens == 0:
        return
    _store_index_k_fp8_kernel[(num_tokens,)](
        idx_k,
        cache,
        slot_mapping,
        idx_k.stride(0),
        cache.stride(0),
        page_size=page_size,
        D=index_head_dim,
        n_sf=n_sf,
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
