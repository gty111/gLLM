"""
Custom Triton kernel for batched rotary embedding with per-token cos_sin_cache offsets.

This replaces vllm's batched_rotary_embedding CUDA kernel which supports
multiple RoPE scaling factors applied simultaneously to different tokens
within a batch (each token uses a different offset into the cos_sin_cache).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _batched_rotary_embedding_kernel(
    positions_ptr,
    query_ptr,
    key_ptr,
    cos_sin_cache_ptr,
    cos_sin_cache_offsets_ptr,
    query_stride,
    key_stride,
    num_heads,
    num_kv_heads,
    head_size,
    rot_dim,
    cache_stride,
    IS_NEOX: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Triton kernel for batched rotary positional embedding with per-token offsets.

    For each token, the effective position in the cos_sin_cache is:
        effective_pos = positions[token] + cos_sin_cache_offsets[token]

    The cos_sin_cache is laid out as:
        [max_positions, rot_dim] where first half is cos, second half is sin

    This kernel applies RoPE in-place to both query and key tensors.
    Supports both NeoX-style (split half) and GPT-J-style (interleaved) rotation.
    """
    token_idx = tl.program_id(0)
    head_block_idx = tl.program_id(1)
    is_key = tl.program_id(2)  # 0 = query, 1 = key

    # Determine number of heads for this pass
    total_heads = tl.where(is_key == 0, num_heads, num_kv_heads)

    head_idx = head_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    head_mask = head_idx < total_heads

    # Compute effective position
    pos = tl.load(positions_ptr + token_idx)
    offset = tl.load(cos_sin_cache_offsets_ptr + token_idx)
    effective_pos = pos + offset

    # Load cos and sin from cache
    # cos_sin_cache layout: [max_positions, rot_dim]
    # First rot_dim//2 elements are cos, next rot_dim//2 are sin
    half_rot_dim = rot_dim // 2
    dim_offs = tl.arange(0, BLOCK_D)
    dim_mask = dim_offs < half_rot_dim

    cos_ptr = cos_sin_cache_ptr + effective_pos * cache_stride + dim_offs
    sin_ptr = cos_sin_cache_ptr + effective_pos * cache_stride + half_rot_dim + dim_offs

    cos_val = tl.load(cos_ptr, mask=dim_mask, other=0.0)
    sin_val = tl.load(sin_ptr, mask=dim_mask, other=0.0)

    # Select tensor pointer and stride
    if is_key == 0:
        tensor_ptr = query_ptr
        tensor_stride = query_stride
    else:
        tensor_ptr = key_ptr
        tensor_stride = key_stride

    # Process each head in this block
    # For NeoX-style: x = [x1, x2], rotated = [x1*cos - x2*sin, x2*cos + x1*sin]
    # For GPT-J-style: x = [x0, x1, x2, x3, ...], pairs are (x0,x1), (x2,x3), ...

    # We process one head at a time within the block
    for h_offset in range(BLOCK_H):
        h = head_block_idx * BLOCK_H + h_offset
        if h >= total_heads:
            break

        base = token_idx * tensor_stride + h * head_size

        if IS_NEOX:
            # NeoX rotation: split into first half and second half
            x1 = tl.load(tensor_ptr + base + dim_offs, mask=dim_mask, other=0.0)
            x2 = tl.load(
                tensor_ptr + base + half_rot_dim + dim_offs, mask=dim_mask, other=0.0
            )

            new_x1 = x1 * cos_val - x2 * sin_val
            new_x2 = x2 * cos_val + x1 * sin_val

            tl.store(tensor_ptr + base + dim_offs, new_x1, mask=dim_mask)
            tl.store(
                tensor_ptr + base + half_rot_dim + dim_offs, new_x2, mask=dim_mask
            )
        else:
            # GPT-J rotation: interleaved pairs (x0, x1), (x2, x3), ...
            pair_offs = tl.arange(0, BLOCK_D)
            pair_mask = pair_offs < half_rot_dim

            even_offs = pair_offs * 2
            odd_offs = pair_offs * 2 + 1

            x_even = tl.load(tensor_ptr + base + even_offs, mask=pair_mask, other=0.0)
            x_odd = tl.load(tensor_ptr + base + odd_offs, mask=pair_mask, other=0.0)

            new_even = x_even * cos_val - x_odd * sin_val
            new_odd = x_odd * cos_val + x_even * sin_val

            tl.store(tensor_ptr + base + even_offs, new_even, mask=pair_mask)
            tl.store(tensor_ptr + base + odd_offs, new_odd, mask=pair_mask)


def batched_rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    rot_dim: int,
    cos_sin_cache_offsets: torch.Tensor,
) -> None:
    """
    Apply rotary positional embedding with per-token cache offsets.

    This handles the case where different tokens in the batch may use different
    RoPE scaling factors (each having its own section in the cos_sin_cache,
    accessed via cos_sin_cache_offsets).

    Args:
        positions: [num_tokens] - position indices
        query: [num_tokens, num_heads * head_size] - query tensor (modified in-place)
        key: [num_tokens, num_kv_heads * head_size] - key tensor (modified in-place)
        head_size: size of each attention head
        cos_sin_cache: [max_positions, rot_dim] - precomputed cos/sin values
        is_neox: whether to use NeoX-style (True) or GPT-J-style (False) rotation
        rot_dim: number of dimensions to rotate
        cos_sin_cache_offsets: [num_tokens] - per-token offset into cos_sin_cache
    """
    num_tokens = positions.shape[0]
    num_heads = query.shape[-1] // head_size
    num_kv_heads = key.shape[-1] // head_size

    half_rot_dim = rot_dim // 2
    BLOCK_D = triton.next_power_of_2(half_rot_dim)
    BLOCK_H = 1  # Process one head at a time for simplicity

    grid = (
        num_tokens,
        triton.cdiv(max(num_heads, num_kv_heads), BLOCK_H),
        2,  # 0=query, 1=key
    )

    _batched_rotary_embedding_kernel[grid](
        positions,
        query,
        key,
        cos_sin_cache,
        cos_sin_cache_offsets,
        query.stride(0),
        key.stride(0),
        num_heads,
        num_kv_heads,
        head_size,
        rot_dim,
        cos_sin_cache.stride(0),
        IS_NEOX=is_neox,
        BLOCK_H=BLOCK_H,
        BLOCK_D=BLOCK_D,
    )
