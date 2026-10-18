"""Masked row scatter into the DeepSeek-V4 paged compressed caches.

A decode step commits a freshly pooled row only for the requests that just
closed a compression group. Expressed in PyTorch that is a read-modify-write::

    old = cache[page, row]
    cache[page, row] = torch.where(boundary[:, None], new, old)

which costs a gather, a select and a scatter -- and the gather reads rows it is
about to overwrite. The main and index caches each do this once per layer, so
41 layers pay it 82 times per step for a handful of 576-wide rows.

Writing only the selected rows collapses that to one launch and removes the
read entirely.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _scatter_rows_where_kernel(
    cache_ptr,        # (pages, rows_per_page, D)
    page_ptr,         # (B,) int64
    row_ptr,          # (B,) int64
    src_ptr,          # (B, D)
    mask_ptr,         # (B,) bool
    rows_per_page,
    D,
    BLOCK_D: tl.constexpr,
):
    b = tl.program_id(0)
    if tl.load(mask_ptr + b) == 0:
        return

    offs = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    m = offs < D
    page = tl.load(page_ptr + b).to(tl.int64)
    row = tl.load(row_ptr + b).to(tl.int64)
    dst = cache_ptr + (page * rows_per_page + row) * D + offs
    tl.store(dst, tl.load(src_ptr + b.to(tl.int64) * D + offs, mask=m), mask=m)


def scatter_rows_where(
    cache: torch.Tensor,
    pages: torch.Tensor,
    rows: torch.Tensor,
    src: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    """``cache[pages[b], rows[b]] = src[b]`` wherever ``mask[b]`` holds."""
    batch, width = src.shape
    if batch == 0:
        return
    block = min(triton.next_power_of_2(width), 1024)
    _scatter_rows_where_kernel[(batch, triton.cdiv(width, block))](
        cache,
        pages,
        rows,
        src,
        mask,
        cache.shape[1],
        width,
        BLOCK_D=block,
        num_warps=4,
    )


__all__ = ["scatter_rows_where"]
