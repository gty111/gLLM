"""Paged visual-embedding rows for encoder disaggregation.

Both ends of the embedding transfer keep rows in :class:`CacheArena` pages
instead of fixed per-item slots:

* the LM registers an ``mm_embed`` cache type in its model cache arena, so
  embedding rows share (and compete for) the same pages as the KV cache and are
  read by prefill in place;
* the encoder stages ViT output in a small arena of its own over the registered
  send buffer.

An item of ``n`` rows occupies ``ceil(n / rows_per_page)`` pages in any order;
row ``j`` lives at row ``j % rows_per_page`` of page ``pages[j // rows_per_page]``.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch

from gllm.runtime.cache_arena import CacheLayout, CacheTensorLayout

MM_EMBED = "mm_embed"


def rows_per_page(page_bytes: int, feat_dim: int, dtype: torch.dtype) -> int:
    row_bytes = feat_dim * torch.empty((), dtype=dtype).element_size()
    return max(1, page_bytes // row_bytes)


def embed_layout(name: str, feat_dim: int, dtype: torch.dtype, rows: int) -> CacheLayout:
    return CacheLayout(
        name, (CacheTensorLayout("rows", dtype, (rows, feat_dim)),), prefer_high=True
    )


def num_pages(rows: int, per_page: int) -> int:
    return -(-rows // per_page)


def row_index(pages: Sequence[int], rows: int, per_page: int) -> Tuple[List[int], List[int]]:
    """(page, row-in-page) of rows ``[0, rows)`` of an item."""
    page = [pages[j // per_page] for j in range(rows)]
    off = [j % per_page for j in range(rows)]
    return page, off


def copy_runs(
    src_pages: Sequence[int], src_per_page: int, src_lo: int,
    dst_pages: Sequence[int], dst_per_page: int, dst_lo: int,
    rows: int,
) -> List[Tuple[int, int, int, int, int]]:
    """Split a copy of ``rows`` rows between two paged layouts into runs that
    are contiguous on both sides: ``(src_page, src_row, dst_page, dst_row, n)``."""
    runs = []
    i = 0
    while i < rows:
        s, d = src_lo + i, dst_lo + i
        sp, so = divmod(s, src_per_page)
        dp, do = divmod(d, dst_per_page)
        n = min(src_per_page - so, dst_per_page - do, rows - i)
        runs.append((src_pages[sp], so, dst_pages[dp], do, n))
        i += n
    return runs


class PagedRows:
    """A seq's visual rows read in place from arena pages.

    Slicing gathers just the requested rows (one indexing kernel); nothing is
    copied out of the pages ahead of time.
    """

    def __init__(self, pool: torch.Tensor, page: torch.Tensor, off: torch.Tensor):
        self.pool = pool  # [num_pages, rows_per_page, feat_dim]
        self.page = page
        self.off = off

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.page.numel(), self.pool.shape[-1])

    def __len__(self) -> int:
        return self.page.numel()

    def prefix(self, rows: int) -> "PagedRows":
        return PagedRows(self.pool, self.page[:rows], self.off[:rows])

    def __getitem__(self, key):
        rows, rest = (key[0], key[1:]) if isinstance(key, tuple) else (key, ())
        out = self.pool[self.page[rows], self.off[rows]]
        return out[(slice(None),) + rest] if rest else out
