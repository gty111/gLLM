"""Encoder-disaggregation state for :class:`gllm.runtime.model_runner.ModelRunner`.

:class:`DisaggMixin` is mixed into ``ModelRunner``. It owns
``self.disagg_embeds`` -- per-seq readiness of the visual embeddings of seqs
admitted before all their embeddings arrived -- which the LM disagg manager
updates, the scheduler reads for gate B and the multimodal embed path consumes.

The embeddings themselves stay where the encoder wrote them: in ``mm_embed``
pages of this rank's model cache arena. Prefill gathers the rows it needs
straight from those pages; the pages are released (through the coordinator's
fanned-out events, so every TP rank frees in the same iteration) once the seq's
prefill no longer needs them.
"""

from typing import Callable, Dict, List, Optional, Tuple

import torch
from attr import Factory, dataclass

from gllm.disagg.paging import MM_EMBED, PagedRows, row_index
from gllm.runtime.sequence import GenerationSequence


@dataclass
class DisaggSeqState:
    """Per-seq encoder-disaggregation overlap state.

    Owned by the :class:`ModelRunner` (keyed by ``seq_id``) so it is immune to
    the scheduler's chunked-prefill ``deepcopy`` of the :class:`GenerationSequence`. The
    LM disagg manager advances ``item_rows[i]`` (and flips ``item_ready[i]``)
    as each item's rows land over NIXL; the scheduler reads them for the
    two-layer prefill gate and the model runner gathers the ready rows from
    ``visual_rows``.

    Items are stored in **image-then-video order** (the order
    ``model.embed_multimodal`` returns its tuple in). Each carries its
    ``[span_start, span_end)`` in the *expanded* token sequence so gate B and
    the ready-prefix embed can be computed.
    """

    num_items: int
    item_span: List[Tuple[int, int]]  # ordered: (start, end) in tokens
    item_modality: List[str]
    item_ready: List[bool]
    image_grid_thw: Optional[torch.Tensor]
    video_grid_thw: Optional[torch.Tensor]
    input_ids_cpu: torch.Tensor  # full expanded prompt ids (cpu)
    is_multimodal_cpu: torch.Tensor  # full mask (cpu)
    prompt_positions: torch.Tensor  # full-prompt mrope positions
    mrope_position_delta: torch.Tensor
    prompt_len: int
    # Rows ``[0, item_rows[i])`` of item ``i`` have landed (streamed items
    # advance segment by segment).
    item_rows: List[int] = Factory(list)
    # Arena ``mm_embed`` pages holding each item's rows, in row order.
    item_pages: List[List[int]] = Factory(list)
    # Every visual row of the prompt in token order, read from the pages
    # (built per rank by :meth:`DisaggMixin.disagg_register`).
    visual_rows: Optional[PagedRows] = None


class DisaggMixin:
    def _init_disagg_state(self) -> None:
        # Encoder-disaggregation overlap: seq_id => per-item
        # readiness for seqs admitted before all their visual embeddings
        # arrived. Populated by the LM disagg manager; consumed by the
        # scheduler (gate B) and the embed path. Empty for the monolith.
        self.disagg_embeds: Dict[int, DisaggSeqState] = {}
        # ``mm_embed`` view ``[num_pages, rows_per_page, feat_dim]`` of the
        # cache arena (set by the LM disagg receiver).
        self._disagg_pool: Optional[torch.Tensor] = None
        self._disagg_rows_per_page = 0
        # seq_id -> item_idx (prompt order) -> pages, from allocation to free.
        self._disagg_pages: Dict[int, Dict[int, List[int]]] = {}
        # Freed pages waiting for in-flight GPU reads: (event, pages).
        self._disagg_quarantine: List[Tuple[torch.cuda.Event, List[int]]] = []
        # TP0: told when a registered seq stops needing its embeddings.
        self._disagg_on_release: Optional[Callable[[int], None]] = None

    def disagg_attach_pool(self, pool: torch.Tensor, rows_per_page: int) -> None:
        self._disagg_pool = pool
        self._disagg_rows_per_page = rows_per_page

    def _disagg_free(self, seq_id: int) -> None:
        """Drop a seq's disagg state (prefill done, freed, or aborted)."""
        if self.disagg_embeds.pop(seq_id, None) is not None and self._disagg_on_release:
            self._disagg_on_release(seq_id)

    def disagg_register(self, seq_id: int, state: DisaggSeqState) -> None:
        """Register a disagg seq for overlapped, readiness-gated prefill.

        Called by the LM disagg manager once *all* per-item ``MmItemMeta`` have
        arrived and every item has its pages (positions/hashes determined; gate
        A satisfied) but before the visual embeddings have necessarily landed.
        Readiness then advances via :meth:`disagg_mark_ready`.
        """
        page: List[int] = []
        off: List[int] = []
        for i in sorted(range(state.num_items), key=lambda i: state.item_span[i][0]):
            start, end = state.item_span[i]
            rows = int(state.is_multimodal_cpu[start:end].sum())
            p, o = row_index(state.item_pages[i], rows, self._disagg_rows_per_page)
            page += p
            off += o
        dev = self._disagg_pool.device
        state.visual_rows = PagedRows(
            self._disagg_pool,
            torch.tensor(page, dtype=torch.long, device=dev),
            torch.tensor(off, dtype=torch.long, device=dev),
        )
        self.disagg_embeds[seq_id] = state

    def disagg_mark_ready(
        self, seq_id: int, ordered_idx: int, rows_end: int, final: bool
    ) -> None:
        """Rows ``[0, rows_end)`` of one item have landed in its pages;
        ``final`` completes the item."""
        st = self.disagg_embeds.get(seq_id)
        if st is None:
            return
        st.item_rows[ordered_idx] = rows_end
        if final:
            st.item_ready[ordered_idx] = True

    # ------------------------------------------------------------------
    # mm_embed pages (applied identically on every TP rank)
    # ------------------------------------------------------------------
    def disagg_alloc_pages(
        self, requests: List[Tuple[int, int, int]]
    ) -> List[Optional[List[int]]]:
        """Allocate ``(seq_id, item_idx, num_pages)`` in order; once one fails
        the rest are skipped, so large items are not starved. Reclaims
        prefix-cache pages as needed."""
        allocator = self.memory_manager.cache_arena.allocator
        out: List[Optional[List[int]]] = []
        blocked = False
        for seq_id, item_idx, n in requests:
            pages = None if blocked else allocator.allocate(MM_EMBED, n)
            if pages is None:
                blocked = True
            else:
                self._disagg_pages.setdefault(seq_id, {})[item_idx] = pages
            out.append(pages)
        return out

    def disagg_free_pages(self, seq_ids: List[int]) -> None:
        """Release a seq's pages once the GPU work queued so far (which may
        still gather from them) has run; see :meth:`disagg_tick`."""
        pages = [
            p for sid in seq_ids for item in self._disagg_pages.pop(sid, {}).values()
            for p in item
        ]
        if pages:
            event = torch.cuda.Event()
            event.record(getattr(self, "forward_stream", None) or torch.cuda.current_stream())
            self._disagg_quarantine.append((event, pages))

    def disagg_tick(self) -> None:
        """Return quarantined pages to the arena. Runs once per iteration on
        every TP rank before that iteration's events, so the arena changes in
        lockstep across ranks. Pages are quarantined at least one iteration,
        long enough that the forward recorded after their last read has
        normally finished (the wait below is then free)."""
        if not self._disagg_quarantine:
            return
        allocator = self.memory_manager.cache_arena.allocator
        for event, pages in self._disagg_quarantine:
            event.synchronize()
            allocator.free(MM_EMBED, pages)
        self._disagg_quarantine = []

    def disagg_prefill_limit(self, seq: GenerationSequence) -> Optional[int]:
        """Gate-B upper bound: the largest token position this
        seq may prefill up to this round = the start of the first image span
        whose embedding hasn't landed yet (or ``prompt_len`` if all ready).
        ``None`` for non-disagg seqs (no cap).

        This deliberately matches :meth:`_disagg_ready_len` (the embed coverage)
        so the scheduler never advances ``computed_token_num`` past the embed
        coverage -- even when a prefix-cache hit would otherwise jump the cursor
        over an item whose embedding is still in flight. Such a (rare) seq waits
        for the embedding to land, then proceeds; the encoder's own embed cache
        keeps that wait short for repeated content.
        """
        st = self.disagg_embeds.get(seq.seq_id)
        if st is None:
            return None
        return self._disagg_ready_len(st)

    @staticmethod
    def _disagg_item_ready_end(st: DisaggSeqState, i: int) -> int:
        """End of item ``i``'s ready part: its span end when complete,
        otherwise the position of its first visual token whose embedding row
        has not landed (the span start when none has)."""
        start, end = st.item_span[i]
        if st.item_ready[i]:
            return end
        rows = st.item_rows[i] if st.item_rows else 0
        if rows == 0:
            return start
        mm_pos = st.is_multimodal_cpu[start:end].nonzero(as_tuple=True)[0]
        return end if rows >= mm_pos.numel() else start + int(mm_pos[rows])

    @staticmethod
    def _disagg_ready_len(st: DisaggSeqState) -> int:
        """Length of the ready prefix ``[0, ready_len)``.

        Stops at the first visual token (in token order) whose embedding has
        not landed, regardless of whether a later item happens to be ready: a
        prefix past it would have more ``is_multimodal`` positions than
        gathered embedding rows and the merge would misalign. A segment-streamed
        item contributes its landed rows.
        """
        rl = st.prompt_len
        for i in range(st.num_items):
            if not st.item_ready[i]:
                rl = min(rl, DisaggMixin._disagg_item_ready_end(st, i))
        return rl

    def _mm_disagg_collect(
        self,
        seq: GenerationSequence,
        st: DisaggSeqState,
        prefill_works: List[Dict],
        batch_positions: List[torch.Tensor],
    ) -> None:
        """Build the prefill work for an overlap disagg seq.

        Positions come from the full-prompt mrope grid (all grids known once
        meta arrived). Visual rows cover the ready prefix; refresh that view
        (kind ``uncached``; no data is copied) whenever the scheduler advances
        past the cached ``coverage_len`` because more rows landed, otherwise
        it is re-sliced for the current chunk (kind ``cached``).
        """
        batch_positions.append(
            st.prompt_positions[:, seq.computed_token_num : seq.seq_len]
        )
        info = self.embedding_cache.get(seq.seq_id)
        need_build = info is None or info.is_multimodal_cpu is None or (
            info.coverage_len is not None and seq.seq_len > info.coverage_len
        )
        if not need_build:
            prefill_works.append({"kind": "cached", "seq": seq, "embedding_info": info})
            return
        ready_len = self._disagg_ready_len(st)
        # The visual rows of the ready prefix, in token order: rows of a
        # partially landed item stop at its first missing row.
        ready_rows = int(st.is_multimodal_cpu[:ready_len].sum())
        prefill_works.append(
            {
                "kind": "uncached",
                "seq": seq,
                "input_ids_cpu": st.input_ids_cpu[:ready_len],
                "is_multimodal_cpu": st.is_multimodal_cpu[:ready_len],
                "mm_input": {},
                "visual_rows": st.visual_rows.prefix(ready_rows),
                "prompt_positions": st.prompt_positions,
                "mrope_position_delta": st.mrope_position_delta,
                "mm_bundle_key": None,
                "coverage_len": ready_len,
            }
        )
