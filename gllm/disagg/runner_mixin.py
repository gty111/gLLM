"""Encoder-disaggregation state for :class:`gllm.runtime.model_runner.ModelRunner`.

Method bodies here were moved verbatim out of ``gllm.runtime.model_runner``
and ``gllm.multimodal.mixin``; :class:`DisaggMixin` is mixed into
``ModelRunner`` so every ``self`` reference and call site keeps its original
meaning. It owns ``self.disagg_embeds`` -- per-seq readiness and visual
embeddings for seqs admitted before all their embeddings arrived -- which the
LM disagg manager fills, the scheduler reads for gate B and the multimodal
embed path consumes.
"""

from typing import Dict, List, Optional, Tuple

import torch
from attr import Factory, dataclass

from gllm.runtime.sequence import GenerationSequence


@dataclass
class DisaggSeqState:
    """Per-seq encoder-disaggregation overlap state.

    Owned by the :class:`ModelRunner` (keyed by ``seq_id``) so it is immune to
    the scheduler's chunked-prefill ``deepcopy`` of the :class:`GenerationSequence`. The
    LM disagg manager fills ``item_embed[i]`` (and flips ``item_ready[i]``) as
    each item's visual embedding lands over NIXL; the scheduler reads
    ``item_ready`` for the two-layer prefill gate and the model runner reads
    ``item_embed`` to embed the ready prefix.

    Items are stored in **image-then-video order** (the order
    ``model.embed_multimodal`` returns its tuple in, which is what the merge
    expects). Each carries its ``[span_start, span_end)`` in the *expanded*
    token sequence so gate B and the ready-prefix embed can be computed.
    """

    num_items: int
    item_span: List[Tuple[int, int]]  # ordered: (start, end) in tokens
    item_modality: List[str]
    item_ready: List[bool]
    item_embed: List[Optional[torch.Tensor]]  # ordered, filled on NIXL notif
    image_grid_thw: Optional[torch.Tensor]
    video_grid_thw: Optional[torch.Tensor]
    input_ids_cpu: torch.Tensor  # full expanded prompt ids (cpu)
    is_multimodal_cpu: torch.Tensor  # full mask (cpu)
    prompt_positions: torch.Tensor  # full-prompt mrope positions
    mrope_position_delta: torch.Tensor
    prompt_len: int
    # Segment-streamed items (video): rows ``[0, item_rows[i])`` have landed so
    # far, copied into a per-item buffer sized for the whole item, which
    # becomes ``item_embed[i]`` when the item completes (no further copy).
    item_rows: List[int] = Factory(list)
    item_buffer: List[Optional[torch.Tensor]] = Factory(list)


class DisaggMixin:
    def _init_disagg_state(self) -> None:
        # Encoder-disaggregation overlap: seq_id => per-item
        # readiness + embeddings for seqs admitted before all their visual
        # embeddings arrived. Populated by the LM disagg manager; consumed by
        # the scheduler (gate B) and the embed path. Empty for the monolith.
        self.disagg_embeds: Dict[int, DisaggSeqState] = {}

    def _disagg_free(self, seq_id: int) -> None:
        """Drop a seq's disagg state (prefill done, freed, or aborted)."""
        self.disagg_embeds.pop(seq_id, None)

    def disagg_register(self, seq_id: int, state: DisaggSeqState) -> None:
        """Register a disagg seq for overlapped, readiness-gated prefill.

        Called by the LM disagg manager once *all* per-item ``MmItemMeta`` have
        arrived (positions/hashes determined; gate A satisfied) but before the
        visual embeddings have necessarily landed. The embeddings are filled in
        progressively via :meth:`disagg_add_embedding`.
        """
        self.disagg_embeds[seq_id] = state

    def disagg_add_embedding(
        self,
        seq_id: int,
        ordered_idx: int,
        rows: torch.Tensor,
        rows_end: int,
        final: bool,
    ) -> None:
        """Record rows ``[rows_end - len(rows), rows_end)`` of one item's
        visual embedding (NIXL write landed). ``final`` completes the item.

        A whole item arrives as one final call and is stored as is. Segment-
        streamed rows are copied into a buffer allocated once for the whole
        item, so readers can slice the landed prefix without concatenating.
        ``rows`` may be a view of the receive slot; it is consumed here.
        """
        st = self.disagg_embeds.get(seq_id)
        if st is None:
            return
        lo = rows_end - rows.shape[0]
        buf = st.item_buffer[ordered_idx]
        if final and lo == 0 and buf is None:
            st.item_embed[ordered_idx] = rows
        else:
            if buf is None:
                start, end = st.item_span[ordered_idx]
                total = int(st.is_multimodal_cpu[start:end].sum())
                buf = rows.new_empty((total, rows.shape[1]))
                st.item_buffer[ordered_idx] = buf
            buf[lo:rows_end].copy_(rows)
            if final:
                st.item_embed[ordered_idx] = buf
                st.item_buffer[ordered_idx] = None
        st.item_rows[ordered_idx] = rows_end
        if final:
            st.item_ready[ordered_idx] = True

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
        meta arrived). Encoded visual rows cover the ready prefix; refresh them
        (kind ``uncached``) whenever the scheduler advances past the cached
        ``coverage_len`` because more items became ready, otherwise the cached
        rows are re-sliced for the current chunk (kind ``cached``).
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
        # Gather the ready-prefix items in token-span order so the concatenated
        # embeddings line up 1-1 with the ``is_multimodal`` True positions. An
        # item cut by ``ready_len`` (segment-streamed, partially landed)
        # contributes the rows of its visual tokens before ``ready_len``.
        ready_items = [
            i for i in range(st.num_items) if st.item_span[i][0] < ready_len
        ]
        ready_items.sort(key=lambda i: st.item_span[i][0])
        ready_embeds = []
        for i in ready_items:
            start, end = st.item_span[i]
            if end <= ready_len:
                ready_embeds.append(st.item_embed[i])
                continue
            n = int(st.is_multimodal_cpu[start:ready_len].sum())
            if n:
                ready_embeds.append(st.item_buffer[i][:n])
        ready_embeds = tuple(ready_embeds)
        prefill_works.append(
            {
                "kind": "uncached",
                "seq": seq,
                "input_ids_cpu": st.input_ids_cpu[:ready_len],
                "is_multimodal_cpu": st.is_multimodal_cpu[:ready_len],
                "mm_input": {},
                "mm_embeddings": ready_embeds if ready_embeds else None,
                "prompt_positions": st.prompt_positions,
                "mrope_position_delta": st.mrope_position_delta,
                "mm_bundle_key": None,
                "coverage_len": ready_len,
            }
        )
