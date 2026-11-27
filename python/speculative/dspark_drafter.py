"""DSpark as a drafter of the shared MTP speculative pipeline.

The pipeline (``speculative/mtp.py``) owns verify, acceptance, GDN checkpoint
promotion, relay/async state, GPU input prep, CUDA graphs and overlap. A
drafter supplies two operations:

* ``kv_sync``: after every target forward, write draft K/V for the rows that
  forward consumed. DSpark projects the target's hidden taps (written to a
  persistent sink by the backbone) into its own layers, which live as extra
  layers of the target KV segment and share its page tables and slot
  mapping. Verify windows are written whole; rows past the accepted prefix
  are overwritten by the next window before any draft attention reads them.
* ``propose``: k draft tokens after each request's uncached last token ``x1``, from
  the draft K/V history ``[0, context_len)``; inputs and outputs stay on the
  device, and shapes depend only on the batch size.
"""

import torch


class DSparkDrafter:
    def __init__(self, draft, *, kv_base_layer: int, taps: torch.Tensor):
        self.draft = draft
        self.kv_base_layer = kv_base_layer
        self.taps = taps
        self.kv_views = None

    @property
    def num_kv_layers(self) -> int:
        return len(self.draft.layers)

    @property
    def num_draft_tokens(self) -> int:
        return self.draft.block_size

    def bind_kv(self, segment) -> None:
        """Bind to the target segment once the cache arena exists."""
        self.kv_views = self.draft.kv_layer_views(segment, self.kv_base_layer)

    def kv_sync(self, input_data, num_tokens: int) -> None:
        """Write draft K/V for the last target forward's first ``num_tokens`` rows."""
        positions = input_data.get_position()
        if positions.ndim == 2:  # MRoPE: text positions are identical per axis
            positions = positions[0]
        self.draft.write_context_kv(
            self.taps[:num_tokens],
            positions[:num_tokens],
            input_data.get_slot_mapping()[:num_tokens],
            self.kv_views,
        )

    def propose(
        self, anchors, context_lens, block_table, *,
        confidence_out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if confidence_out is None:
            return self.draft.propose(anchors, context_lens, block_table, self.kv_views)
        return self.draft.propose(
            anchors, context_lens, block_table, self.kv_views,
            confidence_out=confidence_out,
        )

    def propose_with(
        self, anchors, context_lens, block_table, pick, *,
        confidence_out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Proposals drawn by ``pick`` (see ``Qwen38DSpark.chain_proposals``).

        Used for sampled requests: ``pick`` applies the request's sampling
        transform, records the proposal distribution ``q`` for rejection
        sampling, and draws from it. Optional raw confidence scores use the
        tokens actually drawn; neither ``pick`` nor the LM head is rerun.
        The output buffer belongs to the caller and must not alias inputs.
        """
        if confidence_out is not None:
            self.draft._validate_confidence_out(anchors, confidence_out)
        hidden = self.draft.noise_hidden(anchors, context_lens, block_table, self.kv_views)
        proposals = self.draft.chain_proposals(self.draft._lm_head[0](hidden), anchors, pick)
        if confidence_out is not None:
            confidence_out.copy_(
                self.draft._confidence_from_proposals(hidden, anchors, proposals)
            )
        return proposals
