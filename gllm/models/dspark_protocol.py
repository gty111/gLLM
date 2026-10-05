"""Execution contract shared by every DSpark draft adapter.

DSpark is the joint noisy-block drafter: target-model hidden taps are projected
into the draft width, a fixed block of noisy tokens is run through a short
backbone, and the resulting hidden states are read out through a low-rank
Markov logit correction plus a per-position confidence head.

Every DSpark checkpoint shares that *semantics*; none of them share the
backbone.  DeepSeek-V4 ships three ``mtp.*`` stages built from mHC + MLA + MoE,
while the standalone Qwen3.8 drafter is five plain GQA + SwiGLU layers.  This
module pins the shared pipeline down so one caller can drive any of them, and
so each adapter can be checked against a single contract.

Deliberately absent: attention kernels, cache layouts, tensor-parallel sharding
and weight loading.  Those differ per checkpoint and stay the adapter's own
business.

A draft model borrows the *target* model's token embedding and LM head -- it
ships no vocabulary of its own.  An adapter therefore either keeps references
to the target's modules (DeepSeek-V4) or is handed the already-embedded noise
block and the target LM head by its caller (Qwen3.8).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:  # pragma: no cover - imported for annotations only
    import torch


class DSparkForwardProtocol(ABC):
    """The three calls a DSpark draft model must support.

    The pipeline they name is::

        prefill(target hidden)         -> per-stage target-history caches
        forward_draft(anchor + noise)  -> draft hidden, draft token ids
        forward_head(draft hidden)     -> output ids, logits, confidence

    Cache objects are opaque to the caller: whatever :meth:`prefill` returns is
    handed straight back to :meth:`forward_draft`.  That keeps this contract
    free of any assumption about window caches, paged KV or ``DynamicCache``.
    """

    #: Number of drafted positions per step (``block_size`` in the checkpoint).
    block_size: int

    @abstractmethod
    def prefill(
        self,
        main_hidden: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> Any:
        """Project target hidden states into the per-stage history caches.

        ``main_hidden`` concatenates the checkpoint's target hidden taps and
        has shape ``[B, S, len(target_layer_ids) * hidden]``.  ``input_ids``
        is the anchor token per row, used by adapters that key their cache on
        the token rather than on the hidden state.
        """

    @abstractmethod
    def forward_draft(
        self,
        main_hidden: torch.Tensor,
        input_ids: torch.Tensor,
        *,
        start_pos: int,
        caches: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the noisy draft block and return ``(hidden, draft_ids)``.

        The implementation builds the ``block_size``-wide noisy block itself
        from the anchor in ``input_ids`` (slot 0 is the real token, the rest
        are the checkpoint's mask/noise id) and attends it against ``caches``.
        ``draft_ids`` is returned alongside the hidden states because the
        Markov head conditions on the previous token of each position.
        """

    @abstractmethod
    def forward_head(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        *,
        sample: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply the LM head, the Markov correction and the confidence head.

        Returns ``(output_ids, logits, confidence)``: the ``block_size + 1``
        token ids with the anchor first, the corrected logits per drafted
        position, and one acceptance score per position.  ``sample`` defaults
        to argmax and only ever picks the Markov conditioning token.
        """


__all__ = ["DSparkForwardProtocol"]