"""Contracts shared by the DSpark draft adapters.

DSpark is the joint noisy-block drafter: target-model hidden taps are projected
into the draft width, a fixed block of noisy tokens is run through a short
backbone, and the resulting hidden states are read out through a low-rank
Markov logit correction plus a per-position confidence head.

Every DSpark checkpoint shares that *semantics*; none of them share the
backbone.  DeepSeek-V4 ships three ``mtp.*`` stages built from mHC + MLA + MoE,
while the standalone Qwen3.8 drafter is five plain GQA + SwiGLU layers.  Two
contracts live here so one caller can drive any adapter, and so each adapter
can be checked against a single description:

* :class:`DSparkForwardProtocol` -- the three forward calls.
* :class:`DSparkCheckpointMapping` -- how a checkpoint names and shards the
  draft parameters.

Deliberately absent: attention kernels, cache layouts and tensor-parallel
sharding decisions.  Those differ per checkpoint and stay the adapter's own
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

    from gllm.models.weight_loader import LoadContext, WeightRule


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


class DSparkCheckpointMapping(ABC):
    """How one DSpark checkpoint names and shards its draft parameters.

    gLLM already owns the loading *engine*: ``run_weight_loader`` supplies the
    progress reporting, the pipeline-layer remap, ordered first-match dispatch,
    the shared MoE expert-copy pool and the checkpoint-key fallbacks.  What
    differs between DSpark checkpoints is only the mapping, so that is the
    whole contract:

    * :meth:`checkpoint_key` -- what the checkpoint calls a parameter.  The
      DeepSeek-V4 draft lives under ``mtp.0/1/2.*`` inside the target
      checkpoint; a standalone drafter tends to reuse the names it already
      exposes as module attributes.
    * :meth:`weight_rules` -- how each parameter is sharded.  Ordered, first
      match wins, exactly like every other model's ``weight_rules``.
    * :meth:`load_context` -- the optional per-model call state a handler
      cannot derive from a key alone (vocab shard indices, expert handles).

    An adapter is free to skip :meth:`load_context`, and free to call
    ``run_weight_loader`` from wherever it likes: this fixes the vocabulary of
    the mapping, not the call site.
    """

    @abstractmethod
    def checkpoint_key(self, param_name: str) -> str:
        """Map a module parameter path to its checkpoint key.

        ``param_name`` is the full parameter path including its trailing
        ``.weight`` / ``.bias`` / ``.weight_scale_inv`` suffix, exactly as
        ``run_weight_loader`` hands it over after pipeline remapping.
        """

    @abstractmethod
    def weight_rules(self) -> list[WeightRule]:
        """Ordered rule table; the first match per parameter wins."""

    def load_context(self, weights: dict[str, torch.Tensor]) -> LoadContext | None:
        """Optional per-model ``LoadContext``; ``None`` means no override."""
        return None


__all__ = ["DSparkCheckpointMapping", "DSparkForwardProtocol"]