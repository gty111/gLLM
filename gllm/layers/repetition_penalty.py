"""Repetition-penalty utilities.

The design here is sglang-inspired -- in particular the in-place
``apply_scaling_penalties`` helper mirrors
``sglang/srt/sampling/penaltylib/repetition_penalty.py``. We deliberately
diverge in two places that matter for gLLM:

1. **What goes into the mask.** SGLang only feeds *output tokens* into
   ``cumulated_repetition_penalties`` (see
   ``BatchedRepetitionPenalizer._cumulate_output_tokens``). That works for
   their streaming/session model, but for stateless OpenAI-style chat in
   gLLM the previous assistant turn lives inside the next request's
   *prompt*, not in ``output_ids``. If we mirrored sglang verbatim,
   greedy decoding with ``repetition_penalty=1.05`` from
   ``generation_config.json`` would still happily spell the same reply
   token-by-token (see the "Can you tell a long long story?" repro). So
   we match HF/vLLM semantics and seed the mask with **every token in
   the seq's history** -- prompt and generated alike.

2. **Where the mask lives.** SGLang keeps a persistent ``[batch, vocab]``
   tensor on the orchestrator and updates it incrementally with one
   ``scatter_`` per decode step. gLLM's running batch is composed
   per-step from a pool of running ``Sequence`` objects (the post_schedule
   IPC copy is intentionally lightweight), so we currently rebuild the
   mask from scratch each ``prepare_sample`` instead of carrying
   persistent slots. This costs ``O(sum(len(seq.token_ids)))`` per step
   and is fine for chat-class batches; if/when batch sizes grow we can
   port sglang's incremental design by adding a worker-local slot pool
   and hooking ``ModelRunner.free`` for slot release.
"""

from typing import List, Optional

import torch

from gllm.utils import async_tensor_h2d


def apply_scaling_penalties(
    logits: torch.Tensor, scaling_penalties: torch.Tensor
) -> None:
    """Apply per-token scaling penalty to ``logits`` in-place.

    ``scaling_penalties[i, t]`` is ``1.0`` for tokens that haven't appeared
    in row ``i``'s seq and the seq's ``repetition_penalty`` otherwise.
    Positive logits are divided (decrease their score) and non-positive
    logits are multiplied (push them further negative). Tokens whose mask
    value is ``1.0`` are left unchanged.

    Mirrors sglang's ``apply_scaling_penalties``. Note that this writes
    through ``logits[:]`` instead of using a split ``div_`` / ``mul_``
    pair -- the earlier split-op formulation was a silent no-op because
    its second mask was evaluated against the *already-divided* logits
    (positive logits got ``logit / p * p == logit``).
    """
    logits[:] = torch.where(
        logits < 0,
        logits * scaling_penalties,
        logits / scaling_penalties,
    )


def build_repetition_penalty_mask(
    seqs: list,
    batch_size: int,
    vocab_size: int,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    """Build a ``[batch_size, vocab_size]`` scaling-penalty mask on GPU.

    Returns ``None`` if no sequence in the batch has
    ``repetition_penalty != 1.0`` -- the caller can use this to gate the
    whole penalty step. Otherwise the returned tensor holds:

    * ``seqs[i].repetition_penalty`` at positions of tokens that have
      already appeared in ``seqs[i]`` (prompt + tokens generated so far);
    * ``1.0`` everywhere else.

    Sequences that have ``repetition_penalty == 1.0`` contribute a row of
    all ones, so multiplying through them is a no-op. Sequences whose
    ``token_ids`` mirror was elided on the follower (the delta-broadcast
    path in ``gllm/dist_schedule.py`` sets
    ``FollowerSeq.token_ids = None`` for seqs that don't need it for
    rep-penalty or VL) are skipped gracefully -- a follower will never
    reach this branch for a ``repetition_penalty != 1.0`` seq because
    the driver sets ``needs_token_id_accumulation=True`` at registration
    time. The guard is defensive.
    """
    if not any(getattr(seq, "repetition_penalty", 1.0) != 1.0 for seq in seqs):
        return None

    seq_idx_list: List[int] = []
    token_idx_list: List[int] = []
    for i, seq in enumerate(seqs):
        if seq.repetition_penalty == 1.0:
            continue
        if seq.token_ids is None or len(seq.token_ids) == 0:
            continue
        seq_idx_list.extend([i] * len(seq.token_ids))
        token_idx_list.extend(seq.token_ids)

    mask = torch.ones((batch_size, vocab_size), dtype=dtype, device="cuda")
    if not seq_idx_list:
        return mask

    penalty_vec = async_tensor_h2d(
        [seq.repetition_penalty for seq in seqs], dtype, "cuda", True
    )
    seq_idx_t = async_tensor_h2d(seq_idx_list, torch.long, "cuda", True)
    token_idx_t = async_tensor_h2d(token_idx_list, torch.long, "cuda", True)
    # Scatter the per-seq penalty value into the rows of every token id
    # that already appeared in that seq. Duplicate ``(i, t)`` indices are
    # harmless: they overwrite with the same penalty value.
    mask[seq_idx_t, token_idx_t] = penalty_vec[seq_idx_t]
    return mask
