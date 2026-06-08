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
   ``scatter_`` per decode step. gLLM now follows the same idea: each seq
   with ``repetition_penalty != 1.0`` owns a persistent row in a
   worker-local pool (``MemoryManager._rep_pool``), allocated on first use
   and released via ``MemoryManager.free``. Every decode step scatters only
   the newly appended token into that row and gathers the per-batch
   ``[batch, vocab]`` mask in a single ``index_select`` -- O(batch) work per
   step instead of the old from-scratch ``O(sum(len(seq.token_ids)))``
   rebuild. The build itself lives in
   ``MemoryManager.build_repetition_penalty_mask``; only the GPU-side apply
   helper below is shared.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _scaling_penalty_kernel(
    logits_ptr,
    penalties_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    logit = tl.load(logits_ptr + offsets, mask=mask)
    pen = tl.load(penalties_ptr + offsets, mask=mask)
    out = tl.where(logit < 0, logit * pen, logit / pen)
    tl.store(logits_ptr + offsets, out, mask=mask)


def apply_scaling_penalties(
    logits: torch.Tensor, scaling_penalties: torch.Tensor
) -> None:
    """Apply per-token scaling penalty to ``logits`` in-place.

    ``scaling_penalties[i, t]`` is ``1.0`` for tokens that haven't appeared
    in row ``i``'s seq and the seq's ``repetition_penalty`` otherwise.
    Positive logits are divided (decrease their score) and non-positive
    logits are multiplied (push them further negative). Tokens whose mask
    value is ``1.0`` are left unchanged.

    Mirrors sglang's ``apply_scaling_penalties``. The earlier eager
    formulation (``logits[:] = torch.where(logits < 0, logits * p,
    logits / p)``) made five full-vocab passes -- ``lt`` + ``mul`` + ``div``
    over the whole ``[batch, vocab]`` tensor, then a ``where`` select and a
    ``copy_`` back through ``logits[:]``. The Triton kernel below fuses all
    of that into a single in-place pass over the flattened tensor.
    """
    if not logits.is_contiguous():
        # Fall back to the eager path for the (unexpected) non-contiguous
        # case so the in-place write still lands on ``logits``' storage.
        logits[:] = torch.where(
            logits < 0,
            logits * scaling_penalties,
            logits / scaling_penalties,
        )
        return

    penalties = scaling_penalties.contiguous()
    n_elements = logits.numel()
    if n_elements == 0:
        return
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _scaling_penalty_kernel[grid](logits, penalties, n_elements, BLOCK_SIZE=1024)
