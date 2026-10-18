"""Causal repetition penalties for speculative draft and target distributions."""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from gllm.utils import async_tensor_h2d


@triton.jit
def _apply_kernel(
    logits, history, penalties, candidates,
    VOCAB: tl.constexpr, WIDTH: tl.constexpr,
    LOGIT_STRIDE: tl.constexpr, HISTORY_STRIDE: tl.constexpr,
    CANDIDATE_STRIDE: tl.constexpr, UPDATE_HISTORY: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    seq = row // WIDTH
    position = row % WIDTH
    token = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    valid = token < VOCAB
    scale = tl.load(history + seq * HISTORY_STRIDE + token, valid, other=1)
    penalty = tl.load(penalties + seq)
    seen = tl.full((BLOCK,), False, tl.int1)
    for j in tl.static_range(WIDTH):
        candidate = tl.load(candidates + seq * CANDIDATE_STRIDE + j)
        seen |= (j <= position) & (candidate == token)
    scale = tl.where(seen, penalty, scale)
    value = tl.load(logits + row * LOGIT_STRIDE + token, valid, other=0)
    result = tl.where(value < 0, value * scale, value / scale)
    tl.store(logits + row * LOGIT_STRIDE + token, result, valid)
    if UPDATE_HISTORY:
        tl.store(history + seq * HISTORY_STRIDE + token, scale, valid)


def apply_speculative_penalties(logits, history, penalties, candidates, *, update_history=False):
    """Row j sees history plus candidates[:j+1], never the speculative tail.

    A one-position draft call may update its private scratch history. Target
    verification never mutates committed history. Repeated tokens are penalized
    once, just like the ordinary sampler's set-membership mask.
    """
    batch, width = candidates.shape
    assert logits.shape == (batch * width, history.shape[1])
    assert not update_history or width == 1
    assert logits.stride(1) == history.stride(1) == candidates.stride(1) == 1
    _apply_kernel[(batch * width, triton.cdiv(logits.shape[1], 1024))](
        logits, history, penalties, candidates,
        VOCAB=logits.shape[1], WIDTH=width,
        LOGIT_STRIDE=logits.stride(0), HISTORY_STRIDE=history.stride(0),
        CANDIDATE_STRIDE=candidates.stride(0), UPDATE_HISTORY=update_history,
        BLOCK=1024,
    )


@triton.jit
def _commit_kernel(pool, slots, penalties, candidates, accepted,
                   VOCAB: tl.constexpr, WIDTH: tl.constexpr,
                   STRIDE: tl.constexpr, BLOCK: tl.constexpr):
    seq = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    slot = tl.load(slots + seq)
    count = tl.load(accepted + seq) + 1  # Always commit x1, plus accepted drafts.
    token = tl.load(candidates + seq * STRIDE + col, col < WIDTH, other=-1)
    valid = (slot > 0) & (col < count) & (col < WIDTH) & (token >= 0) & (token < VOCAB)
    # Avoid multiple lanes writing the same token, even when the value agrees.
    for j in tl.static_range(WIDTH):
        earlier = tl.load(candidates + seq * STRIDE + j)
        valid &= ~((col > j) & (token == earlier))
    penalty = tl.load(penalties + seq)
    tl.store(pool + slot * VOCAB + token, penalty, valid)


@dataclass
class SpeculativeRepetitionPenalty:
    manager: object
    history: torch.Tensor
    slots: torch.Tensor
    values: torch.Tensor

    @classmethod
    def prepare(cls, manager, seqs):
        if not any(getattr(s, "repetition_penalty", 1.0) != 1.0 for s in seqs):
            return None
        # The existing pool seeds the prompt once and consumes only new CPU
        # tokens afterwards. GPU commits below cover async accepted tokens even
        # while CPU history still contains optimistic FutureMap placeholders.
        history = manager.build_repetition_penalty_mask(seqs)
        if history is None:
            return None
        slots = async_tensor_h2d(
            [s.rep_slot if s.repetition_penalty != 1.0 else 0 for s in seqs],
            torch.int64, history.device, True,
        )
        values = async_tensor_h2d(
            [s.repetition_penalty for s in seqs], history.dtype, history.device, True,
        )
        return cls(manager, history, slots, values)

    def verify(self, logits, candidates):
        apply_speculative_penalties(logits, self.history, self.values, candidates)

    def commit(self, candidates, accepted):
        # Mixed-prefill sampling may grow the pool after prepare(). Look up the
        # current allocation here; slot identities survive pool growth.
        _commit_kernel[(candidates.shape[0],)](
            self.manager._rep_pool, self.slots, self.values, candidates, accepted,
            VOCAB=self.history.shape[1], WIDTH=candidates.shape[1],
            STRIDE=candidates.stride(0), BLOCK=triton.next_power_of_2(candidates.shape[1]),
        )
