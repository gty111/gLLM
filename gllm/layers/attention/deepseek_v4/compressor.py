"""Reference learned KV pooling for DeepSeek-V4 compressed attention."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from gllm.layers.attention.deepseek_v4.ops import (
    apply_rope_inplace,
    fp8_fake_quantize_inplace,
)
from gllm.layers.linear import ReplicatedLinear
from gllm.layers.ops.deepseek_v4 import compress_decode_batch_fused


@dataclass
class CompressorState:
    kv: torch.Tensor
    score: torch.Tensor


def make_compressor_state(
    batch_size: int,
    ratio: int,
    head_dim: int,
    *,
    device: torch.device | str,
) -> CompressorState:
    overlap = ratio == 4
    coff = 1 + overlap
    shape = (batch_size, coff * ratio, coff * head_dim)
    return CompressorState(
        kv=torch.zeros(shape, device=device, dtype=torch.float32),
        score=torch.full(shape, -torch.inf, device=device, dtype=torch.float32),
    )


def _validate_inputs(
    kv: torch.Tensor, score: torch.Tensor, ape: torch.Tensor, ratio: int
) -> tuple[int, int, int, bool]:
    if kv.dtype != torch.float32 or score.dtype != torch.float32:
        raise TypeError("compressor kv and score inputs must be float32")
    if kv.shape != score.shape or kv.ndim != 3:
        raise ValueError("compressor kv and score must have the same [B,S,C] shape")
    if ratio <= 0:
        raise ValueError("compression ratio must be positive")
    overlap = ratio == 4
    coff = 1 + overlap
    if kv.shape[-1] % coff:
        raise ValueError("compressor channel count is incompatible with ratio")
    head_dim = kv.shape[-1] // coff
    if tuple(ape.shape) != (ratio, coff * head_dim):
        raise ValueError(
            f"ape must have shape {(ratio, coff * head_dim)}, got {ape.shape}"
        )
    return kv.shape[0], kv.shape[1], head_dim, overlap


def compress_prefill(
    kv: torch.Tensor,
    score: torch.Tensor,
    ape: torch.Tensor,
    ratio: int,
) -> tuple[torch.Tensor | None, CompressorState]:
    """Compress a start-position-zero prefill and retain decode state."""
    batch, sequence_length, head_dim, overlap = _validate_inputs(
        kv, score, ape, ratio
    )
    state = make_compressor_state(
        batch, ratio, head_dim, device=kv.device
    )
    cutoff = sequence_length - sequence_length % ratio
    remainder = sequence_length - cutoff
    offset = ratio if overlap else 0

    if overlap and cutoff >= ratio:
        state.kv[:, :ratio].copy_(kv[:, cutoff - ratio : cutoff])
        state.score[:, :ratio].copy_(
            score[:, cutoff - ratio : cutoff] + ape
        )
    if remainder:
        state.kv[:, offset : offset + remainder].copy_(kv[:, cutoff:])
        state.score[:, offset : offset + remainder].copy_(
            score[:, cutoff:] + ape[:remainder]
        )
    if cutoff == 0:
        return None, state

    full_kv = kv[:, :cutoff].unflatten(1, (-1, ratio))
    full_score = score[:, :cutoff].unflatten(1, (-1, ratio)) + ape
    if overlap:
        chunks = full_kv.shape[1]
        overlap_kv = full_kv.new_zeros(batch, chunks, 2 * ratio, head_dim)
        overlap_score = full_score.new_full(
            (batch, chunks, 2 * ratio, head_dim), -torch.inf
        )
        overlap_kv[:, :, ratio:] = full_kv[..., head_dim:]
        overlap_score[:, :, ratio:] = full_score[..., head_dim:]
        overlap_kv[:, 1:, :ratio] = full_kv[:, :-1, :, :head_dim]
        overlap_score[:, 1:, :ratio] = full_score[:, :-1, :, :head_dim]
        full_kv, full_score = overlap_kv, overlap_score
    compressed = (full_kv * full_score.softmax(dim=2)).sum(dim=2)
    return compressed, state


def compress_prefill_batch(
    kv: torch.Tensor,
    score: torch.Tensor,
    ape: torch.Tensor,
    ratio: int,
    lengths: torch.Tensor,
) -> tuple[torch.Tensor | None, CompressorState, torch.Tensor]:
    """Compress a padded variable-length batch in one tensor program."""
    batch, sequence_length, head_dim, overlap = _validate_inputs(
        kv, score, ape, ratio
    )
    if lengths.ndim != 1 or lengths.numel() != batch:
        raise ValueError("lengths must have one entry per prefill row")
    lengths = lengths.to(device=kv.device, dtype=torch.long)

    state = make_compressor_state(batch, ratio, head_dim, device=kv.device)
    full_counts = lengths // ratio
    max_chunks = sequence_length // ratio
    compressed = None
    if max_chunks:
        cutoff = max_chunks * ratio
        full_kv = kv[:, :cutoff].unflatten(1, (max_chunks, ratio))
        full_score = score[:, :cutoff].unflatten(1, (max_chunks, ratio)) + ape
        if overlap:
            overlap_kv = full_kv.new_zeros(
                batch, max_chunks, 2 * ratio, head_dim
            )
            overlap_score = full_score.new_full(
                (batch, max_chunks, 2 * ratio, head_dim), -torch.inf
            )
            overlap_kv[:, :, ratio:] = full_kv[..., head_dim:]
            overlap_score[:, :, ratio:] = full_score[..., head_dim:]
            overlap_kv[:, 1:, :ratio] = full_kv[:, :-1, :, :head_dim]
            overlap_score[:, 1:, :ratio] = full_score[:, :-1, :, :head_dim]
            full_kv, full_score = overlap_kv, overlap_score
        compressed = (full_kv * full_score.softmax(dim=2)).sum(dim=2)
        valid_chunks = (
            torch.arange(max_chunks, device=kv.device).unsqueeze(0)
            < full_counts.unsqueeze(1)
        )
        compressed.masked_fill_(~valid_chunks.unsqueeze(-1), 0)

    rows = torch.arange(batch, device=kv.device).unsqueeze(1)
    within = torch.arange(ratio, device=kv.device).unsqueeze(0)
    cutoffs = full_counts * ratio
    remainder = lengths - cutoffs
    if overlap:
        previous_positions = cutoffs.unsqueeze(1) - ratio + within
        previous_valid = cutoffs.ge(ratio).unsqueeze(1)
        safe_previous = previous_positions.clamp(0, max(sequence_length - 1, 0))
        previous_kv = kv[rows, safe_previous]
        previous_score = score[rows, safe_previous] + ape.unsqueeze(0)
        state.kv[:, :ratio].copy_(
            torch.where(previous_valid.unsqueeze(-1), previous_kv, 0)
        )
        state.score[:, :ratio].copy_(
            torch.where(previous_valid.unsqueeze(-1), previous_score, -torch.inf)
        )
        state_offset = ratio
    else:
        state_offset = 0

    remainder_positions = cutoffs.unsqueeze(1) + within
    remainder_valid = within < remainder.unsqueeze(1)
    safe_remainder = remainder_positions.clamp(0, max(sequence_length - 1, 0))
    remainder_kv = kv[rows, safe_remainder]
    remainder_score = score[rows, safe_remainder] + ape.unsqueeze(0)
    state.kv[:, state_offset : state_offset + ratio].copy_(
        torch.where(remainder_valid.unsqueeze(-1), remainder_kv, 0)
    )
    state.score[:, state_offset : state_offset + ratio].copy_(
        torch.where(
            remainder_valid.unsqueeze(-1), remainder_score, -torch.inf
        )
    )
    return compressed, state, full_counts


def compress_prefill_continue_batch(
    kv: torch.Tensor,
    score: torch.Tensor,
    ape: torch.Tensor,
    ratio: int,
    starts: torch.Tensor,
    lengths: torch.Tensor,
    state: CompressorState,
) -> tuple[torch.Tensor, CompressorState, torch.Tensor]:
    """Consume variable-length prefill suffixes without a token loop.

    ``state`` is the request-owned compressor state after ``starts`` tokens.
    The returned tensor contains only newly completed compression groups,
    padded on the group dimension; ``counts`` identifies the valid prefix for
    each row.  This is the bulk counterpart of repeatedly calling
    :func:`compress_decode` for a chunked/continuation prefill.
    """
    batch, sequence_length, head_dim, overlap = _validate_inputs(
        kv, score, ape, ratio
    )
    if starts.ndim != 1 or starts.numel() != batch:
        raise ValueError("starts must have one entry per prefill row")
    if lengths.ndim != 1 or lengths.numel() != batch:
        raise ValueError("lengths must have one entry per prefill row")
    coff = 1 + overlap
    expected_state = (batch, coff * ratio, coff * head_dim)
    if state.kv.shape != expected_state or state.score.shape != expected_state:
        raise ValueError(f"compressor state must have shape {expected_state}")

    device = kv.device
    starts = starts.to(device=device, dtype=torch.long)
    lengths = lengths.to(device=device, dtype=torch.long)
    cursor = starts.remainder(ratio)
    # cursor <= ratio - 1, so this static bound covers every row without a
    # GPU->CPU max().item() synchronization in each transformer layer.
    max_groups = (sequence_length + 2 * ratio - 2) // ratio
    channels = kv.shape[-1]
    timeline_kv = kv.new_zeros(batch, max_groups * ratio, channels)
    timeline_score = score.new_full(
        (batch, max_groups * ratio, channels), -torch.inf
    )

    current_offset = ratio if overlap else 0
    current_kv = state.kv[:, current_offset : current_offset + ratio]
    current_score = state.score[:, current_offset : current_offset + ratio]
    within_ratio = torch.arange(ratio, device=device).unsqueeze(0)
    prefix_valid = within_ratio < cursor.unsqueeze(1)
    timeline_kv[:, :ratio].copy_(
        torch.where(prefix_valid.unsqueeze(-1), current_kv, 0)
    )
    timeline_score[:, :ratio].copy_(
        torch.where(prefix_valid.unsqueeze(-1), current_score, -torch.inf)
    )

    columns = torch.arange(sequence_length, device=device).unsqueeze(0)
    valid_tokens = columns < lengths.unsqueeze(1)
    destinations = cursor.unsqueeze(1) + columns
    rows = torch.arange(batch, device=device).unsqueeze(1).expand_as(destinations)
    valid_rows = rows[valid_tokens]
    valid_destinations = destinations[valid_tokens]
    ape_rows = destinations.remainder(ratio)
    timeline_kv[valid_rows, valid_destinations] = kv[valid_tokens]
    timeline_score[valid_rows, valid_destinations] = (
        score[valid_tokens] + ape.index_select(0, ape_rows[valid_tokens])
    )

    grouped_kv = timeline_kv.view(batch, max_groups, ratio, channels)
    grouped_score = timeline_score.view(batch, max_groups, ratio, channels)
    counts = (cursor + lengths) // ratio
    group_ids = torch.arange(max_groups, device=device).unsqueeze(0)
    valid_groups = group_ids < counts.unsqueeze(1)

    if overlap:
        previous_kv = torch.cat(
            [state.kv[:, :ratio].unsqueeze(1), grouped_kv[:, :-1]], dim=1
        )
        previous_score = torch.cat(
            [state.score[:, :ratio].unsqueeze(1), grouped_score[:, :-1]], dim=1
        )
        pooled_kv = torch.cat(
            [previous_kv[..., :head_dim], grouped_kv[..., head_dim:]], dim=2
        )
        pooled_score = torch.cat(
            [
                previous_score[..., :head_dim],
                grouped_score[..., head_dim:],
            ],
            dim=2,
        )
    else:
        pooled_kv = grouped_kv
        pooled_score = grouped_score

    # Avoid all--inf softmax rows in padding groups. Their values are masked
    # immediately after reduction and are never committed to the cache.
    pooled_score = torch.where(
        valid_groups.unsqueeze(-1).unsqueeze(-1), pooled_score, 0
    )
    compressed = (pooled_kv * pooled_score.softmax(dim=2)).sum(dim=2)
    compressed.masked_fill_(~valid_groups.unsqueeze(-1), 0)

    # Match the online state's stale-slot semantics exactly: after a boundary,
    # the completed group remains in the working bank and later tokens replace
    # only its prefix until the next boundary.
    completed_idx = (counts - 1).clamp_min(0)
    row_ids = torch.arange(batch, device=device)
    last_completed_kv = grouped_kv[row_ids, completed_idx]
    last_completed_score = grouped_score[row_ids, completed_idx]
    has_completed = counts.gt(0).view(batch, 1, 1)
    base_kv = torch.where(has_completed, last_completed_kv, current_kv)
    base_score = torch.where(has_completed, last_completed_score, current_score)

    remainder = (cursor + lengths).remainder(ratio)
    partial_idx = counts.clamp_max(max_groups - 1)
    partial_kv = grouped_kv[row_ids, partial_idx]
    partial_score = grouped_score[row_ids, partial_idx]
    partial_valid = within_ratio < remainder.unsqueeze(1)
    final_kv = torch.where(partial_valid.unsqueeze(-1), partial_kv, base_kv)
    final_score = torch.where(
        partial_valid.unsqueeze(-1), partial_score, base_score
    )

    if overlap:
        state.kv[:, :ratio].copy_(
            torch.where(has_completed, last_completed_kv, state.kv[:, :ratio])
        )
        state.score[:, :ratio].copy_(
            torch.where(
                has_completed, last_completed_score, state.score[:, :ratio]
            )
        )
        state.kv[:, ratio:].copy_(final_kv)
        state.score[:, ratio:].copy_(final_score)
    else:
        state.kv.copy_(final_kv)
        state.score.copy_(final_score)
    return compressed, state, counts


def compress_decode(
    kv: torch.Tensor,
    score: torch.Tensor,
    ape: torch.Tensor,
    ratio: int,
    position: int,
    state: CompressorState,
) -> torch.Tensor | None:
    """Consume one decode token and emit a compressed row at each boundary."""
    batch, sequence_length, head_dim, overlap = _validate_inputs(
        kv, score, ape, ratio
    )
    if sequence_length != 1:
        raise ValueError("decode compressor expects exactly one token")
    coff = 1 + overlap
    expected_state = (batch, coff * ratio, coff * head_dim)
    if state.kv.shape != expected_state or state.score.shape != expected_state:
        raise ValueError(f"compressor state must have shape {expected_state}")

    cursor = position % ratio
    score = score + ape[cursor]
    boundary = (position + 1) % ratio == 0
    if overlap:
        state.kv[:, ratio + cursor].copy_(kv[:, 0])
        state.score[:, ratio + cursor].copy_(score[:, 0])
        if not boundary:
            return None
        pooled_kv = torch.cat(
            [state.kv[:, :ratio, :head_dim], state.kv[:, ratio:, head_dim:]],
            dim=1,
        )
        pooled_score = torch.cat(
            [
                state.score[:, :ratio, :head_dim],
                state.score[:, ratio:, head_dim:],
            ],
            dim=1,
        )
        output = (pooled_kv * pooled_score.softmax(dim=1)).sum(
            dim=1, keepdim=True
        )
        state.kv[:, :ratio].copy_(state.kv[:, ratio:])
        state.score[:, :ratio].copy_(state.score[:, ratio:])
        return output

    state.kv[:, cursor].copy_(kv[:, 0])
    state.score[:, cursor].copy_(score[:, 0])
    if not boundary:
        return None
    return (state.kv * state.score.softmax(dim=1)).sum(dim=1, keepdim=True)


def compress_decode_batch(
    kv: torch.Tensor,
    score: torch.Tensor,
    ape: torch.Tensor,
    ratio: int,
    positions: torch.Tensor,
    state: CompressorState,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Consume a heterogeneous one-token decode batch without row loops.

    ``positions`` is the absolute position of each row.  The returned boolean
    mask identifies rows that completed a compression group; callers commit
    only those rows to the paged compressed cache.
    """
    batch, sequence_length, head_dim, overlap = _validate_inputs(
        kv, score, ape, ratio
    )
    if sequence_length != 1:
        raise ValueError("decode compressor expects exactly one token")
    if positions.ndim != 1 or positions.numel() != batch:
        raise ValueError("positions must have one entry per decode row")
    coff = 1 + overlap
    expected_state = (batch, coff * ratio, coff * head_dim)
    if state.kv.shape != expected_state or state.score.shape != expected_state:
        raise ValueError(f"compressor state must have shape {expected_state}")

    positions = positions.to(device=kv.device, dtype=torch.long)
    return compress_decode_batch_fused(
        kv, score, ape, ratio, positions, state.kv, state.score
    )


class DeepseekV4Compressor(torch.nn.Module):
    """Learned V4 KV compressor with explicit, externally owned state.

    State is passed in/out instead of being indexed by transient batch row.
    The serving runtime can therefore bind it to a request-owned arena slot
    without changing this numerical implementation.
    """

    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        rope_dim: int,
        compress_ratio: int,
        *,
        norm_eps: float,
        rotate: bool = False,
    ) -> None:
        super().__init__()
        if compress_ratio not in (4, 128):
            raise ValueError("DeepSeek-V4 compression ratio must be 4 or 128")
        self.head_dim = head_dim
        self.rope_dim = rope_dim
        self.compress_ratio = compress_ratio
        self.overlap = compress_ratio == 4
        self.rotate = rotate
        self.norm_eps = norm_eps
        channels = (1 + self.overlap) * head_dim
        # Official V4 pooling promotes checkpoint BF16 weights and activations
        # to FP32 before both projections.
        self.wkv = ReplicatedLinear(
            hidden_size,
            channels,
            bias=False,
            params_dtype=torch.float32,
        )
        self.wgate = ReplicatedLinear(
            hidden_size,
            channels,
            bias=False,
            params_dtype=torch.float32,
        )
        self.ape = torch.nn.Parameter(
            torch.empty(
                compress_ratio,
                channels,
                dtype=torch.float32,
                device="cuda",
            ),
            requires_grad=False,
        )
        self.norm_weight = torch.nn.Parameter(
            torch.ones(head_dim, dtype=torch.float32, device="cuda"),
            requires_grad=False,
        )


    def project(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_fp32 = hidden_states.float()
        return self.wkv(hidden_fp32), self.wgate(hidden_fp32)

    def _finalize(
        self,
        compressed: torch.Tensor | None,
        frequencies: torch.Tensor,
    ) -> torch.Tensor | None:
        if compressed is None:
            return None
        value = compressed.float()
        value = value * torch.rsqrt(
            value.square().mean(-1, keepdim=True) + self.norm_eps
        )
        value = (value * self.norm_weight.float()).to(torch.bfloat16)
        apply_rope_inplace(value[..., -self.rope_dim :], frequencies)
        if self.rotate:
            # Lazy import avoids coupling the compressor primitive to the
            # higher-level indexer module at import time.
            from gllm.layers.attention.deepseek_v4.indexer import (
                mxfp4_fake_quantize,
                normalized_hadamard,
            )

            return mxfp4_fake_quantize(normalized_hadamard(value))
        fp8_fake_quantize_inplace(value[..., : -self.rope_dim], group_size=64)
        return value

    def prefill(
        self,
        hidden_states: torch.Tensor,
        frequencies: torch.Tensor,
    ) -> tuple[torch.Tensor | None, CompressorState]:
        kv, score = self.project(hidden_states)
        compressed, state = compress_prefill(
            kv, score, self.ape, self.compress_ratio
        )
        count = hidden_states.shape[1] // self.compress_ratio
        if frequencies.shape[0] != count:
            raise ValueError(
                f"compressor prefill needs {count} RoPE rows, got "
                f"{frequencies.shape[0]}"
            )
        return self._finalize(compressed, frequencies), state

    def prefill_batch(
        self,
        hidden_states: torch.Tensor,
        frequencies: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor | None, CompressorState, torch.Tensor]:
        """Vectorized variable-length counterpart of :meth:`prefill`."""
        kv, score = self.project(hidden_states)
        compressed, state, counts = compress_prefill_batch(
            kv, score, self.ape, self.compress_ratio, lengths
        )
        expected = hidden_states.shape[1] // self.compress_ratio
        if frequencies.shape[0] != expected:
            raise ValueError(
                f"compressor prefill needs {expected} RoPE rows, got "
                f"{frequencies.shape[0]}"
            )
        return self._finalize(compressed, frequencies), state, counts

    def prefill_continue_batch(
        self,
        hidden_states: torch.Tensor,
        frequencies: torch.Tensor,
        *,
        starts: torch.Tensor,
        lengths: torch.Tensor,
        state: CompressorState,
    ) -> tuple[torch.Tensor, CompressorState, torch.Tensor]:
        """Vectorized continuation-prefill update for request-owned state."""
        kv, score = self.project(hidden_states)
        compressed, state, counts = compress_prefill_continue_batch(
            kv,
            score,
            self.ape,
            self.compress_ratio,
            starts,
            lengths,
            state,
        )
        if frequencies.shape[:2] != compressed.shape[:2]:
            raise ValueError(
                "continuation prefill frequencies must match compressed groups: "
                f"{frequencies.shape[:2]} != {compressed.shape[:2]}"
            )
        return self._finalize(compressed, frequencies), state, counts

    def decode(
        self,
        hidden_states: torch.Tensor,
        frequency: torch.Tensor,
        *,
        position: int,
        state: CompressorState,
    ) -> torch.Tensor | None:
        kv, score = self.project(hidden_states)
        compressed = compress_decode(
            kv,
            score,
            self.ape,
            self.compress_ratio,
            position,
            state,
        )
        if compressed is None:
            return None
        if frequency.ndim == 1:
            frequency = frequency.unsqueeze(0)
        return self._finalize(compressed, frequency)

    def decode_batch(
        self,
        hidden_states: torch.Tensor,
        frequencies: torch.Tensor,
        *,
        positions: torch.Tensor,
        state: CompressorState,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Vectorized heterogeneous-position counterpart of :meth:`decode`."""
        kv, score = self.project(hidden_states)
        compressed, boundary = compress_decode_batch(
            kv,
            score,
            self.ape,
            self.compress_ratio,
            positions,
            state,
        )
        return self._finalize(compressed, frequencies), boundary


__all__ = [
    "CompressorState",
    "DeepseekV4Compressor",
    "compress_decode",
    "compress_decode_batch",
    "compress_prefill",
    "compress_prefill_batch",
    "compress_prefill_continue_batch",
    "make_compressor_state",
]
