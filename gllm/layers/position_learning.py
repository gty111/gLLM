# SPDX-License-Identifier: Apache-2.0
# Source reference: vLLM's Qwen4Exp ngram_embedding.py and ops/ple.py (Apache-2.0).
"""N-gram hashing and eager PLE with caller-owned streaming state.

State belongs to a request, never to the module. Chunked prefill and decode
use the same causal convolution. This reference does not allocate the huge
embedding table: callers pass looked-up embeddings to the PLE layer.
"""

import itertools

import torch
import torch.nn.functional as F
from torch import nn

from gllm.layers.hyper_connection import GroupedGemmaRMSNorm


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    for prime in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if value % prime == 0:
            return value == prime
    exponent, shifts = value - 1, 0
    while exponent % 2 == 0:
        exponent //= 2
        shifts += 1
    for base in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
        if base % value == 0:
            continue
        witness = pow(base, exponent, value)
        if witness in (1, value - 1):
            continue
        for _ in range(shifts - 1):
            witness = pow(witness, 2, value)
            if witness == value - 1:
                break
        else:
            return False
    return True


def ngram_vocab_layout(base: int, heads: int, dense_layer_id: int = 0):
    """Return prime vocabulary sizes, cumulative offsets and unpadded rows."""
    if base <= 0 or heads <= 0 or dense_layer_id < 0:
        raise ValueError("Invalid ngram vocabulary layout")
    sizes, offsets, total = [], [], 0
    candidate = base
    for head in range((dense_layer_id + 1) * heads):
        while not _is_prime(candidate):
            candidate += 1
        if head >= dense_layer_id * heads:
            sizes.append(candidate)
            offsets.append(total)
            total += candidate
        candidate += 1
    return sizes, offsets, total


def ngram_multipliers(ngram_size, vocab_size, dense_layer_id=0, seed=1234):
    """Deterministic fallback; checkpoint layer_multipliers remain authoritative."""
    if ngram_size < 2 or vocab_size <= 0 or vocab_size > (1 << 63) - 1:
        raise ValueError("Invalid ngram size or unigram vocabulary")
    mask = (1 << 64) - 1
    gamma = 0x9E3779B97F4A7C15
    bound = max(1, (((1 << 63) - 1) // vocab_size) // 2)
    result = []
    for i in range(ngram_size):
        value = seed + 10007 * dense_layer_id + gamma * (i + 1)
        value = (value + gamma) & mask
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
        value = (value ^ (value >> 31)) & mask
        result.append(2 * (value % bound) + 1)
    return result


class NGramHash(nn.Module):
    """Packed-request hash with explicit oldest-to-newest predecessor tokens."""

    def __init__(self, config, dense_layer_id=0, *, device=None):
        super().__init__()
        self.ngram_size = config.ngram_size
        self.heads_per_ngram = config.heads_per_ngram
        self.eos_token_id = config.eos_token_id
        sizes, offsets, self.num_rows = ngram_vocab_layout(
            config.ngram_vocab_size_base, config.ngram_heads, dense_layer_id
        )
        for name, values in (
            (
                "layer_multipliers",
                ngram_multipliers(
                    config.ngram_size, config.vocab_size, dense_layer_id, config.seed
                ),
            ),
            ("ngram_heads_vocab_sizes", sizes),
            ("ngram_heads_offsets", offsets),
        ):
            self.register_buffer(
                name, torch.tensor(values, dtype=torch.long, device=device)
            )

    def forward(self, tokens, query_start_loc, context):
        """Return [tokens, ngram_heads] table IDs without modifying context.

        Context [requests, ngram_size-1] is left-padded with EOS for new
        requests. No predecessor before the most recent EOS may affect a hash.
        This eager oracle performs host synchronization; it is not graph-safe.
        """
        if tokens.ndim != 1 or query_start_loc.ndim != 1:
            raise ValueError("Expected packed tokens and one-dimensional boundaries")
        boundaries = query_start_loc.tolist()
        if (
            not boundaries
            or boundaries[0] != 0
            or boundaries[-1] != tokens.numel()
            or any(a > b for a, b in itertools.pairwise(boundaries))
        ):
            raise ValueError("Invalid packed request boundaries")
        if context.shape != (len(boundaries) - 1, self.ngram_size - 1):
            raise ValueError("Context must contain one ngram history per request")
        blocks = []
        for request, (start, end) in enumerate(itertools.pairwise(boundaries)):
            sequence = torch.cat(
                (context[request].to(tokens.device).long(), tokens[start:end].long())
            )
            positions = (
                torch.arange(end - start, device=tokens.device) + self.ngram_size - 1
            )
            current = sequence[positions]
            mixed = current * self.layer_multipliers[0]
            crossed = torch.zeros_like(current, dtype=torch.bool)
            heads = []
            for shift in range(1, self.ngram_size):
                predecessor = sequence[positions - shift]
                predecessor = torch.where(crossed, self.eos_token_id, predecessor)
                crossed = crossed | (predecessor == self.eos_token_id)
                mixed = torch.bitwise_xor(
                    mixed, predecessor * self.layer_multipliers[shift]
                )
                head_start = (shift - 1) * self.heads_per_ngram
                head_end = head_start + self.heads_per_ngram
                heads.append(
                    torch.remainder(
                        mixed[:, None],
                        self.ngram_heads_vocab_sizes[head_start:head_end],
                    )
                    + self.ngram_heads_offsets[head_start:head_end]
                )
            blocks.append(torch.cat(heads, dim=-1))
        if not blocks:
            return tokens.new_empty(
                (0, self.ngram_heads_vocab_sizes.numel()), dtype=torch.long
            )
        return torch.cat(blocks, dim=0)

    def forward_packed(self, tokens, query_start_loc, context):
        """GPU hash for packed requests; boundaries and history are device inputs."""
        positions = torch.arange(tokens.numel(), device=tokens.device)
        requests = torch.bucketize(positions, query_start_loc[1:], right=True)
        starts = query_start_loc[requests].long()
        relative = positions - starts
        mixed = tokens.long() * self.layer_multipliers[0]
        crossed = torch.zeros_like(tokens, dtype=torch.bool)
        heads = []
        for shift in range(1, self.ngram_size):
            predecessor = tokens[(positions - shift).clamp_min(0)]
            context_column = (self.ngram_size - 1 + relative - shift).clamp(
                0, self.ngram_size - 2
            )
            predecessor = torch.where(
                relative >= shift, predecessor, context[requests, context_column]
            )
            predecessor = torch.where(crossed, self.eos_token_id, predecessor)
            crossed = crossed | (predecessor == self.eos_token_id)
            mixed = torch.bitwise_xor(
                mixed, predecessor * self.layer_multipliers[shift]
            )
            lo = (shift - 1) * self.heads_per_ngram
            hi = lo + self.heads_per_ngram
            heads.append(
                torch.remainder(mixed[:, None], self.ngram_heads_vocab_sizes[lo:hi])
                + self.ngram_heads_offsets[lo:hi]
            )
        return torch.cat(heads, -1)


def packed_history_tail(inputs, query_start_loc, history):
    """Last history-length rows per packed request, including short chunks."""
    length = history.shape[1]
    end = query_start_loc[1:].long()
    start = query_start_loc[:-1].long()
    offsets = torch.arange(length, device=inputs.device)
    positions = end[:, None] - length + offsets
    from_input = positions >= start[:, None]
    current = inputs[positions.clamp(0, max(inputs.shape[0] - 1, 0))]
    previous_positions = (length + positions - start[:, None]).clamp(0, length - 1)
    previous = history.gather(
        1,
        previous_positions.reshape(
            *previous_positions.shape, *([1] * (history.ndim - 2))
        ).expand_as(current),
    )
    mask = from_input.reshape(*from_input.shape, *([1] * (history.ndim - 2)))
    return torch.where(mask, current, previous)


def packed_token_histories(inputs, query_start_loc, history):
    """Post-token histories for speculative rollback, including short prefixes."""
    length = history.shape[1]
    positions = torch.arange(inputs.shape[0], device=inputs.device)
    requests = torch.bucketize(positions, query_start_loc[1:], right=True)
    starts = query_start_loc[requests].long()
    previous = (
        positions[:, None] + 1 - length + torch.arange(length, device=inputs.device)
    )
    from_input = previous >= starts[:, None]
    current = inputs[previous.clamp_min(0)]
    context_positions = (length + previous - starts[:, None]).clamp(0, length - 1)
    old = history[requests[:, None], context_positions]
    mask = from_input.reshape(*from_input.shape, *([1] * (history.ndim - 2)))
    return torch.where(mask, current, old)


def dilated_causal_conv(inputs, weight, history=None, *, dilation):
    """[tokens, channels] -> (conv output, new oldest-to-newest history).

    ``weight`` is checkpoint-native [channels, 1, kernel]. Caller supplies
    per-request history [(kernel-1)*dilation, channels]; new history owns its
    storage and never retains an entire prefill allocation.
    """
    if inputs.ndim != 2 or weight.ndim != 3 or weight.shape[1] != 1:
        raise ValueError("Expected token/channel input and depthwise conv weights")
    if dilation <= 0 or weight.shape[0] != inputs.shape[1] or weight.shape[-1] <= 0:
        raise ValueError("Invalid depthwise convolution dimensions")
    window = (weight.shape[-1] - 1) * dilation
    if history is None:
        history = inputs.new_zeros(window, inputs.shape[1])
    if history.shape != (window, inputs.shape[1]):
        raise ValueError("Incorrect PLE convolution history length")
    combined = torch.cat((history, inputs), dim=0)
    if inputs.shape[0]:
        conv = (
            F.conv1d(
                combined.T.unsqueeze(0),
                weight,
                groups=inputs.shape[1],
                dilation=dilation,
            )
            .squeeze(0)
            .T
        )
    else:
        conv = inputs.clone()
    new_history = (
        combined[-window:].clone() if window else inputs.new_empty(0, inputs.shape[1])
    )
    return conv, new_history


class PLELayer(nn.Module):
    """Checkpoint-native PLE gate and convolution for one request chunk.

    Lookup and request routing are intentionally external. ``embeddings`` is
    [tokens, ple_embed_dim], flattened from ngram-head embedding lookups.
    ``streams`` is [tokens, hc_count*hidden_size].
    """

    def __init__(self, config, *, device=None, dtype=None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.hc_count = config.hc_count
        self.dilation = config.ngram_size
        width = config.hyper_hidden_size
        self.key_proj = nn.Linear(
            config.ple_embed_dim, width, bias=False, device=device, dtype=dtype
        )
        self.value_proj = nn.Linear(
            config.ple_embed_dim,
            config.hidden_size,
            bias=False,
            device=device,
            dtype=dtype,
        )
        for name in ("norm_key", "norm_query", "norm_conv"):
            setattr(
                self,
                name,
                GroupedGemmaRMSNorm(
                    config.hidden_size,
                    config.hc_count,
                    config.rms_norm_eps,
                    device=device,
                    dtype=dtype,
                ),
            )
        self.conv1d = nn.Conv1d(
            width,
            width,
            config.ple_conv_kernel_size,
            groups=width,
            bias=False,
            device=device,
            dtype=dtype,
        )

    def forward(self, streams, embeddings, history=None):
        key = self.norm_key(self.key_proj(embeddings))
        query = self.norm_query(streams)
        dot = (key * query).unflatten(-1, (self.hc_count, self.hidden_size)).sum(-1)
        dot = dot / self.hidden_size**0.5
        gate = (dot.sign() * dot.abs().clamp_min(1e-6).sqrt()).sigmoid()
        value = self.value_proj(embeddings)
        gated = (gate.unsqueeze(-1) * value.unsqueeze(-2)).flatten(-2)
        normalized = self.norm_conv(gated)
        conv, history = dilated_causal_conv(
            normalized, self.conv1d.weight, history, dilation=self.dilation
        )
        # Match checkpoint inference boundaries: first round the PLE branch
        # (gate + convolution), then add the outer four-stream residual.
        return streams + (gated + F.silu(conv)), history

    def forward_packed(
        self, streams, embeddings, boundaries, history, *, return_normalized=False
    ):
        key = self.norm_key(self.key_proj(embeddings))
        query = self.norm_query(streams)
        dot = (key * query).unflatten(-1, (self.hc_count, self.hidden_size)).sum(
            -1
        ) / self.hidden_size**0.5
        gate = (dot.sign() * dot.abs().clamp_min(1e-6).sqrt()).sigmoid()
        value = self.value_proj(embeddings)
        gated = (gate.unsqueeze(-1) * value.unsqueeze(-2)).flatten(-2)
        normalized = self.norm_conv(gated)
        positions = torch.arange(streams.shape[0], device=streams.device)
        requests = torch.bucketize(positions, boundaries[1:], right=True)
        relative = positions - boundaries[requests]
        conv = torch.zeros_like(normalized, dtype=torch.float32)
        length = history.shape[1]
        for tap in range(self.conv1d.weight.shape[-1]):
            lag = (self.conv1d.weight.shape[-1] - 1 - tap) * self.dilation
            current = normalized[(positions - lag).clamp_min(0)]
            previous = history[requests, (length + relative - lag).clamp(0, length - 1)]
            values = torch.where((relative >= lag)[:, None], current, previous)
            conv = conv + values.float() * self.conv1d.weight[:, 0, tap].float()
        output = streams + (gated + F.silu(conv.to(streams.dtype)))
        result = output, packed_history_tail(normalized, boundaries, history)
        return (*result, normalized) if return_normalized else result
