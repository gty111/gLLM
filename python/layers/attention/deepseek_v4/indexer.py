"""DeepSeek-V4 C4 indexer reference operations."""

from __future__ import annotations

import torch

from gllm.distributed.parallel_state import get_tp_size
from gllm.layers.attention.deepseek_v4.ops import apply_rope_inplace
from gllm.layers.attention.deepseek_v4.compressor import DeepseekV4Compressor
from gllm.layers.linear import ColumnParallelLinear
from gllm.layers.ops.deepseek_v4 import mxfp4_fake_quantize_fused


def normalized_hadamard(x: torch.Tensor) -> torch.Tensor:
    """Apply the exact normalized Hadamard transform used by DeepSeek-V4."""
    width = x.shape[-1]
    if width <= 0 or width & (width - 1):
        raise ValueError(f"Hadamard width must be a power of two, got {width}")
    if x.dtype is not torch.bfloat16:
        raise TypeError(
            "DeepSeek-V4 Hadamard input must be bfloat16, "
            f"got {x.dtype}"
        )
    try:
        from fast_hadamard_transform import hadamard_transform
    except ImportError as error:
        raise ImportError(
            "DeepSeek-V4 requires fast-hadamard-transform. "
            "Install gLLM's pinned requirements to enable this model."
        ) from error
    return hadamard_transform(x, scale=width**-0.5)


def mxfp4_fake_quantize(x: torch.Tensor, group_size: int = 32) -> torch.Tensor:
    """Quantize/dequantize FP4 E2M1 with power-of-two E8M0 group scales."""
    if x.shape[-1] % group_size:
        raise ValueError(
            f"last dimension {x.shape[-1]} must be divisible by {group_size}"
        )
    return mxfp4_fake_quantize_fused(x, group_size)


def indexer_scores(
    query: torch.Tensor,
    compressed_kv: torch.Tensor,
    head_weights: torch.Tensor,
) -> torch.Tensor:
    """Compute the official ReLU-weighted multi-head index scores."""
    if query.ndim != 4 or compressed_kv.ndim != 3 or head_weights.ndim != 3:
        raise ValueError("query, compressed_kv and head_weights must be 4D/3D/3D")
    b, s, h, d = query.shape
    if compressed_kv.shape[0] != b or compressed_kv.shape[2] != d:
        raise ValueError("compressed_kv does not match query")
    if tuple(head_weights.shape) != (b, s, h):
        raise ValueError("head_weights does not match query")
    if not (
        query.dtype == compressed_kv.dtype == head_weights.dtype
        == torch.bfloat16
    ):
        raise TypeError("DeepSeek-V4 indexer score inputs must all be bfloat16")
    # Keep every intermediate in BF16. This is intentional: the official
    # reference does not upcast this einsum or the head reduction, and an FP32
    # helper can change top-k choices for nearby scores.
    logits = torch.einsum("bshd,btd->bsht", query, compressed_kv)
    return (logits.relu_() * head_weights.unsqueeze(-1)).sum(dim=2)


def causal_indexer_topk(
    scores: torch.Tensor,
    *,
    compress_ratio: int,
    start_pos: int,
    topk: int = 512,
    offset: int = 0,
) -> torch.Tensor:
    """Apply C4 causality and select compressed-cache positions."""
    if scores.ndim != 3:
        raise ValueError("index scores must have shape [B,S,T]")
    _, sequence_length, compressed_length = scores.shape
    masked_scores = scores
    if start_pos == 0:
        valid_count = (
            torch.arange(1, sequence_length + 1, device=scores.device)
            // compress_ratio
        )
        candidates = torch.arange(compressed_length, device=scores.device)
        masked_scores = scores.masked_fill(
            candidates.view(1, 1, -1) >= valid_count.view(1, -1, 1),
            -torch.inf,
        )
    selected = masked_scores.topk(min(topk, compressed_length), dim=-1).indices
    if start_pos == 0:
        valid_count = (
            torch.arange(1, sequence_length + 1, device=scores.device)
            // compress_ratio
        )
        selected = torch.where(
            selected >= valid_count.view(1, -1, 1),
            -1,
            selected + offset,
        )
    else:
        selected = selected + offset
    return selected.to(torch.int32)


def prepare_indexer_query_and_weights(
    query: torch.Tensor,
    head_weights: torch.Tensor,
    *,
    num_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Hadamard+FP4 QAT and the indexer's score-weight scaling."""
    # Keep this helper valid for either full or partitioned head tensors while
    # always applying the official global-head normalization.
    if head_weights.shape != query.shape[:-1]:
        raise ValueError("query/head_weights local shapes do not match")
    if num_heads <= 0 or num_heads % query.shape[-2]:
        raise ValueError("global num_heads must be a multiple of local heads")
    query = mxfp4_fake_quantize(normalized_hadamard(query))
    scale = query.shape[-1] ** -0.5 * num_heads**-0.5
    return query, head_weights * scale


class DeepseekV4Indexer(torch.nn.Module):
    """C4 lightning indexer with externally managed compressed cache/state."""

    def __init__(self, config) -> None:
        super().__init__()
        self.num_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_dim = config.qk_rope_head_dim
        self.topk = config.index_topk
        self.compress_ratio = 4
        tp_size = get_tp_size()
        if self.num_heads % tp_size:
            raise ValueError("V4 index heads must divide tensor parallelism")
        self.local_num_heads = self.num_heads // tp_size
        self.wq_b = ColumnParallelLinear(
            config.q_lora_rank,
            self.num_heads * self.head_dim,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=config.quantization_config,
        )
        self.weights_proj = ColumnParallelLinear(
            config.hidden_size,
            self.num_heads,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=None,
        )
        self.compressor = DeepseekV4Compressor(
            config.hidden_size,
            self.head_dim,
            self.rope_dim,
            self.compress_ratio,
            norm_eps=config.rms_norm_eps,
            rotate=True,
        )


    def prepare_query(
        self,
        hidden_states: torch.Tensor,
        q_lora: torch.Tensor,
        frequencies: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = self.wq_b(q_lora).view(
            *q_lora.shape[:2], self.local_num_heads, self.head_dim
        )
        apply_rope_inplace(query[..., -self.rope_dim :], frequencies)
        head_weights = self.weights_proj(hidden_states)
        return prepare_indexer_query_and_weights(
            query,
            head_weights,
            num_heads=self.num_heads,
        )


__all__ = [
    "causal_indexer_topk",
    "DeepseekV4Indexer",
    "indexer_scores",
    "mxfp4_fake_quantize",
    "normalized_hadamard",
    "prepare_indexer_query_and_weights",
]
