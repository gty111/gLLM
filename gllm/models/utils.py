import os
from typing import Union

import torch
from typing_extensions import TypeAlias

from gllm.dist_utils import (
    dp_gather_hidden,
    dp_local_slice,
    ep_all_reduce,
    get_dp_forward_counts,
    get_dp_size,
)


def get_moe_chunk_size() -> int:
    """Max #tokens fed to the routed experts per call on the DP+EP path.

    Under DP attention the experts run over the *gathered* global batch
    (``dp_size x local_tokens``), so the ``FusedMoE`` intermediate activation
    scales with ``dp_size`` and can dominate memory -- squeezing the KV cache
    (and, before profiling was fixed to a real prefill, OOMing outright).

    Splitting the global batch into row-chunks bounds that peak to ``chunk``
    rows regardless of ``dp_size``. Routing is per-token, so processing the rows
    in chunks is bit-identical to one big call. ``<= 0`` disables chunking.

    Configured via ``GLLM_FUSED_MOE_CHUNK_SIZE`` and defaults to 32768. Lower it
    (e.g. 8192 ~ one replica's prefill budget) to bound the peak harder and free
    more KV, raise it (or set 0 to disable) to favor larger, more efficient
    expert GEMMs.
    """
    try:
        return int(os.environ.get("GLLM_FUSED_MOE_CHUNK_SIZE", "32768"))
    except ValueError:
        return 32768


def dp_ep_moe_routed(experts, gate, hidden_states, num_tokens):
    """Routed-expert core for DP-attention + Expert-Parallel MoE (``EP = DP x TP``).

    Under DP attention each replica arrives with only *its own* tokens while the
    routed experts are sharded across the whole DP/EP group. So we:

    1. all-gather every replica's tokens into the global batch;
    2. run this replica's local expert shard over the global batch (``experts``
       must be built with ``reduce_results=False`` -> partial output, rows whose
       top experts live elsewhere are zero), chunking the rows so the peak
       expert activation stays bounded by :func:`get_moe_chunk_size` regardless
       of ``dp_size``;
    3. all-reduce over the EP group so every token accumulates all its top-k
       experts wherever they live;
    4. slice this replica's own rows back out.

    ``gate`` maps the (global) hidden states to router logits; ``experts`` is the
    block's :class:`FusedMoE`. Returns this replica's local routed output. Shared
    experts / scaling are the caller's responsibility (they differ per model).

    ``get_dp_forward_counts`` gives the per-replica row counts (published by the
    worker each iteration); it is ``None`` only during the profile / graph-warmup
    dummy forward, where every replica runs the same batch so a symmetric gather
    (``num_tokens`` per replica) is exact.
    """
    counts = get_dp_forward_counts()
    if counts is None:
        counts = [num_tokens] * get_dp_size()
    global_hidden = dp_gather_hidden(hidden_states, counts)
    router_logits = gate(global_hidden)

    # Chunk the routed-expert forward over the gathered rows. Routing is
    # per-token, so this is exact; it just caps the intermediate activation at
    # ``chunk`` rows instead of the full ``dp_size x local_tokens`` batch.
    # Decode / CUDA-graph batches are small (``dp_size x bucket``) and stay
    # below ``chunk``, so they take the single-call fast path unchanged.
    chunk = get_moe_chunk_size()
    m = global_hidden.shape[0]
    if chunk <= 0 or m <= chunk:
        routed = experts(hidden_states=global_hidden, router_logits=router_logits)
    else:
        routed = None
        for start in range(0, m, chunk):
            end = min(start + chunk, m)
            out = experts(
                hidden_states=global_hidden[start:end],
                router_logits=router_logits[start:end],
            )
            if routed is None:
                # Allocate from the expert output so dtype/width match exactly.
                routed = out.new_empty((m, out.shape[1]))
            routed[start:end] = out

    routed = ep_all_reduce(routed)
    routed = dp_local_slice(routed, counts)
    return routed


def extract_rope_config(config, default_theta: float = 10000.0):
    """Compatibility shim for ``transformers >= 5.0``.

    Two relevant breaking changes vs ``< 5.0``:

    1. ``config.rope_theta`` no longer exists as a top-level attribute on
       most ``*Config`` classes (e.g. ``Qwen3Config``). Accessing it now
       raises ``AttributeError``; the value lives inside ``rope_scaling``
       as ``rope_scaling["rope_theta"]``.
    2. ``config.rope_scaling`` is now auto-populated with a trivial
       ``{"rope_theta": ..., "rope_type": "default"}`` dict even when the
       model checkpoint did not configure any RoPE scaling (``< 5.0``
       returned ``None`` in that case). Existing model loaders that did
       ``if rope_scaling is None`` to decide between vanilla and
       scaled/MRoPE no longer hit the vanilla branch.

    Returns a ``(rope_theta, rope_scaling)`` tuple where ``rope_scaling``
    is normalized back to ``None`` for the vanilla case so existing
    branches keep working unchanged.
    """
    rope_scaling = getattr(config, "rope_scaling", None) or {}
    rope_theta = rope_scaling.get("rope_theta")
    if rope_theta is None:
        rope_theta = getattr(config, "rope_theta", None)
    if rope_theta is None:
        rope_theta = default_theta
    rope_type = rope_scaling.get("rope_type", "default")
    if rope_type == "default" and "mrope_section" not in rope_scaling and "type" not in rope_scaling:
        return rope_theta, None
    return rope_theta, dict(rope_scaling)

NestedTensors: TypeAlias = Union[
    list["NestedTensors"],
    list["torch.Tensor"],
    "torch.Tensor",
    tuple["torch.Tensor", ...],
]
"""
Uses a list instead of a tensor if the dimensions of each element do not match.
"""


def _embedding_count_expression(embeddings: NestedTensors) -> str:
    """
    Constructs a debugging representation of the number of embeddings in the
    NestedTensors.
    """

    if isinstance(embeddings, torch.Tensor):
        return " x ".join([str(dim) for dim in embeddings.shape[:-1]])

    return " + ".join(_embedding_count_expression(inner) for inner in embeddings)


def _flatten_embeddings(embeddings: NestedTensors) -> torch.Tensor:
    """
    Recursively flattens and concatenates NestedTensors on all but the last
    dimension.
    """

    if isinstance(embeddings, torch.Tensor):
        # Flatten all but the last dimension.
        return embeddings.flatten(0, -2)

    return torch.cat(tuple(_flatten_embeddings(t) for t in embeddings))


def _merge_multimodal_embeddings(
    inputs_embeds: torch.Tensor,
    is_multimodal: torch.Tensor,
    multimodal_embeddings: NestedTensors,
) -> torch.Tensor:
    """
    Merge ``multimodal_embeddings`` into ``inputs_embeds`` by overwriting the
    positions in ``inputs_embeds`` corresponding to placeholder tokens in
    ``input_ids``.

    Note:
        This updates ``inputs_embeds`` in place.
    """
    flattened = _flatten_embeddings(multimodal_embeddings)
    num_expected_tokens = is_multimodal.sum().item()

    if flattened.shape[0] != num_expected_tokens:
        expr = _embedding_count_expression(multimodal_embeddings)
        raise ValueError(
            f"Attempted to assign {expr} = {flattened.shape[0]} "
            f"multimodal tokens to {num_expected_tokens} placeholders"
        )

    try:
        # This is equivalent to: inputs_embeds[is_multimodal] = flattened.
        inputs_embeds.masked_scatter_(
            is_multimodal.unsqueeze(-1), flattened.to(dtype=inputs_embeds.dtype)
        )
    except Exception as e:
        raise ValueError("Error during masked scatter operation") from e

    return inputs_embeds

