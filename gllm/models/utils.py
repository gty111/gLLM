from typing import Union

import torch
from typing_extensions import TypeAlias


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

