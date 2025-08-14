import torch

from typing import Union
from typing_extensions import TypeAlias

NestedTensors: TypeAlias = Union[list["NestedTensors"], list["torch.Tensor"],
                                 "torch.Tensor", tuple["torch.Tensor", ...]]
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

    return " + ".join(
        _embedding_count_expression(inner) for inner in embeddings)

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
    try:
        # This is equivalent to: inputs_embeds[is_multimodal] = flattened.
        inputs_embeds.masked_scatter_(is_multimodal.unsqueeze(-1),
                                      flattened.to(dtype=inputs_embeds.dtype))
    except RuntimeError as e:
        num_expected_tokens = is_multimodal.sum().item()
        assert isinstance(num_expected_tokens, int)

        if flattened.shape[0] != num_expected_tokens:
            expr = _embedding_count_expression(multimodal_embeddings)
            raise ValueError(
                f"Attempted to assign {expr} = {flattened.shape[0]} "
                f"multimodal tokens to {num_expected_tokens} placeholders"
            ) from e
        else:
            raise ValueError("Error during masked scatter operation") from e

    return inputs_embeds

def merge_multimodal_embeddings(
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    multimodal_embeddings: NestedTensors,
    placeholder_token_id: Union[int, list[int]],
) -> torch.Tensor:
    """
    Merge ``multimodal_embeddings`` into ``inputs_embeds`` by overwriting the
    positions in ``inputs_embeds`` corresponding to placeholder tokens in
    ``input_ids``.

    ``placeholder_token_id`` can be a list of token ids (e.g, token ids
    of img_start, img_break, and img_end tokens) when needed: This means
    the order of these tokens in the ``input_ids`` MUST MATCH the order of
    their embeddings in ``multimodal_embeddings`` since we need to
    slice-merge instead of individually scattering.

    For example, if input_ids is "TTTTTSIIIBIIIBIIIETTT", where
    - T is text token
    - S is image start token
    - I is image embedding token
    - B is image break token
    - E is image end token.

    Then the image embeddings (that correspond to I's) from vision encoder
    must be padded with embeddings of S, B, and E in the same order of
    input_ids for a correct embedding merge.

    Note:
        This updates ``inputs_embeds`` in place.
    """
    if isinstance(placeholder_token_id, list):
        placeholder_token_id = torch.tensor(placeholder_token_id,
                                            device=input_ids.device)
        return _merge_multimodal_embeddings(
            inputs_embeds,
            torch.isin(input_ids, placeholder_token_id),
            multimodal_embeddings,
        )

    return _merge_multimodal_embeddings(
        inputs_embeds,
        (input_ids == placeholder_token_id),
        multimodal_embeddings,
    )