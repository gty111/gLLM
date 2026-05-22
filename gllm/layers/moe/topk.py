from typing import Optional

import torch

from gllm import _custom_ops as ops


def topk_softmax(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: Optional[torch.Tensor],
    gating_output: torch.Tensor,
    renormalize: bool,
) -> tuple[torch.Tensor, ...]:
    """Compute top-k softmax (with optional renormalize) in one launch.

    Renormalisation is forwarded to the kernel via the ``renormalize`` flag
    on ``ops.topk_softmax`` so we don't append a follow-up
    ``topk_weights / topk_weights.sum(...)`` (a reduce + elementwise) that
    would silently fight with whatever the caller does after the fact.
    Profile of Qwen3-VL-30B-A3B-Instruct TP=4 showed the pre-fix path
    spending ~20 ms / 100 decode forwards on these two appended kernels.
    """
    from gllm import _custom_ops as ops

    ops.topk_softmax(
        topk_weights,
        topk_indices,
        token_expert_indices,  # kept for API compat, ignored by sgl_kernel
        gating_output,
        renormalize,
    )
    return topk_weights, topk_indices


def fused_topk_sigmoid(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    e_score_correction_bias: torch.Tensor,
    routed_scaling_factor: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused sigmoid + biased topk via sgl_kernel.topk_sigmoid.

    Used when no group hierarchy is applied (num_expert_group == topk_group == 1).
    The kernel internally:
      1. computes sigmoid(gating_output)
      2. adds correction_bias for ranking
      3. picks topk based on biased values
      4. returns the unbiased sigmoid value as the weight (DeepSeek-V3 noaux_tc semantics)
    routed_scaling_factor is applied here since the kernel doesn't accept it.
    """
    assert hidden_states.size(0) == gating_output.size(0), "Number of tokens mismatch"

    M = gating_output.size(0)
    topk_weights = torch.empty(
        M, topk, dtype=torch.float32, device=hidden_states.device
    )
    topk_ids = torch.empty(M, topk, dtype=torch.int32, device=hidden_states.device)

    # sgl_kernel requires correction_bias to be float32.
    bias = e_score_correction_bias
    if bias.dtype != torch.float32:
        bias = bias.to(torch.float32)

    ops.topk_sigmoid(
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        gating_output=gating_output,
        renormalize=renormalize,
        correction_bias=bias,
    )

    if routed_scaling_factor != 1.0:
        topk_weights = topk_weights * routed_scaling_factor

    return topk_weights, topk_ids


def fused_grouped_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    e_score_correction_bias: torch.Tensor,
    num_expert_group: int = 0,
    topk_group: int = 0,
    scoring_func: str = "softmax",
    routed_scaling_factor: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert hidden_states.size(0) == gating_output.size(0), "Number of tokens mismatch"

    if scoring_func == "softmax":
        scores = torch.softmax(gating_output, dim=-1)
    elif scoring_func == "sigmoid":
        scores = gating_output.sigmoid()
    else:
        raise ValueError(f"Unsupported scoring function: {scoring_func}")

    scores_with_bias = scores + e_score_correction_bias.unsqueeze(0)
    topk_values, topk_indices = ops.grouped_topk(
        scores,
        scores_with_bias.to(scores.dtype),
        num_expert_group,
        topk_group,
        topk,
        renormalize,
        routed_scaling_factor,
    )
    return topk_values.to(torch.float32), topk_indices.to(torch.int32)


# This is used by the Deepseek-V2 and Deepseek-V3 model
def grouped_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int = 0,
    topk_group: int = 0,
    scoring_func: str = "softmax",
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Fast path 1: sgl_kernel.moe_fused_gate
    # Constraints: num_expert_group <= 32, topk <= 32,
    # and per-group experts (num_experts / num_expert_group) <= 32.
    num_experts = gating_output.size(-1)
    per_group_experts = (
        num_experts // num_expert_group if num_expert_group > 0 else num_experts
    )
    if (
        num_expert_group <= 32
        and topk <= 32
        and per_group_experts <= 32
        and e_score_correction_bias is not None
    ):
        return fused_grouped_topk(
            hidden_states=hidden_states,
            gating_output=gating_output,
            topk=topk,
            renormalize=renormalize,
            e_score_correction_bias=e_score_correction_bias,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
        )

    # Fast path 2: sgl_kernel.topk_sigmoid for the no-grouping case (e.g. Moonlight,
    # which has n_group=1 with 64 experts, exceeding moe_fused_gate's per-group limit).
    # When num_expert_group == topk_group == 1, no group filtering happens, so a flat
    # topk over (sigmoid + bias) is mathematically equivalent.
    if (
        num_expert_group == 1
        and topk_group == 1
        and scoring_func == "sigmoid"
        and e_score_correction_bias is not None
    ):
        return fused_topk_sigmoid(
            hidden_states=hidden_states,
            gating_output=gating_output,
            topk=topk,
            renormalize=renormalize,
            e_score_correction_bias=e_score_correction_bias,
            routed_scaling_factor=routed_scaling_factor,
        )

    assert hidden_states.size(0) == gating_output.size(0), "Number of tokens mismatch"

    if scoring_func == "softmax":
        scores = torch.softmax(gating_output, dim=-1)
    elif scoring_func == "sigmoid":
        scores = gating_output.sigmoid()
    else:
        raise ValueError(f"Unsupported scoring function: {scoring_func}")

    num_token = scores.size(0)
    if e_score_correction_bias is not None:
        # Store original scores before applying correction bias. We use biased
        # scores for expert selection but original scores for routing weights
        original_scores = scores
        scores = scores + e_score_correction_bias.unsqueeze(0)
        group_scores = (
            scores.view(num_token, num_expert_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
        )
    else:
        group_scores = (
            scores.view(num_token, num_expert_group, -1).max(dim=-1).values
        )  # [n, n_group]

    # For batch invariance, use sorted=True to ensure deterministic expert selection
    use_sorted = False
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=use_sorted)[
        1
    ]  # [n, top_k_group]
    group_mask = torch.zeros_like(group_scores)  # [n, n_group]
    group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_token, num_expert_group, scores.size(-1) // num_expert_group)
        .reshape(num_token, -1)
    )  # [n, e]
    tmp_scores = scores.masked_fill(~score_mask.bool(), float("-inf"))  # [n, e]

    if e_score_correction_bias is not None:
        topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=use_sorted)[1]
        # Use original unbiased scores for the routing weights
        topk_weights = original_scores.gather(1, topk_ids)
    else:
        topk_weights, topk_ids = torch.topk(
            tmp_scores, k=topk, dim=-1, sorted=use_sorted
        )

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    if routed_scaling_factor != 1.0:
        topk_weights = topk_weights * routed_scaling_factor
    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


def fused_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
):
    """Pure-bf16 / fp16 top-k softmax for ungrouped MoE routers (e.g. Qwen3-MoE).

    The previous implementation had three redundant kernels per MoE layer:
      1. ``gating_output.float()`` -- pointless cast, the sgl-kernel
         topk_softmax handles bf16/fp16 logits directly;
      2. ``topk_weights / topk_weights.sum(dim=-1, keepdim=True)`` -- a
         reduce + an elementwise division to renormalise top-k weights,
         already supported in-kernel via ``renormalize=True``;
      3. (gone with #2) the dead ``token_expert_indicies`` scratch buffer
         that sgl-kernel doesn't consume.

    With 48 MoE layers x 100 decode forwards that's ~15k saved kernel
    launches per profile window on Qwen3-VL-30B-A3B-Instruct TP=4 (= less
    rank-0 CPU pressure between AR ops, on top of the direct ~20 ms GPU
    saving). SGLang's trace shows the same one-launch pattern --
    ``topkGatingSoftmax<bf16, top_k=8, n_expert=128, ...>`` followed by no
    Python-side post-processing.
    """
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

    M, _ = hidden_states.shape

    topk_weights = torch.empty(
        M, topk, dtype=torch.float32, device=hidden_states.device
    )
    topk_ids = torch.empty(M, topk, dtype=torch.int32, device=hidden_states.device)

    topk_softmax(
        topk_weights,
        topk_ids,
        None,  # token_expert_indices: kept in API for vLLM/HF compat, sgl-kernel ignores it
        gating_output,
        renormalize,
    )

    return topk_weights, topk_ids


def select_experts(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    top_k: int,
    renormalize: bool,
    use_grouped_topk: bool,
    topk_group: Optional[int] = None,
    num_expert_group: Optional[int] = None,
    scoring_func: str = "softmax",
    e_score_correction_bias: Optional[torch.Tensor] = None,
):
    # DeepSeekv2 uses grouped_top_k
    if use_grouped_topk:
        assert topk_group is not None
        assert num_expert_group is not None
        topk_weights, topk_ids = grouped_topk(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=top_k,
            renormalize=renormalize,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            scoring_func=scoring_func,
            e_score_correction_bias=e_score_correction_bias,
        )
    else:
        topk_weights, topk_ids = fused_topk(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=top_k,
            renormalize=renormalize,
        )

    return topk_weights, topk_ids
