import pytest
import torch

from gllm.layers.moe.topk import deepseek_v4_topk


def _reference_route(
    logits,
    topk,
    *,
    renormalize=True,
    scale=1.0,
    correction_bias=None,
    input_ids=None,
    tid2eid=None,
):
    scores = torch.nn.functional.softplus(logits.float()).sqrt()
    if tid2eid is None:
        selection_scores = (
            scores
            if correction_bias is None
            else scores + correction_bias.float().unsqueeze(0)
        )
        indices = selection_scores.topk(topk, dim=-1).indices
        weights = scores.gather(1, indices)
    else:
        indices = tid2eid[input_ids.reshape(-1)]
        weights = scores.gather(1, indices)
    if renormalize:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights * scale, indices


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_deepseek_v4_sqrtsoftplus_topk_matches_reference(dtype):
    torch.manual_seed(7)
    logits = torch.randn(17, 256, dtype=dtype)

    actual_weights, actual_ids = deepseek_v4_topk(
        logits,
        6,
        renormalize=True,
        routed_scaling_factor=1.5,
    )
    expected_weights, expected_ids = _reference_route(
        logits, 6, renormalize=True, scale=1.5
    )

    torch.testing.assert_close(actual_weights, expected_weights, rtol=0, atol=0)
    torch.testing.assert_close(actual_ids, expected_ids.to(torch.int32), rtol=0, atol=0)


def test_deepseek_v4_hash_topk_matches_reference():
    torch.manual_seed(11)
    logits = torch.randn(9, 256, dtype=torch.bfloat16)
    input_ids = torch.tensor([[0, 3, 9], [12, 21, 34], [55, 89, 127]])
    tid2eid = torch.randint(0, 256, (128, 6), dtype=torch.int64)

    actual_weights, actual_ids = deepseek_v4_topk(
        logits,
        6,
        renormalize=True,
        routed_scaling_factor=1.5,
        input_ids=input_ids,
        hash_indices_table=tid2eid,
    )
    expected_weights, expected_ids = _reference_route(
        logits,
        6,
        renormalize=True,
        scale=1.5,
        input_ids=input_ids,
        tid2eid=tid2eid,
    )

    torch.testing.assert_close(actual_weights, expected_weights, rtol=0, atol=0)
    torch.testing.assert_close(actual_ids, expected_ids.to(torch.int32), rtol=0, atol=0)


def test_deepseek_v4_correction_bias_only_changes_selection():
    logits = torch.tensor([[5.0, 4.0, 0.0, -1.0]])
    bias = torch.tensor([-10.0, 0.0, 20.0, 0.0])

    actual_weights, actual_ids = deepseek_v4_topk(
        logits,
        2,
        renormalize=True,
        routed_scaling_factor=1.5,
        correction_bias=bias,
    )
    expected_weights, expected_ids = _reference_route(
        logits,
        2,
        renormalize=True,
        scale=1.5,
        correction_bias=bias,
    )

    torch.testing.assert_close(actual_weights, expected_weights, rtol=0, atol=0)
    torch.testing.assert_close(actual_ids, expected_ids.to(torch.int32), rtol=0, atol=0)
    assert set(actual_ids[0].tolist()) == {1, 2}


def test_deepseek_v4_hash_topk_validates_inputs():
    logits = torch.randn(2, 8)
    table = torch.zeros(16, 2, dtype=torch.int64)

    with pytest.raises(ValueError, match="requires input_ids"):
        deepseek_v4_topk(logits, 2, hash_indices_table=table)
    with pytest.raises(ValueError, match="one input id"):
        deepseek_v4_topk(
            logits,
            2,
            input_ids=torch.tensor([1]),
            hash_indices_table=table,
        )
