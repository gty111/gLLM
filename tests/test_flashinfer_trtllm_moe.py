from types import SimpleNamespace

import pytest
import torch

from gllm.layers.moe.flashinfer_trtllm import (
    bf16_moe_support_reason,
    convert_bf16_moe_weights,
)


def make_layer(**overrides):
    values = {
        "w13_weight": torch.empty(2, 256, 128, dtype=torch.bfloat16),
        "w2_weight": torch.empty(2, 128, 128, dtype=torch.bfloat16),
        "global_num_experts": 2,
        "activation": "silu",
        "apply_router_weight_on_input": False,
        "scoring_func": "softmax",
        "use_grouped_topk": False,
        "e_score_correction_bias": None,
        "intermediate_size_per_partition": 128,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_support_reason_accepts_sm100_bf16(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (10, 0))
    assert bf16_moe_support_reason(make_layer()) is None


@pytest.mark.parametrize(
    ("override", "reason"),
    [
        ({"activation": "gelu"}, "activation"),
        ({"scoring_func": "sigmoid"}, "scoring_func"),
        ({"use_grouped_topk": True}, "grouped"),
        ({"intermediate_size_per_partition": 96}, "aligned"),
    ],
)
def test_support_reason_rejects_unsupported_routes(monkeypatch, override, reason):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (10, 0))
    assert reason in bf16_moe_support_reason(make_layer(**override))


def test_support_reason_falls_back_outside_sm100(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (9, 0))
    assert "SM100" in bf16_moe_support_reason(make_layer())


def test_block_layout_preserves_storage_size():
    w13 = torch.arange(2 * 256 * 128, dtype=torch.float32).to(torch.bfloat16)
    w13 = w13.view(2, 256, 128)
    w2 = torch.arange(2 * 128 * 128, dtype=torch.float32).to(torch.bfloat16)
    w2 = w2.view(2, 128, 128)

    converted_w13, converted_w2 = convert_bf16_moe_weights(w13, w2)

    assert converted_w13.numel() == w13.numel()
    assert converted_w2.numel() == w2.numel()
    assert converted_w13.shape == (2, 2, 256, 64)
    assert converted_w2.shape == (2, 2, 128, 64)
    assert converted_w13.is_contiguous()
    assert converted_w2.is_contiguous()
