from types import SimpleNamespace

import pytest
import torch

from gllm.layers.attention.deepseek_v4.ops import precompute_rope_frequencies
from gllm.layers.attention.deepseek_v4.projection import (
    DeepseekV4AttentionProjections,
)


FP8_CONFIG = {
    "quant_method": "fp8",
    "activation_scheme": "dynamic",
    "weight_block_size": [128, 128],
    "scale_fmt": "ue8m0",
}


def _config():
    return SimpleNamespace(
        hidden_size=512,
        num_attention_heads=4,
        head_dim=128,
        qk_rope_head_dim=64,
        q_lora_rank=128,
        o_lora_rank=128,
        o_groups=2,
        rms_norm_eps=1e-6,
        quantization_config=FP8_CONFIG,
    )


def _load_fp8(linear, weight):
    from deep_gemm.utils import per_block_cast_to_fp8

    q, scale = per_block_cast_to_fp8(weight, use_ue8m0=True, gran_k=128)
    linear.weight.data.copy_(q)
    linear.weight_scale_inv.data.copy_(scale)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_deepseek_v4_attention_projection_shapes_and_roundtrip():
    if torch.cuda.get_device_capability()[0] not in (10, 12):
        pytest.skip("native block FP8 path requires Blackwell")
    pytest.importorskip("deep_gemm")

    torch.manual_seed(37)
    config = _config()
    module = DeepseekV4AttentionProjections(config)
    for linear in (module.wq_a, module.wq_b, module.wkv, module.wo_a, module.wo_b):
        weight = torch.randn_like(
            linear.weight, dtype=torch.bfloat16
        ) * (0.03 if linear is not module.wo_b else 0.01)
        _load_fp8(linear, weight)
    module.q_norm.weight.data.uniform_(0.9, 1.1)
    module.kv_norm.weight.data.uniform_(0.9, 1.1)

    hidden = torch.randn(
        2, 8, config.hidden_size, device="cuda", dtype=torch.bfloat16
    ) * 0.2
    frequencies = precompute_rope_frequencies(
        config.qk_rope_head_dim,
        8,
        original_sequence_length=0,
        base=10000.0,
        factor=1.0,
        beta_fast=32,
        beta_slow=1,
        device="cuda",
    )
    qr, q, kv = module.prepare_q_kv(hidden, frequencies)
    assert qr.shape == (2, 8, 128)
    assert q.shape == (2, 8, 4, 128)
    assert kv.shape == (2, 8, 128)
    torch.testing.assert_close(
        q.float().square().mean(-1).mean(),
        torch.tensor(1.0, device="cuda"),
        rtol=0.02,
        atol=0.02,
    )

    output = module.project_output(q.clone(), frequencies)
    assert output.shape == hidden.shape
    assert torch.isfinite(output).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_grouped_wo_a_does_not_mix_output_groups():
    if torch.cuda.get_device_capability()[0] not in (10, 12):
        pytest.skip("native block FP8 path requires Blackwell")
    pytest.importorskip("deep_gemm")

    config = _config()
    module = DeepseekV4AttentionProjections(config)
    # Two deliberately different group matrices. If the implementation treats
    # wo_a as one ordinary dense matrix, changing group 0 input leaks into 1.
    w = torch.zeros(256, 256, device="cuda", dtype=torch.bfloat16)
    w[:128].fill_diagonal_(1.0)
    w[128:].fill_diagonal_(2.0)
    _load_fp8(module.wo_a, w)
    x = torch.zeros(1, 1, 2, 256, device="cuda", dtype=torch.bfloat16)
    x[..., 0, :128] = 1
    x[..., 1, :128] = 3
    actual = module._grouped_wo_a(x)
    torch.testing.assert_close(
        actual[..., 0, :], torch.ones_like(actual[..., 0, :]), rtol=0, atol=0
    )
    torch.testing.assert_close(
        actual[..., 1, :], torch.full_like(actual[..., 1, :], 6), rtol=0, atol=0
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_grouped_wo_a_reuses_group_views():
    """Re-slicing per forward silently defeats the packed-scale memo.

    ``fp8LinearMethod`` caches DeepGEMM's packed weight scale *on the scale
    tensor*. A fresh view each call caches onto a temporary, so the pack is
    redone every decode step -- measurable, but never an error.
    """
    if torch.cuda.get_device_capability()[0] not in (10, 12):
        pytest.skip("native block FP8 path requires Blackwell")
    config = _config()
    projections = DeepseekV4AttentionProjections(config).cuda()
    # Real UE8M0 scales: DeepGEMM's packer asserts (device-side, so it does not
    # raise) that every scale is an exact power of two.
    _load_fp8(
        projections.wo_a,
        torch.randn(
            config.o_groups * config.o_lora_rank,
            projections.group_width,
            device="cuda",
            dtype=torch.bfloat16,
        ),
    )
    grouped = torch.randn(
        2,
        1,
        projections.local_num_groups,
        projections.group_width,
        device="cuda",
        dtype=torch.bfloat16,
    )

    projections._grouped_wo_a(grouped)
    first_weight, first_scales = projections._group_views
    projections._grouped_wo_a(grouped)
    again_weight, again_scales = projections._group_views

    # Same objects across forwards, so anything memoized on them survives.
    for a, b in zip(first_scales, again_scales):
        assert a is b
    for a, b in zip(first_weight, again_weight):
        assert a is b

    from gllm.layers.quantization.fp8 import _PACKED_SCALE_ATTR

    assert all(
        getattr(s, _PACKED_SCALE_ATTR, None) is not None for s in again_scales
    ), "the packed scale must stick to the retained view"
