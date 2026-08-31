from types import SimpleNamespace

import pytest
import torch
from torch import nn

from gllm.layers.deepseek_v4_mhc import mhc_post, mhc_pre
from gllm.models.deepseek_v4 import DeepseekV4DecoderLayer


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
        sliding_window=8,
        compress_ratios=(0,),
        compress_rope_theta=40000.0,
        rope_theta=10000.0,
        max_position_embeddings=32,
        rope_scaling={},
        index_n_heads=4,
        index_head_dim=128,
        index_topk=8,
        n_routed_experts=3,
        num_experts_per_tok=2,
        n_shared_experts=1,
        norm_topk_prob=True,
        routed_scaling_factor=1.5,
        swiglu_limit=10.0,
        num_hash_layers=0,
        vocab_size=128,
        moe_intermediate_size=256,
        hc_mult=4,
        hc_sinkhorn_iters=4,
        hc_eps=1e-6,
    )


class _AttentionStub(nn.Module):
    def forward_prefill_with_cache(self, x, cache):
        return x * 0.75, "prefill-cache"

    def forward_decode(self, x, *, position, cache):
        assert position == 7
        assert cache == "decode-cache"
        return x * 0.75


class _FFNStub(nn.Module):
    def forward(self, x, input_ids):
        return x + input_ids.to(x.dtype).unsqueeze(-1) * 0.01


def _reference(layer, hidden, input_ids):
    attn_input, attn_post, attn_comb = mhc_pre(
        hidden,
        layer.hc_attn_fn,
        layer.hc_attn_scale,
        layer.hc_attn_base,
        norm_eps=layer.norm_eps,
        hc_mult=layer.hc_mult,
        sinkhorn_iters=layer.hc_sinkhorn_iters,
        hc_eps=layer.hc_eps,
    )
    attn_output = layer.attn_norm(attn_input) * 0.75
    hidden = mhc_post(attn_output, hidden, attn_post, attn_comb)
    residual = hidden
    ffn_input, ffn_post, ffn_comb = mhc_pre(
        hidden,
        layer.hc_ffn_fn,
        layer.hc_ffn_scale,
        layer.hc_ffn_base,
        norm_eps=layer.norm_eps,
        hc_mult=layer.hc_mult,
        sinkhorn_iters=layer.hc_sinkhorn_iters,
        hc_eps=layer.hc_eps,
    )
    ffn_output = layer.ffn_norm(ffn_input)
    ffn_output = ffn_output + input_ids.to(ffn_output.dtype).unsqueeze(-1) * 0.01
    return mhc_post(ffn_output, residual, ffn_post, ffn_comb)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_deepseek_v4_block_matches_official_mhc_operation_order():
    torch.manual_seed(317)
    layer = DeepseekV4DecoderLayer(0, _config())
    layer.attn = _AttentionStub()
    layer.ffn = _FFNStub()
    layer.attn_norm.weight.data.normal_(mean=1.0, std=0.02)
    layer.ffn_norm.weight.data.normal_(mean=1.0, std=0.02)
    for name in ("hc_attn_fn", "hc_ffn_fn"):
        getattr(layer, name).data.normal_(std=0.01)
    for name in ("hc_attn_base", "hc_ffn_base"):
        getattr(layer, name).data.normal_(std=0.02)
    for name in ("hc_attn_scale", "hc_ffn_scale"):
        getattr(layer, name).data.normal_(std=0.1)

    hidden = torch.randn(
        1, 2, 4, 512, device="cuda", dtype=torch.bfloat16
    ) * 0.2
    input_ids = torch.tensor([[3, 5]], device="cuda")
    expected = _reference(layer, hidden, input_ids)

    prefill, cache = layer.forward_prefill(hidden, input_ids)
    torch.testing.assert_close(prefill, expected, rtol=0, atol=0)
    assert cache == "prefill-cache"

    decoded = layer.forward_decode(
        hidden,
        input_ids,
        position=7,
        cache="decode-cache",
    )
    torch.testing.assert_close(decoded, expected, rtol=0, atol=0)
