from types import SimpleNamespace

import pytest
import torch
from torch import nn

from gllm.layers.attention.deepseek_v4.ops import sparse_attention_reference
from gllm.layers.attention.deepseek_v4.dspark import DeepseekV4DSparkAttention
from gllm.layers.deepseek_v4_mhc import mhc_head, mhc_post, mhc_pre
from gllm.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from gllm.models.deepseek_v4_dspark import DeepseekV4DSpark, DeepseekV4DSparkBlock


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
        rope_theta=10000.0,
        max_position_embeddings=32,
        moe_intermediate_size=256,
        n_routed_experts=3,
        num_experts_per_tok=2,
        n_shared_experts=1,
        norm_topk_prob=True,
        routed_scaling_factor=1.5,
        swiglu_limit=10.0,
        num_hash_layers=1,
        vocab_size=128,
        hc_mult=4,
        hc_sinkhorn_iters=4,
        hc_eps=1e-6,
        num_hidden_layers=3,
        dspark_block_size=5,
        dspark_noise_token_id=127,
        dspark_target_layer_ids=(0, 1, 2),
        dspark_markov_rank=128,
    )


def _load_fp8(linear):
    from deep_gemm.utils import per_block_cast_to_fp8

    weight = torch.randn_like(linear.weight, dtype=torch.bfloat16) * 0.02
    q, scale = per_block_cast_to_fp8(weight, use_ue8m0=True, gran_k=128)
    linear.weight.data.copy_(q)
    linear.weight_scale_inv.data.copy_(scale)


def _initialize(module):
    for linear in (
        module.projections.wq_a,
        module.projections.wq_b,
        module.projections.wkv,
        module.projections.wo_a,
        module.projections.wo_b,
    ):
        _load_fp8(linear)
    module.projections.q_norm.weight.data.fill_(1)
    module.projections.kv_norm.weight.data.fill_(1)
    module.attn_sink.data.normal_(std=0.1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_dspark_prefill_populates_official_circular_window():
    if torch.cuda.get_device_capability()[0] not in (10, 12):
        pytest.skip("native block FP8 path requires Blackwell")
    pytest.importorskip("deep_gemm")
    torch.manual_seed(503)
    module = DeepseekV4DSparkAttention(43, _config())
    _initialize(module)
    main = torch.randn(1, 11, 512, device="cuda", dtype=torch.bfloat16) * 0.2

    frequencies = module.frequencies[: main.shape[1]]
    expected_kv = module.projections.prepare_kv(main, frequencies)
    expected = torch.zeros(1, 8, 128, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(3, 11, device="cuda")
    expected[:, positions % 8] = expected_kv[:, -8:]

    cache = module.prefill_main(main)
    torch.testing.assert_close(cache.window, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_dspark_draft_attention_matches_official_reference_composition():
    if torch.cuda.get_device_capability()[0] not in (10, 12):
        pytest.skip("native block FP8 path requires Blackwell")
    pytest.importorskip("deep_gemm")
    torch.manual_seed(509)
    module = DeepseekV4DSparkAttention(43, _config())
    _initialize(module)
    prompt = torch.randn(1, 6, 512, device="cuda", dtype=torch.bfloat16) * 0.2
    cache = module.prefill_main(prompt)
    expected_window = cache.window.clone()
    current = torch.randn(1, 1, 512, device="cuda", dtype=torch.bfloat16) * 0.2
    draft = torch.randn(1, 5, 512, device="cuda", dtype=torch.bfloat16) * 0.2

    main_kv = module.projections.prepare_kv(current, module.frequencies[6:7])
    expected_window[:, 6] = main_kv[:, 0]
    draft_freq = module.frequencies[7:12]
    _, query, draft_kv = module.projections.prepare_q_kv(draft, draft_freq)
    indices = torch.cat(
        [
            torch.arange(7, device="cuda"),
            8 + torch.arange(5, device="cuda"),
        ]
    ).to(torch.int32).view(1, 1, -1).expand(1, 5, -1).contiguous()
    output = sparse_attention_reference(
        query,
        torch.cat([expected_window, draft_kv], dim=1),
        indices,
        module.attn_sink,
        module.softmax_scale,
    )
    expected = module.projections.project_output(output, draft_freq)

    actual = module.forward_draft_block(
        draft,
        current,
        start_pos=6,
        cache=cache,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(cache.window, expected_window, rtol=0, atol=0)


class _DSparkAttentionStub(nn.Module):
    def forward_draft_block(self, x, main_x, *, start_pos, cache):
        assert start_pos == 9
        assert cache == "cache"
        return x * 0.75 + main_x * 0.125


class _FFNStub(nn.Module):
    def forward(self, x, input_ids):
        return x + input_ids.to(x.dtype).unsqueeze(-1) * 0.01


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_dspark_block_matches_official_mhc_operation_order():
    torch.manual_seed(521)
    config = _config()
    block = DeepseekV4DSparkBlock(0, config)
    block.attn = _DSparkAttentionStub()
    block.ffn = _FFNStub()
    block.attn_norm.weight.data.normal_(mean=1.0, std=0.02)
    block.ffn_norm.weight.data.normal_(mean=1.0, std=0.02)
    for name in ("hc_attn_fn", "hc_ffn_fn"):
        getattr(block, name).data.normal_(std=0.01)
    for name in ("hc_attn_base", "hc_ffn_base"):
        getattr(block, name).data.normal_(std=0.02)
    for name in ("hc_attn_scale", "hc_ffn_scale"):
        getattr(block, name).data.normal_(std=0.1)

    hidden = torch.randn(1, 5, 4, 512, device="cuda", dtype=torch.bfloat16) * 0.2
    main = torch.randn(1, 1, 512, device="cuda", dtype=torch.bfloat16) * 0.2
    ids = torch.tensor([[3, 127, 127, 127, 127]], device="cuda")

    attn_input, post, comb = mhc_pre(
        hidden,
        block.hc_attn_fn,
        block.hc_attn_scale,
        block.hc_attn_base,
        norm_eps=block.norm_eps,
        hc_mult=block.hc_mult,
        sinkhorn_iters=block.hc_sinkhorn_iters,
        hc_eps=block.hc_eps,
    )
    attention_output = block.attn_norm(attn_input) * 0.75 + main * 0.125
    expected = mhc_post(attention_output, hidden, post, comb)
    residual = expected
    ffn_input, post, comb = mhc_pre(
        expected,
        block.hc_ffn_fn,
        block.hc_ffn_scale,
        block.hc_ffn_base,
        norm_eps=block.norm_eps,
        hc_mult=block.hc_mult,
        sinkhorn_iters=block.hc_sinkhorn_iters,
        hc_eps=block.hc_eps,
    )
    ffn_output = block.ffn_norm(ffn_input) + ids.to(torch.bfloat16).unsqueeze(-1) * 0.01
    expected = mhc_post(ffn_output, residual, post, comb)

    actual = block.forward_draft(
        hidden, ids, main, start_pos=9, cache="cache"
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_dspark_head_matches_markov_and_confidence_reference():
    torch.manual_seed(523)
    config = _config()
    embed = VocabParallelEmbedding(128, 512, params_dtype=torch.bfloat16)
    lm_head = ParallelLMHead(128, 512, params_dtype=torch.float32)
    module = DeepseekV4DSpark(config, embed=embed, lm_head=lm_head)
    module.norm.weight.data.normal_(mean=1.0, std=0.02)
    module.hc_head_fn.data.normal_(std=0.01)
    module.hc_head_base.data.normal_(std=0.02)
    module.hc_head_scale.data.normal_(std=0.1)
    module.markov_w1.weight.data.normal_(std=0.02)
    module.markov_w2.weight.data.normal_(std=0.02)
    module.confidence_proj.weight.data.normal_(std=0.02)
    lm_head.weight.data.normal_(std=0.02)

    hidden = torch.randn(2, 5, 4, 512, device="cuda", dtype=torch.bfloat16) * 0.2
    current = torch.tensor([7, 11], device="cuda")
    collapsed = mhc_head(
        hidden,
        module.hc_head_fn,
        module.hc_head_scale,
        module.hc_head_base,
        norm_eps=module.norm_eps,
        hc_mult=module.hc_mult,
        hc_eps=module.hc_eps,
    )
    expected_logits = lm_head(module.norm(collapsed).float())
    expected_ids = current.new_empty(2, 6)
    expected_ids[:, 0] = current
    markov = []
    for position in range(5):
        markov_embed = module.markov_w1(expected_ids[:, position])
        expected_logits[:, position].add_(module.markov_w2(markov_embed.float()))
        markov.append(markov_embed)
        expected_ids[:, position + 1] = expected_logits[:, position].argmax(-1)
    expected_confidence = module.confidence_proj(
        torch.cat([collapsed, torch.stack(markov, dim=1)], dim=-1).float()
    ).squeeze(-1)

    ids, logits, confidence = module.forward_head(hidden, current)
    torch.testing.assert_close(logits, expected_logits, rtol=0, atol=0)
    torch.testing.assert_close(confidence, expected_confidence, rtol=0, atol=0)
    assert torch.equal(ids, expected_ids)
