"""The DeepSeek-V4 weight-rule table, checked without a real checkpoint.

Two things can silently go wrong in a rule table and neither shows up in a
numerics test: a parameter can be matched by an earlier, wrong rule (the
compressor's ``wkv`` vs. the attention ``wkv``), and a parameter path can map
to a checkpoint key that does not exist. Both are cheap to pin down here.
"""

from types import SimpleNamespace

import pytest
import torch

from gllm.models.deepseek_v4 import DeepseekV4ForCausalLM, _v4_src_key


FP8_CONFIG = {
    "quant_method": "fp8",
    "activation_scheme": "dynamic",
    "weight_block_size": [128, 128],
    "scale_fmt": "ue8m0",
}


def _config():
    return SimpleNamespace(
        hidden_size=512,
        num_attention_heads=8,
        head_dim=512,
        qk_rope_head_dim=64,
        q_lora_rank=128,
        o_lora_rank=128,
        o_groups=2,
        rms_norm_eps=1e-6,
        quantization_config=FP8_CONFIG,
        sliding_window=8,
        # one plain, one C4 (indexed), one C128 layer
        compress_ratios=[0, 4, 128],
        compress_rope_theta=40000.0,
        rope_theta=10000.0,
        original_seq_len=16,
        max_position_embeddings=64,
        model_max_length=64,
        rope_scaling={"factor": 4.0, "beta_fast": 32, "beta_slow": 1},
        index_n_heads=8,
        index_head_dim=128,
        index_topk=8,
        num_hidden_layers=3,
        vocab_size=128,
        hc_mult=2,
        hc_sinkhorn_iters=2,
        hc_eps=1e-6,
        n_routed_experts=4,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
        swiglu_limit=10.0,
        n_shared_experts=1,
        moe_intermediate_size=256,
        num_hash_layers=1,
        mlp_layer_types=None,
        mtp_enabled=False,
        dspark_block_size=0,
    )


# parameter-path fragment -> the rule that must claim it.
EXPECTED_RULES = [
    ("model.embed.weight", "vocab"),
    ("model.head.weight", "vocab"),
    ("model.hc_head_fn", "mhc"),
    ("model.layers.0.hc_attn_fn", "mhc"),
    ("model.layers.0.attn.attn_sink", "attn_sink"),
    ("model.layers.0.ffn.experts.w13_weight", "routed_experts"),
    ("model.layers.0.ffn.experts.w2_scale", "routed_experts"),
    ("model.layers.0.ffn.shared_experts.gate_up_proj.weight", "shared_w13"),
    ("model.layers.0.ffn.shared_experts.down_proj.weight_scale_inv", "shared_w2"),
    ("model.layers.0.ffn.tid2eid", "hash_routing"),
    ("model.layers.1.ffn.e_score_correction_bias", "router_bias"),
    # The compressor owns a ``wkv`` too; it must not be claimed by the
    # attention projection rule below it.
    ("model.layers.1.attn.compressor.wkv.weight", "compressor"),
    ("model.layers.1.attn.indexer.compressor.wgate.weight", "compressor"),
    ("model.layers.1.attn.compressor.ape", "compressor"),
    ("model.layers.1.attn.projections.wq_a.weight", "replicated_w"),
    ("model.layers.1.attn.projections.wkv.weight", "replicated_w"),
    ("model.layers.1.attn.projections.wq_a.weight_scale_inv", "replicated_scale"),
    ("model.layers.1.attn.projections.wq_b.weight", "column_w"),
    ("model.layers.1.attn.projections.wq_b.weight_scale_inv", "column_scale"),
    ("model.layers.1.attn.projections.wo_a.weight", "column_w"),
    ("model.layers.1.attn.projections.wo_b.weight", "row_w"),
    ("model.layers.1.attn.projections.wo_b.weight_scale_inv", "row_scale"),
    ("model.layers.1.attn.indexer.wq_b.weight", "column_w"),
    ("model.layers.1.attn.indexer.weights_proj.weight", "column_w"),
    ("model.layers.1.attn_norm.weight", "default"),
    ("model.layers.1.ffn.gate.weight", "default"),
    ("model.norm.weight", "default"),
]


def _match(rules, key):
    for rule in rules:
        if rule.match(key):
            return rule.name
    return None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_every_parameter_is_claimed_by_the_intended_rule():
    model = DeepseekV4ForCausalLM(_config())
    rules = model.weight_rules()
    names = dict(model.named_parameters())

    for path, expected in EXPECTED_RULES:
        assert path in names, f"{path} is not a parameter of this config"
        assert _match(rules, _v4_src_key(path)) == expected, path


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_no_parameter_falls_through_to_an_unintended_rule():
    """Every parameter must be claimed, and none by the bare catch-all unless
    it is a norm/router weight -- those are the only genuinely 'plain' ones."""
    model = DeepseekV4ForCausalLM(_config())
    rules = model.weight_rules()

    default_ok = ("norm.weight", "gate.weight")
    for path, _ in model.named_parameters():
        name = _match(rules, _v4_src_key(path))
        assert name is not None, path
        if name == "default":
            assert path.endswith(default_ok), (
                f"{path} fell through to the catch-all; give it an explicit rule"
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_src_key_closes_the_module_vs_checkpoint_naming_gaps():
    assert _v4_src_key("model.layers.3.attn.projections.wq_b.weight") == (
        "layers.3.attn.wq_b.weight"
    )
    assert _v4_src_key("model.layers.3.attn.projections.wq_b.weight_scale_inv") == (
        "layers.3.attn.wq_b.scale"
    )
    assert _v4_src_key("model.layers.3.attn.compressor.norm_weight") == (
        "layers.3.attn.compressor.norm.weight"
    )
    assert _v4_src_key("model.embed.weight") == "embed.weight"
