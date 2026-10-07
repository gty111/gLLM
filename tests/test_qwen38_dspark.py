"""Qwen3.8 DSpark config parsing and checkpoint mapping, without a GPU.

The adapter's module tree needs CUDA to build (``RMSNorm`` allocates on
``cuda``), but two pieces are pure and can be pinned down here: the config
reader, and the rule table that decides how each checkpoint tensor is sharded.

The key set below is the real one -- 62 tensors read straight out of
``RadixArk/Qwen3.8-27B-DSpark``'s ``model.safetensors`` header -- so this also
catches the failure mode that matters most for a standalone drafter: a module
path that maps to a checkpoint key which does not exist.
"""

import pytest

from gllm.models.qwen38_dspark import (
    Qwen38DSparkConfig,
    Qwen38DSparkMapping,
    _spread_target_layers,
)



# The published config, trimmed to the fields the adapter reads.
PUBLISHED_CONFIG = {
    "architectures": ["DSparkDraftModel"],
    "hidden_size": 5120,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "intermediate_size": 17408,
    "num_hidden_layers": 5,
    "num_target_layers": 64,
    "rms_norm_eps": 1e-06,
    "vocab_size": 248320,
    "block_size": 7,
    "markov_rank": 256,
    "mask_token_id": 248070,
    "target_layer_ids": [5, 19, 33, 47, 61],
    "rope_theta": 10000000,
    "max_position_embeddings": 262144,
    "rope_scaling": {"rope_type": "yarn", "factor": 32.0, "beta_fast": 32.0,
                     "beta_slow": 1.0, "original_max_position_embeddings": 8192},
    "dspark_config": {
        "markov_rank": 256,
        "mask_token_id": 248070,
        "target_layer_ids": [5, 19, 33, 47, 61],
    },
}

# One decoder layer's tensors, in the order the safetensors header lists them.
LAYER_PARAMS = (
    "input_layernorm.weight",
    "mlp.down_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "post_attention_layernorm.weight",
    "self_attn.k_norm.weight",
    "self_attn.k_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.q_proj.weight",
    "self_attn.v_proj.weight",
)

TOP_LEVEL_PARAMS = (
    "fc.weight",
    "hidden_norm.weight",
    "norm.weight",
    "confidence_head.proj.weight",
    "confidence_head.proj.bias",
    "markov_head.markov_w1.weight",
    "markov_head.markov_w2.weight",
)


def _checkpoint_keys(num_layers: int = 5) -> list[str]:
    keys = list(TOP_LEVEL_PARAMS)
    for layer in range(num_layers):
        keys.extend(f"layers.{layer}.{param}" for param in LAYER_PARAMS)
    return sorted(keys)


def _match(rules, key):
    for rule in rules:
        if rule.match(key):
            return rule.name
    return None


def test_config_reads_the_published_layout():
    config = Qwen38DSparkConfig.from_dict(PUBLISHED_CONFIG)

    assert config.hidden_size == 5120
    assert config.num_attention_heads == 32
    assert config.num_key_value_heads == 8
    assert config.head_dim == 128
    assert config.num_hidden_layers == 5
    assert config.block_size == 7
    assert config.markov_rank == 256
    assert config.mask_token_id == 248070
    assert config.target_layer_ids == (5, 19, 33, 47, 61)
    assert config.rope_theta == 10000000
    assert config.rope_scaling["rope_type"] == "yarn"
    # ``fc`` folds one tap per target layer, so this is the width it eats.
    assert config.target_hidden_size == 5 * 5120 == 25600


def test_config_accepts_dspark_fields_at_top_level_too():
    flat = {k: v for k, v in PUBLISHED_CONFIG.items() if k != "dspark_config"}
    config = Qwen38DSparkConfig.from_dict(flat)

    assert config.target_layer_ids == (5, 19, 33, 47, 61)
    assert config.mask_token_id == 248070
    assert config.markov_rank == 256


def test_config_derives_taps_when_they_are_absent():
    flat = {
        k: v
        for k, v in PUBLISHED_CONFIG.items()
        if k not in ("dspark_config", "target_layer_ids")
    }
    config = Qwen38DSparkConfig.from_dict(flat)

    # The reference ``build_target_layer_ids`` formula, which is what the
    # drafter falls back to when a config omits the explicit list.  Note this
    # is *not* the published list ([5, 19, 33, 47, 61]) -- that checkpoint
    # spells its taps out, and the fallback only exists for derived configs.
    assert _spread_target_layers(64, 5) == [1, 16, 31, 46, 61]
    assert config.target_layer_ids == (1, 16, 31, 46, 61)


def test_config_rejects_a_checkpoint_without_a_noise_token():
    flat = {
        k: v
        for k, v in PUBLISHED_CONFIG.items()
        if k not in ("dspark_config", "mask_token_id")
    }
    with pytest.raises(ValueError, match="mask_token_id"):
        Qwen38DSparkConfig.from_dict(flat)


def test_checkpoint_key_is_identity_over_the_real_key_set():
    """A standalone drafter reuses its module names as checkpoint names.

    This is the whole reason the mapping exists as a separate object: the
    sibling DeepSeek-V4 adapter needs ``blocks.*`` -> ``mtp.0/1/2.*``, and this
    one needs nothing at all.
    """
    mapping = Qwen38DSparkMapping()
    keys = _checkpoint_keys()

    # No duplicates, and the count matches the published header.
    assert len(keys) == len(set(keys)) == 62
    for key in keys:
        assert mapping.checkpoint_key(key) == key


# module parameter fragment -> the rule that must claim it.
EXPECTED_RULES = [
    # Attention stays unfused: the dual-source block projects ``q`` from the
    # noisy block and ``k``/``v`` from context ++ noise, so they cannot share
    # one input tensor.
    ("layers.0.self_attn.q_proj.weight", "q_proj"),
    ("layers.0.self_attn.k_proj.weight", "k_proj"),
    ("layers.0.self_attn.v_proj.weight", "v_proj"),
    ("layers.0.self_attn.o_proj.weight", "o_proj"),
    # Only the MLP fuses, and the handler splits gate_proj / up_proj back out
    # of the checkpoint into it.
    ("layers.0.mlp.gate_up_proj.weight", "gate_up_proj"),
    ("layers.0.mlp.down_proj.weight", "down_proj"),
    # Markov tables are vocab-parallel, both of them.
    ("markov_head.markov_w1.weight", "markov"),
    ("markov_head.markov_w2.weight", "markov"),
    # Everything else is small enough to sit on every rank in full.
    ("fc.weight", "replicated"),
    ("hidden_norm.weight", "replicated"),
    ("confidence_head.proj.weight", "replicated"),
    ("confidence_head.proj.bias", "replicated"),
    ("layers.0.input_layernorm.weight", "replicated"),
    ("layers.0.post_attention_layernorm.weight", "replicated"),
    ("layers.0.self_attn.q_norm.weight", "replicated"),
    ("layers.0.self_attn.k_norm.weight", "replicated"),
    ("norm.weight", "replicated"),
]


def test_weight_rules_claim_each_parameter_family():
    rules = Qwen38DSparkMapping().weight_rules()

    for path, expected in EXPECTED_RULES:
        assert _match(rules, path) == expected, path


def test_markov_rule_wins_over_the_catch_all():
    """Ordering matters: ``markov_w*`` must not fall through to ``replicated``.

    At BF16 each table is ~127 MB, so silently replicating them is the kind of
    regression a numerics test would never notice.
    """
    rules = Qwen38DSparkMapping().weight_rules()
    markov_index = next(
        i for i, rule in enumerate(rules) if rule.name == "markov"
    )
    catch_all_index = next(
        i for i, rule in enumerate(rules) if rule.name == "replicated"
    )

    assert markov_index < catch_all_index
