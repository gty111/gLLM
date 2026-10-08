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
import torch

from gllm.models.qwen38_dspark import (
    Qwen38DSparkConfig,
    Qwen38DSparkMapping,
    _rope_cache_length,
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


def test_rope_cache_spans_the_serving_window_not_its_square():
    """A regression here asked YaRN for ~4 GiB instead of ~128 MiB.

    ``YaRNScalingRotaryEmbedding`` builds ``max_position_embeddings *
    scaling_factor`` cache rows, so the length handed to it has to be the one
    *before* scaling.  Passing the published ``max_position_embeddings``
    (262144) at factor 32 asked for 8.4M rows; the serving window only needs
    the 262144 that ``original_max_position_embeddings`` implies.
    """
    config = Qwen38DSparkConfig.from_dict(PUBLISHED_CONFIG)
    factor = float(config.rope_scaling["factor"])

    rows = _rope_cache_length(config)
    assert rows == config.rope_scaling["original_max_position_embeddings"] == 8192
    # The invariant that broke: the cache lands on the serving window, i.e.
    # pre-scaling rows times the factor, not the factor applied twice.
    assert int(rows * factor) == config.max_position_embeddings == 262144
    # ~128 MiB at head_dim 128 in fp32.  A doubled window is ~4 GiB.
    assert rows * config.head_dim * 4 <= 256 * 1024 * 1024

    # A config that omits the pre-scaling length must still land on the serving
    # window rather than past it.
    bare = Qwen38DSparkConfig.from_dict(
        {
            **PUBLISHED_CONFIG,
            "rope_scaling": {"rope_type": "yarn", "factor": 32.0},
        }
    )
    assert int(_rope_cache_length(bare) * 32.0) == bare.max_position_embeddings

    # Without scaling there is nothing to undo.
    plain = Qwen38DSparkConfig.from_dict(
        {**PUBLISHED_CONFIG, "rope_scaling": None}
    )
    assert _rope_cache_length(plain) == plain.max_position_embeddings


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


# --- GPU forward path -------------------------------------------------------


def _tensor_config():
    """A tiny stand-in with the same *structure* as the published config."""
    return Qwen38DSparkConfig(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        intermediate_size=128,
        num_hidden_layers=2,
        rms_norm_eps=1e-6,
        vocab_size=128,
        block_size=5,
        target_layer_ids=(0, 1),
        mask_token_id=127,
        markov_rank=8,
        rope_theta=10000.0,
        max_position_embeddings=64,
        rope_scaling=None,
    )


def _build_model(config):
    """Construct on CUDA under the loader's default dtype.

    ``model_loader`` calls ``torch.set_default_dtype(self.dtype)`` before
    building a model, and every layer here relies on that -- a direct
    construction without it comes out fp32 and will not matmul against bf16
    activations.
    """
    from gllm.layers.vocab_parallel_embedding import (
        ParallelLMHead,
        VocabParallelEmbedding,
    )
    from gllm.models.qwen38_dspark import Qwen38DSpark

    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        embed = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        return Qwen38DSpark(config, embed=embed, lm_head=lm_head)
    finally:
        torch.set_default_dtype(previous_dtype)


def _rows(config, context_len):
    """A ``[1, context_len, taps * hidden]`` block of target hidden states."""
    return torch.randn(
        1, context_len, config.target_hidden_size,
        device="cuda", dtype=torch.bfloat16,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_real_config_rope_cache_is_not_multiplied_twice():
    """Guard the *call site*, not just the arithmetic above.

    ``_build_rope`` has to hand YaRN the pre-scaling length.  Building the real
    cache from the published config is the only way to catch someone bypassing
    ``_rope_cache_length`` -- a regression would allocate ~4 GiB here before
    the assertion fires, which is exactly the failure this test exists to
    prevent from shipping.
    """
    from gllm.models.qwen38_dspark import _build_rope

    config = Qwen38DSparkConfig.from_dict(PUBLISHED_CONFIG)
    rope = _build_rope(config)

    rows, width = rope.cos_sin_cache.shape
    assert width == config.head_dim
    # 8192 x 32, i.e. the serving window -- not 262144 x 32.
    assert rows == config.max_position_embeddings == 262_144


def _source_for(key, shape, dtype):
    '''Deterministic synthetic checkpoint tensor for one key.'''
    generator = torch.Generator().manual_seed(abs(hash(key)) % (2**31))
    return (torch.randn(shape, generator=generator) * 0.1).to(dtype)


def _build_synthetic_checkpoint(model, config):
    '''Build the exact key set the Qwen3.8 mapping expects to read.'''
    mapping = Qwen38DSparkMapping(model)
    weights = {}
    for path, param in model.named_parameters():
        key = mapping.checkpoint_key(path)
        if key.endswith('mlp.gate_up_proj.weight'):
            half = param.shape[0] // 2
            gate_key = key.replace('gate_up_proj', 'gate_proj')
            up_key = key.replace('gate_up_proj', 'up_proj')
            weights[gate_key] = _source_for(
                gate_key, (half, param.shape[1]), param.dtype
            )
            weights[up_key] = _source_for(
                up_key, (half, param.shape[1]), param.dtype
            )
        elif key in (
            'markov_head.markov_w1.weight',
            'markov_head.markov_w2.weight',
        ):
            weights[key] = _source_for(
                key, (config.vocab_size, param.shape[1]), param.dtype
            )
        else:
            weights[key] = _source_for(key, tuple(param.shape), param.dtype)
    return weights


def _assert_loaded_checkpoint(model, weights):
    '''Every model parameter must equal its checkpoint tensor or fused slice.'''
    mapping = Qwen38DSparkMapping(model)
    for path, param in model.named_parameters():
        key = mapping.checkpoint_key(path)
        if key.endswith('mlp.gate_up_proj.weight'):
            half = param.shape[0] // 2
            gate_key = key.replace('gate_up_proj', 'gate_proj')
            up_key = key.replace('gate_up_proj', 'up_proj')
            torch.testing.assert_close(
                param[:half].float().cpu(),
                weights[gate_key].float(),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                param[half:].float().cpu(),
                weights[up_key].float(),
                rtol=0,
                atol=0,
            )
        else:
            torch.testing.assert_close(
                param.float().cpu(),
                weights[key].float(),
                rtol=0,
                atol=0,
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_load_weights_round_trips_the_synthetic_checkpoint():
    '''Exercise the complete checkpoint mapping through the real loader.

    The layer keys have no model. prefix.  Passing the usual
    pp_idx_offset=2 would make resolve_pp_layer_idx try to parse
    self_attn as the layer number and fail before any weight is copied.
    '''
    config = _tensor_config()
    model = _build_model(config)
    weights = _build_synthetic_checkpoint(model, config)

    assert set(weights) == set(_checkpoint_keys(config.num_hidden_layers))

    model._embed[0].weight.data.fill_(17)
    model._lm_head[0].weight.data.fill_(23)

    model.load_weights(weights)

    _assert_loaded_checkpoint(model, weights)
    assert torch.all(model._embed[0].weight == 17)
    assert torch.all(model._lm_head[0].weight == 23)

# --- cache statefulness: context accumulates across draft steps -------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_forward_draft_appends_its_context_to_the_cache():
    """One draft step grows the history by exactly the new context rows."""
    config = _tensor_config()
    model = _build_model(config)
    torch.manual_seed(9)

    prompt_len = 6
    prompt = _rows(config, prompt_len)
    new_context = _rows(config, 1)
    anchor = torch.tensor([5], device="cuda")

    cache = model.prefill(prompt, anchor)
    assert cache.context_length() == prompt_len

    model.forward_draft(
        new_context, anchor, start_pos=prompt_len + 1, caches=cache
    )

    assert cache.context_length() == prompt_len + 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_two_consecutive_draft_steps_reuse_the_grown_cache():
    """The second step builds on the first instead of replaying the prefix.

    Two routes reach the same state:

    * ``prefill(prompt)`` then a step for ``row_a`` and a step for ``row_b``;
    * ``prefill(prompt ++ row_a)`` then a step for ``row_b``.

    The second step's hidden states have to agree.  If the first route
    re-projected the prefix, or handed ``row_b`` a position range that
    overlapped ``row_a``, the two would diverge.
    """
    config = _tensor_config()
    model = _build_model(config)
    torch.manual_seed(11)

    prompt_len = 6
    prompt = _rows(config, prompt_len)
    row_a = _rows(config, 1)
    row_b = _rows(config, 1)
    anchor = torch.tensor([5], device="cuda")

    cache = model.prefill(prompt, anchor)
    model.forward_draft(row_a, anchor, start_pos=prompt_len + 1, caches=cache)
    assert cache.context_length() == prompt_len + 1
    stepped, _ = model.forward_draft(
        row_b, anchor, start_pos=prompt_len + 2, caches=cache
    )
    assert cache.context_length() == prompt_len + 2

    reference_cache = model.prefill(torch.cat([prompt, row_a], dim=1), anchor)
    assert reference_cache.context_length() == prompt_len + 1
    one_shot, _ = model.forward_draft(
        row_b, anchor, start_pos=prompt_len + 2, caches=reference_cache
    )

    # bf16: the two routes batch their matmuls differently, so this is a
    # closeness check rather than bit equality.  A wrong position range moves
    # the rotation far more than this tolerance.
    torch.testing.assert_close(stepped, one_shot, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_draft_noise_never_enters_the_cache():
    """The history holds context only -- the block is attended and dropped.

    SpecForge appends both halves and then ``crop(start)``s the tail back off;
    the end state has to match that.  A leak would show up as ``noise_width``
    extra rows, and the rows that do land have to be exactly what a prefill of
    the same context produces.
    """
    config = _tensor_config()
    model = _build_model(config)
    torch.manual_seed(13)

    prompt_len = 6
    prompt = _rows(config, prompt_len)
    new_row = _rows(config, 1)
    anchor = torch.tensor([5], device="cuda")

    cache = model.prefill(prompt, anchor)
    model.forward_draft(new_row, anchor, start_pos=prompt_len + 1, caches=cache)

    # The noisy block is ``noise_width`` rows wide; only context may land.
    assert model.noise_width > 1
    assert cache.context_length() == prompt_len + 1

    expected = model.prefill(torch.cat([prompt, new_row], dim=1), anchor)
    for got, want in zip(cache.keys, expected.keys):
        torch.testing.assert_close(got, want, rtol=2e-2, atol=2e-2)
    for got, want in zip(cache.values, expected.values):
        torch.testing.assert_close(got, want, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_forward_draft_and_head_run_end_to_end():
    """The real check: the module tree actually forwards.

    Shapes carry the meaning here -- ``forward_head`` must emit
    ``block_size + 1`` ids (anchor first) over ``block_size`` proposal
    positions, and that width has to come out of a ``block_size + 1``-row
    noisy block.
    """
    config = _tensor_config()
    torch.manual_seed(7)
    model = _build_model(config)

    # The checkpoint counts the verifier width; the contract counts proposals.
    assert model.block_size == config.block_size - 1
    assert model.noise_width == config.block_size

    batch, prompt_len = 1, 6
    prompt = torch.randn(
        batch, prompt_len, config.target_hidden_size,
        device="cuda", dtype=torch.bfloat16,
    )
    anchor = torch.tensor([3] * batch, device="cuda")

    cache = model.prefill(prompt, anchor)
    assert cache.context_length() == prompt_len

    current = torch.randn(
        batch, 1, config.target_hidden_size, device="cuda", dtype=torch.bfloat16
    )
    hidden, draft_ids = model.forward_draft(
        current, anchor, start_pos=prompt_len + 1, caches=cache
    )
    assert hidden.shape == (batch, model.noise_width, config.hidden_size)
    assert draft_ids.shape == (batch, model.noise_width)
    assert draft_ids[0, 0].item() == 3
    assert (draft_ids[0, 1:] == config.mask_token_id).all()

    ids, logits, confidence = model.forward_head(hidden, anchor)
    assert ids.shape == (batch, model.block_size + 1)
    assert logits.shape == (batch, model.block_size, config.vocab_size)
    assert confidence.shape == (batch, model.block_size)
    assert ids[0, 0].item() == 3
    assert not torch.isnan(hidden).any()
