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


# --- proposal row alignment: deterministic head-only checks -----------------


def _head_stub(block_size):
    """Exercise the real head without constructing CUDA-only backbone layers."""
    from types import SimpleNamespace

    lm_head = torch.nn.Linear(2, 8, bias=False, device="cpu", dtype=torch.float32)
    markov_w1 = torch.nn.Embedding(8, 1, device="cpu", dtype=torch.float32)
    markov_w2 = torch.nn.Linear(1, 8, bias=False, device="cpu", dtype=torch.float32)
    confidence = torch.nn.Linear(3, 1, device="cpu", dtype=torch.float32)
    with torch.no_grad():
        for module in (lm_head, markov_w1, markov_w2, confidence):
            module.weight.zero_()
        confidence.bias.zero_()
    return SimpleNamespace(
        block_size=block_size,
        _lm_head=[lm_head],
        markov_head=SimpleNamespace(markov_w1=markov_w1, markov_w2=markov_w2),
        confidence_head=SimpleNamespace(proj=confidence),
    )


@pytest.mark.parametrize("block_size", [1, 5, 7])
def test_forward_head_preserves_every_noise_row(block_size):
    """Distinct row values expose a dropped first row or shifted logits."""
    from gllm.models.qwen38_dspark import Qwen38DSpark

    model = _head_stub(block_size)
    with torch.no_grad():
        model._lm_head[0].weight[:2].copy_(torch.eye(2))
        model.confidence_head.proj.weight[0, 0] = 1

    rows = torch.arange(1, block_size + 1, dtype=torch.float32)
    hidden = torch.stack((rows, rows + 10), dim=-1).unsqueeze(0)
    anchor = torch.tensor([4])
    ids, logits, confidence = Qwen38DSpark.forward_head(model, hidden, anchor)

    expected_logits = torch.zeros(1, block_size, 8)
    expected_logits[0, :, :2] = hidden[0]
    torch.testing.assert_close(logits, expected_logits, rtol=0, atol=0)
    torch.testing.assert_close(confidence, rows.unsqueeze(0), rtol=0, atol=0)
    assert ids.tolist() == [[4] + [1] * block_size]


def test_forward_head_aligns_markov_and_confidence_with_previous_tokens():
    """A hand-computed alternating chain pins each proposal's conditioning."""
    from gllm.models.qwen38_dspark import Qwen38DSpark

    model = _head_stub(7)
    with torch.no_grad():
        # Token 0 contributes [-2, 2]; token 1 contributes [3, -3].
        # With zero base logits, greedy proposals alternate 1, 0, 1, ...
        model.markov_head.markov_w1.weight[:2, 0] = torch.tensor([-2., 3.])
        model.markov_head.markov_w2.weight[:2, 0] = torch.tensor([1., -1.])
        # Raw confidence = 2 * hidden[0] + previous-token embedding + 5.
        model.confidence_head.proj.weight[0] = torch.tensor([2., 0., 1.])
        model.confidence_head.proj.bias.fill_(5)

    rows = torch.arange(1, 8, dtype=torch.float32)
    hidden = torch.stack((rows, rows * 10), dim=-1).unsqueeze(0)
    ids, logits, confidence = Qwen38DSpark.forward_head(
        model, hidden, torch.tensor([0])
    )

    expected_logits = torch.zeros(1, 7, 8)
    expected_logits[0, :, :2] = torch.tensor([
        [-2., 2.], [3., -3.], [-2., 2.], [3., -3.],
        [-2., 2.], [3., -3.], [-2., 2.],
    ])
    assert ids.tolist() == [[0, 1, 0, 1, 0, 1, 0, 1]]
    torch.testing.assert_close(logits, expected_logits, rtol=0, atol=0)
    torch.testing.assert_close(
        confidence, torch.tensor([[5., 12., 9., 16., 13., 20., 17.]]),
        rtol=0, atol=0,
    )


def test_forward_head_conditions_on_sampled_token_instead_of_argmax():
    """The next bias must follow the callback's token, even if it is not greedy."""
    from gllm.models.qwen38_dspark import Qwen38DSpark

    model = _head_stub(5)
    with torch.no_grad():
        model.markov_head.markov_w1.weight[0, 0] = -2
        model.markov_head.markov_w1.weight[2, 0] = 7
        model.markov_head.markov_w2.weight[:2, 0] = torch.tensor([1., -1.])
        model.confidence_head.proj.weight[0, 2] = 1

    ids, logits, confidence = Qwen38DSpark.forward_head(
        model, torch.zeros(1, 5, 2), torch.tensor([0]),
        sample=lambda step_logits: torch.full_like(step_logits.argmax(dim=-1), 2),
    )

    # Greedy would choose token 1 first, but the callback always chooses 2.
    expected_logits = torch.zeros(1, 5, 8)
    expected_logits[0, :, :2] = torch.tensor([
        [-2., 2.], [7., -7.], [7., -7.], [7., -7.], [7., -7.],
    ])
    assert ids.tolist() == [[0, 2, 2, 2, 2, 2]]
    torch.testing.assert_close(logits, expected_logits, rtol=0, atol=0)
    torch.testing.assert_close(
        confidence, torch.tensor([[-2., 7., 7., 7., 7.]]), rtol=0, atol=0
    )


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
        # These stand in for loaded target weights. Their constructors use
        # torch.empty, and the draft holds them outside model.parameters().
        with torch.no_grad():
            embed.weight.normal_(0, 0.02)
            lm_head.weight.normal_(0, 0.02)
        return Qwen38DSpark(config, embed=embed, lm_head=lm_head)
    finally:
        torch.set_default_dtype(previous_dtype)


def _rows(config, context_len):
    """A ``[1, context_len, taps * hidden]`` block of target hidden states."""
    return torch.randn(
        1, context_len, config.target_hidden_size,
        device="cuda", dtype=torch.bfloat16,
    )


@pytest.mark.parametrize("block_size", [1, 5, 7])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_model_proposal_width_matches_checkpoint(block_size):
    """Checkpoint block_size counts proposals; the anchor is one extra output id."""
    from dataclasses import replace

    model = _build_model(replace(_tensor_config(), block_size=block_size))
    assert model.noise_width == block_size
    assert model.block_size == block_size


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
    positions, using every row of the ``block_size``-row noisy block.
    """
    config = _tensor_config()
    torch.manual_seed(7)
    model = _build_model(config)

    # The checkpoint and contract count proposals; verification adds the anchor.
    assert model.block_size == config.block_size
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
    assert ids.shape == (batch, config.block_size + 1)
    assert logits.shape == (batch, config.block_size, config.vocab_size)
    assert confidence.shape == (batch, config.block_size)
    assert ids[0, 0].item() == 3
    assert not torch.isnan(hidden).any()


def _paged_context_storage(config, page_size):
    from gllm.runtime.cache_arena import CacheArena, select_physical_page_bytes
    from gllm.runtime.memory_manager import (
        DSparkKVCacheConfig, DSparkRequestCache, MemoryManager, Segment,
    )

    manager = MemoryManager(
        1.0, 1, torch.bfloat16, page_size, 1, config.head_dim, config.vocab_size,
        dspark_kv_cache_config=DSparkKVCacheConfig(
            config.num_hidden_layers, config.num_key_value_heads, config.head_dim
        ),
    )
    layouts = manager._cache_layouts()
    grain = select_physical_page_bytes(layouts.values())
    slot_bytes = max(
        ((layout.entry_bytes + grain - 1) // grain) * grain
        for layout in layouts.values()
    )
    arena = CacheArena(
        torch.zeros(12 * slot_bytes, device="cuda", dtype=torch.uint8),
        physical_page_bytes=grain,
    )
    target = arena.register_cache(layouts["kv_cache"])
    target_page = arena.allocator.allocate(target.name)[0]
    target.tensor("key")[target_page].fill_(11)
    target.tensor("value")[target_page].fill_(13)
    draft = arena.register_cache(layouts["dspark_kv_cache"])
    slots = arena.allocator.allocate(draft.name, 4)
    assert slots is not None
    for slot in slots:
        draft.tensor("key")[slot].fill_(31)
        draft.tensor("value")[slot].fill_(37)
    segment = Segment(
        config.num_hidden_layers, page_size, config.num_key_value_heads,
        config.head_dim, False, draft,
    )
    # Keep one allocated slot outside the request, and reverse the first pages.
    cache = DSparkRequestCache(page_table=[slots[2], slots[0], slots[3]])
    return segment, cache, arena


def _read_paged_context(segment, cache, length):
    # Test-only gather in logical token order, independent of physical grain.
    blocks = [cache.page_table[t // segment.page_size] for t in range(length)]
    offsets = [t % segment.page_size for t in range(length)]
    return (
        [key[blocks, offsets].transpose(0, 1).unsqueeze(0) for key in segment.k_cache],
        [value[blocks, offsets].transpose(0, 1).unsqueeze(0) for value in segment.v_cache],
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("page_size", [3, 4])
def test_paged_context_append_matches_dense_and_preserves_other_rows(
    monkeypatch, page_size
):
    torch.manual_seed(41)
    config = _tensor_config()
    model = _build_model(config)
    segment, cache, arena = _paged_context_storage(config, page_size)
    first, suffix = _rows(config, 5), _rows(config, 4)
    projected_lengths = []
    project = model._project_context

    def record(rows):
        projected_lengths.append(rows.shape[1])
        return project(rows)

    monkeypatch.setattr(model, "_project_context", record)
    model.append_paged_context(first, start_pos=5, segment=segment, cache=cache)
    target = arena.cache("kv_cache")
    target_page = next(iter(target.cache_type.live_slots))
    assert torch.all(target.tensor("key")[target_page] == 11)
    assert torch.all(target.tensor("value")[target_page] == 13)
    draft = arena.cache("dspark_kv_cache")
    spare = next(iter(draft.cache_type.live_slots - set(cache.page_table)))
    assert torch.all(draft.tensor("key")[spare] == 31)
    assert torch.all(draft.tensor("value")[spare] == 37)
    before_keys = [key.clone() for key in segment.k_cache]
    before_values = [value.clone() for value in segment.v_cache]
    model.append_paged_context(suffix, start_pos=9, segment=segment, cache=cache)
    assert projected_lengths == [5, 4]
    assert cache.context_length == 9
    monkeypatch.setattr(model, "_project_context", project)

    anchor = torch.tensor([5], device="cuda")
    reference = model.prefill(torch.cat([first, suffix], dim=1), anchor)
    keys, values = _read_paged_context(segment, cache, 9)
    for got, want in zip(keys + values, reference.keys + reference.values):
        torch.testing.assert_close(got, want, rtol=2e-2, atol=2e-2)
    # All locations except the second append's suffix must be bitwise unchanged,
    # including the old valid prefix, another draft slot and the target bytes.
    keep = torch.ones(segment.num_pages, page_size, device="cuda", dtype=torch.bool)
    for position in range(5, 9):
        keep[cache.page_table[position // page_size], position % page_size] = False
    for got, before in zip(segment.k_cache + segment.v_cache, before_keys + before_values):
        assert torch.equal(got[keep], before[keep])
    assert segment.k_cache[0].stride(0) > segment.k_cache[0][0].numel()
    assert arena.allocator.num_used_physical_pages > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("invalid", [
    "gap", "overlap", "capacity", "batch", "width", "dtype", "geometry", "rope", "tp",
])
def test_paged_context_rejects_invalid_input_before_writing(monkeypatch, invalid):
    import gllm.models.qwen38_dspark as adapter

    config = _tensor_config()
    model = _build_model(config)
    segment, cache, arena = _paged_context_storage(config, 4)
    model.append_paged_context(
        _rows(config, 2), start_pos=2, segment=segment, cache=cache
    )
    rows, end = _rows(config, 2), 4
    if invalid == "gap":
        end = 5
    elif invalid == "overlap":
        end = 3
    elif invalid == "capacity":
        cache.page_table = cache.page_table[:1]
        rows, end = _rows(config, 3), 5
    elif invalid == "batch":
        rows = rows.repeat(2, 1, 1)
    elif invalid == "width":
        rows = rows[..., :-1]
    elif invalid == "dtype":
        rows = rows.float()
    elif invalid == "geometry":
        segment.kv_head_dim += 1
    elif invalid == "rope":
        monkeypatch.setattr(
            model.rotary_emb, "cos_sin_cache", model.rotary_emb.cos_sin_cache[:3]
        )
    elif invalid == "tp":
        monkeypatch.setattr(adapter, "_tp_size", lambda: 2)
    before = arena.backing.clone()

    def unexpected_projection(rows):
        raise AssertionError("invalid input must fail before context projection")

    monkeypatch.setattr(model, "_project_context", unexpected_projection)
    with pytest.raises(ValueError):
        model.append_paged_context(rows, start_pos=end, segment=segment, cache=cache)
    assert cache.context_length == 2
    assert torch.equal(arena.backing, before)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_paged_context_layer_failure_keeps_valid_prefix_and_can_retry(monkeypatch):
    torch.manual_seed(43)
    config = _tensor_config()
    model = _build_model(config)
    segment, cache, _ = _paged_context_storage(config, 4)
    prefix, suffix = _rows(config, 2), _rows(config, 3)
    model.append_paged_context(prefix, start_pos=2, segment=segment, cache=cache)
    keys, values = _read_paged_context(segment, cache, 2)
    keys, values = [key.clone() for key in keys], [value.clone() for value in values]
    attention = model.layers[1].self_attn
    original = attention.prefill_context

    def fail_layer(*args):
        raise RuntimeError("injected context projection failure")

    monkeypatch.setattr(attention, "prefill_context", fail_layer)
    with pytest.raises(RuntimeError, match="injected context projection failure"):
        model.append_paged_context(suffix, start_pos=5, segment=segment, cache=cache)
    assert cache.context_length == 2
    got_keys, got_values = _read_paged_context(segment, cache, 2)
    for got, before in zip(got_keys + got_values, keys + values):
        assert torch.equal(got, before)
    monkeypatch.setattr(attention, "prefill_context", original)
    model.append_paged_context(suffix, start_pos=5, segment=segment, cache=cache)
    assert cache.context_length == 5
    reference = model.prefill(
        torch.cat([prefix, suffix], dim=1), torch.tensor([5], device="cuda")
    )
    got_keys, got_values = _read_paged_context(segment, cache, 5)
    for got, want in zip(got_keys + got_values, reference.keys + reference.values):
        torch.testing.assert_close(got, want, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_paged_context_empty_suffix_is_noop(monkeypatch):
    config = _tensor_config()
    model = _build_model(config)
    segment, cache, arena = _paged_context_storage(config, 4)
    model.append_paged_context(
        _rows(config, 2), start_pos=2, segment=segment, cache=cache
    )
    before = arena.backing.clone()

    def unexpected_projection(rows):
        raise AssertionError("empty context must not be projected")

    monkeypatch.setattr(model, "_project_context", unexpected_projection)
    model.append_paged_context(
        _rows(config, 0), start_pos=2, segment=segment, cache=cache
    )
    assert cache.context_length == 2
    assert torch.equal(arena.backing, before)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("context_length", [0, 3, 17])
@pytest.mark.parametrize("kv_heads", [8, 32])
def test_paged_dual_source_attention_matches_dense_oracle(context_length, kv_heads):
    from dataclasses import replace
    from types import SimpleNamespace

    from gllm.models.qwen38_dspark import Qwen38DSparkAttention

    torch.manual_seed(47)
    config = replace(
        _tensor_config(), num_hidden_layers=1, num_attention_heads=32,
        num_key_value_heads=kv_heads, head_dim=128, block_size=7,
    )
    segment, cache, arena = _paged_context_storage(config, 16)
    key, value = segment.k_cache[0], segment.v_cache[0]
    for page in cache.page_table:
        key[page].fill_(float("nan"))
        value[page].fill_(float("nan"))
    history_k = torch.randn(context_length, kv_heads, 128, device="cuda", dtype=torch.bfloat16)
    history_v = torch.randn_like(history_k)
    if context_length:
        blocks = [cache.page_table[t // 16] for t in range(context_length)]
        offsets = [t % 16 for t in range(context_length)]
        key[blocks, offsets] = history_k
        value[blocks, offsets] = history_v
    q = torch.randn(1, 32, 7, 128, device="cuda", dtype=torch.bfloat16)
    noise_k = torch.randn(1, kv_heads, 7, 128, device="cuda", dtype=torch.bfloat16)
    noise_v = torch.randn_like(noise_k)
    table = cache.get_device_page_table(q.device)[:, :(context_length + 15) // 16]
    attention = SimpleNamespace(
        num_heads=32, num_key_value_groups=32 // kv_heads, head_dim=128, scaling=128**-0.5
    )
    before = arena.backing.clone()
    got = Qwen38DSparkAttention._paged_attention(
        attention, q, noise_k, noise_v, key, value, table, context_length
    )
    # Concatenation is the test oracle only; poison rows beyond valid history
    # expose accidental reads of reserved capacity or a wrong logical page.
    dense_k = torch.cat(
        [history_k.transpose(0, 1).unsqueeze(0), noise_k], dim=2
    ).repeat_interleave(32 // kv_heads, dim=1)
    dense_v = torch.cat(
        [history_v.transpose(0, 1).unsqueeze(0), noise_v], dim=2
    ).repeat_interleave(32 // kv_heads, dim=1)
    scores = torch.matmul(q.float(), dense_k.float().transpose(-1, -2)) * 128**-0.5
    expected = torch.matmul(scores.softmax(-1), dense_v.float())
    torch.testing.assert_close(got.float(), expected, rtol=2e-2, atol=2e-2)
    assert torch.isfinite(got).all()
    assert torch.equal(arena.backing, before)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("poison_vocab_storage", [False, True])
def test_paged_draft_matches_dense_over_multiple_rounds_without_caching_noise(
    monkeypatch, poison_vocab_storage
):
    from dataclasses import replace

    if poison_vocab_storage:
        from gllm.layers.vocab_parallel_embedding import VocabParallelEmbedding

        create_weights = VocabParallelEmbedding.create_weights

        def dirty_vocab_weights(module, *args, **kwargs):
            create_weights(module, *args, **kwargs)
            # Model a reused torch.empty allocation containing NaNs. The
            # synthetic fixture must initialize its borrowed target weights.
            with torch.no_grad():
                module.weight.fill_(float("nan"))

        monkeypatch.setattr(VocabParallelEmbedding, "create_weights", dirty_vocab_weights)

    torch.manual_seed(53)
    config = replace(_tensor_config(), block_size=7)
    model = _build_model(config)
    assert torch.isfinite(model._embed[0].weight).all()
    assert torch.isfinite(model._lm_head[0].weight).all()
    # Centered, bounded synthetic weights give a meaningful numerical comparison
    # rather than nearly identical large positive attention scores.
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.ndim >= 2:
                parameter.normal_(0, 0.02)
    segment, state, arena = _paged_context_storage(config, 4)
    # The caller owns this reserved slot, but exposes it to the request only
    # when the growing context needs a third page.
    next_page = state.page_table.pop()
    prompt = _rows(config, 5)
    anchor = torch.tensor([5], device="cuda")
    dense = model.prefill(prompt, anchor)
    paged = model.prefill_paged(prompt, segment=segment, cache=state)
    assert paged.state is state and paged.segment is segment
    tensor = torch.tensor

    def reject_pageable_upload(data, *args, **kwargs):
        device = kwargs.get("device")
        if device is not None and torch.device(device).type == "cuda" and isinstance(data, list):
            raise AssertionError("draft must reuse the request GPU page table")
        return tensor(data, *args, **kwargs)

    from gllm.layers.ops import triton_decode_attention
    decode = triton_decode_attention.decode_attention_fwd

    def check_shared_table(*args, **kwargs):
        block_table = args[5]
        assert block_table.data_ptr() == state._device_page_table.data_ptr()
        assert block_table.stride(0) == 0
        return decode(*args, **kwargs)

    monkeypatch.setattr(torch, "tensor", reject_pageable_upload)
    monkeypatch.setattr(triton_decode_attention, "decode_attention_fwd", check_shared_table)
    position = 5
    for added in (1, 2, 1):
        context = _rows(config, added)
        position += added
        if position > len(state.page_table) * segment.page_size:
            state.page_table.append(next_page)
        expected, dense_ids = model.forward_draft(
            context, anchor, start_pos=position, caches=dense
        )
        got, paged_ids = model.forward_draft(
            context, anchor, start_pos=position, caches=paged
        )
        assert torch.isfinite(expected).all(), "dense reference produced nonfinite hidden"
        assert torch.isfinite(got).all(), "paged draft produced nonfinite hidden"
        torch.testing.assert_close(got, expected, rtol=2e-2, atol=2e-2)
        assert torch.equal(paged_ids, dense_ids)
        assert got.shape == (1, 7, config.hidden_size)
        assert paged.context_length() == dense.context_length() == position
        keys, values = _read_paged_context(segment, state, position)
        for actual, reference in zip(keys + values, dense.keys + dense.values):
            torch.testing.assert_close(actual, reference, rtol=2e-2, atol=2e-2)
    # Recomputing noise with no new accepted context must not change any arena
    # byte, including reserved suffix rows, target storage and other draft slots.
    before = arena.backing.clone()
    repeated, repeated_ids = model.forward_draft(
        _rows(config, 0), anchor, start_pos=position, caches=paged
    )
    torch.testing.assert_close(repeated, got, rtol=0, atol=0)
    assert torch.equal(repeated_ids, paged_ids)
    assert torch.equal(arena.backing, before)
    assert state.context_length == position


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("invalid", ["anchor_batch", "noise_range"])
def test_paged_draft_rejects_invalid_noise_before_context_append(monkeypatch, invalid):
    config = _tensor_config()
    model = _build_model(config)
    segment, state, arena = _paged_context_storage(config, 4)
    paged = model.prefill_paged(_rows(config, 5), segment=segment, cache=state)
    anchor = torch.tensor([5], device="cuda")
    if invalid == "anchor_batch":
        anchor = anchor.repeat(2)
    else:
        monkeypatch.setattr(
            model.rotary_emb, "cos_sin_cache", model.rotary_emb.cos_sin_cache[:10]
        )
    before = arena.backing.clone()
    with pytest.raises(ValueError):
        model.forward_draft(
            _rows(config, 1), anchor, start_pos=6, caches=paged
        )
    assert state.context_length == 5
    assert torch.equal(arena.backing, before)
