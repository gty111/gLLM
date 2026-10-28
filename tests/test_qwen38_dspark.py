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



def _target_tap_fixture():
    """CPU control-flow oracle with split residuals and destructive final norm."""
    from types import SimpleNamespace

    from gllm.models.qwen3_5 import (
        Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration, Qwen3_5Model,
    )

    class SplitLayer(torch.nn.Module):
        def __init__(self, index):
            super().__init__()
            self.index = index
            self.calls = 0

        def forward(self, data, hidden, residual):
            self.calls += 1
            if residual is None:
                residual = hidden.clone()
            else:
                residual.add_(hidden)
            return torch.full_like(hidden, (self.index + 1) / 32), residual

    class DestructiveNorm(torch.nn.Module):
        def forward(self, hidden, residual):
            residual.add_(hidden)
            hidden.copy_(residual * 3)
            return hidden, residual

    model = Qwen3_5Model.__new__(Qwen3_5Model)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(hidden_size=4)
    model.start_layer, model.end_layer = 0, 64
    model.embed_tokens = torch.nn.Embedding(32, 4)
    with torch.no_grad():
        model.embed_tokens.weight.copy_(torch.arange(128).view(32, 4))
    model.layers = torch.nn.ModuleList(SplitLayer(index) for index in range(64))
    model.norm = DestructiveNorm()
    model._fuse_tail = False

    text_model = Qwen3_5ForCausalLM.__new__(Qwen3_5ForCausalLM)
    torch.nn.Module.__init__(text_model)
    text_model.model = model
    wrapper = Qwen3_5ForConditionalGeneration.__new__(Qwen3_5ForConditionalGeneration)
    torch.nn.Module.__init__(wrapper)
    wrapper.language_model = text_model
    tokens = torch.tensor([2, 5, 7])
    data = SimpleNamespace(
        seqs=[SimpleNamespace(mm_contents=None)], get_tokens=lambda: tokens,
    )
    return wrapper, model, data


@pytest.mark.parametrize("layer_ids", [(5, 19, 33, 47, 61), (61, 5, 47)])
@torch.no_grad()
def test_target_taps_preserve_block_indices_order_and_owned_storage(layer_ids):
    wrapper, model, data = _target_tap_fixture()
    embedded = model.embed_tokens(data.get_tokens()).clone()
    expected = torch.cat([
        embedded + (layer_id + 1) * (layer_id + 2) / 64
        for layer_id in layer_ids
    ], dim=-1).unsqueeze(0)

    normal = model(data)
    final_hidden, taps = wrapper.forward_with_hidden_taps(
        data, target_layer_ids=layer_ids
    )
    torch.testing.assert_close(final_hidden, normal, rtol=0, atol=0)
    torch.testing.assert_close(final_hidden, (embedded + 65) * 3, rtol=0, atol=0)
    torch.testing.assert_close(taps, expected, rtol=0, atol=0)
    assert taps.shape == (1, 3, len(layer_ids) * 4)

    saved = taps.clone()
    next_hidden, next_taps = wrapper.forward_with_hidden_taps(
        data, embedded + 1, target_layer_ids=layer_ids
    )
    assert next_taps.data_ptr() != taps.data_ptr()
    torch.testing.assert_close(next_taps, expected + 1, rtol=0, atol=0)
    torch.testing.assert_close(next_hidden, final_hidden + 3, rtol=0, atol=0)
    torch.testing.assert_close(taps, saved, rtol=0, atol=0)


@torch.no_grad()
def test_target_normal_forward_does_not_allocate_tap_storage(monkeypatch):
    _, model, data = _target_tap_fixture()

    def fail_allocation(*args, **kwargs):
        raise AssertionError("ordinary target forward must not allocate taps")

    monkeypatch.setattr(torch.Tensor, "new_empty", fail_allocation)
    result = model(data)
    expected = (model.embed_tokens(data.get_tokens()) + 65) * 3
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


@pytest.mark.parametrize("layer_ids", [(), (5, 5), (-1,), (64,), (5.5,)])
def test_target_taps_reject_invalid_layers_before_forward(layer_ids):
    wrapper, model, data = _target_tap_fixture()
    with pytest.raises(ValueError, match="target_layer_ids"):
        wrapper.forward_with_hidden_taps(data, target_layer_ids=layer_ids)
    assert all(layer.calls == 0 for layer in model.layers)


@pytest.mark.parametrize("dimension", ["tp", "pp", "dp"])
def test_target_taps_reject_unsupported_parallelism_before_forward(monkeypatch, dimension):
    from gllm.models import qwen3_5

    wrapper, model, data = _target_tap_fixture()
    monkeypatch.setattr(qwen3_5, f"get_{dimension}_size", lambda: 2)
    with pytest.raises(ValueError, match="TP=PP=DP=1"):
        wrapper.forward_with_hidden_taps(data, target_layer_ids=(5,))
    assert all(layer.calls == 0 for layer in model.layers)


@pytest.mark.parametrize("invalid", ["batch", "multimodal", "empty", "hidden", "residual"])
def test_target_taps_reject_invalid_input_before_forward(invalid):
    wrapper, model, data = _target_tap_fixture()
    kwargs = {}
    if invalid == "batch":
        data.seqs *= 2
    elif invalid == "multimodal":
        data.seqs[0].mm_contents = object()
    elif invalid == "empty":
        data.get_tokens = lambda: torch.empty(0, dtype=torch.long)
    else:
        kwargs["hidden_states" if invalid == "hidden" else "residual"] = torch.zeros(4, 4)
    with pytest.raises(ValueError):
        wrapper.forward_with_hidden_taps(data, target_layer_ids=(5,), **kwargs)
    assert all(layer.calls == 0 for layer in model.layers)


def _hybrid_target_fixture(*, mtp_k=0):
    """Real tiny hybrid target and independent request-owned arena states."""
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5ForCausalLM as ReferenceCausalLM,
    )

    from gllm.models.qwen3_5 import Qwen3_5ForCausalLM
    from gllm.runtime.cache_arena import CacheArena, select_physical_page_bytes
    from gllm.runtime.input_data import InputData
    from gllm.runtime.memory_manager import MemoryManager, Segment, SSMSegment
    from gllm.runtime.sequence import GenerationSequence

    torch.manual_seed(71)
    config = Qwen3_5TextConfig(
        vocab_size=128, hidden_size=128, intermediate_size=256,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
        # Quarter-width RoPE is 16-wide, satisfying the FlashInfer alignment.
        head_dim=64, max_position_embeddings=64,
        linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=32, linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        layer_types=["linear_attention", "full_attention", "linear_attention"],
    )
    config._attn_implementation = "eager"
    config.attn_output_gate = True
    config.mtp_enabled = False
    config.mamba_ssm_cache_dtype = "float32"
    reference = ReferenceCausalLM(config).to(device="cuda", dtype=torch.bfloat16).eval()
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        model = Qwen3_5ForCausalLM(config).eval()
        model.load_weights(reference.state_dict())
    finally:
        torch.set_default_dtype(previous_dtype)

    class DensePrefillBackend:
        # Test-only causal attention over the tiny request's valid history.
        # Actual projections, RoPE, arena writes, GDN and residual norms run.
        def forward(self, q, key, value, metadata, scale):
            pages, length = metadata
            k = key[pages].reshape(-1, key.shape[-2], key.shape[-1])[:length]
            v = value[pages].reshape(-1, value.shape[-2], value.shape[-1])[:length]
            # A continuation chunk starts after the existing KV prefix.
            mask = None
            if q.shape[0] != length:
                query_positions = torch.arange(
                    length - q.shape[0], length, device=q.device
                )
                mask = torch.arange(length, device=q.device)[None, :] <= query_positions[:, None]
            output = torch.nn.functional.scaled_dot_product_attention(
                q.transpose(0, 1).unsqueeze(0),
                k.transpose(0, 1).unsqueeze(0),
                v.transpose(0, 1).unsqueeze(0),
                attn_mask=mask, is_causal=mask is None, scale=scale, enable_gqa=True,
            )
            return output[0].transpose(0, 1).contiguous()

    def make_input(tokens):
        manager = MemoryManager(
            1.0, model.num_kv_layers, torch.bfloat16, 16,
            model.num_kv_heads, model.head_dim, config.vocab_size,
            ssm_cache_config=model.ssm_cache_config, mtp_k=mtp_k,
        )
        # Match init_arena's unquantized KV writer metadata.
        manager.kv_cache_dtype = "auto"
        manager.k_scale = torch.tensor(1.0, dtype=torch.float32, device="cuda")
        manager.v_scale = manager.k_scale
        layouts = manager._cache_layouts()
        grain = select_physical_page_bytes(layouts.values())
        pages = 16 * sum((layout.entry_bytes + grain - 1) // grain for layout in layouts.values())
        arena = CacheArena(
            torch.zeros(pages * grain, device="cuda", dtype=torch.uint8),
            physical_page_bytes=grain,
        )
        manager.cache_arena = arena
        manager.segment = Segment(
            model.num_kv_layers, 16, model.num_kv_heads, model.head_dim, False,
            arena.register_cache(layouts["kv_cache"]),
        )
        manager.ssm_segment = SSMSegment(
            model.ssm_cache_config,
            state_cache=arena.register_cache(layouts["ssm_state"]),
            snapshot_cache=arena.register_cache(layouts["ssm_snapshot"]),
        )
        seq = GenerationSequence(1, list(tokens), [], output_len=8, temperature=0)
        seq.page_table = [
            manager.segment.allocate() for _ in range((len(tokens) + 15) // 16)
        ]
        assert manager.allocate_recurrent_slot(seq)
        seq.to_compute_token_num = len(tokens)
        data = InputData(True, manager, max_seq_length=64, max_running_seqs=1)
        data.cal_and_set_input([seq])
        data.forward_metadata_plan.attention_metadata = (seq.page_table, len(tokens))
        return data, manager, seq

    for layer in model.model.layers:
        if layer.self_attn is not None:
            layer.self_attn.attn.set_backend(DensePrefillBackend())
    return model, reference, make_input


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.no_grad()
def test_target_taps_match_hf_hybrid_residual_stream():
    model, reference, make_input = _hybrid_target_fixture()
    ordinary, ordinary_manager, ordinary_seq = make_input((1, 2, 3, 4))
    collected, collected_manager, collected_seq = make_input((1, 2, 3, 4))
    # Independent initial cache/state is essential: a prior forward advances GDN.
    normal = model(ordinary)
    final_hidden, taps = model.forward_with_hidden_taps(
        collected, target_layer_ids=(1, 0)
    )
    output = reference(
        torch.tensor([[1, 2, 3, 4]], device="cuda"),
        use_cache=False, output_hidden_states=True, return_dict=True,
    )
    expected_taps = torch.cat(
        [output.hidden_states[2], output.hidden_states[1]], dim=-1
    )
    assert torch.isfinite(taps).all() and torch.isfinite(final_hidden).all()
    # BF16 fused residual+norm and the unfused reference round differently.
    torch.testing.assert_close(taps, expected_taps, atol=2e-3, rtol=1e-2)
    # Preserve the target's existing final-norm numerics exactly. Its fused
    # BF16 path already differs from the unfused HF final output at baseline;
    # selected pre-norm block features are the independent oracle above.
    torch.testing.assert_close(final_hidden, normal, atol=0, rtol=0)
    for name in ("k_cache", "v_cache"):
        for actual, expected in zip(
            getattr(collected_manager.segment, name), getattr(ordinary_manager.segment, name)
        ):
            torch.testing.assert_close(
                actual[collected_seq.page_table], expected[ordinary_seq.page_table],
                atol=0, rtol=0,
            )
    for name in ("conv_state", "temporal_state"):
        torch.testing.assert_close(
            getattr(collected_manager.ssm_segment, name)[:, collected_seq.recurrent_state_slot],
            getattr(ordinary_manager.ssm_segment, name)[:, ordinary_seq.recurrent_state_slot],
            atol=0, rtol=0,
        )
    collected_manager.free(collected_seq)
    ordinary_manager.free(ordinary_seq)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.no_grad()
def test_target_taps_stream_chunks_into_paged_draft_and_select_accepted_prefix():
    from dataclasses import replace

    from gllm.models.qwen38_dspark import Qwen38DSpark

    model, reference, make_input = _hybrid_target_fixture()
    tokens = (1, 2, 3, 4, 5, 6)
    ordinary, ordinary_manager, ordinary_seq = make_input(tokens)
    collected, collected_manager, collected_seq = make_input(tokens)
    output = reference(
        torch.tensor([tokens], device="cuda"),
        use_cache=False, output_hidden_states=True, return_dict=True,
    )
    expected_taps = torch.cat(
        [output.hidden_states[2], output.hidden_states[1]], dim=-1
    )
    config = replace(
        _tensor_config(), hidden_size=128, intermediate_size=256,
        target_layer_ids=(1, 0), block_size=7,
    )
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        draft = Qwen38DSpark(
            config, embed=model.model.embed_tokens, lm_head=model.lm_head
        ).eval()
        for parameter in draft.parameters():
            if parameter.ndim >= 2:
                parameter.normal_(0, 0.02)
    finally:
        torch.set_default_dtype(previous_dtype)
    assert draft._embed[0] is model.model.embed_tokens
    assert draft._lm_head[0] is model.lm_head

    segment, state, arena = _paged_context_storage(config, 4)
    chunks, snapshots = [], []
    anchor = torch.tensor([7], device="cuda")
    start = 0
    for end in (2, 3, 6):
        for data, seq in ((ordinary, ordinary_seq), (collected, collected_seq)):
            seq.to_compute_token_num = end - start
            data.cal_and_set_input([seq])
            data.forward_metadata_plan.attention_metadata = (seq.page_table, end)
        normal = model(ordinary)
        final_hidden, taps = model.forward_with_hidden_taps(
            collected, target_layer_ids=config.target_layer_ids
        )
        assert taps.shape == (1, end - start, config.target_hidden_size)
        torch.testing.assert_close(
            taps, expected_taps[:, start:end], atol=2e-3, rtol=1e-2
        )
        torch.testing.assert_close(final_hidden, normal, atol=0, rtol=0)
        for old, saved in zip(chunks, snapshots):
            torch.testing.assert_close(old, saved, atol=0, rtol=0)
        chunks.append(taps)
        snapshots.append(taps.clone())
        # Consume only this chunk; no full-prompt raw feature buffer is needed.
        paged = draft.prefill_paged(taps, segment=segment, cache=state)
        dense = draft.prefill(torch.cat(chunks, dim=1), anchor)
        assert paged.context_length() == dense.context_length() == end
        keys, values = _read_paged_context(segment, state, end)
        for got, want in zip(keys + values, dense.keys + dense.values):
            torch.testing.assert_close(got, want, rtol=2e-2, atol=2e-2)
        ordinary_seq.computed_token_num = collected_seq.computed_token_num = end
        start = end

    # The dense path's norm requires a nonempty suffix. Build the same
    # six-row history as five cached rows plus the final collected row.
    features = torch.cat(chunks, dim=1)
    dense = draft.prefill(features[:, :5], anchor)
    before = arena.backing.clone()
    expected, dense_ids = draft.forward_draft(
        features[:, 5:], anchor, start_pos=6, caches=dense
    )
    got, paged_ids = draft.forward_draft(
        features[:, :0], anchor, start_pos=6, caches=paged
    )
    assert torch.isfinite(got).all() and torch.isfinite(expected).all()
    torch.testing.assert_close(got, expected, rtol=2e-2, atol=2e-2)
    assert torch.equal(paged_ids, dense_ids)
    assert torch.equal(arena.backing, before)

    # Simulate a caller selecting an accepted prefix from already-computed
    # target rows. This does not implement verifier decisions or GDN rollback.
    accepted = features[:, :3].clone()
    prefix_segment, prefix_state, _ = _paged_context_storage(config, 4)
    prefix_keys = [key.clone() for key in prefix_segment.k_cache]
    prefix_values = [value.clone() for value in prefix_segment.v_cache]
    draft.prefill_paged(accepted, segment=prefix_segment, cache=prefix_state)
    dense_prefix = draft.prefill(accepted, anchor)
    assert prefix_state.context_length == 3
    keys, values = _read_paged_context(prefix_segment, prefix_state, 3)
    for got, want in zip(keys + values, dense_prefix.keys + dense_prefix.values):
        torch.testing.assert_close(got, want, rtol=2e-2, atol=2e-2)
    keep = torch.ones(
        prefix_segment.num_pages, prefix_segment.page_size,
        device="cuda", dtype=torch.bool,
    )
    keep[prefix_state.page_table[0], :3] = False
    for got, before in zip(
        prefix_segment.k_cache + prefix_segment.v_cache, prefix_keys + prefix_values
    ):
        assert torch.equal(got[keep], before[keep])
    collected_manager.free(collected_seq)
    ordinary_manager.free(ordinary_seq)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.no_grad()
def test_eight_row_target_verify_matches_stepwise_decode_with_fp32_state(monkeypatch):
    from gllm.layers.attention.qkv_backends import (
        bind_qkv_attention_backend,
        create_qkv_attention_backend,
        find_qkv_attention_layers,
    )

    model, _, make_input = _hybrid_target_fixture(mtp_k=7)
    assert model.mtp is None
    backend = create_qkv_attention_backend("fa3", 64, 1)
    assert bind_qkv_attention_backend(find_qkv_attention_layers(model), backend) == 1
    calls = []
    forward = backend.forward

    def record(q, key, value, metadata, scale):
        calls.append((q.shape[0], metadata.max_query_len))
        return forward(q, key, value, metadata, scale)

    monkeypatch.setattr(backend, "forward", record)
    prompt = tuple(range(1, 14))
    verify, verify_manager, verify_seq = make_input(prompt)
    stepwise, step_manager, step_seq = make_input(prompt)
    checked, checked_manager, checked_seq = make_input(prompt)
    requests = (
        (verify, verify_manager, verify_seq),
        (stepwise, step_manager, step_seq),
        (checked, checked_manager, checked_seq),
    )
    spare_pages = []
    for data, manager, seq in requests:
        # Reserve two logical pages in reverse order, with a live spare page
        # between them. The eight-row verify crosses the 16-token page edge.
        spare_pages.append(manager.segment.allocate())
        last_page = manager.segment.allocate()
        seq.page_table = [last_page, seq.page_table[0]]
        table = seq.ssm_block_table
        assert len(table) == 8 and len(set(table)) == 8
        seq.ssm_block_table = [table[i] for i in (3, 0, 6, 2, 7, 1, 5, 4)]
        seq.recurrent_state_slot = seq.ssm_block_table[0]
        assert model.model.embed_tokens.weight.dtype == torch.bfloat16
        assert all(
            cache.dtype == torch.bfloat16
            for cache in manager.segment.k_cache + manager.segment.v_cache
        )
        assert manager.ssm_segment.conv_state.dtype == torch.bfloat16
        assert manager.ssm_segment.temporal_state.dtype == torch.float32
        assert not manager.ssm_segment.temporal_state.is_contiguous()
        data.cal_and_set_input([seq])
        data.forward_metadata_plan.prepare_attention(backend, data)
        model(data)
        seq.computed_token_num = len(prompt)

    candidates = tuple(range(14, 22))
    verify_seq.to_compute_tokens = list(candidates)
    verify_seq.to_compute_token_num = len(candidates)
    verify_seq._mtp_verify = True
    verify.cal_and_set_input([verify_seq])
    plan = verify.forward_metadata_plan
    assert plan.query_lens == (8,) and plan.num_mtp_verify_rows == 1
    assert plan.num_decodes == 0 and verify.is_mtp_verify
    assert verify.get_ssm_block_table_2d().cpu().tolist() == [verify_seq.ssm_block_table]
    assert verify.get_ssm_num_accepted().cpu().tolist() == [1]
    plan.prepare_attention(backend, verify)
    hidden, taps = model.forward_with_hidden_taps(verify, target_layer_ids=(1, 0))
    logits = model.logits_from_hidden(hidden)
    assert hidden.shape == (8, 128) and taps.shape == (1, 8, 256)
    assert logits.shape == (8, 128)
    assert torch.isfinite(logits).all() and torch.isfinite(taps).all()

    for index, token in enumerate(candidates):
        step_seq.to_compute_tokens = [token]
        step_seq.to_compute_token_num = 1
        stepwise.cal_and_set_input([step_seq])
        step_plan = stepwise.forward_metadata_plan
        assert step_plan.query_lens == (1,) and step_plan.num_decodes == 1
        assert step_plan.num_mtp_verify_rows == 0 and not stepwise.is_mtp_verify
        step_plan.prepare_attention(backend, stepwise)
        step_hidden, step_taps = model.forward_with_hidden_taps(
            stepwise, target_layer_ids=(1, 0)
        )
        step_logits = model.logits_from_hidden(step_hidden)
        torch.testing.assert_close(
            taps[:, index:index + 1], step_taps, atol=2e-3, rtol=1e-2,
            msg=f"target taps at verify row {index}",
        )
        torch.testing.assert_close(
            hidden[index:index + 1], step_hidden, atol=2e-2, rtol=2e-2,
            msg=f"target final hidden at verify row {index}",
        )
        torch.testing.assert_close(
            logits[index:index + 1], step_logits, atol=2e-3, rtol=1e-2,
            msg=f"target logits at verify row {index}",
        )
        assert torch.equal(logits[index].argmax(), step_logits[0].argmax())
        checkpoint = verify_seq.ssm_block_table[index]
        rolling = step_seq.recurrent_state_slot
        torch.testing.assert_close(
            verify_manager.ssm_segment.conv_state[:, checkpoint],
            step_manager.ssm_segment.conv_state[:, rolling],
            atol=2e-3, rtol=2e-3, msg=f"conv checkpoint {index}",
        )
        torch.testing.assert_close(
            verify_manager.ssm_segment.temporal_state[:, checkpoint],
            step_manager.ssm_segment.temporal_state[:, rolling],
            atol=2e-5, rtol=2e-5, msg=f"temporal checkpoint {index}",
        )
        step_seq.computed_token_num += 1

    # This is an eight-row target forward, followed by eight actual one-token
    # decode forwards; neither path invokes an MTP head or a dense mock backend.
    from gllm.speculative.dspark import verify_dspark_target

    checked_seq.to_compute_tokens = list(candidates)
    checked_seq.to_compute_token_num = 8
    checked_seq._mtp_verify = True
    checked.cal_and_set_input([checked_seq])
    checked.forward_metadata_plan.prepare_attention(backend, checked)
    before_table = list(checked_seq.ssm_block_table)
    before_calls = list(calls)
    before_arena = checked_manager.cache_arena.backing.clone()
    with torch.autocast("cuda", dtype=torch.float16):
        with pytest.raises(ValueError, match="autocast"):
            verify_dspark_target(model, checked, block_size=7, target_layer_ids=(1, 0))
    assert calls == before_calls
    assert torch.equal(checked_manager.cache_arena.backing, before_arena)
    result = verify_dspark_target(
        model, checked, block_size=7, target_layer_ids=(1, 0)
    )
    predictions = logits.argmax(dim=-1)
    count = int((predictions[:-1] == verify.get_tokens()[1:]).int().cumprod(0).sum())
    assert result.accepted_drafts == count
    assert not result.consumed
    assert result.start_pos == len(prompt) and result.seq_id == checked_seq.seq_id
    torch.testing.assert_close(
        result.committed_ids, verify.get_tokens()[:count + 1].unsqueeze(0), atol=0, rtol=0
    )
    torch.testing.assert_close(
        result.accepted_hidden, taps[:, :count + 1], atol=0, rtol=0
    )
    assert torch.equal(result.next_anchor, predictions[count:count + 1])
    assert checked_seq.computed_token_num == len(prompt)
    assert checked_seq.ssm_block_table == before_table
    assert calls == [(13, 13)] * 3 + [(8, 8)] + [(1, 1)] * 8 + [(8, 8)]
    length = len(prompt) + len(candidates)
    keys, values = _read_paged_context(verify_manager.segment, verify_seq, length)
    expected_keys, expected_values = _read_paged_context(step_manager.segment, step_seq, length)
    for got, want in zip(keys + values, expected_keys + expected_values):
        torch.testing.assert_close(
            got, want, atol=2e-3, rtol=2e-3, msg="target paged KV",
        )
    for (_, manager, seq), spare in zip(requests, spare_pages):
        manager.segment.free(spare)
        manager.free(seq)



def _verify_contract_fixture(accepted=2):
    """CPU oracle: model spies exercise verification control, not target numerics."""
    from types import SimpleNamespace

    from gllm.layers.attention.qkv_backends import PagedAttentionMetadata
    from gllm.runtime.forward_metadata import ForwardMetadataPlan, MetadataMaterialization
    from gllm.runtime.sequence import GenerationSequence

    seq = GenerationSequence(41, list(range(13)), [], output_len=32, temperature=0)
    seq.computed_token_num, seq.to_compute_token_num = 13, 8
    seq.to_compute_tokens = list(range(14, 22))
    seq._mtp_verify, seq._cache_epoch = True, 3
    seq.page_table = [2, 0]
    seq.ssm_block_table = [4, 1, 7, 3, 8, 2, 6, 5]
    seq.recurrent_state_slot = 4
    buffers = {
        "tokens": torch.tensor(seq.to_compute_tokens),
        "position": torch.arange(13, 21),
        "seq_lens": torch.tensor([21], dtype=torch.int32),
        "query_start_loc": torch.tensor([0, 8], dtype=torch.int32),
        "slot_mapping": torch.tensor([45, 46, 47, 0, 1, 2, 3, 4]),
        "block_table": torch.tensor([[2, 0]], dtype=torch.int32),
        "ssm_block_table_2d": torch.tensor([seq.ssm_block_table], dtype=torch.int32),
        "ssm_num_accepted": torch.tensor([1], dtype=torch.int32),
        "recurrent_state_slot_per_seq": torch.tensor([4], dtype=torch.int32),
        "ssm_snapshot_write_slot_per_seq": torch.tensor([-1], dtype=torch.int32),
    }
    kv = SimpleNamespace(
        page_size=16, num_pages=3,
        k_cache=[torch.full((3, 16, 2, 4), 0.25, dtype=torch.bfloat16)],
        v_cache=[torch.full((3, 16, 2, 4), -0.5, dtype=torch.bfloat16)],
    )
    state = SimpleNamespace(
        num_blocks=9,
        conv_state=torch.ones(2, 9, 4, 3, dtype=torch.bfloat16),
        temporal_state=torch.zeros(2, 9, 2, 4, 4, dtype=torch.float32),
    )
    data = SimpleNamespace(
        seqs=[seq], tokens_cpu=buffers["tokens"].clone(),
        memory_manager=SimpleNamespace(segment=kv, ssm_segment=state, dspark_segment=None),
        is_mtp_verify=True, num_mtp_verify_rows=1, num_decodes=0,
    )
    for name in buffers:
        setattr(data, "get_" + name, lambda name=name: buffers[name])
    metadata = PagedAttentionMetadata(
        buffers["block_table"], buffers["seq_lens"], buffers["query_start_loc"], 8, 1
    )
    data.forward_metadata_plan = ForwardMetadataPlan(
        MetadataMaterialization.CPU, (8,), 0, num_mtp_verify_rows=1,
        attention_metadata=metadata,
    )

    class Target(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(vocab_size=32)
            self.max_model_len = 64
            self.model = SimpleNamespace(
                start_layer=0, end_layer=64,
                embed_tokens=torch.nn.Embedding(32, 4, dtype=torch.bfloat16),
            )
            self.calls = self.head_calls = 0
            # Tag rows so the controller head also supports one-row calls.
            self.hidden = torch.arange(8, dtype=torch.bfloat16)[:, None].expand(-1, 4).clone()
            self.taps = torch.arange(64, dtype=torch.bfloat16).reshape(1, 8, 8)
            self.logits = torch.zeros(8, 32)
            predictions = buffers["tokens"][1:].tolist() + [27]
            if accepted < 7:
                predictions[accepted] = (predictions[accepted] + 1) % 32
            self.logits[torch.arange(8), predictions] = 10

        def forward_with_hidden_taps(
            self, data, hidden_states=None, residual=None, *, target_layer_ids,
        ):
            assert target_layer_ids == (1, 0)
            assert hidden_states is None and residual is None
            self.calls += 1
            return self.hidden, self.taps

        def logits_from_hidden(self, hidden):
            self.head_calls += 1
            assert hidden.ndim == 2 and hidden.shape[1] == 4
            return self.logits.index_select(0, hidden[:, 0].long())

        def compute_logits(self, *args):
            raise AssertionError("verify needs all target rows, not only the final row")

    return Target(), data, buffers


@pytest.mark.parametrize("accepted", range(8))
@torch.no_grad()
def test_dspark_verify_selects_longest_prefix_and_owns_results(accepted):
    from gllm.speculative.dspark import verify_dspark_target

    target, data, buffers = _verify_contract_fixture(accepted)
    seq = data.seqs[0]
    expected_ids = buffers["tokens"][:accepted + 1].unsqueeze(0).clone()
    expected_taps = target.taps[:, :accepted + 1].clone()
    expected_next = target.logits.argmax(-1)[accepted:accepted + 1].clone()
    result = verify_dspark_target(target, data, block_size=7, target_layer_ids=(1, 0))
    assert target.calls == 1 and target.head_calls == 8
    assert result.accepted_drafts == accepted
    assert (result.seq_id, result.cache_epoch, result.start_pos) == (41, 3, 13)
    assert result.consumed is False
    assert seq.computed_token_num == 13 and seq.token_ids == list(range(13))
    assert seq.recurrent_state_slot == 4 and seq.ssm_block_table == [4, 1, 7, 3, 8, 2, 6, 5]
    buffers["tokens"].fill_(0)
    target.taps.fill_(-1)
    target.logits.fill_(0)
    torch.testing.assert_close(result.committed_ids, expected_ids, atol=0, rtol=0)
    torch.testing.assert_close(result.accepted_hidden, expected_taps, atol=0, rtol=0)
    torch.testing.assert_close(result.next_anchor, expected_next, atol=0, rtol=0)
    assert result.accepted_hidden.untyped_storage().nbytes() == expected_taps.numel() * 2


def test_dspark_verify_uses_ordinary_head_shape_for_batch_sensitive_logits(monkeypatch):
    from gllm.speculative.dspark import verify_dspark_target

    target, data, buffers = _verify_contract_fixture(accepted=7)
    original_head = target.logits_from_hidden

    def batch_sensitive_head(hidden):
        logits = original_head(hidden).clone()
        if hidden.shape[0] > 1:
            # Model a near-tied choice changed only by the head GEMM shape.
            logits[0].zero_()
            logits[0, int(buffers["tokens"][1]) + 1] = 10
        return logits

    monkeypatch.setattr(target, "logits_from_hidden", batch_sensitive_head)
    assert target.logits_from_hidden(target.hidden).argmax(-1)[0] != (
        target.logits_from_hidden(target.hidden[:1]).argmax(-1)[0]
    )
    target.head_calls = 0
    result = verify_dspark_target(target, data, block_size=7, target_layer_ids=(1, 0))
    assert target.calls == 1 and target.head_calls == 8
    assert result.accepted_drafts == 7
    torch.testing.assert_close(result.committed_ids, buffers["tokens"].unsqueeze(0))
    assert result.next_anchor.tolist() == [27]


@pytest.mark.parametrize("mrope", [False, True], ids=["1d", "mrope"])
@torch.no_grad()
def test_dspark_verify_accepts_conditional_wrapper_text_positions(mrope):
    from gllm.layers.rotary_embedding import MRotaryEmbedding
    from gllm.models.qwen3_5 import Qwen3_5ForConditionalGeneration
    from gllm.runtime.input_data import InputData
    from gllm.speculative.dspark import verify_dspark_target

    target, data, buffers = _verify_contract_fixture()
    data.positions_cpu = data.positions = buffers["position"]
    data.mrope_positions_cpu = None
    if mrope:
        # The conditional runner uses three identical axes even for text.
        positions = MRotaryEmbedding.get_next_input_positions(0, 13, 21)
        data.mrope_positions_cpu = data.mrope_positions = torch.tensor(positions)
    data.get_position = lambda: InputData.get_position(data)
    before = data.get_position().clone()

    wrapper = Qwen3_5ForConditionalGeneration.__new__(Qwen3_5ForConditionalGeneration)
    torch.nn.Module.__init__(wrapper)
    wrapper.language_model = target
    result = verify_dspark_target(wrapper, data, block_size=7, target_layer_ids=(1, 0))

    assert target.calls == 1 and target.head_calls == 8
    assert result.accepted_drafts == 2
    torch.testing.assert_close(result.committed_ids, buffers["tokens"][:3].unsqueeze(0))
    torch.testing.assert_close(result.accepted_hidden, target.taps[:, :3], atol=0, rtol=0)
    assert data.get_position().shape == before.shape
    assert torch.equal(data.get_position(), before)


@pytest.mark.parametrize("invalid", [
    "activation", "key", "value", "conv", "temporal_bf16", "temporal_fp16", "draft_key",
    "sampling", "penalty", "structured", "logprobs", "prompt_logprobs", "prefill",
    "verify_flag", "capacity", "duplicate_state", "rolling_slot", "resume",
    "stale_state_table", "position", "slot_mapping", "seq_lens", "snapshot",
    "vocab", "metadata_missing", "metadata_buffers", "width",
    "mrope_t", "mrope_h", "mrope_w", "position_matrix",
    "position_two_axes", "position_short_mrope",
])
def test_dspark_verify_rejects_before_forward_without_cache_changes(invalid):
    from dataclasses import replace
    from types import SimpleNamespace

    from gllm.speculative.dspark import verify_dspark_target

    target, data, buffers = _verify_contract_fixture()
    manager, seq = data.memory_manager, data.seqs[0]
    kv, state = manager.segment, manager.ssm_segment
    block_size = 7
    if invalid == "activation":
        target.model.embed_tokens.weight.data = target.model.embed_tokens.weight.float()
    elif invalid in ("key", "value"):
        getattr(kv, "k_cache" if invalid == "key" else "v_cache")[0] = kv.k_cache[0].float()
    elif invalid == "conv":
        state.conv_state = state.conv_state.float()
    elif invalid.startswith("temporal"):
        state.temporal_state = state.temporal_state.to(
            torch.bfloat16 if invalid == "temporal_bf16" else torch.float16
        )
    elif invalid == "draft_key":
        manager.dspark_segment = SimpleNamespace(
            k_cache=[torch.zeros(1)], v_cache=[torch.zeros(1, dtype=torch.bfloat16)]
        )
    elif invalid == "sampling":
        seq.temperature, seq.top_k = 1, 10
    elif invalid == "penalty":
        seq.repetition_penalty = 1.1
    elif invalid == "structured":
        seq.structured_output = object()
    elif invalid in ("logprobs", "prompt_logprobs"):
        setattr(seq, invalid + "_enabled", True)
    elif invalid == "prefill":
        seq.computed_token_num = 12
    elif invalid == "verify_flag":
        seq._mtp_verify = False
    elif invalid == "capacity":
        seq.page_table = seq.page_table[:1]
    elif invalid == "duplicate_state":
        seq.ssm_block_table[1] = seq.ssm_block_table[0]
    elif invalid == "rolling_slot":
        seq.recurrent_state_slot = 1
    elif invalid == "resume":
        seq.ssm_num_accepted = 2
    elif invalid == "stale_state_table":
        buffers["ssm_block_table_2d"] = buffers["ssm_block_table_2d"].roll(1, dims=1)
    elif invalid in ("position", "slot_mapping", "seq_lens"):
        buffers[invalid][0] += 1
    elif invalid in ("mrope_t", "mrope_h", "mrope_w"):
        buffers["position"] = buffers["position"].repeat(3, 1)
        axis = ("mrope_t", "mrope_h", "mrope_w").index(invalid)
        buffers["position"][axis, 0] += 1
    elif invalid == "position_matrix":
        buffers["position"] = buffers["position"].unsqueeze(0)
    elif invalid == "position_two_axes":
        buffers["position"] = buffers["position"].repeat(2, 1)
    elif invalid == "position_short_mrope":
        buffers["position"] = buffers["position"].repeat(3, 1)[:, :-1]
    elif invalid == "snapshot":
        buffers["ssm_snapshot_write_slot_per_seq"][0] = 1
    elif invalid == "vocab":
        buffers["tokens"][-1] = data.tokens_cpu[-1] = 32
    elif invalid == "metadata_missing":
        data.forward_metadata_plan.attention_metadata = None
    elif invalid == "metadata_buffers":
        metadata = data.forward_metadata_plan.attention_metadata
        data.forward_metadata_plan.attention_metadata = replace(
            metadata, block_table=metadata.block_table.clone()
        )
    elif invalid == "width":
        block_size = 6
    before = [t.clone() for t in kv.k_cache + kv.v_cache + [state.conv_state, state.temporal_state]]
    with pytest.raises(ValueError):
        verify_dspark_target(target, data, block_size=block_size, target_layer_ids=(1, 0))
    assert target.calls == target.head_calls == 0
    for actual, saved in zip(kv.k_cache + kv.v_cache + [state.conv_state, state.temporal_state], before):
        assert torch.equal(actual, saved)


@pytest.mark.parametrize("dimension", ["tp", "pp", "dp"])
def test_dspark_verify_rejects_parallelism_before_forward(monkeypatch, dimension):
    import gllm.speculative.dspark as verifier

    target, data, _ = _verify_contract_fixture()
    monkeypatch.setattr(verifier, "get_" + dimension + "_size", lambda: 2)
    with pytest.raises(ValueError, match="TP=PP=DP=1"):
        verifier.verify_dspark_target(target, data, block_size=7, target_layer_ids=(1, 0))
    assert target.calls == 0


@pytest.mark.parametrize("layer_ids", [(), (0, 0), (64,), (1.5,)])
def test_dspark_verify_rejects_invalid_layer_ids_before_forward(layer_ids):
    from gllm.speculative.dspark import verify_dspark_target

    target, data, _ = _verify_contract_fixture()
    with pytest.raises(ValueError, match="target_layer_ids"):
        verify_dspark_target(target, data, block_size=7, target_layer_ids=layer_ids)
    assert target.calls == 0



def test_dspark_verify_rejects_autocast_before_forward():
    from gllm.speculative.dspark import verify_dspark_target

    target, data, _ = _verify_contract_fixture()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        with pytest.raises(ValueError, match="autocast"):
            verify_dspark_target(target, data, block_size=7, target_layer_ids=(1, 0))
    assert target.calls == target.head_calls == 0


@pytest.mark.parametrize("field,value", [
    ("fast_path_rows", 0), ("fast_path_tokens", 0),
    ("fast_q_len_per_req", 1), ("context_max_query_len", 8),
])
def test_dspark_verify_rejects_stale_flashinfer_partition(field, value):
    from gllm.layers.attention.qkv_backends import FlashInferPagedAttentionMetadata
    from gllm.speculative.dspark import verify_dspark_target

    target, data, buffers = _verify_contract_fixture()
    metadata = FlashInferPagedAttentionMetadata(
        buffers["block_table"], buffers["seq_lens"], buffers["query_start_loc"], 8, 1,
        cum_seq_lens_q=buffers["query_start_loc"],
        cum_seq_lens_kv=torch.tensor([0, 21], dtype=torch.int32),
        fast_path_rows=1, fast_path_tokens=8, fast_q_len_per_req=8, context_max_query_len=0,
    )
    setattr(metadata, field, value)
    data.forward_metadata_plan.attention_metadata = metadata
    with pytest.raises(ValueError, match="FlashInfer"):
        verify_dspark_target(target, data, block_size=7, target_layer_ids=(1, 0))
    assert target.calls == target.head_calls == 0



@pytest.mark.parametrize("accepted", range(8))
def test_dspark_commit_promotes_once_before_request_progress_changes(accepted):
    from gllm.speculative.dspark import commit_dspark_verify_state, verify_dspark_target

    target, data, _ = _verify_contract_fixture(accepted)
    seq = data.seqs[0]
    result = verify_dspark_target(target, data, block_size=7, target_layer_ids=(1, 0))
    before = list(seq.ssm_block_table)
    pages, tokens = list(seq.page_table), list(seq.token_ids)
    commit_dspark_verify_state(seq, result, committed_count=accepted + 1)
    expected = list(before)
    expected[0], expected[accepted] = expected[accepted], expected[0]
    assert seq.ssm_block_table == expected
    assert seq.recurrent_state_slot == before[accepted] and seq.ssm_num_accepted == 1
    assert set(seq.ssm_block_table) == set(before) and result.consumed
    assert seq.computed_token_num == 13
    assert seq.page_table == pages and seq.token_ids == tokens
    # Keep the old start position: provenance alone cannot catch double commit.
    with pytest.raises(ValueError, match="consumed"):
        commit_dspark_verify_state(seq, result, committed_count=accepted + 1)
    assert seq.ssm_block_table == expected
    assert seq.recurrent_state_slot == expected[0] and seq.ssm_num_accepted == 1


@pytest.mark.parametrize("count", [1, 3, 7])
def test_dspark_commit_truncated_prefix_uses_actual_count(count):
    from gllm.speculative.dspark import commit_dspark_verify_state, verify_dspark_target

    target, data, _ = _verify_contract_fixture(7)
    seq = data.seqs[0]
    result = verify_dspark_target(target, data, block_size=7, target_layer_ids=(1, 0))
    selected = seq.ssm_block_table[count - 1]
    commit_dspark_verify_state(seq, result, committed_count=count)
    assert seq.recurrent_state_slot == selected
    assert result.consumed and seq.computed_token_num == 13
    with pytest.raises(ValueError, match="consumed"):
        commit_dspark_verify_state(seq, result, committed_count=8)


@pytest.mark.parametrize("invalid", [
    "zero", "negative", "past_prefix", "float", "bool",
    "request", "epoch", "position", "missing_table", "short_table",
    "duplicate_slot", "dummy_slot", "rolling_slot", "resume",
])
def test_dspark_commit_rejects_without_mutation_or_consuming(invalid):
    from gllm.speculative.dspark import commit_dspark_verify_state, verify_dspark_target

    target, data, _ = _verify_contract_fixture(2)
    seq = data.seqs[0]
    result = verify_dspark_target(target, data, block_size=7, target_layer_ids=(1, 0))
    count = 3
    if invalid == "zero":
        count = 0
    elif invalid == "negative":
        count = -1
    elif invalid == "past_prefix":
        count = 4
    elif invalid == "float":
        count = 1.5
    elif invalid == "bool":
        count = True
    elif invalid == "request":
        seq.seq_id += 1
    elif invalid == "epoch":
        seq._cache_epoch += 1
    elif invalid == "position":
        seq.computed_token_num += 1
    elif invalid == "missing_table":
        seq.ssm_block_table = None
    elif invalid == "short_table":
        seq.ssm_block_table.pop()
    elif invalid == "duplicate_slot":
        seq.ssm_block_table[1] = seq.ssm_block_table[0]
    elif invalid == "dummy_slot":
        seq.ssm_block_table[1] = 0
    elif invalid == "rolling_slot":
        seq.recurrent_state_slot = 1
    elif invalid == "resume":
        seq.ssm_num_accepted = 2
    before = (
        None if seq.ssm_block_table is None else list(seq.ssm_block_table),
        seq.recurrent_state_slot, seq.ssm_num_accepted,
        seq.computed_token_num, list(seq.page_table), list(seq.token_ids),
    )
    with pytest.raises(ValueError):
        commit_dspark_verify_state(seq, result, committed_count=count)
    after = (
        seq.ssm_block_table, seq.recurrent_state_slot, seq.ssm_num_accepted,
        seq.computed_token_num, seq.page_table, seq.token_ids,
    )
    assert after == before
    assert result.consumed is False



@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.no_grad()
def test_dspark_commits_and_paged_context_follow_greedy_over_multiple_rounds():
    from dataclasses import replace

    from gllm.layers.attention.qkv_backends import (
        bind_qkv_attention_backend, create_qkv_attention_backend, find_qkv_attention_layers,
    )
    from gllm.models.qwen38_dspark import Qwen38DSpark
    from gllm.speculative.dspark import commit_dspark_verify_state, verify_dspark_target

    model, _, make_input = _hybrid_target_fixture(mtp_k=7)
    backend = create_qkv_attention_backend("fa3", 64, 1)
    bind_qkv_attention_backend(find_qkv_attention_layers(model), backend)
    prompt = tuple(range(1, 14))
    data, manager, seq = make_input(prompt)
    oracle, oracle_manager, oracle_seq = make_input(prompt)
    prompt_taps = anchor = None
    for current, owner, request in (
        (data, manager, seq), (oracle, oracle_manager, oracle_seq)
    ):
        # Reserve 48 tokens: the greedy oracle temporarily explores all eight
        # inputs even when only a short prefix will be committed.
        extra = [owner.segment.allocate(), owner.segment.allocate()]
        request.page_table = [extra[1], request.page_table[0], extra[0]]
        table = request.ssm_block_table
        request.ssm_block_table = [table[i] for i in (3, 0, 6, 2, 7, 1, 5, 4)]
        request.recurrent_state_slot = request.ssm_block_table[0]
        request.output_len = 48
        current.cal_and_set_input([request])
        current.forward_metadata_plan.prepare_attention(backend, current)
        hidden, taps = model.forward_with_hidden_taps(current, target_layer_ids=(1, 0))
        next_token = model.logits_from_hidden(hidden[-1:]).argmax(-1)
        if current is data:
            prompt_taps, anchor = taps, next_token
        else:
            assert torch.equal(anchor, next_token)
        request.computed_token_num = len(prompt)

    config = replace(
        _tensor_config(), hidden_size=128, intermediate_size=256,
        target_layer_ids=(1, 0), block_size=7,
    )
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        draft = Qwen38DSpark(
            config, embed=model.model.embed_tokens, lm_head=model.lm_head
        ).eval()
        for parameter in draft.parameters():
            if parameter.ndim >= 2:
                parameter.normal_(0, 0.02)
    finally:
        torch.set_default_dtype(previous_dtype)
    segment, cache, _ = _paged_context_storage(config, 16)
    paged = draft.prefill_paged(prompt_taps, segment=segment, cache=cache)
    chunks, old_results = [prompt_taps], []
    rounds = (0, 0, 2, 7, None, 7)
    for round_index, desired in enumerate(rounds):
        start = seq.computed_token_num
        assert oracle_seq.computed_token_num == paged.context_length() == start
        proposed = None
        if desired is None:
            noise, _ = draft.forward_draft(
                prompt_taps[:, :0], anchor, start_pos=start, caches=paged
            )
            proposed, _, _ = draft.forward_head(noise, anchor)
            proposed = proposed[0].tolist()
            assert proposed[0] == int(anchor.item())

        # Independent ordinary-decode oracle. Save each state solely to select
        # the baseline prefix for the next round; production commit never copies.
        greedy_inputs, next_tokens, oracle_taps, states = [], [], [], []
        token = int(anchor.item())
        for _ in range(8):
            greedy_inputs.append(token)
            oracle_seq.to_compute_tokens = [token]
            oracle_seq.to_compute_token_num = 1
            oracle.cal_and_set_input([oracle_seq])
            oracle.forward_metadata_plan.prepare_attention(backend, oracle)
            hidden, taps = model.forward_with_hidden_taps(
                oracle, target_layer_ids=config.target_layer_ids
            )
            token = int(model.logits_from_hidden(hidden).argmax(-1).item())
            next_tokens.append(token)
            oracle_taps.append(taps)
            slot = oracle_seq.recurrent_state_slot
            states.append((
                oracle_manager.ssm_segment.conv_state[:, slot].clone(),
                oracle_manager.ssm_segment.temporal_state[:, slot].clone(),
            ))
            oracle_seq.computed_token_num += 1
        if proposed is None:
            proposed = list(greedy_inputs)
            if desired < 7:
                proposed[desired + 1] = (proposed[desired + 1] + 1) % config.vocab_size
            accepted = desired
        else:
            accepted = 0
            for candidate, expected in zip(proposed[1:], greedy_inputs[1:]):
                if candidate != expected:
                    break
                accepted += 1

        seq.to_compute_tokens, seq.to_compute_token_num = proposed, 8
        seq._mtp_verify = True
        data.cal_and_set_input([seq])
        data.forward_metadata_plan.prepare_attention(backend, data)
        result = verify_dspark_target(
            model, data, block_size=config.block_size, target_layer_ids=config.target_layer_ids
        )
        assert result.accepted_drafts == accepted
        assert int(result.next_anchor.item()) == next_tokens[accepted]
        count = 3 if round_index == len(rounds) - 1 else accepted + 1
        assert result.committed_ids[0, :count].tolist() == greedy_inputs[:count]
        torch.testing.assert_close(
            result.accepted_hidden[:, :count], torch.cat(oracle_taps[:count], dim=1),
            atol=2e-3, rtol=1e-2,
        )
        table_before = list(seq.ssm_block_table)
        physical_before = manager.cache_arena.backing.clone()
        commit_dspark_verify_state(seq, result, committed_count=count)
        assert seq.recurrent_state_slot == table_before[count - 1]
        assert set(seq.ssm_block_table) == set(table_before) and seq.ssm_num_accepted == 1
        assert seq.computed_token_num == start and result.consumed
        selected_table = list(seq.ssm_block_table)
        with pytest.raises(ValueError, match="consumed"):
            commit_dspark_verify_state(seq, result, committed_count=count)
        assert seq.ssm_block_table == selected_table
        assert torch.equal(manager.cache_arena.backing, physical_before)
        torch.testing.assert_close(
            manager.ssm_segment.conv_state[:, seq.recurrent_state_slot],
            states[count - 1][0], atol=2e-3, rtol=2e-3,
        )
        torch.testing.assert_close(
            manager.ssm_segment.temporal_state[:, seq.recurrent_state_slot],
            states[count - 1][1], atol=2e-5, rtol=2e-5,
        )
        seq.token_ids.extend(result.committed_ids[0, :count].tolist())
        seq.computed_token_num = start + count
        oracle_seq.token_ids.extend(greedy_inputs[:count])
        oracle_seq.computed_token_num = start + count
        oracle_slot = oracle_seq.recurrent_state_slot
        oracle_manager.ssm_segment.conv_state[:, oracle_slot].copy_(states[count - 1][0])
        oracle_manager.ssm_segment.temporal_state[:, oracle_slot].copy_(states[count - 1][1])
        assert seq.token_ids == oracle_seq.token_ids

        # Advancing progress invalidates the old verify metadata even for r=1,
        # where the CPU/GPU slot IDs themselves did not change.
        with pytest.raises(ValueError, match="metadata"):
            verify_dspark_target(
                model, data, block_size=7, target_layer_ids=config.target_layer_ids
            )
        assert torch.equal(manager.cache_arena.backing, physical_before)
        length = seq.computed_token_num
        keys, values = _read_paged_context(manager.segment, seq, length)
        ref_keys, ref_values = _read_paged_context(oracle_manager.segment, oracle_seq, length)
        for actual, expected in zip(keys + values, ref_keys + ref_values):
            torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
        pending = result.accepted_hidden[:, :count].clone()
        chunks.append(pending)
        assert cache.context_length == start
        draft.prefill_paged(pending, segment=segment, cache=cache)
        assert cache.context_length == length
        dense = draft.prefill(torch.cat(chunks, dim=1), anchor)
        keys, values = _read_paged_context(segment, cache, length)
        for actual, expected in zip(keys + values, dense.keys + dense.values):
            torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
        for previous, ids, features in old_results:
            torch.testing.assert_close(previous.committed_ids, ids, atol=0, rtol=0)
            torch.testing.assert_close(previous.accepted_hidden, features, atol=0, rtol=0)
        old_results.append((result, result.committed_ids.clone(), result.accepted_hidden.clone()))
        if count < accepted + 1:
            # The original bonus belongs to a later position. Stop here rather
            # than reusing it after the explicitly truncated commit.
            assert desired == 7
            anchor = None
        else:
            anchor = result.next_anchor
    assert anchor is None
    manager.free(seq)
    oracle_manager.free(oracle_seq)



@pytest.mark.parametrize("checkpoint_limit,serving_limit", [(20, None), (64, 20), (20, 64)])
def test_dspark_verify_rejects_window_beyond_context_limit_before_forward(
    checkpoint_limit, serving_limit
):
    from gllm.speculative.dspark import verify_dspark_target

    target, data, _ = _verify_contract_fixture()
    # Two reserved 16-token pages cover [13, 21), but either the checkpoint
    # or the resolved serving limit permits only 20 positions.
    target.max_model_len = checkpoint_limit
    target.config.model_max_length = serving_limit
    manager = data.memory_manager
    banks = manager.segment.k_cache + manager.segment.v_cache + [
        manager.ssm_segment.conv_state, manager.ssm_segment.temporal_state,
    ]
    before = [bank.clone() for bank in banks]
    with pytest.raises(ValueError, match="context window"):
        verify_dspark_target(target, data, block_size=7, target_layer_ids=(1, 0))
    assert target.calls == target.head_calls == 0
    for bank, saved in zip(banks, before):
        assert torch.equal(bank, saved)


@pytest.mark.parametrize("checkpoint_limit,serving_limit", [(21, None), (64, 21), (21, 64)])
def test_dspark_verify_allows_window_ending_at_context_limit(
    checkpoint_limit, serving_limit
):
    from gllm.speculative.dspark import verify_dspark_target

    target, data, _ = _verify_contract_fixture()
    target.max_model_len = checkpoint_limit
    target.config.model_max_length = serving_limit
    result = verify_dspark_target(target, data, block_size=7, target_layer_ids=(1, 0))
    assert target.calls == 1 and target.head_calls == 8
    assert result.accepted_drafts == 2


def _dspark_loader_target(config, *, embed=None, head=None):
    """Target shell for loader ownership checks; it does not execute a target."""
    from types import SimpleNamespace
    from gllm.models.qwen3_5 import Qwen3_5ForCausalLM

    target = Qwen3_5ForCausalLM.__new__(Qwen3_5ForCausalLM)
    torch.nn.Module.__init__(target)
    target.config = SimpleNamespace(hidden_size=config.hidden_size, vocab_size=config.vocab_size)
    target.max_model_len = config.max_position_embeddings
    target.model = SimpleNamespace(
        start_layer=0, end_layer=3,
        embed_tokens=embed if embed is not None else torch.nn.Embedding(
            config.vocab_size, config.hidden_size, dtype=torch.bfloat16
        ),
    )
    target.lm_head = head if head is not None else torch.nn.Linear(
        config.hidden_size, config.vocab_size, bias=False, dtype=torch.bfloat16
    )
    target.ssm_cache_config = SimpleNamespace(
        dtype=torch.float32, conv_state_dtype=torch.bfloat16
    )
    target.mtp = None
    return target


@pytest.mark.parametrize("field,value", [
    ("block_size", 6), ("hidden_size", 65), ("vocab_size", 129),
    ("mask_token_id", 128), ("target_layer_ids", [0, 0]),
    ("target_layer_ids", [0, 3]),
])
def test_dspark_loader_rejects_incompatible_checkpoint_before_cuda(tmp_path, field, value):
    from dataclasses import asdict, replace
    import json
    from gllm.runtime.model_loader import load_dspark_draft

    config = replace(_tensor_config(), block_size=7)
    target = _dspark_loader_target(config)
    raw = asdict(config)
    raw[field] = value
    (tmp_path / "config.json").write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="incompatible width, vocabulary or target taps"):
        load_dspark_draft(str(tmp_path), target, model_max_length=32)


@pytest.mark.parametrize("length", [8, 65])
def test_dspark_loader_rejects_context_before_allocating_draft(tmp_path, length):
    from dataclasses import asdict, replace
    import json
    from gllm.runtime.model_loader import load_dspark_draft

    config = replace(_tensor_config(), block_size=7)
    (tmp_path / "config.json").write_text(json.dumps(asdict(config)), encoding="utf-8")
    with pytest.raises(ValueError, match="context windows"):
        load_dspark_draft(str(tmp_path), _dspark_loader_target(config), length)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_dspark_loader_reads_indexed_checkpoint_and_borrows_target_weights(tmp_path):
    from dataclasses import asdict, replace
    import json
    from safetensors.torch import save_file
    from gllm.runtime.model_loader import load_dspark_draft

    config = replace(_tensor_config(), block_size=7)
    source = _build_model(config)
    weights = _build_synthetic_checkpoint(source, config)
    save_file(weights, tmp_path / "model.safetensors")
    # An index must select the intended set even beside an alternate shard.
    save_file({"fc.weight": torch.zeros_like(weights["fc.weight"])}, tmp_path / "alternate.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: "model.safetensors" for key in weights}}),
        encoding="utf-8",
    )
    (tmp_path / "config.json").write_text(json.dumps(asdict(config)), encoding="utf-8")
    target = _dspark_loader_target(
        config, embed=source._embed[0], head=source._lm_head[0]
    )
    embed_before = target.model.embed_tokens.weight.clone()
    head_before = target.lm_head.weight.clone()
    previous_dtype = torch.get_default_dtype()
    loaded = load_dspark_draft(str(tmp_path), target, model_max_length=32)

    _assert_loaded_checkpoint(loaded, weights)
    assert torch.get_default_dtype() == previous_dtype
    assert loaded._embed[0] is target.model.embed_tokens
    assert loaded._lm_head[0] is target.lm_head
    assert torch.equal(target.model.embed_tokens.weight, embed_before)
    assert torch.equal(target.lm_head.weight, head_before)
    assert all(p.dtype == torch.bfloat16 and p.device.type == "cuda" for p in loaded.parameters())
    assert not any("embed" in name or "lm_head" in name for name, _ in loaded.named_parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_dspark_loader_rejects_bf16_temporal_state_before_loading(tmp_path):
    from dataclasses import asdict, replace
    import json
    from gllm.runtime.model_loader import load_dspark_draft

    config = replace(_tensor_config(), block_size=7)
    source = _build_model(config)
    target = _dspark_loader_target(config, embed=source._embed[0], head=source._lm_head[0])
    target.ssm_cache_config.dtype = torch.bfloat16
    (tmp_path / "config.json").write_text(json.dumps(asdict(config)), encoding="utf-8")
    with pytest.raises(ValueError, match="FP32 temporal"):
        load_dspark_draft(str(tmp_path), target, model_max_length=32)


def _dspark_runtime_fixture():
    """Real MRoPE target/draft pair, before allocating their shared arena."""
    from dataclasses import replace

    from gllm.layers.attention.qkv_backends import (
        bind_qkv_attention_backend, create_qkv_attention_backend, find_qkv_attention_layers,
    )
    from gllm.models.qwen3_5 import Qwen3_5ForConditionalGeneration
    from gllm.models.qwen38_dspark import Qwen38DSpark
    from gllm.runtime.input_data import InputData
    from gllm.runtime.memory_manager import DSparkKVCacheConfig, MemoryManager
    from gllm.runtime.model_runner import ModelRunner

    text, _, make_input = _hybrid_target_fixture(mtp_k=7)
    # The tiny text fixture defaults to ordinary RoPE. A conditional wrapper
    # carrying three-axis positions needs the real MRoPE path, with sections
    # sized to this fixture's rotary dimension rather than the 27B geometry.
    from gllm.layers.rotary_embedding import MRotaryEmbedding

    for layer in text.model.layers:
        if layer.self_attn is not None:
            rope = layer.self_attn.rotary_emb
            half = rope.rotary_dim // 2
            layer.self_attn.rotary_emb = MRotaryEmbedding(
                head_size=rope.head_size, rotary_dim=rope.rotary_dim,
                max_position_embeddings=text.max_model_len, base=rope.base,
                is_neox_style=True, mrope_interleaved=True,
                mrope_section=[(half + 2) // 3, (half + 1) // 3, half // 3],
            )
    target = Qwen3_5ForConditionalGeneration.__new__(Qwen3_5ForConditionalGeneration)
    torch.nn.Module.__init__(target)
    target.language_model = text
    config = replace(
        _tensor_config(), hidden_size=128, intermediate_size=256,
        target_layer_ids=(1, 0), block_size=7,
    )
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        draft = Qwen38DSpark(
            config, embed=text.model.embed_tokens, lm_head=text.lm_head
        ).eval()
        for param in draft.parameters():
            if param.ndim >= 2:
                param.normal_(0, 0.02)
    finally:
        torch.set_default_dtype(previous_dtype)
    manager = MemoryManager(
        1.0, text.num_kv_layers, torch.bfloat16, 16, text.num_kv_heads, text.head_dim,
        text.config.vocab_size, max_running_seqs=1, ssm_cache_config=text.ssm_cache_config,
        mtp_k=7, dspark_kv_cache_config=DSparkKVCacheConfig(
            len(draft.layers), config.num_key_value_heads, config.head_dim,
        ),
    )
    manager.requires_mrope_buffer = True
    runner = ModelRunner.__new__(ModelRunner)
    runner._dspark_requests = {}
    runner._mtp_relay, runner.use_mm = {}, False
    runner.maxd, runner.maxp, runner.minp, runner.iterp = 1, 8, 1, 1
    runner.init_new_token_ratio = runner.min_new_token_ratio = 0.1
    runner.model, runner.dspark, runner.dspark_enabled = target, draft, True
    runner.model_max_length, runner.max_num_batched_tokens = 32, 32
    runner.memory_manager, runner.uses_mrope = manager, True
    runner.page_size = manager.page_size
    runner.input_data = InputData(
        use_buffer=True, memory_manager=manager, max_seq_length=32, max_running_seqs=1
    )
    backend = create_qkv_attention_backend("fa3", 32, 1)
    bind_qkv_attention_backend(find_qkv_attention_layers(target), backend)
    runner.qkv_attention_backend = backend

    return runner, make_input


def _init_dspark_runtime_test_arena(runner):
    from gllm.runtime.cache_arena import CacheArena, select_physical_page_bytes
    from gllm.runtime.memory_manager import Segment, SSMSegment

    text, draft, manager = runner.model.language_model, runner.dspark, runner.memory_manager
    config = draft.config
    # A small real arena avoids consuming the rest of the GPU in a unit test.
    layouts = manager._cache_layouts()
    grain = select_physical_page_bytes(layouts.values())
    pages = 24 * sum((layout.entry_bytes + grain - 1) // grain for layout in layouts.values())
    arena = CacheArena(
        torch.zeros(pages * grain, device="cuda", dtype=torch.uint8),
        physical_page_bytes=grain,
    )
    caches = {name: arena.register_cache(layout) for name, layout in layouts.items()}
    manager.cache_arena = arena
    manager.segment = Segment(
        text.num_kv_layers, 16, text.num_kv_heads, text.head_dim, False, caches["kv_cache"]
    )
    manager.dspark_segment = Segment(
        len(draft.layers), 16, config.num_key_value_heads, config.head_dim, False,
        caches["dspark_kv_cache"],
    )
    manager.ssm_segment = SSMSegment(
        text.ssm_cache_config, state_cache=caches["ssm_state"],
        snapshot_cache=caches["ssm_snapshot"],
    )
    manager.kv_cache_dtype = "auto"
    manager.k_scale = manager.v_scale = torch.tensor(1.0, device="cuda")
    return arena


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("chunk_size,context_limit", [(8, 9), (8, 32), (16, 32), (32, 32)])
@torch.no_grad()
def test_dspark_startup_profiles_max_chunk_and_eight_row_verify_in_shared_arena(
    chunk_size, context_limit
):
    from gllm.speculative.dspark_profile import profile_dspark_paged_verify

    runner, _ = _dspark_runtime_fixture()
    runner.max_num_batched_tokens = chunk_size
    runner.model_max_length = context_limit
    runner.model.language_model.config.model_max_length = context_limit
    if context_limit == 9:
        # The minimum valid context leaves one row for dense history in the
        # pre-arena probe and one cached row before the eight-row verify.
        runner.dspark.rotary_emb.cos_sin_cache = runner.dspark.rotary_emb.cos_sin_cache[:9]
    # Pre-arena: actual maximum prompt/tap/projection shapes and both heads.
    runner.profile_run()
    assert runner.dspark_profile["prefill_tokens"] == chunk_size
    assert runner.dspark_profile["projected_tokens"] == chunk_size
    assert runner.dspark_profile["noise_rows"] == 7
    assert runner.dspark_profile["target_logits_rows"] == 8
    assert runner.dspark_profile["paged_workspace_allowance_bytes"] > 0

    arena = _init_dspark_runtime_test_arena(runner)
    owned_before = arena.allocator.num_used_physical_pages
    runner.dspark_profile.update(profile_dspark_paged_verify(
        runner.model, runner.dspark, runner.input_data,
        max_prompt_tokens=chunk_size, model_max_length=runner.model_max_length,
        uses_mrope=runner.uses_mrope, prepare_attention=runner._prepare_attention_metadata,
    ))
    assert runner.dspark_profile["verify_tokens"] == 8
    assert runner.dspark_profile["paged_prefill_tokens"] == chunk_size
    # One case fits the verify after the maximum chunk; the other must
    # release that probe and create a shorter context without leaking slots.
    context_size = min(chunk_size, context_limit - 8)
    assert runner.dspark_profile["verify_context_tokens"] == context_size
    assert runner.input_data.forward_metadata_plan.query_lens == (8,)
    assert runner.input_data.get_position().shape == (3, 8)
    assert not runner.input_data.get_position().is_contiguous()
    assert arena.allocator.num_used_physical_pages == owned_before
    assert runner.input_data.seqs[0].ssm_block_table is None
    assert runner.input_data.seqs[0].page_table == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_dspark_loader_failure_closes_weights_and_restores_default_dtype(tmp_path, monkeypatch):
    from dataclasses import asdict, replace
    import json
    from safetensors.torch import save_file
    from gllm.runtime import model_loader as module

    config = replace(_tensor_config(), block_size=7)
    source = _build_model(config)
    target = _dspark_loader_target(config, embed=source._embed[0], head=source._lm_head[0])
    weights = _build_synthetic_checkpoint(source, config)
    del weights["fc.weight"]
    save_file(weights, tmp_path / "model.safetensors")
    (tmp_path / "config.json").write_text(json.dumps(asdict(config)), encoding="utf-8")
    closed = []
    original_close = module.LazySafetensors.close

    def close(mapping):
        original_close(mapping)
        closed.append(True)

    monkeypatch.setattr(module.LazySafetensors, "close", close)
    previous_dtype = torch.get_default_dtype()
    with pytest.raises(KeyError):
        module.load_dspark_draft(str(tmp_path), target, model_max_length=32)
    assert closed == [True]
    assert torch.get_default_dtype() == previous_dtype


@pytest.mark.parametrize("chunk_size", [1, 7])
@pytest.mark.parametrize("phase", ["prefill", "paged_verify"])
def test_dspark_startup_rejects_short_chunks_before_cuda(monkeypatch, chunk_size, phase):
    from types import SimpleNamespace

    from gllm.speculative.dspark_profile import (
        profile_dspark_paged_verify, profile_dspark_prefill,
    )

    def unexpected_work(*args, **kwargs):
        pytest.fail("A short startup chunk must be rejected before CUDA or model work")

    for name in ("reset_peak_memory_stats", "memory_allocated", "synchronize"):
        monkeypatch.setattr(torch.cuda, name, unexpected_work)
    data = SimpleNamespace(
        tokens_cpu=torch.empty(chunk_size, dtype=torch.long, device="cpu"),
        memory_manager=SimpleNamespace(reserve_dspark_pages=unexpected_work),
    )
    target = SimpleNamespace(forward_with_hidden_taps=unexpected_work)
    draft = SimpleNamespace(
        _project_context=unexpected_work, prefill=unexpected_work,
        prefill_paged=unexpected_work, forward_draft=unexpected_work,
        forward_head=unexpected_work,
    )
    with pytest.raises(ValueError, match="at least 8 tokens"):
        if phase == "prefill":
            profile_dspark_prefill(target, draft, data)
        else:
            profile_dspark_paged_verify(
                target, draft, data, max_prompt_tokens=chunk_size,
                model_max_length=32, uses_mrope=False,
                prepare_attention=unexpected_work,
            )


def test_dspark_startup_releases_outputs_before_shorter_prefill(monkeypatch):
    """CPU lifetime oracle; this does not measure CUDA memory peaks."""
    from types import SimpleNamespace
    import weakref

    from gllm.speculative.dspark_profile import profile_dspark_paged_verify

    class ShorterProbeReached(Exception):
        pass

    previous_outputs, freed = [], []

    class Target:
        calls = 0

        def forward_with_hidden_taps(self, data, *, target_layer_ids):
            self.calls += 1
            length = data.seqs[0].to_compute_token_num
            if self.calls == 2:
                assert length == 24 and freed == [32]
                assert all(reference() is None for reference in previous_outputs)
                raise ShorterProbeReached
            assert length == 32
            hidden = torch.empty(length, 4, device="cpu")
            taps = torch.empty(1, length, 8, device="cpu")
            previous_outputs.extend([weakref.ref(hidden), weakref.ref(taps)])
            return hidden, taps

    def release(seq, cache):
        freed.append(seq.computed_token_num)
        cache.reset()

    manager = SimpleNamespace(
        dspark_segment=object(),
        reserve_dspark_pages=lambda *args, **kwargs: True,
        free_dspark_cache=release,
    )
    data = SimpleNamespace(memory_manager=manager)
    data.cal_and_set_input = lambda seqs: setattr(data, "seqs", seqs)
    draft = SimpleNamespace(
        config=SimpleNamespace(target_layer_ids=(0,)),
        prefill_paged=lambda *args, **kwargs: object(),
    )
    for name in ("memory_allocated", "max_memory_allocated"):
        monkeypatch.setattr(torch.cuda, name, lambda: 0)
    for name in ("reset_peak_memory_stats", "synchronize"):
        monkeypatch.setattr(torch.cuda, name, lambda: None)

    with pytest.raises(ShorterProbeReached):
        profile_dspark_paged_verify(
            Target(), draft, data, max_prompt_tokens=32, model_max_length=32,
            uses_mrope=False, prepare_attention=lambda _: None,
        )


def _dspark_prompt_spy(monkeypatch, *, mrope=True):
    """CPU controller oracle; no real attention, allocation or CUDA execution."""
    from types import SimpleNamespace

    from gllm.speculative import dspark_execution as module
    from gllm.runtime.model_runner import ModelRunner
    from gllm.runtime.sequence import GenerationSequence
    from gllm.speculative.dspark import DSparkRequestState

    class Event:
        def __init__(self):
            self.records = 0

        def record(self, stream):
            self.records += 1

        def synchronize(self):
            self.waited = True

    monkeypatch.setattr(
        module, "DSparkRequestState",
        lambda *args: DSparkRequestState(*args, last_use=Event()),
    )
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: None)

    class Data:
        def cal_and_set_input(self, seqs):
            self.seqs = seqs
            seq = seqs[0]
            start, end = seq.computed_token_num, seq.seq_len
            tokens = seq[start:end] if seq.to_compute_tokens is None else seq.to_compute_tokens
            assert len(tokens) == seq.to_compute_token_num
            self.tokens = torch.tensor(tokens, dtype=torch.long)
            self.positions = torch.arange(start, end)
            self.prepared = False

        def set_mrope_position(self, positions):
            self.positions = positions

    class Target:
        def __init__(self):
            self.calls = []
            self.head_calls = 0
            self.scratch = torch.empty(1, 64, 8)

        def forward_with_hidden_taps(self, data, *, target_layer_ids):
            assert data.prepared and target_layer_ids == (1, 0)
            self.calls.append((data.tokens.tolist(), data.positions.clone()))
            count = data.tokens.numel()
            self.scratch[:, :count].copy_(data.tokens[None, :, None] + torch.arange(8))
            hidden = data.tokens[:, None].expand(count, 4).float()
            return hidden, self.scratch[:, :count]

        def __call__(self, data):
            assert data.prepared
            self.calls.append((data.tokens.tolist(), data.positions.clone()))
            return data.tokens[:, None].expand(data.tokens.numel(), 4).float()

        def logits_from_hidden(self, hidden):
            self.head_calls += 1
            assert hidden.shape == (1, 4)
            logits = torch.zeros(1, 32)
            logits[0, (int(hidden[0, 0]) + 3) % 32] = 1
            return logits

    class Draft:
        block_size = noise_width = 7
        config = SimpleNamespace(vocab_size=32, target_layer_ids=(1, 0))
        rotary_emb = SimpleNamespace(cos_sin_cache=torch.empty(64, 2))

        def __init__(self):
            self.chunks = []

        def prefill_paged(self, taps, *, segment, cache):
            self.chunks.append(taps.clone())
            cache.context_length += taps.shape[1]
            return SimpleNamespace(state=cache, segment=segment)

    class Manager:
        dspark_segment = object()
        allow = True

        def __init__(self):
            self.freed, self.reservations, self.freed_tables = [], [], []

        def free(self, seq):
            self.freed.append(seq.seq_id)
            self.freed_tables.append(list(seq.ssm_block_table or []))
            seq.page_table.clear()
            seq.ssm_block_table = None
            seq.recurrent_state_slot = None

        def free_dspark_cache(self, seq, cache):
            cache.reset()
            self.free(seq)

        def reserve_dspark_pages(self, seq, cache, *, target_length, draft_length):
            self.reservations.append((target_length, draft_length))
            if not self.allow or target_length > getattr(self, "limit", 64):
                return False
            for table, length in ((seq.page_table, target_length), (cache.page_table, draft_length)):
                while len(table) * 16 < length:
                    table.append(len(table) + 1)
            if seq.ssm_block_table is None:
                seq.ssm_block_table = list(range(1, 9))
                seq.recurrent_state_slot = 1
            return True

    runner = ModelRunner.__new__(ModelRunner)
    runner.dspark_enabled, runner.uses_mrope = True, mrope
    runner.model_max_length, runner.max_num_batched_tokens = 64, 16
    runner._dspark_requests = {}
    runner._mtp_relay, runner.use_mm = {}, False
    runner.model, runner.dspark = Target(), Draft()
    runner.input_data, runner.memory_manager = Data(), Manager()
    runner.page_size = 16
    runner._prepare_attention_metadata = lambda data: setattr(data, "prepared", True)
    seq = GenerationSequence(61, list(range(13)), [], output_len=16, temperature=0)
    seq.to_compute_token_num = 4
    return runner, seq


@pytest.mark.parametrize("mrope", [False, True])
def test_dspark_prompt_chunks_and_bootstrap_preserve_publication_and_pending(monkeypatch, mrope):
    from gllm.speculative.dspark import commit_dspark_prompt_step

    runner, seq = _dspark_prompt_spy(monkeypatch, mrope=mrope)
    published = []
    for count, end in ((4, 4), (5, 9), (4, 13)):
        seq.to_compute_token_num = count
        previous, history = seq.computed_token_num, list(seq.token_ids)
        result = runner._step_dspark_prompt(seq)
        state = runner._dspark_requests[seq.seq_id]
        assert seq.computed_token_num == previous and seq.token_ids == history
        assert state.cache.context_length == end and result.hidden is None
        assert state.paged_cache.state is state.cache
        assert runner.model.head_calls == (1 if end == 13 else 0)
        positions = torch.arange(previous, end)
        if mrope:
            positions = positions.repeat(3, 1)
        assert torch.equal(runner.model.calls[-1][1], positions)
        with pytest.raises(ValueError, match="uncommitted"):
            runner._step_dspark_prompt(seq)
        tokens = commit_dspark_prompt_step(seq, state, result)
        assert tokens == ([15] if end == 13 else [])
        published.extend(tokens)
        with pytest.raises(ValueError, match="already consumed"):
            commit_dspark_prompt_step(seq, state, result)
    assert published == [15]
    assert seq.token_ids == list(range(13)) + [15]
    assert state.phase == "bootstrap" and state.pending_hidden is None
    assert len(runner.dspark.chunks) == 3

    seq.to_compute_token_num = 1
    result = runner._step_dspark_prompt(seq)
    assert runner.model.calls[-1][0] == [15]
    assert state.cache.context_length == seq.computed_token_num == 13
    assert result.hidden.data_ptr() != runner.model.scratch.data_ptr()
    assert commit_dspark_prompt_step(seq, state, result) == []
    assert seq.token_ids == list(range(13)) + [15]
    assert seq.computed_token_num == state.anchor_pos == len(seq) == 14
    assert state.phase == "verify" and int(state.anchor.item()) == 18
    assert state.cache.context_length + state.pending_hidden.shape[1] == 14
    assert state.pending_hidden.shape == (1, 1, 8)
    assert state.last_use.records == 4
    pending = state.pending_hidden.clone()
    runner.model.scratch.fill_(-123)
    assert torch.equal(state.pending_hidden, pending)


@pytest.mark.parametrize("stop", ["budget", "eos"])
def test_dspark_prompt_first_token_can_finish_without_bootstrap(monkeypatch, stop):
    from gllm.speculative.dspark import commit_dspark_prompt_step

    runner, seq = _dspark_prompt_spy(monkeypatch)
    if stop == "budget":
        seq.output_len = 1
    else:
        seq.finish_tokens = [15]
    seq.to_compute_token_num = seq.prompt_len
    result = runner._step_dspark_prompt(seq)
    state = runner._dspark_requests[seq.seq_id]
    assert commit_dspark_prompt_step(seq, state, result) == [15]
    assert seq.is_finish and state.cache.context_length == 13
    seq.to_compute_token_num = 1
    with pytest.raises(ValueError):
        runner._step_dspark_prompt(seq)
    assert len(runner.model.calls) == 1 and state.pending_hidden is None


@pytest.mark.parametrize("invalid", [
    "sampling", "multimodal", "repetition", "structured", "logprobs",
    "prompt_logprobs", "empty", "budget", "chunk", "override", "mtp", "rope", "token", "autocast",
])
def test_dspark_prompt_rejects_unsupported_input_before_reservation(monkeypatch, invalid):
    runner, seq = _dspark_prompt_spy(monkeypatch)
    if invalid == "sampling":
        seq.temperature, seq.top_k = 0.7, 2
    elif invalid == "multimodal":
        seq.mm_contents = object()
    elif invalid == "repetition":
        seq.repetition_penalty = 1.1
    elif invalid == "structured":
        seq.structured_output = object()
    elif invalid == "logprobs":
        seq.logprobs_enabled = True
    elif invalid == "prompt_logprobs":
        seq.prompt_logprobs_enabled = True
    elif invalid == "empty":
        seq.token_ids, seq.raw_prompt_len, seq.prompt_len = [], 0, 0
    elif invalid == "budget":
        seq.output_len = 52
    elif invalid == "chunk":
        seq.to_compute_token_num = 17
    elif invalid == "override":
        seq.to_compute_tokens = [3] * 4
    elif invalid == "mtp":
        seq._mtp_verify = True
    elif invalid == "rope":
        runner.dspark.rotary_emb = type(runner.dspark.rotary_emb)(cos_sin_cache=torch.empty(2, 2))
    elif invalid == "autocast":
        monkeypatch.setattr(torch, "is_autocast_enabled", lambda *args: True)
    else:
        seq.token_ids[0] = 99
    with pytest.raises(ValueError):
        runner._step_dspark_prompt(seq)
    assert not runner._dspark_requests and not runner.model.calls
    assert not seq.page_table and seq.recurrent_state_slot is None


@pytest.mark.parametrize("existing", [False, True])
def test_dspark_prompt_failed_joint_reservation_preserves_progress(monkeypatch, existing):
    from gllm.runtime.sequence import RequestCapacityError
    from gllm.speculative.dspark import commit_dspark_prompt_step

    runner, seq = _dspark_prompt_spy(monkeypatch)
    if existing:
        result = runner._step_dspark_prompt(seq)
        state = runner._dspark_requests[seq.seq_id]
        commit_dspark_prompt_step(seq, state, result)
    state = runner._dspark_requests.get(seq.seq_id)
    before = (seq.computed_token_num, list(seq.token_ids), list(seq.page_table))
    runner.memory_manager.allow = False
    seq.to_compute_token_num = 4
    with pytest.raises(RequestCapacityError):
        runner._step_dspark_prompt(seq)
    assert before == (seq.computed_token_num, seq.token_ids, seq.page_table)
    assert len(runner.model.calls) == int(existing)
    if existing:
        assert state.cache.context_length == 4 and state.inflight is None
    else:
        assert not runner._dspark_requests


@pytest.mark.parametrize("invalid", ["epoch", "position", "query", "foreign", "abort", "phase"])
def test_dspark_prompt_commit_rejects_stale_result_without_progress(monkeypatch, invalid):
    from dataclasses import replace
    from gllm.speculative.dspark import commit_dspark_prompt_step

    runner, seq = _dspark_prompt_spy(monkeypatch)
    result = runner._step_dspark_prompt(seq)
    state = runner._dspark_requests[seq.seq_id]
    if invalid == "epoch":
        seq._cache_epoch += 1
    elif invalid == "position":
        seq.computed_token_num += 1
    elif invalid == "query":
        seq.to_compute_token_num -= 1
    elif invalid == "foreign":
        result = replace(result)
    elif invalid == "abort":
        seq.is_abort = True
    else:
        result.phase = "bootstrap"
    before = (seq.computed_token_num, list(seq.token_ids), state.phase)
    with pytest.raises(ValueError):
        commit_dspark_prompt_step(seq, state, result)
    assert before == (seq.computed_token_num, seq.token_ids, state.phase)
    assert not result.consumed and state.cache.context_length == 4


def test_dspark_prompt_partial_forward_failure_blocks_retry(monkeypatch):
    runner, seq = _dspark_prompt_spy(monkeypatch)

    def fail_forward(*args, **kwargs):
        raise RuntimeError("target forward failed")

    monkeypatch.setattr(runner.model, "forward_with_hidden_taps", fail_forward)
    with pytest.raises(RuntimeError, match="target forward failed"):
        runner._step_dspark_prompt(seq)
    state = runner._dspark_requests[seq.seq_id]
    assert state.inflight is not None
    assert seq.computed_token_num == state.cache.context_length == 0
    with pytest.raises(ValueError, match="uncommitted"):
        runner._step_dspark_prompt(seq)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.no_grad()
@pytest.mark.parametrize("chunks", [(1,), (23,), (1, 8, 7, 7), (8, 8, 7)])
def test_dspark_prompt_runtime_matches_ordinary_prefill_and_bootstrap_in_shared_arena(chunks):
    from gllm.runtime.sequence import GenerationSequence
    from gllm.speculative.dspark import commit_dspark_prompt_step

    runner, make_input = _dspark_runtime_fixture()
    arena = _init_dspark_runtime_test_arena(runner)
    owned_before = arena.allocator.num_used_physical_pages
    prompt_len = sum(chunks)
    prompt = list(range(1, prompt_len + 1))
    seq = GenerationSequence(72, prompt.copy(), [], output_len=9, temperature=0)
    oracle, oracle_manager, oracle_seq = make_input(prompt)
    text = runner.model.language_model
    captured = []
    original_prefill = runner.dspark.prefill_paged

    def capture(taps, **kwargs):
        captured.append(taps.clone())
        return original_prefill(taps, **kwargs)

    runner.dspark.prefill_paged = capture
    published = []
    for count in chunks:
        start = seq.computed_token_num
        end = start + count
        seq.to_compute_token_num = oracle_seq.to_compute_token_num = count
        oracle.cal_and_set_input([oracle_seq])
        oracle.set_mrope_position(torch.arange(start, end).repeat(3, 1))
        oracle.forward_metadata_plan.prepare_attention(runner.qkv_attention_backend, oracle)
        ordinary = text(oracle)
        result = runner._step_dspark_prompt(seq)
        state = runner._dspark_requests[seq.seq_id]
        assert seq.computed_token_num == start
        assert state.cache.context_length == end
        tokens = commit_dspark_prompt_step(seq, state, result)
        if end == len(prompt):
            expected = int(text.logits_from_hidden(ordinary[-1:]).argmax(-1).item())
            assert tokens == [expected]
        else:
            assert tokens == []
        published.extend(tokens)
        oracle_seq.computed_token_num = end

    assert len(published) == 1 and seq.token_ids == prompt + published
    oracle_seq.append(published[0])
    seq.to_compute_token_num = oracle_seq.to_compute_token_num = 1
    oracle.cal_and_set_input([oracle_seq])
    oracle.set_mrope_position(torch.arange(prompt_len, prompt_len + 1).repeat(3, 1))
    oracle.forward_metadata_plan.prepare_attention(runner.qkv_attention_backend, oracle)
    ordinary = text(oracle)
    expected_anchor = text.logits_from_hidden(ordinary[-1:]).argmax(-1)

    result = runner._step_dspark_prompt(seq)
    assert runner.input_data.get_position().shape == (3, 1)
    assert not runner.input_data.get_position().is_contiguous()
    assert commit_dspark_prompt_step(seq, state, result) == []
    assert torch.equal(state.anchor, expected_anchor)
    assert seq.token_ids == prompt + published
    assert state.anchor_pos == seq.computed_token_num == len(seq) == prompt_len + 1
    assert state.cache.context_length == prompt_len and state.pending_hidden.shape[1] == 1

    # Only prompt rows are valid draft K/V; bootstrap stays in pending until
    # the next draft. Compare the chunked writes with an independent dense view.
    dense = runner.dspark.prefill(torch.cat(captured, dim=1), state.anchor)
    keys, values = _read_paged_context(
        runner.memory_manager.dspark_segment, state.cache, len(prompt)
    )
    for got, want in zip(keys + values, dense.keys + dense.values):
        assert torch.isfinite(got).all() and torch.isfinite(want).all()
        torch.testing.assert_close(got, want, rtol=2e-2, atol=2e-2)
    target_keys, target_values = _read_paged_context(
        runner.memory_manager.segment, seq, prompt_len + 1
    )
    oracle_keys, oracle_values = _read_paged_context(
        oracle_manager.segment, oracle_seq, prompt_len + 1
    )
    for got, want in zip(target_keys + target_values, oracle_keys + oracle_values):
        torch.testing.assert_close(got, want, rtol=0, atol=0)
    for name in ("conv_state", "temporal_state"):
        actual = getattr(runner.memory_manager.ssm_segment, name)[:, seq.recurrent_state_slot]
        expected = getattr(oracle_manager.ssm_segment, name)[:, oracle_seq.recurrent_state_slot]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    runner.free(seq)
    oracle_manager.free(oracle_seq)
    assert arena.allocator.num_used_physical_pages == owned_before


@pytest.mark.parametrize("reuse", ["epoch", "other_request"])
def test_dspark_prompt_request_record_rejects_reuse_before_release(monkeypatch, reuse):
    from gllm.runtime.sequence import GenerationSequence
    from gllm.speculative.dspark import commit_dspark_prompt_step

    runner, seq = _dspark_prompt_spy(monkeypatch)
    result = runner._step_dspark_prompt(seq)
    state = runner._dspark_requests[seq.seq_id]
    commit_dspark_prompt_step(seq, state, result)
    if reuse == "epoch":
        seq._cache_epoch += 1
    else:
        seq = GenerationSequence(62, list(range(13)), [], output_len=16, temperature=0)
    seq.to_compute_token_num = 4
    with pytest.raises(ValueError):
        runner._step_dspark_prompt(seq)
    assert len(runner.model.calls) == 1
    assert list(runner._dspark_requests.values()) == [state]
    assert state.cache.context_length == 4 and state.inflight is None


@pytest.mark.parametrize("invalid", ["query", "missing_first_token", "progress"])
def test_dspark_prompt_bootstrap_rejects_invalid_window_before_forward(monkeypatch, invalid):
    from gllm.speculative.dspark import commit_dspark_prompt_step

    runner, seq = _dspark_prompt_spy(monkeypatch)
    seq.to_compute_token_num = seq.prompt_len
    result = runner._step_dspark_prompt(seq)
    state = runner._dspark_requests[seq.seq_id]
    commit_dspark_prompt_step(seq, state, result)
    seq.to_compute_token_num = 1
    if invalid == "query":
        seq.to_compute_token_num = 2
    elif invalid == "missing_first_token":
        seq.token_ids.pop()
    else:
        seq.computed_token_num += 1
    with pytest.raises(ValueError):
        runner._step_dspark_prompt(seq)
    assert len(runner.model.calls) == 1 and state.cache.context_length == 13
    assert state.pending_hidden is None and state.inflight is None


def _dspark_execution_spy(monkeypatch, *, accepted=2):
    """Direct runner oracle; no Worker/Scheduler or real CUDA execution."""
    from gllm.speculative import dspark_execution as module
    from gllm.speculative.dspark import DSparkVerifyResult, commit_dspark_prompt_step

    runner, seq = _dspark_prompt_spy(monkeypatch)
    seq.output_len = 32
    tables = []

    def draft_forward(pending, anchor, *, start_pos, caches):
        assert caches.state.context_length + pending.shape[1] == start_pos
        caches.state.context_length = start_pos
        return torch.zeros(1, 7, 4), None

    def draft_head(hidden, anchor):
        first = int(anchor.item())
        return torch.tensor([[first] + [(first + i) % 32 for i in range(1, 8)]]), None, None

    def verify(target, data, *, block_size, target_layer_ids):
        request = data.seqs[0]
        assert request._mtp_verify and request.to_compute_token_num == 8
        assert len(request) == request.computed_token_num
        assert block_size == 7 and target_layer_ids == (1, 0)
        tables.append(list(request.ssm_block_table))
        length = accepted + 1
        return DSparkVerifyResult(
            request.seq_id, request._cache_epoch, request.computed_token_num, accepted,
            data.tokens[:length].unsqueeze(0).clone(),
            torch.tensor([(int(data.tokens[length - 1]) + 1) % 32]),
            torch.ones(1, length, 8),
        )

    runner.dspark.forward_draft, runner.dspark.forward_head = draft_forward, draft_head
    monkeypatch.setattr(module, "verify_dspark_target", verify)
    for count in (8, 5, 1):
        seq.to_compute_token_num = count
        result = runner._step_dspark_prompt(seq)
        commit_dspark_prompt_step(seq, runner._dspark_requests[seq.seq_id], result)
    return runner, seq, tables


@pytest.mark.parametrize("accepted", [0, 2, 7])
def test_dspark_execution_verify_commits_prefix_and_reuses_promoted_table(monkeypatch, accepted):
    from gllm.speculative.dspark import commit_dspark_verify_step

    runner, seq, tables = _dspark_execution_spy(monkeypatch, accepted=accepted)
    state = runner._dspark_requests[seq.seq_id]
    for _ in range(2):
        start, anchor = seq.computed_token_num, int(state.anchor.item())
        history, table = list(seq.token_ids), list(seq.ssm_block_table)
        seq.to_compute_token_num = 8
        result = runner._step_dspark_verify(seq)
        assert seq.token_ids == history and seq.computed_token_num == start
        assert result.ready and not result.consumed
        assert seq.to_compute_tokens is None and not seq._mtp_verify
        assert runner.input_data.positions.tolist() == [list(range(start, start + 8))] * 3
        expected = [(anchor + i) % 32 for i in range(accepted + 1)]
        assert commit_dspark_verify_step(seq, state, result) == expected
        assert seq.token_ids == history + expected
        assert seq.computed_token_num == len(seq) == start + accepted + 1
        assert state.cache.context_length == start
        assert state.pending_hidden.shape[1] == accepted + 1
        assert state.anchor_pos == len(seq) and state.inflight is None
        table[0], table[accepted] = table[accepted], table[0]
        assert seq.ssm_block_table == table and seq.recurrent_state_slot == table[0]
        assert result.consumed and result.verification.consumed
        before = (list(seq.token_ids), seq.computed_token_num, list(seq.ssm_block_table))
        with pytest.raises(ValueError, match="already consumed"):
            commit_dspark_verify_step(seq, state, result)
        assert before == (seq.token_ids, seq.computed_token_num, seq.ssm_block_table)
    assert tables[1][0] == tables[0][accepted]


@pytest.mark.parametrize("cutoff", ["eos", "budget"])
def test_dspark_execution_verify_cutoff_promotes_only_retained_inputs(monkeypatch, cutoff):
    from gllm.speculative.dspark import commit_dspark_verify_step

    runner, seq, tables = _dspark_execution_spy(monkeypatch, accepted=7)
    state = runner._dspark_requests[seq.seq_id]
    seq.to_compute_token_num = 8
    result = runner._step_dspark_verify(seq)
    if cutoff == "eos":
        seq.finish_tokens = [20]
        expected = [18, 19, 20]
    else:
        seq.output_len = 3
        expected = [18, 19]
    assert commit_dspark_verify_step(seq, state, result) == expected
    assert seq.computed_token_num == len(seq) == 14 + len(expected)
    assert seq.recurrent_state_slot == tables[0][len(expected) - 1]
    assert seq.is_finish and state.anchor is state.pending_hidden is None
    event = state.last_use
    runner.free(seq)
    assert event.waited and not runner._dspark_requests
    assert seq.page_table == [] and seq.ssm_block_table is None
    assert state.cache.context_length == 0 and state.cache.page_table == []


@pytest.mark.parametrize("phase", ["verify", "tail"])
def test_dspark_execution_failed_reservation_preserves_pending_and_progress(monkeypatch, phase):
    from gllm.runtime.sequence import RequestCapacityError

    runner, seq, tables = _dspark_execution_spy(monkeypatch)
    state = runner._dspark_requests[seq.seq_id]
    state.phase = phase
    seq.to_compute_token_num = 8 if phase == "verify" else 1
    before = (list(seq.token_ids), seq.computed_token_num, list(seq.ssm_block_table))
    pending, anchor = state.pending_hidden, state.anchor
    reservations, calls = len(runner.memory_manager.reservations), len(runner.model.calls)
    runner.memory_manager.allow = False
    with pytest.raises(RequestCapacityError):
        getattr(runner, "_step_dspark_" + phase)(seq)
    assert before == (seq.token_ids, seq.computed_token_num, seq.ssm_block_table)
    assert state.pending_hidden is pending and state.anchor is anchor
    assert state.cache.context_length == 13 and state.inflight is None
    assert len(runner.memory_manager.reservations) == reservations + 1
    assert len(runner.model.calls) == calls and not tables


@pytest.mark.parametrize("limit", ["budget", "rope"])
def test_dspark_execution_rejects_short_verify_window_before_reservation(monkeypatch, limit):
    runner, seq, tables = _dspark_execution_spy(monkeypatch)
    state = runner._dspark_requests[seq.seq_id]
    if limit == "budget":
        seq.output_len = 8  # One token is already published, leaving seven.
    else:
        runner.dspark.rotary_emb.cos_sin_cache = torch.empty(20, 2)
    seq.to_compute_token_num = 8
    pending, anchor = state.pending_hidden, state.anchor
    reservations = list(runner.memory_manager.reservations)
    assert not runner._dspark_verify_fits(seq)
    with pytest.raises(ValueError, match="verify window"):
        runner._step_dspark_verify(seq)
    assert runner.memory_manager.reservations == reservations and not tables
    assert state.pending_hidden is pending and state.anchor is anchor
    assert state.cache.context_length == 13 and state.inflight is None
    assert seq.computed_token_num == len(seq) == 14


@pytest.mark.parametrize("phase", ["verify", "tail"])
@pytest.mark.parametrize("invalid", ["epoch", "query", "foreign"])
def test_dspark_execution_commit_rejects_stale_result_without_mutation(monkeypatch, phase, invalid):
    from dataclasses import replace
    from gllm.speculative.dspark import commit_dspark_tail_step, commit_dspark_verify_step

    runner, seq, _ = _dspark_execution_spy(monkeypatch)
    state = runner._dspark_requests[seq.seq_id]
    state.phase = phase
    seq.to_compute_token_num = 8 if phase == "verify" else 1
    result = getattr(runner, "_step_dspark_" + phase)(seq)
    if invalid == "epoch":
        seq._cache_epoch += 1
    elif invalid == "query":
        seq.to_compute_token_num += 1
    else:
        result = replace(result)
    before = (list(seq.token_ids), seq.computed_token_num, list(seq.ssm_block_table))
    commit = commit_dspark_verify_step if phase == "verify" else commit_dspark_tail_step
    with pytest.raises(ValueError):
        commit(seq, state, result)
    assert before == (seq.token_ids, seq.computed_token_num, seq.ssm_block_table)
    assert not result.consumed
    if result.verification is not None:
        assert not result.verification.consumed


@pytest.mark.parametrize("boundary", ["budget", "rope", "context"])
def test_dspark_execution_tail_consumes_relay_without_growing_draft(monkeypatch, boundary):
    from gllm.speculative.dspark import commit_dspark_tail_step

    runner, seq, tables = _dspark_execution_spy(monkeypatch)
    seq.output_len = 4 if boundary != "rope" else 10
    if boundary == "context":
        runner.model_max_length = seq.raw_prompt_len + seq.output_len
    elif boundary == "rope":
        runner.dspark.rotary_emb.cos_sin_cache = torch.empty(20, 2)
    state = runner._dspark_requests[seq.seq_id]
    assert not runner._dspark_verify_fits(seq)
    state.phase = "tail"
    published = [15]
    while not seq.is_finish:
        start, anchor = seq.computed_token_num, int(state.anchor.item())
        seq.to_compute_token_num = 1
        result = runner._step_dspark_tail(seq)
        assert runner.model.calls[-1][0] == [anchor]
        assert runner.input_data.positions.tolist() == [[start]] * 3
        assert runner.memory_manager.reservations[-1] == (start + 1, 13)
        assert result.hidden is result.verification is None
        published.extend(commit_dspark_tail_step(seq, state, result))
        assert state.cache.context_length == 13
    assert published == [(15 + 3 * i) % 32 for i in range(seq.output_len)]
    assert seq.token_ids == list(range(13)) + published
    assert seq.computed_token_num == len(seq) and not tables
    assert state.anchor is state.pending_hidden is None
    with pytest.raises(ValueError, match="already consumed"):
        commit_dspark_tail_step(seq, state, result)
    runner.free(seq)
    assert not runner._dspark_requests


@pytest.mark.parametrize("failure", ["request", "fatal", "event"])
def test_dspark_execution_failed_verify_requires_a_new_completion_boundary(monkeypatch, failure):
    from gllm.speculative import dspark_execution as module

    runner, seq, _ = _dspark_execution_spy(monkeypatch)
    state = runner._dspark_requests[seq.seq_id]
    seq.to_compute_token_num = 8
    history, progress = list(seq.token_ids), seq.computed_token_num
    records = state.last_use.records

    def fail(*args, **kwargs):
        raise ValueError("request validation") if failure != "fatal" else RuntimeError("CUDA failure")

    monkeypatch.setattr(module, "verify_dspark_target", fail)
    with pytest.raises(ValueError if failure != "fatal" else RuntimeError):
        runner._step_dspark_verify(seq)
    assert seq.token_ids == history and seq.computed_token_num == progress
    assert seq.to_compute_tokens is None and not seq._mtp_verify
    assert state.cache.context_length == progress and state.pending_hidden is None
    assert state.last_use.records == records and not state.inflight.ready
    with pytest.raises(ValueError, match="uncommitted"):
        runner._step_dspark_verify(seq)
    with pytest.raises(RuntimeError, match="safe completion event"):
        runner.free(seq)
    assert not runner.memory_manager.freed
    if failure == "fatal":
        return
    if failure == "event":
        def fail_record(*args, **kwargs):
            raise RuntimeError("CUDA event recording failed")
        monkeypatch.setattr(state.last_use, "record", fail_record)
        with pytest.raises(RuntimeError, match="event recording failed"):
            runner.complete_dspark_failed_step(seq)
        assert not state.inflight.ready and not runner.memory_manager.freed
        return
    runner.complete_dspark_failed_step(seq)
    assert state.last_use.records == records + 1 and state.inflight.ready
    runner.free(seq)
    assert not runner._dspark_requests and state.last_use is None
    assert state.anchor is state.pending_hidden is state.paged_cache is None


def test_dspark_execution_free_clears_request_before_id_reuse(monkeypatch):
    from gllm.runtime.sequence import GenerationSequence
    from gllm.speculative.dspark import commit_dspark_prompt_step

    runner, seq, _ = _dspark_execution_spy(monkeypatch)
    old = runner._dspark_requests[seq.seq_id]
    event = old.last_use
    runner.free(seq)
    assert event.waited and not old.cache.page_table
    assert old.cache.context_length == 0 and not runner._dspark_requests
    assert old.anchor is old.pending_hidden is old.paged_cache is old.last_use is None
    reused = GenerationSequence(seq.seq_id, list(range(13)), [], output_len=16, temperature=0)
    reused.to_compute_token_num = 4
    result = runner._step_dspark_prompt(reused)
    state = runner._dspark_requests[reused.seq_id]
    assert state is not old and state.cache is not old.cache
    assert state.anchor is state.pending_hidden is None
    assert commit_dspark_prompt_step(reused, state, result) == []
    assert state.cache.context_length == reused.computed_token_num == 4


def test_dspark_execution_ordinary_free_preserves_default_path(monkeypatch):
    runner, seq = _dspark_prompt_spy(monkeypatch)
    runner.dspark_enabled = False
    runner._mtp_relay[seq.seq_id] = object()
    runner.free(seq)
    assert runner.memory_manager.freed == [seq.seq_id]
    assert not runner._mtp_relay and not runner._dspark_requests


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.no_grad()
def test_dspark_execution_matches_greedy_with_controlled_and_real_draft():
    from gllm.runtime.input_data import InputData
    from gllm.runtime.sequence import GenerationSequence

    from gllm.speculative.dspark import (
        commit_dspark_prompt_step, commit_dspark_tail_step, commit_dspark_verify_step,
    )

    runner, make_input = _dspark_runtime_fixture()
    runner.model_max_length = 64
    runner.input_data = InputData(True, runner.memory_manager, max_seq_length=64, max_running_seqs=1)
    arena = _init_dspark_runtime_test_arena(runner)
    owned_before = arena.allocator.num_used_physical_pages
    prompt = list(range(1, 14))
    seq = GenerationSequence(101, prompt.copy(), [], output_len=51, temperature=0)
    oracle, oracle_manager, oracle_seq = make_input(prompt)
    oracle_seq.output_len = 51
    text = runner.model.language_model
    features = []
    real_prefill = runner.dspark.prefill_paged

    def capture_prompt(taps, **kwargs):
        features.append(taps.clone())
        return real_prefill(taps, **kwargs)

    runner.dspark.prefill_paged = capture_prompt

    def ordinary_step(token_ids):
        start = oracle_seq.computed_token_num
        oracle_seq.to_compute_tokens = list(token_ids)
        oracle_seq.to_compute_token_num = len(token_ids)
        oracle_manager.pre_allocate_page_for_lengths(
            [oracle_seq], [start + len(token_ids)]
        )
        oracle.cal_and_set_input([oracle_seq])
        oracle.set_mrope_position(torch.arange(start, start + len(token_ids)).repeat(3, 1))
        oracle.forward_metadata_plan.prepare_attention(runner.qkv_attention_backend, oracle)
        hidden = text(oracle)
        oracle_seq.computed_token_num += len(token_ids)
        return int(text.logits_from_hidden(hidden[-1:]).argmax(-1).item())

    for chunk in (prompt[:8], prompt[8:]):
        prediction = ordinary_step(chunk)
        seq.to_compute_token_num = len(chunk)
        record = runner._step_dspark_prompt(seq)
        state = runner._dspark_requests[seq.seq_id]
        published = commit_dspark_prompt_step(seq, state, record)
        assert seq.computed_token_num == oracle_seq.computed_token_num
    assert published == [prediction]
    oracle_seq.append(prediction)
    expected_anchor = ordinary_step([prediction])
    seq.to_compute_token_num = 1
    record = runner._step_dspark_prompt(seq)
    assert commit_dspark_prompt_step(seq, state, record) == []
    assert int(state.anchor.item()) == expected_anchor
    assert seq.token_ids == oracle_seq.token_ids

    real_head = runner.dspark.forward_head
    for desired in (0, 0, 2, 7, None):
        start = seq.computed_token_num
        anchor = int(state.anchor.item())
        expected_inputs, predictions, snapshots = [], [], []
        for _ in range(8):
            expected_inputs.append(anchor)
            anchor = ordinary_step([anchor])
            predictions.append(anchor)
            slot = oracle_seq.recurrent_state_slot
            snapshots.append((
                oracle_manager.ssm_segment.conv_state[:, slot].clone(),
                oracle_manager.ssm_segment.temporal_state[:, slot].clone(),
            ))
        proposals = []

        def head(hidden, first):
            ids, confidence, logits = real_head(hidden, first)
            ids = ids.clone()
            # Controlled boundaries still execute the real draft/head. The
            # final round consumes its genuine proposals without substitution.
            if desired is not None:
                chosen = expected_inputs.copy()
                if desired < 7:
                    chosen[desired + 1] = (chosen[desired + 1] + 1) % runner.dspark.config.vocab_size
                ids[0].copy_(torch.tensor(chosen, device=ids.device))
            proposals[:] = ids[0].cpu().tolist()
            return ids, confidence, logits

        runner.dspark.forward_head = head
        features.append(state.pending_hidden.clone())
        history, table = list(seq.token_ids), list(seq.ssm_block_table)
        seq.to_compute_token_num = 8
        record = runner._step_dspark_verify(seq)
        assert seq.token_ids == history and seq.computed_token_num == start
        assert record.query_len == 8 and record.ready
        assert seq.to_compute_tokens is None and not seq._mtp_verify
        assert runner.input_data.forward_metadata_plan.query_lens == (8,)
        accepted = 0
        for got, want in zip(proposals[1:], expected_inputs[1:]):
            if got != want:
                break
            accepted += 1
        if desired is not None:
            assert accepted == desired
        count = accepted + 1
        assert commit_dspark_verify_step(seq, state, record) == expected_inputs[:count]
        assert record.verification.accepted_drafts == accepted
        assert record.consumed and record.verification.consumed
        assert seq.computed_token_num == start + count == len(seq)
        assert state.cache.context_length == start
        assert state.pending_hidden.shape[1] == count and state.anchor_pos == len(seq)
        assert int(state.anchor.item()) == predictions[accepted]
        assert seq.recurrent_state_slot == table[accepted] and seq.ssm_num_accepted == 1
        for name, expected, tolerance in (
            ("conv_state", snapshots[accepted][0], 2e-3),
            ("temporal_state", snapshots[accepted][1], 2e-5),
        ):
            actual = getattr(runner.memory_manager.ssm_segment, name)[:, seq.recurrent_state_slot]
            torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
        oracle_seq.computed_token_num = start + count
        oracle_seq.token_ids.extend(expected_inputs[:count])
        oracle_slot = oracle_seq.recurrent_state_slot
        oracle_manager.ssm_segment.conv_state[:, oracle_slot].copy_(snapshots[accepted][0])
        oracle_manager.ssm_segment.temporal_state[:, oracle_slot].copy_(snapshots[accepted][1])
        assert seq.token_ids == oracle_seq.token_ids
        actual_k, actual_v = _read_paged_context(runner.memory_manager.segment, seq, len(seq))
        oracle_k, oracle_v = _read_paged_context(oracle_manager.segment, oracle_seq, len(seq))
        for actual, expected in zip(actual_k + actual_v, oracle_k + oracle_v):
            torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
        dense = runner.dspark.prefill(torch.cat(features, dim=1), state.anchor)
        actual_k, actual_v = _read_paged_context(
            runner.memory_manager.dspark_segment, state.cache, start
        )
        for actual, expected in zip(actual_k + actual_v, dense.keys + dense.values):
            torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)

    # Finish through the ordinary one-token relay while keeping draft history
    # fixed. The oracle uses its independent target cache at the same positions.
    draft_length = state.cache.context_length
    seq.output_len = len(seq) - seq.raw_prompt_len + 3
    oracle_seq.output_len = seq.output_len
    state.phase = "tail"
    for _ in range(3):
        anchor = int(state.anchor.item())
        predicted = ordinary_step([anchor])
        seq.to_compute_token_num = 1
        record = runner._step_dspark_tail(seq)
        assert commit_dspark_tail_step(seq, state, record) == [anchor]
        oracle_seq.append(anchor)
        assert seq.token_ids == oracle_seq.token_ids
        assert seq.computed_token_num == oracle_seq.computed_token_num
        assert state.cache.context_length == draft_length
        if not seq.is_finish:
            assert int(state.anchor.item()) == predicted
        for name, tolerance in (("conv_state", 2e-3), ("temporal_state", 2e-5)):
            actual = getattr(runner.memory_manager.ssm_segment, name)[:, seq.recurrent_state_slot]
            expected = getattr(oracle_manager.ssm_segment, name)[:, oracle_seq.recurrent_state_slot]
            torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
        actual_k, actual_v = _read_paged_context(runner.memory_manager.segment, seq, len(seq))
        oracle_k, oracle_v = _read_paged_context(oracle_manager.segment, oracle_seq, len(seq))
        for actual, expected in zip(actual_k + actual_v, oracle_k + oracle_v):
            torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
    assert seq.is_finish and state.anchor is state.pending_hidden is None

    runner.free(seq)
    torch.cuda.synchronize()
    oracle_manager.free(oracle_seq)
    assert not runner._dspark_requests
    assert arena.allocator.num_used_physical_pages == owned_before
