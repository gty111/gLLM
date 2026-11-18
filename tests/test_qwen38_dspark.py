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


@pytest.mark.parametrize("width", [1, 3, 7])
def test_forward_head_aligns_markov_and_confidence_with_previous_tokens(width):
    """A hand-computed alternating chain pins each proposal's conditioning."""
    from gllm.models.qwen38_dspark import Qwen38DSpark

    model = _head_stub(width)
    with torch.no_grad():
        # Token 0 contributes [-2, 2]; token 1 contributes [3, -3].
        # With zero base logits, greedy proposals alternate 1, 0, 1, ...
        model.markov_head.markov_w1.weight[:2, 0] = torch.tensor([-2., 3.])
        model.markov_head.markov_w2.weight[:2, 0] = torch.tensor([1., -1.])
        # Raw confidence = 2 * hidden[0] + previous-token embedding + 5.
        model.confidence_head.proj.weight[0] = torch.tensor([2., 0., 1.])
        model.confidence_head.proj.bias.fill_(5)

    rows = torch.arange(1, width + 1, dtype=torch.float32)
    hidden = torch.stack((rows, rows * 10), dim=-1).unsqueeze(0)
    ids, logits, confidence = Qwen38DSpark.forward_head(
        model, hidden, torch.tensor([0])
    )

    expected_logits = torch.zeros(1, 7, 8)
    expected_logits[0, :, :2] = torch.tensor([
        [-2., 2.], [3., -3.], [-2., 2.], [3., -3.],
        [-2., 2.], [3., -3.], [-2., 2.],
    ])
    assert ids.tolist() == [[0] + [1, 0, 1, 0, 1, 0, 1][:width]]
    torch.testing.assert_close(logits, expected_logits[:, :width], rtol=0, atol=0)
    torch.testing.assert_close(
        confidence, torch.tensor([[5., 12., 9., 16., 13., 20., 17.]])[:, :width],
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


def _build_model(config, *, speculative_num_draft_tokens=None):
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
        return Qwen38DSpark(
            config, embed=embed, lm_head=lm_head, speculative_num_draft_tokens=speculative_num_draft_tokens
        )
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


@pytest.mark.parametrize("invalid", ["empty_batch", "multimodal", "multimodal_second", "empty", "hidden", "residual"])
def test_target_taps_reject_invalid_input_before_forward(invalid):
    wrapper, model, data = _target_tap_fixture()
    kwargs = {}
    if invalid == "empty_batch":
        data.seqs = []
    elif invalid == "multimodal_second":
        from types import SimpleNamespace
        data.seqs.append(SimpleNamespace(mm_contents=object()))
    elif invalid == "multimodal":
        data.seqs[0].mm_contents = object()
    elif invalid == "empty":
        data.get_tokens = lambda: torch.empty(0, dtype=torch.long)
    else:
        kwargs["hidden_states" if invalid == "hidden" else "residual"] = torch.zeros(4, 4)
    with pytest.raises(ValueError):
        wrapper.forward_with_hidden_taps(data, target_layer_ids=(5,), **kwargs)
    assert all(layer.calls == 0 for layer in model.layers)


def _hybrid_target_fixture(*, speculative_num_draft_tokens=0, arena_capacity=16):
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
            ssm_cache_config=model.ssm_cache_config, speculative_num_draft_tokens=speculative_num_draft_tokens,
        )
        # Match init_arena's unquantized KV writer metadata.
        manager.kv_cache_dtype = "auto"
        manager.k_scale = torch.tensor(1.0, dtype=torch.float32, device="cuda")
        manager.v_scale = manager.k_scale
        layouts = manager._cache_layouts()
        grain = select_physical_page_bytes(layouts.values())
        pages = arena_capacity * sum((layout.entry_bytes + grain - 1) // grain for layout in layouts.values())
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
@pytest.mark.parametrize("count", [None, 1, 3, 7])
def test_dspark_loader_reads_indexed_checkpoint_and_borrows_target_weights(tmp_path, count):
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
    loaded = load_dspark_draft(
        str(tmp_path), target, model_max_length=32 if count is None else count + 2,
        speculative_num_draft_tokens=count,
    )
    assert loaded.config.block_size == 7
    assert loaded.block_size == loaded.noise_width == (7 if count is None else count)

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


@pytest.mark.parametrize("lengths", [(1, 3), (2, 1, 4, 3)])
@torch.no_grad()
def test_target_taps_packed_batch_preserves_request_rows_and_storage(lengths):
    from types import SimpleNamespace

    wrapper, model, data = _target_tap_fixture()
    tokens = torch.arange(2, sum(lengths) + 2, dtype=torch.long)
    data.get_tokens = lambda: tokens
    data.seqs = [SimpleNamespace(mm_contents=None) for _ in lengths]
    embedded = model.embed_tokens(tokens).clone()
    hidden, taps = wrapper.forward_with_hidden_taps(data, target_layer_ids=(5, 19))
    expected = torch.cat(
        [embedded + (layer + 1) * (layer + 2) / 64 for layer in (5, 19)], dim=-1
    ).unsqueeze(0)
    assert all(layer.calls == 1 for layer in model.layers)
    begin = 0
    for length in lengths:
        torch.testing.assert_close(
            taps[:, begin:begin + length], expected[:, begin:begin + length], rtol=0, atol=0
        )
        begin += length
    torch.testing.assert_close(hidden, (embedded + 65) * 3, rtol=0, atol=0)
    saved = taps.clone()
    wrapper.forward_with_hidden_taps(data, embedded + 1, target_layer_ids=(19, 5))
    torch.testing.assert_close(taps, saved, rtol=0, atol=0)


def test_paged_batch_rope_keeps_each_request_absolute_positions():
    from gllm.models.qwen38_dspark import _rotate

    torch.manual_seed(101)
    queries = torch.randn(4, 2, 7, 16)
    rotary = torch.randn(64, 16)
    positions = torch.stack([torch.arange(start, start + 7) for start in (0, 3, 17, 31)])
    got = _rotate(queries, rotary, positions)
    expected = torch.cat([
        _rotate(queries[index:index + 1], rotary, positions[index])
        for index in range(4)
    ])
    torch.testing.assert_close(got, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("batch", [2, 4])
@torch.no_grad()
def test_target_taps_packed_batch_matches_independent_hybrid_states(batch):
    from gllm.runtime.input_data import InputData
    from gllm.runtime.sequence import GenerationSequence

    model, _, make_input = _hybrid_target_fixture()
    token_rows = [tuple(range(1 + index, 4 + 2 * index)) for index in range(batch)]
    _, manager, first = make_input(token_rows[0])
    seqs = [first]
    for index, tokens in enumerate(token_rows[1:], start=2):
        seq = GenerationSequence(index, list(tokens), [], output_len=8, temperature=0)
        seq.page_table = [manager.segment.allocate()]
        assert manager.allocate_recurrent_slot(seq)
        seq.to_compute_token_num = len(tokens)
        seqs.append(seq)

    class IndependentCausalAttention:
        # Test oracle only: real target projections/RoPE/GDN/taps remain batched.
        def forward(self, q, key, value, metadata, scale):
            groups = [metadata] if isinstance(metadata, tuple) else metadata
            outputs, begin = [], 0
            for pages, length in groups:
                k = key[pages].reshape(-1, key.shape[-2], key.shape[-1])[:length]
                v = value[pages].reshape(-1, value.shape[-2], value.shape[-1])[:length]
                output = torch.nn.functional.scaled_dot_product_attention(
                    q[begin:begin + length].transpose(0, 1).unsqueeze(0),
                    k.transpose(0, 1).unsqueeze(0), v.transpose(0, 1).unsqueeze(0),
                    is_causal=True, scale=scale, enable_gqa=True,
                )
                outputs.append(output[0].transpose(0, 1))
                begin += length
            return torch.cat(outputs).contiguous()

    for layer in model.model.layers:
        if layer.self_attn is not None:
            layer.self_attn.attn.set_backend(IndependentCausalAttention())
    data = InputData(True, manager, max_seq_length=64, max_running_seqs=batch)
    data.cal_and_set_input(seqs)
    data.forward_metadata_plan.attention_metadata = [
        (seq.page_table, len(tokens)) for seq, tokens in zip(seqs, token_rows)
    ]
    hidden, taps = model.forward_with_hidden_taps(data, target_layer_ids=(1, 0))
    begin = 0
    for seq, tokens in zip(seqs, token_rows):
        reference_data, reference_manager, reference_seq = make_input(tokens)
        expected_hidden, expected_taps = model.forward_with_hidden_taps(
            reference_data, target_layer_ids=(1, 0)
        )
        length = len(tokens)
        torch.testing.assert_close(
            hidden[begin:begin + length], expected_hidden, rtol=2e-2, atol=2e-2
        )
        torch.testing.assert_close(
            taps[:, begin:begin + length], expected_taps, rtol=1e-2, atol=2e-3
        )
        for actual_bank, expected_bank in zip(
            manager.segment.k_cache + manager.segment.v_cache,
            reference_manager.segment.k_cache + reference_manager.segment.v_cache,
        ):
            actual = actual_bank[seq.page_table].reshape(-1, *actual_bank.shape[-2:])[:length]
            expected = expected_bank[reference_seq.page_table].reshape(
                -1, *expected_bank.shape[-2:]
            )[:length]
            torch.testing.assert_close(actual, expected, rtol=1e-2, atol=2e-3)
        for name in ("conv_state", "temporal_state"):
            torch.testing.assert_close(
                getattr(manager.ssm_segment, name)[:, seq.recurrent_state_slot],
                getattr(reference_manager.ssm_segment, name)[:, reference_seq.recurrent_state_slot],
                rtol=1e-2, atol=2e-3,
            )
        reference_manager.free(reference_seq)
        begin += length
    assert torch.isfinite(hidden).all() and torch.isfinite(taps).all()
    for seq in seqs:
        manager.free(seq)


@pytest.mark.parametrize("decodes,width", [(1, 8), (4, 8), (0, 1)])
def test_dspark_backend_reserves_the_shared_verify_window(decodes, width):
    """DSpark drives the MTP pipeline: decode rows reserve 1 + k like MTP."""
    from types import SimpleNamespace

    from gllm.speculative.backend import select_speculative_backend

    runner = SimpleNamespace(
        speculative_num_draft_tokens=7, model=SimpleNamespace(mtp=None),
        _dspark_drafter=object(), mtp_max_batch=0, max_num_batched_tokens=2048,
    )
    backend = select_speculative_backend(dspark_enabled=True, mtp_enabled=True)
    assert backend.name == "dspark"
    assert backend.capabilities.cuda_graph and backend.capabilities.overlap
    assert backend.reservation_width(runner, decodes) == width


# --- fixed runtime proposal count (checkpoint tensors stay unchanged) -------


@pytest.mark.parametrize("width", [0, -1, 8, True, False, 1.5, "3"])
def test_runtime_width_rejected_before_module_allocation(monkeypatch, width):
    from dataclasses import replace
    from gllm.models import qwen38_dspark as module

    def forbidden(*args, **kwargs):
        pytest.fail("invalid width reached a CUDA module constructor")

    monkeypatch.setattr(module, "ReplicatedLinear", forbidden)
    with pytest.raises(ValueError, match="speculative_num_draft_tokens"):
        module.Qwen38DSpark(
            replace(_tensor_config(), block_size=7),
            embed=torch.nn.Identity(), speculative_num_draft_tokens=width,
        )


@pytest.mark.parametrize("shape", [(1, 2, 2), (2, 3, 2), (1, 3, 4), (3, 2)])
def test_runtime_width_head_rejects_bad_shape_before_projection(monkeypatch, shape):
    from gllm.models.qwen38_dspark import Qwen38DSpark

    model = _head_stub(3)

    def forbidden(*args, **kwargs):
        pytest.fail("invalid shape reached the vocabulary projection")

    monkeypatch.setattr(model._lm_head[0], "forward", forbidden)
    with pytest.raises(ValueError, match="draft hidden"):
        Qwen38DSpark.forward_head(model, torch.zeros(shape), torch.tensor([0]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("width", [1, 3, 7])
def test_runtime_width_loads_identical_checkpoint_without_changing_config(width):
    from dataclasses import replace

    config = replace(_tensor_config(), block_size=7)
    reference = _build_model(config)
    weights = _build_synthetic_checkpoint(reference, config)
    reference.load_weights(weights)
    model = _build_model(config, speculative_num_draft_tokens=width)
    model.load_weights(weights)
    assert model.config is config and config.block_size == 7
    assert model.block_size == model.noise_width == width
    assert set(model.state_dict()) == set(reference.state_dict())
    _assert_loaded_checkpoint(model, weights)
    for name, value in reference.state_dict().items():
        torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
    # Width is fixed for this instance, never a per-request mutable field.
    for name in ("block_size", "noise_width"):
        with pytest.raises(AttributeError):
            setattr(model, name, 2)

    if width == 7:
        # Explicit default and omitted default must produce identical outputs,
        # not merely matching shapes or an unchanged state_dict.
        with torch.no_grad():
            model._embed[0].weight.copy_(reference._embed[0].weight)
            model._lm_head[0].weight.copy_(reference._lm_head[0].weight)
        prompt = _rows(config, 4)
        suffix = _rows(config, 1)
        anchor = torch.tensor([5], device="cuda")
        results = []
        for draft in (reference, model):
            cache = draft.prefill(prompt, anchor)
            hidden, noise = draft.forward_draft(suffix, anchor, start_pos=5, caches=cache)
            results.append((hidden, noise, *draft.forward_head(hidden, anchor)))
        for before, after in zip(*results):
            torch.testing.assert_close(before, after, rtol=0, atol=0)


@pytest.mark.parametrize("checkpoint_width", [0, -1, True, 2.5])
def test_runtime_width_rejects_invalid_checkpoint_before_allocation(monkeypatch, checkpoint_width):
    from dataclasses import replace
    from gllm.models import qwen38_dspark as module

    def forbidden(*args, **kwargs):
        pytest.fail("invalid checkpoint width reached a CUDA module constructor")

    monkeypatch.setattr(module, "ReplicatedLinear", forbidden)
    with pytest.raises(ValueError, match="checkpoint block_size"):
        module.Qwen38DSpark(
            replace(_tensor_config(), block_size=checkpoint_width),
            embed=torch.nn.Identity(), speculative_num_draft_tokens=1,
        )


@pytest.mark.parametrize("count", [0, -1, 8, True, False, 2.5])
def test_runtime_width_loader_rejects_invalid_count_before_allocation(tmp_path, count):
    from dataclasses import asdict, replace
    import json
    from gllm.runtime.model_loader import load_dspark_draft

    config = replace(_tensor_config(), block_size=7)
    (tmp_path / "config.json").write_text(json.dumps(asdict(config)), encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint proposal range"):
        load_dspark_draft(str(tmp_path), _dspark_loader_target(config), 32, speculative_num_draft_tokens=count)


@pytest.mark.parametrize("count", [1, 3, 7])
def test_runtime_width_loader_requires_nonempty_context_before_verify(tmp_path, count):
    from dataclasses import asdict, replace
    import json
    from gllm.runtime.model_loader import load_dspark_draft

    config = replace(_tensor_config(), block_size=7)
    (tmp_path / "config.json").write_text(json.dumps(asdict(config)), encoding="utf-8")
    with pytest.raises(ValueError, match="context windows"):
        load_dspark_draft(str(tmp_path), _dspark_loader_target(config),
                          count + 1, speculative_num_draft_tokens=count)


