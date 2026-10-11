"""Flash-Next numerical and checkpoint contracts, using small real tensors."""

import copy
import json
import os

import pytest
import torch
import torch.nn.functional as F
from qwen4_exp_checkpoint_utils import (
    audit_bf16_checkpoint,
    text_tensor_shapes,
)
from safetensors.torch import save_file

from gllm.layers.attention.qsa import (
    compress_index_keys,
    select_qsa_tokens,
    sparse_gqa_attention,
)
from gllm.layers.hyper_connection import GatedResidual, GroupedGemmaRMSNorm
from gllm.layers.position_learning import (
    NGramHash,
    PLELayer,
    dilated_causal_conv,
    ngram_multipliers,
    ngram_vocab_layout,
)
from gllm.models.qwen4_exp import parse_text_config, validate_tp_config
from gllm.models.weight_utils import ShardedEmbeddingLookup


@pytest.fixture
def raw_config():
    return {
        "model_type": "qwen4_exp",
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "text_config": {
            "model_type": "qwen4_exp_text",
            "hidden_size": 8,
            "num_hidden_layers": 4,
            "layer_types": ["linear_attention"] * 3 + ["full_attention"],
            "hc_count": 4,
            "hc_lowrank": 3,
            "rms_norm_eps": 1e-6,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "indexer_n_heads": 2,
            "indexer_kv_heads": 1,
            "indexer_head_dim": 4,
            "indexer_budget": 4,
            "indexer_compress_ratio": 2,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_key_head_dim": 4,
            "linear_value_head_dim": 4,
            "linear_conv_kernel_dim": 4,
            "num_experts": 2,
            "num_experts_per_tok": 1,
            "moe_intermediate_size": 8,
            "shared_expert_intermediate_size": 8,
            "vocab_size": 32,
            "eos_token_id": 31,
            "ple_layer_ids": [2],
            "ple_embed_dim": 8,
            "ple_conv_kernel_size": 4,
            "ngram_size": 3,
            "heads_per_ngram": 2,
            "ngram_vocab_size_base": 11,
            "split_ngram_parts": 3,
            "make_ngram_vocab_size_divisible_by": 8,
        },
    }


@pytest.fixture
def config(raw_config):
    return parse_text_config(raw_config)


@pytest.mark.parametrize(
    "alias", ["full_attention", "qwen_sparse_attention", "indexed_attention"]
)
def test_config_understands_published_and_new_attention_names(raw_config, alias):
    raw_config["text_config"]["layer_types"][-1] = alias
    config = parse_text_config(raw_config)
    assert config.qsa_layer_ids == (3,)
    assert config.gdn_layer_ids == (0, 1, 2)
    assert config.ple_layer_indices == (1,)
    assert config.hyper_hidden_size == 32
    assert config.ple_history_length == 9
    validate_tp_config(config, 2)
    assert raw_config["text_config"]["layer_types"][-1] == alias


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("ple_layer_ids", [0], "one-based"),
        ("ple_layer_ids", [2, 2], "unique"),
        ("layer_types", ["linear_attention"], "one entry"),
        ("layer_types", ["linear_attention"] * 3 + ["unknown"], "Unknown"),
        ("indexer_budget", 3, "compress ratio"),
        ("ple_embed_dim", 7, "ngram heads"),
        ("num_experts_per_tok", 3, "exceeds"),
    ],
)
def test_config_rejects_inconsistent_geometry(raw_config, field, value, match):
    raw_config["text_config"][field] = value
    with pytest.raises(ValueError, match=match):
        parse_text_config(raw_config)


def test_tp_rejects_unsupported_partition(config):
    with pytest.raises(ValueError, match="num_attention_heads"):
        validate_tp_config(config, 3)


@pytest.mark.parametrize("device", ["cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_gr_matches_explicit_equations_and_preserves_residual(device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(73)
    module = GatedResidual(8, 4, 3, device=device, dtype=dtype)
    with torch.no_grad():
        module.hc_norm.weight.normal_(0, 0.1)
    x = torch.randn(7, 32, device=device, dtype=dtype)
    original = x.clone()
    branches = x.float().reshape(7, 4, 8)
    normalized = branches * (branches.square().mean(-1, keepdim=True) + 1e-6).rsqrt()
    normalized = (normalized.flatten(1) * (1 + module.hc_norm.weight.float())).to(dtype)
    low = F.silu(F.linear(normalized, module.input_mix_weight_down.weight) / 4)
    gates = F.linear(low, module.input_mix_weight_up.weight).sigmoid().reshape(7, 4, 8)
    expected_mix = (gates * normalized.reshape(7, 4, 8)).mean(1)
    mixed, residual = module.mix(x)
    torch.testing.assert_close(
        mixed,
        expected_mix,
        rtol=2e-6 if dtype == torch.float32 else 0,
        atol=2e-6 if dtype == torch.float32 else 0,
    )
    block = torch.randn_like(mixed)
    injection = (
        2 * (F.linear(normalized, module.block_inject_weight.weight) / 4).sigmoid()
    )
    expected = x.reshape(7, 4, 8) + injection[:, :, None] * block[:, None, :]
    torch.testing.assert_close(
        module.combine(block, residual),
        expected.flatten(1),
        rtol=2e-6 if dtype == torch.float32 else 0,
        atol=2e-6 if dtype == torch.float32 else 0,
    )
    torch.testing.assert_close(x, original, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_gr_zero_gates_mean_half_normalized_stream_and_unit_injection():
    module = GatedResidual(2, 4, 1, device="cuda")
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.zero_()
    x = torch.arange(8, device="cuda", dtype=torch.float32).reshape(1, 8)
    mixed, residual = module.mix(x)
    branch = x.reshape(1, 4, 2)
    normed = branch * (branch.square().mean(-1, keepdim=True) + 1e-6).rsqrt()
    torch.testing.assert_close(mixed, normed.mean(1) / 2)
    torch.testing.assert_close(
        module.combine(torch.ones(1, 2, device="cuda"), residual), x + 1
    )
    assert module.hc_norm.weight.shape == (8,)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_final_gr_mixer_has_no_injection_parameter():
    module = GatedResidual(8, 4, 3, use_combine=False, device="cuda")
    assert not any("block_inject" in name for name, _ in module.named_parameters())
    x, residual = module.mix(torch.randn(2, 32, device="cuda"))
    with pytest.raises(RuntimeError, match="no injection"):
        module.combine(x, residual)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_grouped_norm_does_not_normalize_across_branches():
    norm = GroupedGemmaRMSNorm(2, 2, device="cuda")
    x = torch.tensor([[1.0, 1.0, 100.0, 100.0]], device="cuda")
    torch.testing.assert_close(norm(x), torch.ones_like(x), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_qsa_block_causality_and_always_visible_tail(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    query = torch.ones(8, 1, 1, device=device)
    # The highest-scoring future block must not leak into earlier rows.
    compressed = torch.tensor([[1.0], [5.0], [3.0], [1000.0]], device=device)
    indices = select_qsa_tokens(
        query, compressed, torch.arange(8, device=device), budget=2, ratio=2
    )
    assert indices.cpu().tolist() == [
        [0, -1, -1],
        [0, 1, -1],
        [0, 1, 2],
        [2, 3, -1],
        [2, 3, 4],
        [2, 3, -1],
        [2, 3, 6],
        [6, 7, -1],
    ]


def test_qsa_scores_relu_each_index_head_before_summing():
    query = torch.tensor([[[1.0], [-2.0]]])
    compressed = torch.tensor([[1.0], [-1.0]])
    indices = select_qsa_tokens(query, compressed, torch.tensor([3]), budget=2, ratio=2)
    assert indices[0, :2].tolist() == [2, 3]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_qsa_under_budget_matches_causal_sdpa(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(79)
    q = torch.randn(7, 4, 8, device=device)
    k = torch.randn(7, 2, 8, device=device)
    v = torch.randn_like(k)
    index_q = torch.randn(7, 2, 4, device=device)
    raw = torch.randn(7, 4, device=device)
    ids = select_qsa_tokens(
        index_q,
        compress_index_keys(raw, 2),
        torch.arange(7, device=device),
        budget=8,
        ratio=2,
    )
    actual = sparse_gqa_attention(q, k, v, ids)
    expected = (
        F.scaled_dot_product_attention(
            q.transpose(0, 1).unsqueeze(0),
            k.repeat_interleave(2, 1).transpose(0, 1).unsqueeze(0),
            v.repeat_interleave(2, 1).transpose(0, 1).unsqueeze(0),
            is_causal=True,
        )
        .squeeze(0)
        .transpose(0, 1)
    )
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


def test_qsa_ignores_unselected_kv_and_handles_empty_rows():
    q = torch.ones(2, 2, 1)
    keys = torch.tensor([[[1.0]], [[1000.0]], [[1.0]]])
    values = torch.tensor([[[2.0]], [[1000.0]], [[4.0]]])
    out = sparse_gqa_attention(q, keys, values, torch.tensor([[0, 2], [-1, -1]]))
    torch.testing.assert_close(out[0], torch.full((2, 1), 3.0))
    assert not out[1].any()


def test_qsa_rejects_incomplete_cache():
    with pytest.raises(ValueError, match="missing a completed"):
        select_qsa_tokens(
            torch.ones(1, 1, 2), torch.ones(1, 2), torch.tensor([7]), budget=4, ratio=2
        )


def test_ngram_prime_layout_and_hash_seed_are_stable():
    assert ngram_vocab_layout(11, 4) == ([11, 13, 17, 19], [0, 11, 24, 41], 60)
    assert ngram_vocab_layout(11, 4, 1) == ([23, 29, 31, 37], [0, 23, 52, 83], 120)
    values = ngram_multipliers(3, 248320)
    assert len(values) == 3 and all(v > 0 and v % 2 for v in values)
    assert values != ngram_multipliers(3, 248320, 1)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_ngram_chunks_and_packed_requests_equal_full_sequence(config, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    hasher = NGramHash(config, device=device)
    tokens = torch.tensor([2, 4, 6, 31, 8, 10, 12], device=device)
    initial = torch.full((1, 2), 31, device=device)
    full = hasher(tokens, torch.tensor([0, 7]), initial)
    history = initial.clone()
    results = []
    for start, end in [(0, 1), (1, 3), (3, 5), (5, 7)]:
        chunk = tokens[start:end]
        results.append(hasher(chunk, torch.tensor([0, end - start]), history))
        history = torch.cat((history[0], chunk))[-2:].unsqueeze(0)
    torch.testing.assert_close(torch.cat(results), full, rtol=0, atol=0)
    packed = hasher(
        torch.cat((tokens, tokens)), torch.tensor([0, 7, 7, 14]), initial.expand(3, -1)
    )
    torch.testing.assert_close(packed, torch.cat((full, full)), rtol=0, atol=0)
    # Changing a token across EOS must not affect hashes after EOS.
    changed = tokens.clone()
    changed[2] = 20
    other = hasher(changed, torch.tensor([0, 7]), initial)
    torch.testing.assert_close(full[4:], other[4:], rtol=0, atol=0)


def test_ngram_matches_integer_xor_formula_and_signed_remainder(config):
    hasher = NGramHash(config)
    # Exercise signed int64 wraparound independently with Python integers.
    hasher.layer_multipliers.copy_(torch.tensor([2**62 + 1, 2**61 + 3, 2**60 + 7]))
    tokens = torch.tensor([2, 3, 5])
    actual = hasher(tokens, torch.tensor([0, 3]), torch.tensor([[31, 31]]))
    expected = []
    for pos in range(3):
        row = []
        for order in (2, 3):
            mixed, crossed = 0, False
            for shift in range(order):
                token = int(tokens[pos - shift]) if pos >= shift else 31
                if shift and crossed:
                    token = 31
                if shift and token == 31:
                    crossed = True
                mixed ^= (token * int(hasher.layer_multipliers[shift])) & (
                    (1 << 64) - 1
                )
            if mixed >= 1 << 63:
                mixed -= 1 << 64
            for head in range((order - 2) * 2, (order - 1) * 2):
                row.append(
                    mixed % int(hasher.ngram_heads_vocab_sizes[head])
                    + int(hasher.ngram_heads_offsets[head])
                )
        expected.append(row)
    assert actual.tolist() == expected


@pytest.mark.parametrize("device", ["cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_ple_chunk_and_decode_state_match_full_prefill(config, device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(83)
    ple = PLELayer(config, device=device, dtype=dtype)
    streams = torch.randn(17, config.hyper_hidden_size, device=device, dtype=dtype)
    embeddings = torch.randn(17, config.ple_embed_dim, device=device, dtype=dtype)
    full, full_state = ple(streams, embeddings)
    chunks, state = [], None
    for start, end in [(0, 2), (2, 7), (7, 8), (8, 12), (12, 17)]:
        out, state = ple(streams[start:end], embeddings[start:end], state)
        chunks.append(out)
    tolerance = (
        {"rtol": 0.025, "atol": 0.025}
        if dtype == torch.bfloat16
        else {"rtol": 2e-5, "atol": 2e-6}
    )
    torch.testing.assert_close(torch.cat(chunks), full, **tolerance)
    torch.testing.assert_close(state, full_state, **tolerance)
    assert state.shape == (9, 32)
    # Reusing the module for another request must start with fresh history.
    again, _ = ple(streams, embeddings)
    torch.testing.assert_close(again, full, rtol=0, atol=0)


def test_ple_convolution_is_dilated_and_retains_only_its_history():
    x = torch.arange(1, 11, dtype=torch.float32).unsqueeze(1)
    weights = torch.tensor([[[1.0, 10.0, 100.0]]])
    actual, state = dilated_causal_conv(x, weights, dilation=3)
    expected = [
        100 * float(x[t])
        + (10 * float(x[t - 3]) if t >= 3 else 0)
        + (float(x[t - 6]) if t >= 6 else 0)
        for t in range(10)
    ]
    torch.testing.assert_close(actual[:, 0], torch.tensor(expected))
    assert state.shape == (6, 1)
    assert state.untyped_storage().nbytes() == state.numel() * state.element_size()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_ple_adds_rounded_branch_before_outer_bf16_residual(config):
    # BF16 addition is not associative. A zero dot gives gate=0.5; the
    # independently computed branch must round before residual addition.
    torch.manual_seed(89)
    ple = PLELayer(config, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        ple.key_proj.weight.zero_()
    streams = torch.randn(
        11, config.hyper_hidden_size, device="cuda", dtype=torch.bfloat16
    )
    embeddings = torch.randn(
        11, config.ple_embed_dim, device="cuda", dtype=torch.bfloat16
    )
    value = ple.value_proj(embeddings)
    gated = (value[:, None, :].expand(-1, 4, -1) * 0.5).flatten(1)
    normalized = ple.norm_conv(gated)
    conv = (
        F.conv1d(
            F.pad(normalized.T.unsqueeze(0), (9, 0)),
            ple.conv1d.weight,
            groups=config.hyper_hidden_size,
            dilation=3,
        )
        .squeeze(0)
        .T
    )
    expected = streams + (gated + F.silu(conv))
    output, _ = ple(streams, embeddings)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert not torch.equal(expected, (streams + gated) + F.silu(conv))


@pytest.fixture
def checkpoint(tmp_path, raw_config, config):
    shapes = text_tensor_shapes(config)
    tensors = {}
    for key, shape in shapes.items():
        dtype = (
            torch.int64
            if key.endswith(
                ("layer_multipliers", "ngram_heads_offsets", "ngram_heads_vocab_sizes")
            )
            else torch.bfloat16
        )
        tensors[key] = torch.zeros(shape, dtype=dtype)
    prefix = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
    offset = 0
    for shard in range(config.split_ngram_parts):
        key = f"{prefix}.shard_{shard}.weight"
        rows, width = tensors[key].shape
        tensors[key] = (
            torch.arange(offset, offset + rows * width)
            .reshape(rows, width)
            .to(torch.bfloat16)
        )
        offset += rows * width
    save_file(tensors, str(tmp_path / "weights.safetensors"))
    (tmp_path / "config.json").write_text(json.dumps(raw_config))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: "weights.safetensors" for key in tensors}})
    )
    return tmp_path


def test_checkpoint_schema_covers_backbone_and_one_based_ple(checkpoint):
    result = audit_bf16_checkpoint(checkpoint)
    assert result["gdn_layers"] == 3 and result["qsa_layers"] == 1
    assert result["ple_layer_indices"] == (1,)


def test_checkpoint_audit_detects_truncation(checkpoint):
    file = checkpoint / "weights.safetensors"
    file.write_bytes(file.read_bytes()[:-1])
    with pytest.raises(ValueError, match="payload"):
        audit_bf16_checkpoint(checkpoint)


def test_checkpoint_audit_detects_missing_and_unmapped_keys(checkpoint):
    path = checkpoint / "model.safetensors.index.json"
    index = json.loads(path.read_text())
    original = copy.deepcopy(index)
    del index["weight_map"]["model.language_model.layers.1.ple.key_proj.weight"]
    path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="Missing"):
        audit_bf16_checkpoint(checkpoint)
    original["weight_map"][
        "model.language_model.unknown.weight"
    ] = "weights.safetensors"
    path.write_text(json.dumps(original))
    with pytest.raises(ValueError, match="Unmapped"):
        audit_bf16_checkpoint(checkpoint)


def test_ple_row_lookup_crosses_shards_and_preserves_repeated_ids(checkpoint, config):
    table = ShardedEmbeddingLookup(
        checkpoint,
        [
            f"model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{i}.weight"
            for i in range(config.split_ngram_parts)
        ],
    )
    ids = torch.tensor([[0, table._ends[0], table.num_rows - 1], [2, 0, 2]])
    expected = (
        ids.unsqueeze(-1) * config.ple_head_dim + torch.arange(config.ple_head_dim)
    ).to(torch.bfloat16)
    torch.testing.assert_close(table(ids), expected, rtol=0, atol=0)
    assert table(torch.empty(0, 4, dtype=torch.long)).shape == (0, 4, 2)
    with pytest.raises(ValueError, match="outside"):
        table(torch.tensor([table.num_rows]))
    table.close()


@pytest.mark.skipif(
    not os.environ.get("GLLM_FLASH_NEXT_CHECKPOINT"),
    reason="Set GLLM_FLASH_NEXT_CHECKPOINT for full local header audit",
)
def test_local_flash_next_checkpoint():
    result = audit_bf16_checkpoint(os.environ["GLLM_FLASH_NEXT_CHECKPOINT"])
    assert result["gdn_layers"] == 36
    assert result["qsa_layers"] == 12
    assert result["ple_layer_indices"] == (1,)


@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("rank", [0, 1, 2])
def test_parallel_table_partition(tmp_path, monkeypatch, fp8, rank):
    import gllm.distributed.parallel_state as parallel

    monkeypatch.setattr(parallel, "get_tp_size", lambda: 3)
    monkeypatch.setattr(parallel, "get_tp_rank", lambda: rank)
    monkeypatch.setattr(parallel, "tensor_model_parallel_all_reduce", lambda x: x)
    dtype = torch.float8_e4m3fn if fp8 else torch.bfloat16
    table = torch.arange(28, dtype=torch.float32).reshape(7, 4).to(dtype)
    keys = [f"table.shard_{i}.weight" for i in range(2)]
    tensors = {keys[0]: table[:4], keys[1]: table[4:]}
    if fp8:
        tensors["table.weight_scale"] = torch.tensor([0.25])
    save_file(tensors, str(tmp_path / "parts.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(tensors, "parts.safetensors")})
    )
    source = ShardedEmbeddingLookup(tmp_path, keys)
    resident = source.load_parallel(device="cpu")
    ids = torch.tensor([[6, 0, 3], [4, 2, 6]])
    expected = source(ids)
    valid = (ids >= rank * 3) & (ids < min(rank * 3 + 3, 7))
    torch.testing.assert_close(
        resident(ids), expected * valid.unsqueeze(-1), rtol=0, atol=0
    )
    assert resident.weight.dtype == dtype
    assert resident.weight.shape == (3, 4)
    source.close()
    torch.testing.assert_close(
        resident(ids), expected * valid.unsqueeze(-1), rtol=0, atol=0
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("fp8", [False, True])
def test_lookup_graph_replays_changed_ids(fp8):
    from gllm.models.weight_utils import ParallelEmbeddingLookup

    dtype = torch.float8_e4m3fn if fp8 else torch.bfloat16
    table = torch.arange(32, device="cuda", dtype=torch.float32).reshape(8, 4).to(dtype)
    lookup = ParallelEmbeddingLookup(
        table, 0, 8, 1, torch.tensor([0.5], device="cuda") if fp8 else None
    )
    ids = torch.tensor([0, 1, 2], device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        lookup(ids)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = lookup(ids)
    ids.copy_(torch.tensor([7, 2, 5], device="cuda"))
    graph.replay()
    torch.testing.assert_close(result, lookup(ids), rtol=0, atol=0)
