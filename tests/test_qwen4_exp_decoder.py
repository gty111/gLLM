"""Tiny native checkpoint exercised through the actual decoder and arena."""

import json
from copy import deepcopy

import pytest
import torch
from qwen4_exp_checkpoint_utils import text_tensor_shapes
from safetensors.torch import save_file
from transformers import AutoConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpVisionModel

from gllm.layers.position_learning import NGramHash
from gllm.models.qwen4_exp import (
    Qwen4ExpForCausalLM,
    Qwen4ExpForConditionalGeneration,
    parse_text_config,
)
from gllm.runtime.cache_arena import CacheArena
from gllm.runtime.input_data import InputData
from gllm.runtime.memory_manager import MemoryManager, Segment, SSMSegment
from gllm.runtime.model_loader import LazySafetensors
from gllm.runtime.sequence import GenerationSequence


def tiny_config():
    return {
        "model_type": "qwen4_exp",
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "tie_word_embeddings": False,
        "text_config": {
            "model_type": "qwen4_exp_text",
            "dtype": "bfloat16",
            "hidden_size": 32,
            "hidden_act": "silu",
            "output_gate_type": "sigmoid",
            "num_hidden_layers": 4,
            "layer_types": ["linear_attention"] * 3 + ["full_attention"],
            "hc_count": 4,
            "hc_lowrank": 16,
            "rms_norm_eps": 1e-6,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 64,
            "indexer_n_heads": 2,
            "indexer_kv_heads": 1,
            "indexer_head_dim": 128,
            "indexer_budget": 64,
            "indexer_compress_ratio": 2,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_key_head_dim": 16,
            "linear_value_head_dim": 16,
            "linear_conv_kernel_dim": 4,
            "mamba_ssm_dtype": "float32",
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 64,
            "shared_expert_intermediate_size": 64,
            "vocab_size": 64,
            "eos_token_id": 31,
            "max_position_embeddings": 128,
            "ple_layer_ids": [2],
            "ple_embed_dim": 32,
            "ple_conv_kernel_size": 4,
            "ngram_size": 3,
            "heads_per_ngram": 2,
            "ngram_vocab_size_base": 11,
            "split_ngram_parts": 3,
            "make_ngram_vocab_size_divisible_by": 8,
            "partial_rotary_factor": 0.5,
            "rope_theta": 10000.0,
            "attention_bias": False,
            "tie_word_embeddings": False,
            "norm_topk_prob": True,
        },
    }


def write_checkpoint(path, raw):
    parsed = parse_text_config(raw)
    torch.manual_seed(107)
    weights = {}
    for key, shape in text_tensor_shapes(parsed).items():
        if key.endswith(
            ("layer_multipliers", "ngram_heads_offsets", "ngram_heads_vocab_sizes")
        ):
            continue
        tensor = torch.randn(shape, dtype=torch.float32) * 0.08
        if key.endswith("linear_attn.norm.weight"):
            tensor.fill_(1.0)
        if key.endswith("A_log"):
            tensor.zero_()
        if "norm" in key and not key.endswith("linear_attn.norm.weight"):
            tensor.zero_()
        weights[key] = tensor.to(torch.bfloat16)
    hasher = NGramHash(parsed)
    for key, buffer in hasher.named_buffers():
        weights[f"model.language_model.layers.1.ple.ple_embedding.{key}"] = buffer
    save_file(weights, str(path / "weights.safetensors"))
    (path / "config.json").write_text(json.dumps(raw))
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: "weights.safetensors" for key in weights}})
    )
    return weights


@pytest.fixture
def tiny_model(tmp_path, request):
    if not torch.cuda.is_available():
        pytest.skip("Decoder requires CUDA")
    raw = tiny_config()
    raw["text_config"].update(getattr(request, "param", {}))
    weights = write_checkpoint(tmp_path, raw)
    config = AutoConfig.from_pretrained(str(tmp_path))
    config.checkpoint_path = str(tmp_path)
    config.mtp_enabled = False
    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        model = Qwen4ExpForCausalLM(config)
        index = {key: (str(tmp_path / "weights.safetensors"), key) for key in weights}
        lazy = LazySafetensors(index)
        with torch.no_grad():
            model.load_weights(lazy)
        lazy.close()
    finally:
        torch.set_default_dtype(original_dtype)
    yield model.eval(), weights
    for layer in model.model.layers:
        if layer.ple_lookup is not None:
            layer.ple_lookup.close()


def make_inputs(model, mtp_k=0):
    manager = MemoryManager.__new__(MemoryManager)
    manager.num_layers = model.num_kv_layers
    manager.dtype = torch.bfloat16
    manager.page_size = 16
    manager.kv_head_num = model.num_kv_heads
    manager.kv_head_dim = model.head_dim
    manager.index_head_dim = model.index_head_dim
    manager.use_mla = False
    manager.dsv4_kv_cache_config = manager.dsv4_state_cache_config = None
    manager.ssm_cache_config = model.ssm_cache_config
    manager.mtp_k = mtp_k
    manager.requires_mrope_buffer = model.requires_mrope_buffer
    kv_layout = manager._kv_cache_layout()
    arena = CacheArena(
        torch.empty(kv_layout.entry_bytes * 512, dtype=torch.uint8, device="cuda"),
        physical_page_bytes=kv_layout.entry_bytes,
    )
    kv = arena.register_cache(kv_layout)
    manager.segment = Segment(
        model.num_kv_layers,
        16,
        model.num_kv_heads,
        model.head_dim,
        False,
        kv,
        index_head_dim=model.index_head_dim,
    )
    manager.ssm_segment = SSMSegment(
        model.ssm_cache_config,
        state_cache=arena.register_cache(manager._ssm_cache_layout("ssm_state")),
        snapshot_cache=arena.register_cache(manager._ssm_cache_layout("ssm_snapshot")),
    )
    return manager, InputData(True, manager, max_seq_length=128, max_running_seqs=4)


def make_request(manager, seq_id, tokens):
    seq = GenerationSequence(seq_id, list(tokens), [], output_len=8)
    seq.page_table = [manager.segment.allocate() for _ in range(4)]
    seq.ssm_block_table = manager.ssm_segment.allocate_block_table(1)
    seq.recurrent_state_slot = seq.ssm_block_table[0]
    return seq


@pytest.mark.parametrize(
    "tiny_model", [{"indexer_budget": 8}, {"indexer_budget": 64}], indirect=True
)
@torch.inference_mode()
def test_decode_graph_replays_changing_tokens_and_positions(tiny_model):
    model, _ = tiny_model
    manager, data = make_inputs(model)
    seq = make_request(manager, 0, list(range(1, 25)))
    run_chunk(model, data, seq, 8)
    seq.to_compute_token_num = 1
    data.cal_and_set_input([seq])
    state = manager.ssm_segment
    tensors = [state.conv_state, state.temporal_state, *state.auxiliary_states.values()]
    baseline = [tensor.clone() for tensor in tensors]

    def restore():
        for tensor, saved in zip(tensors, baseline):
            tensor.copy_(saved)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        model(data)
    torch.cuda.current_stream().wait_stream(stream)
    restore()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = model(data)
    restore()
    for end in (9, 10, 11):
        seq.to_compute_token_num = 1
        data.cal_and_set_input([seq])
        baseline = [tensor.clone() for tensor in tensors]
        expected = model(data).clone()
        expected_states = [tensor.clone() for tensor in tensors]
        restore()
        graph.replay()
        torch.testing.assert_close(actual, expected, atol=0.01, rtol=0.04)
        assert torch.equal(
            model.logits_from_hidden(actual).argmax(-1),
            model.logits_from_hidden(expected).argmax(-1),
        )
        for index, (tensor, saved) in enumerate(zip(tensors, expected_states)):
            slot = seq.recurrent_state_slot
            actual_state = tensor[:, slot] if index < 2 else tensor[slot]
            expected_state = saved[:, slot] if index < 2 else saved[slot]
            torch.testing.assert_close(
                actual_state, expected_state, atol=0.01, rtol=0.04
            )
        seq.computed_token_num = end


@torch.inference_mode()
def run_chunk(model, data, seq, end):
    seq.to_compute_token_num = end - seq.computed_token_num
    data.cal_and_set_input([seq])
    hidden = model(data)
    logits = model.logits_from_hidden(hidden).float()
    seq.computed_token_num = end
    return logits


def test_native_config_preserves_nested_fields(tmp_path):
    raw = tiny_config()
    (tmp_path / "config.json").write_text(json.dumps(raw))
    config = AutoConfig.from_pretrained(str(tmp_path))
    assert config.text_config.hc_count == 4
    assert config.text_config.indexer_head_dim == 128
    assert config.text_config.dtype == torch.bfloat16


def test_auxiliary_state_snapshot_restore_commit_and_free(tiny_model):
    model, _ = tiny_model
    manager, _ = make_inputs(model)
    segment = manager.ssm_segment
    first, second = segment.allocate_block_table(2)
    snapshot = segment.allocate_snapshot()
    state = segment.auxiliary_states["ple_history"]
    state[first].fill_(3.0)
    segment.copy_state("working", first, "snapshot", snapshot)
    state[first].zero_()
    segment.copy_state("snapshot", snapshot, "working", second)
    torch.testing.assert_close(state[second], torch.full_like(state[second], 3.0))
    segment.commit_blocks([(first, second)])
    torch.testing.assert_close(state[first], state[second])
    segment.free_block_table([first, second])
    assert not state[first].any() and not state[second].any()
    segment.free_snapshot(snapshot)
    assert not state[snapshot].any()


@pytest.mark.parametrize("tiny_model", [{}, {"ple_layer_ids": []}], indirect=True)
def test_decoder_full_chunked_prefill_and_decode_logits_match(tiny_model):
    model, _ = tiny_model
    tokens = [2, 5, 7, 11, 13, 31, 17, 19, 23, 29, 3, 9, 12, 15, 21, 22, 24]
    manager, data = make_inputs(model)
    full_seq = make_request(manager, 1, tokens)
    full = run_chunk(model, data, full_seq, len(tokens))
    chunk_seq = make_request(manager, 2, tokens)
    chunks = [run_chunk(model, data, chunk_seq, end) for end in (3, 8, 16, 17)]
    actual = torch.cat(chunks)
    assert torch.isfinite(actual).all() and full.abs().max() > 0.1
    torch.testing.assert_close(actual, full, rtol=0.04, atol=0.035)
    assert torch.equal(actual.argmax(-1), full.argmax(-1))


def test_prefix_state_restore_resumes_decoder_at_chunk_boundary(tiny_model):
    model, _ = tiny_model
    tokens = list(range(1, 24))
    manager, data = make_inputs(model)
    original = make_request(manager, 1, tokens)
    run_chunk(model, data, original, 16)
    snapshot = manager.ssm_segment.allocate_snapshot()
    manager.ssm_segment.copy_state(
        "working", original.recurrent_state_slot, "snapshot", snapshot
    )
    clone = make_request(manager, 2, tokens)
    clone.page_table[0] = original.page_table[0]
    clone.computed_token_num = 16
    manager.ssm_segment.copy_state(
        "snapshot", snapshot, "working", clone.recurrent_state_slot
    )
    first = run_chunk(model, data, original, len(tokens))
    second = run_chunk(model, data, clone, len(tokens))
    torch.testing.assert_close(first, second, rtol=0, atol=0)


@torch.inference_mode()
def test_decoder_packed_requests_keep_independent_history(tiny_model):
    model, _ = tiny_model
    requests = [list(range(1, 18)), [7, 9, 31, 11, 5, 2, 17, 23, 1]]
    manager, data = make_inputs(model)
    expected = []
    for index, tokens in enumerate(requests):
        seq = make_request(manager, index + 1, tokens)
        expected.append(run_chunk(model, data, seq, len(tokens)))
        manager.ssm_segment.free_block_table(seq.ssm_block_table)
    seqs = [
        make_request(manager, index + 3, tokens)
        for index, tokens in enumerate(requests)
    ]
    for seq, end in zip(seqs, (3, 8)):
        run_chunk(model, data, seq, end)
        seq.to_compute_token_num = len(seq.token_ids) - end
    data.cal_and_set_input(seqs)
    actual = model.logits_from_hidden(model(data)).float()
    remaining = torch.cat([expected[0][3:], expected[1][8:]])
    torch.testing.assert_close(actual, remaining, rtol=0.04, atol=0.035)
    # BF16 kernels can reorder almost tied logits across different batch shapes.
    top_two = remaining.topk(2, dim=-1).values
    stable = top_two[:, 0] - top_two[:, 1] > 0.07
    assert stable.any()
    assert torch.equal(actual[stable].argmax(-1), remaining[stable].argmax(-1))


def test_decoder_loads_native_replicated_parameters(tiny_model):
    model, weights = tiny_model
    parameter = model.model.layers[1].ple.key_proj.weight
    expected = weights["model.language_model.layers.1.ple.key_proj.weight"].to(
        parameter.device
    )
    torch.testing.assert_close(parameter, expected, rtol=0, atol=0)
    assert model.model.layers[3].self_attn.attn_output_gate
    assert model.ssm_cache_config.auxiliary_shapes["ple_history"] == (1, 9, 128)


@torch.inference_mode()
def test_startup_profile_before_cache_arena_allocation(tiny_model):
    model, _ = tiny_model
    manager, data = make_inputs(model)
    seq = make_request(manager, 1, [2, 5, 7, 11])
    seq.to_compute_token_num = len(seq.token_ids)
    data.cal_and_set_input([seq])
    manager.segment = manager.ssm_segment = None
    hidden = model(data)
    assert hidden.shape == (4, model.config.hidden_size)
    assert torch.isfinite(hidden).all()


@pytest.mark.parametrize("budget", [64, 8])
@torch.inference_mode()
def test_decoder_logits_match_transformers_native_model(tiny_model, budget):
    from transformers.models.qwen4_exp.modeling_qwen4_exp import (
        Qwen4ExpForCausalLM,
    )

    model, weights = tiny_model
    config = deepcopy(model.config)
    config.indexer_budget = budget
    config._attn_implementation = "eager"
    reference = (
        Qwen4ExpForCausalLM(config).to(device="cuda", dtype=torch.bfloat16).eval()
    )
    state = {}
    table_parts = []
    for name, tensor in weights.items():
        name = name.replace("model.language_model.", "model.")
        if ".ngram_embedding.shard_" in name:
            table_parts.append((int(name.split("shard_")[1].split(".")[0]), tensor))
        else:
            state[name] = tensor
    table_name = "model.layers.1.ple.ple_embedding.ngram_embedding.weight"
    rows = reference.state_dict()[table_name].shape[0]
    state[table_name] = torch.cat([tensor for _, tensor in sorted(table_parts)])[:rows]
    reference.load_state_dict(state, strict=True)
    model.model.layers[3].self_attn.budget = budget
    tokens = [2, 5, 7, 11, 13, 31, 17, 19, 23, 29, 3, 9, 12, 15, 21, 22, 24]
    expected = (
        reference(torch.tensor([tokens], device="cuda"), use_cache=False)
        .logits[0]
        .float()
    )
    manager, data = make_inputs(model)
    seq = make_request(manager, 1, tokens)
    actual = run_chunk(model, data, seq, len(tokens))
    torch.testing.assert_close(actual, expected, rtol=0.04, atol=0.035)
    assert torch.equal(actual.argmax(-1), expected.argmax(-1))


@torch.inference_mode()
def test_piecewise_prefill_replays_ragged_requests(tiny_model):
    from gllm.runtime.piecewise_cuda_graph import PiecewiseGraphRunner

    model, _ = tiny_model
    manager, data = make_inputs(model)
    requests = [make_request(manager, i, list(range(1, 49))) for i in range(3)]
    for seq in requests[:2]:
        seq.to_compute_token_num = 16
    data.cal_and_set_input(requests[:2])
    embedded = model.model.embed_tokens(data.get_tokens())
    runner = PiecewiseGraphRunner(model, [16, 32])
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        runner.capture_bucket(data, embedded)
        for seq in requests[:2]:
            seq.to_compute_token_num = 8
        data.cal_and_set_input(requests[:2])
        runner.capture_bucket(data, model.model.embed_tokens(data.get_tokens()))
    torch.cuda.current_stream().wait_stream(stream)
    state = manager.ssm_segment
    tensors = [state.conv_state, state.temporal_state, *state.auxiliary_states.values()]
    for lengths in ((15, 16), (1, 9, 7), (8, 3, 2)):
        batch = requests[: len(lengths)]
        for seq, length in zip(batch, lengths):
            seq.to_compute_token_num = length
        data.cal_and_set_input(batch)
        embedded = model.model.embed_tokens(data.get_tokens())
        baseline = [tensor.clone() for tensor in tensors]
        expected = model(data, embedded).clone()
        expected_states = [tensor.clone() for tensor in tensors]
        for tensor, saved in zip(tensors, baseline):
            tensor.copy_(saved)
        actual = runner.run(data, embedded)
        assert actual is not None
        torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.04)
        for index, (tensor, saved) in enumerate(zip(tensors, expected_states)):
            slots = [seq.recurrent_state_slot for seq in batch]
            actual_state = tensor[:, slots] if index < 2 else tensor[slots]
            expected_state = saved[:, slots] if index < 2 else saved[slots]
            torch.testing.assert_close(
                actual_state, expected_state, atol=0.02, rtol=0.04
            )
        for seq, length in zip(batch, lengths):
            seq.computed_token_num += length


@torch.inference_mode()
def test_piecewise_mixed_decode_and_prefill_preserves_live_states(tiny_model):
    from gllm.runtime.piecewise_cuda_graph import PiecewiseGraphRunner

    model, _ = tiny_model
    manager, data = make_inputs(model)
    decode = make_request(manager, 0, list(range(1, 9)))
    run_chunk(model, data, decode, 8)
    decode.append(9)
    prefill = make_request(manager, 1, list(range(1, 49)))
    prefill.to_compute_token_num = 16
    data.cal_and_set_input([prefill])
    runner = PiecewiseGraphRunner(model, [16])
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        runner.capture_bucket(data, model.model.embed_tokens(data.get_tokens()))
    torch.cuda.current_stream().wait_stream(stream)
    state = manager.ssm_segment
    tensors = [state.conv_state, state.temporal_state, *state.auxiliary_states.values()]
    for length in (11, 7):
        decode.to_compute_token_num = 1
        prefill.to_compute_token_num = length
        data.cal_and_set_input([decode, prefill])
        assert data.num_decodes == 1
        embedded = model.model.embed_tokens(data.get_tokens())
        baseline = [tensor.clone() for tensor in tensors]
        expected = model(data, embedded).clone()
        expected_states = [tensor.clone() for tensor in tensors]
        for tensor, saved in zip(tensors, baseline):
            tensor.copy_(saved)
        actual = runner.run(data, embedded)
        torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.04)
        slots = [decode.recurrent_state_slot, prefill.recurrent_state_slot]
        for index, (tensor, saved) in enumerate(zip(tensors, expected_states)):
            torch.testing.assert_close(
                tensor[:, slots] if index < 2 else tensor[slots],
                saved[:, slots] if index < 2 else saved[slots],
                atol=0.02,
                rtol=0.04,
            )
        decode.computed_token_num += 1
        decode.append(10)
        prefill.computed_token_num += length


@pytest.fixture
def mtp_model(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    raw = tiny_config()
    weights = write_checkpoint(tmp_path, raw)
    for key, value in list(weights.items()):
        if "model.language_model.layers.3." in key:
            weights[key.replace("model.language_model.layers.3.", "mtp.layers.0.")] = (
                value.clone()
            )
        if "model.language_model.hyper_connection_mixer." in key:
            weights[key.replace("model.language_model.", "mtp.")] = value.clone()
    weights["mtp.fc_embedding.weight"] = torch.randn(32, 32, dtype=torch.bfloat16) * 0.1
    weights["mtp.fc_hidden.weight"] = torch.randn(32, 32, dtype=torch.bfloat16) * 0.1
    weights["mtp.pre_fc_norm_embedding.weight"] = torch.zeros(32, dtype=torch.bfloat16)
    weights["mtp.pre_fc_norm_hidden.weight"] = torch.zeros(128, dtype=torch.bfloat16)
    save_file(weights, str(tmp_path / "weights.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(weights, "weights.safetensors")})
    )
    config = AutoConfig.from_pretrained(tmp_path)
    config.checkpoint_path = str(tmp_path)
    config.mtp_enabled = True
    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        model = Qwen4ExpForCausalLM(config).eval()
        lazy = LazySafetensors(
            {key: (str(tmp_path / "weights.safetensors"), key) for key in weights}
        )
        with torch.no_grad():
            model.load_weights(lazy)
        lazy.close()
    finally:
        torch.set_default_dtype(original_dtype)
    yield model
    for layer in model.model.layers:
        if layer.ple_lookup is not None:
            layer.ple_lookup.close()


@torch.inference_mode()
def test_mtp_carries_all_hidden_streams(mtp_model):
    manager, data = make_inputs(mtp_model, mtp_k=2)
    seq = make_request(manager, 0, list(range(1, 17)))
    run_chunk(mtp_model, data, seq, 8)
    seq.to_compute_token_num = 1
    data.cal_and_set_input([seq])
    hidden = mtp_model(data)
    assert hidden.shape == (1, 128)
    draft = mtp_model.mtp(data, hidden, torch.tensor([10], device="cuda"))
    assert draft.shape == hidden.shape
    assert mtp_model.mtp.logits_from_hidden(draft).shape == (1, 64)
    assert mtp_model.logits_from_hidden(hidden).shape == (1, 64)
    assert mtp_model.num_kv_layers == mtp_model.model.num_kv_layers + 1


@pytest.mark.parametrize("accepted", [0, 1, 2])
@torch.inference_mode()
def test_verify_ple_state_matches_accepted_prefix(mtp_model, accepted):
    model = mtp_model
    tokens = list(range(1, 25))
    baseline_manager, baseline_data = make_inputs(model, mtp_k=2)
    baseline_seq = make_request(baseline_manager, 0, tokens)
    run_chunk(model, baseline_data, baseline_seq, 8)
    for end in range(9, 10 + accepted):
        expected_logits = run_chunk(model, baseline_data, baseline_seq, end)
    baseline_slot = baseline_seq.recurrent_state_slot

    manager, data = make_inputs(model, mtp_k=2)
    seq = make_request(manager, 0, tokens)
    manager.ssm_segment.free_block_table(seq.ssm_block_table)
    seq.ssm_block_table = manager.ssm_segment.allocate_block_table(3)
    seq.recurrent_state_slot = seq.ssm_block_table[0]
    run_chunk(model, data, seq, 8)
    seq._mtp_verify = True
    seq.to_compute_token_num = 3
    seq.to_compute_tokens = tokens[8:11]
    data.cal_and_set_input([seq])
    assert data.num_mtp_verify_rows == 1
    actual_logits = model.logits_from_hidden(model(data)).float()
    torch.testing.assert_close(
        actual_logits[accepted], expected_logits[0], rtol=0.04, atol=0.04
    )
    accepted_slot = seq.ssm_block_table[accepted]
    for name in ("ple_history", "ple_tokens"):
        actual = manager.ssm_segment.auxiliary_states[name][accepted_slot]
        expected = baseline_manager.ssm_segment.auxiliary_states[name][baseline_slot]
        torch.testing.assert_close(actual, expected, rtol=0.04, atol=0.02)
    for name in ("conv_state", "temporal_state"):
        actual = getattr(manager.ssm_segment, name)[:, accepted_slot]
        expected = getattr(baseline_manager.ssm_segment, name)[:, baseline_slot]
        torch.testing.assert_close(actual, expected, rtol=0.04, atol=0.02)
    # Exercise the runtime's accepted-column swap, then overwrite rejected KV.
    seq.ssm_block_table[0], seq.ssm_block_table[accepted] = (
        seq.ssm_block_table[accepted],
        seq.ssm_block_table[0],
    )
    seq.recurrent_state_slot = seq.ssm_block_table[0]
    seq.computed_token_num = 9 + accepted
    seq._mtp_verify = False
    seq.to_compute_tokens = None
    seq.ssm_num_accepted = 1
    actual = run_chunk(model, data, seq, 10 + accepted)
    expected = run_chunk(model, baseline_data, baseline_seq, 10 + accepted)
    torch.testing.assert_close(actual, expected, rtol=0.04, atol=0.04)


@torch.inference_mode()
def test_verify_graph_replays_candidates_and_resume_columns(mtp_model):
    model = mtp_model
    manager, data = make_inputs(model, mtp_k=2)
    seq = make_request(manager, 0, list(range(1, 33)))
    manager.ssm_segment.free_block_table(seq.ssm_block_table)
    seq.ssm_block_table = manager.ssm_segment.allocate_block_table(3)
    seq.recurrent_state_slot = seq.ssm_block_table[0]
    run_chunk(model, data, seq, 8)
    seq._mtp_verify = True
    seq.to_compute_token_num = 3
    seq.to_compute_tokens = [9, 10, 11]
    data.cal_and_set_input([seq])
    state = manager.ssm_segment
    tensors = [state.conv_state, state.temporal_state, *state.auxiliary_states.values()]
    baseline = [tensor.clone() for tensor in tensors]

    def restore():
        for tensor, saved in zip(tensors, baseline):
            tensor.copy_(saved)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        model(data)
    torch.cuda.current_stream().wait_stream(stream)
    restore()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = model(data)
    restore()
    for accepted in (1, 2, 3):
        seq.ssm_num_accepted = accepted
        seq.to_compute_tokens = [12 + accepted, 17, 19]
        data.cal_and_set_input([seq])
        baseline = [tensor.clone() for tensor in tensors]
        expected = model(data).clone()
        expected_states = [tensor.clone() for tensor in tensors]
        restore()
        graph.replay()
        torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.04)
        for index, (tensor, saved) in enumerate(zip(tensors, expected_states)):
            slots = seq.ssm_block_table
            actual_state = tensor[:, slots] if index < 2 else tensor[slots]
            expected_state = saved[:, slots] if index < 2 else saved[slots]
            torch.testing.assert_close(
                actual_state, expected_state, atol=0.02, rtol=0.04
            )
        seq.computed_token_num += accepted


@torch.inference_mode()
def test_piecewise_prefill_preserves_mtp_stream_width(mtp_model):
    from test_qwen4_exp_decoder import test_piecewise_prefill_replays_ragged_requests

    test_piecewise_prefill_replays_ragged_requests((mtp_model, None))


@torch.inference_mode()
def test_piecewise_mixed_verify_and_prefill_updates_live_metadata(mtp_model):
    from gllm.runtime.piecewise_cuda_graph import PiecewiseGraphRunner

    model = mtp_model
    manager, data = make_inputs(model, mtp_k=2)
    seq = make_request(manager, 0, list(range(1, 33)))
    manager.ssm_segment.free_block_table(seq.ssm_block_table)
    seq.ssm_block_table = manager.ssm_segment.allocate_block_table(3)
    seq.recurrent_state_slot = seq.ssm_block_table[0]
    run_chunk(model, data, seq, 8)
    prefill = make_request(manager, 1, list(range(1, 49)))
    prefill.to_compute_token_num = 16
    data.cal_and_set_input([prefill])
    runner = PiecewiseGraphRunner(model, [16])
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        runner.capture_bucket(data, model.model.embed_tokens(data.get_tokens()))
    torch.cuda.current_stream().wait_stream(stream)
    state = manager.ssm_segment
    tensors = [state.conv_state, state.temporal_state, *state.auxiliary_states.values()]
    for accepted, length in ((1, 11), (2, 7)):
        seq._mtp_verify = True
        seq.ssm_num_accepted = accepted
        seq.to_compute_token_num = 3
        seq.to_compute_tokens = [9 + accepted, 12, 13]
        prefill.to_compute_token_num = length
        data.cal_and_set_input([seq, prefill])
        embedded = model.model.embed_tokens(data.get_tokens())
        baseline = [tensor.clone() for tensor in tensors]
        expected = model(data, embedded).clone()
        expected_states = [tensor.clone() for tensor in tensors]
        for tensor, saved in zip(tensors, baseline):
            tensor.copy_(saved)
        actual = runner.run(data, embedded)
        torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.04)
        slots = seq.ssm_block_table + [prefill.recurrent_state_slot]
        for index, (tensor, saved) in enumerate(zip(tensors, expected_states)):
            actual_state = tensor[:, slots] if index < 2 else tensor[slots]
            expected_state = saved[:, slots] if index < 2 else saved[slots]
            torch.testing.assert_close(
                actual_state, expected_state, atol=0.02, rtol=0.04
            )
        seq.computed_token_num += accepted
        prefill.computed_token_num += length


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("grid", [[[1, 4, 4]], [[2, 4, 6]]])
@torch.no_grad()
def test_native_vision_outputs(tmp_path, grid):
    raw = tiny_config()
    raw["vision_config"] = {
        "model_type": "qwen4_exp",
        "depth": 2,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_heads": 4,
        "out_hidden_size": 32,
        "num_position_embeddings": 64,
        "patch_size": 16,
        "temporal_patch_size": 2,
        "spatial_merge_size": 2,
        "in_channels": 3,
        "hidden_act": "gelu_pytorch_tanh",
        "deepstack_visual_indexes": [],
    }
    weights = write_checkpoint(tmp_path, raw)
    config = AutoConfig.from_pretrained(tmp_path)
    config.checkpoint_path = str(tmp_path)
    config.max_num_batched_tokens = 128
    config.mtp_enabled = False
    config.attention_backend = "flashinfer"
    config.vision_config._attn_implementation = "eager"
    torch.manual_seed(319)
    reference = (
        Qwen4ExpVisionModel(config.vision_config)
        .to(device="cuda", dtype=torch.bfloat16)
        .eval()
    )
    weights.update(
        {
            f"model.visual.{key}": value.cpu()
            for key, value in reference.state_dict().items()
        }
    )
    save_file(weights, str(tmp_path / "weights.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(weights, "weights.safetensors")})
    )
    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        model = Qwen4ExpForConditionalGeneration(config).eval()
        lazy = LazySafetensors(
            {key: (str(tmp_path / "weights.safetensors"), key) for key in weights}
        )
        model.load_weights(lazy)
        lazy.close()
    finally:
        torch.set_default_dtype(original_dtype)
    grid_tensor = torch.tensor(grid, device="cuda")
    pixels = torch.randn(
        int(grid_tensor.prod(-1).sum()),
        3 * 2 * 16 * 16,
        device="cuda",
        dtype=torch.bfloat16,
    )
    expected = reference(pixels, grid_thw=grid_tensor).pooler_output
    actual = model.visual(pixels, grid_tensor)
    torch.testing.assert_close(actual, expected, atol=0.006, rtol=0.04)
    assert model.index_head_dim == model.language_model.index_head_dim
    assert model.ssm_cache_config is model.language_model.ssm_cache_config
