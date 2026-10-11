"""MTP hidden-stream width and PLE state rollback across rejected drafts."""

import json

import pytest
import torch
from safetensors.torch import save_file
from test_qwen4_exp_decoder import (
    make_inputs,
    make_request,
    run_chunk,
    tiny_config,
    write_checkpoint,
)
from transformers import AutoConfig

from gllm.models.qwen4_exp import Qwen4ExpForCausalLM
from gllm.runtime.model_loader import LazySafetensors


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
