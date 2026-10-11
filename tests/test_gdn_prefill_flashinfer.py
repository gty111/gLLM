"""Indexed GDN prefill preserves ragged states and supports graph replay."""

import pytest
import torch

from gllm.layers.ops.fla import chunk_gated_delta_rule
from gllm.layers.ops.fla.fused_gdn_gating import fused_gdn_gating
from gllm.layers.ops.fla.index import prepare_chunk_indices, prepare_chunk_offsets
from gllm.layers.ops.fla.l2norm import l2norm_fwd
from gllm.layers.ops.fla.prefill import (
    PrefillInputWorkspace,
    flashinfer_prefill,
    supports_flashinfer_prefill,
)
from gllm.models.qwen3_5 import _partition_query_start_loc
from gllm.runtime.input_data import InputData


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("tokens,hk,hv", [(1, 4, 12), (65, 4, 12), (257, 8, 8)])
@torch.inference_mode()
def test_fused_prefill_inputs_strided_and_graph(dtype, tokens, hk, hv):
    torch.manual_seed(97)
    packed = torch.randn(tokens, (2 * hk + hv) * 128, device="cuda", dtype=dtype)
    q, k, v = [
        x.view(1, tokens, h, 128)
        for x, h in zip(
            packed.split([hk * 128, hk * 128, hv * 128], dim=-1), (hk, hk, hv)
        )
    ]
    gates = torch.randn(tokens, 2 * hv, device="cuda", dtype=dtype)
    a, b = gates.split(hv, dim=-1)
    a_log = torch.randn(hv, device="cuda")
    bias = torch.randn(hv, device="cuda", dtype=dtype)
    # Exercise both branches of thresholded softplus and low-norm rows.
    a[0, 0] = 25
    q[:, 0, 0].zero_()
    k[:, 0, 0].mul_(1e-5)
    workspace = PrefillInputWorkspace()

    def check(actual):
        g, beta = fused_gdn_gating(a_log, a, b, bias)
        torch.testing.assert_close(actual[0], l2norm_fwd(q).squeeze(0), atol=0, rtol=0)
        torch.testing.assert_close(actual[1], l2norm_fwd(k).squeeze(0), atol=0, rtol=0)
        torch.testing.assert_close(actual[2], v.squeeze(0), atol=0, rtol=0)
        torch.testing.assert_close(actual[3], g.squeeze(0).exp(), atol=1e-7, rtol=2e-6)
        torch.testing.assert_close(actual[4], beta.squeeze(0), atol=0, rtol=0)

    prepared = workspace.prepare(q, k, v, a, b, a_log, bias)
    check(prepared)
    pointers = [x.data_ptr() for x in prepared]
    assert pointers == [
        x.data_ptr() for x in workspace.prepare(q, k, v, a, b, a_log, bias)
    ]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = workspace.prepare(q, k, v, a, b, a_log, bias)
    packed.normal_()
    gates.normal_()
    a_log.normal_()
    bias.normal_()
    graph.replay()
    check(captured)
    # Growing the workspace must retain the old captured addresses.
    more = torch.randn(1, 2 * tokens, hk, 128, device="cuda", dtype=dtype)
    more_v = torch.randn(1, 2 * tokens, hv, 128, device="cuda", dtype=dtype)
    more_gate = torch.randn(2 * tokens, hv, device="cuda", dtype=dtype)
    workspace.prepare(more, more, more_v, more_gate, more_gate, a_log, bias)
    graph.replay()
    check(captured)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize("lens", [[65, 129], [1, 255], [128, 128]])
@pytest.mark.parametrize("fused_inputs", [False, True])
@torch.inference_mode()
def test_indexed_prefill_numerics_and_dynamic_graph(lens, fused_inputs):
    if not supports_flashinfer_prefill(128, 128):
        pytest.skip("requires the indexed GDN prefill backend")
    torch.manual_seed(71)
    total = sum(lens)
    q = torch.randn(1, total, 4, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(1, total, 12, 128, device="cuda", dtype=q.dtype)
    g = -torch.rand(1, total, 12, device="cuda")
    beta = torch.rand_like(g)
    fused_kwargs = {}
    if fused_inputs:
        a = torch.randn(total, 12, device="cuda", dtype=q.dtype)
        b = torch.randn_like(a)
        a_log = torch.zeros(12, device="cuda")
        bias = torch.zeros_like(a_log)
        g, beta = fused_gdn_gating(a_log, a, b, bias)
        fused_kwargs = {
            "gate_inputs": (a, b, a_log, bias),
            "workspace": PrefillInputWorkspace(),
        }
    boundaries = torch.tensor([0, lens[0], total], device="cuda", dtype=torch.int32)
    boundaries._cpu_view = boundaries.cpu()
    slots = torch.tensor([3, 1], device="cuda", dtype=torch.int32)
    # Match the per-layer view of the arena: padded outer stride, contiguous H/V/K.
    arena = torch.randn(5, 2, 12, 128, 128, device="cuda", dtype=torch.float32) * 0.01
    state = arena[:, 1]
    original = state.clone()

    def reference():
        return chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=state,
            initial_state_indices=slots,
            cu_seqlens=boundaries,
            scale=128**-0.5,
            head_first=False,
            use_qk_l2norm_in_kernel=True,
        )[:2]

    expected, _ = reference()
    final = state[slots].clone()
    state.copy_(original)
    result = flashinfer_prefill(
        q, k, v, g, beta, state, slots, boundaries, 128**-0.5, **fused_kwargs
    )
    torch.testing.assert_close(result, expected.squeeze(0), atol=0.003, rtol=0.03)
    torch.testing.assert_close(state[slots], final, atol=0.003, rtol=0.03)
    torch.testing.assert_close(state[[0, 2, 4]], original[[0, 2, 4]], atol=0, rtol=0)
    for _ in range(3):
        flashinfer_prefill(
            q, k, v, g, beta, state, slots, boundaries, 128**-0.5, **fused_kwargs
        )
    output_buffer = torch.empty_like(result)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = flashinfer_prefill(
            q,
            k,
            v,
            g,
            beta,
            state,
            slots,
            boundaries,
            128**-0.5,
            out=output_buffer,
            **fused_kwargs,
        )
    assert captured.data_ptr() == output_buffer.data_ptr()
    # Reuse addresses with different sequence boundaries and pool slots.
    boundaries.copy_(
        torch.tensor([0, total // 2, total], device="cuda", dtype=torch.int32)
    )
    slots.copy_(torch.tensor([4, 0], device="cuda", dtype=torch.int32))
    q.normal_()
    state.copy_(original)
    graph.replay()
    graph_state = state.clone()
    state.copy_(original)
    eager = flashinfer_prefill(
        q, k, v, g, beta, state, slots, boundaries, 128**-0.5, **fused_kwargs
    )
    torch.testing.assert_close(captured, eager, atol=0, rtol=0)
    torch.testing.assert_close(state, graph_state, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@torch.inference_mode()
def test_indexed_prefill_state_offsets_exceed_int32():
    if not supports_flashinfer_prefill(128, 128):
        pytest.skip("requires the indexed GDN prefill backend")
    torch.manual_seed(31)
    state = torch.empty_strided(
        (6, 12, 128, 128),
        (2**29, 128 * 128, 128, 1),
        dtype=torch.float32,
        device="cuda",
    )
    slots = torch.tensor([5, 1], device="cuda", dtype=torch.int32)
    original = torch.randn(2, 12, 128, 128, device="cuda") * 0.01
    state.index_copy_(0, slots.long(), original)
    q = torch.randn(1, 2, 4, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(1, 2, 12, 128, device="cuda", dtype=q.dtype)
    g = -torch.rand(1, 2, 12, device="cuda")
    beta = torch.rand_like(g)
    boundaries = torch.tensor([0, 1, 2], device="cuda", dtype=torch.int32)
    boundaries._cpu_view = boundaries.cpu()
    expected, _ = chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=state,
        initial_state_indices=slots,
        cu_seqlens=boundaries,
        scale=128**-0.5,
        head_first=False,
        use_qk_l2norm_in_kernel=True,
    )[:2]
    final = state[slots].clone()
    state.index_copy_(0, slots.long(), original)
    actual = flashinfer_prefill(q, k, v, g, beta, state, slots, boundaries, 128**-0.5)
    torch.testing.assert_close(actual, expected.squeeze(0), atol=0.003, rtol=0.03)
    torch.testing.assert_close(state[slots], final, atol=0.003, rtol=0.03)


def test_forward_metadata_reuses_partition_and_expires_with_new_batch():
    data = InputData.__new__(InputData)
    data.query_start_loc = torch.tensor([0, 1, 66, 196], dtype=torch.int32)
    data.query_start_loc_cpu = data.query_start_loc.clone()
    boundaries = data.get_query_start_loc()
    assert data.get_query_start_loc() is boundaries
    suffix = _partition_query_start_loc(boundaries, data.query_start_loc_cpu, 1)
    assert _partition_query_start_loc(boundaries, data.query_start_loc_cpu, 1) is suffix
    indices = prepare_chunk_indices(suffix, 64)
    offsets = prepare_chunk_offsets(suffix, 64)
    assert prepare_chunk_indices(suffix, 64) is indices
    assert prepare_chunk_offsets(suffix, 64) is offsets
    torch.testing.assert_close(
        indices,
        torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1], [1, 2]], dtype=torch.int32),
    )
    torch.testing.assert_close(offsets, torch.tensor([0, 2, 5], dtype=torch.int32))
    data.query_start_loc.copy_(torch.tensor([0, 1, 2, 3]))
    data.query_start_loc_cpu = data.query_start_loc.clone()
    fresh = data.get_query_start_loc()
    assert fresh is not boundaries
    suffix = _partition_query_start_loc(fresh, data.query_start_loc_cpu, 1)
    torch.testing.assert_close(
        prepare_chunk_offsets(suffix, 64), torch.tensor([0, 1, 2], dtype=torch.int32)
    )


def test_unmanaged_mutable_boundaries_do_not_use_forward_cache():
    boundaries = torch.tensor([0, 65], dtype=torch.int32)
    torch.testing.assert_close(
        prepare_chunk_offsets(boundaries, 64), torch.tensor([0, 2], dtype=torch.int32)
    )
    boundaries[1] = 1
    torch.testing.assert_close(
        prepare_chunk_offsets(boundaries, 64), torch.tensor([0, 1], dtype=torch.int32)
    )
