"""Ownership rules for the fused all-reduce + norm wiring.

The invariant these lock in: ``input_layernorm`` all-reduces only when its
*predecessor* left a per-rank partial behind. Keying it off the layer's own
``_fuse_mlp`` instead happens to work when every layer is configured alike and
PP is off, but reduces an already-reduced tensor at every pipeline stage
boundary -- scaling the hidden state by ``tp_size`` with no error raised.
"""

from types import SimpleNamespace

from gllm.layers.fused_allreduce_norm import link_fused_reduces


def _stack(*deferred):
    return [SimpleNamespace(_fuse_mlp=flag) for flag in deferred]


def test_first_layer_never_owns_the_reduce():
    layers = _stack(True, True, True)
    link_fused_reduces(layers)
    # Layer 0's input is an embedding, or a tensor the previous stage reduced.
    assert layers[0]._fuse_input is False


def test_input_reduce_follows_the_predecessor():
    layers = _stack(True, True, True)
    tail = link_fused_reduces(layers)
    assert [layer._fuse_input for layer in layers] == [False, True, True]
    assert tail is True


def test_heterogeneous_stack_tracks_each_predecessor():
    # qwen3_5 mixes dense and MoE mlps via ``mlp_only_layers``; if one kind
    # cannot defer, only the layer after it must skip the fused path.
    layers = _stack(True, False, True, False)
    tail = link_fused_reduces(layers)
    assert [layer._fuse_input for layer in layers] == [False, True, False, True]
    assert tail is False


def test_no_deferral_anywhere_leaves_every_norm_plain():
    layers = _stack(False, False)
    assert link_fused_reduces(layers) is False
    assert [layer._fuse_input for layer in layers] == [False, False]


def test_empty_stack_reports_no_tail():
    assert link_fused_reduces([]) is False


def test_workspace_is_not_created_or_cached_during_graph_capture(monkeypatch):
    import torch

    import gllm.layers.fused_allreduce_norm as fused

    created = []
    monkeypatch.setattr(fused, "_WORKSPACES", {})
    monkeypatch.setattr(fused, "_create_workspace", lambda *key: created.append(key) or "ws")
    key = (2, 0, 128, 64, torch.bfloat16)

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert fused._workspace(*key) is None
    assert created == [] and fused._WORKSPACES == {}

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    assert fused._workspace(*key) == "ws"
    # Later calls (including during capture) reuse the same workspace.
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert fused._workspace(*key) == "ws"
    assert created == [key]


def test_prepare_is_a_no_op_without_tensor_parallelism(monkeypatch):
    import torch

    import gllm.layers.fused_allreduce_norm as fused

    monkeypatch.setattr(fused, "get_tp_size", lambda: 1)
    monkeypatch.setattr(fused, "_create_workspace", lambda *key: (_ for _ in ()).throw(AssertionError))
    assert fused.prepare_fused_allreduce_norm(64, torch.bfloat16) is False
