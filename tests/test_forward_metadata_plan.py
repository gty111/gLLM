from types import SimpleNamespace

import torch

from gllm.runtime.forward_metadata import (
    ForwardMetadataPlan,
    MetadataMaterialization,
)
from gllm.runtime.input_data import InputData


def _seq(qlen, *, computed_prompt=True, mtp_verify=False):
    return SimpleNamespace(
        to_compute_token_num=qlen,
        computed_prompt=computed_prompt,
        _mtp_verify=mtp_verify,
    )


def test_cpu_mixed_decode_plan_geometry():
    plan = ForwardMetadataPlan.from_sequences(
        [_seq(1), _seq(1), _seq(37, computed_prompt=False)],
    )
    assert plan.materialization is MetadataMaterialization.CPU
    assert plan.batch_size == 3
    assert plan.num_tokens == 39
    assert plan.fast_path_rows == 2
    assert plan.fast_path_tokens == 2
    assert plan.fast_q_len_per_req == 1
    assert plan.context_max_query_len == 37


def test_mtp_mixed_gpu_patch_keeps_one_layout():
    plan = ForwardMetadataPlan.from_sequences(
        [
            _seq(4, mtp_verify=True),
            _seq(4, mtp_verify=True),
            _seq(21, computed_prompt=False),
            _seq(53, computed_prompt=False),
        ],
    )
    corrected = plan.with_gpu_patch(num_rows=2, qlen=4)
    assert corrected.materialization is MetadataMaterialization.CPU_WITH_GPU_PATCH
    assert corrected.gpu_patch_rows == 2
    assert corrected.fast_path_tokens == 8
    assert corrected.context_max_query_len == 53


def test_uniform_gpu_plan_includes_cuda_graph_padding_rows():
    plan = ForwardMetadataPlan.uniform_gpu(
        num_rows=8,
        qlen=4,
        is_mtp_verify=True,
    )
    assert plan.materialization is MetadataMaterialization.GPU_UNIFORM
    assert plan.batch_size == 8
    assert plan.num_tokens == 32
    assert plan.fast_path_rows == 8
    assert plan.fast_q_len_per_req == 4
    assert plan.context_max_query_len == 0


def test_uniform_gpu_snapshot_cpu_mirror_matches_device_skip_sentinel():
    """GPU MTP prep disables snapshots, so its CPU mirror must be all -1.

    Qwen checks this mirror before touching the device snapshot-target tensor;
    an uninitialized shape placeholder can spuriously take the synchronizing
    ``nonzero`` path once per GDN layer.
    """
    data = InputData.__new__(InputData)
    data.use_ssm_cache = True
    data.max_num_block = 8
    plan = ForwardMetadataPlan.uniform_gpu(
        num_rows=32,
        qlen=4,
        is_mtp_verify=True,
    )

    data.mark_gpu_buffer_shapes(seqs=[object()] * 32, plan=plan)

    mirror = data.ssm_snapshot_write_slot_per_seq_cpu
    assert mirror.device.type == "cpu"
    assert mirror.dtype is torch.int32
    assert mirror.tolist() == [-1] * 32
    assert data.ssm_snapshot_src_idx_cpu.numel() == 0
    assert data.ssm_snapshot_dst_idx_cpu.numel() == 0
    assert data._ssm_snapshot_valid_rows == ()


def test_snapshot_copy_indices_reuse_uploaded_slots_and_honor_row_start():
    data = InputData.__new__(InputData)
    data._ssm_snapshot_valid_rows = (1, 3, 7)
    data.ssm_snapshot_src_idx = torch.tensor([11, 13, 17])
    data.ssm_snapshot_dst_idx = torch.tensor([21, 23, 27])

    src, dst = data.get_ssm_snapshot_copy_indices(row_start=3)
    assert src.tolist() == [13, 17]
    assert dst.tolist() == [23, 27]
    assert data.get_ssm_snapshot_copy_indices(row_start=8) is None


def test_plan_rejects_nonuniform_attention_fast_prefix():
    try:
        ForwardMetadataPlan(
            materialization=MetadataMaterialization.CPU,
            query_lens=(1, 2, 8),
            num_decodes=2,
        )
    except ValueError as exc:
        assert "uniform query length" in str(exc)
    else:
        raise AssertionError("nonuniform attention prefix was accepted")


def test_deferred_mtp_plan_cannot_prepare_attention():
    plan = ForwardMetadataPlan.deferred_mtp(2)
    backend = SimpleNamespace(prepare_metadata=lambda *_: object())
    try:
        plan.prepare_attention(backend, SimpleNamespace())
    except RuntimeError as exc:
        assert "before deferred MTP input was materialized" in str(exc)
    else:
        raise AssertionError("deferred MTP metadata reached attention")


def test_plan_owns_materializer_then_install_order():
    events = []

    class _Input:
        def install_forward_metadata_plan(self, plan):
            events.append(("install", plan.materialization))
            self.forward_metadata_plan = plan

    class _Materializer:
        materialization = MetadataMaterialization.GPU_UNIFORM

        def materialize_buffers(self, input_data, plan):
            assert not hasattr(input_data, "forward_metadata_plan")
            events.append(("write", plan.num_tokens))

    plan = ForwardMetadataPlan.uniform_gpu(
        num_rows=2,
        qlen=4,
        is_mtp_verify=True,
    )
    input_data = _Input()
    assert plan.materialize(input_data, _Materializer()) is plan
    assert events == [
        ("write", 8),
        ("install", MetadataMaterialization.GPU_UNIFORM),
    ]
    assert input_data.forward_metadata_plan is plan
