from types import SimpleNamespace

import torch

import gllm.runtime.input_data as input_data_module
from gllm.runtime.input_data import InputData
from gllm.runtime.sequence import GenerationSequence


class _SnapshotSegment:
    def __init__(self):
        self.calls = []
        self.page2ssm_snapshot = [None] * 32
        self.page2ssm_snapshot_valid = [False] * 32

    def reserve_ssm_snapshot(self, page_num, end_tokens):
        self.calls.append((page_num, end_tokens))
        self.page2ssm_snapshot[page_num] = 7
        return 7


def test_overlap_snapshot_is_reserved_and_published_at_launch(monkeypatch):
    monkeypatch.setattr(input_data_module, "get_pp_size", lambda: 1)
    segment = _SnapshotSegment()
    manager = SimpleNamespace(
        page_size=16,
        use_mla=False,
        use_ssm_cache=True,
        segment=segment,
    )
    data = InputData(False, manager, max_seq_length=512)
    seq = GenerationSequence(1, [1] * 256, [])
    seq.ssm_state_slot = 3
    seq.to_compute_token_num = 256
    seq.page_table = list(range(16))

    data._cal_ssm_metadata([seq])

    # Merely prebuilding an overlap batch must not mutate snapshot ownership or
    # make zero-initialized storage visible to a later prefix-cache lookup.
    assert segment.calls == []
    assert data.ssm_snapshot_write_slot_per_seq_cpu.tolist() == [-1]
    assert not any(segment.page2ssm_snapshot_valid)

    data._materialize_ssm_snapshot_targets()
    assert segment.calls == [(15, 256)]
    assert data.ssm_snapshot_write_slot_per_seq_cpu.tolist() == [7]
    assert data._ssm_snapshot_valid_rows == (0,)
    assert data.ssm_snapshot_src_idx_cpu.tolist() == [3]
    assert data.ssm_snapshot_dst_idx_cpu.tolist() == [7]
    assert not any(segment.page2ssm_snapshot_valid)

    # Publication is a separate post-enqueue transition.
    data.mark_ssm_snapshot_writes_enqueued()
    assert segment.page2ssm_snapshot_valid[15]
    assert data._ssm_snapshot_writes_pending == ()
