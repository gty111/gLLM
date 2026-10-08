"""Cross-request pipelined encoder (staging, ordering, in-order notification)."""
import threading
import time
from types import SimpleNamespace

import pytest
import torch

from gllm.engine.encoder_pipeline import (
    EncoderPipeline,
    StagingAllocator,
    WorkItem,
    _PriorityQueue,
    video_segment_rows,
)


def test_staging_allocator_first_fit_and_coalesce():
    a = StagingAllocator(10)
    x, y, z = a.alloc(4), a.alloc(3), a.alloc(3)
    assert (x, y, z) == (0, 4, 7) and a.free_rows == 0
    a.free(y, 3)
    assert a.alloc(2) == 4
    a.free(x, 4)
    a.free(4, 2)
    a.free(z, 3)
    assert a.free_rows == 10 and a.alloc(10) == 0
    with pytest.raises(ValueError):
        a.alloc(11)


def test_staging_allocator_blocks_until_free():
    a = StagingAllocator(4)
    a.alloc(4)
    got = []
    t = threading.Thread(target=lambda: got.append(a.alloc(2)))
    t.start()
    time.sleep(0.05)
    assert got == []
    a.free(0, 4)
    t.join(1)
    assert got == [0]


def test_priority_queue_orders_and_closes():
    q = _PriorityQueue()
    for p in [(2, 0), (1, 1), (1, 0)]:
        q.put(p, p)
    assert [q.get() for _ in range(3)] == [(1, 0), (1, 1), (2, 0)]
    q.close()
    assert q.get() is None


def test_done_prefix_rows_requires_leading_run():
    w = WorkItem(order=0, job=None, num_tokens=9, chash=b"")
    w.seg_rows = [(0, 3), (3, 6), (6, 9)]
    w.seg_done = [False, True, True]
    assert w.done_prefix_rows() == 0
    w.seg_done[0] = True
    assert w.done_prefix_rows() == 9


def test_video_segment_rows_follow_temporal_patches():
    runner = SimpleNamespace(
        video_processor=SimpleNamespace(temporal_patch_size=2), spatial_merge_size=2
    )
    plan = {"grid_thw": torch.tensor([[5, 4, 6]])}  # 9 frames -> t=5
    rows = video_segment_rows(plan, [(0, 4), (4, 8), (8, 9)], runner)
    assert rows == [(0, 12), (12, 24), (24, 30)]  # 6 rows per temporal patch


class _FakeRunner:
    """Video items of ``n_seg`` segments x 2 rows; row value = seg index."""

    feat = 4

    def __init__(self, n_seg=4, delay=None):
        self.n_seg = n_seg
        self.delay = delay or (lambda job, seg: 0.0)
        self.mm_embed_cache = SimpleNamespace(get=lambda k: None, put=lambda k, v: None)
        self.video_processor = SimpleNamespace(temporal_patch_size=2)
        self.spatial_merge_size = 1

    def open_video_stream(self, content):
        return {"grid_thw": torch.tensor([[self.n_seg, 1, 2]]), "content_hash": content.encode()}

    def num_vis_tokens(self, grid):
        return int(grid.prod())

    def video_segment_bounds(self, plan):
        return [(2 * i, 2 * i + 2) for i in range(self.n_seg)]

    def prepare_video_segment(self, plan, lo, hi):
        job_id = plan["content_hash"].decode()
        time.sleep(self.delay(job_id, lo // 2))
        return {"pixel_values_videos": torch.full((2, self.feat), float(lo // 2))}

    def encode_segment(self, mm):
        return mm["pixel_values_videos"].clone()


def _run(pipe, jobs, timeout=10):
    for j in jobs:
        pipe.submit(j)
    started, staged, deadline = 0, [], time.monotonic() + timeout
    while time.monotonic() < deadline:
        for w in pipe.poll_planned():
            pipe.start(w)
            started += 1
        for w, seg, off, rows in pipe.poll_staged():
            staged.append((w, seg, pipe.send_buf[off : off + rows].clone()))
            pipe.segment_written(w, seg, off, rows)
        if started == len(jobs) and not pipe.busy:
            break
        time.sleep(0.001)
    return staged


def test_pipeline_streams_all_segments_with_correct_rows():
    runner = _FakeRunner(n_seg=4)
    pipe = EncoderPipeline(runner, torch.zeros(8, 4), decode_workers=3, max_ready_segments=2)
    jobs = [SimpleNamespace(seq_id=i, item_idx=0, modality="video", content=f"v{i}") for i in range(3)]
    staged = _run(pipe, jobs)
    pipe.close()
    assert len(staged) == 12
    for w, seg, rows in staged:
        assert torch.equal(rows, torch.full((2, 4), float(seg)))
        assert w.seg_rows[seg] == (2 * seg, 2 * seg + 2)
    for w in {id(w): w for w, *_ in staged}.values():
        assert all(w.seg_done) and w.done_prefix_rows() == w.num_tokens == 8


def test_pipeline_overlaps_requests():
    # Each segment takes 50 ms to decode; 2 videos x 4 segments on 8 workers
    # finish together instead of back to back.
    runner = _FakeRunner(n_seg=4, delay=lambda job, seg: 0.05)
    pipe = EncoderPipeline(runner, torch.zeros(16, 4), decode_workers=8)
    jobs = [SimpleNamespace(seq_id=i, item_idx=0, modality="video", content=f"v{i}") for i in range(2)]
    t0 = time.monotonic()
    staged = _run(pipe, jobs)
    pipe.close()
    assert len(staged) == 8
    assert time.monotonic() - t0 < 0.3  # serial would be >= 0.4 s


def test_pipeline_drops_failed_item_but_serves_others():
    class Failing(_FakeRunner):
        def prepare_video_segment(self, plan, lo, hi):
            if plan["content_hash"] == b"bad" and lo == 2:
                raise RuntimeError("corrupt segment")
            return super().prepare_video_segment(plan, lo, hi)

    pipe = EncoderPipeline(Failing(n_seg=3), torch.zeros(8, 4), decode_workers=2)
    jobs = [SimpleNamespace(seq_id=i, item_idx=0, modality="video", content=c)
            for i, c in enumerate(["bad", "good"])]
    staged = _run(pipe, jobs)
    pipe.close()
    good = [s for s in staged if s[0].chash == b"good"]
    assert len(good) == 3 and all(s[0].failed is False for s in good)
    bad = {id(s[0]): s[0] for s in staged if s[0].chash == b"bad"}
    assert all(w.failed for w in bad.values())
