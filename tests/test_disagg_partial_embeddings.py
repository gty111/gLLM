"""Encoder-disaggregation embeddings in cache-arena pages: streamed readiness,
in-place reads, page allocation / release."""
from types import SimpleNamespace

import pytest
import torch

from gllm.disagg.lm_manager import (
    DisaggCoordinator,
    DisaggEvents,
    _PendingItem,
    _PendingSeq,
)
from gllm.disagg.paging import MM_EMBED, PagedRows, copy_runs, embed_layout, row_index
from gllm.disagg.protocol import (
    emb_notif,
    emb_partial_notif,
    parse_emb_notif,
    parse_emb_partial_notif,
)
from gllm.disagg.runner_mixin import DisaggMixin, DisaggSeqState
from gllm.runtime.cache_arena import CacheArena


def test_partial_notif_roundtrip_and_disjoint_from_full():
    msg = emb_partial_notif(7, 1, 4096)
    assert parse_emb_partial_notif(msg) == (7, 1, 4096)
    assert parse_emb_notif(msg) is None
    assert parse_emb_partial_notif(emb_notif(7, 1)) is None
    assert parse_emb_partial_notif(b"embp:1:2") is None


def test_copy_runs_split_at_both_page_boundaries():
    # Source pages of 4 rows, destination pages of 3 rows starting at row 2.
    runs = copy_runs([5, 1], 4, 0, [7, 3, 9], 3, 2, 6)
    assert runs == [(5, 0, 7, 2, 1), (5, 1, 3, 0, 3), (1, 0, 9, 0, 2)]
    assert sum(r[-1] for r in runs) == 6


def test_paged_rows_gathers_in_row_order():
    pool = torch.arange(3 * 2 * 2, dtype=torch.float32).reshape(3, 2, 2)
    page, off = row_index([2, 0], 3, 2)
    rows = PagedRows(pool, torch.tensor(page), torch.tensor(off))
    assert rows.shape == (3, 2)
    assert torch.equal(rows[0:3], torch.stack([pool[2, 0], pool[2, 1], pool[0, 0]]))
    assert torch.equal(rows[1:3, :1], torch.stack([pool[2, 1], pool[0, 0]])[:, :1])
    assert torch.equal(rows.prefix(1)[0:1], pool[2, :1])


def _runner(pool=None, rows_per_page=4):
    r = SimpleNamespace()
    DisaggMixin._init_disagg_state(r)
    if pool is not None:
        DisaggMixin.disagg_attach_pool(r, pool, rows_per_page)
    return r


def _state(pages=([0], [1, 2])):
    # text(0-2) | image span [3, 7) with 4 visual tokens | text(7-8) |
    # video span [9, 19) where positions 10 and 15 are interleaved text
    # (timestamps) and the rest are 8 visual tokens | text(19)
    mask = torch.zeros(20, dtype=torch.bool)
    mask[3:7] = True
    mask[9:19] = True
    mask[10] = mask[15] = False
    return DisaggSeqState(
        num_items=2,
        item_span=[(3, 7), (9, 19)],
        item_modality=["image", "video"],
        item_ready=[False, False],
        image_grid_thw=None,
        video_grid_thw=None,
        input_ids_cpu=torch.arange(20),
        is_multimodal_cpu=mask,
        prompt_positions=torch.zeros(3, 20, dtype=torch.long),
        mrope_position_delta=torch.zeros(1),
        prompt_len=20,
        item_rows=[0, 0],
        item_pages=[list(p) for p in pages],
    )


def _registered():
    # Pages of 4 rows x 2 features; each row holds (page, row) for checking.
    pool = torch.stack(
        [torch.stack([torch.tensor([p, r], dtype=torch.float32) for r in range(4)])
         for p in range(4)]
    )
    r = _runner(pool)
    st = _state()
    DisaggMixin.disagg_register(r, 0, st)
    return r, st


def test_ready_len_follows_landed_rows():
    r, st = _registered()
    mark = lambda *a: DisaggMixin.disagg_mark_ready(r, 0, *a)  # noqa: E731
    assert DisaggMixin._disagg_ready_len(st) == 3  # nothing landed
    mark(0, 4, True)
    assert DisaggMixin._disagg_ready_len(st) == 9  # image done, video untouched
    mark(1, 3, False)
    # 3 video rows -> visual positions 9, 11, 12 ready; next visual is 13
    assert DisaggMixin._disagg_ready_len(st) == 13
    mark(1, 8, True)
    assert DisaggMixin._disagg_ready_len(st) == 20


def test_collect_reads_ready_rows_in_place():
    r, st = _registered()
    DisaggMixin.disagg_mark_ready(r, 0, 0, 4, True)
    DisaggMixin.disagg_mark_ready(r, 0, 1, 3, False)
    works = []
    runner = SimpleNamespace(
        embedding_cache={}, _disagg_ready_len=DisaggMixin._disagg_ready_len
    )
    seq = SimpleNamespace(seq_id=0, computed_token_num=0, seq_len=13)
    DisaggMixin._mm_disagg_collect(runner, seq, st, works, [])
    (work,) = works
    assert work["coverage_len"] == 13
    rows = work["visual_rows"]
    assert rows.shape == (7, 2) and int(work["is_multimodal_cpu"].sum()) == 7
    # image rows: page 0 rows 0-3; video rows: page 1 rows 0-2
    assert rows[0:7].tolist() == [[0, 0], [0, 1], [0, 2], [0, 3], [1, 0], [1, 1], [1, 2]]
    # rows 4.. of the video continue on page 2
    assert st.visual_rows[8:12].tolist() == [[2, 0], [2, 1], [2, 2], [2, 3]]


def test_emit_ready_streams_row_ranges():
    item = SimpleNamespace(
        emitted_final=False, meta=SimpleNamespace(num_tokens=8), embedding_ready=False,
        ready_rows=0, emitted_rows=0, content=object(),
    )
    ps = SimpleNamespace(ordered=[item], seq=SimpleNamespace(seq_id=11))

    def emit():
        ev = DisaggEvents()
        DisaggCoordinator._emit_ready(None, ps, ev)
        return ev.emb_ready

    assert emit() == []
    item.ready_rows = 3
    assert emit() == [(11, 0, 0, 3, False)]
    assert emit() == []
    item.ready_rows, item.embedding_ready = 6, True
    assert emit() == [(11, 0, 3, 8, True)]
    assert item.emitted_final and item.content is None
    assert emit() == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA events")
def test_pages_alloc_in_order_and_free_after_tick():
    arena = CacheArena(torch.zeros(4 * 64, dtype=torch.uint8, device="cuda"), 64)
    arena.register_cache(embed_layout(MM_EMBED, 4, torch.float32, 4))
    r = _runner()
    r.memory_manager = SimpleNamespace(cache_arena=arena)
    got = DisaggMixin.disagg_alloc_pages(r, [(1, 0, 2), (2, 0, 3), (3, 0, 1)])
    # The second request does not fit; the third is skipped behind it.
    assert len(got[0]) == 2 and got[1] is None and got[2] is None
    assert arena.allocator.num_free_slots(MM_EMBED) == 2
    DisaggMixin.disagg_free_pages(r, [1])
    assert arena.allocator.num_free_slots(MM_EMBED) == 2  # quarantined
    DisaggMixin.disagg_tick(r)
    assert arena.allocator.num_free_slots(MM_EMBED) == 4 and r._disagg_pages == {}


def _coordinator(rows_per_page=4, max_pages=4):
    c = DisaggCoordinator.__new__(DisaggCoordinator)
    c.recv = SimpleNamespace(rows_per_page=rows_per_page)
    c.max_mm_pages, c._mm_pages = max_pages, 0
    c._pending, c._page_q, c._requested = {}, [], {}
    c._holders, c._released = {}, set()
    c._encoders = {"e": SimpleNamespace(job_sock=SimpleNamespace(send=lambda b: None))}
    c.redispatch_timeout_s = 20.0
    return c


def _pending(c, seq_id, rows):
    items = [
        _PendingItem(i, "image", meta=SimpleNamespace(num_tokens=n), encoder_identity="e")
        for i, n in enumerate(rows)
    ]
    c._pending[seq_id] = _PendingSeq(seq=SimpleNamespace(seq_id=seq_id), items=items)
    c._page_q += [(seq_id, i) for i in range(len(rows))]
    return c._pending[seq_id]


def test_coordinator_requests_pages_within_budget_and_frees_when_released():
    c = _coordinator()
    a = _pending(c, 1, [5])  # 2 pages
    b = _pending(c, 2, [9])  # 3 pages: over the remaining budget
    ev = DisaggEvents()
    DisaggCoordinator._request_pages(c, ev)
    assert ev.allocs == [(1, 0, 2)]
    DisaggCoordinator.on_pages(c, ev.allocs, [[7, 3]])
    assert a.items[0].pages == [7, 3] and c._mm_pages == 2 and a.pages_complete
    assert a.items[0].target_sent
    assert c._page_q == [(2, 0)] and not b.pages_complete

    # Released while its encoder is still writing: pages stay.
    c._released.add(1)
    ev = DisaggEvents()
    DisaggCoordinator._free_released(c, ev)
    assert ev.frees == []
    a.items[0].embedding_ready = True
    DisaggCoordinator._free_released(c, ev)
    assert ev.frees == [1] and c._mm_pages == 0 and 1 not in c._holders

    ev = DisaggEvents()
    DisaggCoordinator._request_pages(c, ev)
    assert ev.allocs == [(2, 0, 3)]
    DisaggCoordinator.on_pages(c, ev.allocs, [None])  # arena full: retry later
    assert b.items[0].pages is None and not b.items[0].page_requested
    assert c._page_q == [(2, 0)]


def test_pages_granted_after_abort_are_released():
    c = _coordinator()
    _pending(c, 1, [4])
    ev = DisaggEvents()
    DisaggCoordinator._request_pages(c, ev)
    del c._pending[1]  # aborted before the ALLOC result came back
    c._released.discard(1)
    DisaggCoordinator.on_pages(c, ev.allocs, [[0]])
    assert 1 in c._released
    ev = DisaggEvents()
    DisaggCoordinator._free_released(c, ev)  # no encoder was given the pages
    assert ev.frees == [1] and c._mm_pages == 0


def test_fail_notif_roundtrip():
    from gllm.disagg.protocol import emb_fail_notif, parse_emb_fail_notif

    assert parse_emb_fail_notif(emb_fail_notif(3, 1)) == (3, 1)
    assert parse_emb_notif(emb_fail_notif(3, 1)) is None
    assert parse_emb_fail_notif(emb_notif(3, 1)) is None


def _watchdog_coordinator(encoders):
    c = DisaggCoordinator.__new__(DisaggCoordinator)
    c.lm_id = "lm0"
    c.redispatch_timeout_s = 20.0
    c.max_redispatch_attempts = 5
    c._pending = {}
    c._encoders = {e: SimpleNamespace(identity=e) for e in encoders}
    c.sent = []
    c._send_job = lambda seq_id, item, avoid=None: c.sent.append((seq_id, avoid)) or True
    return c


def test_timeout_with_single_busy_encoder_only_restarts_the_timer():
    c = _watchdog_coordinator(["e0"])
    item = _PendingItem(0, "video", encoder_identity="e0", dispatched_at=0.0, attempts=1)
    c._pending[1] = _PendingSeq(seq=None, items=[item])
    DisaggCoordinator._check_watchdog(c, DisaggEvents())
    assert c.sent == [] and item.attempts == 1 and item.dispatched_at > 0
    # A reported failure is re-dispatched even to the same encoder.
    item.encoder_failed = True
    DisaggCoordinator._check_watchdog(c, DisaggEvents())
    assert c.sent == [(1, "e0")] and not item.encoder_failed


def test_timeout_moves_item_to_another_encoder():
    c = _watchdog_coordinator(["e0", "e1"])
    item = _PendingItem(0, "video", encoder_identity="e0", dispatched_at=0.0, attempts=1)
    c._pending[1] = _PendingSeq(seq=None, items=[item])
    DisaggCoordinator._check_watchdog(c, DisaggEvents())
    assert c.sent == [(1, "e0")]
