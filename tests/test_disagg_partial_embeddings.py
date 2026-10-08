"""Segment-streamed (partial) visual embeddings in encoder disaggregation."""
from types import SimpleNamespace

import torch

from gllm.disagg.lm_manager import DisaggCoordinator, DisaggEvents
from gllm.disagg.protocol import (
    emb_notif,
    emb_partial_notif,
    parse_emb_notif,
    parse_emb_partial_notif,
)
from gllm.disagg.runner_mixin import DisaggMixin, DisaggSeqState


def test_partial_notif_roundtrip_and_disjoint_from_full():
    msg = emb_partial_notif(7, 1, 4096)
    assert parse_emb_partial_notif(msg) == (7, 1, 4096)
    assert parse_emb_notif(msg) is None
    assert parse_emb_partial_notif(emb_notif(7, 1)) is None
    assert parse_emb_partial_notif(b"embp:1:2") is None


def _state():
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
        item_embed=[None, None],
        image_grid_thw=None,
        video_grid_thw=None,
        input_ids_cpu=torch.arange(20),
        is_multimodal_cpu=mask,
        prompt_positions=torch.zeros(3, 20, dtype=torch.long),
        mrope_position_delta=torch.zeros(1),
        prompt_len=20,
        item_rows=[0, 0],
        item_buffer=[None, None],
    )


def _add(st, i, rows, end, final):
    DisaggMixin.disagg_add_embedding(
        SimpleNamespace(disagg_embeds={0: st}), 0, i, rows, end, final
    )


def test_ready_len_follows_landed_rows():
    st = _state()
    assert DisaggMixin._disagg_ready_len(st) == 3  # nothing landed
    _add(st, 0, torch.ones(4, 2), 4, True)
    assert DisaggMixin._disagg_ready_len(st) == 9  # image done, video untouched
    _add(st, 1, torch.full((3, 2), 2.0), 3, False)
    # 3 video rows -> visual positions 9, 11, 12 ready; next visual is 13
    assert DisaggMixin._disagg_ready_len(st) == 13
    _add(st, 1, torch.full((5, 2), 3.0), 8, True)
    assert DisaggMixin._disagg_ready_len(st) == 20
    assert st.item_embed[1].shape[0] == 8 and st.item_buffer[1] is None
    assert torch.equal(st.item_embed[1][:3], torch.full((3, 2), 2.0))
    assert torch.equal(st.item_embed[1][3:], torch.full((5, 2), 3.0))


def test_collect_gathers_partial_rows_in_order():
    st = _state()
    _add(st, 0, torch.ones(4, 2), 4, True)
    _add(st, 1, torch.full((3, 2), 2.0), 3, False)
    works = []
    runner = SimpleNamespace(
        embedding_cache={},
        _disagg_ready_len=DisaggMixin._disagg_ready_len,
    )
    seq = SimpleNamespace(seq_id=0, computed_token_num=0, seq_len=13)
    DisaggMixin._mm_disagg_collect(runner, seq, st, works, [])
    (work,) = works
    assert work["coverage_len"] == 13
    embeds = work["mm_embeddings"]
    assert [e.shape[0] for e in embeds] == [4, 3]
    assert int(work["is_multimodal_cpu"].sum()) == 7


def test_emit_ready_streams_row_ranges_then_frees_slot():
    item = SimpleNamespace(
        slot_freed=False, meta=SimpleNamespace(num_tokens=8), embedding_ready=False,
        ready_rows=0, emitted_rows=0, slot_id=5, content=object(),
    )
    ps = SimpleNamespace(ordered=[item], seq=SimpleNamespace(seq_id=11))
    coord = SimpleNamespace(_free_slots=[])

    def emit():
        ev = DisaggEvents()
        DisaggCoordinator._emit_ready(coord, ps, ev)
        return ev.emb_ready

    assert emit() == []
    item.ready_rows = 3
    assert emit() == [(11, 0, 5, 0, 3, False)]
    assert emit() == []
    item.ready_rows, item.embedding_ready = 6, True
    assert emit() == [(11, 0, 5, 3, 8, True)]
    assert coord._free_slots == [5] and item.slot_freed


def test_streamed_rows_are_copied_out_of_the_slot():
    st = _state()
    slot = torch.full((8, 2), 5.0)
    _add(st, 1, slot[:3], 3, False)
    slot.fill_(-1.0)  # slot reused by another item after this point
    assert torch.equal(st.item_buffer[1][:3], torch.full((3, 2), 5.0))
    whole = torch.ones(4, 2)
    _add(st, 0, whole, 4, True)
    assert st.item_embed[0] is whole  # single final event: stored as is
