import msgspec
import pytest

from gllm.distributed import wire
from gllm.distributed.comm import IPCPackage
from gllm.runtime.sequence import GenerationSequence
from gllm.structured_output import StructuredOutput


def _seq(**kw):
    seq = GenerationSequence(
        7, [1, 2, 3], [151645, 151643], 32, False, 0.7, 0.8, 20, 1.05,
        kw.get("mm_contents"), True, 3, True, 2,
        structured_output=kw.get("structured_output"),
    )
    seq.mm_items = kw.get("mm_items")
    seq.requested_output_len = 32
    seq.model_max_length = 4096
    return seq


_FIELDS = (
    "seq_id", "token_ids", "finish_tokens", "output_len", "ignore_eos", "temperature",
    "top_p", "top_k", "repetition_penalty", "mm_contents", "logprobs_enabled",
    "num_top_logprobs", "prompt_logprobs_enabled", "num_prompt_logprobs",
    "structured_output", "mm_items", "requested_output_len", "model_max_length",
    "raw_prompt_len", "prompt_len",
)


def _roundtrip_input(pkg):
    return wire.decode_input(wire.encode_input(pkg))


@pytest.mark.parametrize("kw", [
    {},
    {"structured_output": StructuredOutput(
        '{"type": "object"}', think_start=5, think_end=6, thinking=True,
        allow_think_start=True, prefix_tail=(9, 10), tag='{"type": "structural_tag"}')},
    {"mm_contents": {"image": ["a.png", "data:image/png;base64,AA=="], "video": ["v.mp4"]}},
    {"mm_items": [("image", "a.png"), ("video", "v.mp4")]},
])
def test_request_roundtrip_keeps_every_field(kw):
    seq = _seq(**kw)
    out = _roundtrip_input(IPCPackage([seq])).schedule_lists[0]
    for name in _FIELDS:
        assert getattr(out, name) == getattr(seq, name), name


def test_greedy_requests_still_normalize_top_k():
    seq = GenerationSequence(1, [1], [2], 4, temperature=0.0, top_k=50)
    assert _roundtrip_input(IPCPackage([seq])).schedule_lists[0].top_k == 1


@pytest.mark.parametrize("cmd", [None, "start_profile", "stop_profile",
                                 ("sleep", ("weights", "kv_cache")), ("wake_up", ())])
def test_aborts_and_control_commands(cmd):
    pkg = IPCPackage([])
    pkg.abort_ids = [4, 5]
    pkg.control_cmd = cmd
    pkg.log = False
    out = _roundtrip_input(pkg)
    assert out.abort_ids == [4, 5] and out.control_cmd == cmd and out.log is False
    assert out.schedule_lists == []


def test_output_roundtrip():
    pkg = IPCPackage([])
    pkg.act_schedule_ids = [1, 2]
    pkg.next_tokens = [11, [12, 13, 14]]  # plain and MTP rows
    pkg.logprobs = [None, (-0.5, [12, 3], [-0.5, -1.25])]
    pkg.prompt_logprobs = {2: [None, (5, -0.1, [5], [-0.1])]}
    pkg.free_ids = [2, 9]
    pkg.request_errors = {9: "too long"}
    pkg.control_ack = {"cmd": "sleep", "ok": True, "freed_bytes": 123}
    out = wire.decode_output(wire.encode_output(pkg))
    assert out.act_schedule_ids == [1, 2]
    assert out.next_tokens == [11, [12, 13, 14]]
    assert out.logprobs == [None, (-0.5, [12, 3], [-0.5, -1.25])]
    assert out.prompt_logprobs == {2: [None, (5, -0.1, [5], [-0.1])]}
    assert out.free_ids == [2, 9] and out.request_errors == {9: "too long"}
    assert out.control_ack == pkg.control_ack


def test_unencodable_media_are_refused():
    with pytest.raises(TypeError):
        wire.encode_input(IPCPackage([_seq(mm_contents={"image": [object()], "video": []})]))


def test_messages_are_plain_msgpack_arrays():
    # Any msgpack implementation can read them: no Python-specific types.
    seq = _seq()
    raw = msgspec.msgpack.decode(wire.encode_input(IPCPackage([seq])))
    assert isinstance(raw, list) and raw[0][0][0] == seq.seq_id


def test_preprocessed_media_merge_in_prompt_order():
    import torch

    def tensor(name, t, batched):
        dtype = str(t.dtype).removeprefix("torch.")
        return wire.Tensor(name, dtype, list(t.shape), batched, t.contiguous().view(torch.uint8).numpy().tobytes())

    a, b = torch.randn(8, 6, dtype=torch.bfloat16), torch.randn(4, 6, dtype=torch.bfloat16)
    v = torch.randn(4, 6, dtype=torch.bfloat16)
    items = [
        wire.MediaItem("image", tensors=[tensor("pixel_values", a, False), tensor("image_grid_thw", torch.tensor([1, 2, 4]), True)]),
        wire.MediaItem("video", tensors=[tensor("pixel_values_videos", v, False), tensor("video_grid_thw", torch.tensor([1, 2, 2]), True)]),
        wire.MediaItem("image", tensors=[tensor("pixel_values", b, False), tensor("image_grid_thw", torch.tensor([1, 2, 2]), True)]),
    ]
    req = wire.Request(3, [1, 2], [9], 8, 8, 4096, False, 0.0, 1.0, 0, 1.0, media=items)
    seq = wire.decode_input(msgspec.msgpack.encode(wire.Input([req]))).schedule_lists[0]
    pre = seq.mm_contents["preprocessed"]
    assert torch.equal(pre["pixel_values"], torch.cat([a, b]))
    assert pre["image_grid_thw"].tolist() == [[1, 2, 4], [1, 2, 2]]
    assert torch.equal(pre["pixel_values_videos"], v)
    assert pre["video_grid_thw"].tolist() == [[1, 2, 2]]


def test_arena_tensors_are_views_of_shared_memory(tmp_path, monkeypatch):
    import torch

    pixels = torch.randn(6, 4, dtype=torch.bfloat16)
    grid = torch.tensor([[1, 2, 3]])
    raw_p = pixels.view(torch.uint8).numpy().tobytes()
    raw_g = grid.view(torch.uint8).numpy().tobytes()
    arena = tmp_path / "arena"
    arena.write_bytes(b"\0" * 256 + raw_p + b"\0" * (512 - len(raw_p)) + raw_g + b"\0" * 64)
    monkeypatch.setattr(wire, "_arena", None)
    wire.map_media_arena(str(arena))
    tensors = [wire.Tensor("pixel_values", "bfloat16", [6, 4], False, b"", 256),
               wire.Tensor("image_grid_thw", "int64", [1, 3], True, b"", 768),
               wire.Tensor("patches_per_image", "int64", [1], True, b"", 768)]
    req = wire.Request(3, [1, 2], [9], 8, 8, 4096, False, 0.0, 1.0, 0, 1.0,
                       media=[wire.MediaItem("image")], media_tensors=tensors)
    pre = wire.decode_input(msgspec.msgpack.encode(wire.Input([req]))).schedule_lists[0].mm_contents["preprocessed"]
    assert set(pre) == {"pixel_values", "image_grid_thw"}
    assert torch.equal(pre["pixel_values"], pixels)
    assert pre["image_grid_thw"].tolist() == [[1, 2, 3]]
