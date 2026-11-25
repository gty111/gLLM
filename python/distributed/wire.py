"""Front end <-> worker wire format (msgpack).

The front end and the driver worker of each DP replica exchange these messages
over ZMQ instead of pickled Python objects, so a front end written in another
language can drive the workers. Everything past the driver's receive point
(scheduler, TP/PP fan-out) keeps working on :class:`GenerationSequence` and
:class:`IPCPackage`; the conversions live here.

Structs are encoded as msgpack arrays in field order. Append new fields at the
end with a default so older senders stay decodable.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import msgspec

from gllm.runtime.sequence import GenerationSequence


class Structured(msgspec.Struct, array_like=True):
    schema: Optional[str]
    tag: Optional[str] = None
    think_start: Optional[int] = None
    think_end: Optional[int] = None
    thinking: bool = False
    allow_think_start: bool = False
    prefix_tail: List[int] = []


class Tensor(msgspec.Struct, array_like=True):
    name: str  # processor output name, e.g. pixel_values / image_grid_thw
    dtype: str  # torch dtype name without the "torch." prefix
    shape: List[int]
    # True: one row per item (stacked across items); False: concatenated on dim 0.
    batched: bool
    data: bytes
    # Set when the bytes live in the media arena (shared memory) instead of
    # ``data``: their offset there. The tensor is a view, not a copy.
    offset: Optional[int] = None


class MediaItem(msgspec.Struct, array_like=True):
    modality: str  # "image" | "video"
    # A URL, data URL or local path the worker's processor loads itself.
    ref: Optional[str] = None
    # Or the processor outputs, already computed by the front end (which then
    # sends a prompt with the placeholders expanded).
    tensors: Optional[List[Tensor]] = None


class Request(msgspec.Struct, array_like=True):
    seq_id: int
    token_ids: List[int]
    finish_tokens: List[int]
    output_len: int
    requested_output_len: Optional[int]
    model_max_length: int
    ignore_eos: bool
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float
    logprobs: Optional[int] = None  # number of top logprobs; None = off
    prompt_logprobs: Optional[int] = None
    structured: Optional[Structured] = None
    media: Optional[List[MediaItem]] = None
    # Encoder disaggregation: prompt-ordered raw items for the encoder; the
    # token ids are then a skeleton with one sentinel per item.
    disagg_items: Optional[List[MediaItem]] = None
    # Processor outputs of all ``media`` items, already merged per name in
    # prompt order (the items then carry only their modality).
    media_tensors: Optional[List[Tensor]] = None


class Control(msgspec.Struct, array_like=True):
    cmd: str  # start_profile | stop_profile | sleep | wake_up
    tags: Optional[List[str]] = None


class Input(msgspec.Struct, array_like=True):
    requests: List[Request] = []
    aborts: List[int] = []
    control: Optional[Control] = None
    log: Optional[bool] = None


LogprobRow = Tuple[float, List[int], List[float]]  # sampled, top ids, top values
PromptLogprob = Tuple[int, float, List[int], List[float]]


class Output(msgspec.Struct, array_like=True):
    seq_ids: List[int] = []
    tokens: List[Union[int, List[int]]] = []  # a list when MTP commits several
    logprobs: List[Optional[LogprobRow]] = []
    prompt_logprobs: Dict[int, List[Optional[PromptLogprob]]] = {}
    finished: List[int] = []
    errors: Dict[int, str] = {}
    control_ack: Optional[Dict[str, Any]] = None
    error_codes: Dict[int, str] = {}  # absent: the request exceeds cache capacity


_encoder = msgspec.msgpack.Encoder()
_input_decoder = msgspec.msgpack.Decoder(Input)
_output_decoder = msgspec.msgpack.Decoder(Output)


# ---------------------------------------------------------------------------
# Front end -> worker
# ---------------------------------------------------------------------------
def _media_items(mm_contents) -> Optional[List[MediaItem]]:
    if mm_contents is None:
        return None
    if mm_contents.get("preprocessed") is not None:
        raise NotImplementedError("preprocessed media need the media arena")
    items = []
    for modality in ("image", "video"):
        for ref in mm_contents.get(modality, ()):
            if not isinstance(ref, str):
                raise TypeError(f"{modality} inputs must be URLs or paths, got {type(ref).__name__}")
            items.append(MediaItem(modality, ref))
    return items


def request_from_seq(seq: GenerationSequence) -> Request:
    so = seq.structured_output
    structured = None
    if so is not None:
        structured = Structured(
            so.schema, so.tag, so.think_start, so.think_end, so.thinking,
            so.allow_think_start, list(so.prefix_tail),
        )
    mm_items = getattr(seq, "mm_items", None)
    return Request(
        seq_id=seq.seq_id,
        token_ids=list(seq.token_ids),
        finish_tokens=list(seq.finish_tokens),
        output_len=seq.output_len,
        requested_output_len=getattr(seq, "requested_output_len", None),
        model_max_length=getattr(seq, "model_max_length", 0),
        ignore_eos=seq.ignore_eos,
        temperature=seq.temperature,
        top_p=seq.top_p,
        top_k=seq.top_k,
        repetition_penalty=seq.repetition_penalty,
        logprobs=seq.num_top_logprobs if seq.logprobs_enabled else None,
        prompt_logprobs=seq.num_prompt_logprobs if seq.prompt_logprobs_enabled else None,
        structured=structured,
        media=_media_items(seq.mm_contents),
        disagg_items=(
            [MediaItem(modality, ref) for modality, ref in mm_items] if mm_items else None
        ),
    )


_TORCH_DTYPES = {
    "float32": "float32", "bfloat16": "bfloat16", "float16": "float16",
    "int64": "int64", "int32": "int32", "uint8": "uint8", "bool": "bool",
}


_arena = None


def map_media_arena(path: str) -> None:
    """Map the front end's shared-memory media arena into this process.

    Driver workers call it during startup with ``engine_server --media-arena``:
    the front end's open descriptor (``/proc/<pid>/fd/<fd>``), since the file
    itself is unlinked at creation so a crash cannot leave it behind.
    """
    global _arena
    import mmap

    with open(path, "r+b") as f:
        _arena = mmap.mmap(f.fileno(), 0)


def media_arena():
    if _arena is None:
        raise ValueError("media tensors reference an arena, but none is mapped")
    return _arena


def _tensor(t: Tensor):
    import math

    import torch

    name = _TORCH_DTYPES.get(t.dtype)
    if name is None:
        raise ValueError(f"unsupported tensor dtype {t.dtype}")
    dtype = getattr(torch, name)
    if t.offset is not None:
        count = math.prod(t.shape)
        if count == 0:
            return torch.empty(t.shape, dtype=dtype)
        # A view of shared memory; the front end keeps it until the request ends.
        return torch.frombuffer(media_arena(), dtype=dtype, count=count, offset=t.offset).reshape(t.shape)
    if not t.data:
        return torch.empty(t.shape, dtype=dtype)
    # A writable copy: the receive buffer is read-only.
    return torch.frombuffer(bytearray(t.data), dtype=dtype).reshape(t.shape)


# Processor outputs the vision towers consume; others (front-end bookkeeping
# such as per-item patch counts) are dropped.
_MODEL_INPUTS = frozenset({
    "pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw",
    "second_per_grid_ts",
})


def _preprocessed(items: List[MediaItem]) -> Dict[str, Any]:
    """Merge per-item processor outputs into one batch, in prompt order."""
    import torch

    parts: Dict[str, list] = {}
    batched: Dict[str, bool] = {}
    for item in items:
        if not item.tensors:
            raise ValueError("a request cannot mix preprocessed and raw media")
        for t in item.tensors:
            if t.name not in _MODEL_INPUTS:
                continue
            parts.setdefault(t.name, []).append(_tensor(t))
            batched[t.name] = t.batched
    return {
        name: torch.stack(ts) if batched[name] else torch.cat(ts, dim=0)
        for name, ts in parts.items()
    }


def seq_from_request(req: Request) -> GenerationSequence:
    structured_output = None
    if req.structured is not None:
        from gllm.structured_output import StructuredOutput

        s = req.structured
        structured_output = StructuredOutput(
            s.schema, think_start=s.think_start, think_end=s.think_end,
            thinking=s.thinking, allow_think_start=s.allow_think_start,
            prefix_tail=tuple(s.prefix_tail), tag=s.tag,
        )
    mm_contents = None
    if req.media:
        mm_contents = {"image": [], "video": []}
        if req.media_tensors:
            mm_contents["preprocessed"] = {
                t.name: _tensor(t) for t in req.media_tensors if t.name in _MODEL_INPUTS
            }
        elif any(item.tensors for item in req.media):
            mm_contents["preprocessed"] = _preprocessed(req.media)
        else:
            for item in req.media:
                mm_contents[item.modality].append(item.ref)
    seq = GenerationSequence(
        req.seq_id,
        req.token_ids,
        req.finish_tokens,
        req.output_len,
        req.ignore_eos,
        req.temperature,
        req.top_p,
        req.top_k,
        req.repetition_penalty,
        mm_contents,
        req.logprobs is not None,
        req.logprobs or 0,
        req.prompt_logprobs is not None,
        req.prompt_logprobs or 0,
        structured_output=structured_output,
    )
    seq.mm_items = (
        [(item.modality, item.ref) for item in req.disagg_items] if req.disagg_items else None
    )
    seq.requested_output_len = req.requested_output_len
    seq.model_max_length = req.model_max_length
    return seq


def encode_input(pkg) -> bytes:
    """Encode a front-end :class:`IPCPackage`."""
    control = None
    cmd = pkg.control_cmd
    if isinstance(cmd, tuple):  # sleep mode: (name, tags)
        control = Control(cmd[0], list(cmd[1]) if cmd[1] is not None else None)
    elif cmd is not None:
        control = Control(cmd)
    return _encoder.encode(Input(
        [request_from_seq(seq) for seq in pkg.schedule_lists],
        list(pkg.abort_ids),
        control,
        pkg.log,
    ))


def decode_input(data: bytes):
    """Decode into the :class:`IPCPackage` the driver worker consumes."""
    from gllm.distributed.comm import IPCPackage

    msg = _input_decoder.decode(data)
    pkg = IPCPackage([seq_from_request(r) for r in msg.requests])
    pkg.abort_ids = msg.aborts
    pkg.log = msg.log
    if msg.control is not None:
        c = msg.control
        pkg.control_cmd = (c.cmd, tuple(c.tags or ())) if c.cmd in ("sleep", "wake_up") else c.cmd
    return pkg


# ---------------------------------------------------------------------------
# Worker -> front end
# ---------------------------------------------------------------------------
def encode_output(pkg) -> bytes:
    """Encode a worker :class:`IPCPackage`."""
    return _encoder.encode(Output(
        pkg.act_schedule_ids,
        pkg.next_tokens,
        pkg.logprobs,
        pkg.prompt_logprobs,
        pkg.free_ids,
        getattr(pkg, "request_errors", None) or {},
        pkg.control_ack,
        getattr(pkg, "request_error_codes", None) or {},
    ))


def decode_output(data: bytes):
    from gllm.distributed.comm import IPCPackage

    msg = _output_decoder.decode(data)
    pkg = IPCPackage([])
    pkg.act_schedule_ids = msg.seq_ids
    pkg.next_tokens = msg.tokens
    pkg.logprobs = msg.logprobs
    pkg.prompt_logprobs = msg.prompt_logprobs
    pkg.free_ids = msg.finished
    pkg.request_errors = msg.errors
    pkg.control_ack = msg.control_ack
    pkg.request_error_codes = msg.error_codes
    return pkg
