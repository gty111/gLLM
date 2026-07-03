from collections.abc import Sequence

import torch
import torch.distributed as dist
from logger import logger


def send_pp_data(output, dst):
    if type(output) == tuple:
        assert len(output) == 2
        dist.isend(output[0], dst)
        dist.isend(output[1], dst)
    else:
        dist.isend(output, dst)


def recv_pp_data(src, num_tokens, recv_hidden_states, recv_residual, has_residual):
    if has_residual:
        dist.recv(recv_hidden_states[:num_tokens], src)
        dist.recv(recv_residual[:num_tokens], src)
    else:
        dist.recv(recv_hidden_states[:num_tokens], src)


def send_obj_list(obj_list, dst):
    dist.send_object_list(obj_list, dst=dst)


def recv_obj_list(obj_list, src):
    dist.recv_object_list(obj_list, src=src)


_RANK = 0
_PP_RANK = 0
_TP_RANK = 0
_EP_RANK = 0
_LOCAL_RANK = 0
_PP_SIZE = 1
_TP_SIZE = 1
_EP_SIZE = 1
_WORLD_SIZE = 1
# Data-parallel (DP) attention state. When ``_DP_SIZE > 1`` the engine runs
# ``_DP_SIZE`` full-model replicas (one per GPU). Each replica owns its own
# scheduler + KV cache and serves a disjoint shard of requests (round-robined
# by the frontend), so the *attention* path is pure data-parallel -- the
# compressed MLA latent KV cache is sharded across replicas instead of being
# replicated on every TP rank.
#
# The *experts* are, in contrast, sharded across all replicas with
# ``EP = DP x TP`` (the DeepSeek/SGLang MoE deployment): every replica holds
# ``1/DP`` of the routed experts. Each MoE layer therefore all-gathers the
# per-replica tokens into the global batch, runs its local expert shard, and
# all-reduces the result back before each replica slices out its own rows. To
# do that the replicas *do* form a shared NCCL communicator -- the torch
# *default* process group of size ``_DP_SIZE`` (see :func:`init_dp_ep`). gLLM's
# own TP/PP/world bookkeeping is deliberately left in single-rank mode so each
# replica still behaves as a self-contained column driver (own frontend
# sockets, no PP/TP followers); only ``_EP_*`` and the DP collectives use the
# default group.
_DP_RANK = 0
_DP_SIZE = 1
# Per-forward token counts across the DP group (one int per replica), published
# by the worker before each forward and consumed by the MoE layers to drive the
# variable-length all-gather. ``None`` outside a driven forward (profile /
# graph-warmup), where the MoE falls back to a symmetric gather.
_DP_FWD_COUNTS = None
_ASSIGNED_LAYERS = None
_TP_GROUP = None
# Dedicated TP communicator for the per-iter IPC broadcast (rank 0 ->
# PP=0 TP peers). It covers the same ranks as ``_TP_GROUP`` but is a
# *separate* NCCL communicator so its kernels don't queue behind the
# forward path's all_reduces. Empirically, sharing a communicator
# pushed each broadcast to ~6 ms (the .item() readback waited for the
# previous iter's per-layer ARs to drain on the same NCCL stream); on
# a dedicated group the same broadcast finishes in ~10-20 us.
_IPC_TP_GROUP = None
# DP process group: the ranks that share this replica's ``tp_rank`` across all
# DP groups (``{d*tp_size + tp_rank : d in range(dp_size)}``). Used by the MoE
# to all-gather tokens across the DP dimension. ``None`` outside DP.
_DP_GROUP = None
# EP process group: the routed experts are sharded across the ranks of a single
# pipeline stage (``dp_size * tp_size`` ranks). For ``pp_size == 1`` this is the
# whole world; :func:`ep_all_reduce` sums the expert shards over it. ``None``
# outside DP+EP (the non-DP MoE path reduces over the TP group instead).
_EP_GROUP = None
_USE_EP = True


def get_rank():
    return _RANK


def get_world_size():
    return _WORLD_SIZE


def get_pp_rank():
    return _PP_RANK


def get_tp_rank():
    return _TP_RANK


def get_ep_rank():
    return _EP_RANK


def get_dp_rank():
    return _DP_RANK


def get_dp_size():
    return _DP_SIZE


def get_dp_group():
    return _DP_GROUP


def get_ep_group():
    return _EP_GROUP


def is_dp_attn():
    """True when DP-attention (independent per-GPU replicas) is enabled."""
    return _DP_SIZE > 1


def set_dp_info(dp_rank, dp_size, local_rank=None):
    """Record this replica's DP rank/size (bookkeeping/logging only).

    Used on the ``dp_size == 1`` path (and as a fallback) to record
    ``local_rank`` in dist_utils: with ``pp_size == tp_size == 1`` the engine
    never calls :func:`init_dist`, so ``_LOCAL_RANK`` would otherwise stay ``0``
    on every process. Several call sites key off ``get_local_rank()`` -- most
    importantly the weight-load progress array (indexed ``rank*2`` /
    ``rank*2+1``), which would collide onto slot 0 and hang the frontend's
    ``load_progress`` wait. The ``dp_size > 1`` path uses :func:`init_dp_ep`
    instead, which records the same state and additionally forms the NCCL group.
    """
    global _DP_RANK, _DP_SIZE, _LOCAL_RANK
    _DP_RANK = dp_rank
    _DP_SIZE = dp_size
    if local_rank is not None:
        _LOCAL_RANK = local_rank


def init_dp_ep(
    pp_size,
    dp_size,
    tp_size,
    pp_rank,
    dp_rank,
    tp_rank,
    use_ep,
    local_rank,
    master_addr,
    master_port,
    assigned_layers=None,
):
    """Bring up (PP x) DP-attention + Expert-Parallel (``EP = DP x TP``).

    Layout: ``world = pp_size * dp_size * tp_size`` ranks arranged as a
    ``pp x dp x tp`` grid with

        ``global_rank = pp_rank * S + dp_rank * tp_size + tp_rank``,  ``S = dp_size * tp_size``.

    ``S`` is the *stage size*: within one pipeline stage the ``dp x tp`` grid is
    exactly the single-stage DP+EP layout, offset by ``pp_rank * S``. Four group
    families are formed (every rank calls ``new_group`` the same number of times
    in the same order, since all iterate the identical nested loops):

    * **TP subgroups** -- fix ``(pp, dp)``, vary ``tp``. Standard tensor
      parallelism *within* a DP group at one stage (attention heads,
      embedding/lm-head, o_proj/down_proj all-reduce). The MLA latent KV is
      replicated across these ``tp_size`` ranks and sharded across DP groups.
    * **DP subgroups** -- fix ``(pp, tp)``, vary ``dp``. Used by the MoE to
      all-gather tokens across the DP dimension *within a stage*
      (:func:`dp_gather_hidden`, :func:`dp_all_gather_meta`).
    * **EP groups** -- fix ``pp``, all ``(dp, tp)`` (the ``S`` ranks of one
      stage). The routed experts are sharded across a stage's ``S`` ranks;
      :func:`ep_all_reduce` sums the shards over :func:`get_ep_group`.
    * **PP** -- adjacent stages talk via ``rank +/- S`` (same ``(dp, tp)``),
      see :func:`get_next_pp_rank` / :func:`get_last_pp_rank`.

    For ``pp_size == 1`` this reduces to the single-stage DP+EP layout (EP group
    == whole world). For ``tp_size == 1`` the TP subgroups are singletons.
    """
    global _DP_RANK, _DP_SIZE, _DP_GROUP, _EP_RANK, _EP_SIZE, _EP_GROUP, _USE_EP
    global _RANK, _WORLD_SIZE, _TP_SIZE, _PP_SIZE, _TP_RANK, _PP_RANK, _LOCAL_RANK
    global _TP_GROUP, _IPC_TP_GROUP, _ASSIGNED_LAYERS

    stage_size = dp_size * tp_size
    world_size = pp_size * stage_size
    global_rank = pp_rank * stage_size + dp_rank * tp_size + tp_rank

    _DP_RANK = dp_rank
    _DP_SIZE = dp_size
    _TP_SIZE = tp_size
    _TP_RANK = tp_rank
    _PP_SIZE = pp_size
    _PP_RANK = pp_rank
    _RANK = global_rank
    _WORLD_SIZE = world_size
    _USE_EP = use_ep
    _ASSIGNED_LAYERS = assigned_layers
    # EP spans one pipeline stage (EP = DP x TP); each rank owns 1/S experts.
    _EP_SIZE = stage_size if use_ep else tp_size
    _EP_RANK = (dp_rank * tp_size + tp_rank) if use_ep else tp_rank
    _LOCAL_RANK = local_rank

    init_method = f"tcp://{master_addr}:{master_port}"
    logger.info(
        f"NCCL(DP+EP): Init_method {init_method}, Rank {global_rank}, "
        f"PP {pp_rank}/{pp_size}, DP {dp_rank}/{dp_size}, TP {tp_rank}/{tp_size}, "
        f"EP size {_EP_SIZE}, World_size {world_size}"
    )
    dist.init_process_group(
        init_method=init_method,
        backend="nccl",
        world_size=world_size,
        rank=global_rank,
    )

    def _stage_base(pp):
        return pp * stage_size

    # TP subgroups (attention / embedding / lm-head + a separate IPC
    # communicator for the input broadcast; see ``_IPC_TP_GROUP``).
    tp_groups = [
        [_stage_base(pp) + d * tp_size + t for t in range(tp_size)]
        for pp in range(pp_size)
        for d in range(dp_size)
    ]
    for ranks in tp_groups:
        g = dist.new_group(ranks)
        if global_rank in ranks:
            _TP_GROUP = g
    for ranks in tp_groups:
        g = dist.new_group(ranks)
        if global_rank in ranks:
            _IPC_TP_GROUP = g

    # DP subgroups (MoE token all-gather across the DP dimension, per stage).
    dp_groups = [
        [_stage_base(pp) + d * tp_size + t for d in range(dp_size)]
        for pp in range(pp_size)
        for t in range(tp_size)
    ]
    for ranks in dp_groups:
        g = dist.new_group(ranks)
        if global_rank in ranks:
            _DP_GROUP = g

    # EP groups (routed-expert all-reduce, one per pipeline stage).
    ep_groups = [
        [_stage_base(pp) + i for i in range(stage_size)] for pp in range(pp_size)
    ]
    for ranks in ep_groups:
        g = dist.new_group(ranks)
        if global_rank in ranks:
            _EP_GROUP = g


def set_dp_forward_counts(counts):
    """Publish this forward's per-replica token counts (see ``_DP_FWD_COUNTS``)."""
    global _DP_FWD_COUNTS
    _DP_FWD_COUNTS = counts


def get_dp_forward_counts():
    return _DP_FWD_COUNTS


def dp_all_gather_num_tokens(local_ntok: int):
    """All-gather each replica's forward token count over the DP group.

    Called once per iteration by every replica (an unconditional barrier that
    keeps the replicas in lockstep). Returns the per-replica counts as a list.
    """
    t = torch.tensor([local_ntok], dtype=torch.long, device="cuda")
    out = torch.empty(_DP_SIZE, dtype=torch.long, device="cuda")
    dist.all_gather_into_tensor(out, t, group=_DP_GROUP)
    return out.tolist()


def dp_all_gather_meta(local_ntok: int, is_decode: bool):
    """All-gather each DP group's ``(token count, is_decode)`` this iteration.

    A single fixed-size collective (one row per DP group) that doubles as the
    per-iter lockstep barrier. ``is_decode`` lets the driver decide whether the
    whole world can take the CUDA-graph path (only when *every* group is a pure
    decode / idle-dummy step; any prefill forces the eager, variable-length
    gather). Returns ``(counts, decode_flags)`` as two lists of length
    ``dp_size``.
    """
    t = torch.tensor(
        [local_ntok, 1 if is_decode else 0], dtype=torch.long, device="cuda"
    )
    out = torch.empty(_DP_SIZE, 2, dtype=torch.long, device="cuda")
    dist.all_gather_into_tensor(out, t, group=_DP_GROUP)
    out = out.tolist()
    counts = [row[0] for row in out]
    decode_flags = [row[1] for row in out]
    return counts, decode_flags


def _dp_counts_uniform(counts) -> bool:
    return all(c == counts[0] for c in counts)


def dp_gather_hidden(x: torch.Tensor, counts) -> torch.Tensor:
    """Gather per-replica token rows into the global batch.

    ``x`` is this replica's ``[counts[dp_rank], H]`` hidden states. Two layouts:

    * **Uniform** (``counts`` all equal -- SGLang's MAX_LEN mode, used on the
      CUDA-graph decode path where every group is padded to a common bucket):
      the fixed-size all-gather output *is* the contiguous global batch
      ``[dp_size*B, H]`` already, so we return it directly. No ``torch.cat`` ->
      no data-dependent allocation, so this is safe to capture in a CUDA graph.
    * **Ragged** (variable ``counts`` -- eager prefill / mixed): pad to the max,
      all-gather, then concatenate the real slices into ``[sum(counts), H]``.
    """
    max_n = max(counts)
    hidden = x.shape[1]
    padded = x.new_zeros((max_n, hidden))
    padded[: x.shape[0]] = x
    gathered = x.new_empty((_DP_SIZE * max_n, hidden))
    dist.all_gather_into_tensor(gathered, padded, group=_DP_GROUP)
    if _dp_counts_uniform(counts):
        return gathered
    gathered = gathered.view(_DP_SIZE, max_n, hidden)
    parts = [gathered[d, : counts[d]] for d in range(_DP_SIZE)]
    return torch.cat(parts, dim=0)


def dp_local_slice(x_global: torch.Tensor, counts) -> torch.Tensor:
    """Slice this replica's own rows back out of the global batch.

    Mirrors :func:`dp_gather_hidden`: for the uniform layout each group owns a
    contiguous ``B``-row block at ``dp_rank*B``; for the ragged layout the
    offset is the prefix sum of ``counts``.
    """
    if _dp_counts_uniform(counts):
        block = counts[0]
        start = _DP_RANK * block
        return x_global[start : start + block]
    start = sum(counts[:_DP_RANK])
    end = start + counts[_DP_RANK]
    return x_global[start:end]


def ep_all_reduce(x: torch.Tensor) -> torch.Tensor:
    """Sum expert-shard contributions across the EP group (one pipeline stage).

    EP spans the ``dp_size * tp_size`` ranks of a single stage (``_EP_GROUP``).
    For ``pp_size == 1`` that is the whole world.
    """
    dist.all_reduce(x, group=_EP_GROUP)
    return x


def get_local_rank():
    return _LOCAL_RANK


def get_output_rank():
    return (get_pp_size() - 1) * get_tp_size()


def is_output_rank():
    return is_last_pp_rank() and is_first_tp_rank()


def is_first_pp_rank():
    return get_pp_rank() == 0


def is_first_tp_rank():
    return get_tp_rank() == 0


def is_last_pp_rank():
    return get_pp_rank() == get_pp_size() - 1


def is_use_ep():
    return _USE_EP


def _pp_stage_size():
    """Ranks per pipeline stage: ``dp_size * tp_size`` (``tp_size`` when no DP)."""
    return get_dp_size() * get_tp_size()


def get_next_pp_rank():
    return get_rank() + _pp_stage_size()


def get_last_pp_rank():
    return get_rank() - _pp_stage_size()


def get_pp_size():
    return _PP_SIZE


def get_tp_size():
    return _TP_SIZE


def get_ep_size():
    return _EP_SIZE


def get_assigned_layers():
    return _ASSIGNED_LAYERS


def get_tp_group():
    return _TP_GROUP


def get_ipc_tp_group():
    """Process group used by :meth:`zmqComm.broadcast_input_to_tp`.

    Same ranks as :func:`get_tp_group` but a different NCCL
    communicator -- see ``_IPC_TP_GROUP`` for why this matters.
    """
    return _IPC_TP_GROUP


def init_tp_group():
    global _TP_GROUP, _IPC_TP_GROUP
    tp_groups = [
        list(range(_pp_rank * get_tp_size(), (_pp_rank + 1) * get_tp_size()))
        for _pp_rank in range(get_pp_size())
    ]
    # Two passes so that ``dist.new_group`` is called the same number of
    # times on every rank (it is collective). Order matters: forward AR
    # uses _TP_GROUP, IPC broadcast uses _IPC_TP_GROUP.
    for tp_ranks in tp_groups:
        tp_group = dist.new_group(tp_ranks)
        if _RANK in tp_ranks:
            _TP_GROUP = tp_group
    for tp_ranks in tp_groups:
        ipc_group = dist.new_group(tp_ranks)
        if _RANK in tp_ranks:
            _IPC_TP_GROUP = ipc_group


def init_dist(
    pp_size,
    tp_size,
    use_ep,
    local_rank,
    pp_rank,
    tp_rank,
    master_addr,
    master_port,
    assigned_layers,
):
    global _RANK, _PP_RANK, _TP_RANK, _PP_SIZE, _TP_SIZE, _WORLD_SIZE, _ASSIGNED_LAYERS, _LOCAL_RANK, _TP_GROUP
    global _EP_SIZE, _EP_RANK, _USE_EP
    _RANK = pp_rank * tp_size + tp_rank
    _PP_RANK = pp_rank
    _TP_RANK = tp_rank
    _EP_RANK = _TP_RANK if use_ep else 0
    _LOCAL_RANK = local_rank
    _PP_SIZE = pp_size
    _TP_SIZE = tp_size
    _EP_SIZE = _TP_SIZE if use_ep else 1
    _USE_EP = use_ep
    _WORLD_SIZE = pp_size * tp_size
    _ASSIGNED_LAYERS = assigned_layers

    self_tp_ranks = list(range(pp_rank * tp_size, (pp_rank + 1) * tp_size))

    init_method = f"tcp://{master_addr}:{master_port}"
    backend = "nccl"
    tp_ep_log = "TP Groups" if not use_ep or tp_size == 1 else "TP/EP Groups"
    logger.info(
        f"NCCL: Init_method {init_method}, Backend {backend}, Rank {_RANK}, {tp_ep_log} {self_tp_ranks}, Word_size {_WORLD_SIZE}"
    )
    dist.init_process_group(
        init_method=init_method, backend=backend, world_size=_WORLD_SIZE, rank=_RANK
    )

    init_tp_group()


def get_pp_layers(num_layers):
    if _ASSIGNED_LAYERS is None:
        num_layers_pp = num_layers // get_pp_size()

        if num_layers % get_pp_size() != 0:
            num_layers_pp += 1

        if get_pp_rank() != get_pp_size() - 1:
            assigned_layers = num_layers_pp * get_pp_rank(), num_layers_pp * (
                get_pp_rank() + 1
            )
        else:
            assigned_layers = num_layers_pp * get_pp_rank(), num_layers
    else:
        total_assigned_layers = [int(i) for i in _ASSIGNED_LAYERS.split(",")]
        assert (
            len(total_assigned_layers) == get_pp_size()
            and sum(total_assigned_layers) == num_layers
        )
        assigned_layers = [
            sum(total_assigned_layers[: get_pp_rank()]),
            sum(total_assigned_layers[: get_pp_rank() + 1]),
        ]

    if get_pp_size() > 1:
        logger.info(
            "Assigned %2d layers: (%3d,%3d)"
            % (
                assigned_layers[1] - assigned_layers[0],
                assigned_layers[0],
                assigned_layers[1] - 1,
            )
        )

    return assigned_layers


# Set the correct layer index for PP
def resolve_pp_layer_idx(layer_name, idx, start_layer_idx):
    if "layers" in layer_name:
        layer_name_list = layer_name.split(".")
        layer_name_list[idx] = str(int(layer_name_list[idx]) + start_layer_idx)
        return ".".join(layer_name_list)
    else:
        return layer_name


def resolve_ep_expert_idx(expert_idx, expert_map):
    if expert_map is not None:
        local_expert_idx = expert_map[expert_idx]
    else:
        local_expert_idx = expert_idx
    return local_expert_idx


def tensor_model_parallel_all_gather(input_: torch.Tensor, dim=-1) -> torch.Tensor:
    """All-gather the input tensor across model parallel group."""
    if dim < 0:
        # Convert negative dim to positive.
        dim += input_.dim()
    input_size = input_.size()
    # NOTE: we have to use concat-style all-gather here,
    # stack-style all-gather has compatibility issues with
    # torch.compile . see https://github.com/pytorch/pytorch/issues/138795
    output_size = (input_size[0] * get_tp_size(),) + input_size[1:]
    # Allocate output tensor.
    output_tensor = torch.empty(output_size, dtype=input_.dtype, device=input_.device)
    # All-gather.
    dist.all_gather_into_tensor(output_tensor, input_, group=get_tp_group())
    # Reshape
    output_tensor = output_tensor.reshape((get_tp_size(),) + input_size)
    output_tensor = output_tensor.movedim(0, dim)
    output_tensor = output_tensor.reshape(
        input_size[:dim] + (get_tp_size() * input_size[dim],) + input_size[dim + 1 :]
    )
    return output_tensor


def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce ``input_`` across the TP group.

    Fast path: when an NVLink-P2P-capable ``CustomAllreduce`` has been
    initialised and the message fits its registered staging window, route
    through ``sgl_kernel``'s two-shot AR kernel (~3-5x lower latency than
    NCCL's RING_LL for the <1 MB / small-batch decode regime on H100).

    Fall back to ``dist.all_reduce`` otherwise (custom AR disabled, too
    large a tensor, non-contiguous input, etc.).

    Return semantics: callers must use the returned tensor; the custom-AR
    path may return a buffer distinct from ``input_`` (out-of-place
    kernel). Every existing call site already obeys this contract
    (``output = tensor_model_parallel_all_reduce(...)``), so we deliberately
    *do not* mirror the result back into ``input_``. The previous
    ``input_.copy_(out)`` write-back was issuing one ``memcpy32_post`` per
    AR (~31 ms / 1.7 s total GPU on a 60-prompt decode-heavy profile -- a
    pure 2 % waste with no semantic benefit; SGLang doesn't do the copy
    either, which is exactly the source of its memcpy32_post=0 in our
    side-by-side trace comparison).
    """
    # Import lazily to avoid a circular import at module init (dist_utils is
    # imported by gllm.distributed.cuda_wrapper transitively via ``logger``).
    from gllm.distributed import get_custom_allreduce

    car = get_custom_allreduce()
    if car is not None and car.should_custom_ar(input_):
        return car.all_reduce(input_)
    dist.all_reduce(input_, group=get_tp_group())
    return input_


def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, "{} is not divisible by {}".format(
        numerator, denominator
    )


def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


def split_tensor_along_last_dim(
    tensor: torch.Tensor,
    num_partitions: int,
    contiguous_split_chunks: bool = False,
) -> Sequence[torch.Tensor]:
    """Split a tensor along its last dimension.

    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.

    Returns:
        A list of Tensors
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    last_dim_size = divide(tensor.size()[last_dim], num_partitions)
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # NOTE: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list
