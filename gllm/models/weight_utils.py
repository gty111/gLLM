# Resolve weights loading for TP

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Callable, Optional

from gllm.dist_utils import (
    get_tp_rank,
    get_tp_size,
    is_use_ep,
    resolve_ep_expert_idx,
)


def copy_qkv_proj(dst_qkv, src_q, src_k, src_v, num_heads, num_kv_heads, head_dim):
    dst_qkv[: num_heads * head_dim] = src_q[
        get_tp_rank()
        * num_heads
        * head_dim : (get_tp_rank() + 1)
        * num_heads
        * head_dim
    ]
    dst_qkv[num_heads * head_dim : (num_heads + num_kv_heads) * head_dim] = src_k[
        get_tp_rank()
        * num_kv_heads
        * head_dim : (get_tp_rank() + 1)
        * num_kv_heads
        * head_dim
    ]
    dst_qkv[(num_heads + num_kv_heads) * head_dim :] = src_v[
        get_tp_rank()
        * num_kv_heads
        * head_dim : (get_tp_rank() + 1)
        * num_kv_heads
        * head_dim
    ]


def copy_qkv_a_proj(dst, q_a_proj, kv_a_proj_with_mqa):
    size_partition = q_a_proj.shape[0]
    dst[:size_partition] = q_a_proj
    dst[size_partition:] = kv_a_proj_with_mqa


def copy_gate_up_proj(dst, src_gate, src_up, partition_tp=True):
    size_partition = dst.shape[0] // 2
    if partition_tp:
        dst[:size_partition] = src_gate[
            get_tp_rank() * size_partition : (get_tp_rank() + 1) * size_partition
        ]
        dst[size_partition:] = src_up[
            get_tp_rank() * size_partition : (get_tp_rank() + 1) * size_partition
        ]
    else:
        dst[:size_partition] = src_gate
        dst[size_partition:] = src_up


def copy_single_proj_dim1(dst, src, partition_tp=True):
    # partition on column
    if partition_tp:
        size_partition = dst.shape[1]
        dst.copy_(
            src[
                :, get_tp_rank() * size_partition : (get_tp_rank() + 1) * size_partition
            ]
        )
    else:
        dst.copy_(src)


def copy_single_proj_dim0(dst, src):
    # partition on row
    size_partition = dst.shape[0]
    dst.copy_(
        src[get_tp_rank() * size_partition : (get_tp_rank() + 1) * size_partition]
    )
    

def get_tensor_from_dict(weights, k):
    k_language = k.replace('model', 'model.language_model')
    k_visual = k.replace('visual', 'model.visual')
    if k in weights:
        return weights[k]
    elif k_language in weights:
        return weights[k_language]
    elif k_visual in weights:
        return weights[k_visual]
    else:
        for key, value in weights.items():
            print(key)
        
        raise KeyError(f"Fail to extract {k} from weights")


# ---------------------------------------------------------------------------
# MoE expert-weight loaders
# ---------------------------------------------------------------------------
#
# Two checkpoint layouts are supported:
#
#   * "per-expert" — every expert is stored as a separate tensor in the
#     checkpoint, e.g. ``layers.0.mlp.experts.{i}.gate_proj.weight`` for
#     ``i in [0, num_experts)``. Used by Qwen2/3-MoE, DeepSeek-V2/V3, and
#     Mixtral (Mixtral uses ``w1``/``w2``/``w3`` instead of
#     ``gate_proj``/``down_proj``/``up_proj``).
#
#   * "stacked" — all experts are pre-stacked into one tensor in the
#     checkpoint, e.g. shape ``(E, H, 2I)`` for gate_up and ``(E, I, H)`` for
#     down. Used by Qwen3-VL-MoE's ``gate_up_proj`` / ``down_proj``.
#
# For the per-expert layout the source tensors live in different mmap'd
# offsets so loading the fused MoE weight inherently requires ``num_experts``
# small CPU->GPU copies per layer. The Python+CUDA per-call overhead
# (typically ~3-5 ms each) dominates load time when ``num_experts`` is large
# (Qwen3-30B-A3B has 128 experts x 48 layers => >12k copies). PyTorch
# releases the GIL during ``.copy_()``, so a small ThreadPool lets multiple
# H2D transfers be in-flight on the PCIe bus simultaneously and cuts the
# load-weights phase by ~25-30x on Qwen3-30B-A3B with TP=4.
#
# For the stacked layout, the EP-off + TP>1 path can collapse the per-expert
# loop into ``O(1)`` bulk slice + permute + H2D copies, which is even faster
# than threading because there is no per-call overhead at all. EP-on (each
# rank owns a subset of experts) still needs a per-expert loop, so we thread
# that path too.

_MOE_LOAD_MAX_WORKERS = 8


@contextmanager
def moe_expert_load_pool(num_experts: int):
    """Context-managed ThreadPool for parallelizing per-expert MoE H2D copies.

    Usage::

        with moe_expert_load_pool(num_experts) as pool:
            for k, v in parameters.items():
                ...
                if k.find("w13_weight") != -1:
                    load_fused_w13_per_expert(v.data, weights, ..., pool=pool)

    Sized as ``min(_MOE_LOAD_MAX_WORKERS, num_experts)``. Empirically 8
    threads is enough to saturate PCIe Gen4 with bf16 expert tensors; beyond
    that the GIL contention starts to hurt.
    """
    pool = ThreadPoolExecutor(
        max_workers=min(_MOE_LOAD_MAX_WORKERS, max(1, num_experts)),
        thread_name_prefix="moe-load",
    )
    try:
        yield pool
    finally:
        pool.shutdown(wait=True)


def _iter_experts(num_experts: int, body: Callable[[int], None],
                  pool: Optional[ThreadPoolExecutor]):
    """Run ``body(expert_idx)`` for every expert, optionally in parallel."""
    if pool is None:
        for i in range(num_experts):
            body(i)
    else:
        # ``list(...)`` forces materialization so exceptions surface here.
        list(pool.map(body, range(num_experts)))


def load_fused_w13_per_expert(
    dst,
    weights,
    key_for_gate: Callable[[int], str],
    key_for_up: Callable[[int], str],
    expert_map,
    num_experts: int,
    pool: Optional[ThreadPoolExecutor] = None,
):
    """Load the fused ``w13_weight`` (gate ++ up) MoE parameter from a
    per-expert checkpoint.

    Args:
        dst: GPU param tensor of shape
            ``(local_num_experts, 2 * inter_per_partition, hidden)``.
        weights: dict of source CPU tensors.
        key_for_gate: ``expert_idx -> source key`` for the gate projection.
        key_for_up: ``expert_idx -> source key`` for the up projection.
        expert_map: EP expert map (``None`` when EP is disabled).
        num_experts: global expert count.
        pool: optional pool from :func:`moe_expert_load_pool`. When provided
            the per-expert copies run concurrently; otherwise they run
            serially.
    """
    partition_tp = not is_use_ep()

    def _one(expert_idx: int):
        local_expert_idx = resolve_ep_expert_idx(expert_idx, expert_map)
        if local_expert_idx == -1:
            return
        copy_gate_up_proj(
            dst[local_expert_idx],
            weights[key_for_gate(expert_idx)],
            weights[key_for_up(expert_idx)],
            partition_tp,
        )

    _iter_experts(num_experts, _one, pool)


def load_w2_per_expert(
    dst,
    weights,
    key_for_down: Callable[[int], str],
    expert_map,
    num_experts: int,
    pool: Optional[ThreadPoolExecutor] = None,
):
    """Load the ``w2_weight`` (down) MoE parameter from a per-expert
    checkpoint.

    See :func:`load_fused_w13_per_expert` for argument semantics; ``dst`` has
    shape ``(local_num_experts, hidden, inter_per_partition)`` here.
    """
    partition_tp = not is_use_ep()

    def _one(expert_idx: int):
        local_expert_idx = resolve_ep_expert_idx(expert_idx, expert_map)
        if local_expert_idx == -1:
            return
        copy_single_proj_dim1(
            dst[local_expert_idx],
            weights[key_for_down(expert_idx)],
            partition_tp,
        )

    _iter_experts(num_experts, _one, pool)


def load_fused_w13_stacked(
    dst,
    w13_weight,
    expert_map,
    num_experts: int,
    pool: Optional[ThreadPoolExecutor] = None,
):
    """Load the fused ``w13_weight`` MoE parameter from a stacked checkpoint.

    Checkpoint layout: ``(E, H, 2I)`` where gate and up are concatenated
    along the last dim. Target layout: ``(local_E, 2 * I_p, H)``.

    EP-off + TP>1 fast path: replaces the per-expert Python loop with two
    bulk slice + permute + H2D copies (one for the gate half, one for the
    up half). Per :commit:`df725dd`, the per-expert slicing path was
    ~4m23s for Qwen3-VL-30B-A3B because each ``w13_weight[e][:, slice]``
    view is doubly non-contiguous after ``.permute(1, 0)``.

    EP-on (or TP==1) slow path: per-expert loop, optionally threaded.
    """
    if not is_use_ep() and get_tp_size() > 1:
        inter = w13_weight.shape[-1] // 2
        size_p = inter // get_tp_size()
        tp_rank = get_tp_rank()
        dst[:, :size_p, :].copy_(
            w13_weight[
                :, :, tp_rank * size_p : (tp_rank + 1) * size_p
            ].permute(0, 2, 1)
        )
        dst[:, size_p:, :].copy_(
            w13_weight[
                :,
                :,
                inter + tp_rank * size_p : inter + (tp_rank + 1) * size_p,
            ].permute(0, 2, 1)
        )
        return

    def _one(expert_idx: int):
        local_expert_idx = resolve_ep_expert_idx(expert_idx, expert_map)
        if local_expert_idx == -1:
            return
        # w13_weight[expert_idx]: (H, 2I) -> dst slot: (2 * I_p, H)
        # When EP is on every rank owns the full intermediate dim (I_p = I),
        # so we just need a permute, no further slicing.
        dst[local_expert_idx].copy_(w13_weight[expert_idx].permute(1, 0))

    _iter_experts(num_experts, _one, pool)


def load_w2_stacked(
    dst,
    w2_weight,
    expert_map,
    num_experts: int,
    pool: Optional[ThreadPoolExecutor] = None,
):
    """Load the ``w2_weight`` MoE parameter from a stacked checkpoint.

    Checkpoint layout: ``(E, I, H)``. Target layout:
    ``(local_E, H, I_p)``. EP-off + TP>1 is collapsed into one bulk slice +
    permute + H2D copy. EP-on falls back to a per-expert loop (optionally
    threaded via ``pool``).
    """
    if not is_use_ep() and get_tp_size() > 1:
        inter = w2_weight.shape[1]
        size_p = inter // get_tp_size()
        tp_rank = get_tp_rank()
        dst.copy_(
            w2_weight[
                :, tp_rank * size_p : (tp_rank + 1) * size_p, :
            ].permute(0, 2, 1)
        )
        return

    def _one(expert_idx: int):
        local_expert_idx = resolve_ep_expert_idx(expert_idx, expert_map)
        if local_expert_idx == -1:
            return
        dst[local_expert_idx].copy_(w2_weight[expert_idx].permute(1, 0))

    _iter_experts(num_experts, _one, pool)

