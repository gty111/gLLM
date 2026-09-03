"""Fuse the TP all-reduce with the RMSNorm that immediately follows it.

Under tensor parallelism a ``RowParallelLinear`` (attention ``o_proj`` /
``out_proj``) produces a per-rank partial sum that is all-reduced, and the
result is fed straight into the layer's norm, which adds the residual and
normalizes. Measured on a 2-way-TP decode graph replay of Qwen3.5-27B those are
two separate launches per site -- ``sglang::cross_device_reduce_1stage`` at
9.3 us plus ``gemma_rmsnorm_bf16_kernel`` at 6.1 us -- and there are 128 sites
per step (64 layers x 2). flashinfer's fused kernel does all-reduce + residual
add + RMSNorm in one 7.5 us launch, so the 128 sites collapse from 256
launches to 128 and the standalone RMSNorm kernels disappear entirely.

The Gemma convention (the stored weight means ``weight + 1``) is expressed with
the kernel's ``weight_bias`` argument, so no weight rewriting is needed --
folding the ``+1`` into a bf16 weight ahead of time would round away the
checkpoint's small learned offsets.

Every gate falls back to ``all_reduce`` + ``norm``, which is the numerically
identical unfused path.
"""

from __future__ import annotations

import functools
from typing import Optional, Tuple

import torch
from logger import logger

from gllm.distributed.parallel_state import (
    get_tp_group,
    get_tp_rank,
    get_tp_size,
    tensor_model_parallel_all_reduce,
)

_MiB = 1024 * 1024

# Per-(device capability, TP size) workspace budget.
# The Lamport buffers scale with this, so it is a memory/coverage trade rather
# than a correctness bound: an oversize batch simply falls back.
_MAX_SIZE_MB = {
    (9, 0): {2: 64.0, 4: 2.0, 8: 0.5},
    (10, 0): {2: 64.0, 4: 32.0, 8: 1.0},
    (10, 3): {2: 64.0, 4: 64.0, 8: 4.0},
    (12, 0): {2: 64.0, 4: 32.0, 8: 1.0},
}

_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16)

# Which branch each call took. A fallback is silent by design (an unsupported
# topology must not break the model), which makes it easy to benchmark or
# validate the unfused path by accident; tests assert on these.
_STATS = {"fused": 0, "fallback": 0}


@functools.lru_cache(maxsize=1)
def _flashinfer_comm():
    """Return ``flashinfer.comm`` plus the fusion pattern, or ``(None, None)``."""
    try:
        import flashinfer.comm as comm

        return comm, comm.AllReduceFusionPattern.kARResidualRMSNorm
    except (ImportError, AttributeError, RuntimeError):
        return None, None


def _max_token_num(tp_size: int, hidden_size: int, dtype: torch.dtype) -> Optional[int]:
    budget = _MAX_SIZE_MB.get(torch.cuda.get_device_capability(), {}).get(tp_size)
    if not budget:
        return None
    element_size = torch.tensor([], dtype=dtype).element_size()
    return int(budget * _MiB) // (hidden_size * element_size)


@functools.lru_cache(maxsize=4)
def _workspace(tp_size: int, rank: int, max_token_num: int, hidden_dim: int, dtype):
    """Create (once) the Lamport workspace this rank's fused all-reduce uses.

    Cached because the workspace registers IPC buffers across the TP group: it
    must be built exactly once per configuration, and its addresses have to stay
    fixed for CUDA-graph replay.  ``None`` means the topology cannot support the
    fused kernel (no NVSwitch, multicast unavailable), and callers fall back.
    """
    comm, _ = _flashinfer_comm()
    if comm is None:
        return None
    try:
        # flashinfer's default handle exchange goes through mpi4py, which this
        # stack does not ship; its own torch.distributed backend does the same
        # allgather/bcast over the TP process group.
        from flashinfer.comm.mnnvl import TorchDistBackend

        workspace = comm.create_allreduce_fusion_workspace(
            backend="auto",
            world_size=tp_size,
            rank=rank,
            max_token_num=max_token_num,
            hidden_dim=hidden_dim,
            dtype=dtype,
            comm_backend=TorchDistBackend(group=get_tp_group()),
            group=get_tp_group(),
        )
    except Exception as error:  # noqa: BLE001 - topology/deps dependent
        logger.warning(
            "Fused all-reduce + RMSNorm unavailable (%s: %s); "
            "falling back to separate all-reduce and norm.",
            type(error).__name__,
            error,
        )
        return None
    if workspace is not None and workspace.backend == "mnnvl" and not getattr(
        workspace, "mc_ptr", 0
    ):
        workspace.destroy()
        return None
    return workspace


def _plan(x: torch.Tensor, tp_size: int):
    """Return ``(workspace, max_token_num)`` when the fused path applies."""
    comm, pattern = _flashinfer_comm()
    if comm is None or pattern is None:
        return None, 0
    if (
        not x.is_cuda
        or x.dim() != 2
        or not x.is_contiguous()
        or x.dtype not in _SUPPORTED_DTYPES
    ):
        return None, 0
    num_tokens, hidden = x.shape
    budget = _max_token_num(tp_size, hidden, x.dtype)
    if budget is None or num_tokens > budget:
        return None, 0
    workspace = _workspace(tp_size, get_tp_rank(), budget, hidden, x.dtype)
    if workspace is None:
        return None, 0
    # The token bound above spends the whole budget, but a backend may use only
    # a fraction per call (mnnvl rotates three Lamport buffers), so ask rather
    # than let the kernel reject the launch.
    if not workspace.is_buffer_size_sufficient(
        tp_size=tp_size, num_tokens=num_tokens, hidden_dim=hidden, dtype=x.dtype
    ):
        return None, 0
    return workspace, budget


# Attribute names under which a module keeps the ``RowParallelLinear`` that
# performs its output all-reduce. Ordered so the attention/MLP output projection
# is found before anything else a module might expose.
_OUTPUT_PROJ_NAMES = ("o_proj", "out_proj", "dense", "down_proj", "wo_b")


def defer_reduce(module: torch.nn.Module) -> bool:
    """Stop ``module`` all-reducing its output; return whether that worked.

    The point is to hand the *next* norm a per-rank partial sum so it can fuse
    the all-reduce into its own kernel. Call sites keep the returned flag and
    only use the fused norm when it is true, so a module whose output reduce
    lives somewhere this does not recognise keeps its existing behaviour rather
    than silently emitting an un-reduced tensor -- the one failure mode of this
    optimization that would corrupt results instead of just being slow.
    """
    if get_tp_size() == 1:
        return False
    if getattr(module, "reduce_results", None) is True:
        # The module owns the reduce itself (e.g. a MoE block's tail).
        module.reduce_results = False
        return True
    for name in _OUTPUT_PROJ_NAMES:
        proj = getattr(module, name, None)
        if getattr(proj, "reduce_results", None) is True:
            proj.reduce_results = False
            return True
    return False


def link_fused_reduces(layers) -> bool:
    """Wire ``_fuse_input`` across a decoder stack; return the tail flag.

    ``input_layernorm`` consumes the *previous* layer's mlp output, so whether
    it owns an all-reduce depends on that predecessor's deferral, not on the
    layer's own. Layer 0 keeps ``False``: it receives an embedding, or a tensor
    the previous pipeline stage reduced before sending it -- keying it off the
    layer's own ``_fuse_mlp`` instead would all-reduce an already-reduced
    tensor at every PP stage boundary, scaling it by ``tp_size``.

    The return value says whether the last layer's mlp output is still a
    per-rank partial, which the caller needs both for the final norm and to
    decide whether to reduce before handing off to the next stage.
    """
    previous = False
    for layer in layers:
        layer._fuse_input = previous
        previous = bool(getattr(layer, "_fuse_mlp", False))
    return previous


def maybe_fused_norm(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    norm: torch.nn.Module,
    deferred: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``norm(all_reduce(x), residual)`` if the producer's reduce was deferred.

    ``deferred`` is what :func:`defer_reduce` returned for the module that
    produced ``hidden_states``. When it is false the tensor is already reduced
    and this is exactly ``norm(hidden_states, residual)``.
    """
    if not deferred:
        return norm(hidden_states, residual)
    return fused_all_reduce_rms_norm(hidden_states, residual, norm)


_GAMMA_ATTR = "_fused_ar_norm_gamma"


def _kernel_gamma(norm: torch.nn.Module, dtype: torch.dtype):
    """Return ``norm.weight`` in the kernel's dtype, memoized on the module.

    flashinfer's fused kernel reads ``rms_gamma`` as the activation dtype;
    handing it the fp32 parameter that ``RMSNorm``/``GemmaRMSNorm`` store makes
    it reinterpret the buffer and silently produce wrong values (a ~14x larger
    deviation from an fp32 reference than the unfused path -- it is not a
    tolerable rounding difference). Cast once and cache: casting per call would
    add a launch and give back what the fusion saves.

    Returns ``None`` when the parameter is not a plain 1-D contiguous tensor,
    so the caller falls back rather than guessing.
    """
    weight = norm.weight
    if weight.dtype == dtype:
        return weight
    if weight.dim() != 1:
        return None
    # Key on the parameter's version counter so a later in-place weight write
    # (checkpoint load, any reload) cannot leave a stale copy behind.
    stamp = (dtype, tuple(weight.shape), weight._version)
    cached = getattr(norm, _GAMMA_ATTR, None)
    if cached is not None and cached[0] == stamp:
        return cached[1]
    gamma = weight.detach().to(dtype).contiguous()
    setattr(norm, _GAMMA_ATTR, (stamp, gamma))
    return gamma


def fused_all_reduce_rms_norm(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    norm: torch.nn.Module,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``norm(all_reduce(hidden_states), residual)`` in one kernel where possible.

    ``hidden_states`` must be the *un-reduced* per-rank output of a row-parallel
    linear.  Returns ``(normed, new_residual)`` exactly like ``norm``'s
    residual form.
    """
    tp_size = get_tp_size()
    if tp_size == 1:
        _STATS["fallback"] += 1
        return norm(hidden_states, residual)

    flat = hidden_states
    reshaped = flat.dim() != 2
    if reshaped:
        flat = flat.view(-1, flat.shape[-1])
        residual_flat = residual.view(-1, residual.shape[-1])
    else:
        residual_flat = residual

    workspace, max_token_num = _plan(flat, tp_size)
    if workspace is None:
        if not _STATS["fallback"]:
            logger.info("Fused all-reduce + RMSNorm inactive; using separate kernels")
        _STATS["fallback"] += 1
        reduced = tensor_model_parallel_all_reduce(hidden_states)
        return norm(reduced, residual)

    comm, pattern = _flashinfer_comm()
    # The norm class declares its own gain convention: a Gemma-style norm
    # stores the learned *offset*, so the kernel has to add 1 before scaling.
    # Anything that does not declare one is not a norm this op understands.
    weight_bias = getattr(norm, "weight_bias", None)
    if weight_bias is None:
        _STATS["fallback"] += 1
        reduced = tensor_model_parallel_all_reduce(hidden_states)
        return norm(reduced, residual)
    gamma = _kernel_gamma(norm, flat.dtype)
    if gamma is None:
        _STATS["fallback"] += 1
        reduced = tensor_model_parallel_all_reduce(hidden_states)
        return norm(reduced, residual)
    # Alias exactly like the unfused ``ops.fused_add_rms_norm`` this replaces:
    # the normalized result lands in the *input* buffer and the new residual in
    # the *residual* buffer. Passing a separate ``norm_out`` instead makes the
    # kernel put the new residual in the input buffer -- and that buffer belongs
    # to the producer (a MoE workspace, a piecewise-graph slot), which the next
    # layer overwrites, silently clobbering the residual. That showed up as
    # out-of-range token ids once the batch was large enough for the buffers to
    # actually be recycled.
    comm.allreduce_fusion(
        input=flat,
        workspace=workspace,
        pattern=pattern,
        launch_with_pdl=True,
        output=None,
        residual_out=residual_flat,
        norm_out=flat,
        quant_out=None,
        scale_out=None,
        residual_in=residual_flat,
        rms_gamma=gamma,
        rms_eps=norm.variance_epsilon,
        scale_factor=None,
        layout_code=None,
        use_oneshot=None,
        fp32_acc=True,
        weight_bias=float(weight_bias),
        # The one-shot Lamport all-reduce signals PDL completion before its
        # output buffer is committed, so a following PDL kernel can read an
        # uninitialized buffer and produce NaN at small token counts.
        trigger_completion_at_end=True,
    )
    if not _STATS["fused"]:
        logger.info("Fused all-reduce + RMSNorm active (%s tokens/site)", flat.shape[0])
    _STATS["fused"] += 1
    if reshaped:
        return flat.view_as(hidden_states), residual_flat.view_as(residual)
    return flat, residual_flat


__all__ = [
    "defer_reduce",
    "fused_all_reduce_rms_norm",
    "link_fused_reduces",
    "maybe_fused_norm",
]
