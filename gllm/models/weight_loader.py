"""Declarative, table-driven model weight loading.

Historically every model's ``load_weights`` was a monolithic
``for k, v in named_parameters(): if key.find(...) elif ...`` chain. Subclasses
(DeepSeek, Kimi, Qwen3-MoE) inherited one base chain and had their
architecture-specific branches (MLA fused QKV-a, int4 packed experts, ...)
jammed into it, so the loaders were tightly coupled.

This module replaces that with an ordered **rule table**: each model declares a
list of :class:`WeightRule` ``(match, handler)`` pairs, and :func:`run_weight_loader`
provides the shared dispatch engine (progress bar, PP layer-index remap,
first-match dispatch, default copy, optional MoE thread pool, optional
pre-passes). Subclasses compose ``parent_rules + own_rules`` instead of copying
a loop.

Key invariants preserved from the old loaders:

* Keys are remapped via :func:`resolve_pp_layer_idx` before both matching and
  checkpoint lookup.
* Rules are matched **in order, first match wins** — exactly mirroring the old
  ``if/elif`` priority. Ordering matters where one pattern is a substring of
  another (e.g. ``w13_weight_packed`` must precede ``w13_weight``).
* All checkpoint reads go through :func:`get_tensor_from_dict` (a superset of
  the old direct ``weights[k]`` access: it tries ``k`` first, then the
  ``language_model.`` / ``visual.`` fallbacks).
* ``mp_load_progress`` / tqdm progress semantics and counts are unchanged.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Set

import torch

from gllm.dist_utils import get_local_rank, resolve_pp_layer_idx
from gllm.utils import get_model_load_pbar

from .weight_utils import (
    copy_gate_up_proj,
    copy_qkv_a_proj,
    copy_qkv_proj,
    copy_qkv_proj_gqa,
    copy_single_proj_dim0,
    copy_single_proj_dim1,
    get_tensor_from_dict,
    has_tensor_in_dict,
    load_fused_w13_per_expert,
    load_fused_w13_stacked,
    load_fused_w13_stacked_natural,
    load_w2_per_expert,
    load_w2_stacked,
    load_w2_stacked_natural,
    moe_expert_load_pool,
)


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------


@dataclass
class LoadContext:
    """Mutable per-load context threaded through every handler.

    Carries the checkpoint dict plus the metadata handlers need (attention head
    geometry, MoE expert map / count / thread pool). ``extra`` holds
    model-specific bits (e.g. FP8 block size, GQA replication factor) so the
    handler signature stays uniform.
    """

    weights: Dict[str, torch.Tensor]
    num_heads: int = 0
    num_kv_heads: int = 0
    head_dim: int = 0
    expert_map: object = None
    num_experts: Optional[int] = None
    pool: object = None
    extra: dict = field(default_factory=dict)


# A handler mutates ``param`` (a parameter's ``.data``) from ``ctx.weights``
# using the already-PP-remapped ``key``.
Handler = Callable[[LoadContext, str, torch.Tensor], None]


@dataclass
class WeightRule:
    match: Callable[[str], bool]
    handler: Handler
    name: str = ""


# A pre-pass runs before the per-parameter loop (e.g. fused GDN load). It
# returns the set of *local* parameter names it has already filled so the main
# loop skips them. It is responsible for calling ``update`` once per filled key.
PrePass = Callable[
    [torch.nn.Module, Dict[str, torch.nn.Parameter], LoadContext, Callable[[], None]],
    Set[str],
]


# ---------------------------------------------------------------------------
# Match helpers
# ---------------------------------------------------------------------------


def contains(*subs: str) -> Callable[[str], bool]:
    """Match if any of ``subs`` is a substring of the key (mirrors ``k.find``)."""
    return lambda k: any(s in k for s in subs)


# ---------------------------------------------------------------------------
# Dispatch engine
# ---------------------------------------------------------------------------


def run_weight_loader(
    model: torch.nn.Module,
    weights: Dict[str, torch.Tensor],
    rules: Sequence[WeightRule],
    mp_load_progress,
    *,
    pp_idx_offset: int,
    start_layer: int,
    ctx: LoadContext,
    pre_passes: Sequence[PrePass] = (),
    src_key_fn: Optional[Callable[[str], str]] = None,
) -> None:
    """Drive weight loading from a rule table.

    Args:
        model: the module whose ``named_parameters()`` to fill.
        weights: checkpoint CPU tensor dict.
        rules: ordered rule table; first matching rule per key wins.
        mp_load_progress: shared progress array (multi-proc) or ``None`` (tqdm).
        pp_idx_offset: which dotted segment holds the layer index (2 for most,
            3 for ChatGLM) — passed to :func:`resolve_pp_layer_idx`.
        start_layer: this PP rank's first global layer index.
        ctx: :class:`LoadContext` shared with every handler.
        pre_passes: optional callables run before the per-parameter loop; each
            returns the set of local parameter names it already filled.
        src_key_fn: optional transform applied to the PP-resolved key before
            matching/handling (e.g. ChatGLM rewrites ``embedding`` ->
            ``embedding.word_embeddings`` to match its checkpoint names).
    """
    parameters = dict(model.named_parameters())
    if mp_load_progress is not None:
        rank = get_local_rank()
        mp_load_progress[rank * 2] = len(parameters)
        mp_load_progress[rank * 2 + 1] = 0

        def update() -> None:
            mp_load_progress[rank * 2 + 1] += 1

    else:
        pbar = get_model_load_pbar(len(parameters))

        def update() -> None:
            pbar.update(1)

    def body() -> None:
        filled: Set[str] = set()
        for pre in pre_passes:
            filled |= pre(model, parameters, ctx, update)

        for k, v in parameters.items():
            if k in filled:
                continue
            rk = resolve_pp_layer_idx(k, pp_idx_offset, start_layer)
            if src_key_fn is not None:
                rk = src_key_fn(rk)
            for rule in rules:
                if rule.match(rk):
                    rule.handler(ctx, rk, v.data)
                    break
            else:
                v.data.copy_(get_tensor_from_dict(weights, rk))
            update()

    # MoE models open a shared thread pool for the whole loop so per-expert H2D
    # copies overlap. Dense models (num_experts is None) skip it.
    if ctx.num_experts is not None:
        with moe_expert_load_pool(ctx.num_experts) as pool:
            ctx.pool = pool
            body()
    else:
        body()


# ---------------------------------------------------------------------------
# Generic dense handlers
# ---------------------------------------------------------------------------


def h_qkv_proj(ctx: LoadContext, k: str, p: torch.Tensor) -> None:
    """Fused QKV from separate q/k/v checkpoints (Qwen-style).

    ``head_dim_patch`` collapses to 1 for FP8 ``weight_scale`` rows (one scale
    per output row group) and is ``head_dim`` for the weight itself.
    """
    head_dim_patch = (
        ctx.head_dim if (k.find("scale") == -1 or k.find("weight") == -1) else 1
    )
    w = ctx.weights
    copy_qkv_proj(
        p,
        get_tensor_from_dict(w, k.replace("qkv_proj", "q_proj")),
        get_tensor_from_dict(w, k.replace("qkv_proj", "k_proj")),
        get_tensor_from_dict(w, k.replace("qkv_proj", "v_proj")),
        ctx.num_heads,
        ctx.num_kv_heads,
        head_dim_patch,
    )


def h_qkv_a_proj(ctx: LoadContext, k: str, p: torch.Tensor) -> None:
    """DeepSeek/Kimi MLA fused ``q_a_proj`` ++ ``kv_a_proj_with_mqa``."""
    w = ctx.weights
    copy_qkv_a_proj(
        p,
        get_tensor_from_dict(w, k.replace("fused_qkv_a_proj", "q_a_proj")),
        get_tensor_from_dict(w, k.replace("fused_qkv_a_proj", "kv_a_proj_with_mqa")),
    )


def h_gate_up(ctx: LoadContext, k: str, p: torch.Tensor) -> None:
    """Fused gate_up from separate gate_proj/up_proj (dense or shared expert)."""
    w = ctx.weights
    copy_gate_up_proj(
        p,
        get_tensor_from_dict(w, k.replace("gate_up_proj", "gate_proj")),
        get_tensor_from_dict(w, k.replace("gate_up_proj", "up_proj")),
    )


def h_proj_dim1(ctx: LoadContext, k: str, p: torch.Tensor) -> None:
    """Row-parallel proj (o_proj / down_proj): TP slices the input (col) dim."""
    copy_single_proj_dim1(p, get_tensor_from_dict(ctx.weights, k))


def h_proj_dim0(ctx: LoadContext, k: str, p: torch.Tensor) -> None:
    """Column-parallel proj (embed / lm_head / DeepSeek q_proj/kv_b/q_b): TP
    slices the output (row) dim."""
    copy_single_proj_dim0(p, get_tensor_from_dict(ctx.weights, k))


def h_default(ctx: LoadContext, k: str, p: torch.Tensor) -> None:
    """Verbatim copy (norms, biases, scalar params)."""
    p.copy_(get_tensor_from_dict(ctx.weights, k))


# ---- ChatGLM handlers ------------------------------------------------------
# ChatGLM stores QKV as one fused ``query_key_value`` tensor (sliced by
# precomputed q/k index bounds) and the MLP gate+up as one fused
# ``dense_h_to_4h`` tensor (sliced at ``intermediate_size``). The slice bounds
# live in ``ctx.extra``.


def h_chatglm_qkv(ctx: LoadContext, k: str, p: torch.Tensor) -> None:
    w = get_tensor_from_dict(ctx.weights, k)
    qi = ctx.extra["q_index"]
    ki = ctx.extra["k_index"]
    copy_qkv_proj(
        p, w[:qi], w[qi:ki], w[ki:], ctx.num_heads, ctx.num_kv_heads, ctx.head_dim
    )


def h_chatglm_gate_up(ctx: LoadContext, k: str, p: torch.Tensor) -> None:
    w = get_tensor_from_dict(ctx.weights, k)
    isz = ctx.extra["intermediate_size"]
    copy_gate_up_proj(p, w[:isz], w[isz:])


# ---------------------------------------------------------------------------
# MoE expert handler factories
# ---------------------------------------------------------------------------
#
# The FusedMoE params are ``w13_weight`` (gate++up) and ``w2_weight`` (down).
# Checkpoints store experts per-index under model-specific sub-keys, so each
# model passes the per-expert key templates: e.g. Qwen uses
# ``{i}.gate_proj.weight``, Mixtral uses ``{i}.w1.weight``, int4 uses
# ``{i}.gate_proj.weight_packed``. ``replace`` is the param token to substitute.


def make_w13_loader(replace: str, gate_to: str, up_to: str) -> Handler:
    def h(ctx: LoadContext, k: str, p: torch.Tensor) -> None:
        load_fused_w13_per_expert(
            p,
            ctx.weights,
            key_for_gate=lambda i, k=k: k.replace(replace, f"{i}.{gate_to}"),
            key_for_up=lambda i, k=k: k.replace(replace, f"{i}.{up_to}"),
            expert_map=ctx.expert_map,
            num_experts=ctx.num_experts,
            pool=ctx.pool,
        )

    return h


def make_w2_loader(replace: str, down_to: str) -> Handler:
    def h(ctx: LoadContext, k: str, p: torch.Tensor) -> None:
        load_w2_per_expert(
            p,
            ctx.weights,
            key_for_down=lambda i, k=k: k.replace(replace, f"{i}.{down_to}"),
            expert_map=ctx.expert_map,
            num_experts=ctx.num_experts,
            pool=ctx.pool,
        )

    return h


# ---- Stacked-expert variants ----------------------------------------------
# Some checkpoints pre-stack all experts into a single tensor instead of one
# tensor per expert index. ``stacked`` is the transposed ``(E, H, 2I)`` /
# ``(E, I, H)`` convention (Qwen3-VL-MoE); ``stacked_natural`` is the
# ``(E, 2I, H)`` / ``(E, H, I)`` convention (Qwen3.5-MoE bf16).


def make_w13_stacked_loader(replace: str, stacked_to: str) -> Handler:
    def h(ctx: LoadContext, k: str, p: torch.Tensor) -> None:
        load_fused_w13_stacked(
            p,
            get_tensor_from_dict(ctx.weights, k.replace(replace, stacked_to)),
            expert_map=ctx.expert_map,
            num_experts=ctx.num_experts,
            pool=ctx.pool,
        )

    return h


def make_w2_stacked_loader(replace: str, stacked_to: str) -> Handler:
    def h(ctx: LoadContext, k: str, p: torch.Tensor) -> None:
        load_w2_stacked(
            p,
            get_tensor_from_dict(ctx.weights, k.replace(replace, stacked_to)),
            expert_map=ctx.expert_map,
            num_experts=ctx.num_experts,
            pool=ctx.pool,
        )

    return h


# ---- Qwen3.5-MoE hybrid (GDN + FP8) expert/attn handlers -------------------
# Qwen3.5-MoE stores experts either stacked (bf16: single ``gate_up_proj`` /
# ``down_proj`` in *natural* (E, 2I, H) / (E, H, I) layout) or per-expert (FP8:
# ``{i}.gate_proj.weight`` + ``{i}.gate_proj.weight_scale_inv``). The same
# parameter (``w13_weight`` / ``w13_weight_scale_inv``) routes here; we detect
# the stacked tensor at load time and fall back to per-expert otherwise.


def _expert_key_base(k: str, weight_token: str) -> tuple:
    """Split ``...experts.{weight_token}[_scale_inv]`` into (base, suffix)."""
    is_scale = k.endswith("weight_scale_inv")
    suffix = "weight_scale_inv" if is_scale else "weight"
    drop = f"{weight_token}_scale_inv" if is_scale else weight_token
    base = k.replace(drop, "")
    return base, suffix, is_scale


def h_w13_hybrid(ctx: LoadContext, k: str, p: torch.Tensor) -> None:
    base, suffix, is_scale = _expert_key_base(k, "w13_weight")
    stacked_key = f"{base}gate_up_proj"
    if not is_scale and has_tensor_in_dict(ctx.weights, stacked_key):
        load_fused_w13_stacked_natural(
            p,
            get_tensor_from_dict(ctx.weights, stacked_key),
            expert_map=ctx.expert_map,
            num_experts=ctx.num_experts,
            pool=ctx.pool,
        )
    else:
        load_fused_w13_per_expert(
            p,
            ctx.weights,
            key_for_gate=lambda i, base=base, s=suffix: f"{base}{i}.gate_proj.{s}",
            key_for_up=lambda i, base=base, s=suffix: f"{base}{i}.up_proj.{s}",
            expert_map=ctx.expert_map,
            num_experts=ctx.num_experts,
            pool=ctx.pool,
        )


def h_w2_hybrid(ctx: LoadContext, k: str, p: torch.Tensor) -> None:
    base, suffix, is_scale = _expert_key_base(k, "w2_weight")
    stacked_key = f"{base}down_proj"
    if not is_scale and has_tensor_in_dict(ctx.weights, stacked_key):
        load_w2_stacked_natural(
            p,
            get_tensor_from_dict(ctx.weights, stacked_key),
            expert_map=ctx.expert_map,
            num_experts=ctx.num_experts,
            pool=ctx.pool,
        )
    else:
        load_w2_per_expert(
            p,
            ctx.weights,
            key_for_down=lambda i, base=base, s=suffix: f"{base}{i}.down_proj.{s}",
            expert_map=ctx.expert_map,
            num_experts=ctx.num_experts,
            pool=ctx.pool,
        )


def h_qkv_proj_gqa(ctx: LoadContext, k: str, p: torch.Tensor) -> None:
    """GQA/FP8-aware fused QKV (Qwen3.5).

    Reads from ``ctx.extra``: ``num_q_rows`` (q rows per rank, doubled when
    ``attn_output_gate``), ``num_kv_heads`` (per-rank), ``num_kv_head_replicas``
    (TP ranks sharing a kv head), ``qkv_block_n`` (FP8 block size or None). For
    the FP8 ``weight_scale_inv`` the per-component stride is ``head_dim //
    block_n`` rows instead of ``head_dim``.
    """
    e = ctx.extra
    is_scale = k.endswith("weight_scale_inv")
    if is_scale and e.get("qkv_block_n"):
        head_dim_or_blocks = ctx.head_dim // e["qkv_block_n"]
    else:
        head_dim_or_blocks = ctx.head_dim
    w = ctx.weights
    copy_qkv_proj_gqa(
        p,
        get_tensor_from_dict(w, k.replace("qkv_proj", "q_proj")),
        get_tensor_from_dict(w, k.replace("qkv_proj", "k_proj")),
        get_tensor_from_dict(w, k.replace("qkv_proj", "v_proj")),
        e["num_q_rows"],
        ctx.num_kv_heads,
        e["num_kv_head_replicas"],
        head_dim_or_blocks,
    )


def h_qkv_proj_gated(ctx: LoadContext, k: str, p: torch.Tensor) -> None:
    """Fused QKV where q rows may be doubled by ``attn_output_gate`` (Qwen3.5
    dense).

    ``ctx.extra['num_q_rows']`` is the per-rank q-row count (in head units,
    already doubled for the gate). For the plain weight tensor each head spans
    ``head_dim`` rows. For an FP8 ``weight_scale_inv`` tensor the rows are
    block-quantized, so each head spans ``head_dim // block_n`` scale rows;
    we derive that per-head stride from the source shape instead of hardcoding
    it (the old code collapsed it to ``1``, which corrupted the K/V offsets and
    crashed on block-quantized checkpoints such as Qwen3.5-27B-FP8).
    """
    w = ctx.weights
    src_q = get_tensor_from_dict(w, k.replace("qkv_proj", "q_proj"))
    src_k = get_tensor_from_dict(w, k.replace("qkv_proj", "k_proj"))
    src_v = get_tensor_from_dict(w, k.replace("qkv_proj", "v_proj"))

    if k.endswith("weight_scale_inv"):
        # Per-head scale-row stride = (head_dim // block_n). ``num_q_rows``
        # counts the q+gate head-row groups, so dividing the q-scale row count
        # by it recovers the stride without needing block_n explicitly.
        num_q_rows = ctx.extra["num_q_rows"]
        head_dim_patch = src_q.shape[0] // num_q_rows
    else:
        head_dim_patch = ctx.head_dim

    copy_qkv_proj(
        p,
        src_q,
        src_k,
        src_v,
        ctx.extra["num_q_rows"],
        ctx.num_kv_heads,
        head_dim_patch,
    )


def make_gdn_pre_pass(gdn_subs, gdn_load_fn):
    """Build a pre-pass that fuses each layer's GDN block en bloc.

    ``gdn_subs`` are the local parameter sub-keys the fused load fills (so the
    main loop skips them). ``gdn_load_fn(linear_attn, prefix, weights)`` does
    the actual fused load. Mirrors the old two-pass GDN handling.
    """

    def pre(model, parameters, ctx, update):
        filled = set()
        for local_idx, layer in enumerate(model.model.layers):
            if getattr(layer, "linear_attn", None) is None:
                continue
            global_idx = local_idx + model.start_layer
            prefix = f"model.layers.{global_idx}.linear_attn"
            gdn_load_fn(layer.linear_attn, prefix, ctx.weights)
            for sub in gdn_subs:
                local_key = f"model.layers.{local_idx}.linear_attn.{sub}"
                if local_key in parameters:
                    filled.add(local_key)
                    update()
        return filled

    return pre


# ---------------------------------------------------------------------------
# Vision-tower loading (VL wrappers)
# ---------------------------------------------------------------------------
#
# VL vision towers are loaded separately from the language model: keys come
# from ``visual.named_parameters()`` and the checkpoint stores them under a
# ``visual.`` prefix, with no PP layer remap. Vision handlers reuse the
# ``Handler(ctx, k, p)`` signature; ``ctx.extra["prefix"]`` carries the
# checkpoint namespace so a handler resolves its source as
# ``f"{prefix}{k}"``. ``ctx.extra["src_split"]`` (when set) means the fused
# qkv tensor is stored as one tensor to be split, not separate q/k/v files.


def _vsrc(ctx: LoadContext, k: str) -> str:
    return f"{ctx.extra.get('prefix', '')}{k}"


def hv_default(ctx: LoadContext, k: str, p: torch.Tensor) -> None:
    p.copy_(get_tensor_from_dict(ctx.weights, _vsrc(ctx, k)))


def hv_proj_dim1(ctx: LoadContext, k: str, p: torch.Tensor) -> None:
    copy_single_proj_dim1(p, get_tensor_from_dict(ctx.weights, _vsrc(ctx, k)))


def hv_proj_dim0(ctx: LoadContext, k: str, p: torch.Tensor) -> None:
    copy_single_proj_dim0(p, get_tensor_from_dict(ctx.weights, _vsrc(ctx, k)))


def hv_gate_up(ctx: LoadContext, k: str, p: torch.Tensor) -> None:
    w = ctx.weights
    copy_gate_up_proj(
        p,
        get_tensor_from_dict(w, _vsrc(ctx, k).replace("gate_up_proj", "gate_proj")),
        get_tensor_from_dict(w, _vsrc(ctx, k).replace("gate_up_proj", "up_proj")),
    )


def hv_qkv_fused_split(ctx: LoadContext, k: str, p: torch.Tensor) -> None:
    """Vision fused QKV stored as a single tensor: split rows into q/k/v."""
    src_qkv = get_tensor_from_dict(ctx.weights, _vsrc(ctx, k))
    n = src_qkv.shape[0] // 3
    q, kk, vv = src_qkv.split([n, n, n], dim=0)
    copy_qkv_proj(p, q, kk, vv, ctx.num_heads, ctx.num_heads, ctx.head_dim)


def hv_merger_mlp(ctx: LoadContext, k: str, p: torch.Tensor) -> None:
    """Qwen2.5-VL ``merger.mlp``: first linear (``...0``) is column-parallel,
    second (``...2.weight``) is row-parallel, biases copy verbatim."""
    src = get_tensor_from_dict(ctx.weights, _vsrc(ctx, k))
    if k.find("0") != -1:
        copy_single_proj_dim0(p, src)
    elif k.find("2.weight") != -1:
        copy_single_proj_dim1(p, src)
    else:
        p.copy_(src)


def run_vision_loader(
    visual: torch.nn.Module,
    weights: Dict[str, torch.Tensor],
    rules: Sequence[WeightRule],
    ctx: LoadContext,
) -> None:
    """Load a vision tower from a ``ctx.extra['prefix']``-namespaced checkpoint.

    No PP layer remap, no MoE pool, no ``mp_load_progress`` accounting (matches
    the old VL loaders, which only counted the language model). First-match
    dispatch; unmatched keys fall back to :func:`hv_default`.
    """
    for k, v in visual.named_parameters():
        for rule in rules:
            if rule.match(k):
                rule.handler(ctx, k, v.data)
                break
        else:
            hv_default(ctx, k, v.data)
