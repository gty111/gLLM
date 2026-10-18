"""Shared helpers for MTP (multi-token prediction) head weight loading.

Three model families (DeepSeek-V3.2/V4 DSpark, Qwen3.5 dense, Qwen3.5-MoE)
each open-coded the same two patterns:

* detaching the head submodule for the base-model weight pass and reattaching
  it in a ``finally`` (:func:`detached_head`);
* filling the head's parameters through the *parent* model's rule table, with
  each local parameter name remapped to its checkpoint key by the head's own
  ``_src_key`` (:func:`load_remapped_weights`), optionally under the shared
  MoE expert-copy thread pool (:func:`maybe_expert_pool`).
"""

import contextlib

from .weight_utils import (
    copy_single_proj_dim0,
    get_tensor_from_dict,
    moe_expert_load_pool,
)


@contextlib.contextmanager
def detached_head(owner, attr: str):
    """Temporarily set ``owner.<attr>`` to None, restoring it afterwards.

    The base loader iterates ``owner.named_parameters()``; the head's params
    live under a checkpoint namespace the base rule table cannot resolve
    (``mtp.*`` vs ``model.layers.N.*``), so the head is detached for the base
    pass and loaded separately by the caller. Yields the detached module.
    """
    saved = getattr(owner, attr)
    setattr(owner, attr, None)
    try:
        yield saved
    finally:
        setattr(owner, attr, saved)


@contextlib.contextmanager
def maybe_expert_pool(ctx):
    """Open the shared MoE expert-copy thread pool when ``ctx`` is MoE.

    Mirrors the tail of :func:`run_weight_loader`: per-expert H2D copies
    overlap inside the pool; dense contexts (``num_experts is None``) skip it.
    """
    if ctx.num_experts is not None:
        with moe_expert_load_pool(ctx.num_experts) as pool:
            ctx.pool = pool
            yield
    else:
        yield


def load_remapped_weights(
    module,
    rules,
    ctx,
    src_key,
    *,
    dim0_prefixes=(),
    skip=None,
    after=None,
):
    """Fill ``module``'s parameters through a borrowed parent rule table.

    Each local parameter name is mapped to its checkpoint key by
    ``src_key(name)`` (e.g. ``mtp_block.<x>`` -> ``mtp.layers.0.<x>``); the
    first matching rule handles it, unmatched keys copy verbatim.

    ``dim0_prefixes``: parameter-name prefixes that bypass the rule table and
    take the vocab-parallel dim-0 slicer directly (their remapped keys lack
    the embed/lm_head substrings the table keys off -- e.g. DeepSeek's
    ``shared_head.``/``embed_tokens.``).

    ``skip(name)`` / ``after(name)`` let a caller (the Qwen3.5-MoE pre-pass)
    restrict the loop to a subset of parameters and do per-parameter
    bookkeeping (filled-set accounting + progress ``update``).
    """
    for name, p in module.named_parameters():
        if skip is not None and skip(name):
            continue
        src = src_key(name)
        if dim0_prefixes and name.startswith(dim0_prefixes):
            copy_single_proj_dim0(p.data, get_tensor_from_dict(ctx.weights, src))
        else:
            for rule in rules:
                if rule.match(src):
                    rule.handler(ctx, src, p.data)
                    break
            else:
                p.data.copy_(get_tensor_from_dict(ctx.weights, src))
        if after is not None:
            after(name)
