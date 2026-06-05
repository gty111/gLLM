"""Encoder-disaggregation control + data plane (design §4-§6).

This package wires the Encoder node (vision-only ViT, :mod:`gllm.encoder_engine`)
to the LM PP0 worker:

    Frontend  --skeleton tokens + raw mm items-->  LM PP0 worker
    LM PP0    --EncoderJob(slot desc) (ZMQ)----->  Encoder runtime
    Encoder   --MmItemMeta (ZMQ)--------------->   LM PP0 (token-id expansion)
    Encoder   --visual embedding (NIXL WRITE)-->   LM PP0 recv slot pool
    Encoder   --notif "emb:<seq>:<item>"------>    LM PP0 (data-ready gate)

Phase 3b holds each request until *all* its items' meta + embeddings have
landed, then hands a fully-formed (expanded token_ids, grids, embeddings)
Sequence to the scheduler -- so the existing monolith prefill path runs
unchanged and the result is byte-identical with the monolith. Intra-request
encode/prefill overlap (two-layer gating) is layered on in Phase 6.
"""

__all__ = ["EncoderJob", "MmItemMeta"]


def __getattr__(name):
    # Lazy re-export so that importing a lightweight submodule (e.g.
    # ``gllm.disagg.config``) from the monolith path does not eagerly import
    # ``protocol`` -> ``gllm.transfer.nixl_transfer`` and pull NIXL into a
    # process that never disaggregates.
    if name in __all__:
        from gllm.disagg.protocol import EncoderJob, MmItemMeta

        return {"EncoderJob": EncoderJob, "MmItemMeta": MmItemMeta}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
