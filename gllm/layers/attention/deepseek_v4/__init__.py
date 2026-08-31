"""DeepSeek-V4 sparse attention.

The modules are separated by responsibility so each numerical boundary can be
verified on its own:

* :mod:`ops` -- stateless primitives: YaRN RoPE, FP8/MXFP4 QAT round trips,
  sliding-window/compressed index construction, and the two sparse-attention
  kernels (fused FlashMLA and the eager oracle DSpark also uses).
* :mod:`projection` -- every learned matrix of a V4 attention layer.
* :mod:`compressor` -- the learned KV pooling and its request-owned state.
* :mod:`indexer` -- the C4 lightning indexer that selects compressed rows.
* :mod:`layer` -- the serving layer: paged prefill/decode over the cache arenas.
* :mod:`reference` -- token-at-a-time numerical oracles used by tests only.
* :mod:`dspark` -- the DSpark speculative block attention stage.
"""
