# Qwen multimodal MTP KV refresh

The Qwen MTP head attends to its own KV layer in the shared cache. A target
prefill must populate that layer before speculative decoding reads it. The
entry at position `p` pairs `target_hidden[p]` with the input embedding at
`p + 1`. Image/video placeholder IDs alone do not reproduce visual embeddings.

For the Qwen head with a colocated encoder and a single pipeline stage,
multimodal preparation now retains sparse visual replacements for the shifted
span. This includes the first visual row of the next prefill chunk, when the
chunk boundary splits an image. The refresh places those rows at each request's
batch offset before the MTP embedding normalization. The target hidden state
already includes any deepstack contributions; those residual columns are not
reapplied to the head's input embedding.

The replacements hold views of encoded visual rows until the refresh is queued,
even when the last prefill chunk releases the request's visual cache. They do
not retain full prompt embeddings. Index construction uses CPU masks, and GPU
indices are copied from pinned memory without a GPU-to-CPU round trip. Text-only
refresh keeps its fused embedding/norm path, and decode/verify CUDA graphs keep
their existing token-only input path.

The existing unsupported path remains for MTP heads without visual input
support, pipeline stages that do not own visual features, or a partially ready
disaggregated encoder that has not supplied the shifted lookahead row. Those
cases emit a warning and are not covered by the colocated-encoder fix.

Regression tests: `tests/test_mtp_multimodal_kv.py`, together with
`tests/test_text_embedding_chunks.py` and `tests/test_mtp_embed_norm.py`.
