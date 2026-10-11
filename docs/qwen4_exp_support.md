# Qwen3.8-Flash-Next support

The native `Qwen4ExpForConditionalGeneration` architecture is registered for
BF16/FP8 text and multimodal inference with TP/EP. Transformers 5.19.0 or newer
provides the native config and processor. PP and DP-attention are unsupported.

## Implemented

- The 48-layer text backbone combines 36 Gated DeltaNet layers, 12 Qwen Sparse
  Attention (QSA) layers, four Gated Residual streams and PLE. GDN uses the
  published sigmoid output gate. Native stacked BF16 and per-expert FP8 MoE
  weights reuse existing gLLM loaders and kernels.
- PLE hashes packed n-grams on GPU. Its embedding table is row-sharded across TP
  ranks, loaded in bounded CPU chunks and kept on GPU. Official FP8 lookup
  preserves the published cast-then-global-scale order. Gating and dilated
  convolution run on GPU with per-request histories.
- QSA caches completed, normalized and rotated index blocks once.
  GPU kernels score paged compressed blocks, select causal blocks and attend
  to paged K/V without gathering the full main-attention history. The visible
  incomplete tail is included. Decode, prefill and verify share these kernels.
  Physical page and snapshot addresses use int64 before stride multiplication
  so arenas larger than 4 GB do not overflow int32 offsets.
- BF16/FP16 QSA groups query heads sharing K/V and uses Tensor Core dot
  products. Small decode/verify grids use split-K with an FP32 merge; larger
  prefill grids avoid the merge. Grouped Gemma normalization, HC gate mixing
  and residual injection use fused CUDA kernels and preserve the eager
  low-precision rounding boundaries. Equation references live in tests.
- Decode and MTP verify FULL CUDA Graph and MTP draft graphs are enabled. Batch padding uses
  dummy slots/pages. GPU snapshot copies read mutable target slots, allowing
  snapshot metadata to change between replays.
- Prefill supports piecewise CUDA Graph: MoE and both attention/MLP HC
  projections and residual mixing are captured. GDN/QSA input and output
  projections and GDN output gating/normalization are captured too. Their
  variable-length cores, PLE and request-state updates remain eager boundaries
  using current request metadata, for ordinary and MTP mixed batches.
  Persistent boundary buffers are shared per tensor shape and input slot;
  residual streams, injection logits and GDN output gates are bridged across
  graph segments without retaining graph-pool activations.
- HC down/injection projections share one GEMM, packed after weight loading
  with alignment padding. Original checkpoint parameter names remain views of
  the packed storage. No projection packing occurs on graph replay.
- Block-128 FP8 SwiGLU experts with renormalized softmax routing use
  FlashInfer TRTLLM when the device and installed kernels pass support checks.
  EP offsets and graph replay are supported. Weight permutations happen once
  at load time; published FP32 scales retain their precision.
- GDN prefill uses FlashInfer's indexed state-pool kernel for supported
  128-dimensional K/V heads. Other devices and shapes retain the vendored FLA
  backend, whose boundary partitions and chunk metadata are reused within each
  forward.
- QSA index scoring uses Tensor Core dot products for BF16/FP16 grids with at
  least eight rows. Top-k block expansion and causal tail generation use one
  kernel; one-row scoring retains the faster scalar reduction.
- QSA block selection uses FlashInfer ragged top-k with its graph-safe path,
  deterministic output and smaller-index tie-breaking. The scoring kernel
  publishes GPU row lengths and zero offsets without a separate metadata
  launch. Selection emits all completed blocks directly when they fit in the
  budget, including on graph replay; otherwise it selects the highest scores.
  Unused block IDs are -1, and expansion appends the incomplete causal tail.
- QSA preparation fuses Q/K normalization, text or MRoPE rotation, K/V and
  raw-index cache publication, and packed-request mapping. Compressed-index
  maintenance remains a separate launch because it reads across tokens.
  Eager selection bounds its search using the CPU page-table width and the
  engine's logical context limit. When that bound fits within the selection
  budget, all causal tokens are emitted directly without scoring or top-k.
  Full decode/verify graphs retain the full search extent so growing request
  contexts can replay the same graph safely. Tiles beyond each query's visible
  compressed prefix skip cache reads and dot products.
- The existing Qwen3-VL vision tower loads native vision weights. QSA keeps
  the first token's resolved MRoPE frequencies with each index block,
  preserving image/video positions through compression and prefix reuse.
- MTP carries the pre-final-mixer `hc_count * hidden_size` streams between
  target and draft steps. Its final mixer produces LM-head inputs. GDN, PLE
  convolution histories and int64 token histories keep per-candidate
  checkpoints; accepting a prefix selects the corresponding state slot.
- `temperature=0` selects greedy decoding, including mixed sampling batches
  and MTP. Penalties and grammar are applied before argmax.

Generic layers live in `layers/hyper_connection.py`, `layers/position_learning.py`,
`layers/attention/{qsa,paged_sparse}.py` and
`layers/ops/{hyper_connection,qsa,state_snapshot}.py`.
There are no model-named directories under `layers`.

## Implementation details

### Shared state and graph buffers

GDN and PLE snapshots copy directly between disjoint working and snapshot
slots. Snapshot publication happens after the copy kernels are enqueued.
Piecewise boundaries can write into persistent output buffers; ordinary GDN
decode and indexed prefill use those buffers directly. Mixed forwards write
partitions into separate slices. The FLA fallback and MTP verify copy their
results into the corresponding slices. Input/residual bridges and padding
clears preserve graph-pool lifetime safety.

The shared GDN snapshot path also applies to Qwen3.8-27B. That decoder wraps
whole attention in a dynamic callback, so it does not yet use the GDN
`forward_piecewise` direct output-buffer path. PLE snapshots apply to Flash-Next.

### GDN prefill preparation

A single Triton kernel normalizes Q/K, materializes contiguous V, computes
sigmoid gates and converts log decay to alpha. Q/K keep projection-dtype
rounding. Beta rounds sigmoid to the projection dtype before FP32 storage;
alpha uses FP32 expf. Layers share a workspace under serialized CUDA-stream
execution. Workspace capacities are retained so captured graphs keep valid
scratch addresses after growth. Flash-Next and Qwen3.8-27B share this path.

### HC projections and mixing

Supported BF16 inputs with 1--32 token rows, 10240 input features, 320 lowrank
features and four injection features use a two-stage Triton down projection.
A 16-way split-K GEMM writes FP32 partials; the epilogue reduces them, rounds
to BF16, divides with another BF16 rounding, and emits scaled SiLU plus the
injection projection. Scratch and results persist for each captured shape.
Other device/shape combinations and all up projections use cuBLAS. This is an
HC optimization; Qwen3.8-27B has no HC down projection.

Attention residual combination is fused with the following MLP mixer
normalization while materializing the streams needed by PLE and MTP.
Gate-mixing launch geometry depends on token count, stream count, hidden width
and dtype. These operations are captured by decode, verify and piecewise graphs.

### MTP staging lifetime

Sequence metadata, cohort-remap indices, completed-prefill relay rows and KV
boundary token patches use alternating pinned host buffers. A copy-completion
event protects each slot before CPU reuse. The successor can be launched before
the predecessor is collected without overwriting an in-flight H2D source.
Device destinations remain persistent and are reused in CUDA-stream order.

## Validation

Tests cover native Transformers logits, sparse selection above budget,
chunked and packed prefill, prefix restore, BF16/FP8 embedding partitions,
graph replay with changing IDs and positions, image/video outputs, MTP verify
logits and accepted-state rollback, and zero-temperature sampling. Piecewise
replay covers ragged prefill and mixed ordinary/speculative batches.

Kernel tests cover FP64 projection references, low-norm inputs, strided tensors,
workspace growth, offsets above 2^31 elements, deterministic top-k ties and
snapshot copies. Delayed-stream tests exercise staging reuse, cohort
reorder/compaction, prefill relay admission and boundary patches. The latest
CUDA-enabled validation passed 254 tests; one optional full-checkpoint test
was skipped. Routing bounds and decoders without PLE layers are covered too.

The latest 1400-question MMLU-Pro regression used concurrency 32, greedy
decoding, no thinking and a 1024-token output cap. MTP k=3 scored 1091/1400
(77.93%); ordinary decode scored 1089/1400 (77.79%). Both runs had no empty
responses or CUDA errors. Unparsed answers were counted as wrong.

Different target decode/verify kernel shapes can reorder nearly tied BF16
logits. MTP tests compare logits and accepted states numerically rather than
requiring whole generated paragraphs to be identical across modes.

```bash
python -m pytest tests/test_qwen4_exp*.py tests/test_output_budget.py -q

GLLM_FLASH_NEXT_CHECKPOINT=/path/to/bf16/snapshot python -m pytest tests/test_qwen4_exp.py -q

python tests/qwen4_exp_checkpoint_utils.py /path/to/bf16/snapshot
```

## Usage and limits

Use the official BF16 or FP8 indexed safetensors directory. FP8 supports
routed-expert block-128 quantization and globally scaled PLE tables; attention,
GDN and shared-expert projections stay BF16 as specified by that checkpoint.
NVFP4/INT4 and quantized dense attention projections are not implemented here.

```python
from gllm import LLM

llm = LLM("/path/to/Qwen3.8-Flash-Next-FP8", tp_size=4, use_ep=True,
          model_max_length=8192, mtp_enabled=True, mtp_k=2,
          disable_cuda_graph=False)
outputs = llm.generate(prompts=["你好"], output_lens=[32], temperature=0)
```

Set `mtp_enabled=False` for ordinary decode. Native MTP config is eligible for
automatic detection. Enable `piecewise_cuda_graph=True` for prefill graphs
and `disable_cuda_graph=False` for decode, draft and verify graphs. Prefill
uses graph segments with dynamic eager boundaries rather than one full graph.
Requests containing image/video data retain the existing eager prefill fallback.

Weight payload sizes (decimal GB; exclude caches, activations and graphs):

| Component | BF16 | FP8 |
|---|---:|---:|
| PLE embedding table | 102.400 | 51.200 |
| Other text weights | 251.487 | 130.706 |
| Vision | 0.898 | 0.898 |
| MTP | 5.214 | 2.698 |
| Total | 360.000 | 185.502 |
