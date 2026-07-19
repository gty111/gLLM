# DeepSeek Sparse Attention (DSA / V3.2) — Design & Notes

This document describes how gLLM supports **DeepSeek-V3.2**, whose only
architectural addition over V3 is **DeepSeek Sparse Attention (DSA)**: a
per-layer "lightning indexer" that, for every query, scores all cached keys and
keeps only the top `index_topk` (2048) — feeding those into sparse MLA attention
so attention cost stays bounded at long context.

It records the design, the accuracy bugs found while bringing V3.2 up (several of
which were *not* DSA-specific), the CUDA-graph and memory pitfalls, the sparse
prefill implementation, and the validation runs.

Model config (DeepSeek-V3.2): `index_topk=2048`, `index_n_heads=64`,
`index_head_dim=128`, `qk_rope_head_dim=64`, `kv_lora_rank=512`, MoE routing
`sigmoid` + group-limited (`n_group=8`, `topk_group=4`), FP8 block-quant weights
with `quantization_config.scale_fmt="ue8m0"`.

---

## 1. Architecture

V3.2's config and weights are a strict superset of V3 — identical MLA
projections, MoE block, YaRN RoPE, FP8 block-quant — so everything except
attention is inherited from `gllm/models/deepseek_v2.py`. The additions live in
`gllm/models/deepseek_v32.py`:

- **`DeepseekV32Indexer`**: a lightweight side path
  (`self_attn.indexer.{wq_b, wk, k_norm, weights_proj}`). Per token it computes a
  64-head indexer query and a single-head key (LayerNorm'd), applies
  **non-interleaved (neox) RoPE** to the rope slice (unlike MLA's interleaved
  RoPE), and scores `score[q,k] = Σ_h weight[q,h] · ReLU(softmax_scale · q[q,h,:] · k[k,:])`.
- **`DeepseekV32MLAAttention`**: inherits MLA, adds the indexer, writes the
  indexer key into a parallel **paged index-key cache** (`store_index_k`), selects
  per-query top-`index_topk` **physical KV slots**, and passes them to
  `MLAAttention` so it runs the *sparse* FlashMLA kernels.

For any sequence no longer than `index_topk` the top-k selects **all** keys, so
DSA is mathematically identical to dense MLA there. This is the correctness
oracle used throughout: for prompts ≤ 2048, sparse output must equal dense.

### Physical-slot indexing

The sparse kernels consume `indices` as **absolute physical KV cache slots**
(`block_table[i, p // page_sz] * page_sz + p % page_sz`), not per-sequence token
positions — FlashMLA does not re-apply the page table to sparse indices. So the
page-table transform (token position → physical slot) is folded into the
selector, and the same physical-slot convention is used everywhere (decode,
prefill, bf16 and fp8 caches).

---

## 2. Decode DSA

`DeepseekV32MLAAttention._select_topk_decode` scores each decode query (1 per
sequence) against its entire cached history, top-k's, and returns
`[num_decode, index_topk]` int32 physical slots (−1 padded). `MLAAttention`
routes by cache dtype (`_forward_decode`):

- **bf16 cache** → FA3 `flash_attn_with_kvcache` with the top-k slots as a
  `page_size=1` page table (paged bf16 sparse is rejected by FlashMLA on SM90).
- **fp8 cache** → FlashMLA sparse `flash_mla_with_kvcache(..., indices=..., is_fp8_kvcache=True)`.

The scorer is shared with prefill via `_topk_slots` (see §4).

---

## 3. Accuracy bugs found during bring-up

V3.2 initially scored ~73 on MMLU-Pro vs vLLM's ~78. Three independent bugs, only
one DSA-specific:

### 3.1 MoE double-sigmoid (primary, affects all DeepSeek-V3 models)

`gllm/layers/moe/topk.py::fused_grouped_topk` pre-computed `scores =
gating.sigmoid()` and fed **that** to `sgl_kernel.moe_fused_gate`, which applies
sigmoid **internally** — i.e. `sigmoid(sigmoid(logits))`. Because sigmoid is
monotonic the ranking is mostly preserved (so it degraded a few points rather
than crashing), but the correction-bias add and the group top-2 sum happen on a
compressed scale, shifting expert selection for every token.

**Fix**: pass the **raw** router logits + **raw** correction bias straight to
`moe_fused_gate` (route in float32 to match the HF reference). Numerically:
raw-logits matches HF `modeling_deepseek_v32` routing 16/16 on indices, weights
to ~3e-8; the old path matched 0/N. `routed_scaling_factor` passed to the kernel
is inert (the kernel doesn't apply it; gLLM renormalizes, then MoEBlock multiplies
by 2.5 separately).

### 3.2 UE8M0 activation-scale rounding (config-driven)

The checkpoint declares `scale_fmt="ue8m0"`: FP8 group scales are rounded **up to
a power of two** (`sf = 2^ceil(log2(amax/448))`), for both weights (offline) and
activations (online). gLLM used a plain `amax/448` activation scale — the
un-aligned path. Confirmed against the reference `inference/kernel.py::act_quant`
(`round_scale=True`) and HF `deep_gemm.per_token_cast_to_fp8(use_ue8m0=True)`.

**Fix**: a `round_scale` constexpr in the Triton per-token-group quant kernels
(`gllm/layers/quantization/fp8.py`), enabled **from config**
(`quantization_config.scale_fmt == "ue8m0"`) — threaded through the dense linear
path (`fp8LinearMethod`, bypassing the sgl/flashinfer fast paths which don't
round) **and** the MoE experts (`_fp8_quantize` via `Fp8MoEMethod`). DeepGEMM's
`fp8_gemm_nt` accepts plain fp32 scales and uses their values as-is (it only does
TMA layout repacking, not value rounding), so feeding rounded scales works and
measurably changes output.

### 3.3 CUDA-graph gibberish (DSA-specific, pre-existing)

With CUDA graph enabled (the api_server default), V3.2 produced pure gibberish
(`1. (1# # -*- coding ...`). Bisecting proved it was **not** the MoE/UE8M0 fixes
(neutralizing both still gibberished under graph). Root cause: the decode
selector used `max_L = int(decode_seq_lens_cpu.max().item())` — a **data-dependent
dynamic shape**. CUDA graph freezes shapes at capture time (dummy seq_len≈2), so
on replay every downstream tensor ran at the captured `max_L`, truncating the KV
history and corrupting the top-k. V3-only path (V3 has no indexer), which is why
other models were fine.

**Fix**: make `max_L = block_table.shape[1] * page_sz` **static** (full addressable
block-table width); the real per-seq length is applied via a GPU mask
(`valid[i,p] = p < seq_len[i]`), which stays correct per replay. See §5 for the
memory tiling this required.

**MMLU-Pro trajectory**: 72.86 (baseline) → 77.79 (§3.1–3.3 fixed, bf16 cache) →
78.64 (+ prefill DSA); vLLM reference 78.29. FP8 KV cache: 77.36 (expected FP8
quality trade-off).

---

## 4. Prefill DSA

Prefill originally fell back to dense MLA (exact for prompts ≤ 2048, an
approximation beyond). Wiring DSA into prefill is harder than decode: prefill has
**many query tokens per sequence, each with its own causal horizon** — the query
at absolute position `p` may attend only to keys `[0, p]`.

The reference builds a dense `[bsz, seqlen, seqlen]` causal-masked score matrix
and top-k's per row — incompatible with gLLM's varlen/paged flash kernels.
Instead, following SGLang's NSA backend, gLLM computes **per-prefill-query top-k
physical slots** and feeds the sparse kernel.

### `_select_topk_prefill` + shared `_topk_slots`

`_topk_slots` (extracted from the decode selector) is the shared scorer: given a
per-row block table and a per-row `valid` mask, it maps positions → physical
slots, scores (tiled, see §5), masks, top-k's, and returns physical slots. Decode
and prefill differ only in the mask:

- **decode**: `valid[i,p] = p < seq_len[i]`.
- **prefill**: `valid[t,p] = (p <= abs_pos[t]) & (p < seq_len[t])`, where
  `abs_pos[t] = context_len[seq(t)] + intra-seq offset`. Each prefill token's
  sequence, block-table row, and abs position are recovered from
  `query_start_loc` (bucketize) + new `MLACommonPrefillMetadata.seq_lens` /
  `context_lens` fields.

### `_forward_prefill_sparse`

Mirrors the absorbed decode path: absorb `q_nope → latent` via `W_UK_T`, call
`flash_mla_sparse_fwd(q, kv, indices, sm_scale, d_v)`, then `W_UV` (`_v_up_proj`).
Because the top-k slots already span each query's **full** causal history (cached
prefix + earlier new tokens — all written to the KV cache before the call), the
sparse call **replaces both** the dense new-token (suffix) and context (prefix)
pieces; no `merge_attn_states`.

`kv` is the flat latent cache the physical slots index into:
- **bf16 cache** → a direct `reshape(-1, 1, dim)` view.
- **fp8 cache** → dequantized to that layout (see §6).

---

## 5. CUDA-graph safety & memory tiling

Two shape/memory pitfalls, both from the `[rows, max_L, dim]` key gather:

1. **Decode (graph-captured)**: needs static shapes (§3.3). `max_L` is the full
   block-table width, so the naive full gather `[num_decode, max_L, dim]` fp32 is
   multiple GB and OOMs during large-bucket capture. **Fix**: tile the position
   axis (`_INDEX_SCORE_TILE=512`) — fixed tile + static tile count keep every
   per-tile tensor's shape static, capping scratch at `[rows, tile, dim]`.

2. **Prefill (eager, NOT graph-captured)**: only decode buckets are captured, so
   prefill can use a **dynamic** `max_L` = actual longest sequence (sliced block
   table), which the profile-run OOM'd on when it was static. But the key gather
   also scales with `num_prefill_tokens` (a full `maxp`-token prefill), so tile
   the **query axis** too (`_INDEX_QUERY_CHUNK=512`). Together the two tilings cap
   peak scratch at `[512, 512, 128]` ≈ 0.13 GB regardless of context length or
   prefill batch size.

Head padding: `flash_mla_sparse_fwd` requires the query head count to be a
multiple of 64 (SM90) / 128 (SM100+). Under TP the per-rank head count is smaller
(128/8 = 16), so zero-pad the head axis and trim the output (matches SGLang).

---

## 6. FP8 MLA cache: sparse-prefill dequant (gather-only)

The V3.2 FP8-packed MLA cache uses a 656-byte/token layout (512 fp8 nope + 4×fp32
per-tile scales + 64 bf16 rope). `flash_mla_sparse_fwd` needs a bf16 latent buffer
its physical-slot indices can address, so the cache is dequantized first.

`dequant_mla_fp8_slots` (`gllm/layers/ops/cache_kernels.py`): dequantizes **only
the physical slots the top-k actually references** (`torch.unique(topk_indices)`,
≥0) into a **full-width, physical-slot-indexed** bf16 buffer. Unreferenced rows
are never written (the kernel never reads them), so absolute slots index the
output directly — **no index remapping**.

**Contrast with SGLang** (`dequantize_k_cache_paged`): SGLang dequantizes each
sequence's whole KV history into a **compact** buffer and remaps the top-k
indices into that compact space (fused into its top-k kernel). Trade-offs:

| | gLLM (physical-slot) | SGLang (compact) |
|---|---|---|
| dequant amount (long ctx) | only selected slots | whole seq history |
| dequant amount (prefix-shared) | unique → once | may repeat shared slots |
| output buffer | full-width (sparsely filled) | compact (minimal) |
| index remapping | none | required (fused into top-k) |

i.e. gLLM trades a full-width buffer allocation for less dequant compute and a
simpler, remapping-free index space; SGLang trades a heavier index-transform
machine for minimal memory. A compact gather is a possible follow-up if the
full-width allocation ever dominates.

---

## 7. Validation

- **Dense-equivalence oracle** (bf16 & fp8): a 631-token prompt (≤ index_topk)
  produces **identical** output with sparse prefill on vs forced off — confirming
  sparse == dense when the causal horizon ≤ index_topk.
- **RULER 4096** (bf16, >2048 → sparse active): **96.15%** (250/260); all
  needle-in-haystack / variable-tracking / word-extraction tasks at 100%,
  QA at 70–80. Proves the indexer selects the right KV at long context (a wrong
  top-k would miss the needle).
- **MMLU-Pro regression**: **78.64%** bf16 (no short-context break; ≥ the 77.79
  pre-prefill-DSA number), **77.36%** fp8 KV cache; empty responses 0/1400 in
  both — no gibberish, prefix-cache poisoning fix holds at concurrency 128.
- **FP8 dequant unit test**: `dequant_mla_fp8_slots` matches the full-cache
  dequant on referenced slots to 0.0 abs diff.

### 7.1 gLLM vs vLLM alignment (RULER 4096, full FP8 indexer scoring)

Head-to-head against vLLM 0.22 (same DeepSeek-V3.2, tp8, EP, max-len 8192,
enforce-eager), all runs `num_per_task=20`, concurrency 16:

| engine / config                     | RULER 4096         |
| ------------------------------------ | ------------------ |
| vLLM (fp8 indexer + UE8M0)           | 96.54 / 96.15 (two runs) |
| gLLM fp32 scoring                    | 96.15              |
| gLLM full-fp8 + UE8M0 (Hadamard off) | 95.38 (248/260)    |
| gLLM full-fp8 + UE8M0 (Hadamard on)  | 94.23 / 95.77 (two runs) |
| gLLM full-fp8, no UE8M0              | 93.85              |

Findings from the alignment deep-dive:

- **Run-to-run nondeterminism dominates the "~1pt gap".** Same server, same
  greedy config, two runs differ by up to ~1.5pt — all variance in the fuzzy
  generative tasks (`qa`, `cwe`, `vt`); the exact-match tasks (niah, fwe) are
  100% on every run and both engines. vLLM itself varies 96.54↔96.15. The cause
  is batch-composition-dependent floating-point reduction order flipping greedy
  argmax at tie-break points.
- **Q and K FP8 quant are bit-exact with vLLM.** vLLM's indexer hard-sets
  `scale_fmt="ue8m0"` and quantizes q via `per_token_group_quant_fp8(use_ue8m0=True)`
  and k via `indexer_k_quant_and_cache(scale_fmt="ue8m0")`. Running both engines'
  kernels on identical input: q scale identical (0.250000), **k fp8 bytes 100.00%
  identical (25600/25600, zero off-by-1 ulp)**. UE8M0 formula
  `exp2(ceil(log2(scale)))` is character-identical on both sides.
- **UE8M0 matters, and gLLM matches vLLM's use of it.** Enabling UE8M0 lifted
  gLLM full-fp8 from 93.85 → 95.38. (A `sgl_kernel==0.3.21` build was found to have
  a no-op `scale_ue8m0` flag — that is a build bug, not the reference semantics.)
- **Hadamard is net-negative on this model → default OFF.** gLLM applied a
  Hadamard transform to q/k before fp8 quant (SGLang's NSA recipe); vLLM does not.
  Per-question diffing showed the only task systematically lost was `vt` (variable
  tracking): 15/20 with Hadamard vs 19/20 without, matching vLLM's 20/20. Logprob
  analysis: Hadamard flattens the output distribution at the answer's opening
  greedy tie-break, so the model takes a different narrative path and drops a
  variable from the long answer. Disabling it recovered all 5 lost `vt` questions
  and moved logprobs toward vLLM (identical top-3 at the first generated token,
  Δlp≈0.008). See §9.
- **chat template is identical**: the model ships a standalone `chat_template.jinja`
  (not in `tokenizer_config.json`); both engines load it and render token-identical
  `input_ids`, so the prompt is not a variable.
- **Single-request determinism**: gLLM in steady state is bit-deterministic across
  repeated single requests (logprob spread 0.0); vLLM is NOT (spread up to ~1.08,
  token sequence diverges by pos 13) — so per-token logprob equality between the
  two is not achievable regardless, but the distributions align.

RULER eval client: `benchmarks/evaluate_ruler.py` (buckets 4096/8192/16384).

---

## 8. Usage

```
# bf16 MLA cache (default), CUDA graph on:
python -m gllm.entrypoints.api_server --model-path <V3.2> \
  --tp 8 --enable-ep --page-size 64 --mla-decode-backend flashmla \
  --gpu-memory-util 0.6 --maxd 512 --model-max-length 8192 --port 8100

# FP8 MLA cache (smaller KV, ~1pt MMLU trade-off):
#   add  --mla-cache-dtype fp8
```

DSA (decode + prefill) is always on for V3.2; it is a no-op for prompts ≤
`index_topk` and only diverges from dense beyond that.

## 9. FP8 indexer scoring (default ON, `GLLM_DSA_FP8_SCORE=1`)

The indexer scores via SGLang's FP8 path by default (`GLLM_DSA_FP8_SCORE=1`) —
10-50× faster indexer scoring at long context, verified bit-aligned with vLLM's
quant (§7.1). Set `GLLM_DSA_FP8_SCORE=0` to fall back to the exact fp32 einsum
selector (`_select_topk_prefill` / `_select_topk_decode`).

- **prefill** (`_select_topk_prefill_fp8`): reads the fp32 index-K cache, applies
  Hadamard to q and k, quantizes to e4m3 + per-token scale, scores with
  `deep_gemm.fp8_mqa_logits` (ragged, per-query causal `ks/ke`). No cache change.
- **decode** (`_select_topk_decode_fp8`): scores with `deep_gemm.fp8_paged_mqa_logits`
  over a **persistent paged FP8 index-K cache** (`index_k_fp8_cache`, 132-byte
  block-contiguous layout: per page `[page_size*128 fp8][page_size*4 scale]`),
  written by the `store_index_k_fp8` Triton kernel. Both the metadata call and the
  kernel are CUDA-graph-capturable, so the decode selector stays graph-safe.

Key correctness rules (both learned the hard way):
- `fp8_mqa_logits` / `fp8_paged_mqa_logits` apply ReLU internally and fold the
  per-head weights (carrying `q_scale * softmax_scale`) + cache scale — matching
  `Σ_h w·ReLU(scale·q·k)`.
- **Hadamard is OFF by default** (`GLLM_DSA_HADAMARD=0`): it was net-negative on
  RULER (§7.1). When re-enabled, the FP8 path scores `Hadamard(q)·Hadamard(k)`, so
  BOTH q and k must be Hadamard'd: H is orthogonal so `(Hq)·(Hk)=q·k`, but
  `(Hq)·k ≠ q·k`. Hadamard-ing only q (and storing the raw key) silently corrupts
  selection (RULER 84% vs 94%). The two sides are toggled together by the flag.
- `store_index_k_fp8` must write the fp8 **byte pattern** via a `float8e4nv`-cast
  pointer, not `.to(uint8)` (which truncates the fp8 value to an integer).

**Accuracy (RULER 4096, retrieval):** fp32 96.15% · full-fp8 + UE8M0, Hadamard off
(default) 95.38% (248/260) · full-fp8 + UE8M0, Hadamard on 94.23–95.77% (run-to-run)
· full-fp8 without UE8M0 93.85%. vLLM reference 96.15–96.54%. UE8M0 is required for
alignment; Hadamard hurts (default off). See §7.1 for the full comparison.

## 10. Known limitations / follow-ups

- FP8 sparse-prefill dequant (KV read) is gather-only but still allocates a
  full-width bf16 buffer; a compact gather + remap would cut that allocation.
- The prefill FP8 selector re-quantizes K from the fp32 index cache each layer; it
  could read the persistent paged FP8 index cache (as decode does) to avoid that.
