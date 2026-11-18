# DSpark speculative decoding

DSpark is a block draft model: from the target's hidden states it proposes
`k` tokens in one parallel forward instead of `k` autoregressive steps. gLLM
runs it as a drafter of the shared MTP speculative pipeline, so it gets the
same verify, acceptance, GDN checkpoint promotion, GPU input preparation,
CUDA graphs, overlap scheduling, prefix caching and structured-output
handling as MTP.

## Usage

```sh
python -m gllm.entrypoints.api_server \
    --model-path Qwen/Qwen3.8-27B \
    --speculative-backend dspark \
    --dspark-model-path /path/to/Qwen3.8-27B-DSpark \
    [--speculative-num-draft-tokens K]   # 1..7, default 7
```

* Single GPU only for now (TP = PP = DP = 1): the draft reads the target's
  hidden taps on the same device.
* Attention backend `fa3`, `flashinfer` (default) or `fa4`, chosen
  explicitly; a startup fallback to another backend is an error.
* Greedy and sampled requests both speculate. Sampled requests draw each
  draft from the request's temperature/top-k/top-p distribution and are
  verified with lossless rejection sampling, as with MTP. Batches with
  logprobs requests decode without speculation.
* Acceptance is logged as `MTP metrics: Mean acceptance length ...`.

## How it plugs into the MTP pipeline

`speculative/dspark_drafter.py` implements the two drafter operations:

* **`kv_sync`** runs after every target forward (prefill chunks, mixed
  batches, plain decode and verify; inside the verify CUDA graph too). The target backbone writes the hidden states of the draft's tap
  layers to a persistent buffer during every forward
  (`Qwen3_5Model.set_hidden_tap_sink`); the draft projects them into K/V for
  the same positions and slots. Verify windows are written whole: rows past
  the accepted prefix are overwritten by the next window before any draft
  attention can read them.
* **`propose`** produces `k` drafts after each request's uncached last token
  `x1` from the
  draft K/V history `[0, context)`: greedy, or sampled from the request's
  transformed distribution `q`, which is kept for the accept test. Its inputs (anchor, context length,
  page table) are the `MtpGpuPrep` staging buffers, which a chained overlap
  step fills from the predecessor's device state, so drafting never touches
  the host. One CUDA graph is captured per verify bucket for greedy drafts
  and one for sampled drafts with restricted `top_k` (sparse `q`); sampled
  batches with unrestricted `top_k` draft eagerly.

The draft's K/V are extra layers of the target KV segment (the draft's
8 x 128 heads have the target's 4 x 256 bytes per token per layer), sharing
the target's page tables, slot mapping and prefix-cache pages.

## Performance

Qwen3.8-27B BF16 target, all serving defaults (CUDA graphs, overlap,
piecewise graphs, prefix caching), 32 chat prompts with thinking, 512-token
greedy outputs. Both drafters at the same `k`, one job, relative to no
speculation:

| Drafter | `k` | conc 1 | conc 4 | conc 16 | Mean acceptance length |
| --- | --- | --- | --- | --- | --- |
| MTP | 3 | 2.15x | 2.02x | 1.80x | 2.88 |
| DSpark | 3 | 2.27x | 2.20x | 1.78x | 2.93 |
| MTP | 5 | 2.32x | 2.04x | 1.77x | 3.41 |
| DSpark | 5 | 2.71x | 2.30x | 1.80x | 3.50 |
| MTP | 7 | 2.26x | 2.05x | 1.70x | 3.65 |
| DSpark | 7 | 2.91x | 2.33x | 1.92x | 3.84 |

Sampled requests (temperature 0.6, top-p 0.95, top-k 20; lossless rejection
sampling), same setup. The last column is a separate run of 64 such requests
at concurrency 16, started after the server has already handled a few
thousand requests:

| Drafter | `k` | conc 1 | conc 4 | conc 16 | Mean acceptance length | Long conc 16 acceptance |
| --- | --- | --- | --- | --- | --- | --- |
| MTP | 3 | 2.16x | 1.98x | 1.74x | 2.88 | 2.93 |
| DSpark | 3 | 2.27x | 2.06x | 1.81x | 2.83 | 2.92 |
| MTP | 5 | 2.21x | 1.84x | 1.82x | 3.40 | 3.43 |
| DSpark | 5 | 2.56x | 2.28x | 1.79x | 3.49 | 3.56 |
| MTP | 7 | 2.22x | 2.01x | 1.71x | 3.72 | 3.77 |
| DSpark | 7 | 2.71x | 2.43x | 1.85x | 3.85 | 3.90 |

DSpark drafts all `k` tokens in one forward, so a larger `k` costs it less
than MTP's `k` sequential head steps: up to `k = 7` its low-concurrency
speedup keeps growing while concurrency-16 throughput holds, whereas MTP's
speedup flattens at `k = 5` and its concurrency-16 throughput drops at
`k = 7` (the verify window grows with `k`).

## Limitations

* TP/PP/DP = 1.
* Only the Qwen3.8 DSpark checkpoint layout is supported
  (`models/qwen38_dspark.py`). `models/deepseek_v4_dspark.py` is modeled but
  not wired to the runtime.
