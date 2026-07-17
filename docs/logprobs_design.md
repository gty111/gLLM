# Logprobs & Prompt-Logprobs — Design & Usage

This document describes how gLLM returns **log probabilities** for both the
generated tokens (`logprobs`) and the input prompt tokens (`prompt_logprobs`),
covering the OpenAI-compatible API surface, the end-to-end data flow through the
worker/scheduler/frontend, the special handling needed for Pipeline Parallelism
(PP) and Tensor Parallelism (TP), and the current limitations.

---

## 1. What logprobs mean

For a token at position `i`, its logprob is the model's conditional
log-probability given all preceding tokens:

```
logprob_i = log P(token_i | token_0, ..., token_{i-1})
```

gLLM exposes two flavors:

- **`logprobs`** (generation): the logprob of each token the model *samples*,
  optionally with the top-`k` alternative tokens at that step. Computed once per
  decode step.
- **`prompt_logprobs`** (input scoring): the logprob the model assigns to each
  token of the *input prompt*, i.e. how "expected" your prompt text is to the
  model. Computed during prefill. Useful for perplexity, candidate re-ranking,
  zero-shot classification via prompt scoring, and OOD detection.

Both are the same quantity (a conditional log-softmax value); they differ only in
whether they apply to generated tokens or to the given input tokens. The first
prompt token always has `null` (no preceding context).

---

## 2. API

### 2.1 Chat Completions (`/v1/chat/completions`)

Mirrors the OpenAI schema:

| Field | Type | Meaning |
|-------|------|---------|
| `logprobs` | bool | Turn generation logprobs on. |
| `top_logprobs` | int (0–20) | How many alternatives to report per token (requires `logprobs=true`). |
| `prompt_logprobs` | int (0–20) | Turn prompt logprobs on and report this many alternatives per prompt token. (gLLM extension.) |

Response: each choice carries `logprobs.content` (list of per-token entries) and
`prompt_logprobs` (list aligned with the prompt tokens).

### 2.2 Completions (`/v1/completions`)

| Field | Type | Meaning |
|-------|------|---------|
| `logprobs` | int | Number of top alternatives per generated token (the sampled token's logprob is always included). `null` disables. |
| `prompt_logprobs` | int (0–20) | Prompt logprobs with this many alternatives. |

Response: `logprobs` is the OpenAI `CompletionLogProbs` object
(`tokens`, `token_logprobs`, `top_logprobs`, `text_offset`); `prompt_logprobs`
is the gLLM extension list.

### 2.3 Per-token entry shape

Every per-token entry (used by both generation and prompt logprobs) looks like:

```json
{
  "token_id": 9822,
  "token": " France",
  "logprob": -3.71,
  "bytes": [32, 70, 114, 97, 110, 99, 101],
  "top_logprobs": [
    { "token_id": 9822, "token": " France", "logprob": -3.71, "bytes": [...] },
    { "token_id": 5578, "token": " Paris",  "logprob": -4.02, "bytes": [...] }
  ]
}
```

`prompt_logprobs` is a `List[Optional[Dict]]` of length `prompt_len`, where index
0 is `null` and every later index is an entry like the one above (for the actual
prompt token at that position). It is delivered once, on the streaming chunk
where the prompt finishes prefill.

Both requests clamp the alternative count to **20** (the OpenAI ceiling) to bound
the per-step top-k work — see `api_server.create_chat_completion` /
`create_completion`.

---

## 3. Where the numbers come from

Logprobs are computed on the **output rank** (the last PP stage's `tp0`), the only
rank that holds the gathered full-vocab logits.

### 3.1 Generation logprobs

`Sampler.compute_logprobs` (`gllm/layers/sampler.py`) returns, on GPU:

- `sampled` `[B]` — the logprob of the chosen token (always reported), and
- `top_vals` / `top_ids` `[B, k]` — the top-`k` alternatives.

It runs on the **same logits used to sample**, i.e. *after* temperature scaling
and repetition penalty. So for greedy decoding (temperature effectively 1 via the
argmax shortcut) the values are the raw model logprobs; for `temperature != 1` or
with penalties they reflect the *effective* sampling distribution. (This is a
deliberate choice — see §6.)

### 3.2 Prompt logprobs

`ModelRunner._compute_prompt_logprobs` runs the LM head over *every* prefill
position of a requesting sequence via the model-level `logits_from_hidden`
method, takes `log_softmax`, and records the logprob of the *actual next* prompt
token at each position (plus top-k). It:

- is computed from **raw** hidden→logits (no temperature/penalty), which is the
  right semantics for scoring given text;
- handles **chunked prefill** by filling only the positions the current chunk
  covers; prefix-cache-skipped positions stay `null`;
- accumulates into `Sequence.prompt_logprobs_data`, a list of length
  `raw_prompt_len` (`None` or `(token_id, logprob, top_ids, top_vals)`).

### 3.3 `logits_from_hidden`

To keep LM-head placement a model-internal detail (tied weights, TP gather,
multimodal nesting), every model exposes:

```python
def logits_from_hidden(self, hidden_states):  # -> full-vocab logits
    return self.lm_head(hidden_states)
```

Leaf CausalLM models implement it directly; `compute_logits` now calls it after
selecting each seq's last position. Multimodal wrappers (Qwen*-VL, Kimi-K2.5)
delegate to their inner `language_model`. A model lacking it simply doesn't
support prompt logprobs (silent no-op).

---

## 4. Data flow

```
                       output rank (last PP stage, tp0)
  ┌────────────────────────────────────────────────────────────┐
  │  compute_logits ──► logits (full vocab, TP-gathered)         │
  │       │                                                      │
  │       ├─ Sampler.forward_gpu(return_logprobs) ─► next_tokens │
  │       │                                          + (sampled, │
  │       │                                            top_vals, │
  │       │                                            top_ids)  │
  │       │                                                      │
  │       └─ _compute_prompt_logprobs (prefill only)            │
  │              └─► seq.prompt_logprobs_data                    │
  │                  (PP>1: also -> _last_prompt_logprobs, sent  │
  │                   to rank 0 over the token socket)           │
  └────────────────────────────────────────────────────────────┘
              │ per-batch-row list                 │
              ▼                                     ▼
     scheduler.add_next_tokens(next_tokens, logprobs)
              │
              ▼
     process_output / process_output_finalize
        · IPCPackage.logprobs[]     (aligned with next_tokens)
        · IPCPackage.prompt_logprobs{seq_id: data}  (emitted once)
              │  ZMQ output socket
              ▼
     LLM._apply_ipc_package (frontend)
        · _make_logprob_entry / _make_prompt_logprobs  (decode ids → pieces)
        · StreamOutput(text, logprob, prompt_logprobs)  ─► AsyncStream
              │
              ▼
     serving_chat / serving_completions  ─► OpenAI JSON
```

Key carriers:

- **`ModelRunner._last_logprobs`** — the per-batch-row generation logprobs from
  the latest non-overlap `step_once`; a list aligned with the batch, `None` for
  seqs that didn't request logprobs, else `(sampled, top_ids, top_vals)` sliced
  to that seq's `num_top_logprobs`.
- **Overlap path** stages logprobs into double-buffered pinned tensors
  (`_lp_sampled_bufs` / `_lp_topval_bufs` / `_lp_topid_bufs`) alongside the
  sampled-token buffer, then `OverlapWorker._read_logprobs` materializes them
  after `copy_done` fires.
- **`IPCPackage.logprobs`** (list) and **`IPCPackage.prompt_logprobs`** (dict
  keyed by seq_id) — empty on the common no-logprobs path, so they add nothing to
  the pickled payload.
- **`StreamOutput`** (`gllm/utils/__init__.py`) — wraps `text`, `logprob`,
  `prompt_logprobs` so the async stream carries structured data, not a bare str.

`Scheduler._attach_prompt_logprobs` emits a seq's completed
`prompt_logprobs_data` exactly once (on the step its prompt finishes prefill),
latching `_prompt_logprobs_sent` so later decode steps don't resend it.

---

## 5. Parallelism

### 5.1 Tensor Parallelism (TP > 1)

- **Generation logprobs: supported.** They reuse the full-vocab logits that
  `compute_logits` already all-gathers on every TP rank, so `compute_logprobs`
  (a local `log_softmax` + top-k) only runs on the output rank and needs no extra
  collective.
- **Prompt logprobs: supported.** `_compute_prompt_logprobs` re-runs the LM head
  via `logits_from_hidden`, and `ParallelLMHead.forward` issues a
  `tensor_model_parallel_all_gather`. This collective must be balanced, so the
  helper is invoked on **every** TP rank of the last PP stage — not just the
  output rank. That is safe because every TP rank of the stage holds identical
  seqs (real `Sequence` for PP=1, identical `FollowerSeq` mirrors for PP>1) with
  full `token_ids`, `raw_prompt_len`, `computed_token_num`, and identical
  `hidden_states` / `query_start_loc`. The per-seq `project` calls therefore
  match count and
  shape bit-for-bit across ranks (including heterogeneous batches, since all
  ranks agree on which seqs requested it), the all-gather stays balanced, every
  rank computes the same result, and only the output rank's copy is shipped to
  the frontend (the rest are discarded).

### 5.2 Pipeline Parallelism (PP > 1)

- **Generation logprobs: supported.** The sampling rank is a follower on the last
  PP stage. `SeqRegister` / `FollowerSeq` carry `logprobs_enabled` and
  `num_top_logprobs` so the follower knows what to compute. The sampled tokens
  **and** their logprobs ride the same token ZMQ socket back to rank 0
  (`send_tokens((next_tokens, logprobs))` / `recv_tokens`), where
  `process_output` attaches them.
- **Prompt logprobs: supported.** The output-rank follower has the hidden
  states + LM head, so it computes prompt logprobs locally. `SeqRegister` /
  `FollowerSeq` now mirror `prompt_logprobs_enabled`, `num_prompt_logprobs` and
  `raw_prompt_len`, and the follower keeps the full prompt `token_ids`
  (`_keeps_token_ids` is forced on) to look up the target token at each
  position. `_compute_prompt_logprobs` accumulates into the mirror's
  `prompt_logprobs_data`; the step a prompt finishes prefill, the completed list
  is stashed in `ModelRunner._last_prompt_logprobs` and shipped to rank 0 over
  the **token socket** alongside the sampled tokens + generation logprobs
  (`send_tokens((next_tokens, gen_logprobs, prompt_logprobs))`). Rank 0's
  `process_output` attaches the received dict to the `IPCPackage` keyed by
  seq_id. PP does not change reduction order, so PP=2 output is **bit-identical**
  to PP=1.

### 5.3 Summary

| | Generation `logprobs` | `prompt_logprobs` |
|---|:---:|:---:|
| Single GPU | ✅ | ✅ |
| TP > 1 | ✅ | ✅ |
| PP > 1 | ✅ | ✅ |
| TP > 1 **and** PP > 1 | ✅ | ✅ |
| Overlap scheduling | ✅ | ✅ (PP=1) |

---

## 6. Notes & limitations

- **Generation logprobs use post-temperature/penalty logits.** They match the
  effective sampling distribution, not necessarily the raw model distribution.
  For greedy decoding they coincide (argmax skips the temperature divide). If raw
  logprobs are needed regardless of sampling params, capture them before the
  in-place temperature/penalty ops in `Sampler.forward_gpu`.
- **`prompt_logprobs` is a gLLM extension** (not part of the OpenAI schema),
  attached to the response choice. It is supported across single-GPU, TP>1,
  PP>1, and TP>1+PP>1 (see §5).
- **`text_offset`** (completions) accumulates the length of the detokenized delta,
  while each token string comes from a single-id decode; these can diverge for
  multi-byte tokens (best-effort, as in OpenAI).
- **Cost.** Both paths add a full-vocab `log_softmax` + top-k, gated to run only
  when some seq in the batch actually requested logprobs, so the common path is
  unaffected. In the overlap path, `_compute_prompt_logprobs` forces a `.cpu()`
  sync per prefill chunk (only when prompt logprobs are requested). Under TP>1,
  `prompt_logprobs` runs the LM-head projection redundantly on every TP rank
  (required to keep the all-gather balanced); only the output rank's result is
  used, but the extra projection compute on the other ranks is unavoidable.

---

## 7. Validation

Verified against a dense model (`Qwen3-0.6B`) and a multimodal model
(`Qwen3.5-2B`):

- Generation logprobs: PP=1 (overlap + `--no-overlap-scheduling`), PP=2, and TP=2,
  including concurrent/batched requests and long prompts. PP=1 vs PP=2 greedy
  logprobs match numerically.
- Prompt logprobs:
  - **TP=2** output matches TP=1 within TP floating-point noise (max
    |Δlogprob| ≈ 0.07, smaller than the ≈ 0.14 seen on generation logprobs; all
    token ids identical).
  - **PP=2** output is **bit-identical** to PP=1 (max |Δlogprob| = 0, including a
    280-token chunked-prefill prompt), confirming the follower→rank-0 socket
    path and cross-chunk accumulation.
  - Heterogeneous batches (some seqs requesting it, some not), concurrent
    requests, and chat all work without deadlock on both TP>1 and PP>1.

### Quick smoke test

```bash
# Generation logprobs (works under TP/PP > 1)
curl -s localhost:8200/v1/completions -H 'Content-Type: application/json' -d '{
  "model": "<model>", "prompt": "The capital of France is",
  "max_tokens": 5, "temperature": 0, "logprobs": 3
}'

# Prompt logprobs (works under TP/PP > 1)
curl -s localhost:8200/v1/completions -H 'Content-Type: application/json' -d '{
  "model": "<model>", "prompt": "The capital of France is",
  "max_tokens": 1, "temperature": 0, "prompt_logprobs": 3
}'

# Chat logprobs
curl -s localhost:8200/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "<model>", "messages": [{"role": "user", "content": "Say hi"}],
  "max_tokens": 5, "logprobs": true, "top_logprobs": 2
}'
```

---

## 8. Touched files

| File | Role |
|------|------|
| `gllm/entrypoints/protocol.py` | request/response schema fields |
| `gllm/entrypoints/api_server.py` | parse + clamp request params |
| `gllm/entrypoints/serving_chat.py`, `serving_completions.py` | build OpenAI logprob objects |
| `gllm/layers/sampler.py` | `compute_logprobs` on GPU |
| `gllm/model_runner.py` | `_build_logprob_rows`, `_compute_prompt_logprobs`, overlap staging |
| `gllm/models/*.py` | `logits_from_hidden` (LM-head projection) |
| `gllm/scheduler.py` | queue logprobs with tokens, `_attach_prompt_logprobs` |
| `gllm/comm.py` | `IPCPackage.logprobs` / `prompt_logprobs`, token socket tuple |
| `gllm/worker.py`, `overlap_worker.py` | carry logprobs alongside tokens (incl. PP>1) |
| `gllm/dist_schedule.py` | `SeqRegister` / `FollowerSeq` logprob flags (PP>1) |
| `gllm/sequence.py` | per-seq logprob flags + `prompt_logprobs_data` |
| `gllm/llm_engine.py`, `async_llm_engine.py` | decode ids → entries, plumb params |
| `gllm/utils/__init__.py` | `StreamOutput` |
