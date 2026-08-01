# MTP (Multi-Token Prediction) Speculative Decoding — Design & Plan

Status: **IMPLEMENTED & VALIDATED** for DeepSeek-V3.2 (8×H20). §§1–8 below are the
original design/plan (kept for the motivation). The **as-built state, final API,
env vars, and validation results** are in §10 at the end — read that first for how
the shipped feature actually works.

This document describes how to add **MTP speculative decoding** to gLLM for the
DeepSeek-V3/V3.2 and GLM-MoE-DSA (GLM-4.5/4.6/5.2) family, whose checkpoints ship
a **built-in MTP head** (`num_nextn_predict_layers = 1`, stored as the extra
`model.layers.{num_hidden_layers}.*` block).

---

## 1. Background: what MTP is, and why it fits this family

MTP = the model ships **one extra transformer layer** (the "NextN" / MTP head)
trained to predict token *t+2* given the hidden state that produced token *t+1*
plus the embedding of token *t+1*. In one decode step you can therefore:

1. run the **main model** to get token `x1` (as today), keeping its last hidden
   state `h`;
2. feed `(h, embed(x1))` into the **MTP head** to *draft* token `x2`
   (and, by looping, `x3 … x_{1+k}`);
3. run the **main model once more** over the `k` drafted tokens to **verify**
   them in a single forward, accepting the longest correct prefix via rejection
   sampling.

Net effect: up to `k+1` tokens committed per target forward instead of 1, at the
cost of one extra (cheap, single-layer) draft forward per drafted token. Typical
acceptance on this family gives **~1.8–2.5× decode throughput**.

Why this family is the easy case:

* The MTP head is **architecturally identical to a main decoder layer**
  (MLA + MoE + DSA indexer) — gLLM already has all of it in
  `deepseek_v2.py` / `deepseek_v32.py`. The head is just `enorm`, `hnorm`,
  `eh_proj`, one `DeepseekV2DecoderLayer`, and a `shared_head.norm`.
* It **shares the target model's KV cache layout** (same MLA latent cache, same
  DSA index cache) — we do NOT need a separate draft KV pool. The MTP head is an
  extra module that writes into the same paged cache.
* `num_nextn_predict_layers = 1`, so multi-step drafting **reuses the single MTP
  layer cyclically** (`spec_step_idx % 1 == 0`). No tree needed — a **linear
  chain (topk=1)** is enough and is the recommended default for this family
  (`num_steps=3, topk=1`).

### GLM-5.2 specifics

* MTP head is `model.layers.78.*` (config `num_hidden_layers = 78`,
  `num_nextn_predict_layers = 1`). Weight keys:
  `enorm`, `hnorm`, `eh_proj`, `shared_head.norm`, plus a full decoder layer's
  `self_attn.*` / `mlp.*` (MoE) / `input_layernorm` / `post_attention_layernorm`.
* `index_share_for_mtp_iteration = true` in the config → the MTP head **reuses
  the last full layer's DSA top-k index selection** rather than recomputing it.
  This is a perf detail we can honor in stage 2 (see §7).

---

## 2. Chosen approach: shared-KV MTP head + one-mode-per-batch discipline

We deliberately do **NOT** build a separate draft worker + own KV pool + tree
attention + per-step attention backends + KV compaction. That is the right design
for external EAGLE draft models and tree drafting, but it is a large, invasive
rewrite for an engine that has *zero* spec infra today.

Instead we make two design choices that fit gLLM's existing engine:

* **Model structure & KV.** The MTP head is an extra module on the *same*
  `*ForCausalLM`, sharing the target's KV cache. No second worker, no second KV
  pool, no tree.
* **Batching & CUDA-graph discipline.** A batch runs **one mode at a time**
  (all-decode-verify, or all-prefill), so the verify forward is **shape-uniform**
  (`1+k` query tokens per seq) and reuses gLLM's existing batch-size-bucketed
  **full CUDA graphs** with only the token dimension changed from `1` to `1+k`.
  We do **not** mix spec-decode and prefill in one batch.

Rationale for the batching choice:

* Mixing spec + non-spec + prefill in one batch (per-request draft counts via
  `query_start_loc`) forces a non-uniform batch that **falls back to
  piecewise/eager** graphs. Forbidding mixing keeps **full graphs always live**.
* gLLM's decode path is already full-CUDA-graph + chunked-prefill (decode and
  prefill are already scheduled as separable batches), so one-mode-per-batch is a
  *smaller* change and preserves the graph speedup that matters most on a
  ~750B MoE (MoE all-to-all + weight bandwidth bound).

| Concern | Decision |
|---|---|
| Draft model | The **same** `*ForCausalLM` object + an MTP head module. No second worker/weight path beyond the head. |
| Draft KV | **Shared** with target (same MLA latent + DSA index paged cache). Draft tokens occupy real KV slots; rejected slots rolled back. |
| Draft structure | **Linear chain**, `topk = 1`. `k = num_speculative_tokens` (default 3). No tree mask. |
| Verify | **One** target forward over `1+k` tokens per decode seq (like a tiny `1+k` "prefill"), then rejection sampling. |
| **Batch composition** | **One mode per batch.** All decode seqs in a verify batch carry the **same** `1+k`. Spec-decode batches and prefill batches are **not mixed**; a new prefill request waits for the next batch. |
| **CUDA graph** | Verify reuses existing **full decode graphs**, batch-size-bucketed, token dim `1→1+k` (fixed `k`). No `(batch, variable-k)` 2-D bucket explosion. |
| Rollback | Truncate each seq back to `base_len + num_accepted`; free KV slots beyond it. |
| Scope | Decode path only. Prefill unchanged. Off by default; `num_speculative_tokens=0` ≡ today. |

---

## 3. gLLM extension points (from an architecture audit)

Control flow today (non-overlap path):
`Worker.run_pp0` → `Scheduler.schedule_once` → `ModelRunner.prepare_input`
(`InputData.cal_input`) → `ModelRunner.step_once` (`forward` → `compute_logits`
→ `Sampler.forward_gpu`) → `Scheduler.process_output` (`seq.append`).

Hard assumptions to relax (all in the **decode** path):

1. **`seq.to_compute_token_num == 1` for decode** (`scheduler.py`). MTP verify
   needs `1 + k` tokens per decode seq in the verify forward.
2. **`slot_mapping` maps one token → one slot** (`input_data.py`). Verify needs
   `1 + k` contiguous slots per seq.
3. **Sampling is final / one token per seq** (`sampler.py`). Need rejection
   sampling over `k` drafts + a bonus token. Also verified: the whole commit path
   (`Sampler`→`step_once`→IPC→`process_output`) is structurally 1-token-per-step,
   so the spec loop lives **inside the worker** and commits extra accepted tokens
   worker-side (see §4.7), rather than making the scheduler emit N tokens/seq.
4. **KV pages freed only on seq finish** (`memory_manager.py`). Need
   mid-sequence rollback of rejected draft slots.
5. **CUDA graphs assume batch×1-token decode** (`model_runner.py`, verified:
   `create_dummy_seqs` captures at `to_compute_token_num=1`). The verify forward
   is **not** a decode; it rides the **prefill-with-context** path (§5), which
   gLLM runs **eager** today. So Stage 1 verify is eager for free; a *captured*
   uniform-`1+k` verify graph is **new Stage-2 work** (not free decode-graph
   reuse). Draft steps stay token-dim `1` and *do* reuse today's decode graph.

---

## 4. Component plan

### 4.1 Model layer — `gllm/models/deepseek_mtp.py` (new)

A thin module reusing existing V2/V3.2 building blocks:

```python
class DeepseekMTPLayer(nn.Module):
    def __init__(self, config, layer_id):
        self.enorm = RMSNorm(hidden_size, eps)
        self.hnorm = RMSNorm(hidden_size, eps)
        self.eh_proj = ReplicatedLinear(2*hidden_size, hidden_size, bias=False)
        # reuse the exact decoder layer of the base model (V3.2 => DSA indexer):
        self.mtp_block = DeepseekV32DecoderLayer(glb_layer_id, layer_id, config)
        self.shared_head_norm = RMSNorm(hidden_size, eps)

    def forward(self, input_data, prev_hidden, input_ids_embeds):
        e = self.enorm(input_ids_embeds)
        h = self.hnorm(prev_hidden)
        x = self.eh_proj(torch.cat([e, h], dim=-1))
        x, residual = self.mtp_block(input_data, x, residual=None)
        x = residual + x
        return self.shared_head_norm(x)   # -> feed to the SHARED lm_head
```

Notes:
* `lm_head` and `embed_tokens` are **shared** with the base model (the MTP head's
  `shared_head.head` is tied to the main `lm_head`; do not load a second copy).
* For GLM-MoE-DSA we build `GlmMoeDsaDecoderLayer` here instead of the V3.2 one
  (same pattern, once GLM base support from the earlier plan exists).
* Weight loading: the base `weight_loader` is parameter-driven, so simply
  registering the MTP module's params under prefix `model.layers.{N}.` and adding
  its keys to the model's `weight_rules()` is enough — no `_keys_to_ignore`
  needed. The head is only *built* when spec decoding is enabled, so today's
  "MTP weights silently skipped" behavior is preserved when it's off.

### 4.2 Proposer — `gllm/spec_decode/mtp_proposer.py` (new)

Owns the **draft loop** (runs inside the worker, after the target forward):

```python
class MtpProposer:
    def propose(self, input_data, target_hidden, first_token_ids) -> draft_tokens:
        # draft_tokens: [num_decode, k]
        hidden = target_hidden               # last hidden of target forward
        tok = first_token_ids                # x1 sampled by target
        drafts = []
        for step in range(self.k):
            emb = self.model.embed_input_ids(tok)
            hidden = self.mtp.forward(draft_input_data, hidden, emb)
            logits = self.model.logits_from_hidden(hidden)
            tok = logits.argmax(-1)          # greedy chain draft
            drafts.append(tok)
            # advance positions + write draft KV slots for step+1
            draft_input_data = self._advance(draft_input_data, tok)
        return torch.stack(drafts, dim=1)
```

Each draft step:
* writes its KV into the **next real slot** of each decode seq (slots
  pre-allocated by the scheduler, see §4.4);
* advances `positions` by 1;
* for DSA: honor `index_share_for_mtp_iteration` (reuse last full layer's top-k)
  in stage 2; in stage 1 just recompute the indexer (correct, slightly slower).

### 4.3 Verifier + rejection sampler — `gllm/layers/sampler.py` (extend)

Add `verify_draft_tokens(target_logits, draft_tokens, draft_probs=None)`.

**Interface note (verified):** the existing `Sampler.forward_gpu` assumes
`logits: [batch, vocab]` (2-D; `softmax(dim=-1)` + fused top-k/top-p). The verify
logits are `[num_seq, 1+k, vocab]` (3-D). Do **not** try to thread this through
`forward_gpu` — write a **dedicated rejection/greedy-verify routine** that either
flattens to `[num_seq*(1+k), vocab]` for the sampling primitives or implements the
accept/reject scan directly. Keep it separate from `forward_gpu`.

* Target verify forward produces `[num_seq, k+1, vocab]` logits (logits at the
  position *before* each draft token + one bonus position).
* **Greedy target** (default): accept draft `d_i` iff `argmax(target_logits_i) ==
  d_i`; stop at first mismatch; emit the target's argmax at the mismatch position
  as the corrected token (the standard "≥1 token always commits" guarantee).
* **Sampling target**: standard rejection sampling
  `accept iff u < min(1, p_target(d_i)/p_draft(d_i))`, with the usual
  residual-distribution resample on rejection. Needs `draft_probs` from the
  proposer (store them in the chain loop).
* Returns per-seq `num_accepted ∈ [1, k+1]` and the committed token ids.

### 4.4 Scheduler + sequence state — `gllm/scheduler.py`, `gllm/sequence.py`

* When spec is on, for each decode seq **pre-allocate `k` extra KV slots** (so the
  draft chain has somewhere to write): `to_compute_token_num` becomes `1+k` on the
  verify forward.
* **Batch discipline (§5b):** a scheduled batch is **either** a spec-verify batch
  (all decode seqs, uniform `1+k`) **or** a prefill batch — never mixed. Reuse the
  existing chunked-prefill split that already keeps decode and prefill separable;
  when spec is on, a decode batch becomes a verify batch and new prefill requests
  are scheduled in their own batch (they wait one step). All spec
  seqs in the batch share the same `k` (shape-uniform for CUDA graph).
* New `Scheduler.process_spec_output(num_accepted, committed_tokens)`:
  * `seq.extend(committed_tokens[:num_accepted])` (append 1..k+1 tokens);
  * `memory_manager.rollback_kv(seq, base_len + num_accepted)` to free unused
    draft slots;
  * finish / logprob / stream handling as today, but possibly emitting multiple
    tokens per step.

### 4.5 Memory manager — `gllm/memory_manager.py` (extend)

Verified API: a sequence's page list is `seq.page_table` (list of page ids); pages
are freed via `segment.free(page_num)` → `IDAllocator.free` (a FIFO free-list).
The allocator's design frees a seq's pages **atomically on finish/preempt**, not
mid-sequence, and `PrefixSegment.free` is **ref-counted** for cross-sequence
prefix sharing. So the rollback must be **page-aligned and prefix-cache aware**:

```python
def rollback_kv(self, seq, keep_len):
    pages_needed = ceil(keep_len / self.page_size)
    for pid in seq.page_table[pages_needed:]:
        self.segment.free(pid)          # PrefixSegment.free decrements refcount
    seq.page_table = seq.page_table[:pages_needed]
```

Caveats to honor:
* Roll back at **page granularity**, not per token — only free pages entirely
  beyond `keep_len`. The partially-filled page stays; its stale slots are
  overwritten before the next read.
* Under prefix caching, a rolled-back page may be **shared**; rely on the
  ref-counted `PrefixSegment.free` rather than a raw allocator free. Confirm a
  draft-written page was never registered into the prefix hash before freeing
  (draft slots must not be prefix-cache-published until accepted).
* **Cheaper alternative to consider:** pre-allocate `k` extra slots *within the
  seq's existing tail page* when possible, so a rejected chain needs **no page
  free at all** (just a length rewind). Rollback only frees when the `1+k` draft
  spilled into a fresh page. This avoids most allocator churn.

MLA latent cache + DSA index cache roll back the same way (page-table
truncation). No per-slot zeroing needed.

### 4.6 Config & entry — `gllm` args + `model_loader.py`

* New args: `--num-speculative-tokens k` (0 = off, default 0),
  `--speculative-method mtp`. Auto-detect eligible when the checkpoint has
  `num_nextn_predict_layers >= 1` and arch ∈ {DeepseekV3, DeepseekV32,
  GlmMoeDsa}.
* `model_loader.py`: when spec on, after building the base `*ForCausalLM`, attach
  `MtpProposer` (which builds `DeepseekMTPLayer` and loads the head weights).

### 4.7 Where the loop lives — keep it inside `ModelRunner.step_once`

**Verified structural constraint.** The commit pipeline is **1-token-per-step end
to end**: `Sampler` returns `[batch]`, `step_once` returns a flat `next_tokens`
list, the IPC package carries one token per seq (`ipc_package.next_tokens.append`),
and `Scheduler.process_output` does `seq.append(next_tokens[idx])` (single token).
Making the scheduler/IPC path emit a *variable* number of tokens per seq is a
large, invasive refactor of the frontend protocol.

**Decision: do the entire draft→verify→accept cycle *inside* the worker forward,
and still emit exactly one “next_token” per seq to the existing pipeline**, while
committing the *extra* accepted tokens into `seq.token_ids` directly on the worker
side. Concretely:

* `step_once` (last PP rank) after the target forward: retain `hidden`
  (already available — verified at `model_runner.py:~1843`), run
  `MtpProposer.propose`, run the verify forward (prefill-with-context, §5), run
  `verify_draft_tokens`.
* The `num_accepted-1` *bonus* accepted draft tokens are appended to each seq's
  `token_ids` **and** their KV kept; the single "official" `next_token` handed
  back to the scheduler is the **last committed token** (so streaming / stop-check
  still works). The intermediate accepted tokens are surfaced to the frontend via
  a small extension to the IPC package (a per-seq `extra_tokens: list[int]`), which
  is an **additive** change, far smaller than making the whole path variable-width.
* This localizes ~all MTP complexity to the worker/runner, leaving
  `Sampler.forward_gpu`, the scheduler's batch loop, and the frontend mostly
  intact. The spec loop lives in the worker, not the scheduler.

Open sub-question: gLLM's **OverlapScheduler** pre-places token placeholders and
finalizes them a step later (`process_output_finalize`); the `extra_tokens` path
must slot into that finalize, or spec runs only on the non-overlap scheduler
first (acceptable for Stage 1).

---

## 5. Verify forward: route it through the PREFILL path, not decode

**Critical correctness point (verified against the code).** gLLM's **decode**
attention path hard-assumes **exactly one query token per sequence** and cannot
be reused for the `1+k`-token verify forward:

* `attention.py` decode branch does `q = q.unsqueeze(1); q_len = q.shape[1]` with
  an explicit comment "Decode always has q_len == 1"; the FlashMLA tile metadata
  `num_q_tokens_per_head_k = q_len * num_heads` is sized for `q_len==1`.
* The DSA decode selector `_select_topk_decode` returns `[num_decode, index_topk]`
  — **one row per sequence**, not per query token.
* Decode CUDA graphs are captured with dummy seqs at `to_compute_token_num = 1`
  (`model_runner.create_dummy_seqs`), so the shape contract is `[batch]×1`.

**Therefore the verify forward must be expressed as a "prefill with context"**,
which gLLM already fully supports via chunked prefill: a seq with `L` cached
tokens gets `1+k` new query tokens, attending to its cached `[0, L)` context plus
causal masking among the `1+k`. The relevant machinery already exists and needs
**no kernel changes**:

* `_run_prefill_context_chunk` + `_run_prefill_new_tokens` + `merge_attn_states`
  handle "q_len tokens per seq attending to `context_len` cached + causal among
  the new tokens" — exactly the verify shape.
* The DSA **prefill** selector `_select_topk_prefill` already computes **per-query
  causal top-k** (`abs_pos = context_len + intra`, one row per query token), which
  is precisely what `1+k` verify queries need. `_select_topk_prefill_fp8` too.

So the verify forward is built as InputData in the **prefill** family:

* `tokens = [x1, d1, …, dk]` per seq, `positions = [L, …, L+k]`,
  `context_lens = L`, `seq_lens = L+1+k`, `query_start_loc` groups the `1+k`
  per seq, `slot_mapping` = the `1+k` pre-allocated slots.
* This means a "verify batch" is scheduled as a **prefill-mode batch** whose
  sequences happen to each have a small `1+k` query and a large cached context —
  not as a decode batch.

**Implication for §5b/CUDA graphs (correction).** Verify does **not** reuse the
decode graph with token dim `1+k`. It rides the **prefill path, which gLLM runs
eager (uncaptured) today**. So:
* Stage 1 verify is eager "for free" — that is simply how prefill already runs.
* The CUDA-graph win for verify is **not** automatic (the earlier §5b framing was
  too optimistic). Capturing a graph for a *uniform* `1+k` prefill-with-context
  batch is a **new** capability (gLLM captures only decode today) and belongs in
  Stage 2 as real work, not a free reuse. The **uniform-batch discipline still
  matters** because it's the precondition that *makes* such a capture feasible.
* The genuinely-free graph reuse is on the **draft steps** (token dim `1`,
  identical to today's decode) — those do ride the existing decode graph.

---

## 5b. CUDA graphs & batch composition (the key perf decision)

This is the key axis for CUDA-graph performance. Two possible designs:

* **Mixing allowed:** mix spec + non-spec + prefill in one batch (per-request
  draft counts tracked via `query_start_loc`, logits split by `cu_num_logits`).
  But a non-uniform batch **cannot use full CUDA graphs** — it falls back to
  piecewise/eager. Draft steps would be captured as **per-step** graphs (each step
  = batch×1 token); verify captured only when the batch happens to be uniform.
* **Mixing forbidden:** a batch has a **single** mode (decode for draft, verify,
  or prefill). Every request in a verify batch shares the **same** number of draft
  tokens, so shapes are always static and **full CUDA graphs are always live**. A
  new prefill request waits for the next batch (explicit barrier when spec +
  DP-attn).

**gLLM decision — forbid mixing (one mode per batch):**

1. **One mode per batch.** A batch is either a *verify* batch (all decode seqs,
   each `1+k` tokens, run as **prefill-with-context**, §5) or a *prefill* batch
   (as today). gLLM's chunked-prefill scheduler already separates decode from
   prefill, so this is a small tightening, not a rewrite.
2. **Uniform `1+k` across the verify batch.** All spec seqs carry the same `1+k`
   query length → the verify forward is shape-static. This uniformity is the
   **precondition** for eventually CUDA-graph-capturing the verify forward
   (Stage 2, new work — see correction below), and it keeps rejection sampling /
   logit slicing regular.
3. **Draft steps: token dim `1`.** Each of the `k` draft-chain steps is a
   batch×1-token forward — *identical in shape to today's decode* — so it reuses
   the current decode graph as-is. Start with **per-step replay** (`k`
   replays); optionally fuse the whole chain into one captured graph
   later if the `k` launch overheads matter.
4. **Acceptance variance is post-forward, not a shape problem.** The verify
   forward always emits `1+k` logits per seq (static). How many are *accepted*
   (rejection sampling changes effective `seq_lens`) is computed **after** the
   forward, by slicing logits with `query_start_loc` and rolling back KV. The
   forward itself never sees a variable shape.

**Correction vs. an earlier draft of this section.** Verify runs on the
**prefill** path (§5), which gLLM currently executes **eager (uncaptured)**. So
verify does *not* get a free CUDA-graph by "reusing the decode graph with token
dim `1+k`" — capturing a uniform `1+k` prefill-with-context batch is a **new**
Stage-2 capability. Only the **draft steps** (token dim `1`) reuse the existing
decode graph for free. Net: Stage 1 verify is eager (which is fine and how
prefill already runs); the graph speedup for verify is real Stage-2 work, not
automatic.

---

## 6. Staging (each stage independently shippable & testable)

**Stage 0 — GLM base inference** (prerequisite, separate plan): non-speculative
GLM-5.2 running. MTP is meaningless without it.

**Stage 1 — MTP head loads + drafts, eager verify, correctness oracle.**
* Build `DeepseekMTPLayer`, load head weights, wire `MtpProposer`.
* Verify forward runs **eager** (no CUDA graph) → sidesteps the graph-shape
  problem entirely for first bring-up.
* **Correctness gate**: with greedy target + greedy draft, output token stream
  must be **bit-identical** to the non-spec run (spec decoding is exact by
  construction). This is the primary test — run the same prompt with `k=0` and
  `k=3`, assert identical generations.

**Stage 2 — perf: CUDA graph for verify + DSA index sharing.**
* **New capability:** capture a graph for the **uniform `1+k` prefill-with-context**
  verify batch (gLLM captures only decode today, so this is real work, not decode
  -graph reuse). Keyed by batch-size buckets at fixed `k`; the uniform-batch
  discipline (§5b) is what makes capture possible.
* Draft steps already reuse today's batch×1 decode graph (per-step replay);
  optionally fuse the `k`-step chain into one captured graph later.
* Honor `index_share_for_mtp_iteration` so the MTP head reuses the last full
  layer's DSA top-k (skip recompute).
* Measure acceptance rate + tokens/s; tune default `k`.

**Stage 3 — sampling path + robustness.**
* Rejection sampling for non-greedy requests (`draft_probs` plumbed through).
* Interaction with: DP-attention/EP, PP (draft loop runs on last PP rank where
  hidden + lm_head live), prefix cache, chunked prefill mixing, preemption
  during a draft chain.

---

## 7. Open questions / risks

1. **PP placement.** The MTP head + lm_head live on the **last** PP rank. The
   draft chain must run there and its drafted token ids broadcast back. Confirm
   gLLM's PP output-rank plumbing (`worker.py` output_rank) can
   carry `k` tokens + hidden.
2. **DSA index cache during draft.** Each draft token must `store_index_k` into
   the index cache and (if FP8 scoring) the FP8 index cache, then roll back on
   rejection. Confirm rollback frees index-cache slots symmetrically with the
   latent cache.
3. **CUDA-graph for verify is new work, not free reuse (corrected).** Verify runs
   on the eager prefill path, so Stage 1 works uncaptured. A captured uniform-`1+k`
   verify graph is a genuine Stage-2 task (gLLM captures only decode today). The
   uniform-batch discipline (§5b) is the enabler, not itself the speedup. Only the
   draft steps reuse the existing decode graph for free.
4. **DP/EP sync.** Draft steps are extra forwards; all DP ranks must run the same
   number of draft steps to keep MoE all-to-all in lockstep. Use a fixed-`k` (no
   early-exit) draft loop under DP-EP, accepting some wasted work. The one-mode
   -per-batch discipline (§5b) also gives the "don't mix prefill and decode when
   spec + DP-attn" barrier — reuse gLLM's DP batch-mode sync point.
5. **Overlap scheduler.** gLLM's `OverlapScheduler` places token placeholders;
   multi-token commit per step needs its finalize path (`process_output_finalize`)
   generalized to write `num_accepted` real tokens into `1+k` placeholders.

---

## 8. Rough effort estimate

| Item | Size |
|---|---|
| `deepseek_mtp.py` head module + weight rules | S (reuses existing layers; verified signatures/methods exist) |
| `MtpProposer` chain loop | M |
| Rejection / greedy-verify sampler (standalone, NOT via `forward_gpu`) | M |
| Worker-side multi-token commit (§4.7) + `extra_tokens` IPC field | M |
| KV rollback, page-aligned + prefix-cache-aware (§4.5) | M |
| `InputData` **verify = prefill-with-context** build (`1+k` per seq) | M |
| CUDA graph for uniform `1+k` verify (new capture, Stage 2) | M–L |
| DP/EP + PP + OverlapScheduler correctness (stage 3) | L |

Stage 1 (correct but eager, greedy-only) is the bulk of the value for a demo and
is **M–L-sized** (revised up from M after review: the verify-as-prefill routing
and worker-side multi-token commit are more than trivial). Stages 2–3 are where
the real perf + production-hardening effort (L) lives.

---

## 10. As-built status (DeepSeek-V3.2, 8×H20)

This section supersedes §§4–8 wherever they disagree; those describe the *plan*,
this describes what actually shipped and was validated on hardware.

### 10.1 Public API — how to enable

MTP is a constructor concern, not an env var. `LLM(...)` (→ `ModelRunner`) takes:

| param | default | meaning |
| --- | --- | --- |
| `mtp_enabled` | `None` | `None` = **auto-detect**: enable iff the checkpoint has `num_nextn_predict_layers >= 1`. `True`/`False` force on/off. |
| `mtp_k` | `3` | draft chain length (tokens drafted per target forward). |

`mtp_enabled` is stamped onto `config.mtp_enabled` and read by `deepseek_v32.py`
(`want_mtp = config.mtp_enabled and num_nextn >= 1`) to decide whether to build
`self.mtp = DeepseekMTP(...)`. `ModelRunner.init()` sets
`_mtp_k = mtp_k if model.mtp is not None else 0`.

Usage:
```python
LLM(model, tp_size=8)                 # V3.2 → MTP auto-on, k=3, fused, graphs
LLM(model, tp_size=8, mtp_enabled=False)  # force off
LLM(model, tp_size=8, mtp_k=2)            # shorter draft chain
```

### 10.2 Environment variables

Only **one** MTP env var remains (down from ~11 during development):

| env var | default | purpose |
| --- | --- | --- |
| `GLLM_MTP_FUSED` | `1` (on) | escape hatch to disable fused mode (`=0`). Fused is correct in every validated combo; the flag only exists for debugging. |

Everything else is automatic: draft/verify CUDA graphs capture whenever CUDA
graphs are on (respect `--disable-cuda-graph`); rejection sampling activates
per-batch by runtime detection (see 10.4). `GLLM_DSA_FP8_SCORE` / `GLLM_DSA_HADAMARD`
are pre-existing DSA knobs, unrelated to MTP.

### 10.3 Fused MTP (the default)

The plan's two-forward step (decode `x1` → draft → verify) collapses to **one
target forward per step**. After verify, the accept step stashes each seq's
*bonus* token + its verify hidden into `self._mtp_relay[seq_id]`. The next step's
fused fast-path (`step_once`, gated on `_mtp_fused and not is_dp_attn and
is_last_pp_rank and pure-decode and all seqs have relay`) skips the `x1`-decode
forward entirely — the relayed `(bonus, hidden)` seed the draft directly.

Bonus discipline (the subtle correctness point): the bonus is **relayed, not
committed**, when fused (it is committed once as the *next* step's `x1`).
Committing it both places double-emits it (observed as token repetition). The
non-fused path instead appends the bonus to the committed list. Both the greedy
and rejection accept branches honor this split.

The non-fused code path is retained (not dead): it is the fallback for
bootstrap (a seq's first decode step has no relay), DP+EP (fused is gated off
under DP-attn), and any relay miss.

### 10.4 Greedy vs rejection sampling

Runtime detection — no env flag:
`_rej_active = any((s.temperature>1e-5 and s.temperature != 1.0) or s.top_k != 1
for s in decode_seqs)`.

* **Greedy** (`_rej_active` false): accept draft `d_p` iff it equals the target's
  argmax at that verify position. Bit-exact vs non-MTP greedy (validated 82.14%
  = baseline on MMLU-Pro 28q).
* **Rejection sampling** (distribution-lossless under temperature/top-p/top-k):
  draft tokens are drawn from the per-step draft dist `q`; accept `d_p` with prob
  `min(1, p/q)`, on reject resample from the residual `(p-q)+`, else sample a
  bonus from `p`. The draft proposal uses the **Gumbel-max trick**
  (`argmax(q / Exp(1))`, `_gumbel_argmax`) instead of `torch.multinomial` so the
  draft step is CUDA-graph-capturable (multinomial's device-side validity assert
  is graph-hostile). `exponential_` on the default CUDA generator IS capturable
  (verified by micro-test).

### 10.5 CUDA graphs

Three graph families, all captured at init on a dedicated stream:

* **Draft chain** — `_capture_draft_graphs` captures a single draft step per
  bucket; `_draft_chain_graph` replays it `k` times, advancing positions /
  slot_mapping / seq_lens in the static input buffers between replays.
  When `_mtp_can_sample`, a second **sampled** draft step is captured
  (`_draft_step_forward_sampled`: forward + `_mtp_probs_static` + Gumbel draw + q
  stash into `_d_q`), replayed by `_draft_chain_graph_sampled`.
* **Verify** — `_capture_verify_graphs` captures the uniform `1+k`-query verify
  forward (fp8 decode-sparse kernel via `is_mtp_verify` metadata, see
  `attention.py::_forward_verify_sparse`).
* All graph paths fall back to eager when `nd > max captured bucket`.

### 10.6 TP consistency

* Draft/verify **logits are bit-identical across TP ranks** (LM-head is
  `ParallelLMHead` — per-shard `F.linear` + pure-copy `all_gather`, no
  partial-sum all-reduce → no fp epsilon). Greedy argmax is therefore identical
  on all ranks with no broadcast.
* Rejection accept uses `p`/`q` that DO carry attention-all-reduce epsilon and
  per-rank RNG draws, so `_mtp_decode` broadcasts TP-rank-0's committed token
  grid + relayed bonus token to make rank 0 authoritative.
* Draft tokens under the sampled graph are broadcast between replays
  (`_mtp_bcast_tp`) since the captured Gumbel RNG isn't guaranteed rank-synced.

### 10.7 Overlap scheduler

MTP verify runs as a **synchronous** draft→verify→accept block
(`OverlapWorker._run_mtp_sync`), not through the future-map relay pipeline: any
in-flight overlapped batch is drained first so committed token_ids have no
`-future_slot_id` placeholders. Seqs freed under us mid-step (EOS/max_len from
the drained batch) are marked `_overlap_freed` and filtered out of both the
forward batch and the `batch_running` entry (double-free fix). DP+EP is out of
scope for the sync path (variable per-step token count breaks the cross-DP
collective lockstep) → plain TP only.

### 10.8 Validation results (auto-detect path, no MTP env vars)

| scenario | accuracy | notes |
| --- | --- | --- |
| greedy MMLU-Pro 28q | **82.14%** (= baseline) | bit-exact; mean accept len ~3.14 (k=3) |
| sampling MMLU-Pro 140q (T=1.0, top_p=0.95, top_k=40) | 75.71% | rejection; mean accept len ~2.5, per-pos accept 0.79/0.51/0.17 |

Full matrix validated: greedy/sampling × fused/non-fused × overlap/non-overlap ×
draft-graph/verify-graph. Fused C=1 speedup ~2.09×.

### 10.9 Key files (as-built)

* `gllm/models/deepseek_mtp.py` — `DeepseekMTP` head (layer-61 nextn block).
* `gllm/model_runner.py` — `_mtp_decode` (draft→verify→accept), `_draft_chain_*`,
  `_capture_{draft,verify}_graphs`, `_gumbel_argmax`, `_mtp_probs_static`,
  `step_once` fused fast-path, `_record_mtp_metrics`.
* `gllm/layers/attention.py` — `_forward_verify_sparse` (fp8 verify kernel).
* `gllm/input_data.py` — `is_mtp_verify` metadata + verify prefill-with-context path.
* `gllm/overlap_worker.py` — `_run_mtp_sync`, all-rank sampling in `run_batch_async`.
* `gllm/worker.py` — hard-exit (`os._exit`) crash/shutdown handlers (avoids NCCL
  destroy deadlock), MTP token-list broadcast skip.

Removed during cleanup: `gllm/spec_decode/mtp_proposer.py` (superseded by the
`_draft_chain_*` methods) and its `spec_decode/` package.

### 10.10 GPU-native input prep (`gllm/mtp_gpu_prep.py`)

An MTP step prepares **two** batches (the draft chain's batch×1 and the verify
batch's uniform `1+k`) on top of whatever the scheduler already prepared. Doing
that with `InputData.cal_input` meant rebuilding every per-token array in Python
per phase per step. Measured on Qwen3.5-0.8B (1×H200, 64 concurrent greedy
decodes, `GLLM_MTP_PROF=2`): **4.3 ms of the 12.4 ms step was host-side prep**.

`MtpGpuPrep` replaces it with the vLLM-model-runner-V2 pattern:

* per-*sequence* facts (context length, mrope delta, `x1`, page table, SSM block
  table) live in persistent **pinned** staging buffers and cross to the device in
  three small H2D copies, staged **once per step** (draft + verify share it,
  memoized on `(epoch, bucket)`);
* every per-*token* array is derived by vectorized CUDA ops written straight into
  the static buffers the captured graphs read -- the verify token ids come from
  the draft chain's GPU tensor and never round-trip through the host;
* CUDA-graph bucket padding writes dummy rows on the GPU instead of building
  throwaway `Sequence` objects.

Supporting changes in `ModelRunner`: one page pre-allocation per step for the
whole `1+k` speculative window (was one per phase); `orig_tokens` kept by
reference so the step no longer costs O(context) per seq; the draft chain's
mid-step `drafts.tolist()` folded into the single end-of-step packed D2H
(`[n_accepted | bonus | drafts]`); and `mtp_fused_prep_eligible` /
`prepare_input_mtp_fused`, which skip the scheduler-side decode prep that a fused
step would only overwrite.

Env vars: `GLLM_MTP_GPUPREP=0` reverts to the CPU builders (both paths are kept);
`GLLM_MTP_GPUPREP_ASSERT=1` rebuilds the verify batch on the host every step and
compares all 10 device buffers (dev/CI only); `GLLM_MTP_PROF=2` reports the
per-sub-step **host** time breakdown (`=1` stays the cuda-synced draft/verify/
accept split).

Result at nd=64 greedy: host prep 4.3 ms → 1.3 ms, step 12.4 → 8.3 ms, decode
throughput **8.8k → 10.9k tok/s** (nd=20: 3.1k → 4.5k). Token streams are
bit-identical to the CPU-prep path.

Also fixed here, because the MTP draft chain is what exposed it:
`create_dummy_seqs` gave its padding rows `page_table = [seq_id]`, i.e. pages
`0..size-1`. Harmless at graph-capture time, but the draft chain pads its bucket
at **runtime**, where those pages belong to live sequences -- the dummy rows
overwrote real KV, so any step with `bucket > nd` diverged. Runtime callers now
pass `runtime=True`, which points every pad row at the reserved `dummy_page`.

### 10.11 Rejection-sampling accept: vectorized

The sampling (rejection) accept was a per-sequence python loop that read
`float(q_dists[i,p,d])` / `float(p_dists[start+p,d])` — each one a device-scalar
sync, up to `2·nd·k` (384 at nd=64) per step — and then ran one
`torch.multinomial` + `.item()` per sequence. It measured **20.5 ms of a 34.6 ms**
sampling step at nd=64.

Now every decision is a batched tensor op:

* `p(d_p)` / `q(d_p)` come from two `gather`s → `[nd, k]`;
* accept mask `u < min(1, p/q)`, then `cumprod(...).sum(1)` gives `n_accepted`
  (the same trick the greedy accept uses);
* the bonus draw needs exactly **one** distribution row per sequence — the
  residual `(p-q)+` at the rejected position, or `p` at the tail when all drafts
  were accepted. Both are row `n_accepted` of `p`, so one gather covers both
  cases and a **single batched `multinomial`** replaces the per-seq loop;
* the host learns the outcome from one packed D2H (`[n_accepted | bonus |
  drafts]`), same shape as the greedy path.

The verify forward now also returns raw **logits** instead of an argmax, so the
greedy argmax and the rejection `p` transform share one lm-head pass (the
rejection path used to re-run `logits_from_hidden` over the same hidden), and the
per-row sampling params are expanded with `repeat_interleave` instead of building
a `nd·(1+k)`-entry python seq list per step.

Result at nd=64, T=0.8/top_k=20: accept **20.5 → 5.3 ms**, step **34.6 → 19.3 ms**,
throughput **3.5k → 5.7k tok/s**. Mean acceptance length is unchanged (2.21).
Distribution check (512 samples, T=1.0, no top-k/p): pooled total-variation
distance to non-MTP sampling is 0.175, *below* the 0.218 split-half noise floor
of the non-MTP sample itself — i.e. no detectable bias.

### 10.12 `max_tokens` overshoot (fixed)

`Scheduler.process_output` stops committing at `seq.is_finish`, but shipped the
**untruncated** speculative burst in `ipc_package.next_tokens`, so
`LLM._apply_ipc_package` appended all `1+n_accepted` tokens to the frontend's copy
of the sequence: a request with `max_tokens=16` returned 16–19 tokens (and text
could continue past the stop token). It now ships `committed[:kept]`.

### 10.13 Sparse (top-k) `p` / `q` for rejection sampling

With the per-seq loop gone, what remained was the *distribution transform*
itself. Measured at this vocab (248k) on H200: `softmax -> top_k_renorm ->
top_p_renorm` costs **1.5 ms for 64 rows** and **3.1 ms for 256 rows**, and the
sampling step pays it `k` times in the draft chain (once per draft step, 64 rows)
plus once in the accept (`nd·(1+k)` = 256 rows) — ~7 ms of the 17 ms step.

But when a request restricts `top_k`, the transformed distribution has **at most
`top_k` nonzero entries**, so carrying it as a dense `[rows, vocab]` tensor is
pure waste. `_mtp_sparse_probs` computes the same distribution as
`(vals, idx)` over its top-k support:

```
vals, idx = topk(logits, k_pad)          # descending
keep      = vals >= vals[:, top_k-1]     # tie-inclusive, like the dense kernel
probs     = softmax(vals.masked_fill(~keep, -inf) / temp)
probs     = zero where exclusive-cumsum >= top_p; renormalize
```

`softmax` restricted to the kept set *is* the dense renormalization of that set,
and `keep` is a prefix of the descending order, so this is mathematically the
dense chain — verified to 1e-7 against the sgl kernels on non-tied logits. Cost:
**0.2 ms / 0.6 ms** for the same 64 / 256 rows.

Two details matter:

* the reference kernel is **tie-inclusive** (`probs=[.4,.2,.2,.2]`, `top_k=2`
  keeps all four), and bf16 logits over a 248k vocab tie often enough to see it
  (support for `top_k=20` measured 20–24), hence `keep = vals >= kth` and the
  `_SPARSE_TIE_MARGIN` (64) of headroom. `topk`'s cost is dominated by the vocab
  scan, so the margin is nearly free. A device-side counter reports (at the 1 Hz
  metrics log, never on the hot path) if ties ever spill past the window.
* `q` and `p` must be built by the *same* code, so sparseness is decided **once
  per step** (`_mtp_sparse_eligible`: every request's `top_k` within the window)
  and threaded into both the draft chain and the accept. An unrestricted request
  (`top_k=-1`, top-p only) keeps the dense path end to end.

`q` now travels as `MtpQDist` — either dense `[nd, k, vocab]` or sparse
`vals`/`idx` `[nd, k, k_pad]` plus `drawn` `[nd, k]` (the probability of the token
each draft step actually sampled, so the accept needs no lookup at all). The
residual `(p-q)+` is supported inside p's kept set (p is 0 outside it), so it is
computed on `[nd, k_pad]` and the bonus is drawn with one `multinomial` over that.
The sparse sampled-draft step is captured as its own graph family
(`_draft_size_to_graph_sampled_sparse`, `k_pad` baked in at capture).

Result at nd=64, T=0.8/top_k=20: draft **7.5 → 3.7 ms**, accept **5.3 → 2.5 ms**,
step **17.4 → 12.7 ms**, throughput **6.0k → 8.1k tok/s**. Cumulative for the
sampling path across §10.11 + §10.13: **3.5k → 8.1k tok/s (2.3×)**, step
34.6 → 12.7 ms. Distribution check (512 samples, T=1.0, top_k=20): pooled TV to
non-MTP sampling 0.138 vs a 0.136 split-half noise floor — no detectable bias.
Validated on greedy / sampling × sparse / dense × graph / eager × TP=1 / TP=2.

### 10.14 What the torch profiler found (online serving, sampling + MTP)

Profiled through the server's own hooks (`POST /start_profile` /
`/stop_profile`, trace + `key_averages` summary under
`GLLM_TORCH_PROFILER_DIR`) over a **steady-state 2 s window** at 64-way
concurrency, T=0.8 / top_k=20, k=3 — warmed up first with *disjoint* prompts so
the window contains neither first-use JIT nor prefix-cache hits on the measured
requests. Two things came out of it and are fixed:

* **`_mtp_sample_params` was 3 device syncs per step, 558 ms of the 2 s window.**
  `torch.tensor(list, device="cuda")` copies from pageable memory, which torch
  serializes with a `cudaStreamSynchronize`; sitting right after the verify graph
  was enqueued, it blocked the host on the entire outstanding GPU queue. Now
  staged through persistent pinned buffers with `non_blocking` copies, and the
  sampled draft chain fills `_d_temp`/`_d_topk`/`_d_topp` from that same staging
  (D2D). `k_pad` likewise moved to a host-side max instead of
  `int(top_ks.max().item())`. Syncs per step **9.0 → 2.1** (562 ms → 0.6 ms),
  eager launches **215 → 178**.
* **The verify conv loop was over half of all GPU ops in the step** — see §10.15.

Still open, in profile order: `fused_recurrent_gdn_spec_fwd_kernel` (the
`1+k` full recurrent-state checkpoints, ~17% of GPU time and the reason MTP
trails plain decode on a 0.8B model — the bytes per checkpoint are now halved by
the engine-wide `mamba_ssm_cache_dtype` knob, which stores the recurrent state in
the activation dtype by default like vLLM does, but the *count* of checkpoints is
inherent to verifying `1+k` tokens); the sparse `topk` (16 radix passes/step —
`_sparse_kpad_capture` is a fixed 128 even when a request asks for `top_k=20`, so
per-bucket `k_pad` capture would cut it); and ~178 eager launches/step of prep +
accept glue that could be fused into a couple of Triton kernels. The step is also
still ~60% GPU-idle in the window because MTP runs synchronously
(`_run_mtp_sync` drains the pipeline), so the scheduler + output processing sit
in the critical path instead of overlapping the MTP GPU work.

### 10.15 Verify conv: one kernel instead of one per token

`Qwen3_5GatedDeltaNet._forward_mtp_verify` used to walk the `1+k` verify tokens
in python, doing `index_select` + `index_copy_` + a single-token
`causal_conv1d_update` per step. At k=3 over 18 GDN layers that is **~380 GPU ops
per MTP step** (72 conv kernels, 72 `index_copy`, 150 index, 86 gather) — more
than half of every op in the step.

`causal_conv1d_update` already supports the whole thing in one launch (vLLM's
spec-decode path): `x` as `[nseq, dim, 1+k]` plus `intermediate_conv_window`,
which records the post-window after **every** token. Two details make it fit
gLLM's block-table layout:

* that path needs a window physically wider than the query (`width-1 + k`), while
  a state block holds exactly `width-1`; so the resume window is staged into a
  small shared scratch buffer of the right width and the kernel's own rolling
  write-back lands there and is discarded;
* `num_accepted_tokens` is passed as all-ones (rewind offset 0) because the
  resume column is chosen while staging, not by the kernel's rewind.

Then one `index_copy_` scatters `intermediate_conv_window` into the seqs'
block-table columns (its `[seq][step]` layout matches the block table's row-major
flattening). Per layer: **4 ops instead of 4·(1+k)**, with the conv running once.

Bit-exactness was checked two ways: a standalone comparison against the old loop
(outputs *and* every touched/untouched state block, `max|diff| = 0`, including
non-trivial resume columns), and end-to-end — greedy MTP token streams are
20/20 identical to the pre-change engine under the same protocol.

Measured at nd=64 (same protocol, back-to-back):

| | conv loop | batched conv |
| --- | --- | --- |
| verify phase | 6.3 ms | **5.4 ms** |
| step (greedy) | 9.1 ms | **8.2 ms** |
| greedy throughput | 11.0k | **12.0k tok/s** |
| sampling throughput | 9.3k | **10.0k tok/s** |
| greedy @ conc=20 | 4.9k | **5.7k tok/s** |

### 10.16 Select the top-k support on the *raw* logits

`_mtp_sparse_probs` (§10.13) opened with `topk(logits.float(), k_pad)`. Measured
on this vocab (248k), `topk` is **flat in `k`** over the whole range that matters
— 0.313 ms at `k` = 64, 84, 96 and 128 (64 rows) — but it scales with the *bytes
it scans*, so selecting on the widened copy costs exactly double:

| 248k vocab | select on fp32 copy | select on bf16 logits |
| --- | --- | --- |
| 64 rows (draft step) | 0.313 ms | **0.163 ms** |
| 256 rows (accept) | 1.014 ms | **0.552 ms** |

So `topk` runs on the logits as they come out of the LM head and only the
selected `[rows, k_pad]` slice is widened to fp32 for the softmax. This is
**bit-identical**, not an approximation: bf16 → fp32 is lossless and order
preserving, so the selected set, its order, and every value the softmax then sees
are unchanged (checked over top_k ∈ {1,20,50} × top_p ∈ {0.9,0.95,1.0} × fp32/bf16
logits: `max|diff| = 0.000e+00` against the previous implementation).

Worth recording as a negative result too: this measurement killed a planned
optimization. `k_pad` was fixed at 128 to cover `top_k ≤ 64`, and bucketing it
(a batch of `top_k=20` needs only 84) looked like an easy win — but since `topk`
is flat in `k`, it would have bought nothing.

Sampling, k=2, conc=64: **11.6k → 12.4k tok/s (+6.5%)**.

### 10.17 MTP is a *scheduling* decision: the batch-size gate

Every measurement above holds the batch fixed and asks "how much cheaper can the
MTP step get?". The other question is *when to take an MTP step at all*.
Speculating multiplies the target work per step by `1+k` and buys back
`mean_accept` tokens, so it wins only while the decode batch leaves the GPU
under-utilized.

A measurement note first, because it changes the numbers: comparing two engine
configurations by total wall time at `out=256` is **not safe at low concurrency**.
There is a fixed ~0.5 s of prefill + first-step + settling in every run, it
differs between modes, and at `conc=1` it is most of the wall. That artifact
alone produced an apparent "+26% at conc=1" which does not survive a longer run
(and an apparent "−11% at conc=4" in the other direction at `out=512`). The
numbers below are `out=1024`, 3 repetitions each, spread ~1%:

| conc | MTP off | k=2 always | |
| --- | --- | --- | --- |
| 1 | 502 | 517 | **+2.9%** |
| 2 | 740 | 864 | **+16.6%** |
| 4 | 1656 | 1796 | **+8.5%** |
| 6 | 2564 | 2266 | −11.6% |
| 8 | 3541 | 2976 | −16.0% |
| 32 | 12422 | 9513 | −23% |
| 64 | 20373 | 15343 | −25% |

Acceptance is healthy throughout (mean 1.98 of 3 at k=2, 2.17 of 4 at k=3), so
this is not a draft-quality problem — it is GPU saturation. A 0.8B model
saturates an H200 at a *very* small batch, so the profitable window here is
`conc <= 4`; it widens with model size (a larger target forward amortizes the
1-layer draft head better), which is exactly why the crossover cannot be a
constant in the code.

Hence `--mtp-max-batch N` (vLLM's speculative `disable_by_batch_size` analogue): speculate only while the decode batch is at
most `N` sequences.

It is a *scheduling* decision, not a second code path: a declined step takes the
plain decode path that already exists as MTP's bootstrap path. Three details:

* the decision is taken **once per iteration** (`mtp_begin_iter`, called by both
  workers) and cached for the four sites that consult it (fused-prep eligibility
  in each worker, `step_once`'s fused and non-fused gates). If they could
  disagree within a step, a step whose prep was skipped as "fused MTP" could then
  take the plain path — running the plain forward on minimal input prep;
* it is a pure function of `num_decodes` and of state each rank derives from its
  own (identical) steps, so TP/PP ranks agree without a collective — MTP's sync
  design requires TP-identical tokens;
* a declined step advances every seq by one token *without* refreshing the fused
  relay, so the stashed `(bonus_tok, bonus_hidden)` no longer describes the seq's
  last position. `_mtp_drop_relay()` invalidates it, costing one bootstrap
  (non-fused) step when the batch falls back below the threshold.

Default is `0` (always speculate): the crossover depends on the model, `k`, the
GPU and the acceptance rate, and a baked-in default would silently disable
speculation on models where it wins far above this batch size.

Measured with a threshold of 4, same `out=1024` protocol as the table above:

| conc | always speculate | **gate** | MTP off |
| --- | --- | --- | --- |
| 4 | 1724 | **1779** | 1648 |
| 64 | 15989 | **20288** | 20284 |

The gate recovers essentially all of the ~22% that always-speculating gives away
at large batch while keeping the small-batch win, which is what makes it safe to
leave MTP enabled in a serving deployment. It matters more under sampling, where
the speculating step is the more expensive of the two: `conc=64`, T=0.8 /
top_k=20, `out=256` — always 12490, **gate 18171**, off 18474.

Accuracy: MMLU-Pro, 1400 questions, k=2, greedy, concurrency 32, threshold 16 —
chosen so the batch crosses the gate continuously and the MTP↔plain transitions
and relay invalidation are exercised throughout: **32.5%**, against 27.6% for
MTP-always at the same settings. A stale-relay bug would have collapsed this to
chance (~10%); the spread is the batch-composition numerics effect these kernels
already show (§10.14).

### 10.18 Finding the crossover automatically: what went wrong (not shipped)

`--mtp-max-batch` asks the operator for a number they do not have. The criterion
itself is simple and, as measured, correct:

    speculating wins  <=>  accepted_tokens_per_step / t_spec > 1 / t_plain

Pinning each mode and reading the steady state off real steps confirms it:

| | t_spec | accept | spec ms/token | t_plain | criterion | truth |
| --- | --- | --- | --- | --- | --- | --- |
| conc=4 | 4.55 ms | 2.22 | 2.05 | 2.07 ms | speculate | +4.5% |
| conc=64 | 7.24 ms | 2.24 | 3.23 | 2.76 ms | plain | −21% |

What does not work is estimating those quantities *online*, from a gate that is
also choosing the mode. Four attempts, each fixing a real and separately
verified bias, and the controller still ended up bistable — the same command run
twice settled all-speculate once and all-plain the next. The failure modes are
worth recording, because they are properties of the measurement and not of the
particular controller:

* **the switch step is not a sample of the mode it switches to.** The first
  speculating step after a plain one is the non-fused bootstrap step (a second
  target forward) and it also drains/refills the overlap pipeline. Timing
  speculation there is self-fulfilling: it reads slow, the gate picks plain, and
  the estimate is never revisited. Skipping the first few steps of each streak
  fixes the bias but makes exploration blocks long and rare.
* **whichever mode explores first eats the cold start.** Exploring
  sequentially (all of A, then all of B) charged post-prefill warm-up entirely to
  A — measured at nd=1, +9% on speculation, enough to invert the decision.
  Alternating blocks fixes this one.
* **acceptance is high-variance.** A step accepts 1, 2 or 3 tokens; a short
  block's EMA read 1.93 against a 2.22 steady state. A cumulative mean fixes it.
* **hysteresis must anchor on the last decision, not the last sample.**
  Anchoring on the last *sampled* mode makes the margin depend on which mode
  exploration happened to end with — always plain, so the margin always pushed
  against speculation.
* **the estimates are only valid for the batch they were taken at.** Bucketing
  by power of two still mixes nd=33 with nd=64, and a batch that is filling or
  draining moves through buckets while the samples are being taken.

A controller that samples the mode it is not currently running, on a workload it
is simultaneously perturbing, needs more care than the 25% it is protecting.
The fixed threshold is predictable, and the table in §10.17 is how to pick it:
sweep concurrency at a fixed `out` of ~1024 with a few repetitions, and take the
batch size where the two curves cross.
