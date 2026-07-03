# DP Attention (Data-Parallel Attention + Expert Parallelism) — Design & Notes

This document describes how gLLM runs **DP attention** for MLA models (Deepseek-V2/V3,
Kimi/Moonlight): the Multi-head Latent Attention (MLA) KV cache is *sharded* across
data-parallel replicas instead of being *replicated* on every tensor-parallel rank,
while the MoE experts are sharded with Expert Parallelism (`EP = DP × TP`).

It also records the bug fixes made while bringing this up and the validation runs.

---

## 1. Motivation

MLA compresses the KV cache into a small per-token *latent* vector. Under plain
Tensor Parallelism (TP) that latent KV cache is **replicated on every TP rank** —
each rank stores the full latent KV for all sequences. For MLA this is wasteful:
the attention compute is cheap relative to the KV, so TP mostly buys you *duplicated
memory* rather than useful parallelism, and it caps how many sequences fit in the
KV cache.

**DP attention** fixes this by running `dp_size` independent full-model **replicas**.
Each replica:

- owns its **own scheduler + KV cache** and serves a *disjoint shard* of requests
  (round-robined by the frontend), so the MLA latent KV cache is **sharded across
  replicas** instead of replicated;
- runs attention entirely locally (TP *within* the replica, if `tp_size > 1`).

The MoE layers, however, must see the *global* token batch to route to experts that
live on other replicas. So the routed experts are sharded across the whole
**Expert-Parallel (EP)** group with the strict relation

```
EP = DP × TP        (per pipeline stage)
```

i.e. every GPU in a pipeline stage owns `1/EP` of the experts. Each MoE layer
gathers the global batch across the DP dimension, runs its local expert shard, and
all-reduces over the EP group (details in §5).

---

## 2. Parallelism layout & process groups

The world is a `pp × dp × tp` grid of `world = pp_size · dp_size · tp_size` ranks:

```
global_rank = pp_rank · S + dp_rank · tp_size + tp_rank,     S = dp_size · tp_size
```

`S` is the **stage size** (ranks per pipeline stage). Group construction lives in
`gllm/dist_utils.py::init_dp_ep`. Four group families are formed (every rank calls
`dist.new_group` the same number of times in the same order, so the NCCL groups line
up across ranks):

| Group | Members (fix / vary) | Used for |
|-------|----------------------|----------|
| **TP** subgroup | fix `(pp, dp)`, vary `tp` | attention heads, embedding/lm-head, o_proj/down_proj all-reduce; MLA latent KV replicated here |
| **DP** subgroup (`_DP_GROUP`) | fix `(pp, tp)`, vary `dp` | MoE token all-gather across the DP dimension (`dp_gather_hidden`) + the per-iter barrier (`dp_all_gather_meta`) |
| **EP** group (`_EP_GROUP`) | fix `pp`, all `(dp, tp)` = the `S` ranks of one stage | routed-expert all-reduce (`ep_all_reduce`) |
| **PP** | adjacent stages via `rank ± S` | pipeline hidden-state exchange |

Example — `pp1 dp2 tp2` (4 GPUs, `S = 4`):

```
             tp0        tp1
   dp0    rank0      rank1        ← DP group 0 (one MLA KV shard, TP-replicated across rank0/1)
   dp1    rank2      rank3        ← DP group 1 (another MLA KV shard)

   TP groups : {0,1} {2,3}          (attention / dense all-reduce within a replica)
   DP groups : {0,2} {1,3}          (MoE gather + barrier, across replicas at same tp)
   EP group  : {0,1,2,3}            (expert all-reduce over the whole stage)
```

Key accessors in `dist_utils.py`: `is_dp_attn()` (`_DP_SIZE > 1`), `get_dp_size()`,
`get_dp_group()`, `get_ep_group()`, `get_ep_rank()`.

> On the `dp_size == 1` (non-DP) path gLLM keeps its usual TP/PP bookkeeping. On the
> DP path each replica is a self-contained **column driver** (own frontend sockets,
> own scheduler); only the `_DP_*` / `_EP_*` collectives cross replicas.

---

## 3. Request lifecycle & data flow

```
                         ┌──────────────────────── frontend (LLM engine) ───────────────────────┐
   HTTP  ─► add_requests ─► wait_lists ─► send_ipc_package                                        │
                                             │  dp_size>1: round-robin one seq per package        │
                                             ▼                                                    │
                    ┌───────── request_socket_dp0 ─────────┐   ┌──── request_socket_dp1 ────┐     │
                    ▼                                       │   ▼                            │     │
          DP0 driver (tp0)                                  │  DP1 driver (tp0)              │     │
             recv_ipc_package                               │    recv_ipc_package            │     │
             broadcast_input_to_tp ─► DP0 tp1 (follower)    │    broadcast_input_to_tp ─► ...│     │
             schedule_forward (barrier + forward)           │    schedule_forward            │     │
             process_output ─► send_output ─┐               │    ... ─► send_output ─┐       │     │
                                            ▼               │                        ▼       │     │
                                    frontend output PULL  ◄─┴────────────────────────┘  (fan-in)   │
                                            │  recv_ipc_package → running_maps[seq_id] → AsyncStream│
                                            └───────────────────────────────────────────────────► HTTP
```

- **Frontend → replica (round-robin):** `LLM._send_ipc_package_dp` sends each new
  sequence to exactly one DP replica (`request_socket_dp{d}`), advancing `_dp_rr`.
  That replica owns the seq's KV for its whole lifetime. Aborts are *broadcast* to
  all replicas (a replica that doesn't own the id ignores it).
- **Driver → TP peers:** only each replica's `tp_rank == 0` polls the frontend
  (`Worker._polls_frontend`). It fans the `IPCPackage` out to its TP followers over
  a CPU-side zmq PUSH/PULL (`broadcast_input_to_tp`) so every column driver in the
  replica adds the same requests/aborts in lockstep — no NVLink contention with the
  model's per-layer all-reduce.
- **Replica → frontend (fan-in):** each driver PUSHes its output package into the
  shared frontend PULL. Packages are self-describing (`act_schedule_ids` aligned with
  `next_tokens`/`free_ids`) and keyed by `seq_id`, so ordering across replicas is
  irrelevant; the engine drains *all* queued packages each tick.

Server flag: `--dp N` (see `gllm/entrypoints/api_server.py`). World size is
`pp·dp·tp`; with `--enable-ep` the experts shard across `EP = dp·tp` per stage.

---

## 4. The per-iteration lockstep barrier

Because each replica schedules *independently*, at any step some replicas may have
work and others may be idle. But the MoE runs collectives over the DP/EP group, so
**all replicas must enter — and skip — the forward together**. This is enforced by a
single unconditional all-gather every iteration (`Worker._schedule_forward_dp`):

```python
# gllm/worker.py
real_counts, decode_flags = dp_all_gather_meta(real_ntok, is_decode)  # over _DP_GROUP
if sum(real_counts) == 0:
    return                      # every replica idle → all skip in unison
fwd_counts = [c if c > 0 else 1 for c in real_counts]
if real_ntok == 0:              # idle replica rides along with a 1-token dummy
    dummy_seqs = self.model_runner.create_dummy_seqs(1)
    self.model_runner.prepare_input(dummy_seqs)
```

`dp_all_gather_meta` (in `dist_utils.py`) all-gathers each replica's
`(token_count, is_decode)` — a fixed-size collective that doubles as the barrier and
tells everyone (a) whether anyone has work and (b) whether the *whole world* is a
pure-decode step (needed for the CUDA-graph decision, §6).

**Correctness invariant.** The barrier is per-DP-group (`{0,2}` and `{1,3}` in the
example). Because the two TP peers of a replica (`rank0`/`rank1`) run an *identical*
column-driver schedule (same requests via `broadcast_input_to_tp`, same tokens via
greedy sampling / `broadcast_tokens_to_tp`), both DP groups compute the same
`sum(real_counts)` and therefore make the same run/skip decision. If that invariant
were ever violated (TP peers disagreeing on `real_ntok`), one DP group would run the
`ep_all_reduce` while the other skipped it and the stage-wide collective would hang —
see the determinism fixes in §7.

---

## 5. MoE forward under DP + EP

`DeepseekV2MOE._forward_dp_ep` (in `gllm/models/deepseek_v2.py`) implements the
gather → shard → reduce → slice dance. Per MoE layer:

```python
# 1. shared (dense) experts on local tokens; TP-reduce their partials
shared_output = tensor_model_parallel_all_reduce(self.shared_experts(hidden_states))

# 2. gather every replica's tokens into the global batch (over _DP_GROUP)
counts = get_dp_forward_counts()          # published by the worker this step
global_hidden = dp_gather_hidden(hidden_states, counts)

# 3. run this rank's local expert shard over the *global* batch
routed = self.experts(global_hidden, self.gate(global_hidden))  # reduce_results=False

# 4. all-reduce over the EP group so each token sums all its top-k experts
routed = ep_all_reduce(routed)            # over _EP_GROUP (the whole stage)

# 5. slice this replica's own rows back out, fold in shared output
routed = dp_local_slice(routed, counts)
```

`set_dp_forward_counts(...)` publishes the agreed per-replica row counts each step so
every MoE layer sizes/slices its gather identically. Two layouts:

- **Uniform** (all counts equal — CUDA-graph decode path): the fixed-size all-gather
  output *is* the contiguous global batch (`dp_size·B` rows), no data-dependent
  `cat`/alloc → safe to capture in a graph.
- **Ragged** (variable counts — eager prefill/mixed): pad to `max`, all-gather, then
  concatenate the real slices.

---

## 6. CUDA graph & overlap scheduling compatibility

- **CUDA graph.** Only taken when *every* replica is a pure-decode (or idle-dummy)
  step (`all(decode_flags)`) and a captured bucket covers the largest replica. Then
  every replica pads to one common bucket (`dp_select_bucket(max(fwd_counts))`), so the
  global MoE batch is a static `dp_size · bucket` (uniform fixed-length layout) that
  the captured gather/all-reduce can replay. Any prefill/mixed/bucket-miss → eager,
  variable-length gather. The graph/eager decision is derived from `decode_flags`
  gathered in the barrier, so it is identical on all ranks.
- **Overlap (async) scheduling.** The DP path is compatible; the per-iter barrier and
  the published counts flow through the same `SchedulePayload` mechanism. For `pp > 1`
  the barrier result (counts + graph bucket) rides the payload to PP-other stages so
  they replay without re-running the barrier (`_schedule_forward_dp_pp`).

---

## 7. Bug fixes made during bring-up

### 7.1 Lost-request race on `wait_lists` (tail-of-run hang) — **primary fix**

**Symptom.** Under high concurrency, MMLU-Pro runs reliably stalled near the tail
(e.g. 294–323/400). All workers were **idle-spinning** the barrier (not deadlocked),
the engine event loop was idle, yet the client waited forever on ~5–20% of requests.
A freshly `curl`ed request was still served correctly. Reproduced under *every* worker
config (overlap+graph, eager+graph, eager+no-graph) → the bug was **engine-side**, not
in the worker collectives.

**Root cause.** The async server runs both request intake (`add_requests`) and the
engine step (`send_ipc_package`) on the event loop's **default `ThreadPoolExecutor`**
(many threads, via `make_async`). They touched `self.wait_lists` with no lock:

```
Thread A (send_ipc_package): for seq in wait_lists: dispatch(seq)
Thread B (add_requests):      wait_lists.extend([new_seq])   # lands in the gap
Thread A:                     self.wait_lists = []           # new_seq silently dropped
```

A request appended in the window between the dispatch loop and the `wait_lists = []`
reset was **discarded** — never sent to any worker, so its `AsyncStream` never
finished and the HTTP request hung. The DP path (`_send_ipc_package_dp`) has a wider
window (per-seq zmq sends before the reset), which is why DP+EP tripped it most often.

**Fix** (`gllm/llm_engine.py`, `gllm/async_llm_engine.py`): guard the intake queues
with `self._pending_lock` and **atomically snapshot-and-clear** them:

```python
def send_ipc_package(self, log=True):
    with self._pending_lock:
        if not self.wait_lists and not self.abort_ids:
            return
        wait_lists, abort_ids = self.wait_lists, self.abort_ids
        self.wait_lists, self.abort_ids = [], []      # anything new goes to the next tick
    for seq in wait_lists:
        self.running_maps[seq.seq_id] = seq
    ...
```

`add_requests` appends under the same lock; `check_abort_seqs` iterates a snapshot of
`running_maps` (`list(...)`) and appends aborts under the lock.

### 7.2 Non-deterministic decode-token-budget jitter (TP/PP desync)

`scheduler.get_balanced_decode_token_budget` used `random.randint`, whose RNG diverged
across TP peers, so peers picked different batch sizes and deadlocked the stage-wide
MoE/EP all-reduce (and PP hidden exchange). Replaced with a **deterministic rotating
counter** (`_decode_budget_jitter`) advanced in lockstep by every rank.

### 7.3 Broadcast source rank for a TP subgroup not containing global rank 0

`dist.broadcast(..., src=get_output_rank())` failed with *"Global rank 0 is not part of
group"* when a DP replica's TP subgroup didn't include rank 0. Fixed to use the group's
own tp0 (`get_rank() - get_tp_rank()`) as the source.

### 7.4 `KeyError` on stale post-free token

Under overlap scheduling a worker can emit a trailing token for a sequence it freed one
step earlier (EOS detected after the next step launched). The engine's
`_apply_ipc_package` now uses `running_maps.get(id)` and drops such stale tokens instead
of crashing.

### 7.5 Per-replica scheduler logging

Scheduler stats were gated on `get_rank() == 0`, so only DP0 logged. Changed to
`get_pp_rank() == 0 and get_tp_rank() == 0` so **each** replica's driver logs its own
`#wait/#run/#prefill/#decode/…` line.

---

## 8. Known behavior: idle busy-spin

When there are no requests the workers still show ~15–30% GPU utilization. This is
expected, not a leak: the DP path runs the unconditional `dp_all_gather_meta` barrier
every loop iteration (a small NCCL all-gather that also spin-waits on the SM), and the
main loop has no idle sleep (frontend poll is non-blocking). It is a coordination cost,
not real model compute, and does not affect correctness. A future *idle-backoff*
(sleep after N consecutive idle steps, or a lighter CPU-side idle signal) could drive
this toward 0 at the cost of a small first-request latency bump — must preserve the DP
lockstep invariant of §4.

---

## 9. Validation

Model: `Moonlight-16B-A3B-Instruct` (FP8), 4× H20, MMLU-Pro, greedy
(`temperature=0`), `--no-overlap-scheduling --disable-cuda-graph --model-max-length 8192`.

**DP correctness vs non-DP baseline** — same 4 GPUs, same `EP=4`, full 1400-question
MMLU-Pro (14 subjects), concurrency 256:

| Config | TOTAL accuracy | empty/lost |
|--------|----------------|------------|
| **DP**  `--tp 2 --dp 2 --enable-ep` (TP2×DP2×EP4) | **48.79 %** (683/1400) | 0/1400 |
| **non-DP** `--tp 4 --dp 1 --enable-ep` (TP4×EP4)  | **48.43 %** (678/1400) | 0/1400 |

Δ = 0.36 % (5 questions), within floating-point noise (TP reduction-order differences
occasionally flip a greedy argmax). Per-subject deltas are single-digit and mixed-sign
— no systematic bias → **the DP-attention implementation is numerically aligned with
the non-DP path.**

**Stress / stability.** 1400 requests @ concurrency 256 on the DP instance completed
with **0 empty responses and no tail stall**; GPUs stayed 47–80 % utilized. Before the
§7.1 fix the same load reliably hung near the tail.

---

## 10. Usage

```bash
# DP attention: TP2 × DP2, experts sharded EP = DP·TP = 4 (4 GPUs)
python -m gllm.entrypoints.api_server \
    --model-path <MLA-model> \
    --tp 2 --dp 2 --enable-ep \
    --model-max-length 8192
```

- `--dp N` — number of DP-attention replicas (MLA KV cache is sharded across them).
- `--enable-ep` — shard MoE experts across `EP = dp·tp` ranks per pipeline stage.
- Works together with `--pp`, CUDA graph, and overlap scheduling (§6).
- Constraint: `EP = DP × TP` (per stage) — enforced by the group layout in `init_dp_ep`.
