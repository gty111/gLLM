# Encoder Disaggregation — Usage Guide

This guide explains how to run **encoder disaggregation** in gLLM: the vision
encoder (ViT) of a multimodal model is split out of the language model (LM) and
runs in its own process/GPU. Visual embeddings are sent back to the LM over NIXL
(GPU→GPU).

> For internals and rationale, see
> [`encoder_disaggregation_design.md`](./encoder_disaggregation_design.md).
> This document only covers how to use it.

---

## 1. What it is / when to use it

In a normal multimodal request, vision encoding (ViT) and the language model
(prefill + decode) run on the same GPUs. When there are many / high-resolution
images, the ViT consumes a lot of compute and hurts the LM's TTFT and throughput.

Disaggregation splits the two apart:

```
                 ┌─────────────┐   NIXL (GPU→GPU, visual embeddings)
   images/video ►│ Encoder ×N  │ ───────────────────────────────┐
                 │ (ViT only)  │                                  ▼
                 └─────────────┘                          ┌──────────────┐
                       ▲  ZMQ control plane (job / meta)   │   LM node    │
                       └──────────────────────────────────│ (LM only)    │
                                                           │ prefill+decode│
   text ──────────────────────────────────────────────────►└──────────────┘
```

Use it when the workload is **vision-heavy** (visual tokens ≫ text tokens:
multi-image, high resolution) or when you want to scale the ViT across extra
GPUs independently of the LM. It is *not* worth it for text-only or
vision-light workloads.

> Reference (Qwen3.5-35B-A3B-FP8, random 4–8 images @1080p): at an **equal GPU
> budget**, disaggregation cut median TTFT by −28% (E1LM1, 2 GPUs) up to −50%
> (E3LM1, 4 GPUs) and raised total throughput +32% to +49%. See section 10.

---

## 2. Components & communication

| Component | Entry point | Role |
|-----------|-------------|------|
| **Discovery registry** | `gllm.entrypoints.discovery_server` | A tiny ZMQ rendezvous service. The LM and encoders register with it and discover each other (TTL leases, heartbeats, ADD/REMOVE events). |
| **LM node** | `gllm.entrypoints.lm_server` | Frontend + scheduler + PP0 worker + KV/SSM cache, **without the vision tower**. Serves the OpenAI-compatible API. |
| **Encoder node** | `gllm.entrypoints.encoder_server` | **Vision tower only**. Runs the ViT per item and writes embeddings back to the LM over NIXL. One process per GPU; run several replicas. |

Two planes:

- **Control plane (ZMQ):** the LM pushes `EncoderJob`s to encoders; encoders
  push `MmItemMeta` (per-item metadata) back to the LM.
- **Data plane (NIXL/UCX):** visual embedding tensors go GPU→GPU directly.

**Discovery does the wiring for you.** Once every process points at the same
discovery endpoint, hosts/ports are advertised and resolved automatically — you
do *not* hand-configure peer addresses. The launch commands below are therefore
minimal; the networking flags only matter for cross-machine / multi-replica
cases (sections 7 and 9).

---

## 3. Supported models

Encoder disaggregation is wired for the Qwen-VL families (LM side skips the
vision tower, encoder side skips the language model, and output stays
byte-identical to the monolith). Verified:

| Model | Architecture | Notes |
|-------|--------------|-------|
| Qwen2.5-VL (e.g. `Qwen2.5-VL-3B-Instruct`) | `Qwen2_5_VLForConditionalGeneration` | dense, no deepstack |
| Qwen3-VL (e.g. `Qwen3-VL-30B-A3B-Instruct-FP8`) | `Qwen3VLMoeForConditionalGeneration` | MoE, deepstack `[8,16,24]` |
| Qwen3.5 / Qwen3.5-MoE (e.g. `Qwen3.5-0.8B`, `Qwen3.5-35B-A3B-FP8`) | `Qwen3_5(Moe)ForConditionalGeneration` | hybrid linear-attn (SSM cache — see section 8) |

> To add a new model, its `*ForConditionalGeneration` class must support
> `skip_visual` / `skip_language` in `__init__`, expose
> `embed_multimodal_single(**mm_input)`, and guard both skip branches in
> `load_weights`. The Qwen3-VL base class already does this; subclasses inherit
> it for free.

---

## 4. Prerequisites

- NIXL (UCX backend) available for GPU→GPU transfer.
- **One GPU per process:** each encoder/LM binds a single card via
  `--encoder-gpu` / `--lm-gpu` (it sets `CUDA_VISIBLE_DEVICES` internally).
  Make sure the target card is free.
- Keep the LM↔encoder NIXL traffic within the same NUMA domain (cross-NUMA UCX
  wireup can fail — see section 9).
- `python` below refers to the gLLM environment's interpreter; `$MODEL` is the
  path to a supported model directory.

---

## 5. Quick start (single node: 1 LM + 1 encoder)

Minimal flags only — discovery handles the rest. Pick free GPUs (here LM on
GPU 2, encoder on GPU 3) and a registry port (here 9500).

```bash
MODEL=/path/to/Qwen2.5-VL-3B-Instruct
DISC=127.0.0.1:9500   # registry address; reused for bind and connect

# 1) Discovery registry
python -m gllm.entrypoints.discovery_server --listen $DISC &

# 2) LM node (port 8100). Vision tower is skipped by default on this entry point.
python -m gllm.entrypoints.lm_server \
    --model-path $MODEL --lm-gpu 2 --port 8100 \
    --discovery-endpoint $DISC &

# 3) Encoder node (vision only)
python -m gllm.entrypoints.encoder_server \
    --model-path $MODEL --encoder-gpu 3 \
    --discovery-endpoint $DISC &
```

That's it. Everything else uses defaults: advertise host = `auto`,
control-plane ports = ephemeral, prefix caching on, `--gpu-memory-util 0.9`,
`--schedule-method chunked_prefill`, `--encoder-dp 1`.

Readiness:

- LM log shows `Uvicorn running on http://0.0.0.0:8100` and
  `encoder ... connected; live encoders=N`.
- Encoder log shows `READY; entering job loop` / `serving jobs`.

Send a request (OpenAI-compatible, against the LM port):

```bash
python - <<'EOF'
import base64
from openai import OpenAI
c = OpenAI(api_key="EMPTY", base_url="http://127.0.0.1:8100/v1")
uri = "data:image/png;base64," + base64.b64encode(open("/tmp/test_img.png","rb").read()).decode()
r = c.chat.completions.create(
    model=c.models.list().data[0].id,
    messages=[{"role":"user","content":[
        {"type":"text","text":"Describe this image."},
        {"type":"image_url","image_url":{"url":uri}}]}],
    max_tokens=128, temperature=0.0, extra_body={"top_k":1})
print(r.choices[0].message.content)
EOF
```

---

## 6. Configuration reference

You normally only need `--model-path`, the GPU index, the LM `--port`, and
`--discovery-endpoint`. The flags below are for tuning / non-default setups.

### Essential

| Flag | Applies to | Purpose |
|------|-----------|---------|
| `--model-path` | LM, encoder | Same model directory on both sides. |
| `--lm-gpu` / `--encoder-gpu` | LM / encoder | Physical GPU to bind. |
| `--port` | LM | OpenAI API port (default 8000). |
| `--discovery-endpoint` | LM, encoder | Registry `HOST:PORT`. **If omitted on the LM, it falls back to a plain text-only server.** |
| `--listen` | discovery | Registry bind address (default `0.0.0.0:9500`). |

### Optional (defaults shown)

| Flag | Default | Purpose |
|------|---------|---------|
| `--encoder-dp` (LM) | `1` | Number of encoder replicas to wait for before dispatching. |
| `--gpu-memory-util` | `0.9` | Memory fraction. |
| `--maxd` (LM) | `512` | Max concurrent decode slots. **Drives SSM cache size for hybrid models — see section 8.** |
| `--enable-prefix-caching` / `--no-...` (LM) | on | Prefix caching. |
| `--schedule-method` (LM) | `chunked_prefill` | Scheduling mode. |
| `--disable-cuda-graph` (LM) | off | Turn off CUDA graphs if a model/config hits graph issues. |
| `--model-max-length` (LM) | model config | Cap context length (bounds KV/SSM buffers). |

### Networking (only for cross-machine / multi-replica — see 7 & 9)

| Flag | Default | Purpose |
|------|---------|---------|
| `--nixl-advertise-host` (LM) / `--advertise-host` (encoder) | `auto` | Address peers use to connect back. `auto` detects the egress IP; use an explicit IP for cross-machine, or `127.0.0.1` to force loopback. |
| `--meta-port` (LM) | `0` (ephemeral) | Fixed port for the per-item meta intake (pin it behind a firewall). |
| `--nixl-backend` (LM / encoder) | `UCX` | NIXL transport backend. The data-plane endpoint is auto-negotiated via the exchanged metadata, so there is no port to configure (UCX picks ephemeral ports; same-host encoders need no distinct NIXL port). |
| `--zmq-listen` (encoder) | `0.0.0.0:0` | Job control-plane port (ephemeral by default). |
| `--max-vis-tokens` (encoder) | `16384` | Upper bound on N_vis per item; sizes the send buffer. |
| `--mm-embed-cache-size` (encoder) | `256` (MB) | Per-replica content_hash→embedding dedup cache. |

### Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `GLLM_DISAGG_OVERLAP` | `1` | **Intra-request** encode/prefill overlap (see below). |
| `GLLM_DISAGG_REDISPATCH_TIMEOUT_S` | `20.0` | Re-dispatch in-flight jobs after this timeout (watchdog). |
| `GLLM_DISAGG_MAX_REDISPATCH` | `5` | Max re-dispatch attempts per item. |
| `GLLM_ENC_FAIL_FIRST_N` | `0` | **Test only:** make an encoder drop its first N jobs to exercise re-dispatch. |

#### `GLLM_DISAGG_OVERLAP` in detail

This controls how the LM admits a **single** request whose images are still
being encoded — it is *intra-request* overlap, not the (always-on) pipelining
between requests. The LM tracks two gates per request:

- **Gate A (metadata):** every visual item's `MmItemMeta` has arrived, so token
  positions, prompt length, and prefix-cache hashes are fixed.
- **Gate B (data):** the actual embedding for each image span has landed over
  NIXL. Prefill is capped at the start of the first span whose embedding has not
  yet arrived; as more embeddings land, prefill advances.

| Value | Behavior | Trade-off |
|-------|----------|-----------|
| `1` (default) | Admit as soon as **Gate A** is met. The request starts prefilling its text and already-arrived images immediately, and progressively fills in the remaining image spans (Gate B) as their embeddings arrive — overlapping LM prefill with ongoing ViT encode/transfer for the *same* request. | Best when encoding is on the critical path. Prefill runs in multiple chunks, so output is subject to the engine's normal chunked-prefill bf16 rounding (not bit-exact vs an unchunked run). |
| `0` | Wait until **Gate A and Gate B** are both met (all embeddings landed) before admitting, then prefill the whole prompt in one chunk. | Determinism baseline: byte-identical to the unchunked monolith. TTFT no worse than `1` when encoders keep up; higher when the LM would otherwise sit idle waiting on a request's last image. |

Note: regardless of this flag, the LM can still prefill/decode request A while
the encoders work on request B — that cross-request pipelining is inherent to
the two-plane design and is not gated by `GLLM_DISAGG_OVERLAP`.

---

## 7. Multiple encoders (DP, e.g. E3LM1)

3 encoders + 1 LM. The only extra vs the quick start is `--encoder-dp 3` on the
LM. Same-host encoders need no distinct NIXL port (UCX auto-negotiates the
data-plane endpoint); the ZMQ job intake defaults to an ephemeral port too.

```bash
MODEL=/path/to/Qwen3.5-35B-A3B-FP8
DISC=127.0.0.1:9500   # registry address; reused for bind and connect

python -m gllm.entrypoints.discovery_server --listen $DISC &

python -m gllm.entrypoints.lm_server \
    --model-path $MODEL --lm-gpu 4 --port 8100 \
    --discovery-endpoint $DISC --encoder-dp 3 --maxd 64 &

for g in 5 6 7; do
  python -m gllm.entrypoints.encoder_server \
      --model-path $MODEL --encoder-gpu $g \
      --discovery-endpoint $DISC &
done
```

The LM dispatches per item across the live encoders. Encoders can be added or
removed **dynamically**: a new replica joins automatically once registered; if
one goes away, the LM watchdog re-dispatches its in-flight items to the others.

---

## 8. Memory tuning (hybrid models with SSM cache)

Qwen3.5-family models have hybrid linear-attention. The LM's SSM snapshot pool
scales as `~4 × maxd + 1` slots, so with the default `--maxd 512` the SSM cache
can reach ~39 GB on a single TP1 GPU and OOM on top of the weights.

- If concurrency is moderate, lower `--maxd` (e.g. 64): SSM cache drops from
  ~39 GB to ~5 GB.
- Still OOM? Lower `--gpu-memory-util` or `--model-max-length`.
- Dense models (Qwen2.5-VL) have no SSM cache and need no special handling.

---

## 9. Cross-machine deployment

- **Discovery:** bind `--listen` to a routable address; every node points at the
  same `HOST:PORT`.
- **Advertise host:** keep `auto` (detects the egress IP) or pass an explicit
  routable IP on the LM (`--nixl-advertise-host`) and encoders
  (`--advertise-host`). Do **not** use `127.0.0.1` across machines.
- **Fixed ports:** behind a firewall, pin the LM `--meta-port` and encoder
  `--zmq-listen` (e.g. `0.0.0.0:9300`) so static rules can target them.
- **Data plane (NIXL/UCX, RDMA):** set UCX env vars when needed, e.g.
  ```bash
  export UCX_TLS=rc,cuda_copy,cuda_ipc   # RDMA (rc) across hosts; cuda_ipc same host
  export UCX_NET_DEVICES=mlx5_0:1        # pick the RDMA NIC
  ```
- On a single host, keep the LM and encoders on GPUs in the **same NUMA domain**,
  otherwise UCX may report `NIXL_ERR_REMOTE_DISCONNECT`.

---

## 10. Performance reference

Model `Qwen3.5-35B-A3B-FP8`, workload = random 4–8 images @1080p
(`sglang.bench_serving` image dataset, 64 prompts, concurrency 16, out-len 128),
**equal 2-GPU budget per side**, `--disable-cuda-graph`, prefix caching on:

| Metric | Monolith TP2 | E1LM1 (disagg) |
|--------|--------------|----------------|
| TTFT median (ms) | 5445 | **3938** |
| TTFT mean (ms) | 6235 | **4687** |
| TPOT median (ms) | 169.0 | **113.5** |
| Total throughput (tok/s) | 5457 | **7180** |
| E2E mean (ms) | 27296 | **19136** |
| Bench duration (s) | 110.2 | **83.8** |

Layout: monolith = TP2 (ViT + LM share 2 GPUs); disagg = LM TP1 on 1 GPU + 1
vision-only encoder on a 2nd GPU. Even at this minimal 1:1 split, moving vision
off the LM's critical path gives **−28% median TTFT** and **+32% throughput**.
Adding encoders scales the win: with 3 encoders + 1 LM vs an equal-budget TP4
monolith, the same 1080p workload reaches −50% TTFT / +49% throughput (the more
GPUs spent on parallel ViT, the shorter the encode wall).

### Intra-request overlap (`GLLM_DISAGG_OVERLAP`)

The E1LM1 config above is **encoder-bound** (one encoder serializes every ViT),
which is exactly where intra-request overlap (section 6) helps. Same workload:

| Metric | `OVERLAP=0` | `OVERLAP=1` (default) |
|--------|-------------|------------------------|
| TTFT median (ms) | 5015 | **3938** |
| TTFT mean (ms) | 5435 | **4687** |
| Total throughput (tok/s) | 6808 | **7180** |

Admitting on Gate A and overlapping prefill with the encoder's remaining ViTs
cuts median TTFT **−21%**. With several encoders the encode wall is short and
parallel, so embeddings are essentially all present by admission and overlap has
little left to hide (within noise) — it matters most when encode is the
bottleneck.

---

## 11. Validating correctness (byte-identical to monolith)

Run a **monolith** (`api_server`) and a **disaggregated stack** side by side,
ask the same question about the same image, and the outputs should be identical
byte-for-byte (greedy decoding + prefix cache, cold == warm).

```bash
# Monolith (api_server) on port 8200
python -m gllm.entrypoints.api_server --model-path $MODEL --port 8200 \
    --pp 1 --tp 1 --disable-cuda-graph &

# Compare disagg(8100) vs mono(8200)
python tests/disagg_vlbug_check.py \
    --disagg-port 8100 --mono-port 8200 \
    --image /tmp/test_img.png --repeats 3 --max-tokens 128
```

Expect `ALL GOOD: True` (`disagg == monolith: True`, no SVG).

> Note: with chunked prefill, bf16 rounding at chunk boundaries causes *expected*
> tiny numerical differences (independent of disaggregation); under greedy
> decoding a long sequence may diverge after some token. That is not a
> disaggregation bug.

---

## 12. Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| LM OOM at startup, a GPU already full | A worker from a previous run didn't exit (`multiprocessing-fork`; after its parent is killed it reparents to `init`, PPID=1, and keeps holding GPU memory). Find PPID=1 workers (`ps -eo pid,ppid,cmd | grep multiprocessing`) and `kill -9` them; wait ~8s before relaunching. |
| LM OOM (hybrid model) | `--maxd` too large → oversized SSM cache. Lower `--maxd` (section 8). |
| `Address already in use` (fixed port) | A leftover worker still holds the port. Same fix as above. |
| Encoder is up but `live encoders` stays 0 | Processor/config hash mismatch (LM and encoder must use the same model dir / processor settings), or wrong discovery endpoint. |
| `NIXL_ERR_REMOTE_DISCONNECT` | Cross-NUMA UCX wireup. Put LM and encoder on GPUs in the same NUMA domain, or set UCX env vars. |
| Requests hang in the queue | No live encoder at that moment. Once an encoder connects, the watchdog re-dispatches in-flight items. |

---

## 13. Graceful shutdown

```bash
# Kill the stack (LM / encoder / discovery / monolith) and clean up stray workers
ps -eo pid,cmd | grep -E "lm_server|encoder_server|api_server|discovery_server" \
  | grep -v grep | awk '{print $1}' | xargs -r kill -9
for w in $(ps -eo pid,ppid,cmd | grep multiprocessing | grep -v grep | awk '$2==1{print $1}'); do
  kill -9 $w
done
```
