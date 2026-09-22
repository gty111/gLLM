# Frontend/Worker Decoupling (Standalone Deployment)

gLLM traditionally runs the OpenAI **frontend** (HTTP API) and the GPU
**worker** fleet as tightly coupled processes: the frontend *spawns* the
workers, and a crash of either half tears the whole system down
(`check_worker_alive` does `sys.exit()` when a worker dies; killing the
frontend orphan-then-loses the GPU fleet). This document describes the
**decoupled deployment** where the two halves are independent processes that
can be restarted (or crash) without dragging the other down.

## Why

* A frontend crash (bad request handler bug, OOM in the Python HTTP layer, a
  restart for a config change) should **not** reload the model weights /
  re-capture CUDA graphs on the GPU.
* A worker crash (CUDA error, OOM, a kernel panic) should **not** kill the
  HTTP listeners that clients are attached to. In-flight requests fail fast
  and new ones resume as soon as a worker is back.
* Independent scaling / patching of the API layer and the inference layer.

## Architecture

```
   ┌────────────────────────┐         ┌──────────────────────────────┐
   │  frontend (api_server) │  zmq    │  worker fleet (worker_server)│
   │  --standalone-frontend │◄───────►│  (GPU child process)         │
   │  no GPU, no spawn      │  ipc/tcp│  binds schedule PULL,        │
   │  loads tokenizer+cfg   │         │  output PUSH, runs the model │
   └────────────────────────┘         └──────────────────────────────┘
                │                                   │
                └────── rendezvous: worker endpoint file ──────┘
                     (<endpoint-file>.json)
```

The two sides rendezvous through a small JSON **worker endpoint file**
(`gllm.entrypoints.worker_endpoint.py`):

```json
{
  "version": 1,
  "uuid": "<transport id, random per worker launch>",
  "updated_at": 1789974106.123,
  "endpoints": {
    "0": {
      "schedule": "ipc:///tmp/<uuid>_gllm_schedule",
      "output":   "ipc:///tmp/<uuid>_gllm_output",
      "token":    "ipc:///tmp/<uuid>_gllm_token"
    }
  }
}
```

* The **worker parent** writes it (only *after* the GPU child has bound its
  sockets and finished init) and removes it via `atexit` + a child-supervision
  loop when the fleet goes away.
* The **frontend** polls it. A change in `uuid` means the worker fleet
  restarted; the frontend tears down and re-connects its ZMQ sockets
  **in-process** (`FleetSupervisor.reconnect`) — no frontend process restart.

Transport: `ipc://` on a single machine, or fixed `tcp://` ports
(`--worker-transport-base-port`) so a frontend on another host can connect.

### Frontend session epoch

Every frontend process mints a random **session epoch**, bound to the
*request*: the frontend stamps `seq.frontend_session` on every sequence
before dispatching it. The worker then makes the identity UNAMBIGUOUS
internally: at admission each seq's client id is remapped to a
fleet-unique monotonic **internal id** (client id + stamp preserved as
`seq.client_seq_id` / `seq.frontend_session`), so two surviving sessions'
request 0s can coexist in the same scheduler, abort set, KV tables and
output pipelines without cross-talk. OUTPUT packages are translated back
in flight (`Worker.translate_output_for_frontend`, hooked into
`comm.send_output`): internal ids -> client ids, with per-row session
stamps (`IPCPackage.sessions` / `free_sessions`), so the standalone
frontend applies only rows stamped with its own epoch -- a late completion
of the dead session's request 0 can never terminate the new session's
request 0, and `abort(0)` from the new session frees only its own
request. Dispatch is gated by a cheap endpoint-file liveness probe so a
dead fleet cannot silently absorb requests into its 512MB send buffer.

### Wire protocol (single definition)

The four session fields, their directions, and the **positional
alignment** contract are defined once on `IPCPackage` (the class
docstring in `gllm/distributed/comm.py` is the authoritative table).
All producers/consumers route through the package helpers instead of
touching the stamp lists directly:

| Helper                        | Side     | Purpose                                    |
|-------------------------------|----------|--------------------------------------------|
| `merge_aligned(other)`        | worker   | the ONLY sanctioned drain merge (keeps `abort_sessions` aligned with `abort_ids`) |
| `abort_stamps_valid()`        | worker   | request-dir alignment predicate            |
| `output_stamps_valid()`       | frontend | one O(1) per-package alignment check       |
| `act_session_at(i, epoch)`    | frontend | positional row gate for acted tokens       |
| `free_session_at(i, epoch)`   | frontend | positional row gate for free rows          |

Malformed stamp lists fail CLOSED: a misaligned packet is dropped
wholesale rather than guessed, and a missing stamp list is legacy
(monolith) semantics, never "foreign". Covered by
`tests/test_ipc_protocol.py`.

## Crash semantics (verified end-to-end)

| Event                              | Frontend                              | Worker fleet                     |
|------------------------------------|---------------------------------------|----------------------------------|
| **Worker GPU child dies**          | stays up; in-flight requests fail fast; auto-reconnects when a new worker publishes | parent watchdog removes the endpoint file and exits |
| **New worker launched**            | detects the new `uuid`, re-connects in-process, serves immediately | fresh fleet, weights re-loaded |
| **Frontend dies**                  | (gone) — new frontend re-reads the endpoint file and reconnects | **survives**; weights stay loaded, keeps serving |
| **New frontend launched**          | connects to the still-running worker (same `uuid`); mints a fresh session epoch, so late outputs for the dead frontend's ids are dropped by the worker's stamp | unchanged |

## Usage

### 1) Launch the GPU worker fleet (no HTTP)

```bash
python -m gllm.entrypoints.worker_server \
    --model-path /path/to/model \
    --worker-gpu 1 \                    # physical GPU(s); length must equal --tp
    --worker-endpoint-file /tmp/gllm_worker_endpoint.json \
    --master-addr 127.0.0.1 --master-port 29611 \
    --tp 1 --gpu-memory-util 0.9 \
    [--worker-transport-base-port 50001]   # cross-machine frontends
    [--worker-transport-advertise-host FLEET_IP]   # see below
```

> **Cross-machine tip:** with `--worker-transport-base-port`, the worker
> *listens* on the bind host (`--master-addr`, `0.0.0.0` = all
> interfaces) but the endpoint file must carry a *routable* address —
> frontends on other hosts cannot dial `0.0.0.0`. If
> `--master-addr` is already a real IP it is reused; otherwise pass
> `--worker-transport-advertise-host FLEET_IP`. Publishing a wildcard
> is refused at startup.

### 2) Launch the stateless frontend (no GPU)

```bash
python -m gllm.entrypoints.api_server \
    --model-path /path/to/model \
    --host 0.0.0.0 --port 8000 \
    --standalone-frontend \
    --worker-endpoint-file /tmp/gllm_worker_endpoint.json
```

The frontend blocks (up to 5 min) until the worker endpoint file appears, then
serves. `GET /health` probes the worker fleet and returns `503
worker_unavailable` when it is down.

> `--model-path` is required on **both** sides: the frontend loads the
> tokenizer + HF config (CPU only, no weights / no CUDA) to tokenize requests
> and resolve the tool-call parser; the worker loads the actual model.

## Scope & limitations (current)

* Single-rank worker fleets are fully supported (`--tp 1`), including the
  cross-machine `tcp://` transport (`--worker-transport-base-port`): the
  worker child binds the exact fixed addresses published in the endpoint
  file (schedule=base, output=base+1, token=base+2), and the published
  rows use `--worker-transport-advertise-host` (falling back to
  `--master-addr`) so remote frontends get a dialable address. The
  published file carries a single rank-0 transport row; multi-rank
  fleets (TP>1 / PP>1) coordinate internally behind that one leg and
  are not independently addressable by the frontend yet.
* Encoder-disaggregation (`lm_server`) and DP-attention per-replica endpoints
  are orthogonal and not combined with standalone mode yet.
* On a worker *restart* the KV cache is lost (expected — it lived in the dead
  process); in-flight requests are terminated and must be retried by the client.

## Implementation notes

* `ModelRunner.load_metadata(...)` builds the CPU-only subset of the runner
  (tokenizer + config + `model_max_length`) used by the standalone frontend so
  it never touches CUDA.
* `zmqComm(..., standalone_remote=True)` is the frontend's connect-only socket
  layout (PUSH schedule / PULL output) pointing at the worker's bound
  endpoints. Both legs CONNECT from the frontend: the worker Binds its request
  PULL and, over `tcp://`, also Binds its output PUSH (`make_socket`'s PUSH
  connects, which is backwards for a remote frontend); over `ipc://` the
  classic PULL-bind/PUSH-connect roles are kept.
* The standalone worker **parent** must not create a second frontend-role ZMQ
  comm on the same endpoints: a `PUSH→PULL` leg load-balances across all PULLs,
  so a stray parent PULL on the output leg would silently swallow half the
  worker's output frames.
* `mp.set_warmup_delay(...)` is set on the spawn context so each GPU child
  re-reads its own `CUDA_VISIBLE_DEVICES` before any CUDA call — otherwise a
  parent that probed `torch.cuda` (seeing all GPUs) initialises the primary
  context on GPU 0 and the child inherits it, allocating on the wrong (often
  busy) GPU.
