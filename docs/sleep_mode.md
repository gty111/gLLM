# Sleep mode

Sleep mode lets an idle server give back most of its GPU memory and get it
back when requests return. The process, NCCL groups and captured CUDA graphs
stay alive, so waking up does not reload or recompile anything.

## Enabling

Sleep mode is on by default:

```sh
python -m gllm.entrypoints.api_server --model-path <model> \
    --sleep-idle-timeout 300   # optional; 0 (default) disables idle sleep
```

* `--sleep-mode` / `--no-sleep-mode` is the master switch (default on).
  With `--no-sleep-mode` weights and the cache arena use the ordinary
  allocator and the sleep endpoints answer 400.
* `--sleep-idle-timeout S` puts the server to sleep after `S` seconds with no
  request in flight. Leave it at `0` to sleep only through the endpoint.

A request that arrives while the server is asleep (or falling asleep) is
queued and wakes the server automatically; it runs as soon as the memory is
back. Its first-token latency therefore includes the wake-up time.

## Endpoints

| Endpoint | Description |
| --- | --- |
| `POST /sleep` | Release memory. Query: `tags` (repeatable, default all), `mode=wait` (default: let in-flight requests finish) or `mode=abort` (cancel them), `level=1` (the only supported level). |
| `POST /wake_up` | Restore memory. Query: `tags` (repeatable, default all). |
| `GET /is_sleeping` | `{"is_sleeping": bool, "sleeping_tags": [...]}` |

```sh
curl -X POST localhost:8000/sleep
curl -X POST 'localhost:8000/sleep?tags=kv_cache'      # KV cache only
curl -X POST localhost:8000/wake_up
curl localhost:8000/is_sleeping
```

`/sleep` and `/wake_up` return once every worker has finished, with the
released/restored GiB per tag (summed over all ranks) and the duration on the
slowest DP replica's driver. New requests are held from the moment `/sleep` is called; requests
resume only when no tag is asleep any more.

## What is released

| Tag | Memory | While asleep | On wake |
| --- | --- | --- | --- |
| `weights` | everything allocated during model loading | copied to pinned host memory | copied back |
| `kv_cache` | the cache arena (KV pages, recurrent states, prefix snapshots) | discarded; the prefix cache is emptied | zero-filled |

CUDA graph pools, sampler/runtime buffers, NCCL and custom all-reduce buffers,
the CUDA context and the allocator's cached activation blocks stay resident,
so once both tags are back a forward pass cannot run out of memory because of
what another process took while the server was asleep.

## Failures

* If waking fails (typically: another process now holds the GPU memory),
  `/wake_up` answers 500 and the server stays asleep. Requests that triggered
  an automatic wake-up get HTTP 503 with code `wake_up_failed`; the next
  request tries again.
* If sleeping fails on some rank, the frontend wakes every rank again (a
  no-op where nothing was released) before answering 500. If even that fails
  the server is treated as asleep, so no request runs on released memory.
* A sleep or wake-up runs to completion even if the HTTP caller disconnects.

The pinned host buffer for weights is allocated on the first sleep and kept
for later cycles, so a server that has slept once holds one copy of its
weight shard in host memory per GPU.

## How it works

With sleep mode on, model loading and the arena backing allocation run inside
PyTorch memory pools whose allocator uses CUDA virtual memory management
(`gllm/runtime/sleep_mode.py`). Each allocation reserves a virtual address
range and maps one physical handle into it. Sleeping unmaps and releases the
physical handles but keeps the reservations, so every tensor and every
captured graph keeps pointing at valid addresses; waking maps new physical
memory at the same addresses.

The frontend serializes sleep/wake, holds new requests, waits until no request
is in flight, then sends the command to every DP replica. Each column driver
runs it once its scheduler and overlap pipeline are empty, forwards it to its
PP followers, and acknowledges to the frontend. Every rank releases its own
memory.

## Encoder disaggregation

The LM registers its cache arena with NIXL so encoders can write embeddings
into it. Sleeping `kv_cache` deregisters the arena before releasing it; waking
re-registers it at the same address, gathers every TP rank's new agent
metadata on TP0 and republishes the LM to discovery before `/wake_up`
returns. Encoders reconnect on the discovery UPDATE; an encoder that writes
with the old metadata first fails, reconnects from discovery and retries.

## Limitations

* `level=2` (discarding weights entirely) is not supported; weights always
  stay in host memory.
* Idle sleep runs only in `gllm.entrypoints.api_server`.
