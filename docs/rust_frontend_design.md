# Rust front end: design

Goal: replace the Python API server process with a Rust front end, for
throughput/latency and to take Python off the request path entirely. Workers
stay Python.

## Process layout

```
client ──HTTP──▶ gllm-frontend (Rust, tokio + axum)
                   │ spawns + supervises
                   ├──▶ engine launcher (Python): builds EngineConfig + ModelRunner,
                   │      spawns the workers, reports load progress / readiness,
                   │      then only supervises (not on the request path)
                   │
                   │ ZMQ + msgpack (this document)        /dev/shm media arena
                   ▼                                       ▲
                 driver worker of each DP replica ─────────┘ (TP/PP peers as today)
```

The launcher exists because the workers are `spawn` children that receive the
pickled `ModelRunner`; keeping that in Python avoids porting engine bring-up.
Workers are daemon children of the launcher, so the launcher stays alive as
their supervisor and exits (taking them down) when the front end goes away.

## Crate reuse

The front end builds on the protocol-agnostic crates of vLLM's Rust front end,
used as git dependencies of the upstream repository at one pinned commit (no
fork, no vendored source):

```
gllm-frontend (ours): axum HTTP, OpenAI chat/completions/responses wire types,
                      SSE, gLLM admin routes (sleep/wake/profile)
  └─ vllm-chat      chat templates, tool/reasoning parsing, multimodal (llm-multimodal)
     └─ vllm-text   tokenize, incremental detokenize, stop strings
        └─ dyn vllm_llm::Llm
              └─ GllmLlm (ours): ZMQ + msgpack to the workers, DP routing,
                                 abort, media arena
```

`vllm-server` is not used: it always builds an engine-core backend and its
admin routes call the engine-core client directly. Our HTTP layer covers only
what gLLM serves.

## Responsibilities

| Concern | Today (Python API server) | Rust front end |
|---|---|---|
| OpenAI protocol, validation, streaming | api_server / serving_* | yes |
| Chat template, tokenize, incremental detokenize | tokenizers/mixin | minijinja + `tokenizers` |
| Tool-call / reasoning parsing | tokenizers/tool_parsers, reasoning | per-model parser plugins |
| Structured output | validates + compiles, worker compiles again | sends the spec; worker compiles (and rejects) |
| seq_id allocation, abort, DP routing | engine/llm.py | yes |
| Sampling defaults, output-length budget | engine/llm.py `allocate_seq` | yes (reads generation_config.json) |
| Sleep/wake, idle sleep, profile | async_llm.py | yes, as control messages |
| Multimodal preprocessing | processor runs twice (front end + worker hot loop) | once, in per-model Rust plugins; raw-bytes fallback to the worker's HF processor |
| Responses API store | response_store | yes |

## Phase 0: language-neutral IPC (Python front end unchanged otherwise)

The only crossing points are, on the worker driver, `zmqComm.recv_ipc_package`
(PULL, worker binds) and `zmqComm.send_output` (PUSH to the front end's bound
PULL); on the front end their counterparts in `engine/llm.py`. Both sides
currently `send_pyobj` whole `GenerationSequence` / `IPCPackage` objects.

Phase 0 replaces those pickles with msgpack messages (`msgspec` in Python,
`rmp-serde` in Rust) defined once in `python/distributed/wire.py`. The worker
converts an incoming request into a `GenerationSequence` before
`_admit_requests`; nothing past that point changes, and TP/PP fan-out keeps its
existing internal format. The Python front end is switched to the same wire
format, so it is exercised by every existing test and A/B'd before any Rust
lands.

### Messages

`python/distributed/wire.py` and `rust/frontend/src/wire.rs` define the same
structs (msgpack arrays in field order); keep them in sync, and append new
fields at the end with a default.

```
Input   { requests: [Request], aborts: [seq_id], control: None | Control, log: None | bool }
Control { cmd: "start_profile" | "stop_profile" | "sleep" | "wake_up", tags: None | [str] }
Request {
  seq_id, token_ids,            # placeholders already expanded for media
  finish_tokens,                # eos ids + request stop_token_ids
  output_len, requested_output_len, model_max_length, ignore_eos,
  temperature, top_p, top_k, repetition_penalty,
  logprobs, prompt_logprobs,    # number of top entries; None = off
  structured: None | {schema, tag, think_start, think_end, thinking,
                      allow_think_start, prefix_tail},
  media: None | [MediaItem],    # unused by the Rust front end
  disagg_items: None | [MediaItem {modality, reference}],   # encoder disaggregation
  media_tensors: None | [Tensor],                           # merged per name, prompt order
}
Tensor  { name, dtype, shape, batched, data: bytes, offset: None | u64 }  # offset => in the arena

Output {
  seq_ids, tokens,              # a token, or a list when MTP commits several
  logprobs, prompt_logprobs, finished,
  errors: {seq_id: message}, control_ack: None | map,
  error_codes: {seq_id: code},  # absent = KV-cache capacity error
}
```

## Media arena (zero copy)

- One fixed-size `/dev/shm` file (`--media-arena-gb`, default 4; 0 sends
  tensors inline) created by the front end and passed to the engine as
  `--media-arena`. The front end unlinks the name at once, so nothing is
  left behind after a crash, and passes `/proc/<pid>/fd/<fd>` instead; every
  local worker maps it during start-up, including workers started again after
  sleep level 2.
- A request's processor outputs are merged per tensor name and written once;
  the worker builds tensors with `torch.frombuffer` over the mapping.
- A range is released when its request finishes (the front end sees it
  retire), so no extra message is needed.
- Arena full: the request waits up to 30 s for space, then gets a 503.
- Only the DP drivers, all on the front end's node, read the arena; TP/PP
  peers (possibly on other nodes) receive the tensors from their driver.

## Multimodal preprocessing

Per-model processors come from `llm-multimodal` through `vllm-chat` (registered
by model type; shared decode/resize/normalize/patchify blocks). `GllmLlm` moves
the resulting tensors into the media arena. Models it does not cover use the
`raw` fallback. Each supported model gets an element-wise parity test against
the HF processor plus an end-to-end greedy comparison.

## Status and remaining work

Done (phases 0-2):

- Wire format on both front ends; outputs identical to the pickle path (single
  GPU and DP=2), no throughput change.
- `rust/frontend`: `GllmLlm` (ZMQ + msgpack, DP round robin, abort on client
  disconnect, logprobs / prompt logprobs, structured outputs with the reasoning
  gate), sleep/wake with auto wake and idle sleep, chat + completions (stream
  and non-stream), models, health, version, profile.
- Tool-call parser picked from the architecture as before (`--tool-call-parser`
  overrides); the DeepSeek V4/V4.1 reasoning parser too, others auto-detected.
- Constraints are compiled on the driver before admission; a bad one is
  answered with a 400-class error instead of reaching the sampler.
- Media preprocessed in Rust (`llm-multimodal`), merged per tensor name and
  written once into the shared-memory arena; workers build views on it.
- Sleep level 2: the front end stops the engine process (and its workers)
  and starts it again on wake-up, reconnecting to the new workers.
- Responses API: input items (messages, function/custom tool calls and
  outputs, input images and files), function/custom tools, namespaces, client
  tool search, `store` / `previous_response_id`, streaming events.

Behaviour that differs from the Python front end:

- Stop strings and `stop_token_ids` are supported.
- `max_tokens` above the context length is refused; a smaller budget that
  does not fit next to the prompt is clamped instead of refused.
- Reasoning text and content are trimmed around `</think>`.
- A response format under thinking is compiled into one grammar that covers
  the reasoning (free text up to `</think>`, then the schema), so the tokens
  right after `</think>` may differ; the output still satisfies the schema.
- Media are passed as data URLs or http(s) URLs; bare local paths are not
  fetched. JPEG decoding matches PIL only with libturbojpeg on the loader path;
  video needs an `ffmpeg` executable on `PATH`.

Also done: encoder disaggregation (the LM front end sends skeleton prompts and
raw items as `disagg_items`), multi-node launch modes, the binary built by
`setup.py`, and the Python API server removed.

DeepSeek reasoning follows DeepSeek's own encoders: thinking is on unless
`thinking` / `enable_thinking` is false or the effort is `none`, and the
default effort is `high`. V4.1 renders the numeric budget (`low`/`high`/`max`
= 50/75/100, or 1-100). V4 uses the 0731 effort prompts, except for a
checkpoint whose shipped `encoding/encoding_dsv4` predates them: there only
`max` adds a prefix, as that encoder does.

Build and checks: Rust >= 1.95 (`rust-version`, enforced by `setup.py`); CI
runs clippy and unit tests; `rust/frontend/parity/` holds the parity scripts
and the crate-upgrade procedure. A multimodal server warns at startup when
`libturbojpeg` or `ffmpeg` is missing.

Remaining:

1. Custom tools with a Lark grammar are not checked after generation (regex
   grammars are).
2. Kimi has not been compared against its checkpoint encoder.

## Phases

0. Wire format + launcher split; Python front end on the new wire; worker-side
   `preprocessed` and `raw` media paths; arena. A/B vs master.
1. Rust MVP: `GllmLlm` + our HTTP layer over `vllm-chat`; Qwen3 text, chat +
   completions, streaming, logprobs, DP routing, abort, sleep/wake, profile.
   A/B vs the Python front end in one job.
2. Tool/reasoning parsing and structured output wired through; Qwen-VL media
   through the arena.
3. Responses API, DeepSeek V4 / Kimi encoders, encoder disaggregation.
4. Remove the Python API server.
