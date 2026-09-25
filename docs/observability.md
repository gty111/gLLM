# Prometheus Observability

gLLM ships built-in metrics collection based on
[prometheus_client](https://github.com/prometheus/client_python), covering the
request lifecycle and internal engine state. The dependency is soft: when
`prometheus-client` is not installed or metrics are explicitly disabled, every
instrumentation site degrades to a no-op with no performance impact.

## Enabling

Metrics are **off by default** (consistent with SGLang) and must be turned on
explicitly:

```bash
# 1. Install the dependency
pip install prometheus-client

# 2. Pass --enable-metrics at startup
python -m gllm.entrypoints.api_server --model-path /path/to/model \
    --enable-metrics ...
```

Without `--enable-metrics` (or without `prometheus-client` installed), all
instrumentation sites become no-ops and `/metrics` responds with a single
informational line, so there is no measurable overhead.

The scrape endpoint is mounted directly on the API server's main port:

```bash
curl http://<host>:<port>/metrics
```

## Metric Reference

All metric names are prefixed with `gllm_`.

### Request lifecycle (frontend, main process)

| Metric | Type | Description |
|---|---|---|
| `gllm_requests_total{model,method,streaming}` | Counter | Total inference requests received, split by `/v1/chat/completions`, `/v1/completions`, `/v1/responses` |
| `gllm_request_success_total{model,finish_reason}` | Counter | Requests completed normally (stop/length) |
| `gllm_request_e2e_latency_seconds` | Histogram | End-to-end latency (request received → all output produced) |
| `gllm_time_to_first_token_seconds` | Histogram | TTFT (queuing + first-token generation) |
| `gllm_time_per_output_token_seconds` | Histogram | Mean inter-token latency over the streaming window |
| `gllm_prompt_tokens_total{model}` | Counter | Total prompt tokens fed to the engine |
| `gllm_generation_tokens_total{model}` | Counter | Total output tokens generated |
| `gllm_prompt_tokens_per_request` | Histogram | Prompt length distribution per request |
| `gllm_generation_tokens_per_request` | Histogram | Output length distribution per request |
| `gllm_token_throughput_per_second` | Gauge | Output token throughput (exponential moving average) |

### Scheduling and caching (reported by workers, relayed over ZMQ)

| Metric | Type | Description |
|---|---|---|
| `gllm_num_requests_running` | Gauge | Sequences currently executing in the engine |
| `gllm_num_requests_waiting` | Gauge | New sequences awaiting admission (frontend queue) |
| `gllm_num_steps_total` | Counter | Engine iteration steps |
| `gllm_iteration_tokens_total` | Counter | Token rows computed per step (prefill + decode) |
| `gllm_prefill_tokens_total` | Counter | Prompt query tokens scheduled per engine step (actual forward compute, prefill phase) |
| `gllm_decode_tokens_total` | Counter | Decode query tokens scheduled per engine step (includes the MTP verify query width) |

Throughput should be derived from counters, not the legacy EMA gauge:

```promql
# Total generated-token throughput
sum(rate(gllm_generation_tokens_total[1m])) by (instance)
# Per-phase scheduled throughput
sum(rate(gllm_prefill_tokens_total[1m])) by (instance)
sum(rate(gllm_decode_tokens_total[1m])) by (instance)
```

> Note: `gllm_token_throughput_per_second` is a legacy EMA gauge that only
> updates on request completion and holds its last value while idle. Prefer
> the `rate()` expressions above.
| `gllm_finished_requests_total` | Counter | Sequences that reached a terminal state |
| `gllm_num_preemptions_total` | Counter | Sequences preempted |
| `gllm_kv_cache_pages_total` | Gauge | Total KV cache physical pages |
| `gllm_kv_cache_pages_free` | Gauge | Free KV cache pages |
| `gllm_kv_cache_usage_pct` | Gauge | KV cache utilization (%) |
| `gllm_batch_prefill_seqs` | Gauge | Prefill rows in the most recent step |
| `gllm_batch_decode_seqs` | Gauge | Decode rows in the most recent step |
| `gllm_batch_tokens` | Gauge | Total tokens in the most recent step |
| `gllm_gpu_memory_allocated_bytes` | Histogram | torch CUDA allocated memory |
| `gllm_gpu_memory_reserved_bytes` | Histogram | torch CUDA reserved memory |
| `gllm_prefix_cache_hit_rate` | Gauge | Prefix cache hit rate (hit pages / (hit pages + allocated pages)); only populated with `--enable-prefix-caching` |
| `gllm_uptime_seconds` | Gauge | Process uptime |

## Architecture

```
┌─────────────────────────── Main process ───────────────────────────┐
│  FastAPI /metrics  ──►  FrontendMetrics (CollectorRegistry)        │
│        ▲                          ▲                                │
│        │ request lifecycle        │ stats merge                    │
│  api_server entry points      llm.recv_ipc_package()              │
│  (begin/first_token/token/    (update_from_engine_stats)          │
│   finish via RequestTracker)                                       │
└──────────────────────────────▲────────────────────────────────────┘
                               │ IPCPackage.stats (ZMQ, existing output channel)
┌──────────────────────────────┴────────────────────────────────────┐
│  Worker subprocess (per TP/DP rank)                               │
│  Scheduler.engine_stats: EngineStats                              │
│    · record_iteration / record_kv / record_queues                 │
│      (every scheduling tick)                                      │
│    · record_prefix_cache / record_gpu_memory                      │
│      (at prefill admission / on outbound package)                 │
│  On outbound IPC package: snapshot() → IPCPackage.stats           │
└───────────────────────────────────────────────────────────────────┘
```

No new port is introduced on the worker side: statistics reuse the existing
worker→frontend ZMQ output channel (the `IPCPackage.stats` field) and ride
along with every output package.

## Grafana Query Examples

```promql
# Live output throughput
gllm_token_throughput_per_second

# P99 TTFT
histogram_quantile(0.99, sum(rate(gllm_time_to_first_token_seconds_bucket[1m])) by (le))

# KV cache utilization
gllm_kv_cache_usage_pct

# Tokens generated per minute
rate(gllm_generation_tokens_total[1m]) * 60
```
