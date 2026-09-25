# Prometheus 可观测性

gLLM 内置基于 [prometheus_client](https://github.com/prometheus/client_python)
的指标采集，覆盖请求生命周期与引擎内部状态。依赖为软依赖：未安装
`prometheus-client` 或显式关闭时，所有埋点自动降级为无操作，不影响性能。

## 启用方式

指标**默认关闭**（与 SGLang 一致），需要显式开启：

```bash
# 1. 安装依赖
pip install prometheus-client

# 2. 启动时加 --enable-metrics
python -m gllm.entrypoints.api_server --model-path /path/to/model \
    --enable-metrics ...
```

不加 `--enable-metrics`（或未安装 `prometheus-client`）时，所有埋点自动降级为
无操作，`/metrics` 返回一行说明，对性能无影响。

抓取端点直接挂在 API server 主端口上：

```bash
curl http://<host>:<port>/metrics
```

## 指标一览

指标名均以 `gllm_` 为前缀。

### 请求生命周期（前端，主进程）

| 指标 | 类型 | 说明 |
|---|---|---|
| `gllm_requests_total{model,method,streaming}` | Counter | 收到的推理请求总数，按 `/v1/chat/completions`、`/v1/completions`、`/v1/responses` 区分 |
| `gllm_request_success_total{model,finish_reason}` | Counter | 正常完成的请求数（stop/length） |
| `gllm_request_e2e_latency_seconds` | Histogram | 端到端延迟（收到请求 → 全部产出） |
| `gllm_time_to_first_token_seconds` | Histogram | TTFT（含排队 + 首 token 生成） |
| `gllm_time_per_output_token_seconds` | Histogram | 平均 inter-token 延迟（流式窗口内） |
| `gllm_prompt_tokens_total{model}` | Counter | 进入引擎的 prompt token 总数 |
| `gllm_generation_tokens_total{model}` | Counter | 生成的 output token 总数 |
| `gllm_prompt_tokens_per_request` | Histogram | 单请求 prompt 长度分布 |
| `gllm_generation_tokens_per_request` | Histogram | 单请求输出长度分布 |
| `gllm_token_throughput_per_second` | Gauge | 输出 token 吞吐（指数滑动平均） |

### 调度与缓存（worker 上报，经 ZMQ 回传）

| 指标 | 类型 | 说明 |
|---|---|---|
| `gllm_num_requests_running` | Gauge | 引擎中正在执行的序列数 |
| `gllm_num_requests_waiting` | Gauge | 等待准入的新序列数（前端队列） |
| `gllm_num_steps_total` | Counter | 引擎迭代步数 |
| `gllm_iteration_tokens_total` | Counter | 每步计算的 token 行数（prefill+decode） |
| `gllm_finished_requests_total` | Counter | 到达终态的序列数 |
| `gllm_num_preemptions_total` | Counter | 被抢占的序列数 |
| `gllm_kv_cache_pages_total` | Gauge | KV cache 物理页总数 |
| `gllm_kv_cache_pages_free` | Gauge | 空闲 KV 页数 |
| `gllm_kv_cache_usage_pct` | Gauge | KV cache 使用率（%） |
| `gllm_batch_prefill_seqs` | Gauge | 最近一步的 prefill 行数 |
| `gllm_batch_decode_seqs` | Gauge | 最近一步的 decode 行数 |
| `gllm_batch_tokens` | Gauge | 最近一步的总 token 数 |
| `gllm_gpu_memory_allocated_bytes` | Histogram | torch CUDA 已分配显存 |
| `gllm_gpu_memory_reserved_bytes` | Histogram | torch CUDA 已保留显存 |
| `gllm_prefix_cache_hit_rate` | Gauge | 前缀缓存命中率（命中页 / (命中页+分配页)），仅 `--enable-prefix-caching` 时有值 |
| `gllm_uptime_seconds` | Gauge | 进程运行时长 |

## 架构

```
┌─────────────────────────── 主进程 ───────────────────────────┐
│  FastAPI /metrics  ──►  FrontendMetrics (CollectorRegistry)  │
│        ▲                          ▲                          │
│        │ 请求生命周期埋点           │ stats 合并                │
│  api_server 三个入口点        llm.recv_ipc_package()          │
│  (begin/first_token/token/    (update_from_engine_stats)      │
│   finish via RequestTracker)                                 │
└──────────────────────────────▲───────────────────────────────┘
                               │ IPCPackage.stats (ZMQ, 已有的输出通道)
┌──────────────────────────────┴───────────────────────────────┐
│  Worker 子进程（每 TP/DP rank）                                │
│  Scheduler.engine_stats: EngineStats                         │
│    · record_iteration / record_kv / record_queues            │
│      （每个调度 tick）                                         │
│    · record_prefix_cache / record_gpu_memory                 │
│      （prefill 准入时 / 出包时）                                │
│  输出 IPC 包时 snapshot() → IPCPackage.stats                  │
└──────────────────────────────────────────────────────────────┘
```

worker 侧不新增任何端口：统计复用既有的 worker→frontend ZMQ 输出
通道（`IPCPackage.stats` 字段），随每个输出包附带。

## Grafana 查询示例

```promql
# 实时输出吞吐
gllm_token_throughput_per_second

# P99 TTFT
histogram_quantile(0.99, sum(rate(gllm_time_to_first_token_seconds_bucket[1m])) by (le))

# KV cache 使用率
gllm_kv_cache_usage_pct

# 每分钟生成 token 数
rate(gllm_generation_tokens_total[1m]) * 60
```
