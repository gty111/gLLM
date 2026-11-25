# Front-end parity checks

Scripts that check `gllm-frontend` against a reference. Run them after changing
the front end, and always after moving the vLLM crates to a new `rev` in
`../Cargo.toml` (all `vllm-*` entries share one rev; keep the `pcre2` patch and
`Cargo.lock` in step with that rev's workspace).

| Script | Needs | Checks |
| --- | --- | --- |
| `render_parity.py <model> [dsv4\|dsv32]` | CPU | prompt tokens: offline Python encode path vs `--render-request` |
| `encoder_parity.py <model>` | CPU | prompt tokens vs a DeepSeek V4 / V4.1 checkpoint's own encoder |
| `compare_servers.py <ref> <new>` | 2 servers | chat, completions, logprobs, structured output, tools, media, streaming, abort, profile, sleep/wake, Responses |
| `compare_disagg.py <ref> <new>` | 2 LM nodes + encoders | encoder disaggregation answers (text, PNG, JPEG, two images) |
| `dp_check.py <base>` | 1 server, `--dp 2` | concurrency, abort, profile, sleep/wake across replicas |
| `sleep_check.py <base> [--idle]` | 1 server | sleep levels 1 and 2: same answers after each cycle, GPU memory released, auto wake, abort; `--idle` for a server started with `--sleep-idle-level 2` |
| `bench.py <base> <conc> <n> <max_tokens>` | 1 server | throughput and TTFT |

The CPU scripts use the binary given by `--frontend` (default: the
`gllm-frontend` on `PATH`) and need the gLLM Python package importable.

## Upgrading the vLLM crates

1. Build the current binary and keep a copy as the reference.
2. Change the rev, copy the new `Cargo.lock` and `[patch.crates-io]` from that
   vLLM rev, then `cargo build --release` and `cargo clippy --all-targets`.
3. CPU: `render_parity.py` on a Qwen model, and on DeepSeek V4 with `dsv4`;
   `encoder_parity.py` on DeepSeek V4 and V4.1.
4. GPU, in one job: start the reference and the new binary on separate GPUs
   with the same model and engine flags (a multimodal Qwen model covers the
   most), then `compare_servers.py ref new`. Run `dp_check.py` on the new
   binary with `--dp 2`, `sleep_check.py` if the change touches sleep or
   the engine process, and `compare_disagg.py` if the change touches media.
5. `bench.py` on both binaries in the same job; compare relative numbers only.

`media/` holds the small images and clip the media cases send as data URLs.
