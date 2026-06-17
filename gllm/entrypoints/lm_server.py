"""LM Server entrypoint for encoder disaggregation (design §7.2.1).

The LM node = Frontend + Router + Scheduler + PP0 Worker + KV/SSM cache, but
WITHOUT the vision tower (``--skip-visual``). It always does prefill + decode
itself (no PD split; design §1.2). Phase 1 scope: start with the vision tower
skipped and serve text-only requests (byte-identical to the monolith text
path). The NIXL receive slot pool, the per-item meta aggregator, the router,
and discovery are layered on in Phases 3-5.

    python -m gllm.entrypoints.lm_server \
        --model-path /path/to/Qwen3.5-VL --lm-gpu 0 --port 8000 \
        --service-name dev --discovery-endpoint 127.0.0.1:9500
"""

import argparse
import asyncio
import os


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="gLLM LM Server (encoder-disaggregated)")
    # --- Model ---
    p.add_argument("--model-path", required=True, type=str)
    p.add_argument("--load-format", choices=["auto", "dummy"], default="auto")
    p.add_argument("--model-max-length", type=int, default=None)
    # The defining flag of the LM node. On by default for this entrypoint.
    p.add_argument(
        "--skip-visual",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Do not load the vision tower; visual embeds arrive over NIXL.",
    )
    # --- Network / GPU ---
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument(
        "--lm-gpu",
        type=str,
        default="0",
        help="Physical GPU(s) for the LM. A single ordinal (e.g. '0') for "
        "tp_size=1, or a comma-separated list (e.g. '0,1') whose length equals "
        "--tp for tensor parallelism.",
    )
    p.add_argument(
        "--tp",
        type=int,
        default=1,
        help="LM tensor-parallel size. The full visual embedding is "
        "multi-written by the encoder into every TP rank's slot pool; "
        "pp_size stays 1.",
    )
    p.add_argument("--master-addr", type=str, default="0.0.0.0")
    p.add_argument("--master-port", type=str, default="8001")
    p.add_argument("--zmq-port-base", type=int, default=8002)
    # NIXL transport backend (PP0 receive side). The data-plane endpoint is
    # auto-negotiated via the metadata exchanged over the ZMQ control plane, so
    # there is no fixed listen port to configure here.
    p.add_argument("--nixl-backend", choices=["UCX"], default="UCX")
    p.add_argument(
        "--meta-port",
        type=int,
        default=0,
        help="Fixed TCP port for the PP0 per-item meta intake socket (encoders "
        "PUSH MmItemMeta here). 0 (default) = ephemeral; pin it for a firewalled "
        "cross-machine deploy so a static ingress rule can target it.",
    )
    p.add_argument(
        "--nixl-advertise-host",
        type=str,
        default="auto",
        help="Host the encoder connects back to for ZMQ meta. 'auto' (default) "
        "detects the routable egress IP toward the discovery server; pass an "
        "explicit IP to override, or '127.0.0.1' for single-node loopback.",
    )
    # Receive slot pool (design §5.3). Used from Phase 3b.
    p.add_argument("--mm-recv-slots", type=int, default=None)
    p.add_argument("--mm-max-vis-tokens", type=int, default=None)
    p.add_argument(
        "--encoder-dp",
        type=int,
        default=1,
        help="Number of encoder replicas expected in discovery before LM starts dispatching.",
    )
    # --- Discovery (Phase 4) ---
    p.add_argument("--service-name", type=str, default="gllm-lm-prod")
    p.add_argument(
        "--discovery-endpoint",
        type=str,
        default=None,
        help="discovery registry HOST:PORT (run discovery_server there)",
    )
    # --- Runtime (mirror api_server defaults) ---
    p.add_argument("--gpu-memory-util", type=float, default=0.9)
    p.add_argument("--page-size", type=int, default=16)
    p.add_argument(
        "--mla-decode-backend",
        type=str,
        choices=["flashmla", "triton"],
        default="flashmla",
        help=(
            "MLA decode attention backend. 'flashmla' (default) auto-bumps "
            "page_size to 64 and falls back to Triton if unavailable."
        ),
    )
    p.add_argument("--disable-cuda-graph", action="store_true")
    p.add_argument("--max-cuda-graph-bs", type=int, default=512)
    p.add_argument(
        "--overlap-scheduling",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--enable-prefix-caching",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--maxd", type=int, default=512)
    p.add_argument("--maxp", type=int, default=8192)
    p.add_argument("--minp", type=int, default=32)
    p.add_argument("--iterp", type=int, default=8)
    p.add_argument("--init-new-token-ratio", type=float, default=0.7)
    p.add_argument("--min-new-token-ratio", type=float, default=0.1)
    p.add_argument(
        "--schedule-method",
        choices=["split_pd", "chunked_prefill", "token_throttling"],
        default="chunked_prefill",
    )
    p.add_argument("--mm-processor-min-pixels", type=int, default=None)
    p.add_argument("--mm-processor-max-pixels", type=int, default=None)
    return p


def main():
    args = build_arg_parser().parse_args()

    # Pin the LM to its physical GPU(s) before any CUDA init (design §3.2). For
    # tp_size>1 the visible-device list length must match --tp (the spawned
    # workers use local ranks 0..tp-1 as cuda:0..tp-1 within this mask).
    lm_gpus = [g.strip() for g in str(args.lm_gpu).split(",") if g.strip() != ""]
    if len(lm_gpus) != args.tp:
        raise SystemExit(
            f"--lm-gpu lists {len(lm_gpus)} GPU(s) ({args.lm_gpu!r}) but --tp={args.tp}; "
            f"pass exactly {args.tp} comma-separated ordinal(s)."
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(lm_gpus)

    from gllm.disagg.config import DisaggConfig

    # Encoder-disaggregation config, threaded explicitly into the engine /
    # worker (no GLLM_DISAGG_* / GLLM_SKIP_* env). The LM-side manager (slot
    # pool + per-item aggregator) + the skeleton frontend path are enabled when
    # a discovery endpoint is wired AND the vision tower is skipped. Without a
    # discovery endpoint this entrypoint is a plain text-only LM (Phase 1).
    disagg_config = DisaggConfig(skip_visual=bool(args.skip_visual))
    if args.skip_visual and args.discovery_endpoint:
        from gllm.disagg.discovery import resolve_advertise_host
        from gllm.mm_common import processor_config_hash

        advertise_host = resolve_advertise_host(
            args.nixl_advertise_host, args.discovery_endpoint
        )
        disagg_config.is_lm = True
        disagg_config.discovery_endpoint = args.discovery_endpoint
        disagg_config.processor_config_hash = processor_config_hash(args.model_path)
        disagg_config.advertise_host = advertise_host
        disagg_config.meta_bind = f"tcp://0.0.0.0:{int(args.meta_port)}"
        disagg_config.nixl_backend = args.nixl_backend
        disagg_config.num_slots = args.mm_recv_slots
        disagg_config.max_vis_tokens = args.mm_max_vis_tokens
        disagg_config.encoder_dp = max(1, args.encoder_dp)

    import gllm.entrypoints.api_server as api
    from gllm.async_llm_engine import PipeAsyncLLM

    api.llm = PipeAsyncLLM(
        host=args.host,
        master_addr=args.master_addr,
        master_port=args.master_port,
        zmq_port_base=args.zmq_port_base,
        launch_mode="normal",
        worker_ranks=None,
        load_format=args.load_format,
        model_path=args.model_path,
        gpu_memory_util=args.gpu_memory_util,
        page_size=args.page_size,
        maxd=args.maxd,
        maxp=args.maxp,
        minp=args.minp,
        iterp=args.iterp,
        init_new_token_ratio=args.init_new_token_ratio,
        min_new_token_ratio=args.min_new_token_ratio,
        enable_prefix_caching=args.enable_prefix_caching,
        pp_size=1,
        tp_size=args.tp,
        use_ep=False,
        assigned_layers=None,
        schedule_method=args.schedule_method,
        overlap_scheduling=args.overlap_scheduling,
        disable_cuda_graph=args.disable_cuda_graph,
        max_cuda_graph_bs=args.max_cuda_graph_bs,
        model_max_length=args.model_max_length,
        mm_processor_min_pixels=args.mm_processor_min_pixels,
        mm_processor_max_pixels=args.mm_processor_max_pixels,
        disagg_config=disagg_config,
        mla_decode_backend=args.mla_decode_backend,
    )

    asyncio.run(api.run_server(args))


if __name__ == "__main__":
    main()
