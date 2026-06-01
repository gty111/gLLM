"""LM Server entrypoint for encoder disaggregation (design §7.2.1).

The LM node = Frontend + Router + Scheduler + PP0 Worker + KV/SSM cache, but
WITHOUT the vision tower (``--skip-visual``). It always does prefill + decode
itself (no PD split; design §1.2). Phase 1 scope: start with the vision tower
skipped and serve text-only requests (byte-identical to the monolith text
path). The NIXL receive slot pool, the per-item meta aggregator, the router,
and discovery are layered on in Phases 3-5.

    python -m gllm.entrypoints.lm_server \
        --model-path /path/to/Qwen3.5-VL --lm-gpu 0 --port 8000 \
        --service-name dev --discovery-mode file --discovery-endpoint /tmp/gllm-disc
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
    p.add_argument("--disable-thinking", action="store_true")
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
    p.add_argument("--lm-gpu", type=int, default=0)
    p.add_argument("--master-addr", type=str, default="0.0.0.0")
    p.add_argument("--master-port", type=str, default="8001")
    p.add_argument("--zmq-port-base", type=int, default=8002)
    # NIXL receive endpoint (PP0). Used from Phase 3b.
    p.add_argument("--nixl-listen", type=str, default="0.0.0.0:9001")
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
        help="network mode: registry HOST:PORT; file mode: shared directory",
    )
    p.add_argument(
        "--discovery-mode", choices=["network", "file"], default="network"
    )
    # --- Runtime (mirror api_server defaults) ---
    p.add_argument("--gpu-memory-util", type=float, default=0.9)
    p.add_argument("--page-size", type=int, default=16)
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
    p.add_argument("--kvthresh", type=float, default=0.05)
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

    # Pin the LM to its single physical GPU before any CUDA init (design §3.2).
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.lm_gpu)
    if args.skip_visual:
        os.environ["GLLM_SKIP_VISUAL"] = "1"
    os.environ.setdefault("GLLM_SKIP_LANGUAGE", "0")

    # Encoder-disaggregation: enable the LM-side manager (slot pool + per-item
    # aggregator) + the skeleton frontend path when a discovery endpoint is
    # wired AND the vision tower is skipped. Without a discovery endpoint this
    # entrypoint is a plain text-only LM (Phase 1). These env vars are inherited
    # by the spawned worker subprocesses.
    if args.skip_visual and args.discovery_endpoint:
        from gllm.disagg.discovery import resolve_advertise_host
        from gllm.mm_common import processor_config_hash

        advertise_host = resolve_advertise_host(
            args.nixl_advertise_host, args.discovery_endpoint
        )
        os.environ["GLLM_DISAGG_LM"] = "1"
        os.environ["GLLM_DISAGG_DIR"] = args.discovery_endpoint
        os.environ["GLLM_DISAGG_MODE"] = args.discovery_mode
        os.environ["GLLM_DISAGG_PROC_HASH"] = processor_config_hash(args.model_path)
        os.environ["GLLM_DISAGG_HOST"] = advertise_host
        os.environ["GLLM_DISAGG_META_BIND"] = f"tcp://0.0.0.0:{int(args.meta_port)}"
        if args.mm_recv_slots is not None:
            os.environ["GLLM_DISAGG_NUM_SLOTS"] = str(args.mm_recv_slots)
        if args.mm_max_vis_tokens is not None:
            os.environ["GLLM_DISAGG_MAX_VIS_TOKENS"] = str(args.mm_max_vis_tokens)
        os.environ["GLLM_DISAGG_ENCODER_DP"] = str(max(1, args.encoder_dp))
        # network mode is pure-network; only file mode needs a shared directory.
        if args.discovery_mode == "file":
            os.makedirs(args.discovery_endpoint, exist_ok=True)

    # Import after env is set so the model loader sees GLLM_SKIP_VISUAL.
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
        kvthresh=args.kvthresh,
        enable_prefix_caching=args.enable_prefix_caching,
        pp_size=1,
        tp_size=1,
        use_ep=False,
        assigned_layers=None,
        schedule_method=args.schedule_method,
        overlap_scheduling=args.overlap_scheduling,
        use_thinking=not args.disable_thinking,
        disable_cuda_graph=args.disable_cuda_graph,
        max_cuda_graph_bs=args.max_cuda_graph_bs,
        model_max_length=args.model_max_length,
        mm_processor_min_pixels=args.mm_processor_min_pixels,
        mm_processor_max_pixels=args.mm_processor_max_pixels,
    )

    asyncio.run(api.run_server(args))


if __name__ == "__main__":
    main()
