"""Encoder Server entrypoint (one process per DP replica, one GPU).

Phase 1 scope: load the vision tower + processor and idle, advertising
readiness. Later phases attach the ZMQ EncoderJob intake, NIXL send segment,
discovery publish/watch, and the per-item processor->ViT->NIXL-write loop
(design §4.2 / §7).

This is a standalone process: it does NOT import the LM scheduler / worker
machinery. Startup is fully decoupled from the LM server (design §7.3).

    python -m gllm.entrypoints.encoder_server \
        --model-path /path/to/Qwen3.5-VL --encoder-gpu 2 \
        --service-name dev --discovery-mode file \
        --discovery-endpoint /tmp/gllm-disc
"""

import argparse
import os
import signal
import sys


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="gLLM Encoder Server (vision-only)")
    p.add_argument("--model-path", required=True, type=str)
    p.add_argument("--load-format", choices=["auto", "dummy"], default="auto")
    # This process owns exactly one GPU (design §3.2 / §7.2.2).
    p.add_argument("--encoder-gpu", type=int, default=0)
    # Self endpoints; peers discover these via the registry, never hard-coded.
    # Port 0 = ephemeral (default; fine same-host / multi-replica). For a
    # firewalled cross-machine deploy, pin a fixed port (e.g. 0.0.0.0:9100) so a
    # static ingress rule can be written; the advertised host is --advertise-host.
    p.add_argument("--zmq-listen", type=str, default="0.0.0.0:0")
    p.add_argument("--nixl-listen", type=str, default="0.0.0.0:9101")
    # Service-group membership + discovery (design §7.3).
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
    p.add_argument("--encoder-id", type=str, default=None)
    p.add_argument("--mm-embed-cache-size", type=float, default=256.0,
                   help="Per-replica content_hash->embed dedup cache size (MB)")
    p.add_argument("--mm-processor-min-pixels", type=int, default=None)
    p.add_argument("--mm-processor-max-pixels", type=int, default=None)
    p.add_argument("--nixl-backend", choices=["UCX"], default="UCX")
    p.add_argument(
        "--advertise-host",
        type=str,
        default="auto",
        help="Host the LM connects back to for ZMQ. 'auto' (default) detects the "
        "routable egress IP toward the discovery server; pass an explicit IP to "
        "override, or '127.0.0.1' for single-node loopback.",
    )
    p.add_argument(
        "--max-vis-tokens",
        type=int,
        default=16384,
        help="Upper bound on N_vis per item; sizes the registered send buffer",
    )
    return p


def default_encoder_id(gpu: int) -> str:
    import socket

    return f"{socket.gethostname()}-{os.getpid()}-gpu{gpu}"


def main():
    args = build_arg_parser().parse_args()

    os.environ["GLLM_SKIP_LANGUAGE"] = "1"
    os.environ["GLLM_SKIP_VISUAL"] = "0"

    # Pin this process to its single physical GPU by selecting the device
    # directly (design §3.2 mutual-exclusion invariant). We do this instead of
    # masking with CUDA_VISIBLE_DEVICES because the gllm import chain can create
    # a CUDA context on device 0 before any in-process env mask would take
    # effect; ``set_device`` picks the physical ordinal regardless and must run
    # as the very first CUDA call so the primary context lands on --encoder-gpu.
    import torch

    torch.cuda.set_device(args.encoder_gpu)

    from logger import logger

    from gllm.utils import init_logger

    init_logger()

    from gllm.encoder_engine import EncoderEngine

    encoder_id = args.encoder_id or default_encoder_id(args.encoder_gpu)
    logger.info(
        f"Starting encoder_server id={encoder_id} gpu={args.encoder_gpu} "
        f"service={args.service_name} discovery={args.discovery_mode}"
    )

    engine = EncoderEngine(
        model_path=args.model_path,
        load_format=args.load_format,
        mm_processor_min_pixels=args.mm_processor_min_pixels,
        mm_processor_max_pixels=args.mm_processor_max_pixels,
        mm_embed_cache_mb=args.mm_embed_cache_size,
    )
    engine.init()

    if not args.discovery_endpoint:
        raise SystemExit(
            "--discovery-endpoint is required to connect the encoder to an LM "
            "node (network mode: registry HOST:PORT; file mode: a directory)"
        )

    from gllm.disagg.discovery import resolve_advertise_host
    from gllm.disagg.encoder_runtime import EncoderRuntime
    from gllm.mm_common import processor_config_hash

    advertise_host = resolve_advertise_host(
        args.advertise_host, args.discovery_endpoint
    )
    logger.info(
        f"Encoder {encoder_id} advertising ZMQ job intake on host {advertise_host} "
        f"(--advertise-host={args.advertise_host})"
    )

    runtime = EncoderRuntime(
        engine,
        encoder_id=encoder_id,
        discovery_endpoint=args.discovery_endpoint,
        discovery_mode=args.discovery_mode,
        processor_config_hash=processor_config_hash(args.model_path),
        advertise_host=advertise_host,
        # Honor the configured host:port (port 0 -> ephemeral). A fixed port is
        # what a cross-machine firewall rule targets.
        job_bind=f"tcp://{args.zmq_listen}",
        max_vis_tokens=args.max_vis_tokens,
        nixl_backend=args.nixl_backend,
    )
    runtime.setup()
    logger.info(f"Encoder {encoder_id} READY; entering job loop")

    def _sig(_signum, _frame):
        logger.info(f"Encoder {encoder_id} shutting down")
        sys.stdout.flush()
        os._exit(0)

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    runtime.serve_forever()


if __name__ == "__main__":
    main()
