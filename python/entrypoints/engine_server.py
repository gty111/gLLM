"""Headless engine for an external front end (the Rust ``gllm-frontend``).

Starts the workers, prints one ``GLLM_READY <json>`` line on stdout once they
are serving, then only supervises them: requests flow straight between the
front end and the driver workers over ZMQ (see ``gllm.distributed.wire``).
The front end binds the output socket and connects to the workers' request
sockets under ``ipc:///tmp/<ipc-prefix>_gllm_*``.

Exits (taking the workers down) when a worker dies, on SIGTERM/SIGINT, or when
the parent front end goes away.

Multi-node: the master node runs ``gllm-frontend ... -- --launch-mode master
--ranks ...``; every other node runs this module directly with
``--launch-mode slave --ranks ...`` (no front end, no ``--ipc-prefix``).
"""

import argparse
import json
import os
import signal
import sys
import time

from logger import logger

from gllm.engine.llm import LLM
from gllm.entrypoints import cli_args


class HeadlessLLM(LLM):
    """Workers only: the front end lives in another process."""

    def __init__(self, ipc_prefix: str, media_arena=None, **kwargs):
        self.ipc_prefix = ipc_prefix
        self.media_arena = media_arena
        super().__init__(**kwargs)

    def _init_frontend_comm(self):
        pass


def add_disagg_lm_args(p: argparse.ArgumentParser) -> None:
    """Flags of an LM node that receives visual embeddings from encoders."""
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


def disagg_config_from_args(args):
    """The engine's encoder-disaggregation config for these flags.

    The LM-side manager (slot pool + per-item aggregator) and the skeleton
    front-end path are enabled when a discovery endpoint is wired and the
    vision tower is skipped; without one this is a plain text-only LM.
    """
    from gllm.disagg.config import DisaggConfig

    config = DisaggConfig(skip_visual=bool(args.skip_visual))
    if args.skip_visual and args.discovery_endpoint:
        from gllm.disagg.discovery import resolve_advertise_host
        from gllm.multimodal.common import processor_config_hash

        config.is_lm = True
        config.discovery_endpoint = args.discovery_endpoint
        config.processor_config_hash = processor_config_hash(args.model_path)
        config.advertise_host = resolve_advertise_host(
            args.nixl_advertise_host, args.discovery_endpoint
        )
        config.meta_bind = f"tcp://0.0.0.0:{int(args.meta_port)}"
        config.nixl_backend = args.nixl_backend
        config.encoder_dp = max(1, args.encoder_dp)
    return config


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ipc-prefix", default=None,
                   help="IPC socket prefix chosen by the front end. Omit on a "
                   "`--launch-mode slave` node, which runs workers only.")
    p.add_argument("--host", default=None)
    p.add_argument("--media-arena", default=None,
                   help="Shared-memory media arena of the front end (a path the "
                   "workers open and map).")
    cli_args.add_engine_args(p)
    cli_args.add_topology_args(p)
    # Encoder disaggregation: run as the LM node (no vision tower; visual
    # embeddings arrive from encoder servers found through discovery).
    p.add_argument("--skip-visual", action="store_true", default=False,
                   help="Do not load the vision tower (LM node of encoder disaggregation).")
    add_disagg_lm_args(p)
    return p


def ready_info(llm: HeadlessLLM) -> dict:
    runner = llm.model_runner
    gen = llm.generation_config
    return {
        "ipc_prefix": llm.ipc_prefix,
        "dp_size": llm.dp_size,
        "max_model_len": llm.model_max_length,
        "eos_token_ids": list(llm.finish_tokens),
        "use_mm": bool(runner.use_mm),
        # Dtype of preprocessed media tensors (the vision encoder's dtype).
        "dtype": str(getattr(runner.model_loader, "dtype", "")).removeprefix("torch."),
        "sleep_mode": bool(llm.config.sleep_mode),
        # The front end then sends skeleton prompts plus raw media references.
        "disagg_lm": bool(llm.is_disagg_lm),
        "sleep_idle_timeout": float(llm.config.sleep_idle_timeout or 0.0),
        "sleep_idle_level": int(llm.config.sleep_idle_level),
        # Sleep level 2 stops and restarts this process, so every worker must
        # be its child.
        "full_sleep": bool(llm.config.supports_full_sleep),
        # Sampling defaults the front end applies when a request leaves them unset.
        "defaults": {
            "temperature": gen.temperature,
            "top_p": gen.top_p,
            "top_k": gen.top_k,
            "repetition_penalty": gen.repetition_penalty,
        },
    }


def _exit_with_parent() -> None:
    """Ask the kernel to SIGTERM us when the front end dies (Linux only)."""
    try:
        import ctypes

        PR_SET_PDEATHSIG = 1
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(PR_SET_PDEATHSIG, signal.SIGTERM)
    except Exception:
        pass


def main():
    from gllm.runtime.model_loader import quiet_hub_logging

    quiet_hub_logging()
    args = build_arg_parser().parse_args()
    if args.ipc_prefix is not None:
        # Started by a front end; a slave node instead outlives its shell.
        _exit_with_parent()
    parent = os.getppid()

    def stop(signum, frame):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    disagg_config = None
    if args.skip_visual:
        disagg_config = disagg_config_from_args(args)
    if args.ipc_prefix is None and args.launch_mode != "slave":
        raise SystemExit("--ipc-prefix is required unless --launch-mode slave")
    llm = HeadlessLLM(args.ipc_prefix, media_arena=args.media_arena, host=args.host, disagg_config=disagg_config,
                      **cli_args.engine_kwargs(args))
    print("GLLM_READY " + json.dumps(ready_info(llm)), flush=True)

    while True:
        llm.check_worker_alive()  # exits the process if a worker failed
        if args.ipc_prefix is not None and os.getppid() != parent:
            logger.info("front end exited; stopping workers")
            sys.exit(0)
        time.sleep(1)


if __name__ == "__main__":
    main()
