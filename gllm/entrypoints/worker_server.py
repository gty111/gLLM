"""Standalone GPU worker entrypoint (frontend/worker decoupling).

Runs the GPU worker fleet *without* the OpenAI HTTP frontend, so the two
halves of a gLLM deployment can be managed (and restarted) independently:

    # 1) GPU worker fleet (e.g. GPU 1) -- binds ipc/tcp transport, publishes
    #    its endpoints to the rendezvous file, serves inference forever.
    python -m gllm.entrypoints.worker_server \
        --model-path /path/to/model --worker-gpu 1 \
        --worker-endpoint-file /tmp/gllm_worker_endpoint.json \
        [--worker-transport-base-port 50001]   # for cross-machine frontends

    # 2) Stateless frontend (any process, no GPU) -- connects to the fleet.
    python -m gllm.entrypoints.api_server \
        --model-path /path/to/model --port 8000 \
        --standalone-frontend \
        --worker-endpoint-file /tmp/gllm_worker_endpoint.json

Crash semantics:

* kill the worker -> in-flight requests fail fast on the (still alive)
  frontend; when a new worker publishes the endpoint file again, the
  frontend reconnects in-process. No frontend restart.
* kill the frontend -> the worker keeps running; a new frontend simply
  reconnects. No worker restart (weights stay loaded).

Currently supported topologies: single-GPU (``--tp 1``). Multi-rank fleets
(TP>1 / PP>1) publish additional endpoint rows but the standalone frontend
connects rank 0's transport; wire them up once needed.
"""

import argparse
import os

from gllm.entrypoints import cli_args


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="gLLM standalone worker (GPU fleet without the HTTP frontend)"
    )
    cli_args.add_engine_args(p)
    p.add_argument(
        "--worker-gpu",
        type=str,
        default=None,
        help=(
            "Physical GPU ordinal(s) for this worker fleet. A single ordinal "
            "for --tp 1, or a comma-separated list whose length equals --tp."
        ),
    )
    return p


def main():
    from logger import logger

    from gllm.entrypoints import cli_args as ca
    from gllm.runtime.model_loader import quiet_hub_logging

    args = build_arg_parser().parse_args()

    if not args.worker_endpoint_file:
        raise SystemExit("worker_server requires --worker-endpoint-file")

    # Pin to the requested physical GPU(s) before any CUDA init, mirroring
    # lm_server (--lm-gpu): spawned children then use local ranks 0..tp-1.
    if args.worker_gpu is not None:
        gpus = [g.strip() for g in str(args.worker_gpu).split(",") if g.strip() != ""]
        if len(gpus) != args.tp:
            raise SystemExit(
                f"--worker-gpu lists {len(gpus)} GPU(s) ({args.worker_gpu!r}) but "
                f"--tp={args.tp}; pass exactly {args.tp} comma-separated ordinal(s)."
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
    elif args.tp > 1:
        logger.warning(
            "--worker-gpu not set with --tp %d; workers will use the first %d "
            "visible devices. Set it explicitly in decoupled deployments.",
            args.tp,
            args.tp,
        )

    # Fail fast BEFORE the LLM ctor (which spawns the fleet and loads
    # weights): a TCP transport that would publish an undialable
    # wildcard must not cost a full model load to discover.
    from gllm.engine.llm import LLM
    preflight = LLM.__new__(LLM)
    preflight.host = args.master_addr if args.master_addr else "0.0.0.0"
    preflight.worker_transport_base_port = args.worker_transport_base_port
    preflight.worker_transport_advertise_host = args.worker_transport_advertise_host
    preflight.check_tcp_advertise_host()

    kwargs = ca.engine_kwargs(args)
    kwargs.update(
        host=args.master_addr if args.master_addr else "0.0.0.0",
        launch_mode="normal",
        worker_ranks=None,
        pp_size=1,
        dp_size=1,
        use_ep=False,
        assigned_layers=None,
        standalone_worker=True,
        standalone_frontend=False,
    )

    quiet_hub_logging()

    engine = LLM(**kwargs)

    logger.info(
        "Standalone worker fleet ready (endpoint file: %s). Waiting for "
        "frontend connections; requests arrive via the rendezvous transport.",
        args.worker_endpoint_file,
    )

    # Drive the engine loop directly: recv outputs -> send new work. This is
    # the same cadence AsyncLLM.schedule() runs, minus the asyncio layer (the
    # worker has no HTTP streams; the frontend owns the async streams).
    engine.mainloop()


if __name__ == "__main__":
    main()
