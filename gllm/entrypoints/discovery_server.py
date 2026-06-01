"""Standalone discovery registry for encoder disaggregation (design §7.3.2).

A tiny, dependency-free ZMQ rendezvous service that the LM and Encoder servers
publish to and watch -- the network (etcd-style) alternative to a shared
filesystem. Start one of these per service group; point every ``lm_server`` and
``encoder_server`` at it with ``--discovery-mode network --discovery-endpoint
HOST:PORT``.

    python -m gllm.entrypoints.discovery_server --listen 0.0.0.0:9500
"""

import argparse
import signal
import sys


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="gLLM discovery registry (network)")
    p.add_argument(
        "--listen",
        type=str,
        default="0.0.0.0:9500",
        help="host:port (or tcp://host:port) to bind the ROUTER socket on",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    from gllm.utils import init_logger

    init_logger()

    from logger import logger

    from gllm.disagg.discovery import DiscoveryServer

    server = DiscoveryServer(args.listen)

    def _sig(_signum, _frame):
        logger.info("discovery_server shutting down")
        sys.stdout.flush()
        server.stop()

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    server.serve_forever()


if __name__ == "__main__":
    main()
