"""Regression test: entrypoints pass some engine kwargs explicitly alongside
``**engine_kwargs(args)`` — the two must never overlap (TypeError: multiple
values for keyword argument)."""

import sys

from gllm.entrypoints import api_server, cli_args

# Kwargs the entrypoints pass to AsyncLLM explicitly (api_server main and
# lm_server). Keep in sync with those call sites.
_EXPLICIT_KWARGS = {
    "host",
    "launch_mode",
    "worker_ranks",
    "pp_size",
    "dp_size",
    "use_ep",
    "assigned_layers",
    "disagg_config",
}


def test_engine_kwargs_do_not_collide_with_explicit_entrypoint_kwargs():
    sys.argv = ["api_server", "--model-path", "/tmp/x"]
    args = api_server.build_arg_parser().parse_args()
    kwargs = cli_args.engine_kwargs(args)
    assert not (set(kwargs) & _EXPLICIT_KWARGS)


def test_engine_kwargs_still_forward_engine_fields():
    sys.argv = ["api_server", "--model-path", "/tmp/x"]
    args = api_server.build_arg_parser().parse_args()
    kwargs = cli_args.engine_kwargs(args)
    # Engine knobs must flow through; renames are converted.
    assert kwargs["model_path"] == "/tmp/x"
    assert kwargs["tp_size"] == args.tp
    assert "host" not in kwargs  # entrypoint-only
