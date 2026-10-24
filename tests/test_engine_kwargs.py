"""engine_kwargs is the single source of engine kwargs: entrypoints never pass
EngineConfig fields explicitly (they mutate the args namespace or rely on
defaults), so kwargs can never collide at the AsyncLLM/LLM call sites."""

import dataclasses
import sys

import pytest

from gllm.entrypoints import api_server, cli_args, lm_server
from gllm.runtime.config import EngineConfig

_FIELDS = {f.name for f in dataclasses.fields(EngineConfig)}


def _parse(parser_builder, argv):
    old = sys.argv
    sys.argv = argv
    try:
        return parser_builder().parse_args()
    finally:
        sys.argv = old


def test_api_server_forwards_all_parser_defined_engine_fields():
    args = _parse(api_server.build_arg_parser, ["api_server", "--model-path", "/tmp/x"])
    kwargs = cli_args.engine_kwargs(args)
    # Every parser arg whose dest is an EngineConfig field must be forwarded.
    missing = {n for n in vars(args) if n in _FIELDS and n not in kwargs}
    assert not missing
    # Entrypoint-owned server args never leak in.
    assert "host" not in kwargs and "port" not in kwargs
    # Topology args defined by api_server's own parser arrive under their
    # EngineConfig names.
    assert kwargs["pp_size"] == 1 and kwargs["dp_size"] == 1
    assert kwargs["use_ep"] is False
    assert kwargs["launch_mode"] == "normal"
    assert kwargs["worker_ranks"] is None and kwargs["assigned_layers"] is None
    # Renames/tri-state conversions still work.
    assert kwargs["tp_size"] == args.tp
    assert kwargs["mtp_enabled"] is None


def test_lm_server_defaults_and_ep_flag():
    args = _parse(lm_server.build_arg_parser, ["lm_server", "--model-path", "/tmp/x"])
    kwargs = cli_args.engine_kwargs(args)
    # The LM node's fixed role is just the EngineConfig defaults: the parser
    # does not define launch_mode/pp_size at all.
    assert "launch_mode" not in kwargs and "pp_size" not in kwargs
    # EP is off by default but user-controllable.
    assert kwargs["use_ep"] is False
    args = _parse(
        lm_server.build_arg_parser,
        ["lm_server", "--model-path", "/tmp/x", "--enable-ep"],
    )
    assert cli_args.engine_kwargs(args)["use_ep"] is True


def test_engine_config_validation():
    base = dict(model_path="/tmp/x")
    with pytest.raises(ValueError, match="launch_mode"):
        EngineConfig(**base, launch_mode="bogus")
    with pytest.raises(ValueError, match="worker_ranks"):
        EngineConfig(**base, launch_mode="master")
    with pytest.raises(ValueError, match="pp_size"):
        EngineConfig(**base, pp_size=0)
    # Sane defaults pass.
    EngineConfig(**base)


def test_dspark_startup_cuda_failure_reaches_worker_exit_path_without_cleanup(monkeypatch):
    from types import SimpleNamespace

    import torch
    from gllm.runtime.model_runner import ModelRunner
    from gllm.workers.worker import run_worker

    failure = RuntimeError("CUDA error: an illegal memory access was encountered")
    handled = []

    def fail_forward(*args, **kwargs):
        raise failure

    def unsafe_cleanup(*args, **kwargs):
        pytest.fail("A failed CUDA context must not be synchronized or reclaimed")

    runner = SimpleNamespace(
        max_num_batched_tokens=16, model_max_length=32, uses_mrope=False,
        dspark=SimpleNamespace(config=SimpleNamespace(target_layer_ids=(0, 1))),
        input_data=SimpleNamespace(cal_and_set_input=lambda seqs: None),
        memory_manager=SimpleNamespace(
            reserve_dspark_pages=lambda *args, **kwargs: True,
            free_dspark_cache=unsafe_cleanup,
        ),
        _prepare_attention_metadata=lambda data: None,
        model=SimpleNamespace(forward_with_hidden_taps=fail_forward),
    )
    monkeypatch.setattr(torch.cuda, "synchronize", unsafe_cleanup)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    worker = SimpleNamespace(
        init=lambda: ModelRunner._profile_dspark_paged_verify(runner),
        handle_exception=handled.append,
    )
    run_worker(worker)
    assert handled == [failure]
