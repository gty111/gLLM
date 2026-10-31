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


def _dspark_engine_options():
    return dict(
        model_path="/tmp/target", speculative_backend="dspark",
        dspark_model_path="/tmp/draft", maxd=1, maxp=16,
        overlap_scheduling=False, enable_prefix_caching=False,
        disable_cuda_graph=True, piecewise_cuda_graph=False,
        attention_backend="fa3",
    )


@pytest.mark.parametrize("builder", [api_server.build_arg_parser, lm_server.build_arg_parser])
def test_dspark_cli_forwards_explicit_backend_and_local_draft(builder):
    args = _parse(builder, [
        "server", "--model-path", "/tmp/target",
        "--speculative-backend", "dspark", "--dspark-model-path", "/tmp/draft",
    ])
    values = cli_args.engine_kwargs(args)
    assert values["speculative_backend"] == "dspark"
    assert values["dspark_model_path"] == "/tmp/draft"


@pytest.mark.parametrize("enabled,layers,architecture,expected", [
    (None, 0, "Qwen3_5ForConditionalGeneration", False),
    (None, 1, "Qwen3_5ForConditionalGeneration", True),
    (False, 1, "Qwen3_5ForConditionalGeneration", False),
    (True, 0, "Qwen3_5ForConditionalGeneration", True),
    (None, 3, "DeepseekV4ForCausalLM", False),
    (True, 3, "DeepseekV4ForCausalLM", True),
])
def test_legacy_mtp_selection_is_unchanged(enabled, layers, architecture, expected):
    config = EngineConfig(model_path="/tmp/target", mtp_enabled=enabled)
    assert config.resolve_mtp_enabled(architecture, layers) is expected


@pytest.mark.parametrize("backend,enabled,layers,expected", [
    ("none", None, 1, False), ("none", False, 1, False),
    ("mtp", None, 1, True), ("mtp", True, 1, True),
    ("dspark", None, 1, False), ("dspark", False, 1, False),
])
def test_explicit_backend_resolves_head_construction(backend, enabled, layers, expected):
    values = _dspark_engine_options() if backend == "dspark" else dict(model_path="/tmp/x")
    values.update(speculative_backend=backend, mtp_enabled=enabled)
    config = EngineConfig(**values)
    assert config.resolve_mtp_enabled("Qwen3_5ForConditionalGeneration", layers) is expected
    assert config.mtp_k == 3  # DSpark's seven checkpoints are a separate capacity choice.


@pytest.mark.parametrize("options,match", [
    ({"speculative_backend": "unknown"}, "speculative_backend"),
    ({"dspark_model_path": None}, "dspark_model_path"),
    ({"mtp_enabled": True}, "mtp_enabled"),
    ({"tp_size": 2}, "tp_size"), ({"pp_size": 2}, "pp_size"),
    ({"dp_size": 2}, "dp_size"), ({"maxd": 0}, "maxd"),
    ({"overlap_scheduling": True}, "overlap_scheduling"),
    ({"enable_prefix_caching": True}, "enable_prefix_caching"),
    ({"disable_cuda_graph": False}, "disable_cuda_graph"),
    ({"piecewise_cuda_graph": True}, "piecewise_cuda_graph"),
    ({"piecewise_cuda_graph": None}, "piecewise_cuda_graph"),
    ({"attention_backend": "auto"}, "attention_backend"),
    ({"schedule_method": "split_pd"}, "schedule_method"),
    ({"maxp": 7}, "maxp"), ({"load_format": "dummy"}, "load_format"),
])
def test_dspark_rejects_unsupported_engine_options(options, match):
    values = _dspark_engine_options()
    values.update(options)
    with pytest.raises(ValueError, match=match):
        EngineConfig(**values)


@pytest.mark.parametrize("backend,enabled", [("none", True), ("mtp", False)])
def test_explicit_backend_rejects_conflicting_mtp_switch(backend, enabled):
    with pytest.raises(ValueError, match="conflicts"):
        EngineConfig(model_path="/tmp/x", speculative_backend=backend, mtp_enabled=enabled)


def test_draft_path_does_not_silently_enable_dspark():
    with pytest.raises(ValueError, match="requires speculative_backend"):
        EngineConfig(model_path="/tmp/x", dspark_model_path="/tmp/draft")


@pytest.mark.parametrize("architecture,layers,k", [
    ("Qwen3_5ForConditionalGeneration", 0, 3),
    ("DeepseekV4ForCausalLM", 3, 3),
    ("Qwen3_5ForConditionalGeneration", 1, 0),
])
def test_explicit_mtp_requires_compatible_head(architecture, layers, k):
    config = EngineConfig(model_path="/tmp/x", speculative_backend="mtp", mtp_k=k)
    with pytest.raises(ValueError, match="compatible MTP"):
        config.resolve_mtp_enabled(architecture, layers)


@pytest.mark.parametrize("backend", [None, "mtp", "dspark"])
def test_dspark_rejects_actual_fa3_fallback_before_loading(monkeypatch, backend):
    from types import SimpleNamespace

    import torch
    from gllm.runtime import model_runner as module

    attempts, loaded = [], []

    class Backend:
        def __init__(self, name):
            self.name = name

        def smoke_test(self, page_size):
            if self.name == "fa3":
                raise RuntimeError("FA3 probe failed")

    def make_backend(name, *_):
        attempts.append(name)
        return Backend(name)

    def load_model(*_):
        loaded.append(True)
        raise LookupError("weight loading boundary")

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (9, 0))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(module, "create_qkv_attention_backend", make_backend)
    monkeypatch.setattr(module, "propagate_serving_config", lambda config: None)
    runner = module.ModelRunner.__new__(module.ModelRunner)
    runner.dspark_enabled = backend == "dspark"
    runner.config = SimpleNamespace(speculative_backend=backend)
    runner.attention_backend = "fa3"
    runner.model_max_length, runner.max_running_seqs, runner.page_size = 64, 1, 16
    runner.use_mla = False
    runner.model_loader = SimpleNamespace(config=SimpleNamespace(), load_model=load_model)

    expected = RuntimeError if backend == "dspark" else LookupError
    with pytest.raises(expected, match="resolved and bound FA3|weight loading boundary"):
        runner.init()
    assert runner.attention_backend == "fa4"
    assert runner._validated_qkv_attention_backend.name == "fa4"
    assert attempts == ["fa3", "fa4"]
    assert loaded == ([] if backend == "dspark" else [True])


def test_dspark_rejects_backend_object_disagreeing_with_fa3_setting():
    from types import SimpleNamespace
    from gllm.runtime.model_runner import ModelRunner

    runner = ModelRunner.__new__(ModelRunner)
    runner.dspark_enabled = True
    runner.verify_config = lambda: None
    runner.attention_backend = "fa3"
    runner._validated_qkv_attention_backend = SimpleNamespace(name="fa4")
    with pytest.raises(RuntimeError, match="resolved and bound FA3"):
        runner.init()


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
    runner.input_data.memory_manager = runner.memory_manager
    monkeypatch.setattr(torch.cuda, "synchronize", unsafe_cleanup)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    worker = SimpleNamespace(
        init=lambda: ModelRunner._profile_dspark_paged_verify(runner),
        handle_exception=handled.append,
    )
    run_worker(worker)
    assert handled == [failure]


def test_dspark_request_dispatch_rejects_an_unprepared_batch():
    from types import SimpleNamespace
    from gllm.runtime.model_runner import ModelRunner

    runner = ModelRunner.__new__(ModelRunner)
    runner.dspark_enabled = True
    runner.input_data = SimpleNamespace(seqs=[])
    with pytest.raises(ValueError, match="unpadded batch"):
        runner.step_once()


@pytest.mark.parametrize("invalid", ["missing", "empty", "multiple", "prebuilt"])
def test_dspark_input_preparation_rejects_invalid_batches_before_materialization(invalid):
    from types import SimpleNamespace
    from gllm.runtime.model_runner import ModelRunner

    def materialize(*args, **kwargs):
        pytest.fail("Invalid DSpark batches must be rejected before materializing input")

    runner = ModelRunner.__new__(ModelRunner)
    runner.dspark_enabled = True
    previous = [object()]
    runner.input_data = SimpleNamespace(
        seqs=previous,
        cal_and_set_input=materialize,
        set_input_from_prebuilt=materialize,
    )
    seqs, prebuilt = [object()], None
    if invalid == "missing":
        seqs = None
    elif invalid == "empty":
        seqs = []
    elif invalid == "multiple":
        seqs.append(object())
    else:
        prebuilt = object()
    with pytest.raises(ValueError, match="live batch"):
        runner.prepare_input(seqs, input_data=prebuilt)
    assert runner.input_data.seqs is previous

@pytest.mark.parametrize("batch", [1, 2, 4])
def test_dspark_config_accepts_a_full_verify_batch_token_budget(batch):
    values = _dspark_engine_options()
    values.update(maxd=batch, maxp=8 * batch)
    config = EngineConfig(**values)
    assert config.maxd == batch and config.maxp == 8 * batch
    assert not config.resolve_mtp_enabled("Qwen3_5ForConditionalGeneration", 1)


@pytest.mark.parametrize("batch,tokens", [(2, 15), (4, 31)])
def test_dspark_config_rejects_a_partial_maximum_verify_window(batch, tokens):
    values = _dspark_engine_options()
    values.update(maxd=batch, maxp=tokens)
    with pytest.raises(ValueError, match="maxp >= 8"):
        EngineConfig(**values)


def test_dspark_prepare_and_dispatch_delegates_a_supported_batch(monkeypatch):
    from types import SimpleNamespace
    from gllm.runtime.model_runner import ModelRunner

    runner = ModelRunner.__new__(ModelRunner)
    runner.dspark_enabled, runner.max_running_seqs = True, 2
    runner.input_data = SimpleNamespace(seqs=[])
    seqs, delegated = [object(), object()], []
    expected = object()
    def execute(batch):
        delegated.append(batch)
        return expected
    runner._execute_dspark_batch = execute
    runner.prepare_input(seqs)
    assert runner.step_once() is expected
    assert delegated == [seqs]
