import torch


def test_deepgemm_w8a8_rejects_blackwell_before_launch(monkeypatch):
    from gllm.layers.quantization import fp8

    fp8.deepgemm_available.cache_clear()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (10, 0))
    monkeypatch.setattr(
        torch,
        "randn",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("DeepGEMM probe launched on an unsupported device")
        ),
    )

    assert fp8.deepgemm_available() is False
    fp8.deepgemm_available.cache_clear()


def test_non_fp8_models_do_not_probe_fp8_backends(monkeypatch):
    from gllm.layers.quantization import fp8

    def unexpected_probe():
        raise AssertionError("FP8 backend probe ran for a non-FP8 model")

    monkeypatch.setattr(fp8, "deepgemm_available", unexpected_probe)
    monkeypatch.setattr(fp8, "flashinfer_swapab_available", unexpected_probe)
    monkeypatch.setattr(fp8, "flashinfer_groupwise_available", unexpected_probe)

    assert fp8.fp8_backend_requires_bucket_warmup(None) is False
    assert fp8.fp8_backend_requires_bucket_warmup({}) is False
    assert (
        fp8.fp8_backend_requires_bucket_warmup(
            {"quant_method": "int4_moe"}
        )
        is False
    )


def _case(m=8, round_scale=False, n=1024, k=2048):
    import torch

    from gllm.layers.quantization import fp8

    return fp8.Fp8GemmCase(M=m, N=n, K=k, block_n=128, block_k=128,
                           round_scale=round_scale, input_dtype=torch.bfloat16)


def _only(monkeypatch, **available):
    from gllm.layers.quantization import fp8

    for name in ("flashinfer_swapab_available", "flashinfer_groupwise_available",
                 "deepgemm_packed_available", "deepgemm_available"):
        monkeypatch.setattr(fp8, name, lambda v=available.get(name, False): v)
    monkeypatch.setattr(fp8, "_groupwise_weight_scale", lambda ws: ws)
    return fp8


def test_rule_table_priorities(monkeypatch):
    import torch

    scale = torch.ones(8, 16)
    fp8 = _only(monkeypatch, flashinfer_swapab_available=True, deepgemm_available=True)
    # swapAB only for small M; larger M goes to DeepGEMM.
    assert fp8.select_fp8_gemm_rule(_case(m=8), scale).name == "flashinfer_swapab"
    assert fp8.select_fp8_gemm_rule(_case(m=64), scale).name == "deepgemm"

    fp8 = _only(monkeypatch, flashinfer_groupwise_available=True, deepgemm_packed_available=True)
    # FP32 scales -> groupwise; UE8M0 scales -> packed DeepGEMM.
    assert fp8.select_fp8_gemm_rule(_case(), scale).name == "flashinfer_groupwise"
    assert fp8.select_fp8_gemm_rule(_case(round_scale=True), scale).name == "deepgemm_packed"

    fp8 = _only(monkeypatch)
    assert fp8.select_fp8_gemm_rule(_case(), scale).name == "triton"
    # Unaligned shapes never reach the block-scale kernels.
    fp8 = _only(monkeypatch, flashinfer_groupwise_available=True, deepgemm_available=True)
    assert fp8.select_fp8_gemm_rule(_case(n=1000), scale).name == "triton"
