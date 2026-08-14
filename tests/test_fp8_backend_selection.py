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

    assert fp8.fp8_backend_requires_bucket_warmup(None) is False
    assert fp8.fp8_backend_requires_bucket_warmup({}) is False
    assert (
        fp8.fp8_backend_requires_bucket_warmup(
            {"quant_method": "int4_moe"}
        )
        is False
    )
