"""Round-trip DeepSeek-V4 weight loading against a synthetic checkpoint.

The rule-matching test next door proves each parameter reaches the *intended*
handler. This one proves the handlers are wired correctly: that every
checkpoint key a handler asks for is one the mapping produces, and that the
value that lands is the value the checkpoint held. Running the real loader is
the only way to catch a handler whose lookup key does not match the context it
was given.
"""

import pytest
import torch

from gllm.layers.quantization.fp8 import block_fp8_scale_to_float32
from gllm.models.deepseek_v4 import DeepseekV4ForCausalLM, _v4_src_key

from test_deepseek_v4_weight_rules import _config


def _source_for(key: str, shape, dtype: torch.dtype) -> torch.Tensor:
    """Deterministic checkpoint tensor for ``key``, in the checkpoint's dtype."""
    generator = torch.Generator().manual_seed(abs(hash(key)) % (2**31))
    if dtype is torch.uint8:
        return torch.randint(
            0, 255, shape, generator=generator, dtype=torch.uint8
        )
    if dtype is torch.float8_e4m3fn:
        return (
            torch.randn(shape, generator=generator) * 0.1
        ).to(torch.float8_e4m3fn)
    if dtype is torch.int32:
        return torch.randint(0, 4, shape, generator=generator, dtype=torch.int32)
    return (torch.randn(shape, generator=generator) * 0.1).to(dtype)


def _build_checkpoint(model, config):
    """One checkpoint tensor per key the rule table will ask for (TP=1)."""
    weights = {}
    hidden = config.hidden_size
    inter = config.moe_intermediate_size

    for path, param in model.named_parameters():
        key = _v4_src_key(path)

        if ".ffn.experts." in key:
            prefix, field = key.rsplit(".", 1)
            suffix = "scale" if field.endswith("scale") else "weight"
            names = ("w1", "w3") if field.startswith("w13") else ("w2",)
            per_expert = param.shape[1] // len(names)
            for expert in range(config.n_routed_experts):
                for name in names:
                    shape = (per_expert, param.shape[2])
                    weights[f"{prefix}.{expert}.{name}.{suffix}"] = _source_for(
                        f"{prefix}.{expert}.{name}.{suffix}", shape, torch.uint8
                    )
            continue

        if "shared_experts.gate_up_proj" in key:
            base, field = key.rsplit(".gate_up_proj.", 1)
            suffix = "scale" if field == "scale" else "weight"
            half = param.shape[0] // 2
            for name in ("w1", "w3"):
                dtype = torch.uint8 if suffix == "scale" else param.dtype
                weights[f"{base}.{name}.{suffix}"] = _source_for(
                    f"{base}.{name}.{suffix}", (half, param.shape[1]), dtype
                )
            continue

        if "shared_experts.down_proj" in key:
            base, field = key.rsplit(".down_proj.", 1)
            suffix = "scale" if field == "scale" else "weight"
            dtype = torch.uint8 if suffix == "scale" else param.dtype
            weights[f"{base}.w2.{suffix}"] = _source_for(
                f"{base}.w2.{suffix}", tuple(param.shape), dtype
            )
            continue

        if key.endswith("e_score_correction_bias"):
            key = key.replace("e_score_correction_bias", "gate.bias")
        elif key.endswith("tid2eid"):
            key = key.replace("tid2eid", "gate.tid2eid")

        shape = tuple(param.shape)
        dtype = param.dtype
        if key.endswith(".scale"):
            dtype = torch.uint8
        elif key in ("embed.weight", "head.weight"):
            # The checkpoint holds the unpadded vocabulary.
            shape = (config.vocab_size, hidden)
            dtype = torch.bfloat16
        elif dtype is torch.float32:
            # The checkpoint stores these in BF16; the model promotes them.
            dtype = torch.bfloat16
        weights[key] = _source_for(key, shape, dtype)
    return weights


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_load_weights_places_every_tensor_it_was_given(monkeypatch):
    config = _config()
    model = DeepseekV4ForCausalLM(config)
    weights = _build_checkpoint(model, config)

    # ``load_weights`` finishes by repacking the routed-expert tensors into the
    # kernel's layout, which would hide what actually landed. Suppress that one
    # step so the assertions below see the loaded bytes; the repack itself is
    # covered by ``test_mxfp4_moe``.
    monkeypatch.setattr(
        type(model.model.layers[0].ffn.experts),
        "process_weights_after_loading",
        lambda self: None,
    )

    # Loading must not raise: every key a handler builds has to be one the
    # synthetic checkpoint (built from the same mapping) actually contains.
    model.load_weights(weights)

    projections = model.model.layers[1].attn.projections
    torch.testing.assert_close(
        projections.wq_b.weight.float().cpu(),
        weights["layers.1.attn.wq_b.weight"].float(),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        projections.wq_b.weight_scale_inv.cpu(),
        block_fp8_scale_to_float32(weights["layers.1.attn.wq_b.scale"]),
        rtol=0,
        atol=0,
    )
    # Replicated down-projection: whole tensor, not a shard.
    torch.testing.assert_close(
        projections.wq_a.weight.float().cpu(),
        weights["layers.1.attn.wq_a.weight"].float(),
        rtol=0,
        atol=0,
    )

    # The compressor owns a ``wkv`` of its own and must have taken its own key.
    compressor = model.model.layers[1].attn.compressor
    torch.testing.assert_close(
        compressor.wkv.weight.cpu(),
        weights["layers.1.attn.compressor.wkv.weight"].float(),
        rtol=0,
        atol=0,
    )

    # Shared expert: gate then up, stacked along the output dim.
    shared = model.model.layers[1].ffn.shared_experts
    half = shared.gate_up_proj.weight.shape[0] // 2
    torch.testing.assert_close(
        shared.gate_up_proj.weight[:half].float().cpu(),
        weights["layers.1.ffn.shared_experts.w1.weight"].float(),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        shared.gate_up_proj.weight[half:].float().cpu(),
        weights["layers.1.ffn.shared_experts.w3.weight"].float(),
        rtol=0,
        atol=0,
    )

    # Routed experts: per-expert w1/w3 stacked into one w13 row.
    experts = model.model.layers[1].ffn.experts
    per_expert = config.moe_intermediate_size
    torch.testing.assert_close(
        experts.w13_weight[2, :per_expert].cpu(),
        weights["layers.1.ffn.experts.2.w1.weight"],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        experts.w13_weight[2, per_expert:].cpu(),
        weights["layers.1.ffn.experts.2.w3.weight"],
        rtol=0,
        atol=0,
    )

    # Vocab-parallel head, and the padded tail left at zero.
    head = model.model.head
    torch.testing.assert_close(
        head.weight[: config.vocab_size].cpu(),
        weights["head.weight"].float(),
        rtol=0,
        atol=0,
    )
    assert torch.all(head.weight[config.vocab_size :] == 0)

    # Per-head attention sinks and the FP32-promoted mHC coefficients.
    torch.testing.assert_close(
        model.model.layers[1].attn.attn_sink.cpu(),
        weights["layers.1.attn.attn_sink"].float(),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        model.model.layers[1].hc_attn_fn.cpu(),
        weights["layers.1.hc_attn_fn"].float(),
        rtol=0,
        atol=0,
    )
