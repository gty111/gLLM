"""Native vision weights, interpolation, image/video and MRoPE index keys."""

import json

import pytest
import torch
from safetensors.torch import save_file
from test_qwen4_exp_decoder import tiny_config, write_checkpoint
from transformers import AutoConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpVisionModel

from gllm.models.qwen4_exp import Qwen4ExpForConditionalGeneration
from gllm.runtime.model_loader import LazySafetensors


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("grid", [[[1, 4, 4]], [[2, 4, 6]]])
@torch.no_grad()
def test_native_vision_outputs(tmp_path, grid):
    raw = tiny_config()
    raw["vision_config"] = {
        "model_type": "qwen4_exp",
        "depth": 2,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_heads": 4,
        "out_hidden_size": 32,
        "num_position_embeddings": 64,
        "patch_size": 16,
        "temporal_patch_size": 2,
        "spatial_merge_size": 2,
        "in_channels": 3,
        "hidden_act": "gelu_pytorch_tanh",
        "deepstack_visual_indexes": [],
    }
    weights = write_checkpoint(tmp_path, raw)
    config = AutoConfig.from_pretrained(tmp_path)
    config.checkpoint_path = str(tmp_path)
    config.max_num_batched_tokens = 128
    config.mtp_enabled = False
    config.attention_backend = "flashinfer"
    config.vision_config._attn_implementation = "eager"
    torch.manual_seed(319)
    reference = (
        Qwen4ExpVisionModel(config.vision_config)
        .to(device="cuda", dtype=torch.bfloat16)
        .eval()
    )
    weights.update(
        {
            f"model.visual.{key}": value.cpu()
            for key, value in reference.state_dict().items()
        }
    )
    save_file(weights, str(tmp_path / "weights.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(weights, "weights.safetensors")})
    )
    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        model = Qwen4ExpForConditionalGeneration(config).eval()
        lazy = LazySafetensors(
            {key: (str(tmp_path / "weights.safetensors"), key) for key in weights}
        )
        model.load_weights(lazy)
        lazy.close()
    finally:
        torch.set_default_dtype(original_dtype)
    grid_tensor = torch.tensor(grid, device="cuda")
    pixels = torch.randn(
        int(grid_tensor.prod(-1).sum()),
        3 * 2 * 16 * 16,
        device="cuda",
        dtype=torch.bfloat16,
    )
    expected = reference(pixels, grid_thw=grid_tensor).pooler_output
    actual = model.visual(pixels, grid_tensor)
    torch.testing.assert_close(actual, expected, atol=0.006, rtol=0.04)
    assert model.index_head_dim == model.language_model.index_head_dim
    assert model.ssm_cache_config is model.language_model.ssm_cache_config
