"""Model architecture registry.

Single source of truth mapping HuggingFace ``config.architectures`` names to
the gLLM model class plus that architecture's capability flags. Adding a new
model means adding one entry here instead of editing the scattered
architecture-name branches in ``gllm.runtime.model_loader``.

Capability flags (an absent key means False):

- ``mla``: the model uses MLA attention (DeepSeek family / Kimi-K2.5), which
  switches the KV cache to the MLA layout (``ModelLoader.use_mla``).
- ``mm``: the model is multimodal and consumes multimodal inputs
  (``ModelLoader.use_mm``).
- ``hybrid``: the model has linear-attention (Mamba/GDN) layers that need a
  recurrent-state cache in addition to the regular KV cache
  (``ModelLoader.use_hybrid_state``).
- ``normalize_kimi_quant``: ``ModelLoader.load_config`` must translate this
  checkpoint family's compressed-tensors config into gLLM's int4-MoE hint
  (``ModelLoader._normalize_kimi_quant_config``).
"""

from typing import Dict, Tuple

from gllm.models.chatglm import ChatGLMForCausalLM
from gllm.models.deepseek_v2 import DeepseekV2ForCausalLM
from gllm.models.deepseek_v32 import DeepseekV32ForCausalLM
from gllm.models.deepseek_v4 import DeepseekV4ForCausalLM
from gllm.models.kimi_k25 import KimiK25ForConditionalGeneration
from gllm.models.llama import LlamaForCausalLM
from gllm.models.mixtral import MixtralForCausalLM
from gllm.models.qwen2 import Qwen2ForCausalLM
from gllm.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from gllm.models.qwen2_moe import Qwen2MoeForCausalLM
from gllm.models.qwen3 import Qwen3ForCausalLM
from gllm.models.qwen3_5 import Qwen3_5ForConditionalGeneration
from gllm.models.qwen3_5_moe import Qwen3_5MoeForConditionalGeneration
from gllm.models.qwen3_moe import Qwen3MoeForCausalLM
from gllm.models.qwen3_vl import Qwen3VLForConditionalGeneration
from gllm.models.qwen3_vl_moe import Qwen3VLMoeForConditionalGeneration

MODEL_ARCH_REGISTRY: Dict[str, Tuple[type, Dict[str, bool]]] = {
    "LlamaForCausalLM": (LlamaForCausalLM, {}),
    "ChatGLMModel": (ChatGLMForCausalLM, {}),
    "Qwen2ForCausalLM": (Qwen2ForCausalLM, {}),
    "Qwen3ForCausalLM": (Qwen3ForCausalLM, {}),
    "Qwen2MoeForCausalLM": (Qwen2MoeForCausalLM, {}),
    "Qwen3MoeForCausalLM": (Qwen3MoeForCausalLM, {}),
    "MixtralForCausalLM": (MixtralForCausalLM, {}),
    "DeepseekV2ForCausalLM": (DeepseekV2ForCausalLM, {"mla": True}),
    "DeepseekV3ForCausalLM": (DeepseekV2ForCausalLM, {"mla": True}),
    "DeepseekV32ForCausalLM": (DeepseekV32ForCausalLM, {"mla": True}),
    "DeepseekV4ForCausalLM": (DeepseekV4ForCausalLM, {"mla": True}),
    "Qwen2_5_VLForConditionalGeneration": (Qwen2_5_VLForConditionalGeneration, {"mm": True}),
    "Qwen3VLForConditionalGeneration": (Qwen3VLForConditionalGeneration, {"mm": True}),
    "Qwen3VLMoeForConditionalGeneration": (Qwen3VLMoeForConditionalGeneration, {"mm": True}),
    "Qwen3_5ForConditionalGeneration": (Qwen3_5ForConditionalGeneration, {"mm": True, "hybrid": True}),
    "Qwen3_5MoeForConditionalGeneration": (Qwen3_5MoeForConditionalGeneration, {"mm": True, "hybrid": True}),
    "KimiK25ForConditionalGeneration": (
        KimiK25ForConditionalGeneration,
        {"mla": True, "mm": True, "normalize_kimi_quant": True},
    ),
}
