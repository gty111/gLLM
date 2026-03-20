import torch
import torch.nn as nn

from gllm.dist_utils import get_ep_rank, get_ep_size, get_local_rank, get_tp_size, is_first_pp_rank, is_last_pp_rank, is_use_ep, resolve_ep_expert_idx, resolve_pp_layer_idx
from gllm.input_data import InputData
from gllm.layers.moe.fused_moe_triton.layer import determine_expert_map
from gllm.models.weight_utils import copy_gate_up_proj, copy_qkv_proj, copy_single_proj_dim0, copy_single_proj_dim1, get_tensor_from_dict
from gllm.utils import get_model_load_pbar

from .qwen3_moe import Qwen3MoeForCausalLM, Qwen3MoeModel
from .qwen3_vl import Qwen3VLForConditionalGeneration, Qwen3_VisionTransformer


class Qwen3MoeLLMModel(Qwen3MoeModel):
    def forward(
        self,
        input_data: InputData,
        hidden_states=None,
        residual=None,
        deepstack_input_embeds=None,
    ):
        if is_first_pp_rank() and hidden_states is None:
            hidden_states = self.embed_tokens(input_data.get_tokens())

        for local_layer_idx, layer in enumerate(self.layers):
            layer_idx = local_layer_idx + self.start_layer

            hidden_states, residual = layer(
                input_data,
                hidden_states,
                residual,
            )

            if deepstack_input_embeds is not None and layer_idx in range(
                0, len(deepstack_input_embeds)
            ):
                hidden_states = (
                    hidden_states
                    + deepstack_input_embeds[f"deepstack_input_embeds_{layer_idx}"]
                )

        if not is_last_pp_rank():
            return hidden_states, residual

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3MoeLLMForCausalLM(Qwen3MoeForCausalLM):
    def __init__(self, config):
        super().__init__(config, model_type=Qwen3MoeLLMModel)
        
    def load_weights(self, weights, mp_load_progress=None):
        parameters = dict(self.named_parameters())
        if mp_load_progress is not None:
            mp_load_progress[get_local_rank() * 2] = len(parameters)
            mp_load_progress[get_local_rank() * 2 + 1] = 0
        else:
            pbar = get_model_load_pbar(len(parameters))

        attn = self.model.layers[0].self_attn
        num_heads = attn.num_heads
        num_kv_heads = attn.num_kv_heads
        head_dim = attn.head_dim

        num_experts = getattr(
            self.config, "num_experts", getattr(self.config, "n_routed_experts", None)
        )
        assert num_experts is not None

        _, expert_map = determine_expert_map(get_ep_size(), get_ep_rank(), num_experts)

        for k, v in parameters.items():
            k = resolve_pp_layer_idx(k, 2, self.model.start_layer)
            if k.find("self_attn.qkv_proj") != -1:
                head_dim_patch = (
                    head_dim if k.find("scale") == -1 or k.find("weight") == -1 else 1
                )
                copy_qkv_proj(
                    v.data,
                    get_tensor_from_dict(weights, k.replace("qkv_proj", "q_proj")),
                    get_tensor_from_dict(weights, k.replace("qkv_proj", "k_proj")),
                    get_tensor_from_dict(weights, k.replace("qkv_proj", "v_proj")),
                    num_heads,
                    num_kv_heads,
                    head_dim_patch,
                )
            elif k.find("w13_weight") != -1:  # expert
                w13_weight = get_tensor_from_dict(weights, k.replace("w13_weight", "gate_up_proj"))
                for expert_idx in range(num_experts):
                    local_expert_idx = resolve_ep_expert_idx(expert_idx, expert_map)
                    if local_expert_idx == -1:
                        continue
                    assert is_use_ep() or get_tp_size() == 1
                    v.data[local_expert_idx].copy_(w13_weight[expert_idx].permute(1, 0))
            elif k.find("w2_weight") != -1:  # expert
                w2_weight = get_tensor_from_dict(weights, k.replace("w2_weight", "down_proj"))
                for expert_idx in range(num_experts):
                    local_expert_idx = resolve_ep_expert_idx(expert_idx, expert_map)
                    if local_expert_idx == -1:
                        continue
                    assert is_use_ep() or get_tp_size() == 1
                    v.data[local_expert_idx].copy_(w2_weight[expert_idx].permute(1, 0))
            elif k.find("gate_up_proj.weight") != -1:  # shared expert or dense layer
                copy_gate_up_proj(
                    v.data,
                    get_tensor_from_dict(weights, k.replace("gate_up_proj", "gate_proj")),
                    get_tensor_from_dict(weights, k.replace("gate_up_proj", "up_proj")),
                )
            elif k.find("down_proj.weight") != -1:  # shared expert or dense layer
                copy_single_proj_dim1(v.data, get_tensor_from_dict(weights, k))
            elif k.find("self_attn.o_proj") != -1:
                copy_single_proj_dim1(v.data, get_tensor_from_dict(weights, k))
            elif k.find("embed_tokens") != -1 or k.find("lm_head") != -1:
                copy_single_proj_dim0(v.data, get_tensor_from_dict(weights, k))
            else:
                v.data.copy_(get_tensor_from_dict(weights, k))
            if mp_load_progress is not None:
                mp_load_progress[get_local_rank() * 2 + 1] += 1
            else:
                pbar.update(1)


class Qwen3VLMoeForConditionalGeneration(Qwen3VLForConditionalGeneration):
    def __init__(self, config):
        nn.Module.__init__(self)
        quant_config = getattr(config, "quant_config", None)

        self.config = config

        self.use_deepstack = hasattr(config.vision_config, "deepstack_visual_indexes")
        self.deepstack_num_level = (
            len(config.vision_config.deepstack_visual_indexes)
            if self.use_deepstack
            else 0
        )
        self.visual_dim = config.vision_config.out_hidden_size
        self.multiscale_dim = self.visual_dim * self.deepstack_num_level

        self.visual = Qwen3_VisionTransformer(
            config.vision_config,
            norm_eps=getattr(config, "rms_norm_eps", 1e-6),
            quant_config=quant_config,
        )

        if self.use_deepstack:
            self.deepstack_input_embeds = [
                torch.zeros(
                    config.max_num_batched_tokens,
                    config.text_config.hidden_size,
                )
                for _ in range(self.deepstack_num_level)
            ]

        self.language_model = Qwen3MoeLLMForCausalLM(config.text_config)

        if not is_first_pp_rank() and hasattr(
            config.vision_config, "deepstack_visual_indexes"
        ):
            assert self.language_model.start_layer >= len(
                config.vision_config.deepstack_visual_indexes
            ), (
                "start_layer should be greater than or equal to "
                "len(deepstack_visual_indexes)"
            )

        self.num_layers = self.language_model.num_layers
        self.num_kv_heads = self.language_model.num_kv_heads
        self.head_dim = self.language_model.head_dim
        self.ret_residual = self.language_model.ret_residual
