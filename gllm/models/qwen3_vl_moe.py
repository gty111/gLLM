from gllm.dist_utils import get_ep_rank, get_ep_size, get_local_rank, is_first_pp_rank, is_last_pp_rank, resolve_pp_layer_idx
from gllm.input_data import InputData
from gllm.layers.moe.fused_moe_triton.layer import determine_expert_map
from gllm.models.weight_utils import (
    copy_gate_up_proj,
    copy_qkv_proj,
    copy_single_proj_dim0,
    copy_single_proj_dim1,
    get_tensor_from_dict,
    load_fused_w13_stacked,
    load_w2_stacked,
    moe_expert_load_pool,
)
from gllm.utils import get_model_load_pbar

from .qwen3_moe import Qwen3MoeForCausalLM, Qwen3MoeModel
from .qwen3_vl import Qwen3VLForConditionalGeneration


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

        with moe_expert_load_pool(num_experts) as expert_pool:
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
                    # checkpoint shape: (E, H, 2I) — stacked across experts.
                    # internal shape:   (local_E, 2 * I_p, H).
                    load_fused_w13_stacked(
                        v.data,
                        get_tensor_from_dict(
                            weights, k.replace("w13_weight", "gate_up_proj")
                        ),
                        expert_map=expert_map,
                        num_experts=num_experts,
                        pool=expert_pool,
                    )
                elif k.find("w2_weight") != -1:  # expert
                    # checkpoint shape: (E, I, H) — stacked across experts.
                    # internal shape:   (local_E, H, I_p).
                    load_w2_stacked(
                        v.data,
                        get_tensor_from_dict(
                            weights, k.replace("w2_weight", "down_proj")
                        ),
                        expert_map=expert_map,
                        num_experts=num_experts,
                        pool=expert_pool,
                    )
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
        super().__init__(config, language_model_type=Qwen3MoeLLMForCausalLM)
