from gllm.dist_utils import is_first_pp_rank, is_last_pp_rank
from gllm.input_data import InputData

from .qwen3_moe import Qwen3MoeForCausalLM, Qwen3MoeModel
from .qwen3_vl import Qwen3VLForConditionalGeneration
from .weight_loader import (
    WeightRule,
    contains,
    make_w13_stacked_loader,
    make_w2_stacked_loader,
)


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

    def expert_weight_rules(self):
        # Qwen3-VL-MoE pre-stacks experts: ``gate_up_proj`` (E, H, 2I) and
        # ``down_proj`` (E, I, H) single tensors instead of per-expert keys.
        return [
            WeightRule(
                contains("w13_weight"),
                make_w13_stacked_loader("w13_weight", "gate_up_proj"),
                "w13_stacked",
            ),
            WeightRule(
                contains("w2_weight"),
                make_w2_stacked_loader("w2_weight", "down_proj"),
                "w2_stacked",
            ),
        ]



class Qwen3VLMoeForConditionalGeneration(Qwen3VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config, language_model_type=Qwen3MoeLLMForCausalLM)
