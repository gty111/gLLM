from typing import Optional

import torch
from torch import nn

from gllm.dist_utils import (
    get_pp_layers,
    is_first_pp_rank,
    is_last_pp_rank,
)
from gllm.input_data import InputData
from gllm.layers.activation import SiluAndMul
from gllm.layers.attention import FlashAttention
from gllm.layers.layernorm import RMSNorm
from gllm.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from gllm.layers.rotary_embedding import MRotaryEmbedding, RotaryEmbedding
from gllm.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from gllm.modules.attention import Attention

from .utils import extract_rope_config
from .weight_loader import (
    LoadContext,
    WeightRule,
    contains,
    h_gate_up,
    h_proj_dim0,
    h_proj_dim1,
    h_qkv_proj,
    run_weight_loader,
)


class Qwen2MLP(nn.Module):

    def __init__(self, config, shared_expert=False):
        super().__init__()
        quant_config = getattr(config, "quantization_config", None)
        if not shared_expert:
            intermediate_size = config.intermediate_size
        else:
            intermediate_size = config.shared_expert_intermediate_size
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
        )

        self.down_proj = RowParallelLinear(
            intermediate_size, config.hidden_size, bias=False, quant_config=quant_config
        )

        self.act_fn = SiluAndMul()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_up_proj(x)))


class Qwen2Attention(Attention):
    def __init__(self, layer_id: int, config, qkv_bias=True):
        super().__init__(
            config.num_attention_heads, config.num_key_value_heads, config.hidden_size
        )

        self.rope_theta, _rope_scaling_normalized = extract_rope_config(
            config, default_theta=10000.0
        )
        self.max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        quant_config = getattr(config, "quantization_config", None)

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
            quant_config=quant_config,
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
        )
        rope_scaling = _rope_scaling_normalized
        if rope_scaling is None:
            self.rotary_emb = RotaryEmbedding(
                self.head_dim,
                self.head_dim,
                self.max_position_embeddings,
                self.rope_theta,
                True,
            )
        else:
            assert "mrope_section" in rope_scaling
            self.rotary_emb = MRotaryEmbedding(
                self.head_dim,
                self.head_dim,
                self.max_position_embeddings,
                self.rope_theta,
                True,
                rope_scaling["mrope_section"],
            )
        self.attn = FlashAttention(
            layer_id, self.scaling, self.num_heads, self.num_kv_heads, self.head_dim
        )

    def forward(self, input_data: InputData, hidden_states: torch.Tensor):
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(input_data.get_position(), q, k)
        attn_output = self.attn.forward(q, k, v, input_data)
        output = self.o_proj(attn_output)
        return output


class Qwen2DecoderLayer(nn.Module):
    def __init__(
        self, layer_id: int, config, attention_type=Qwen2Attention, mlp_type=Qwen2MLP
    ):
        super().__init__()
        self.self_attn = attention_type(layer_id, config)
        self.mlp = mlp_type(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        input_data: InputData,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(input_data, hidden_states)

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen2Model(nn.Module):
    def __init__(self, config, decoder_layer_type=Qwen2DecoderLayer):
        super().__init__()
        if is_first_pp_rank() or (config.tie_word_embeddings and is_last_pp_rank()):
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
            )
        self.start_layer, self.end_layer = get_pp_layers(config.num_hidden_layers)

        self.layers = nn.ModuleList(
            [
                decoder_layer_type(i - self.start_layer, config)
                for i in range(self.start_layer, self.end_layer)
            ]
        )
        if is_last_pp_rank():
            self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, input_data: InputData, hidden_states=None, residual=None):
        if is_first_pp_rank() and hidden_states is None:
            hidden_states = self.embed_tokens(input_data.get_tokens())
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual = layer(input_data, hidden_states, residual)
        if is_last_pp_rank():
            hidden_states, _ = self.norm(hidden_states, residual)
            return hidden_states
        else:
            return hidden_states, residual
        
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)


class Qwen2ForCausalLM(nn.Module):
    def __init__(self, config, model_type=Qwen2Model):
        super().__init__()
        self.config = config
        self.max_model_len = config.max_position_embeddings
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.model = model_type(config)
        self.num_layers = len(self.model.layers)
        self.start_layer = self.model.start_layer
        self.end_layer = self.model.end_layer
        self.ret_residual = True
        if is_last_pp_rank():
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
            )
            if config.tie_word_embeddings:
                self.lm_head.tie_weights(self.model.embed_tokens)

    def forward(self, input_data: InputData, hidden_states=None, residual=None):
        return self.model(input_data, hidden_states, residual)

    def compute_logits(self, input_data: InputData, hidden_states: torch.Tensor):
        # fetch hidden_states of last token in each seq
        idx_list = input_data.get_query_start_loc() - 1
        return self.logits_from_hidden(hidden_states[idx_list[1:]])

    def logits_from_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project the given hidden states to full-vocab logits.

        ``compute_logits`` gathers only each seq's last position (for
        sampling); this projects *every* supplied position and is used by the
        prompt-logprobs path. Keeping it here means LM-head placement (tied
        weights, TP gather, multimodal nesting) stays a model-internal detail.
        """
        return self.lm_head(hidden_states)

    def weight_rules(self):
        """Ordered ``(match, handler)`` table for this rank's parameters.

        Subclasses compose ``super().weight_rules() + [...]`` (or override) to
        add architecture-specific rules. First match wins, mirroring the old
        ``if/elif`` priority.
        """
        return [
            WeightRule(contains("self_attn.qkv_proj"), h_qkv_proj, "qkv_proj"),
            WeightRule(contains("self_attn.o_proj"), h_proj_dim1, "o_proj"),
            WeightRule(contains("gate_up_proj"), h_gate_up, "gate_up_proj"),
            WeightRule(contains("down_proj"), h_proj_dim1, "down_proj"),
            WeightRule(
                contains("embed_tokens", "lm_head"), h_proj_dim0, "embed_lm_head"
            ),
        ]

    def _make_load_context(self, weights):
        """Build the :class:`LoadContext` shared with every handler.

        Reads attention head geometry off layer 0. Dense models leave the MoE
        fields (``expert_map`` / ``num_experts`` / ``pool``) at their ``None``
        defaults; MoE subclasses populate them in their override.
        """
        attn = self.model.layers[0].self_attn
        return LoadContext(
            weights=weights,
            num_heads=attn.num_heads,
            num_kv_heads=attn.num_kv_heads,
            head_dim=attn.head_dim,
        )

    def load_weights(self, weights, mp_load_progress=None):
        run_weight_loader(
            self,
            weights,
            self.weight_rules(),
            mp_load_progress,
            pp_idx_offset=2,
            start_layer=self.model.start_layer,
            ctx=self._make_load_context(weights),
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)
