from copy import deepcopy

import torch
from torch import nn

from gllm.dist_utils import (
    get_pp_layers,
    get_tp_size,
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
from gllm.layers.rotary_embedding import RotaryEmbedding
from gllm.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from gllm.modules.attention import Attention

from .weight_loader import (
    LoadContext,
    WeightRule,
    contains,
    h_chatglm_gate_up,
    h_chatglm_qkv,
    h_proj_dim0,
    h_proj_dim1,
    run_weight_loader,
)


class GLMAttention(Attention):
    def __init__(self, layer_id: int, config):
        total_num_kv_heads = (
            config.multi_query_group_num
            if config.multi_query_attention
            else config.num_attention_heads
        )
        super().__init__(
            config.num_attention_heads, total_num_kv_heads, config.hidden_size
        )

        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            self.head_dim // 2,
            config.seq_length,
            getattr(config, "rope_theta", 10000),
            False,
        )
        self.attn = FlashAttention(
            layer_id, self.scaling, self.num_heads, self.num_kv_heads, self.head_dim
        )

        self.projection_size = config.kv_channels * self.num_heads
        self.qkv_hidden_size = (
            self.projection_size + 2 * self.head_dim * config.multi_query_group_num
        )

        self.query_key_value = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=config.add_bias_linear or config.add_qkv_bias,
        )

        self.dense = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=config.add_bias_linear,
        )

    def forward(self, input_data: InputData, hidden_states: torch.Tensor):
        qkv = self.query_key_value(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(input_data.get_position(), q, k)
        attn_output = self.attn.forward(q, k, v, input_data)
        output = self.dense(attn_output)
        return output


class GLMMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.add_bias = config.add_bias_linear

        self.dense_h_to_4h = MergedColumnParallelLinear(
            config.hidden_size, [config.ffn_hidden_size] * 2, bias=self.add_bias
        )

        self.activation_func = SiluAndMul()

        self.dense_4h_to_h = RowParallelLinear(
            config.ffn_hidden_size, config.hidden_size, bias=self.add_bias
        )

    def forward(self, hidden_states):
        # [s, b, 4hp]
        intermediate_parallel = self.dense_h_to_4h(hidden_states)
        intermediate_parallel = self.activation_func(intermediate_parallel)
        # [s, b, h]
        output = self.dense_4h_to_h(intermediate_parallel)
        return output


class GLMBlock(nn.Module):
    def __init__(self, layer_id, config):
        super().__init__()
        self.apply_residual_connection_post_layernorm = (
            config.apply_residual_connection_post_layernorm
        )
        self.fp32_residual_connection = config.fp32_residual_connection

        assert config.rmsnorm
        self.input_layernorm = RMSNorm(config.hidden_size, config.layernorm_epsilon)

        self.self_attention = GLMAttention(layer_id, config)
        self.hidden_dropout = config.hidden_dropout

        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, config.layernorm_epsilon
        )

        self.mlp = GLMMLP(config)

    def forward(self, hidden_states: torch.Tensor, input_data: InputData):
        # hidden_states: [num_tokens, h]
        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)
        # Self attention.
        attention_output = self.self_attention(input_data, layernorm_output)

        # Residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        layernorm_input = residual + attention_output

        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)

        # Second residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        output = self.mlp(layernorm_output) + residual

        return output


class GLMTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # assume post_layer_norm is true
        self.post_layer_norm = True

        self.start_layer, self.end_layer = get_pp_layers(config.num_layers)
        self.layers = nn.ModuleList(
            [
                GLMBlock(i - self.start_layer, config)
                for i in range(self.start_layer, self.end_layer)
            ]
        )

        if is_last_pp_rank() and self.post_layer_norm:
            assert config.rmsnorm
            layer_norm_func = RMSNorm
            self.final_layernorm = layer_norm_func(
                config.hidden_size, config.layernorm_epsilon
            )

    def forward(self, input_data: InputData, hidden_states: torch.Tensor):
        for layer in self.layers:
            hidden_states = layer(hidden_states, input_data)
        # Final layer norm.
        if is_last_pp_rank():
            if self.post_layer_norm:
                hidden_states = self.final_layernorm(hidden_states)

        return hidden_states


class ChatGLMModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embedding = VocabParallelEmbedding(
            config.padded_vocab_size,
            config.hidden_size,
        )

        self.multi_query_group_num = config.multi_query_group_num
        self.kv_channels = config.kv_channels

        self.encoder = GLMTransformer(config)
        self.output_layer = ParallelLMHead(
            config.padded_vocab_size,
            config.hidden_size,
            bias=False,
        )

    def forward(self, input_data: InputData, hidden_states=None):
        if is_first_pp_rank() and hidden_states is None:
            hidden_states = self.embedding(input_data.get_tokens())

        # Run encoder.
        hidden_states = self.encoder(input_data, hidden_states)
        return hidden_states


class ChatGLMForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.max_model_len = config.seq_length
        self.num_kv_heads = config.multi_query_group_num
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.transformer = ChatGLMModel(config)
        self.num_layers = len(self.transformer.encoder.layers)
        self.ret_residual = False
        if is_last_pp_rank():
            self.lm_head = self.transformer.output_layer

    def forward(
        self,
        input_data: InputData,
        hidden_states=None,
        residual=None,
        input_embeds=None,
    ):
        if input_embeds is not None:
            assert hidden_states is None and residual is None
            hidden_states = input_embeds
        return self.transformer(input_data, hidden_states)

    def compute_logits(self, input_data: InputData, hidden_states: torch.Tensor):
        # fetch hidden_states of last token in each seq
        idx_list = input_data.get_query_start_loc() - 1
        return self.lm_head(hidden_states[idx_list[1:]])

    def _chatglm_rules(self):
        return [
            WeightRule(
                contains("dense_h_to_4h.weight"), h_chatglm_gate_up, "h_to_4h"
            ),
            WeightRule(contains("dense_4h_to_h.weight"), h_proj_dim1, "4h_to_h"),
            WeightRule(contains("query_key_value"), h_chatglm_qkv, "qkv"),
            WeightRule(contains("dense.weight"), h_proj_dim1, "dense"),
            WeightRule(
                contains("embedding", "output_layer"), h_proj_dim0, "embed_out"
            ),
        ]

    def load_weights(self, weights, mp_load_progress=None):
        attn = self.transformer.encoder.layers[0].self_attention
        q_index = attn.num_heads * attn.head_dim * get_tp_size()
        k_index = (attn.num_heads + attn.num_kv_heads) * attn.head_dim * get_tp_size()
        ctx = LoadContext(
            weights=weights,
            num_heads=attn.num_heads,
            num_kv_heads=attn.num_kv_heads,
            head_dim=attn.head_dim,
            extra={
                "q_index": q_index,
                "k_index": k_index,
                "intermediate_size": self.config.ffn_hidden_size,
            },
        )
        # ChatGLM names its embedding ``embedding.weight`` in the module tree
        # but ``embedding.word_embeddings.weight`` in the checkpoint.
        run_weight_loader(
            self,
            weights,
            self._chatglm_rules(),
            mp_load_progress,
            pp_idx_offset=3,
            start_layer=self.transformer.encoder.start_layer,
            ctx=ctx,
            src_key_fn=lambda k: (
                k.replace("embedding", "embedding.word_embeddings")
                if "embedding" in k
                else k
            ),
        )

    def process_response(self, output, history):
        content = ""
        history = deepcopy(history)
        for response in output.split("<|assistant|>"):
            if "\n" in response:
                metadata, content = response.split("\n", maxsplit=1)
            else:
                metadata, content = "", response
            if not metadata.strip():
                content = content.strip()
                history.append(
                    {"role": "assistant", "metadata": metadata, "content": content}
                )
                content = content.replace("[[训练时间]]", "2023年")
            else:
                history.append(
                    {"role": "assistant", "metadata": metadata, "content": content}
                )
                if history[0]["role"] == "system" and "tools" in history[0]:
                    content = "\n".join(content.split("\n")[1:-1])
                    parameters = eval(content)
                    content = {"name": metadata.strip(), "parameters": parameters}
                else:
                    content = {"name": metadata.strip(), "content": content}
        return content, history
