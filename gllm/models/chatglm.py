import torch

from torch import nn
from copy import deepcopy

from gllm.layers.linear import MergedColumnParallelLinear, RowParallelLinear, QKVParallelLinear
from gllm.layers.vocab_parallel_embedding import VocabParallelEmbedding, ParallelLMHead
from gllm.input_data import InputData
from gllm.layers.rotary_embedding import RotaryEmbedding
from gllm.layers.attention import FlashAttention
from gllm.layers.activation import SiluAndMul
from gllm.layers.layernorm import RMSNorm
from gllm.dist_utils import (get_pp_layers, get_pp_rank, get_local_rank, is_last_pp_rank, 
                             resolve_pp_layer_idx, get_tp_size)
from gllm.utils import get_model_load_pbar
from gllm.modules.attention import Attention

from .weight_utils import (copy_qkv_proj_weight, copy_qkv_proj_bias, 
                           copy_gate_up_proj_weight, copy_single_proj_col,
                           copy_single_proj_row)


class GLMAttention(Attention):
    def __init__(self, layer_id: int, config):
        total_num_kv_heads = (config.multi_query_group_num 
                              if config.multi_query_attention else 
                              config.num_attention_heads)
        super().__init__(config.num_attention_heads,
                         total_num_kv_heads,
                         config.hidden_size)

        self.rotary_emb = RotaryEmbedding(
            self.head_dim, self.head_dim // 2, config.seq_length, getattr(config,'rope_theta',10000), False)
        self.attn = FlashAttention(
            layer_id, self.scaling, self.num_heads, self.num_kv_heads, self.head_dim, self.hidden_size)

        self.projection_size = config.kv_channels * self.num_heads
        self.qkv_hidden_size = self.projection_size + 2 * \
            self.head_dim * config.multi_query_group_num
            
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
        q, k = self.rotary_emb(input_data.positions, q, k)
        attn_output = self.attn.forward(q, k, v, input_data)
        output = self.dense(attn_output)
        return output


class GLMMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.add_bias = config.add_bias_linear
        
        self.dense_h_to_4h = MergedColumnParallelLinear(
            config.hidden_size,
            [config.ffn_hidden_size] * 2,
            bias = self.add_bias
        )
        
        self.activation_func = SiluAndMul()
        
        self.dense_4h_to_h = RowParallelLinear(
            config.ffn_hidden_size,
            config.hidden_size,
            bias = self.add_bias
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
        self.apply_residual_connection_post_layernorm = config.apply_residual_connection_post_layernorm
        self.fp32_residual_connection = config.fp32_residual_connection

        assert config.rmsnorm
        self.input_layernorm = RMSNorm(
            config.hidden_size, config.layernorm_epsilon)

        self.self_attention = GLMAttention(layer_id, config)
        self.hidden_dropout = config.hidden_dropout

        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, config.layernorm_epsilon)

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
            [GLMBlock(i-self.start_layer, config) for i in range(self.start_layer, self.end_layer)])

        if is_last_pp_rank() and self.post_layer_norm:
                assert config.rmsnorm
                layer_norm_func = RMSNorm
                self.final_layernorm = layer_norm_func(
                    config.hidden_size, config.layernorm_epsilon)

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
        if get_pp_rank() == 0:
            hidden_states = self.embedding(input_data.tokens)

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

    def forward(self, input_data: InputData, hidden_states=None, residual=None):
        return self.transformer(input_data, hidden_states)

    def compute_logits(self, input_data: InputData, hidden_states: torch.Tensor):
        # fetch hidden_states of last token in each seq
        idx_list = input_data.query_start_loc - 1
        return self.lm_head(hidden_states[idx_list[1:]])

    def load_weights(self, weights, mp_load_progress):
        parameters = dict(self.named_parameters())
        if mp_load_progress is not None:
            mp_load_progress[get_local_rank()*2] = len(parameters)
            mp_load_progress[get_local_rank()*2+1] = 0
        else:
            pbar = get_model_load_pbar(len(parameters))
            
        attn = self.transformer.encoder.layers[0].self_attention
        num_heads = attn.num_heads
        num_kv_heads = attn.num_kv_heads
        head_dim = attn.head_dim
        
        intermediate_size = self.config.ffn_hidden_size
        intermediate_size_partition = intermediate_size // get_tp_size()
        vocab_size_partition = self.config.padded_vocab_size // get_tp_size()
        
        q_index = num_heads*head_dim*get_tp_size()
        k_index = (num_heads+num_kv_heads)*head_dim*get_tp_size()
        
        for k, v in parameters.items():
            k = resolve_pp_layer_idx(k, 3, self.transformer.encoder.start_layer)
            if 'embedding' in k:
                k = k.replace('embedding', 'embedding.word_embeddings')
            
            weight = weights[k]
            if 'dense_h_to_4h.weight' in k:
                copy_gate_up_proj_weight(v.data, weight[:intermediate_size], weight[intermediate_size:], intermediate_size_partition)
            elif 'dense_4h_to_h.weight' in k:
                copy_single_proj_col(v.data, weight, intermediate_size_partition)
            elif 'query_key_value.weight' in k:
                copy_qkv_proj_weight(v.data, weight[:q_index], weight[q_index:k_index], weight[k_index:], num_heads, num_kv_heads, head_dim)
            elif 'query_key_value.bias' in k:
                copy_qkv_proj_bias(v.data, weight[:q_index], weight[q_index:k_index], weight[k_index:], num_heads, num_kv_heads, head_dim)
            elif 'dense.weight' in k:
                copy_single_proj_col(v.data, weight, num_heads*head_dim)
            elif 'embedding' in k or 'output_layer' in k:
                copy_single_proj_row(v.data, weight, vocab_size_partition)
            else:
                v.data.copy_(weight)
            if mp_load_progress is not None:
                mp_load_progress[get_local_rank()*2+1] += 1
            else:
                pbar.update(1)

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
                    {"role": "assistant", "metadata": metadata, "content": content})
                content = content.replace("[[训练时间]]", "2023年")
            else:
                history.append(
                    {"role": "assistant", "metadata": metadata, "content": content})
                if history[0]["role"] == "system" and "tools" in history[0]:
                    content = "\n".join(content.split("\n")[1:-1])
                    parameters = eval(content)
                    content = {"name": metadata.strip(),
                               "parameters": parameters}
                else:
                    content = {"name": metadata.strip(), "content": content}
        return content, history
