import torch

from torch import nn
from copy import deepcopy

from gllm.input_data import InputData
from gllm.layers.rotary_embedding import RotaryEmbedding
from gllm.layers.attention import FlashAttention
from gllm.layers.activation import SiluAndMul
from gllm.layers.layernorm import RMSNorm
from gllm.layers.sampler import Sampler
from gllm.dist_utils import get_pp_layers, get_pp_rank, get_pp_size, get_local_rank
from gllm.utils import get_model_load_pbar


class GLMAttention(nn.Module):
    def __init__(self, layer_id: int, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.multi_query_group_num
        self.head_dim = self.hidden_size // self.num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.rotary_emb = RotaryEmbedding(
            self.head_dim, self.head_dim // 2, config.seq_length, getattr(config,'rope_theta',10000), False)
        self.attn = FlashAttention(
            layer_id, self.scaling, self.num_heads, self.num_kv_heads, self.head_dim, self.hidden_size)

        self.projection_size = config.kv_channels * self.num_heads
        self.qkv_hidden_size = self.projection_size + 2 * \
            self.head_dim * config.multi_query_group_num
        self.query_key_value = nn.Linear(self.hidden_size, self.qkv_hidden_size,
                                         bias=config.add_bias_linear or config.add_qkv_bias)
        self.dense = nn.Linear(self.projection_size, self.hidden_size,
                               bias=config.add_bias_linear)

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
        self.dense_h_to_4h = nn.Linear(
            config.hidden_size, config.ffn_hidden_size*2, bias=self.add_bias)
        self.activation_func = SiluAndMul()
        self.dense_4h_to_h = nn.Linear(
            config.ffn_hidden_size, config.hidden_size, bias=self.add_bias)

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

        if get_pp_rank() == get_pp_size() - 1:
            if self.post_layer_norm:
                assert config.rmsnorm
                layer_norm_func = RMSNorm
                self.final_layernorm = layer_norm_func(
                    config.hidden_size, config.layernorm_epsilon)

    def forward(self, input_data: InputData, hidden_states: torch.Tensor):
        for layer in self.layers:
            hidden_states = layer(hidden_states, input_data)
        # Final layer norm.
        if get_pp_rank() == get_pp_size() - 1:
            if self.post_layer_norm:
                hidden_states = self.final_layernorm(hidden_states)

        return hidden_states


class ChatGLMModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embedding = nn.Embedding(
            config.padded_vocab_size, config.hidden_size)

        self.multi_query_group_num = config.multi_query_group_num
        self.kv_channels = config.kv_channels

        self.encoder = GLMTransformer(config)
        self.output_layer = nn.Linear(
            config.hidden_size, config.padded_vocab_size, bias=False)

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
        if get_pp_rank() == get_pp_size() - 1:
            self.lm_head = self.transformer.output_layer
        self.sampler = Sampler()

    def forward(self, input_data: InputData, hidden_states=None, residual=None):
        return self.transformer(input_data, hidden_states)

    def compute_logits(self, input_data: InputData, hidden_states: torch.Tensor):
        # fetch hidden_states of last token in each seq
        idx_list = input_data.query_start_loc - 1
        return self.lm_head(hidden_states[idx_list[1:]])

    def sample(self, input_data: InputData, logits: torch.Tensor):
        return self.sampler.forward(logits, input_data)

    def load_weights(self, weights, mp_load_progress):
        parameters = dict(self.named_parameters())
        if mp_load_progress is not None:
            mp_load_progress[get_local_rank()*2] = len(parameters)
            mp_load_progress[get_local_rank()*2+1] = 0
        else:
            pbar = get_model_load_pbar(len(parameters))
        
        for k, v in parameters.items():
            # resolve PP layer
            if 'layers' in k:
                k_list = k.split('.')
                k_list[3] = str(int(k_list[3])+self.transformer.encoder.start_layer)
                k = '.'.join(k_list)
            if 'embedding' in k:
                k = k.replace('embedding', 'embedding.word_embeddings')
            v.data.copy_(weights[k])
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
