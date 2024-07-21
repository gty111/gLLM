import json
import glob
from safetensors import safe_open
from tqdm import tqdm

from gllm.llama3 import LlamaForCausalLM


class ModelLoader():
    def __init__(self, model_path):
        self.model_path = model_path

    def load_weights(self):
        weights = {}
        weights_path = glob.glob(f"{self.model_path}/*.safetensors")
        for weight_path in weights_path:
            with safe_open(weight_path, framework="pt", device="cpu") as f:
                for k in f.keys():
                    weights[k] = f.get_tensor(k)

        return weights

    def load_config(self):
        config_path = f"{self.model_path}/config.json"
        with open(config_path, 'r') as f:
            model_config = json.load(f)
        return model_config

    def load_model(self):
        # print(args.model)
        weights = self.load_weights()
        model_config = self.load_config()
        llama3 = LlamaForCausalLM(model_config)

        parameters = dict(llama3.named_parameters())

        # assert len(parameters) == len(weights)
        num_attn_heads = model_config['num_attention_heads']
        head_dim = model_config['hidden_size'] // num_attn_heads
        num_kv_heads = model_config['num_key_value_heads']
        intermediate_size = model_config['intermediate_size']
        for k, v in tqdm(parameters.items()):
            if k.find('self_attn.qkv_proj') != -1:
                v.data[:num_attn_heads*head_dim, :] = weights[k.replace(
                    'qkv_proj', 'q_proj')]
                v.data[num_attn_heads*head_dim:(num_attn_heads +
                       num_kv_heads)*head_dim, :] = weights[k.replace('qkv_proj', 'k_proj')]
                v.data[(num_attn_heads +
                       num_kv_heads)*head_dim:, :] = weights[k.replace('qkv_proj', 'v_proj')]
            elif k.find('gate_up_proj') != -1:
                v.data[:intermediate_size, :] = weights[k.replace(
                    'gate_up_proj', 'gate_proj')]
                v.data[intermediate_size:, :] = weights[k.replace(
                    'gate_up_proj', 'up_proj')]
            else:
                v.data.copy_(weights[k])
            # v.data.copy_(weights[k])

        return llama3
