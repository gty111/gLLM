import json
import glob
import torch
from safetensors import safe_open

from gllm.models.llama import LlamaForCausalLM
from gllm.models.chatglm import ChatGLMForCausalLM
from gllm.models.qwen2 import Qwen2ForCausalLM


class ModelLoader():
    def __init__(self, model_path):
        self.model_path = model_path

    def get_dtype(self, dtype: str):
        if dtype == 'float16':
            return torch.float16
        elif dtype == 'bfloat16':
            return torch.bfloat16
        else:
            assert 0

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
        model_config['torch_dtype'] = self.get_dtype(
            model_config['torch_dtype'])
        return model_config

    def load_model(self):
        # print(args.model)
        weights = self.load_weights()
        model_config = self.load_config()

        architecture = model_config['architectures'][0]
        if architecture == 'LlamaForCausalLM':
            if 'rope_theta' not in model_config:
                model_config['rope_theta'] = 10000
            model = LlamaForCausalLM(model_config) 
        elif architecture == 'ChatGLMModel':
            if 'rope_theta' not in model_config:
                model_config['rope_theta'] = 10000
            model = ChatGLMForCausalLM(model_config)
        elif architecture == 'Qwen2ForCausalLM':
            model = Qwen2ForCausalLM(model_config)
        else:
            assert 0
            
        model.load_weights(weights)
        return model