import json
import glob
import torch
import torch.distributed as dist

from logger import logger
from safetensors import safe_open

from gllm.models.llama import LlamaForCausalLM
from gllm.models.chatglm import ChatGLMForCausalLM
from gllm.models.qwen2 import Qwen2ForCausalLM


class ModelLoader():
    def __init__(self, model_path):
        self.model_path = model_path
        self.model_config = self.load_config()
        self.post_process_config()
        
    def get_dtype(self, dtype: str):
        if dtype == 'float16':
            return torch.float16
        elif dtype == 'bfloat16':
            return torch.bfloat16
        else:
            assert 0

    def get_finish_tokens(self):
        return self.get_model_type().get_finish_tokens(self.model_config)

    def load_weights(self):
        weights = {}
        
        # load .safetensor
        weights_path = glob.glob(f"{self.model_path}/*.safetensors")
        for weight_path in weights_path:
            with safe_open(weight_path, framework="pt", device="cpu") as f:
                for k in f.keys():
                    weights[k] = f.get_tensor(k)
        
        if len(weights) != 0:
            return weights
        
        # load .bin
        weights_path = glob.glob(f'{self.model_path}/*.bin')
        for weight_path in weights_path:
            weights.update(torch.load(weight_path,weights_only=True))
        
        if len(weights) != 0:
            return weights

        assert 0
        
    def post_process_config(self):
        if self.architecture == 'ChatGLMModel':
            if 'rope_scaling' not in self.model_config:
                self.model_config['rope_theta'] = 10000
            else:
                self.model_config['rope_theta'] = self.model_config['rope_scaling'] * 10000
        elif self.architecture == 'LlamaForCausalLM':
            if 'rope_theta' not in self.model_config:
                self.model_config['rope_theta'] = 10000

    def load_config(self):
        config_path = f"{self.model_path}/config.json"
        with open(config_path, 'r') as f:
            model_config = json.load(f)
        self.dtype = self.get_dtype(
            model_config['torch_dtype'])
        model_config['torch_dtype'] = self.dtype
        self.architecture = model_config['architectures'][0]
        if 'vocab_size' in model_config:
            self.vocab_size = int(model_config['vocab_size'])
        elif 'padded_vocab_size' in model_config:
            self.vocab_size = int(model_config['padded_vocab_size'])
        else:
            assert 0
        return model_config
    
    def get_model_type(self):
        model_type = None
        if self.architecture == 'LlamaForCausalLM':
            model_type = LlamaForCausalLM
        elif self.architecture == 'ChatGLMModel':
            model_type = ChatGLMForCausalLM
        elif self.architecture == 'Qwen2ForCausalLM':
            model_type = Qwen2ForCausalLM
        else:
            assert 0
        return model_type

    def load_model(self):
        weights = self.load_weights()
        model_type = self.get_model_type()
        model = model_type(self.model_config)
        
        logger.info(f"Worker {dist.get_rank()} loading model ...")
        model.load_weights(weights)
        return model
