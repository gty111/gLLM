import glob
import torch

from logger import logger
from safetensors import safe_open
from transformers import AutoConfig

from gllm.models.llama import LlamaForCausalLM
from gllm.models.chatglm import ChatGLMForCausalLM
from gllm.models.qwen2 import Qwen2ForCausalLM
from gllm.dist_utils import get_pp_rank


class ModelLoader():
    def __init__(self, load_format, model_path):
        self.model_path = model_path
        self.config = self.load_config()
        self.load_format = load_format
        
    def get_dtype(self, dtype: str):
        if dtype == 'float16':
            return torch.float16
        elif dtype == 'bfloat16':
            return torch.bfloat16
        else:
            assert 0

    def get_finish_tokens(self):
        return self.get_model_type().get_finish_tokens(self.config)

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

        raise Exception('No weights(.bin/.safetensor) found in the directory!')

    def load_config(self):
        config = AutoConfig.from_pretrained(self.model_path,trust_remote_code=True)
        self.dtype = config.torch_dtype
        self.architecture = config.architectures[0]
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        return config
    
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
        model_type = self.get_model_type()
        model = model_type(self.config)
        
        if self.load_format == 'auto':
            weights = self.load_weights()
            logger.info(f"Worker {get_pp_rank()} loading model ...")
            model.load_weights(weights)
        else:
            assert self.load_format == 'dummy'
        return model
