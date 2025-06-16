import glob
import torch

from safetensors import safe_open
from transformers import AutoConfig, GenerationConfig
from huggingface_hub import snapshot_download
from logger import logger

from gllm.models.llama import LlamaForCausalLM
from gllm.models.chatglm import ChatGLMForCausalLM
from gllm.models.qwen2 import Qwen2ForCausalLM
from gllm.models.qwen2_moe import Qwen2MoeForCausalLM
from gllm.models.qwen3 import Qwen3ForCausalLM
from gllm.models.qwen3_moe import Qwen3MoeForCausalLM
from gllm.models.mixtral import MixtralForCausalLM
from gllm.utils import get_lock


class ModelLoader():
    def __init__(self, load_format, model_path):
        self.model_path = model_path
        self.load_config()
        self.load_format = load_format

    def get_finish_tokens(self):
        return self.get_model_type().get_finish_tokens(self.config)
    
    def load_safetensors(self, path):
        weights_path = glob.glob(f"{path}/*.safetensors")
        for weight_path in weights_path:
            with safe_open(weight_path, framework="pt", device="cpu") as f:
                for k in f.keys():
                    self.weights[k] = f.get_tensor(k)
        return len(self.weights) != 0
                    
    def load_bin(self, path):
        weights_path = glob.glob(f'{path}/*.bin')
        for weight_path in weights_path:
            self.weights.update(torch.load(weight_path,weights_only=True))
        return len(self.weights) != 0
            
    def load_weights_from_local(self,path):
        if self.load_safetensors(path):
            return True
        
        if self.load_bin(path):
            return True
        
        return False
        
    def load_weights_from_huggingface(self, path):
        try:
            with get_lock(path, None):
                cached_path = snapshot_download(path, 
                                                allow_patterns=["*.safetensors", "*.bin"],
                                                ignore_patterns=["original/**/*"])
                return self.load_weights_from_local(cached_path)
        except Exception as e:
            raise Exception(f'Failed to load {self.model_path} because of {e}!')

    def load_weights(self):
        self.weights = {}
        
        if self.load_weights_from_local(self.model_path):
            return
        
        if self.load_weights_from_huggingface(self.model_path):
            return

        raise Exception(f'Failed to load {self.model_path} from local or huggingface!')

    def load_config(self):
        self.config = AutoConfig.from_pretrained(self.model_path,trust_remote_code=True)
        self.generation_config = GenerationConfig.from_pretrained(self.model_path)
        self.dtype = self.config.torch_dtype
        self.architecture = self.config.architectures[0]
        self.vocab_size = self.config.vocab_size
        self.hidden_size = self.config.hidden_size
    
    def get_model_type(self):
        model_type = None
        if self.architecture == 'LlamaForCausalLM':
            model_type = LlamaForCausalLM
        elif self.architecture == 'ChatGLMModel':
            model_type = ChatGLMForCausalLM
        elif self.architecture == 'Qwen2ForCausalLM':
            model_type = Qwen2ForCausalLM
        elif self.architecture == 'Qwen3ForCausalLM':
            model_type = Qwen3ForCausalLM
        elif self.architecture == 'Qwen2MoeForCausalLM':
            model_type = Qwen2MoeForCausalLM
        elif self.architecture == 'Qwen3MoeForCausalLM':
            model_type = Qwen3MoeForCausalLM
        elif self.architecture == 'MixtralForCausalLM':
            model_type = MixtralForCausalLM
        else:
            raise Exception(f'Unsupported model: {self.architecture}')
        return model_type

    def load_model(self, mp_load_progress=None):
        model_type = self.get_model_type()
        
        torch.set_default_dtype(self.dtype)
        
        # Load weights to CPU memory
        if self.load_format == 'auto':
            self.load_weights()
        
        torch.set_default_device('cuda')
        
        # Init model whose weights are on GPU memory 
        free_gpu_memory_before, _ = torch.cuda.mem_get_info()
        model = model_type(self.config)
        free_gpu_memory_after, _ = torch.cuda.mem_get_info()
        model_size_gb = round((free_gpu_memory_before - free_gpu_memory_after)/(2**30),2)
        logger.info(f'Model architecture: {self.architecture}, Default dtype: {self.dtype}, Model weights {model_size_gb} GB')
        
        # Load weights from CPU memory to GPU memory 
        if self.load_format == 'auto':
            model.load_weights(self.weights,mp_load_progress)
        return model
