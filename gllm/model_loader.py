import json
import glob
from safetensors import safe_open

from gllm.models.llama import LlamaForCausalLM
from gllm.models.chatglm import ChatGLMForCausalLM


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

        architecture = model_config['architectures'][0]
        if architecture == 'LlamaForCausalLM':
            model = LlamaForCausalLM(model_config)
            model.load_weights(weights)
            return model
        elif architecture == 'ChatGLMModel':
            model = ChatGLMForCausalLM(model_config)
            model.load_weights(weights)
            return model
        else:
            assert 0
