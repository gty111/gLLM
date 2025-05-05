from .qwen2_moe import Qwen2MoeMLP as Qwen3MoeMLP
from .qwen2_moe import Qwen2MoeSparseMoeBlock
from .qwen3 import Qwen3Attention as Qwen3MoeAttention
from .qwen2_moe import Qwen2MoeDecoderLayer
from .qwen2_moe import Qwen2MoeModel
from .qwen2_moe import Qwen2MoeForCausalLM

class Qwen3MoeSparseMoeBlock(Qwen2MoeSparseMoeBlock):
    def __init__(self, config):
        super().__init__(config)

class Qwen3MoeDecoderLayer(Qwen2MoeDecoderLayer):
    def __init__(self, layer_id, config, 
                 moe_block_type=Qwen3MoeSparseMoeBlock,
                 mlp_type=Qwen3MoeMLP,
                 attn_type=Qwen3MoeAttention):
        super().__init__(layer_id, config, moe_block_type, mlp_type, attn_type)
        
class Qwen3MoeModel(Qwen2MoeModel):
    def __init__(self, config, decoder_layer_type=Qwen3MoeDecoderLayer):
        super().__init__(config, decoder_layer_type)
        
class Qwen3MoeForCausalLM(Qwen2MoeForCausalLM):
    def __init__(self, config, model_type=Qwen3MoeModel):
        super().__init__(config, model_type)