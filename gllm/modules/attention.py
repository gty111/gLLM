from torch import nn

from gllm.dist_utils import get_tp_size


class Attention(nn.Module):
    def __init__(self, total_num_heads, total_num_kv_heads, hidden_size, head_dim=None):
        super().__init__()
        
        self.hidden_size = hidden_size
        tp_size = get_tp_size()
        
        self.total_num_heads = total_num_heads
        if self.total_num_heads % tp_size != 0:
            raise Exception(f'total_num_heads({self.total_num_heads}) is not divisible by tp_size({tp_size})')
        self.num_heads = self.total_num_heads // tp_size
        
        self.total_num_kv_heads = total_num_kv_heads
        if self.total_num_kv_heads % tp_size != 0:
            raise Exception(f'total_num_kv_heads({self.total_num_kv_heads}) is not divisible by tp_size({tp_size})')
        self.num_kv_heads =  self.total_num_kv_heads // tp_size
        
        self.head_dim = (self.hidden_size // self.total_num_heads 
                            if head_dim is None else head_dim)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        