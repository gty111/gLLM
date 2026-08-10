from torch import nn

from gllm.distributed.parallel_state import get_tp_size


class AttentionLayerBase(nn.Module):
    """Shared TP-sharded head geometry for model-specific attention layers."""

    def __init__(self, total_num_heads, total_num_kv_heads, hidden_size, head_dim=None):
        super().__init__()

        self.hidden_size = hidden_size
        tp_size = get_tp_size()

        self.total_num_heads = total_num_heads
        if self.total_num_heads % tp_size != 0:
            raise Exception(
                f"total_num_heads({self.total_num_heads}) is not divisible by "
                f"tp_size({tp_size})"
            )
        self.num_heads = self.total_num_heads // tp_size

        self.total_num_kv_heads = total_num_kv_heads
        if self.total_num_kv_heads % tp_size != 0:
            raise Exception(
                f"total_num_kv_heads({self.total_num_kv_heads}) is not divisible "
                f"by tp_size({tp_size})"
            )
        self.num_kv_heads = self.total_num_kv_heads // tp_size

        self.head_dim = (
            self.hidden_size // self.total_num_heads if head_dim is None else head_dim
        )
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
