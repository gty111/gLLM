import torch
import contextlib

try:
    import gllm._C
except ImportError as e:
    print(e)
    
supports_moe_ops = False
with contextlib.suppress(ImportError):
    import gllm._moe_C  # noqa: F401
    supports_moe_ops = True

# cache ops
def reshape_and_cache_flash(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    torch.ops._C.reshape_and_cache_flash(key, value, key_cache, value_cache, slot_mapping)

# activation ops
def silu_and_mul(out: torch.Tensor, x: torch.Tensor) -> None:
    torch.ops._C.silu_and_mul(out, x)


# pos encoding ops
def rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
) -> None:
    torch.ops._C.rotary_embedding(positions, query, key, head_size,
                                  cos_sin_cache, is_neox)


def batched_rotary_embedding(positions: torch.Tensor, query: torch.Tensor,
                             key: torch.Tensor, head_size: int,
                             cos_sin_cache: torch.Tensor, is_neox: bool,
                             rot_dim: int,
                             cos_sin_cache_offsets: torch.Tensor) -> None:
    torch.ops._C.batched_rotary_embedding(positions, query, key, head_size,
                                          cos_sin_cache, is_neox, rot_dim,
                                          cos_sin_cache_offsets)


# layer norm ops
def rms_norm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor,
             epsilon: float) -> None:
    # TODO: Remove this contiguous call when the kernel is updated to support non-contiguous input
    input = input.contiguous()
    torch.ops._C.rms_norm(out, input, weight, epsilon)


def fused_add_rms_norm(input: torch.Tensor, residual: torch.Tensor,
                       weight: torch.Tensor, epsilon: float) -> None:
    torch.ops._C.fused_add_rms_norm(input, residual, weight, epsilon)

def topk_softmax(topk_weights: torch.Tensor, topk_ids: torch.Tensor,
                 token_expert_indicies: torch.Tensor,
                 gating_output: torch.Tensor) -> None:
    torch.ops._moe_C.topk_softmax(topk_weights, topk_ids,
                                  token_expert_indicies, gating_output)

# moe
def moe_sum(input: torch.Tensor, output: torch.Tensor):
    torch.ops._moe_C.moe_sum(input, output)

def moe_align_block_size(topk_ids: torch.Tensor, num_experts: int,
                         block_size: int, sorted_token_ids: torch.Tensor,
                         experts_ids: torch.Tensor,
                         num_tokens_post_pad: torch.Tensor) -> None:
    torch.ops._moe_C.moe_align_block_size(topk_ids, num_experts, block_size,
                                          sorted_token_ids, experts_ids,
                                          num_tokens_post_pad)

def sgl_moe_align_block_size(topk_ids: torch.Tensor, num_experts: int,
                             block_size: int, sorted_token_ids: torch.Tensor,
                             experts_ids: torch.Tensor,
                             num_tokens_post_pad: torch.Tensor) -> None:
    torch.ops._moe_C.sgl_moe_align_block_size(topk_ids, num_experts,
                                              block_size, sorted_token_ids,
                                              experts_ids, num_tokens_post_pad)