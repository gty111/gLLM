# Resolve weights loading for TP

from gllm.dist_utils import get_tp_rank
    
def copy_qkv_proj(dst_qkv, src_q, src_k, src_v, num_heads, num_kv_heads, head_dim):
    dst_qkv[:num_heads*head_dim] = src_q[get_tp_rank()*num_heads*head_dim:(get_tp_rank()+1)*num_heads*head_dim]
    dst_qkv[num_heads*head_dim:(num_heads +
            num_kv_heads)*head_dim] = src_k[get_tp_rank()*num_kv_heads*head_dim:(get_tp_rank()+1)*num_kv_heads*head_dim]
    dst_qkv[(num_heads +
            num_kv_heads)*head_dim:] = src_v[get_tp_rank()*num_kv_heads*head_dim:(get_tp_rank()+1)*num_kv_heads*head_dim]

def copy_qkv_a_proj(dst, q_a_proj, kv_a_proj_with_mqa):
    size_partition = q_a_proj.shape[0]
    dst[:size_partition] = q_a_proj
    dst[size_partition:] = kv_a_proj_with_mqa

def copy_gate_up_proj(dst, src_gate, src_up, partition_tp=True):
    size_partition = dst.shape[0] // 2
    if partition_tp:
        dst[:size_partition] = src_gate[get_tp_rank()*size_partition:(get_tp_rank()+1)*size_partition]
        dst[size_partition:] = src_up[get_tp_rank()*size_partition:(get_tp_rank()+1)*size_partition]
    else:
        dst[:size_partition] = src_gate
        dst[size_partition:] = src_up
    
def copy_single_proj_dim1(dst, src, partition_tp=True):
    # partition on column
    if partition_tp:
        size_partition = dst.shape[1]
        dst.copy_(src[: , get_tp_rank()*size_partition:(get_tp_rank()+1)*size_partition])
    else:
        dst.copy_(src)

def copy_single_proj_dim0(dst, src):
    # partition on row
    size_partition = dst.shape[0]
    dst.copy_(src[get_tp_rank()*size_partition:(get_tp_rank()+1)*size_partition])