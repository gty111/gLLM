# Resolve weights loading for TP

from gllm.dist_utils import get_tp_rank

def copy_qkv_proj_weight(dst_qkv, src_q, src_k, src_v, num_heads, num_kv_heads, head_dim):
    dst_qkv[:num_heads*head_dim, :] = src_q[get_tp_rank()*num_heads*head_dim:(get_tp_rank()+1)*num_heads*head_dim,:]
    dst_qkv[num_heads*head_dim:(num_heads +
            num_kv_heads)*head_dim, :] = src_k[get_tp_rank()*num_kv_heads*head_dim:(get_tp_rank()+1)*num_kv_heads*head_dim,:]
    dst_qkv[(num_heads +
            num_kv_heads)*head_dim:, :] = src_v[get_tp_rank()*num_kv_heads*head_dim:(get_tp_rank()+1)*num_kv_heads*head_dim,:]
    
def copy_qkv_proj_bias(dst_qkv, src_q, src_k, src_v, num_heads, num_kv_heads, head_dim):
    dst_qkv[:num_heads*head_dim] = src_q[get_tp_rank()*num_heads*head_dim:(get_tp_rank()+1)*num_heads*head_dim]
    dst_qkv[num_heads*head_dim:(num_heads +
            num_kv_heads)*head_dim] = src_k[get_tp_rank()*num_kv_heads*head_dim:(get_tp_rank()+1)*num_kv_heads*head_dim]
    dst_qkv[(num_heads +
            num_kv_heads)*head_dim:] = src_v[get_tp_rank()*num_kv_heads*head_dim:(get_tp_rank()+1)*num_kv_heads*head_dim]
    
def copy_gate_up_proj_weight(dst, src_gate, src_up, intermediate_size_partition, partition_tp=True):
    if partition_tp:
        dst[:intermediate_size_partition, :] = src_gate[get_tp_rank()*intermediate_size_partition:(get_tp_rank()+1)*intermediate_size_partition, :]
        dst[intermediate_size_partition:, :] = src_up[get_tp_rank()*intermediate_size_partition:(get_tp_rank()+1)*intermediate_size_partition, :]
    else:
        dst[:intermediate_size_partition, :] = src_gate
        dst[intermediate_size_partition:, :] = src_up
    
def copy_single_proj_col(dst, src, size_partition, partition_tp=True):
    # partition on column
    if partition_tp:
        dst.copy_(src[: , get_tp_rank()*size_partition:(get_tp_rank()+1)*size_partition])
    else:
        dst.copy_(src)

def copy_single_proj_row(dst, src, size_partition):
    # partition on row
    dst.copy_(src[get_tp_rank()*size_partition:(get_tp_rank()+1)*size_partition, :])