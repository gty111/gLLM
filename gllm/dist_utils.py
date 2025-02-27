import os
import torch.distributed as dist
import torch

def send_pp_data(output, dst):
    # output: (hidden_states, residual)
    tensor_shape = list(output[0].shape)
    assert len(tensor_shape) == 2
    dist.send_object_list(tensor_shape, dst)
    dist.send(output[0], dst)
    dist.send(output[1], dst)
    
def recv_pp_data(dtype, src):
    tensor_shape = [0,0]
    dist.recv_object_list(tensor_shape,src)
    hidden_states = torch.zeros(torch.Size(tensor_shape),dtype=dtype, device=f'cuda:{dist.get_rank()}')
    residual = hidden_states.clone().detach()
    dist.recv(hidden_states,src)
    dist.recv(residual,src)
    return hidden_states, residual
    
def get_min_num_pages(num_pages):
    num_pages_all = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(num_pages_all, num_pages)
    return min(num_pages_all)

def init_dist(pp_size, pp_rank, master_addr, master_port):
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    dist.init_process_group(backend='nccl', world_size=pp_size, rank=pp_rank)
    torch.cuda.set_device(f'cuda:{pp_rank}')

def get_pp_layers(num_layers):
    assert num_layers % dist.get_world_size() == 0
    num_layers_pp = num_layers // dist.get_world_size()
    return num_layers_pp * dist.get_rank(), num_layers_pp * (dist.get_rank()+1)
