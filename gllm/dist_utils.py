import os
import torch.distributed as dist
import torch

def send_tensor(tensor:torch.Tensor, dst):
    tensor_shape = list(tensor.shape)
    # send dim
    dist.send_object_list([len(tensor_shape)],dst)
    # send shape
    dist.send_object_list(tensor_shape,dst)
    # send tensor
    dist.send(tensor,dst)
    
def recv_tensor(dtype, src):
    # recv dim
    dim = [None]
    dist.recv_object_list(dim,src)
    # recv shape
    tensor_shape = [None for _ in range(dim[0])]
    dist.recv_object_list(tensor_shape,src)
    # recv tensor
    tensor = torch.zeros(torch.Size(tensor_shape),dtype=dtype,device=f'cuda:{dist.get_rank()}')
    dist.recv(tensor,src)
    return tensor

def send_pp_data(output, dst):
    if type(output) == tuple:
        assert len(output) == 2
        dist.isend(output[0],dst)
        dist.isend(output[1],dst)
    else:
        dist.isend(output,dst)

def recv_pp_data(src, dtype, shape, has_residual):
    hidden_states = torch.zeros(torch.Size(shape),dtype=dtype,device=f'cuda:{dist.get_rank()}')
    if has_residual:
        residual = hidden_states.clone().detach()
        hidden_states_future = dist.irecv(hidden_states,src)
        residual_future = dist.irecv(residual,src)
        return hidden_states_future, residual_future, hidden_states, residual
    else:
        hidden_states_future = dist.irecv(hidden_states,src)
        return hidden_states_future, hidden_states
    
def init_dist(pp_size, pp_rank, master_addr, master_port):
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    dist.init_process_group(backend='nccl', world_size=pp_size, rank=pp_rank)

def get_pp_layers(num_layers):
    assert num_layers % dist.get_world_size() == 0
    num_layers_pp = num_layers // dist.get_world_size()
    return num_layers_pp * dist.get_rank(), num_layers_pp * (dist.get_rank()+1)
