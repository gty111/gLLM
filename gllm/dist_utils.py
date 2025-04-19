import os
import torch.distributed as dist
import torch

from logger import logger

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

_PP_RANK=0
_PP_SIZE=1
_ASSIGNED_LAYERS=None

def get_pp_rank():
    return _PP_RANK

def get_pp_size():
    return _PP_SIZE

def get_assigned_layers():
    return _ASSIGNED_LAYERS

def init_dist(pp_size, pp_rank, device_size, device_rank, master_addr, master_port, assigned_layers):
    global _PP_RANK, _PP_SIZE, _ASSIGNED_LAYERS
    _PP_RANK = pp_rank
    _PP_SIZE = pp_size
    _ASSIGNED_LAYERS = assigned_layers
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    dist.init_process_group(backend='nccl', world_size=device_size, rank=device_rank)

def get_pp_layers(num_layers):
    if _ASSIGNED_LAYERS is None:
        num_layers_pp = num_layers // get_pp_size()
        
        if get_pp_size() <= 4 or num_layers % get_pp_size() != 0:
            num_layers_pp += 1

        if get_pp_rank() != get_pp_size() - 1:
            assigned_layers = num_layers_pp * get_pp_rank(), num_layers_pp * (get_pp_rank()+1)
        else:
            assigned_layers = num_layers_pp * get_pp_rank(), num_layers
    else:
        total_assigned_layers = [int(i) for i in _ASSIGNED_LAYERS.split(',')]
        assert len(total_assigned_layers) == get_pp_size() and sum(total_assigned_layers) == num_layers
        assigned_layers = [sum(total_assigned_layers[:get_pp_rank()]), sum(total_assigned_layers[:get_pp_rank()+1])]
    
    if get_pp_size() > 1:
        logger.info('Assigned layers: (%3d,%3d) #total: %2d'%
                    (
                        assigned_layers[0],
                        assigned_layers[1]-1,
                        assigned_layers[1]-assigned_layers[0]
                    ))
    
    return assigned_layers
