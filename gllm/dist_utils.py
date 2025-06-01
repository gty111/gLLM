import torch.distributed as dist
import torch

from logger import logger

def send_pp_data(output, dst):
    if type(output) == tuple:
        assert len(output) == 2
        dist.isend(output[0],dst)
        dist.isend(output[1],dst)
    else:
        dist.isend(output,dst)

def recv_pp_data(src, shape, has_residual):
    hidden_states = torch.zeros(torch.Size(shape))
    if has_residual:
        residual = hidden_states.clone().detach()
        hidden_states_future = dist.irecv(hidden_states,src)
        residual_future = dist.irecv(residual,src)
        return hidden_states_future, residual_future, hidden_states, residual
    else:
        hidden_states_future = dist.irecv(hidden_states,src)
        return hidden_states_future, hidden_states
    
def send_obj_list(obj_list, dst):
    dist.send_object_list(obj_list, dst=dst)
    
def recv_obj_list(obj_list, src):
    dist.recv_object_list(obj_list, src=src)

_PP_RANK=0
_LOCAL_RANK=0
_PP_SIZE=1
_ASSIGNED_LAYERS=None

def get_pp_rank():
    return _PP_RANK

def get_local_rank():
    return _LOCAL_RANK

def is_pp_last_rank():
    return get_pp_rank() == get_pp_size() - 1

def get_pp_size():
    return _PP_SIZE

def get_assigned_layers():
    return _ASSIGNED_LAYERS

def init_dist(pp_size, local_rank, pp_rank, master_addr, master_port, assigned_layers):
    global _PP_RANK, _PP_SIZE, _ASSIGNED_LAYERS, _LOCAL_RANK
    _PP_RANK = pp_rank
    _LOCAL_RANK = local_rank
    _PP_SIZE = pp_size
    _ASSIGNED_LAYERS = assigned_layers
    init_method = f'tcp://{master_addr}:{master_port}'
    backend = 'nccl'
    logger.info(f'NCCL: Init_method {init_method}, Backend {backend}, Word_size {pp_size}')
    dist.init_process_group(init_method=init_method, backend=backend, world_size=pp_size, rank=pp_rank)

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

# Set the correct layer index
def resolve_pp_layer(layer_name, idx, start_layer_idx):
    if 'layers' in layer_name:
        layer_name_list = layer_name.split('.')
        layer_name_list[idx] = str(int(layer_name_list[idx])+start_layer_idx)
        return '.'.join(layer_name_list)
    else:
        return layer_name
