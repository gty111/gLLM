import torch.distributed as dist
import torch

from logger import logger
from collections.abc import Sequence

def send_pp_data(output, dst):
    if type(output) == tuple:
        assert len(output) == 2
        dist.isend(output[0], dst)
        dist.isend(output[1], dst)
    else:
        dist.isend(output, dst)

def recv_pp_data(src, shape, has_residual):
    hidden_states = torch.zeros(torch.Size(shape))
    if has_residual:
        residual = hidden_states.clone().detach()
        hidden_states_future = dist.irecv(hidden_states, src)
        residual_future = dist.irecv(residual, src)
        return hidden_states_future, residual_future, hidden_states, residual
    else:
        hidden_states_future = dist.irecv(hidden_states, src)
        return hidden_states_future, hidden_states
    
def send_obj_list(obj_list, dst):
    dist.send_object_list(obj_list, dst=dst)
    
def recv_obj_list(obj_list, src):
    dist.recv_object_list(obj_list, src=src)

_RANK=0
_PP_RANK=0
_TP_RANK=0
_EP_RANK=0
_LOCAL_RANK=0
_PP_SIZE=1
_TP_SIZE=1
_EP_SIZE=1
_WORLD_SIZE=1
_ASSIGNED_LAYERS=None
_TP_GROUP=None
_USE_EP=True

def get_rank():
    return _RANK

def get_world_size():
    return _WORLD_SIZE

def get_pp_rank():
    return _PP_RANK

def get_tp_rank():
    return _TP_RANK

def get_ep_rank():
    return _EP_RANK

def get_local_rank():
    return _LOCAL_RANK

def get_output_rank():
    return (get_pp_size() - 1) * get_tp_size()

def is_output_rank():
    return is_last_pp_rank() and is_first_tp_rank()

def is_first_tp_rank():
    return get_tp_rank() == 0

def is_last_pp_rank():
    return get_pp_rank() == get_pp_size() - 1

def is_use_ep():
    return _USE_EP

def get_next_pp_rank():
    return get_rank() + get_tp_size()

def get_last_pp_rank():
    return get_rank() - get_tp_size()

def get_pp_size():
    return _PP_SIZE

def get_tp_size():
    return _TP_SIZE

def get_ep_size():
    return _EP_SIZE

def get_assigned_layers():
    return _ASSIGNED_LAYERS

def get_tp_group():
    return _TP_GROUP

def init_tp_group():
    global _TP_GROUP
    tp_groups = [list(range(_pp_rank*get_tp_size(), (_pp_rank+1)*get_tp_size())) for _pp_rank in range(get_pp_size())]
    for tp_ranks in tp_groups:
        tp_group = dist.new_group(tp_ranks)
        if _RANK in tp_ranks:
            _TP_GROUP = tp_group

def init_dist(pp_size, tp_size, use_ep, local_rank, pp_rank, tp_rank, master_addr, master_port, assigned_layers):
    global _RANK, _PP_RANK, _TP_RANK, _PP_SIZE, _TP_SIZE, _WORLD_SIZE, _ASSIGNED_LAYERS, _LOCAL_RANK, _TP_GROUP
    global _EP_SIZE, _EP_RANK, _USE_EP
    _RANK = pp_rank * tp_size + tp_rank
    _PP_RANK = pp_rank
    _TP_RANK = tp_rank
    _EP_RANK = _TP_RANK if use_ep else 0
    _LOCAL_RANK = local_rank
    _PP_SIZE = pp_size
    _TP_SIZE = tp_size
    _EP_SIZE = _TP_SIZE if use_ep else 1
    _USE_EP = use_ep
    _WORLD_SIZE = pp_size * tp_size
    _ASSIGNED_LAYERS = assigned_layers
    
    self_tp_ranks = list(range(pp_rank*tp_size, (pp_rank+1)*tp_size))
    
    init_method = f'tcp://{master_addr}:{master_port}'
    backend = 'nccl'
    tp_ep_log = 'TP Groups' if not use_ep else 'TP/EP Groups'
    logger.info(f'NCCL: Init_method {init_method}, Backend {backend}, Rank {_RANK}, {tp_ep_log} {self_tp_ranks}, Word_size {_WORLD_SIZE}')
    dist.init_process_group(init_method=init_method, backend=backend, world_size=_WORLD_SIZE, rank=_RANK)
    
    init_tp_group()

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
        logger.info('Assigned %2d layers: (%3d,%3d)'%
                    (
                        assigned_layers[1]-assigned_layers[0],
                        assigned_layers[0],
                        assigned_layers[1]-1,
                    ))
    
    return assigned_layers

# Set the correct layer index for PP
def resolve_pp_layer_idx(layer_name, idx, start_layer_idx):
    if 'layers' in layer_name:
        layer_name_list = layer_name.split('.')
        layer_name_list[idx] = str(int(layer_name_list[idx])+start_layer_idx)
        return '.'.join(layer_name_list)
    else:
        return layer_name
    
def resolve_ep_expert_idx(expert_idx, expert_map):
    if expert_map is not None:
        local_expert_idx = expert_map[expert_idx]
    else:
        local_expert_idx = expert_idx
    return local_expert_idx
    
def tensor_model_parallel_all_gather(input_: torch.Tensor, dim=-1) -> torch.Tensor:
    """All-gather the input tensor across model parallel group."""
    if dim < 0:
        # Convert negative dim to positive.
        dim += input_.dim()
    input_size = input_.size()
    # NOTE: we have to use concat-style all-gather here,
    # stack-style all-gather has compatibility issues with
    # torch.compile . see https://github.com/pytorch/pytorch/issues/138795
    output_size = (input_size[0] * get_tp_size(), ) + input_size[1:]
    # Allocate output tensor.
    output_tensor = torch.empty(output_size,
                                dtype=input_.dtype,
                                device=input_.device)
    # All-gather.
    dist.all_gather_into_tensor(output_tensor,
                                input_,
                                group=get_tp_group())
    # Reshape
    output_tensor = output_tensor.reshape((get_tp_size(), ) + input_size)
    output_tensor = output_tensor.movedim(0, dim)
    output_tensor = output_tensor.reshape(input_size[:dim] +
                                            (get_tp_size() *
                                            input_size[dim], ) +
                                            input_size[dim + 1:])
    return output_tensor

def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    dist.all_reduce(input_, group=get_tp_group())
    return input_

def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, "{} is not divisible by {}".format(
        numerator, denominator)


def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator

def split_tensor_along_last_dim(
    tensor: torch.Tensor,
    num_partitions: int,
    contiguous_split_chunks: bool = False,
) -> Sequence[torch.Tensor]:
    """ Split a tensor along its last dimension.

        Arguments:
            tensor: input tensor.
            num_partitions: number of partitions to split the tensor
            contiguous_split_chunks: If True, make each chunk contiguous
                                     in memory.

        Returns:
            A list of Tensors
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    last_dim_size = divide(tensor.size()[last_dim], num_partitions)
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # NOTE: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list