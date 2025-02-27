import os
import torch.distributed as dist
import torch

from gllm.input_data import InputData

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

def send_pp_data(input_data:InputData, output, dst):
    '''output: (hidden_states, residual)'''
    
    send_tensor(input_data.temperature, dst)
    send_tensor(input_data.top_p, dst)
    send_tensor(input_data.top_k, dst)

    send_tensor(input_data.slot_mapping_tensor, dst)
    send_tensor(input_data.positions, dst)
    
    # send computed_prompt, prefix_prefill
    obj_list = [input_data.computed_prompt, input_data.prefix_prefill]
    dist.send_object_list(obj_list, dst)
    
    if not input_data.computed_prompt:
        send_tensor(input_data.seq_start_loc, dst)
        if input_data.prefix_prefill:
            obj_list = [input_data.max_seq_len, input_data.max_query_len]
            dist.send_object_list(obj_list, dst)
            send_tensor(input_data.block_table, dst)
            send_tensor(input_data.query_start_loc, dst)
        else:
            obj_list = [input_data.max_seq_len]
            dist.send_object_list(obj_list, dst)
    else:
        send_tensor(input_data.cache_seqs_len, dst)
        send_tensor(input_data.block_table, dst)
    
    # send hidden_states, residual
    send_tensor(output[0], dst)
    send_tensor(output[1], dst)
    
def recv_pp_data(dtype, memory_manager, src):
    temperature = recv_tensor(dtype, src)
    print('temperature',temperature)
    top_p = recv_tensor(dtype, src)
    print('top_p',top_p)
    top_k = recv_tensor(dtype, src)
    print('top_k',top_k)
    
    slot_mapping_tensor = recv_tensor(torch.int64, src)
    print('slot_mapping_tensor',slot_mapping_tensor)
    positions = recv_tensor(torch.long, src)
    print('positions',positions)
    
    # recv computed_prompt, prefix_prefill
    obj_list = [None, None]
    dist.recv_object_list(obj_list, src)
    
    input_data = None
    
    computed_prompt, prefix_prefill = obj_list
    print(computed_prompt, prefix_prefill)
    if not computed_prompt:
        seq_start_loc = recv_tensor(torch.int32, src)
        print('seq_start_loc',seq_start_loc)
        if prefix_prefill:
            obj_list = [None, None]
            dist.recv_object_list(obj_list,src)
            max_seq_len,max_query_len = obj_list
            block_table = recv_tensor(torch.int32, src)
            query_start_loc = recv_tensor(torch.int32, src)
            input_data = InputData.build_prefill(
                temperature, top_p, top_k,
                prefix_prefill, memory_manager, slot_mapping_tensor,
                positions,max_seq_len, seq_start_loc,
                block_table, max_query_len, query_start_loc)
        else:
            obj_list = [None]
            dist.recv_object_list(obj_list,src)
            max_seq_len = obj_list[0]
            input_data = InputData.build_prefill(
                temperature, top_p, top_k,
                prefix_prefill, memory_manager, slot_mapping_tensor,
                positions,max_seq_len, seq_start_loc)
    else:
        cache_seqs_len = recv_tensor(torch.int32, src)
        block_table = recv_tensor(torch.int32, src)
        input_data = InputData.build_decode(
            temperature, top_p, top_k,
            slot_mapping_tensor,memory_manager,positions,
            cache_seqs_len,block_table)
    
    # recv hidden_states, residual
    hidden_states = recv_tensor(dtype, src)
    print('hidden_states', hidden_states)
    residual = recv_tensor(dtype, src)
    print('residual', residual)
    return input_data, hidden_states, residual

def init_dist(pp_size, pp_rank, master_addr, master_port):
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    dist.init_process_group(backend='nccl', world_size=pp_size, rank=pp_rank)
    torch.cuda.set_device(f'cuda:{pp_rank}')

def get_pp_layers(num_layers):
    assert num_layers % dist.get_world_size() == 0
    num_layers_pp = num_layers // dist.get_world_size()
    return num_layers_pp * dist.get_rank(), num_layers_pp * (dist.get_rank()+1)
