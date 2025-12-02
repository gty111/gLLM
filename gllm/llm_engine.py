import tqdm
import torch.multiprocessing as mp
import sys
import time

from logger import logger
from typing import List, Dict

from gllm.model_runner import ModelRunner
from gllm.sequence import Sequence
from gllm.id_allocator import IDAllocator
from gllm.utils import init_logger, random_uuid, get_model_load_pbar
from gllm.comm import zmqComm, IPCPackage
from gllm.worker import Worker, run_worker
from gllm.async_worker import AsyncWorker, run_worker_async

class LLM():
    def __init__(self, model_path, host=None, master_addr:str='0.0.0.0', master_port:str='8000', 
                 zmq_port_base:int=8001, launch_mode:str='normal', worker_ranks:str=None, 
                 load_format:str='auto', gpu_memory_util=0.9, page_size=16, maxd=2048, maxp=2048, minp=32, 
                 iterp=8, kvthresh=0.05, enable_prefix_caching=True, pp_size=1, tp_size=1, use_ep=True,
                 assigned_layers=None, use_cp_schedule=False, use_async_worker=False, use_thinking=True):
        init_logger()
        self.model_path = model_path
        self.load_format = load_format
        self.model_runner = ModelRunner(
            load_format, model_path, gpu_memory_util, page_size, enable_prefix_caching, use_thinking,
            maxp, maxd, kvthresh, minp, iterp, use_cp_schedule)
        self.pp_size = pp_size
        self.tp_size = tp_size
        self.use_ep = use_ep
        self.host = host
        self.master_addr = master_addr
        self.master_port = master_port
        self.zmq_port_base = zmq_port_base
        self.launch_mode = launch_mode
        self.worker_ranks = worker_ranks
        self.id_allocator = IDAllocator(0, 99999)
        self.finish_tokens = self.model_runner.model_loader.generation_config.eos_token_id
        if type(self.finish_tokens) == int:
            self.finish_tokens = [self.finish_tokens]
        self.model_max_length = self.model_runner.tokenizer.model_max_length
        self.generation_config = self.model_runner.model_loader.generation_config
        
        self.assigned_layers = assigned_layers
        self.use_cp_schedule = use_cp_schedule
        self.use_async_worker = use_async_worker
        
        schedule_method = 'Sarathi-Serve (chunked prefill)' if self.use_cp_schedule else 'Token Throttling'
        logger.info(f'Schedule method: {schedule_method}')
        
        # Interact with workers
        self.wait_lists: List[Sequence] = []
        self.abort_ids: List[int] = []
        self.running_maps: Dict[int, Sequence] = dict() # seq_id => Sequence
        self.async_streams = None
        
        # Init workers
        self.init_workers()

        # wait worker start
        self.wait_workers()
    
    def wait_workers(self):
        while True:
            num_worker_start = 0
            for i in self.mp_alive:
                if i==-1:
                    sys.exit()
                num_worker_start += i
            if num_worker_start == self.num_workers:
                break
            time.sleep(1)
        
    def init_workers(self):
        if self.launch_mode != 'normal':
            if self.worker_ranks is None:
                logger.error('Please specify arg --ranks when the launching mode is master/slave')
                sys.exit(1)
            self.act_worker_ranks = [int(i) for i in self.worker_ranks.split(',')]
            assert len(self.act_worker_ranks) != 0
        else:
            self.act_worker_ranks = list(range(self.pp_size*self.tp_size))
        self.num_workers = len(self.act_worker_ranks)

        self.ctx = mp.get_context('spawn')
        self.mp_alive = self.ctx.Array('i', [0 for i in range(self.num_workers)])
        self.mp_load_progress = self.ctx.Array(
            'i', [0 for _ in range(self.num_workers*2)])

        ipc_path_prefix = random_uuid()
        self.schedule_path = f'ipc:///tmp/{ipc_path_prefix}_gllm_schedule'
        self.output_path = f'ipc:///tmp/{ipc_path_prefix}_gllm_output'
        self.token_path = f'ipc:///tmp/{ipc_path_prefix}_gllm_token'

        self.comm = zmqComm(self.host, self.zmq_port_base, self.launch_mode, self.master_addr, 
                            self.schedule_path, self.output_path, self.token_path, frontend=True)
        self.comm.init()

        logger.info(f'Launching worker {self.act_worker_ranks}, PP size {self.pp_size}, TP size {self.tp_size} ...')
        if self.use_async_worker:
            logger.warning(f'AsyncWorker is an experimental feature')
            
        self.process_list = []
        for local_rank, rank in enumerate(self.act_worker_ranks):
            pp_rank = rank // self.tp_size
            tp_rank = rank % self.tp_size
            self.start_worker(local_rank, pp_rank, tp_rank)

        if self.load_format == 'auto':
            self.load_progress()
        
    def start_worker(self, local_rank, pp_rank, tp_rank):
        worker_cls = Worker if not self.use_async_worker else AsyncWorker
        comm = zmqComm(self.host,
                       self.zmq_port_base,
                       self.launch_mode,
                       self.master_addr,
                       self.schedule_path,
                       self.output_path,
                       self.token_path)
        worker = worker_cls(self.model_runner,
                            local_rank,
                            pp_rank,
                            tp_rank,
                            self.pp_size,
                            self.tp_size,
                            self.use_ep,
                            self.master_addr,
                            self.master_port,
                            comm,
                            self.mp_alive,
                            self.mp_load_progress,
                            self.assigned_layers,
                            self.use_cp_schedule)
        process = self.ctx.Process(
                    target=run_worker if not self.use_async_worker else run_worker_async,
                    args=(worker,),
                    daemon=True)
        self.process_list.append(process)
        process.start()
    
    def load_progress(self):
        total_weights = 0
        while True:
            self.check_worker_alive()
            ready = True
            total_weights = 0
            for i in range(self.num_workers):
                if self.mp_load_progress[i*2] == 0:
                    ready = False
                    continue
                total_weights += self.mp_load_progress[i*2]
            if ready:
                break
        pbar = get_model_load_pbar(total_weights)
        last_total_weights = 0
        while True:
            self.check_worker_alive()
            cur_total_weights = 0
            for i in range(self.num_workers):
                cur_total_weights += self.mp_load_progress[i*2+1]
            pbar.update(cur_total_weights-last_total_weights)
            last_total_weights = cur_total_weights
            if cur_total_weights == total_weights:
                break
        
    def check_worker_alive(self):
        for i in self.mp_alive:
            if i==-1:
                sys.exit()
        
    def add_requests(self, requests: List[Sequence]):
        self.wait_lists.extend(requests)
    
    def recv_ipc_package(self):
        '''
        return: number of finished requests in each schedule
        '''
        ipc_package:IPCPackage = self.comm.recv_output()
        if ipc_package is not None:
            for idx, id in enumerate(ipc_package.act_schedule_ids):
                if len(ipc_package.next_tokens) != 0:
                    seq: Sequence = self.running_maps[id]
                    seq.append(ipc_package.next_tokens[idx])
                    if self.async_streams:
                        self.async_streams[id].put(
                            seq.detokenize_inc(self.model_runner.tokenizer))
                if id in ipc_package.free_ids:
                    self.running_maps.pop(id)
                    if self.async_streams:
                        self.async_streams[id].finish()
                        del self.async_streams[id]
            self.free_finish_ids(ipc_package.free_ids)
            return len(ipc_package.free_ids)
        return 0

    def send_ipc_package(self, log=True):
        if len(self.wait_lists) != 0 or len(self.abort_ids) != 0:
            for seq in self.wait_lists:
                self.running_maps[seq.seq_id] = seq
            ipc_package = IPCPackage(self.wait_lists)
            if len(self.abort_ids) != 0:
                logger.warning(f'Abort {len(self.abort_ids)} request(s) due to loss of network connection')
            ipc_package.abort_ids = self.abort_ids
            ipc_package.log = log
            self.wait_lists = []
            self.abort_ids = []
            self.comm.send_ipc_package(ipc_package)
            
    def schedule(self, log=True):
        self.check_worker_alive()
        num_finish_seqs = self.recv_ipc_package()
        self.send_ipc_package(log)
        return num_finish_seqs

    def check_seq_length(self, token_ids: List[int], output_len: int):
        max_seq_length = len(
            token_ids) + output_len if output_len is not None else len(token_ids)
        if max_seq_length > self.model_max_length:
            logger.warning(
                f'Ignore seq due to the length ({max_seq_length}) exceeds max model len({self.model_max_length})')
            return False
        else:
            return True

    def allocate_seq(self, token_ids: List[int], output_len=None, ignore_eos=False,
                     temperature=None, top_p=None, top_k=None, repetition_penalty=None,
                     mm_contents=None):
        temperature = self.generation_config.temperature if temperature is None else temperature
        top_p = self.generation_config.top_p if top_p is None else top_p
        top_k = 1 if top_k is None else top_k
        repetition_penalty = self.generation_config.repetition_penalty if repetition_penalty is None else repetition_penalty
        return Sequence(self.id_allocator.allocate(), token_ids,
                        self.finish_tokens, output_len, ignore_eos,
                        temperature, top_p, top_k, repetition_penalty,
                        mm_contents)

    def free_finish_ids(self, finish_ids:List[int]):
        for id in finish_ids:
            self.id_allocator.free(id)

    def generate(self, prompts: List[str] = None, tokens: List[List[int]] = None, output_lens: List[int] = None,
                 temperature=None, top_p=None, top_k=None):
        seqs: List[Sequence] = []
        assert prompts is not None or tokens is not None
        num_seqs = len(prompts) if prompts is not None else len(tokens)
        for idx in range(num_seqs):
            token_ids = tokens[idx] if tokens is not None else self.model_runner.encode(prompts[idx])
            output_len_each = output_lens[idx] if output_lens is not None else None
            if self.check_seq_length(token_ids, output_len_each):
                seq = self.allocate_seq(token_ids, output_len_each, False, temperature,
                                        top_p, top_k)
                seqs.append(seq)
        self.add_requests(seqs)

        pbar = tqdm.tqdm(total=len(seqs),ncols=100)
        while pbar.n != len(seqs):
            num_finish_seqs = self.schedule(log=False)
            pbar.update(num_finish_seqs)

        for seq in seqs:
            seq.prompt = self.model_runner.decode(seq[:seq.prompt_len])
            seq.output = self.model_runner.decode(seq[seq.prompt_len:])

        return seqs

    def chat(self):
        architecture = self.model_runner.model_loader.architecture
        print("\nWelcome to the chatbot!\n"
              "Type '\\exit' to exit the chatbot.\n"
              "Type '\\clear' to clear the chatbot's history.\n")
        history = []
        while True:
            prompt = input(">>> ")
            print()
            if prompt == '\\clear':
                history = []
                continue
            elif prompt == '\\exit':
                break

            if architecture == 'ChatGLMModel' and hasattr(self.model_runner.tokenizer, 'build_chat_input'):
                tokens = self.model_runner.tokenizer.build_chat_input(
                    prompt, history=history, role='user').get("input_ids").numpy().tolist()[0]
            else:
                history.append({"role": "user", "content": prompt})
                tokens = self.model_runner.encode(history, chat=True)
            
            seq = self.allocate_seq(tokens)
            self.add_requests([seq])
            while len(self.running_maps) != 0 or len(self.wait_lists) != 0:
                self.schedule(log=False)
                print(seq.detokenize_inc(self.model_runner.tokenizer), end='', flush=True)
            print("\n")
            
            output_text = self.model_runner.decode(seq[seq.prompt_len:])

            if architecture == 'ChatGLMModel' and hasattr(self.model_runner.tokenizer, 'build_chat_input'):
                _, history = self.model_runner.model.process_response(
                    output_text, history)
            else:
                history.append({"role": "assistant", "content": output_text})
