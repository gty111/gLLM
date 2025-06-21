import tqdm

from logger import logger
from typing import List

from gllm.model_runner import ModelRunner
from gllm.sequence import Sequence
from gllm.id_allocator import IDAllocator
from gllm.frontend_scheduler import FrontendScheduler
from gllm.input_data import InputData
from gllm.utils import init_logger


class LLM():
    def __init__(self, model_path, host=None, master_addr=None, master_port=None, 
                 zmq_port_base=None, launch_mode=None, worker_ranks=None, load_format='auto', 
                 gpu_memory_util=0.9, page_size=16, maxd=256,maxp=2048, minp=32, 
                 iterp=8, kvthresh=0.05, enable_prefix_caching=True, pp_size=1, tp_size=1, use_ep=True,
                 assigned_layers=None, use_naive_schedule=False, use_async_worker=False, use_thinking=True):
        init_logger()
        self.model_path = model_path
        self.model_runner = ModelRunner(
            load_format, model_path, gpu_memory_util, page_size, enable_prefix_caching, use_thinking,
            maxp, maxd, kvthresh, minp, iterp)
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
        self.scheduler = FrontendScheduler(
            maxd, maxp, kvthresh, page_size)
        self.finish_tokens = self.model_runner.model_loader.generation_config.eos_token_id
        if type(self.finish_tokens) == int:
            self.finish_tokens = [self.finish_tokens]
        self.model_max_length = self.model_runner.tokenizer.model_max_length
        self.generation_config = self.model_runner.model_loader.generation_config
        
        self.assigned_layers = assigned_layers
        self.use_naive_schedule = use_naive_schedule
        self.use_async_worker = use_async_worker

    def check_seq_length(self, token_ids: List[int], output_len: int):
        max_seq_length = len(
            token_ids) + output_len if output_len is not None else len(token_ids)
        if max_seq_length > self.model_max_length:
            logger.warning(
                f'Ignore seq due to the length({max_seq_length}) exceeds max model len({self.model_runner.model.max_model_len})')
            return False
        else:
            return True

    def allocate_seq(self, token_ids: List[int], output_len=None, ignore_eos=False,
                     temperature=None, top_p=None, top_k=None, repetition_penalty=None):
        temperature = self.generation_config.temperature if temperature is None else temperature
        top_p = self.generation_config.top_p if top_p is None else top_p
        top_k = self.generation_config.top_k if top_k is None else top_k
        repetition_penalty = self.generation_config.repetition_penalty if repetition_penalty is None else repetition_penalty
        return Sequence(self.id_allocator.allocate(), token_ids,
                        self.finish_tokens, output_len, ignore_eos,
                        temperature, top_p, top_k, repetition_penalty)

    def add_requests(self, requests: List[Sequence]):
        self.scheduler.add_requests(requests)

    def free_finish_ids(self, finish_ids:List[int]):
        for id in finish_ids:
            self.id_allocator.free(id)

    def init(self):
        self.model_runner.init()
        self.scheduler.set_total_num_free_pages(self.model_runner.memory_manager.get_num_free_pages())

    def step(self):
        if self.model_runner.model is None:
            self.init()
        scheduleOutput = self.scheduler.schedule(self.model_runner.memory_manager)
        next_tokens = self.model_runner.step_once(InputData(scheduleOutput.schedule_lists, self.model_runner.memory_manager))
        self.scheduler.update_seqs(scheduleOutput, next_tokens, self.model_runner.memory_manager)

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
        while len(self.scheduler.finish_ids) != len(seqs):
            cur_finish_num = len(self.scheduler.finish_ids)
            self.step()
            pbar.update(len(self.scheduler.finish_ids)-cur_finish_num)

        for seq in seqs:
            seq.prompt = self.model_runner.decode(seq[:seq.prompt_len])
            seq.output = self.model_runner.decode(seq[seq.prompt_len:])

        self.free_finish_ids(self.scheduler.get_finish_ids())
        return seqs

    def chat(self):
        self.model_runner.init()
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
            output_text = self.model_runner.stream_inference(seq)

            if architecture == 'ChatGLMModel' and hasattr(self.model_runner.tokenizer, 'build_chat_input'):
                _, history = self.model_runner.model.process_response(
                    output_text, history)
            else:
                history.append({"role": "assistant", "content": output_text})
