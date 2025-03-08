import tqdm

from logger import logger
from typing import List

from gllm.model_runner import ModelRunner
from gllm.sequence import Sequence
from gllm.allocatorID import AllocatorID
from gllm.scheduler import Scheduler
from gllm.dist_utils import init_dist
from gllm.input_data import InputData


class LLM():
    def __init__(self, model_path, load_format='auto', gpu_memory_util=0.9, page_size=16, max_decode_seqs=256,
                 max_batch_tokens=8192, ratio_threshold_free_pages=0.2, enable_prefix_caching=True, pp_size=1):
        self.model_path = model_path
        self.model_runner = ModelRunner(
            load_format, model_path, gpu_memory_util, page_size, enable_prefix_caching)
        self.pp_size = pp_size
        self.master_addr = '127.0.0.1'
        self.master_port = '49082'
        self.allocatorID = AllocatorID(0, 99999)
        self.scheduler = Scheduler(
            max_decode_seqs, max_batch_tokens, ratio_threshold_free_pages, page_size, pp_size)
        self.finish_tokens = self.model_runner.model_loader.get_finish_tokens()
        self.model_max_length = self.model_runner.tokenizer.model_max_length

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
                     temperature=0.6, top_p=0.9, top_k=10):
        return Sequence(self.allocatorID.allocate(), token_ids,
                        self.finish_tokens, output_len, ignore_eos,
                        temperature, top_p, top_k)

    def add_requests(self, requests: List[Sequence]):
        self.scheduler.add_requests(requests)

    def free_finish_requests(self):
        for seq in self.scheduler.finish_lists:
            self.allocatorID.free(seq.seq_id)
        self.scheduler.finish_lists = []

    def init(self):
        self.model_runner.init()
        self.scheduler.set_total_num_free_pages(self.model_runner.memory_manager.get_num_free_pages())

    def step(self):
        if self.model_runner.model is None:
            self.init()
        scheduleOutput = self.scheduler.schedule(
            self.model_runner.memory_manager.get_num_free_pages())
        next_tokens = self.model_runner.step_once(InputData(scheduleOutput.schedule_lists, self.model_runner.memory_manager))
        self.scheduler.update_seqs(scheduleOutput, next_tokens)

    def generate(self, prompts: List[str] = None, tokens: List[List[int]] = None, output_lens: List[int] = None,
                 temperature=0.6, top_p=0.9, top_k=10):
        requests: List[Sequence] = []
        assert prompts is not None or tokens is not None
        num_seqs = len(prompts) if prompts is not None else len(tokens)
        for idx in range(num_seqs):
            token_ids = tokens[idx] if tokens is not None else self.model_runner.tokenizer.encode(
                prompts[idx])
            output_len_each = output_lens[idx] if output_lens is not None else None
            if self.check_seq_length(token_ids, output_len_each):
                seq = self.allocate_seq(token_ids, output_len_each, False, temperature,
                                        top_p, top_k)
                requests.append(seq)
        self.add_requests(requests)

        pbar = tqdm.tqdm(total=len(requests))
        while len(self.scheduler.finish_lists) != len(requests):
            cur_finish_num = len(self.scheduler.finish_lists)
            self.step()
            pbar.update(len(self.scheduler.finish_lists)-cur_finish_num)

        for request in requests:
            request.prompt = self.model_runner.tokenizer.decode(
                request.token_ids[:request.prompt_len], True, True)
            request.output = self.model_runner.tokenizer.decode(
                request.token_ids[request.prompt_len:], True, True)

        self.free_finish_requests()
        return requests

    def chat(self):
        self.model_runner.init()
        architecture = self.model_runner.model_loader.architecture
        print("\nWelcome to the chatbot!\n"
              "Type '\exit' to exit the chatbot.\n"
              "Type '\clear' to clear the chatbot's history.\n")
        history = []
        while True:
            prompt = input(">>> ")
            if prompt == '\clear':
                history = []
                continue
            elif prompt == '\exit':
                break

            if architecture == 'ChatGLMModel' and hasattr(self.model_runner.tokenizer, 'build_chat_input'):
                tokens = self.model_runner.tokenizer.build_chat_input(
                    prompt, history=history, role='user').get("input_ids").numpy().tolist()[0]
            else:
                history.append({"role": "user", "content": prompt})
                tokens = self.model_runner.tokenizer.apply_chat_template(
                    history, add_generation_prompt=True)
            seq = self.allocate_seq(tokens)
            output_text = self.model_runner.stream_inference(seq)

            if architecture == 'ChatGLMModel' and hasattr(self.model_runner.tokenizer, 'build_chat_input'):
                _, history = self.model_runner.model.process_response(
                    output_text, history)
            else:
                history.append({"role": "assistant", "content": output_text})
