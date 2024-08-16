import tqdm
from typing import List

from gllm.model_runner import ModelRunner
from gllm.sequence import Sequence
from gllm.allocatorID import AllocatorID
from gllm.scheduler import Scheduler


class LLM():
    def __init__(self, model_path, gpu_memory_utilization=0.9, page_size=16, max_decode_seqs=256, max_batch_tokens=8192, ratio_threshold_free_pages=0.2):
        self.model_path = model_path
        self.model_runner = ModelRunner(
            model_path, gpu_memory_utilization, page_size)
        self.allocatorID = AllocatorID(0, 99999)
        self.scheduler = Scheduler(
            max_decode_seqs, max_batch_tokens, ratio_threshold_free_pages,
            self.model_runner.memory_manager.get_num_free_pages(),
            self.model_runner.model.finish_tokens)

    def allocate_seq(self, token_ids: List[int], output_len=None, ignore_eos=False):
        return Sequence(self.allocatorID.allocate(), token_ids, 
                        self.model_runner.model.finish_tokens, output_len, ignore_eos)

    def add_requests(self, requests: List[Sequence]):
        self.scheduler.add_requests(requests)

    def free_finish_requests(self):
        for seq in self.scheduler.finish_lists:
            self.allocatorID.free(seq.seq_id)
        self.scheduler.finish_lists = []

    def step(self, temperature, top_p):
        scheduled_seqs = self.scheduler.schedule(self.model_runner.memory_manager.get_num_free_pages())
        self.model_runner.step_once(scheduled_seqs, temperature, top_p)
        self.scheduler.update_seqs(scheduled_seqs)

    def generate(self, prompts: List[str] = None, tokens: List[List[int]] = None, output_lens: List[int] = None, temperature=0.6, top_p=0.9):
        requests: List[Sequence] = []
        if tokens is not None:
            for idx, token_ids in enumerate(tokens):
                if output_lens is not None:
                    seq = self.allocate_seq(token_ids, output_lens[idx])
                else:
                    seq = self.allocate_seq(token_ids)
                requests.append(seq)
        elif prompts is not None:
            for idx, prompt in enumerate(prompts):
                token_ids = self.model_runner.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}])
                if output_lens is not None:
                    seq = self.allocate_seq(token_ids, output_lens[idx])
                else:
                    seq = self.allocate_seq(token_ids)
                requests.append(seq)
        else:
            return None
        self.add_requests(requests)

        pbar = tqdm.tqdm(total=len(requests))
        while len(self.scheduler.finish_lists) != len(requests):
            cur_finish_num = len(self.scheduler.finish_lists)
            self.step(temperature, top_p)
            pbar.update(len(self.scheduler.finish_lists)-cur_finish_num)

        for request in requests:
            request.prompt = self.model_runner.tokenizer.decode(
                request.token_ids[:request.prompt_len], True, True)
            request.output = self.model_runner.tokenizer.decode(
                request.token_ids[request.prompt_len:], True, True)

        self.free_finish_requests()
        return requests

    def chat(self, temperature=0.6, top_p=0.9):
        architecture = self.model_runner.model.model_config['architectures'][0]
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
            output_text = self.model_runner.stream_inference(
                seq, temperature, top_p)

            if architecture == 'ChatGLMModel' and hasattr(self.model_runner.tokenizer, 'build_chat_input'):
                _, history = self.model_runner.model.process_response(
                    output_text, history)
            else:
                history.append({"role": "assistant", "content": output_text})
