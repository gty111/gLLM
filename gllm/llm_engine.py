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
            self.model_runner.model.finish_tokens, page_size)

    def allocate_seq(self, token_ids: List[int], output_len=None, ignore_eos=False, 
                     temperature=0.6, top_p=0.9, top_k=10):
        return Sequence(self.allocatorID.allocate(), token_ids, 
                        self.model_runner.model.finish_tokens, output_len, ignore_eos,
                        temperature, top_p, top_k)

    def add_requests(self, requests: List[Sequence]):
        self.scheduler.add_requests(requests)

    def free_finish_requests(self):
        for seq in self.scheduler.finish_lists:
            self.allocatorID.free(seq.seq_id)
        self.scheduler.finish_lists = []

    def step(self):
        scheduled_seqs = self.scheduler.schedule(self.model_runner.memory_manager.get_num_free_pages())
        self.model_runner.step_once(scheduled_seqs)
        self.scheduler.update_seqs(scheduled_seqs)

    def generate(self, prompts: List[str] = None, tokens: List[List[int]] = None, output_lens: List[int] = None, 
                 temperature=0.6, top_p=0.9, top_k=10):
        requests: List[Sequence] = []
        assert prompts is not None or tokens is not None
        num_seqs = len(prompts) if prompts is not None else len(tokens)
        for idx in range(num_seqs):
            token_ids = tokens[idx] if tokens is not None else self.model_runner.tokenizer.encode(prompts[idx])
            output_len_each = output_lens[idx] if output_lens is not None else None
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
            output_text = self.model_runner.stream_inference(seq)

            if architecture == 'ChatGLMModel' and hasattr(self.model_runner.tokenizer, 'build_chat_input'):
                _, history = self.model_runner.model.process_response(
                    output_text, history)
            else:
                history.append({"role": "assistant", "content": output_text})
