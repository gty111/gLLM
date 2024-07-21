import tqdm
from typing import List

from gllm.model_runner import ModelRunner
from gllm.sequence import Sequence
from gllm.allocatorID import AllocatorID
from gllm.scheduler import Scheduler


class LLM():
    def __init__(self, model_path):
        self.model_runner = ModelRunner(model_path)
        self.allocatorID = AllocatorID(1000, 9999)
        self.scheduler = Scheduler(self.model_runner)

    def allocate_seq(self, token_ids: List[int]):
        return Sequence(self.allocatorID.allocate(), token_ids)

    def add_requests(self, requests: List[Sequence]):
        self.scheduler.add_requests(requests)

    def free_requests(self, requests: List[Sequence]):
        for seq in requests:
            del self.scheduler.finish_lists[seq.seq_id]
            self.allocatorID.free(seq.seq_id)

    def step(self):
        self.model_runner.step_once(self.scheduler.schedule())

    def generate(self, prompts: List[Sequence] = None):
        requests: List[Sequence] = []
        for prompt in prompts:
            token_ids = self.model_runner.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}])
            seq = self.allocate_seq(token_ids)
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

        self.free_requests(requests)
        return requests

    def chat(self):
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
            history.append({"role": "user", "content": prompt})
            tokens = self.model_runner.tokenizer.apply_chat_template(history)
            seq = self.allocate_seq(tokens)
            output_text = self.model_runner.inference(seq)
            history.append({"role": "assistant", "content": output_text})
            # print(f'Answer: {output_text}')
