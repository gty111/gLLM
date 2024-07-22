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

    def allocate_seq(self, token_ids: List[int], output_len=None):
        return Sequence(self.allocatorID.allocate(), token_ids, output_len)

    def add_requests(self, requests: List[Sequence]):
        self.scheduler.add_requests(requests)

    def free_requests(self, requests: List[Sequence]):
        for seq in requests:
            del self.scheduler.finish_lists[seq.seq_id]
            self.allocatorID.free(seq.seq_id)

    def step(self):
        scheduled_seqs = self.scheduler.schedule()
        self.model_runner.step_once(scheduled_seqs)

    def generate(self, prompts: List[str] = None, tokens: List[List[int]] = None, output_lens: List[int] = None):
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
            output_text = self.model_runner.stream_inference(seq)
            history.append({"role": "assistant", "content": output_text})
            # print(f'Answer: {output_text}')
