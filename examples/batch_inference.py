import json
import random
import time
import argparse
from gllm import LLM

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Benchmark offline serving throughput')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--sharegpt-path', type=str, required=True)
    parser.add_argument('--num-prompt', type=int, default=8)
    parser.add_argument('--print-output', action="store_true")
    parser.add_argument('--gpu-memory-util', type=float, default=0.9)
    args = parser.parse_args()

    llm = LLM(args.model, gpu_memory_util=args.gpu_memory_util)
    llm.init()
    with open(args.sharegpt_path) as f:
        completions = json.load(f)
        tokens = []
        output_lens = []
        random.shuffle(completions)
        for completion in completions:
            if len(completion['conversations']) < 2:
                continue
            if completion['conversations'][0]['from'] == 'gpt':
                continue
            prompt = completion['conversations'][0]['value']
            answer = completion['conversations'][1]['value']
            tokens_each = llm.model_runner.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], add_generation_prompt=True)
            input_len = len(tokens_each)
            output_len = len(llm.model_runner.tokenizer.apply_chat_template(
                [{"role": "assistant", "content": answer}]))
            if input_len > 1024 or input_len+output_len > 2048:
                continue

            tokens.append(tokens_each)
            output_lens.append(output_len)
            if len(tokens) == args.num_prompt:
                break
        start = time.time()
        seqs = llm.generate(tokens=tokens, output_lens=output_lens)
        end = time.time()
        num_input_tokens = 0
        num_output_tokens = 0
        for seq in seqs:
            num_input_tokens += seq.prompt_len
            num_output_tokens += len(seq) - seq.prompt_len
            if args.print_output:
                print('*'*10)
                print(f'prompt:\n{seq.prompt}')
                print('-'*10)
                print(f'Answer:\n{seq.output}')
        print()
        print(f'[Throughput(reqs/s)]: {round(len(seqs)/(end-start),2)}')
        print(
            f'[Input tokens throughput(toks/s)]: {round(num_input_tokens/(end-start),2)}')
        print(
            f'[Output tokens throughput(toks/s)]: {round(num_output_tokens/(end-start),2)}')
