import json
import random
import time
import argparse
from gllm import LLM

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark offline serving throughput')
    parser.add_argument('--model-path',type=str,required=True)
    parser.add_argument('--share-gpt-path',type=str,required=True)
    parser.add_argument('--num-prompt',type=int,default=8)
    args = parser.parse_args()
    
    llm = LLM(args.model_path)
    with open(args.share_gpt_path) as f:
        completions = json.load(f)
        prompts = []
        random.shuffle(completions)
        for completion in completions:
            if len(completion['conversations']) == 0:
                continue
            if completion['conversations'][0]['from'] == 'gpt':
                continue
            if len(llm.model_runner.tokenizer.encode(completion['conversations'][0]['value'])) > 1024:
                continue
            prompts.append(completion['conversations'][0]['value'])
            if len(prompts) == args.num_prompt:
                break
        start = time.time()
        seqs = llm.generate(prompts)
        end = time.time()
        num_input_tokens = 0
        num_output_tokens = 0
        for seq in seqs:
            num_input_tokens += seq.prompt_len
            num_output_tokens += len(seq.token_ids) - seq.prompt_len
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

