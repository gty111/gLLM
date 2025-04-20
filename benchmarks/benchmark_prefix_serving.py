"""Benchmark online serving throughput.

On the server side, run one of the following commands:
    (vLLM backend)
    python -m vllm.entrypoints.api_server \
        --model <your_model> --swap-space 16 \
        --disable-log-requests

    (TGI backend)
    ./launch_tgi_server.sh <your_model> <max_batch_total_tokens>

On the client side, run:
    python benchmarks/benchmark_serving.py \
        --backend <backend> \
        --model <your_model> --dataset <target_dataset> \
        --num-min-rounds <min_rounds> --num-max-rounds <max_rounds> \
        --num-max-users <max_users>
"""
import argparse
import asyncio
import json
import random
import time
from dataclasses import dataclass
from datetime import datetime
from typing import AsyncGenerator, List, Optional, Tuple
import warnings

import numpy as np
from tqdm.asyncio import tqdm
from transformers import PreTrainedTokenizerBase
try:
    from vllm.transformers_utils.tokenizer import get_tokenizer
except ImportError:
    from backend_request_func import get_tokenizer

from backend_request_func import (
    ASYNC_REQUEST_FUNCS,
    RequestFuncInput,
    RequestFuncOutput,
)


@dataclass
class BenchmarkMetrics:
    completed: int
    total_input: int
    total_output: int
    request_throughput: float
    input_throughput: float
    output_throughput: float
    mean_ttft_ms: float
    median_ttft_ms: float
    std_ttft_ms: float
    p99_ttft_ms: float
    mean_tpot_ms: float
    median_tpot_ms: float
    std_tpot_ms: float
    p99_tpot_ms: float
    mean_itl_ms: float
    median_itl_ms: float
    std_itl_ms: float
    p99_itl_ms: float
    avg_latency_ms: float
    
    
def sample_requests(
    dataset_path: str,
    num_min_rounds: int,
    num_max_rounds: int,
    num_max_users: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: Optional[int],
) -> List[Tuple[str, int, int, int, int]]:
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")

    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    
    assert num_min_rounds <= num_max_rounds

    dataset = [data for data in dataset if len(
        data["conversations"]) >= num_min_rounds*2]
    dataset = [data for data in dataset if data[
        "conversations"][0]["from"] == "human"]
    
    filtered_dataset = []
    user_id = 1
    for i in dataset:
        history = []
        epoch = 0
        for chat in i["conversations"]:
            if chat["from"] == "human":
                history.append({"role":"user","content":chat["value"]})
            elif chat["from"] == "gpt":
                if len(history) != 0:
                    # if len(tokenizer(gen_prompt(history)).input_ids) > 4096:
                    #     break
                    prompt_tokens = tokenizer.apply_chat_template(history,add_generation_prompt=True)
                    prompt = tokenizer.decode(prompt_tokens,True)
                    epoch += 1
                    if fixed_output_len:
                        filtered_dataset.append((prompt,
                                            len(tokenizer(prompt).input_ids),
                                            fixed_output_len,
                                            user_id,
                                            epoch))
                    else:
                        filtered_dataset.append((prompt,
                                            len(tokenizer(prompt).input_ids),
                                            len(tokenizer(chat['value']).input_ids),
                                            user_id,
                                            epoch))
                history.append({"role":"gpt","content":chat["value"]})
            if epoch == num_max_rounds:
                break
        user_id += 1
        if user_id-1 == num_max_users:
            break
    users = min(user_id,num_max_users)
    epochs = len(filtered_dataset) // users
    print(f"#users : {min(user_id,num_max_users)}, #epochs : {epochs}")
    return filtered_dataset


def calculate_metrics(
    input_requests: List[Tuple[str, int, int]],
    outputs: List[RequestFuncOutput],
    dur_s: float,
    tokenizer: PreTrainedTokenizerBase,
) -> Tuple[BenchmarkMetrics, List[int]]:
    actual_output_lens: List[int] = []
    total_input = 0
    completed = 0
    itls: List[float] = []
    tpots: List[float] = []
    ttfts: List[float] = []
    latencys: List[float] = []
    for i in range(len(outputs)):
        if outputs[i].success:
            # We use the tokenizer to count the number of output tokens for all
            # serving backends instead of looking at len(outputs[i].itl) since
            # multiple output tokens may be bundled together
            # Note : this may inflate the output token count slightly
            output_len = len(
                tokenizer(outputs[i].generated_text,
                          add_special_tokens=False).input_ids)
            actual_output_lens.append(output_len)
            total_input += input_requests[i][1]
            if output_len > 1:
                tpots.append(
                    (outputs[i].latency - outputs[i].ttft) / (output_len - 1))
            itls += outputs[i].itl
            ttfts.append(outputs[i].ttft)
            latencys.append(outputs[i].latency)
            completed += 1
        else:
            actual_output_lens.append(0)

    if completed == 0:
        warnings.warn(
            "All requests failed. This is likely due to a misconfiguration "
            "on the benchmark arguments.",
            stacklevel=2)
    metrics = BenchmarkMetrics(
        completed=completed,
        total_input=total_input,
        total_output=sum(actual_output_lens),
        request_throughput=completed / dur_s,
        input_throughput=total_input / dur_s,
        output_throughput=sum(actual_output_lens) / dur_s,
        mean_ttft_ms=np.mean(ttfts or 0) *
        1000,  # ttfts is empty if streaming is not supported by backend
        median_ttft_ms=np.median(ttfts or 0) * 1000,
        std_ttft_ms=np.std(ttfts or 0) * 1000,
        p99_ttft_ms=np.percentile(ttfts or 0, 99) * 1000,
        mean_tpot_ms=np.mean(tpots or 0) * 1000,
        median_tpot_ms=np.median(tpots or 0) * 1000,
        std_tpot_ms=np.std(tpots or 0) * 1000,
        p99_tpot_ms=np.percentile(tpots or 0, 99) * 1000,
        mean_itl_ms=np.mean(itls or 0) * 1000,
        median_itl_ms=np.median(itls or 0) * 1000,
        std_itl_ms=np.std(itls or 0) * 1000,
        p99_itl_ms=np.percentile(itls or 0, 99) * 1000,
        avg_latency_ms=np.mean(latencys) * 1000,
    )

    return metrics, actual_output_lens

async def user_request(request_func,
                       model_id,
                       api_url,
                       best_of,
                       use_beam_search,
                       pbar,
                       requests,):
    cur_task = None
    outputs = []
    for request in requests:
        if cur_task is not None:
            # wait for the last request finished
            outputs.append(await cur_task)
            # assume the interval between requests is 5s
            # await asyncio.sleep(5)
        request_func_input = RequestFuncInput(
            model=model_id,
            prompt=request[0],
            api_url=api_url,
            prompt_len=request[1],
            output_len=request[2],
            best_of=best_of,
        )
        cur_task = asyncio.create_task(
            request_func(request_func_input=request_func_input,
                         pbar=pbar))
    outputs.append(await cur_task)
    return outputs

async def benchmark(
    backend: str,
    api_url: str,
    model_id: str,
    tokenizer: PreTrainedTokenizerBase,
    input_requests: List[Tuple[str, int, int, int, int]],
    best_of: int,
    use_beam_search: bool,
    disable_tqdm: bool,
):
    if backend in ASYNC_REQUEST_FUNCS:
        request_func = ASYNC_REQUEST_FUNCS.get(backend)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    pbar = None if disable_tqdm else tqdm(total=len(input_requests))
    
    # rearrange request by its users
    user_id = 1
    request_rearrange = []
    while True:
        request_users = []
        for request in input_requests:
            if request[3] == user_id:
                request_users.append(request)
        if len(request_users) == 0:
            break
        request_rearrange.append(request_users)
        user_id += 1

    benchmark_start_time = time.perf_counter()
    tasks = []
    for request_epoch in request_rearrange:
        tasks.append(asyncio.create_task(user_request(
            request_func=request_func,
            model_id=model_id,
            api_url=api_url,
            best_of=best_of,
            use_beam_search=use_beam_search,
            pbar=pbar,
            requests=request_epoch,
        )))
        # await asyncio.sleep(1)
        
    output_all = await asyncio.gather(*tasks)
    outputs = []
    for output_each in output_all:
        outputs.extend(output_each)

    if not disable_tqdm:
        pbar.close()

    benchmark_duration = time.perf_counter() - benchmark_start_time

    metrics, actual_output_lens = calculate_metrics(
        input_requests=input_requests,
        outputs=outputs,
        dur_s=benchmark_duration,
        tokenizer=tokenizer,
    )

    print("{s:{c}^{n}}".format(s=' Serving Benchmark Result ', n=50, c='='))
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):",
                                    benchmark_duration))
    print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
    print("{:<40} {:<10}".format("Total generated tokens:",
                                 metrics.total_output))
    print("{:<40} {:<10.2f}".format("Request throughput (req/s):",
                                    metrics.request_throughput))
    print("{:<40} {:<10.2f}".format("Input token throughput (tok/s):",
                                    metrics.input_throughput))
    print("{:<40} {:<10.2f}".format("Output token throughput (tok/s):",
                                    metrics.output_throughput))
    print("{:<40} {:<10.2f}".format("Avg latency (ms):", metrics.avg_latency_ms))
    print("{s:{c}^{n}}".format(s='Time to First Token', n=50, c='-'))
    print("{:<40} {:<10.2f}".format("Mean TTFT (ms):", metrics.mean_ttft_ms))
    print("{:<40} {:<10.2f}".format("Median TTFT (ms):",
                                    metrics.median_ttft_ms))
    print("{:<40} {:<10.2f}".format("P99 TTFT (ms):", metrics.p99_ttft_ms))
    print("{s:{c}^{n}}".format(s='Time per Output Token (excl. 1st token)',
                               n=50,
                               c='-'))
    print("{:<40} {:<10.2f}".format("Mean TPOT (ms):", metrics.mean_tpot_ms))
    print("{:<40} {:<10.2f}".format("Median TPOT (ms):",
                                    metrics.median_tpot_ms))
    print("{:<40} {:<10.2f}".format("P99 TPOT (ms):", metrics.p99_tpot_ms))
    print("{s:{c}^{n}}".format(s='Inter-token Latency', n=50, c='-'))
    print("{:<40} {:<10.2f}".format("Mean ITL (ms):", metrics.mean_itl_ms))
    print("{:<40} {:<10.2f}".format("Median ITL (ms):", metrics.median_itl_ms))
    print("{:<40} {:<10.2f}".format("P99 ITL (ms):", metrics.p99_itl_ms))
    print("=" * 50)

    result = {
        "duration": benchmark_duration,
        "completed": metrics.completed,
        "total_input_tokens": metrics.total_input,
        "total_output_tokens": metrics.total_output,
        "request_throughput": metrics.request_throughput,
        "input_throughput": metrics.input_throughput,
        "output_throughput": metrics.output_throughput,
        "mean_ttft_ms": metrics.mean_ttft_ms,
        "median_ttft_ms": metrics.median_ttft_ms,
        "std_ttft_ms": metrics.std_ttft_ms,
        "p99_ttft_ms": metrics.p99_ttft_ms,
        "mean_tpot_ms": metrics.mean_tpot_ms,
        "median_tpot_ms": metrics.median_tpot_ms,
        "std_tpot_ms": metrics.std_tpot_ms,
        "p99_tpot_ms": metrics.p99_tpot_ms,
        "mean_itl_ms": metrics.mean_itl_ms,
        "median_itl_ms": metrics.median_itl_ms,
        "std_itl_ms": metrics.std_itl_ms,
        "p99_itl_ms": metrics.p99_itl_ms,
        "input_lens": [output.prompt_len for output in outputs],
        "output_lens": actual_output_lens,
        "ttfts": [output.ttft for output in outputs],
        "itls": [output.itl for output in outputs],
        "generated_texts": [output.generated_text for output in outputs],
        "errors": [output.error for output in outputs],
    }
    return result


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    backend = args.backend
    model_id = args.model
    tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model

    if args.base_url is not None:
        api_url = f"{args.base_url}{args.endpoint}"
    else:
        api_url = f"http://{args.host}:{args.port}{args.endpoint}"
    tokenizer = get_tokenizer(tokenizer_id,
                              trust_remote_code=args.trust_remote_code)
    input_requests = sample_requests(args.dataset, 
                                     args.num_min_rounds, 
                                     args.num_max_rounds, 
                                     args.num_max_users,
                                     tokenizer,
                                     args.output_len,
                                     )
    
    benchmark_result = asyncio.run(
        benchmark(
            backend=backend,
            api_url=api_url,
            model_id=model_id,
            tokenizer=tokenizer,
            input_requests=input_requests,
            best_of=args.best_of,
            use_beam_search=args.use_beam_search,
            disable_tqdm=args.disable_tqdm,
        ))

    # Save config and results to json
    if args.save_result:
        result_json = {}

        # Setup
        current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
        result_json["date"] = current_dt
        result_json["backend"] = backend
        result_json["version"] = args.version
        result_json["model_id"] = model_id
        result_json["tokenizer_id"] = tokenizer_id
        result_json["best_of"] = args.best_of
        result_json["use_beam_search"] = args.use_beam_search

        # Merge with benchmark result
        result_json = {**result_json, **benchmark_result}

        # Save to file
        base_model_id = model_id.split("/")[-1]
        file_name = f"{backend}-{args.num_min_rounds}-{args.num_max_rounds}-{args.num_max_users}-{base_model_id}-{current_dt}.json"
        with open(file_name, "w") as outfile:
            json.dump(result_json, outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the online serving throughput.")
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=list(ASYNC_REQUEST_FUNCS.keys()),
    )
    parser.add_argument(
        "--version",
        type=str,
        default="N/A",
        help="Version of the serving backend/engine.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Server or API base url if not using http host and port.",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--endpoint",
        type=str,
        default="/v1/completions",
        help="API endpoint.",
    )
    parser.add_argument("--dataset",
                        type=str,
                        required=True,
                        help="Path to the dataset.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the model.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help=
        "Name or path of the tokenizer, if not using the default model tokenizer.",
    )
    parser.add_argument(
        "--best-of",
        type=int,
        default=1,
        help="Generates `best_of` sequences per prompt and "
        "returns the best one.",
    )
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code from huggingface",
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Specify to disable tqdm progress bar.",
    )
    parser.add_argument(
        "--save-result",
        action="store_true",
        help="Specify to save benchmark results to a json file",
    )
    parser.add_argument("--num-max-users",
                        type=int,
                        default=128,
                        help="Max number of users")
    parser.add_argument("--num-max-rounds",
                        type=int,
                        default=100,
                        help="Max number of rounds at least in one conversation")
    parser.add_argument("--num-min-rounds",
                        type=int,
                        default=50,
                        help="Min number of rounds at least in one conversation")
    parser.add_argument("--output-len",
                        type=int,
                        default=None,
                        help="Specific length of the output")
    args = parser.parse_args()
    main(args)