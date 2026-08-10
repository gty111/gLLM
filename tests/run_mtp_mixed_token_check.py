"""Drive a deterministic two-wave workload and compare generated token ids."""

import argparse
import concurrent.futures
import json
import time
import urllib.request

from transformers import AutoTokenizer


PHASE_A = [
    "Write a numbered list of 40 facts about the solar system. Item one:",
    "Continue this sequence with explanations: 1, 1, 2, 3, 5, 8,",
    "Explain dynamic programming carefully in ten short paragraphs.",
    "Write the integers from 100 to 200, separated by commas: 100,",
]
PHASE_B = [
    "The capital of France is",
    "Count from one to thirty using English words:",
    "List the first twenty prime numbers:",
    "Complete the sentence: Water freezes at",
]


def complete(port: int, prompt: str, max_tokens: int) -> str:
    payload = json.dumps(
        {
            "model": "test",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "top_p": 1,
        }
    ).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)["choices"][0]["text"]


def workload(port: int, wave_delay: float):
    started = time.perf_counter()
    outputs = [None] * (len(PHASE_A) + len(PHASE_B))
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(outputs)) as pool:
        first = [pool.submit(complete, port, prompt, 64) for prompt in PHASE_A]
        # Admit the second wave only after the first cohort is decoding.  This
        # is the deterministic mixed verify-prefix + prefill-suffix boundary.
        time.sleep(wave_delay)
        second = [
            pool.submit(complete, port, prompt, 64) for prompt in PHASE_B
        ]
        for i, future in enumerate(first + second):
            outputs[i] = future.result()
    return outputs, time.perf_counter() - started


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-port", type=int, required=True)
    parser.add_argument("--mixed-port", type=int, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--save", required=True)
    parser.add_argument("--wave-delay", type=float, default=0.03)
    args = parser.parse_args()

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        reference_future = pool.submit(
            workload, args.reference_port, args.wave_delay
        )
        mixed_future = pool.submit(workload, args.mixed_port, args.wave_delay)
        reference, reference_seconds = reference_future.result()
        mixed, mixed_seconds = mixed_future.result()

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    comparisons = []
    all_equal = True
    for index, (reference_text, mixed_text) in enumerate(zip(reference, mixed)):
        reference_ids = tokenizer.encode(reference_text, add_special_tokens=False)
        mixed_ids = tokenizer.encode(mixed_text, add_special_tokens=False)
        first_difference = next(
            (
                i
                for i, (left, right) in enumerate(zip(reference_ids, mixed_ids))
                if left != right
            ),
            None,
        )
        equal = reference_ids == mixed_ids
        if not equal and first_difference is None:
            first_difference = min(len(reference_ids), len(mixed_ids))
        all_equal &= equal
        comparisons.append(
            {
                "request": index,
                "equal": equal,
                "first_difference": first_difference,
                "reference_ids": reference_ids,
                "mixed_ids": mixed_ids,
                "reference_text": reference_text,
                "mixed_text": mixed_text,
            }
        )

    result = {
        "all_equal": all_equal,
        "reference_seconds": reference_seconds,
        "mixed_seconds": mixed_seconds,
        "comparisons": comparisons,
    }
    with open(args.save, "w", encoding="utf-8") as output:
        json.dump(result, output, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "comparisons"}))
    for comparison in comparisons:
        print(
            f"request={comparison['request']} equal={comparison['equal']} "
            f"first_difference={comparison['first_difference']} "
            f"reference_tokens={len(comparison['reference_ids'])} "
            f"mixed_tokens={len(comparison['mixed_ids'])}"
        )
    raise SystemExit(0 if all_equal else 1)


if __name__ == "__main__":
    main()
