"""Online RULER long-context retrieval eval (OpenAI-compatible chat endpoint).

RULER (https://huggingface.co/datasets/simonjegou/ruler) ships pre-generated
buckets at 4096 / 8192 / 16384 tokens. Each row is a synthetic long-context
retrieval task:

    context + question + answer_prefix  ->  model completes  ->  the completion
    must contain one of the acceptable ``answer`` strings.

Because every bucket length exceeds a typical sparse-attention budget
(e.g. DeepSeek-V3.2's ``index_topk`` = 2048), this is a good end-to-end check
that long-context attention (sparse or dense) still retrieves the right tokens.

Modeled on ``evaluate_mmlu_pro.py`` (same async OpenAI-chat client). Point it at
a running server and a local RULER parquet directory:

    python benchmarks/evaluate_ruler.py \
        --model /path/to/DeepSeek-V3.2 --port 8100 \
        --length 4096 --num-per-task 50 --concurrency 32 \
        --data-path /path/to/hf_cache/ruler

``--data-path`` expects the HF layout ``<data-path>/<length>/test-*.parquet``.
"""

import argparse
import asyncio
import os

import pandas as pd
from backend_request_func import (
    RequestFuncInput,
    async_request_openai_chat_completions,
)
from tqdm import tqdm

API_KEY = "EMPTY"


def _acceptable_answers(answer) -> list[str]:
    """RULER's ``answer`` is a list/ndarray of acceptable answer strings."""
    if isinstance(answer, (list, tuple)):
        return [str(x) for x in answer]
    try:
        return [str(x) for x in answer.tolist()]
    except AttributeError:
        return [str(answer)]


def _find_length_parquet(data_path: str, length: int) -> str:
    """Locate ``<data-path>/<length>/test-*.parquet``."""
    d = os.path.join(data_path, str(length))
    if not os.path.isdir(d):
        raise FileNotFoundError(
            f"RULER length dir not found: {d} (expected <data-path>/<length>/)."
        )
    for name in sorted(os.listdir(d)):
        if name.startswith("test") and name.endswith(".parquet"):
            return os.path.join(d, name)
    raise FileNotFoundError(f"No test-*.parquet under {d}.")


def load_ruler(data_path: str, length: int, num_per_task: int) -> list[dict]:
    df = pd.read_parquet(_find_length_parquet(data_path, length))
    rows: list[dict] = []
    for _, group in df.groupby("task"):
        rows.extend(group.head(num_per_task).to_dict("records"))
    return rows, df["task"].nunique()


async def evaluate(args):
    rows, n_tasks = load_ruler(args.data_path, args.length, args.num_per_task)
    print(
        f"RULER length={args.length}: {len(rows)} rows across {n_tasks} tasks "
        f"(num_per_task={args.num_per_task})",
        flush=True,
    )

    api_url = f"http://{args.host}:{args.port}/v1/chat/completions"
    sem = asyncio.Semaphore(args.concurrency)
    pbar = tqdm(total=len(rows))

    async def run_one(row):
        prompt = f"{row['context']}\n\n{row['question']}\n{row['answer_prefix']}"
        req = RequestFuncInput(
            prompt="",
            messages=[{"role": "user", "content": prompt}],
            api_url=api_url,
            prompt_len=len(prompt),
            output_len=(args.output_len or int(row.get("max_new_tokens", 128))),
            model=args.model,
            no_thinking=args.no_thinking,
        )
        async with sem:
            out = await async_request_openai_chat_completions(req, pbar=pbar)
        gen = out.generated_text or ""
        gold = _acceptable_answers(row["answer"])
        ok = any(g.lower() in gen.lower() for g in gold)
        return row["task"], ok

    results = await asyncio.gather(*[run_one(r) for r in rows])
    pbar.close()

    per_task: dict[str, list[int]] = {}
    for task, ok in results:
        rec = per_task.setdefault(task, [0, 0])
        rec[0] += int(ok)
        rec[1] += 1

    print("\n" + "=" * 52)
    total_correct = total = 0
    for task in sorted(per_task):
        correct, count = per_task[task]
        total_correct += correct
        total += count
        print(f"  {task:<22}: {100 * correct / count:6.2f}  ({correct}/{count})")
    print(f"TOTAL retrieval accuracy: {100 * total_correct / total:.2f}  "
          f"({total_correct}/{total})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", "-m", type=str, required=True, help="Served model name/path."
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument(
        "--port",
        type=str,
        default="8000",
        help="Server port for the OpenAI-compatible chat endpoint.",
    )
    parser.add_argument(
        "--length",
        type=int,
        default=4096,
        choices=[4096, 8192, 16384],
        help="RULER context-length bucket to evaluate.",
    )
    parser.add_argument(
        "--num-per-task",
        type=int,
        default=50,
        help="Number of examples per RULER task (13 tasks total).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=32,
        help="Max in-flight requests to the server.",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=0,
        help="Override max new tokens; 0 uses each row's max_new_tokens.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Local RULER dir with <length>/test-*.parquet (e.g. hf_cache/ruler).",
    )
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help="Ask reasoning models to answer directly (chat_template_kwargs).",
    )
    args = parser.parse_args()
    asyncio.run(evaluate(args))


if __name__ == "__main__":
    main()
