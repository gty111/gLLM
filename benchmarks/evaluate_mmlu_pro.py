# adopt from https://github.com/TIGER-AI-Lab/MMLU-Pro/blob/main/evaluate_from_api.py

import argparse
import asyncio
import os
import random
import re

from backend_request_func import RequestFuncInput, async_request_openai_chat_completions
from datasets import load_dataset
from tqdm import tqdm

API_KEY = "EMPTY"
random.seed(12345)

_ANSWER_IS_RE = re.compile(r"answer is \(?([A-J])\)?")
_ANSWER_LINE_RE = re.compile(r"[aA]nswer:\s*([A-J])")
_STANDALONE_CHOICE_RE = re.compile(r"\b[A-J]\b")


def load_mmlu_pro():
    # ``--data-path`` points at a local copy of MMLU-Pro for offline / air-gapped
    # runs. It may be either:
    #   * a directory holding ``test-*.parquet`` / ``validation-*.parquet``
    #     (the layout under the HF ``data/`` folder), or
    #   * any path/name accepted by ``datasets.load_dataset`` directly.
    # When empty, fall back to streaming ``TIGER-Lab/MMLU-Pro`` from the Hub.
    data_path = args.data_path
    if data_path:
        test_parquet = _find_split_parquet(data_path, "test")
        val_parquet = _find_split_parquet(data_path, "validation")
        if test_parquet and val_parquet:
            dataset = load_dataset(
                "parquet",
                data_files={"test": test_parquet, "validation": val_parquet},
            )
        else:
            dataset = load_dataset(data_path)
    else:
        dataset = load_dataset("TIGER-Lab/MMLU-Pro")
    test_df, val_df = dataset["test"], dataset["validation"]
    test_df = preprocess(test_df)
    val_df = preprocess(val_df)
    return test_df, val_df


def _find_split_parquet(data_path, split):
    """Return the parquet file for ``split`` under ``data_path`` if present."""
    if not os.path.isdir(data_path):
        return None
    for fname in sorted(os.listdir(data_path)):
        if fname.startswith(split) and fname.endswith(".parquet"):
            return os.path.join(data_path, fname)
    return None


def preprocess(test_df):
    res_df = []
    for each in test_df:
        options = []
        for opt in each["options"]:
            if opt == "N/A":
                continue
            options.append(opt)
        each["options"] = options
        res_df.append(each)
    res = {}
    for each in res_df:
        if each["category"] not in res:
            res[each["category"]] = []
        res[each["category"]].append(each)
    return res


def format_example(question, options, cot_content=""):
    if cot_content == "":
        cot_content = "Let's think step by step."
    if cot_content.startswith("A: "):
        cot_content = cot_content[3:]
    example = "Question: {}\nOptions: ".format(question)
    choice_map = "ABCDEFGHIJ"
    for i, opt in enumerate(options):
        example += "{}. {}\n".format(choice_map[i], opt)
    if cot_content == "":
        example += "Answer: "
    else:
        example += "Answer: " + cot_content + "\n\n"
    return example


def format_question(question, options):
    """The question + lettered options only (a single chat user turn)."""
    text = "Question: {}\nOptions: ".format(question)
    choice_map = "ABCDEFGHIJ"
    for i, opt in enumerate(options):
        text += "{}. {}\n".format(choice_map[i], opt)
    text += "Answer: Let's think step by step."
    return text


def format_answer(cot_content):
    """The assistant turn for a few-shot example: its CoT + final answer line."""
    if cot_content.startswith("A: "):
        cot_content = cot_content[3:]
    return cot_content


def extract_answer(text):
    match = _ANSWER_IS_RE.search(text)
    if match:
        return match.group(1)
    else:
        # print("1st answer extract failed\n" + text)
        return extract_again(text)


def extract_again(text):
    # Preserve the old ``.*Answer:`` semantics exactly: choose the first line
    # containing a marker and, if that line has several, the final marker on
    # that line. Scanning linearly avoids the old regex backtracking cost.
    for line in text.splitlines():
        last_on_line = None
        for match in _ANSWER_LINE_RE.finditer(line):
            last_on_line = match.group(1)
        if last_on_line is not None:
            return last_on_line
    return extract_final(text)


def extract_final(text):
    last = None
    for match in _STANDALONE_CHOICE_RE.finditer(text):
        last = match.group(0)
    return last


def single_request(api_url, single_question, cot_examples_dict, pbar):
    category = single_question["category"]
    cot_examples = cot_examples_dict[category]
    question = single_question["question"]
    options = single_question["options"]
    # Render few-shot examples as real multi-turn chat: a system instruction,
    # then one user(question)/assistant(CoT+answer) turn per example, then the
    # target question as the final user turn. This keeps the model from
    # re-answering the shots (which happens when they are flattened into one
    # user message) so the "The answer is (X)" it emits is for THIS question.
    system = (
        "The following are multiple choice questions (with answers) about {}. Think"
        ' step by step and then output the answer in the format of "The answer is (X)"'
        " at the end.".format(category)
    )
    messages = [{"role": "system", "content": system}]
    for each in cot_examples[: args.num_shots]:
        messages.append(
            {"role": "user", "content": format_question(each["question"], each["options"])}
        )
        messages.append(
            {"role": "assistant", "content": format_answer(each["cot_content"])}
        )
    messages.append({"role": "user", "content": format_question(question, options)})

    request_func_input = RequestFuncInput(
        prompt="",
        messages=messages,
        api_url=api_url,
        prompt_len=sum(len(m["content"]) for m in messages),
        output_len=args.output_len,
        model=args.model,
        no_thinking=args.no_thinking,
    )
    return async_request_openai_chat_completions(
        # The evaluator advances the bar only after this response has also
        # been parsed and scored, keeping its count and live score atomic.
        request_func_input=request_func_input, pbar=None
    )


async def evaluate(subjects):
    # ``--port`` may be a comma-separated list (e.g. per-DP endpoints); requests
    # are round-robined across the resulting URLs so load is split across them.
    ports = [p.strip() for p in str(args.port).split(",") if p.strip()]
    api_urls = [f"http://{args.host}:{p}/v1/chat/completions" for p in ports]
    print("endpoints:", api_urls)
    test_df, dev_df = load_mmlu_pro()
    if not subjects:
        subjects = list(test_df.keys())
    print("assigned subjects", subjects)
    category_record = {"total": {"#correct": 0, "#wrong": 0}}

    # Cap in-flight requests so we don't dump all 1400 prompts onto the server
    # at once. A bounded client-side concurrency keeps the engine's running set
    # small enough that its KV cache / scheduler stays in a healthy regime
    # (no page-exhaustion throttling) and makes the run reproducible.
    sem = asyncio.Semaphore(args.concurrency)

    async def bounded_request(api_url, index, each):
        async with sem:
            completion = await single_request(api_url, each, dev_df, pbar)
        return index, each, completion

    print(f"Sending requests (concurrency={args.concurrency}) ...")
    pbar = tqdm()
    tasks = []
    test_data_total = []
    for subject in subjects:
        test_data_total.extend(test_df[subject][: args.num_per_sub])
    if args.shuffle_seed is not None:
        # Vary only *which questions share a batch*: same questions, same
        # prompts, same few-shot examples, different dispatch order. Kernels are
        # not batch-invariant, so two runs of one configuration do not score
        # identically; shuffling measures that noise floor, which is what any
        # A/B difference has to be compared against.
        random.Random(args.shuffle_seed).shuffle(test_data_total)
    for index, each in enumerate(test_data_total):
        api_url = api_urls[index % len(api_urls)]
        tasks.append(asyncio.create_task(bounded_request(api_url, index, each)))
    pbar.total = len(tasks)
    n_empty = 0
    processed = 0
    output = open(args.save, "w", buffering=1) if args.save else None
    try:
        # Score and persist each response as soon as it arrives. Besides making
        # live accuracy visible, this bounds retained response memory and removes
        # the old multi-minute, post-request "Processing completions" phase.
        for task in asyncio.as_completed(tasks):
            idx, each, completion = await task
            label = each["answer"]
            response = completion.generated_text
            if not response:
                n_empty += 1
            response = (response or "").replace("**", "")
            pred = extract_answer(response)
            category = each["category"]
            if category not in category_record:
                category_record[category] = {"#correct": 0, "#wrong": 0}
            each["pred"] = pred
            each["model_outputs"] = response
            correct = pred is not None and pred == label
            if correct:
                category_record[category]["#correct"] += 1
                category_record["total"]["#correct"] += 1
            else:
                category_record[category]["#wrong"] += 1
                category_record["total"]["#wrong"] += 1
            if output is not None:
                import json as _json
                output.write(_json.dumps({
                    "qid": each.get("question_id", idx),
                    "category": category,
                    "gold": label,
                    "pred": pred,
                    "correct": correct,
                    "response": response,
                }, ensure_ascii=False) + "\n")

            processed += 1
            correct_so_far = category_record["total"]["#correct"]
            live_score = 100 * correct_so_far / processed
            pbar.set_postfix(
                score=f"{live_score:.2f}",
                correct=correct_so_far,
                empty=n_empty,
                refresh=False,
            )
            pbar.update(1)
    finally:
        if output is not None:
            output.close()
        pbar.close()
    total = category_record["total"]
    total["score"] = round(
        100 * total["#correct"] / (total["#correct"] + total["#wrong"]), 2
    )
    print(f"empty responses: {n_empty}/{len(test_data_total)}")
    for cat in sorted(category_record):
        if cat == "total":
            continue
        r = category_record[cat]
        n = r["#correct"] + r["#wrong"]
        print(f"  {cat:20s}: {100 * r['#correct'] / n:5.2f}  ({r['#correct']}/{n})")
    print("=" * 50)
    print(
        f"TOTAL accuracy: {total['score']}  "
        f"({total['#correct']}/{total['#correct'] + total['#wrong']})"
    )
    if args.save:
        print(f"Saved {processed} per-question rows to {args.save}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, required=True)
    parser.add_argument(
        "--assigned_subjects",
        "-a",
        type=str,
        default="all",
        help="business, law, psychology, biology, chemistry, history, other, health, "
        "economics, math, physics, computer science, philosophy, engineering",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument(
        "--port",
        type=str,
        default="8000",
        help="Server port, or a comma-separated list of ports to round-robin "
        "across (e.g. per-DP endpoints).",
    )
    parser.add_argument("--output-len", type=int, default=1024)
    parser.add_argument("--num-per-sub", type=int, default=100)
    parser.add_argument("--num-shots", type=int, default=5)
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=None,
        help="Shuffle the dispatch order with this seed. Same questions and "
        "prompts, different batch composition -- use it to measure a "
        "configuration's own run-to-run spread before reading anything into an "
        "A/B difference.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=128,
        help="Max number of in-flight requests sent to the server at once.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="",
        help="Local MMLU-Pro path for offline runs: either a directory with "
        "test-*.parquet / validation-*.parquet, or any path/name accepted by "
        "datasets.load_dataset. Empty = stream TIGER-Lab/MMLU-Pro from the Hub.",
    )
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help="Send chat_template_kwargs={'thinking'/'enable_thinking': False} so "
        "reasoning models (e.g. Kimi-K2.5) answer directly instead of emitting a "
        "long reasoning trace that gets truncated by --output-len.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="",
        help="Optional path to dump per-question {id, gold, pred, correct, "
        "response} as JSONL for base-vs-MTP prediction diffing.",
    )
    assigned_subjects = []
    args = parser.parse_args()

    if args.assigned_subjects == "all":
        assigned_subjects = []
    else:
        assigned_subjects = args.assigned_subjects.split(",")
    asyncio.run(evaluate(assigned_subjects))
