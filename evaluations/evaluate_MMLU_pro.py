# adopt from https://github.com/TIGER-AI-Lab/MMLU-Pro/blob/main/evaluate_from_api.py

import os
import re
import random
from tqdm import tqdm
from datasets import load_dataset
import argparse
from benchmarks.backend_request_func import async_request_openai_chat_completions, RequestFuncInput
import asyncio


API_KEY = "EMPTY"
random.seed(12345)


def load_mmlu_pro():
    dataset = load_dataset("TIGER-Lab/MMLU-Pro")
    test_df, val_df = dataset["test"], dataset["validation"]
    test_df = preprocess(test_df)
    val_df = preprocess(val_df)
    return test_df, val_df


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


def extract_answer(text):
    pattern = r"answer is \(?([A-J])\)?"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        # print("1st answer extract failed\n" + text)
        return extract_again(text)


def extract_again(text):
    match = re.search(r'.*[aA]nswer:\s*([A-J])', text)
    if match:
        return match.group(1)
    else:
        return extract_final(text)


def extract_final(text):
    pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(0)
    else:
        return None


def single_request(api_url, single_question, cot_examples_dict, pbar):
    category = single_question["category"]
    cot_examples = cot_examples_dict[category]
    question = single_question["question"]
    options = single_question["options"]
    prompt = "The following are multiple choice questions (with answers) about {}. Think step by" \
             " step and then output the answer in the format of \"The answer is (X)\" at the end.\n\n" \
        .format(category)
    for each in cot_examples:
        prompt += format_example(each["question"],
                                 each["options"], each["cot_content"])
    input_text = format_example(question, options)

    prompt = prompt + input_text

    request_func_input = RequestFuncInput(prompt=prompt,
                                          api_url=api_url,
                                          prompt_len=len(prompt),
                                          output_len=args.output_len,
                                          model=args.model,
                                          )
    return async_request_openai_chat_completions(request_func_input=request_func_input, pbar=pbar)



async def evaluate(subjects):
    api_url = f"http://{args.host}:{args.port}/v1/chat/completions"
    test_df, dev_df = load_mmlu_pro()
    if not subjects:
        subjects = list(test_df.keys())
    print("assigned subjects", subjects)
    category_record = {'total':{'#correct':0,'#wrong':0}}
    
    print(f"Sending requests ...")
    pbar = tqdm()
    tasks = []
    test_data_total = []
    for subject in subjects:
        test_data = test_df[subject][:args.num_per_sub]
        test_data_total.extend(test_data)
        for each in test_data:
            tasks.append(single_request(api_url, each, dev_df, pbar))
    pbar.total = len(tasks)
    completions = await asyncio.gather(*tasks)
    pbar.close()
    print(f"Processing completions ...")
    for idx, each in tqdm(enumerate(test_data_total),total=len(tasks)):
        label = each["answer"]
        response = completions[idx].generated_text
        response = response.replace('**', '')
        pred = extract_answer(response)
        category = each["category"]
        if response is not None:
            if category not in category_record:
                category_record[category] = {"#correct": 0, "#wrong": 0}
            each["pred"] = pred
            each["model_outputs"] = response
            if pred is not None:
                if pred == label:
                    category_record[category]["#correct"] += 1
                    category_record['total']['#correct'] += 1
                else:
                    category_record[category]["#wrong"] += 1
                    category_record['total']['#wrong'] += 1
            else:
                category_record[category]["#wrong"] += 1
                category_record['total']['#wrong'] += 1
    category_record['total']['score'] = round(
        100*category_record['total']['#correct'] / (
            category_record['total']['#correct'] + category_record['total']['#wrong']),2)
    print(category_record)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, required=True)
    parser.add_argument("--assigned_subjects", "-a", type=str, default="all",
                        help="business, law, psychology, biology, chemistry, history, other, health, "
                             "economics, math, physics, computer science, philosophy, engineering")
    parser.add_argument("--host", type=str, default='0.0.0.0')
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--output-len", type=int, default=1024)
    parser.add_argument("--num-per-sub", type=int, default=100)
    assigned_subjects = []
    args = parser.parse_args()

    if args.assigned_subjects == "all":
        assigned_subjects = []
    else:
        assigned_subjects = args.assigned_subjects.split(",")
    asyncio.run(evaluate(assigned_subjects))
