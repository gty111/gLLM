# adopt from https://github.com/TIGER-AI-Lab/MMLU-Pro/blob/main/evaluate_from_api.py
import re
import random
import argparse
import asyncio

from gllm import LLM

from tqdm import tqdm
from datasets import load_dataset

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


def single_request(single_question, cot_examples_dict):
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
    
    return prompt



async def evaluate(subjects):
    test_df, dev_df = load_mmlu_pro()
    if not subjects:
        subjects = list(test_df.keys())
    print("assigned subjects", subjects)
    category_record = {'total':{'#correct':0,'#wrong':0}}
    
    llm = LLM(model_path=args.model,
              gpu_memory_util=args.gpu_memory_util,
              kvthresh=args.kvthresh,
              pp_size=args.pp,
              tp_size=args.tp,
              enable_prefix_caching=True,
              use_thinking=False)
    
    print(f"generating requests ...")
    prompts = []
    test_data_total = []
    for subject in subjects:
        test_data = test_df[subject][:args.num_per_sub]
        test_data_total.extend(test_data)
        for each in test_data:
            prompts.append(single_request(each, dev_df))
    
    seqs = llm.generate(prompts, output_lens=[args.output_len for i in range(len(prompts))])
    
    outputs = [seq.output for seq in seqs]
    
    print(f"Processing completions ...")
    for idx, each in tqdm(enumerate(test_data_total),total=len(prompts)):
        label = each["answer"]
        response = outputs[idx]
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
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument('--gpu-memory-util', type=float, default=0.9)
    parser.add_argument('--kvthresh', type=float, default=0.2)
    parser.add_argument("--output-len", type=int, default=1024)
    parser.add_argument("--num-per-sub", type=int, default=100)
    assigned_subjects = []
    args = parser.parse_args()

    if args.assigned_subjects == "all":
        assigned_subjects = []
    else:
        assigned_subjects = args.assigned_subjects.split(",")
    asyncio.run(evaluate(assigned_subjects))
