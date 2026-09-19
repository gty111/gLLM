"""Slurm-only Qwen online integration / graph / regression smoke suite."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import httpx
import jsonschema

MODEL = "/mnt/lustre/hf-models/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
SCHEMA = {"type": "object", "properties": {
    "answer": {"type": "integer"}, "label": {"type": "string", "enum": ["中文", "ok"]}},
    "required": ["answer", "label"], "additionalProperties": False}


def run_suite(url, full=True, constraints=True):
    client = httpx.Client(base_url=url, timeout=180)
    records = []

    def chat(i=0, stream=False, thinking=False, structured=True, sample=False, budget=256, long=False, top_k=20):
        constrained = structured and constraints
        schema = SCHEMA if i % 2 == 0 else {
            "type": "object", "properties": {"values": {"type": "array", "items": {"type": "boolean"}}},
            "required": ["values"], "additionalProperties": False}
        body = {"model": "Qwen/Qwen3.8-27B", "messages": [{"role": "user", "content":
            ("Context. " * 800 if long else "") +
            ("Return JSON: answer is 42 and label is 中文; or values is [true,false]." if structured else
             "Write the numbers from 1 to 200, one per line.")}],
            "max_completion_tokens": budget, "temperature": 0.7 if sample else 0,
            "chat_template_kwargs": {"enable_thinking": thinking}, "stream": stream}
        if sample:
            body.update(top_k=top_k, top_p=0.9)
        if constrained:
            body["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "answer", "schema": schema, "strict": True}}
        else:
            body["ignore_eos"] = True
        start = time.perf_counter()
        if stream:
            content, reason, finish, ttft = "", "", None, None
            with client.stream("POST", "/v1/chat/completions", json=body) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    event = json.loads(line[6:])
                    for choice in event["choices"]:
                        delta = choice["delta"]
                        content += delta.get("content") or ""
                        reason += delta.get("reasoning_content") or ""
                        if delta.get("content") or delta.get("reasoning_content"):
                            ttft = ttft or time.perf_counter() - start
                        finish = choice.get("finish_reason") or finish
            data = dict(content=content, reasoning=reason, finish=finish, ttft=ttft)
        else:
            response = client.post("/v1/chat/completions", json=body)
            response.raise_for_status()
            raw = response.json()
            choice = raw["choices"][0]
            data = dict(content=choice["message"]["content"],
                        reasoning=choice["message"].get("reasoning_content"),
                        finish=choice["finish_reason"], usage=raw["usage"])
        data.update(elapsed=time.perf_counter() - start, structured=constrained,
                    thinking=thinking, stream=stream, sample=sample, budget=budget)
        records.append(data)
        if constrained and budget > 1:
            assert data["finish"] == "stop", data
            jsonschema.validate(json.loads(data["content"]), schema)
            if thinking:
                assert data["reasoning"], data
        if budget == 1:
            assert data["finish"] == "length", data
        return data

    # Identical ordinary greedy workload for before/after parity and throughput.
    chat(structured=False, budget=256)
    plain_trials = []
    for _ in range(3):
        start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
            plain = list(pool.map(lambda i: chat(i, structured=False, budget=256), range(32)))
        plain_trials.append(dict(wall=time.perf_counter()-start,
            output_tokens=sum(r["usage"]["completion_tokens"] for r in plain),
            outputs=[r["content"] for r in plain]))
    structured_trials = []
    if full and constraints:
        # Fixed, nontrivial grammar forces exactly the same answer with and
        # without speculation. Isolate B1 latency and B32 aggregate throughput.
        literal = {"values": list(range(1, 65))}
        benchmark_schema = {"type": "object", "properties": {"values": {"const": literal["values"]}},
                            "required": ["values"], "additionalProperties": False}
        benchmark_body = {"model": "Qwen/Qwen3.8-27B", "messages": [{"role": "user", "content":
            "Return a JSON object with values containing integers from 1 through 64, in order."}],
            "max_completion_tokens": 512, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "numbers", "strict": True, "schema": benchmark_schema}}}

        def structured_benchmark(_):
            response = client.post("/v1/chat/completions", json=benchmark_body)
            response.raise_for_status()
            raw = response.json()
            choice = raw["choices"][0]
            assert choice["finish_reason"] == "stop", raw
            assert json.loads(choice["message"]["content"]) == literal, raw
            return raw

        print("Structured-only benchmark starting (B1/B32)", flush=True)
        structured_benchmark(0)
        for concurrency in (1, 32):
            for _ in range(3):
                start = time.perf_counter()
                with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                    results = list(pool.map(structured_benchmark, range(concurrency)))
                structured_trials.append(dict(concurrency=concurrency, wall=time.perf_counter()-start,
                    output_tokens=sum(r["usage"]["completion_tokens"] for r in results),
                    outputs=[r["choices"][0]["message"]["content"] for r in results]))
        print("Structured-only benchmark passed", flush=True)
    if full:
        chat()
        chat(stream=True)
        chat(thinking=True, budget=1024)
        chat(thinking=True, stream=True, budget=1024)
        chat(sample=True)
        chat(sample=True, top_k=-1)
        chat(sample=True, thinking=True, budget=1024)
        chat(budget=1)
        chat(long=True)
        start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
            mixed = list(pool.map(lambda i: chat(i, stream=i % 3 == 0,
                structured=i % 5 != 0, sample=i % 4 == 0), range(64)))
        mixed_wall = time.perf_counter()-start
        if not constraints:
            return dict(records=records, plain_trials=plain_trials, mixed_wall=mixed_wall)
        # JSON mode and Responses use the same backend but different protocol shapes.
        body = {"model": "Qwen/Qwen3.8-27B", "messages": [{"role": "user", "content": "Return JSON with x equal to 1."}],
                "response_format": {"type": "json_object"}, "max_completion_tokens": 128,
                "chat_template_kwargs": {"enable_thinking": False}, "temperature": 0}
        response = client.post("/v1/chat/completions", json=body)
        response.raise_for_status()
        assert isinstance(json.loads(response.json()["choices"][0]["message"]["content"]), dict)
        body["response_format"] = {"type": "json_schema", "json_schema": {
            "name": "literal", "strict": True, "schema": {
                "type": "object", "properties": {"literal": {"const": "中文\\\"\n<think>"}},
                "required": ["literal"], "additionalProperties": False}}}
        body.update(logprobs=True, top_logprobs=20, repetition_penalty=1.05)
        response = client.post("/v1/chat/completions", json=body)
        response.raise_for_status()
        jsonschema.validate(json.loads(response.json()["choices"][0]["message"]["content"]),
                            body["response_format"]["json_schema"]["schema"])
        for streaming in [False, True]:
            rbody = {"model": "Qwen/Qwen3.8-27B", "input": "Return answer 42 and label ok.",
                     "reasoning": {"effort": "none"}, "max_output_tokens": 256, "stream": streaming,
                     "text": {"format": {"type": "json_schema", "name": "answer", "strict": True, "schema": SCHEMA}}}
            r = client.post("/v1/responses", json=rbody)
            r.raise_for_status()
            if streaming:
                events = [json.loads(l[6:]) for l in r.text.splitlines() if l.startswith("data: ")]
                final = next(e["response"] for e in events if e["type"] == "response.completed")
                text = "".join(e["delta"] for e in events if e["type"] == "response.output_text.delta")
            else:
                final = r.json()
                text = "".join(c["text"] for o in final["output"] if o["type"] == "message" for c in o["content"])
            assert final["status"] == "completed", final
            jsonschema.validate(json.loads(text), SCHEMA)
        rbody.update(stream=False, max_output_tokens=1)
        limited = client.post("/v1/responses", json=rbody)
        limited.raise_for_status()
        assert limited.json()["status"] == "incomplete", limited.text
        assert limited.json()["incomplete_details"]["reason"] == "max_output_tokens"
        body["response_format"] = {"type": "json_schema", "json_schema": {
            "name": "bad", "schema": {"type": "integer", "minimum": 1}}}
        assert client.post("/v1/chat/completions", json=body).status_code == 400
        body["response_format"]["json_schema"]["schema"] = {"$ref": "#/$defs/missing"}
        assert client.post("/v1/chat/completions", json=body).status_code == 400
        # Client abort while reasoning must release its state without affecting
        # the next request, even if the numeric sequence ID is reused.
        body.update(stream=True, max_completion_tokens=1024,
                    chat_template_kwargs={"enable_thinking": True})
        body["response_format"]["json_schema"]["schema"] = SCHEMA
        with client.stream("POST", "/v1/chat/completions", json=body) as aborted:
            aborted.raise_for_status()
            for line in aborted.iter_lines():
                if line.startswith("data: "):
                    break
        # Ensure an invalid request did not poison the worker.
        chat()
    else:
        mixed_wall = None
    return dict(records=records, plain_trials=plain_trials, mixed_wall=mixed_wall,
                structured_trials=structured_trials)


def main():
    assert os.environ.get("SLURM_JOB_ID"), "GPU tests must run under Slurm"
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--baseline")
    parser.add_argument("--output", required=True)
    parser.add_argument("--modes", default="overlap,sync,pp,baseline")
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for mode in args.modes.split(","):
        baseline = mode in ("baseline", "baseline-pp")
        pp = mode in ("pp", "baseline-pp")
        source = args.baseline if baseline or mode == "before" else args.source
        command = [sys.executable, "-u", "-m", "gllm.entrypoints.api_server", "--model-path", MODEL,
                   "--host", "127.0.0.1", "--port", "18135", "--tp", "1" if mode == "sync" or pp else "2",
                   "--pp", "2" if pp else "1", "--maxp", "512", "--maxd", "32",
                   "--max-cuda-graph-bs", "32", "--model-max-length", "4096", "--gpu-memory-util", "0.65",
                   "--piecewise-cuda-graph", "on", "--mtp-enabled", "off" if pp else "on"]
        if mode != "sync":
            command.append("--overlap-scheduling")
        print(mode, command, flush=True)
        with (out / f"{mode}.server.log").open("w") as log:
            pythonpath = source + os.pathsep + os.environ.get("STRUCTURED_DEBUG_DIR", "")
            process = subprocess.Popen(command, cwd=source, env=dict(os.environ, PYTHONPATH=pythonpath),
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                for _ in range(600):
                    if process.poll() is not None:
                        raise RuntimeError(f"{mode} server exited: see {log.name}")
                    try:
                        if httpx.get("http://127.0.0.1:18135/health", timeout=2).status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    time.sleep(2)
                else:
                    raise TimeoutError("server startup timed out")
                result = run_suite("http://127.0.0.1:18135", full=mode != "baseline", constraints=not baseline)
                (out / f"{mode}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
                print(mode, "PASS", flush=True)
            finally:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


if __name__ == "__main__":
    main()
