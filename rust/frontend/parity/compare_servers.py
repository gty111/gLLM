"""Two front ends on the same model and engine flags: same requests, same answers.

usage: compare_servers.py <reference base> <candidate base>
Run the reference build (e.g. the last release) and the candidate (e.g. after a
crate upgrade) side by side on separate GPUs in one job.
Compares what a client reads (text, reasoning, tool calls, logprobs, usage,
finish reason, error status), not the exact JSON layout.
"""
import concurrent.futures as cf
import json
import sys
import time
import urllib.error
import urllib.request

A, B = sys.argv[1], sys.argv[2]
failures = []


def call(base, path, body=None, timeout=900):
    if body is not None and path.startswith("/v1/"):
        body = {"model": MODELS[base], **body}
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data, {"Content-Type": "application/json"},
                                 method="POST" if data is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            return r.status, (json.loads(raw) if raw else {})
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw or b"{}")
        except ValueError:
            return e.code, {"raw": raw.decode(errors="replace")}


def stream(base, path, body, stop_after=None):
    body = {"model": MODELS[base], **body, "stream": True}
    req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    chunks = []
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            for line in r:
                line = line.decode().strip()
                if line.startswith("data: ") and line != "data: [DONE]":
                    chunks.append(json.loads(line[6:]))
                    if stop_after and len(chunks) >= stop_after:
                        break
    except urllib.error.HTTPError as e:
        return [{"status": e.code}]
    return chunks


def lp_view(lp):
    if not lp:
        return None
    if "content" in lp:  # chat
        return [(e["token"], round(e["logprob"], 2), [t["token"] for t in e.get("top_logprobs", [])])
                for e in lp["content"] or []]
    return list(zip(lp["tokens"], [round(x, 2) for x in lp["token_logprobs"]]))


def prompt_lp_view(p):
    if not p:
        return None
    return [None if e is None else (e["token_id"], round(e["logprob"], 2)) for e in p]


def view(status, r):
    if status != 200:
        return {"status": status}
    c = r["choices"][0]
    usage = {k: (r.get("usage") or {}).get(k) for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
    out = {"finish": c.get("finish_reason"), "usage": usage,
           "logprobs": lp_view(c.get("logprobs")), "prompt_logprobs": prompt_lp_view(c.get("prompt_logprobs"))}
    if "message" in c:
        m = c["message"]
        # Compare text without surrounding whitespace (older builds kept it around </think>).
        out["content"] = (m.get("content") or "").strip() or None
        out["reasoning"] = (m.get("reasoning_content") or "").strip() or None
        out["tools"] = [(t["function"]["name"], json.loads(t["function"]["arguments"]))
                        for t in m.get("tool_calls") or []]
    else:
        out["text"] = c.get("text")
    return out


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {'' if ok else detail}", flush=True)
    if not ok:
        failures.append(name)


def same(name, path, body):
    (s1, a), (s2, b) = call(A, path, body), call(B, path, body)
    va, vb = view(s1, a), view(s2, b)
    check(name, va == vb, f"\n  ref={json.dumps(va)[:600]}\n  new={json.dumps(vb)[:600]}"
          f"\n  new raw={json.dumps(b)[:300]}")
    return s2, b


MODELS = {b: call(b, "/v1/models")[1]["data"][0]["id"] for b in (A, B)}
NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}
msgs = [{"role": "user", "content": "Name three primary colors. Answer briefly."}]
greedy = {"temperature": 0, "max_tokens": 48}

same("chat", "/v1/chat/completions", {"messages": msgs, **greedy, **NO_THINK})
same("chat thinking", "/v1/chat/completions", {"messages": msgs, "temperature": 0, "max_tokens": 300})
same("chat logprobs", "/v1/chat/completions",
     {"messages": msgs, **greedy, "logprobs": True, "top_logprobs": 3, **NO_THINK})
same("chat prompt_logprobs", "/v1/chat/completions",
     {"messages": msgs, **greedy, "prompt_logprobs": 2, **NO_THINK})
same("completion", "/v1/completions", {"prompt": "The capital of France is", "temperature": 0, "max_tokens": 16})
same("completion logprobs + prompt_logprobs", "/v1/completions",
     {"prompt": "The capital of France is", "temperature": 0, "max_tokens": 16, "logprobs": 2,
      "prompt_logprobs": 1})
same("completion ignore_eos", "/v1/completions",
     {"prompt": "1, 2, 3,", "temperature": 0, "max_tokens": 40, "ignore_eos": True})
same("multi-turn", "/v1/chat/completions", {"messages": [
    {"role": "system", "content": "You are terse."},
    {"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello."},
    {"role": "user", "content": "What is 2+2?"}], **greedy, **NO_THINK})
schema = {"type": "object", "properties": {"city": {"type": "string"},
                                           "population": {"type": "integer"}},
          "required": ["city", "population"], "additionalProperties": False}
def json_keys(status, r):
    try:
        return status, sorted(json.loads(r["choices"][0]["message"]["content"]))
    except Exception:
        return status, None


s1, a = call(A, "/v1/chat/completions", {
    "messages": [{"role": "user", "content": "Give the largest city of Japan as JSON."}],
    "temperature": 0, "max_tokens": 1024, "chat_template_kwargs": {"enable_thinking": True},
    "response_format": {"type": "json_schema", "json_schema": {"name": "c", "schema": schema}}})
s2, b = call(B, "/v1/chat/completions", {
    "messages": [{"role": "user", "content": "Give the largest city of Japan as JSON."}],
    "temperature": 0, "max_tokens": 1024, "chat_template_kwargs": {"enable_thinking": True},
    "response_format": {"type": "json_schema", "json_schema": {"name": "c", "schema": schema}}})
check("json_schema thinking=True (valid, same keys)", json_keys(s1, a) == json_keys(s2, b) == (200, ["city", "population"]),
      f"{json_keys(s1, a)} {json_keys(s2, b)} {str(b)[:300]}")
for think in (False,):
    same(f"json_schema thinking={think}", "/v1/chat/completions", {
        "messages": [{"role": "user", "content": "Give the largest city of Japan as JSON."}],
        "temperature": 0, "max_tokens": 1024, "chat_template_kwargs": {"enable_thinking": think},
        "response_format": {"type": "json_schema", "json_schema": {"name": "c", "schema": schema}}})
same("json_object", "/v1/chat/completions", {
    "messages": [{"role": "user", "content": "Describe a cat as a JSON object."}],
    "temperature": 0, "max_tokens": 256, "response_format": {"type": "json_object"}, **NO_THINK})
tools = [{"type": "function", "function": {
    "name": "get_weather", "description": "Get the weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                   "required": ["city"]}}}]
same("tool call auto", "/v1/chat/completions", {
    "messages": [{"role": "user", "content": "What's the weather in Paris?"}],
    "tools": tools, "temperature": 0, "max_tokens": 256, **NO_THINK})
same("tool result turn", "/v1/chat/completions", {"messages": [
    {"role": "user", "content": "What's the weather in Paris?"},
    {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function",
     "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]},
    {"role": "tool", "tool_call_id": "call_1", "content": "Sunny, 21C"}],
    "tools": tools, **greedy, **NO_THINK})
same("too long refused", "/v1/completions", {"prompt": "hi", "temperature": 0, "max_tokens": 10_000_000})
same("bad model refused", "/v1/chat/completions", {"messages": msgs, "model": "nope", **greedy})


# Multimodal: both sides get the same data URLs.
import base64
from pathlib import Path

MEDIA = Path(__file__).parent / "media"


def data_url(name, mime):
    return f"data:{mime};base64," + base64.b64encode((MEDIA / name).read_bytes()).decode()


for name, parts in [
    ("image png", [("image_url", data_url("shapes.png", "image/png"))]),
    ("image jpeg", [("image_url", data_url("shapes.jpg", "image/jpeg"))]),
    ("two images", [("image_url", data_url("shapes.png", "image/png")), ("image_url", data_url("tri.png", "image/png"))]),
    ("video", [("video_url", data_url("clip.mp4", "video/mp4"))]),
]:
    content = [{"type": k, k: {"url": u}} for k, u in parts] + [
        {"type": "text", "text": "Describe what you see in one sentence."}]
    body = {"messages": [{"role": "user", "content": content}], "temperature": 0, "max_tokens": 64, **NO_THINK}
    (s1, a), (s2, b) = call(A, "/v1/chat/completions", body), call(B, "/v1/chat/completions", body)
    va, vb = view(s1, a), view(s2, b)
    check(name, va == vb, f"\n  ref={json.dumps(va)[:400]}\n  new={json.dumps(vb)[:400]}\n  new raw={json.dumps(b)[:300]}")


def stream_text(chunks):
    return "".join((c["choices"][0].get("delta") or {}).get("content") or c["choices"][0].get("text") or ""
                   for c in chunks if c.get("choices"))


sa = stream(A, "/v1/chat/completions", {"messages": msgs, **greedy, **NO_THINK,
                                        "stream_options": {"include_usage": True}})
sb = stream(B, "/v1/chat/completions", {"messages": msgs, **greedy, **NO_THINK,
                                        "stream_options": {"include_usage": True}})
check("chat streamed", stream_text(sa) == stream_text(sb) and sa[-1].get("usage") == sb[-1].get("usage"),
      f"{stream_text(sa)!r} vs {stream_text(sb)!r} | {sa[-1:]} vs {sb[-1:]}")
sa = stream(A, "/v1/completions", {"prompt": "The capital of France is", "temperature": 0, "max_tokens": 16})
sb = stream(B, "/v1/completions", {"prompt": "The capital of France is", "temperature": 0, "max_tokens": 16})
check("completion streamed", stream_text(sa) == stream_text(sb), f"{stream_text(sa)!r} vs {stream_text(sb)!r}")
sb = stream(B, "/v1/chat/completions", {"messages": [{"role": "user", "content": "What's the weather in Paris?"}],
                                        "tools": tools, **greedy, **NO_THINK})
names = [d["function"].get("name") for c in sb if c.get("choices")
         for d in (c["choices"][0].get("delta") or {}).get("tool_calls") or []]
check("new streamed tool call", "get_weather" in names, str(sb)[:400])

# Rust-only: stop strings and stop_token_ids (the Python front end refuses them).
s, r = call(B, "/v1/completions", {"prompt": "1, 2, 3, 4, 5, 6,", "temperature": 0, "max_tokens": 16,
                                   "stop": [" 9"]})
check("new stop string", s == 200 and "9" not in r["choices"][0]["text"] and r["choices"][0]["finish_reason"] == "stop",
      f"{s} {r}")

bodies = [{"messages": [{"role": "user", "content": f"Count from {i} to {i + 12}."}],
           "temperature": 0, "max_tokens": 64, **NO_THINK} for i in range(32)]
with cf.ThreadPoolExecutor(64) as pool:
    ra = list(pool.map(lambda b: view(*call(A, "/v1/chat/completions", b)), bodies))
    rb = list(pool.map(lambda b: view(*call(B, "/v1/chat/completions", b)), bodies))
check("32 concurrent", ra == rb, f"{[i for i, (x, y) in enumerate(zip(ra, rb)) if x != y]}")

stream(B, "/v1/chat/completions", {"messages": msgs, "max_tokens": 4000, "ignore_eos": True, **NO_THINK},
       stop_after=5)
time.sleep(2)
same("after abort", "/v1/chat/completions", {"messages": msgs, **greedy, **NO_THINK})
for path in ("/start_profile", "/stop_profile"):
    s, _ = call(B, path, {})
    check(f"new {path}", s == 200, s)
s1, a = call(B, "/sleep", {})
s2, b = call(B, "/is_sleeping")
s3, c = call(B, "/wake_up", {})
s4, d = call(B, "/is_sleeping")
check("new sleep/wake", s1 == s3 == 200 and b.get("is_sleeping") is True and d.get("is_sleeping") is False,
      f"{s1} {a} | {b} | {s3} {c} | {d}")
call(B, "/sleep", {})
same("request wakes a sleeping engine", "/v1/chat/completions", {"messages": msgs, **greedy, **NO_THINK})
# --- Responses API -----------------------------------------------------------
def resp_view(status, r):
    if status != 200:
        return {"status": status}
    out = []
    for o in r.get("output", []):
        if o["type"] == "message":
            out.append(("message", "".join(c.get("text", "") for c in o["content"]).strip()))
        elif o["type"] == "reasoning":
            out.append(("reasoning", "".join(c.get("text", "") for c in o["content"]).strip()))
        elif o["type"] == "function_call":
            out.append(("function_call", o["name"], json.loads(o["arguments"])))
        elif o["type"] == "custom_tool_call":
            out.append(("custom_tool_call", o["name"], o["input"]))
    return {"status": r["status"], "output": out, "usage": {k: r["usage"][k] for k in ("input_tokens", "output_tokens")}}


def same_resp(name, body):
    (s1, a), (s2, b) = call(A, "/v1/responses", body), call(B, "/v1/responses", body)
    va, vb = resp_view(s1, a), resp_view(s2, b)
    check(name, va == vb, f"\n  ref={json.dumps(va)[:500]}\n  new={json.dumps(vb)[:500]}\n  new raw={json.dumps(b)[:300]}")
    return a, b


same_resp("responses text", {"input": "Name three primary colors. Answer briefly.", "temperature": 0,
                             "max_output_tokens": 64, "reasoning": {"effort": "none"}})
same_resp("responses thinking", {"input": "Name three primary colors. Answer briefly.", "temperature": 0,
                                 "max_output_tokens": 400})
rtools = [{"type": "function", "name": "get_weather", "description": "Get the weather for a city",
           "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}]
same_resp("responses function call", {"input": "What's the weather in Paris?", "tools": rtools, "temperature": 0,
                                      "max_output_tokens": 256, "reasoning": {"effort": "none"}})
same_resp("responses custom tool", {"input": "Use the shell tool to list files in /tmp.", "temperature": 0,
    "tools": [{"type": "custom", "name": "shell", "description": "Run a shell command"}],
    "max_output_tokens": 256, "reasoning": {"effort": "none"}})
a, b = same_resp("responses stored turn 1", {"input": "My name is Ada. Reply with OK.", "temperature": 0,
                                             "max_output_tokens": 32, "store": True, "reasoning": {"effort": "none"}})
if a.get("id") and b.get("id"):
    (s1, a2), (s2, b2) = (call(A, "/v1/responses", {"input": "What is my name?", "previous_response_id": a["id"],
                                                    "temperature": 0, "max_output_tokens": 32, "reasoning": {"effort": "none"}}),
                          call(B, "/v1/responses", {"input": "What is my name?", "previous_response_id": b["id"],
                                                    "temperature": 0, "max_output_tokens": 32, "reasoning": {"effort": "none"}}))
    check("responses previous_response_id", resp_view(s1, a2) == resp_view(s2, b2),
          f"{resp_view(s1, a2)} vs {resp_view(s2, b2)}")
s, r = call(B, "/v1/responses", {"input": "hi", "previous_response_id": "resp_missing"})
check("responses unknown previous id", s == 404, f"{s} {r}")


def resp_stream(base, body):
    body = {"model": MODELS[base], **body, "stream": True}
    req = urllib.request.Request(base + "/v1/responses", json.dumps(body).encode(), {"Content-Type": "application/json"})
    events = []
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            line = line.decode().strip()
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    return events


body = {"input": "What's the weather in Paris?", "tools": rtools, "temperature": 0, "max_output_tokens": 256,
        "reasoning": {"effort": "none"}}
ea, eb = resp_stream(A, body), resp_stream(B, body)
check("responses streamed events", [e["type"] for e in ea] == [e["type"] for e in eb]
      and resp_view(200, ea[-1]["response"]) == resp_view(200, eb[-1]["response"]),
      f"{[e['type'] for e in ea]}\n  vs {[e['type'] for e in eb]}")
body = {"input": "Name three primary colors. Answer briefly.", "temperature": 0, "max_output_tokens": 64,
        "reasoning": {"effort": "none"}}
ea, eb = resp_stream(A, body), resp_stream(B, body)
check("responses streamed text", [e["type"] for e in ea] == [e["type"] for e in eb]
      and resp_view(200, ea[-1]["response"]) == resp_view(200, eb[-1]["response"]),
      f"{[e['type'] for e in ea]}\n  vs {[e['type'] for e in eb]}")

print("FAILURES:", failures if failures else "none")
