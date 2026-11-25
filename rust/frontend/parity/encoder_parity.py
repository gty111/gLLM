"""Prompt tokens: `gllm-frontend --render-request` vs a DeepSeek checkpoint's own encoder.

usage: encoder_parity.py <model dir> [--frontend BIN]   (CPU only)
Supports V4 (`encoding/encoding_dsv4.py`, or only its .pyc) and V4.1
(`encoding/encoding.py`). Each case states the thinking mode and effort the
official encoder is called with, i.e. what the front end must resolve to.
"""
import argparse
import importlib.machinery
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from transformers import AutoTokenizer

_ap = argparse.ArgumentParser()
_ap.add_argument("model", type=Path)
_ap.add_argument("--frontend", default=shutil.which("gllm-frontend"),
                 help="gllm-frontend binary (default: the one on PATH)")
_args = _ap.parse_args()
M, BIN = _args.model, _args.frontend
MODEL_TYPE = json.loads((M / "config.json").read_text())["model_type"]


def encoder_file():
    enc = M / "encoding"
    if MODEL_TYPE == "deepseek_v41":
        return enc / "encoding.py"
    src = enc / "encoding_dsv4.py"
    return src if src.exists() else next((enc / "__pycache__").glob("encoding_dsv4.*.pyc"))


def load(path):
    if path.suffix == ".pyc":
        loader = importlib.machinery.SourcelessFileLoader("enc", str(path))
        spec = importlib.util.spec_from_loader("enc", loader)
    else:
        spec = importlib.util.spec_from_file_location("enc", path)
        loader = spec.loader
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


ENC_FILE = encoder_file()
enc = load(ENC_FILE)
tok = AutoTokenizer.from_pretrained(M)
V41 = MODEL_TYPE == "deepseek_v41"
# An unset or "high" effort: V4.1 and the 0731 V4 encoder render "high"; the
# original V4 encoder only knows "max", so everything else is the plain prompt.
LEGACY_V4 = not V41 and b"Beyond maximum " not in ENC_FILE.read_bytes()
HIGH = None if LEGACY_V4 else "high"


def rust(body):
    with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
        json.dump({"model": "m", **body}, f)
        f.flush()
        p = subprocess.run([BIN, "--model-path", str(M), "--render-request", f.name], capture_output=True, text=True)
    if p.returncode:
        return "ERROR " + (p.stderr.strip().splitlines() or ["?"])[-1][:200]
    return json.loads(p.stdout.strip().splitlines()[-1])["prompt_token_ids"]


tools = [{"type": "function", "function": {"name": "get_weather", "description": "Get the weather",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
user = [{"role": "user", "content": "Name three primary colors."}]
system = [{"role": "system", "content": "You are terse."}] + user
turn = [{"role": "user", "content": "Weather in Paris?"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function",
         "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "Sunny"}]
with_tools = [{"role": "system", "content": "", "tools": tools}]
cases = [  # (name, request, official messages, thinking_mode, effort)
    ("no controls", {"messages": user}, user, "thinking", HIGH),
    ("thinking=false", {"messages": user, "chat_template_kwargs": {"thinking": False}}, user, "chat", None),
    ("enable_thinking=false", {"messages": user, "chat_template_kwargs": {"enable_thinking": False}}, user, "chat", None),
    ("effort none", {"messages": user, "reasoning_effort": "none"}, user, "chat", None),
    ("thinking=true", {"messages": user, "chat_template_kwargs": {"thinking": True}}, user, "thinking", HIGH),
    ("effort low", {"messages": user, "reasoning_effort": "low"}, user, "thinking", "low" if V41 else None),
    ("effort high", {"messages": user, "reasoning_effort": "high"}, user, "thinking", HIGH),
    ("effort max", {"messages": user, "reasoning_effort": "max"}, user, "thinking", "max"),
    ("system", {"messages": system}, system, "thinking", HIGH),
    ("tools, no thinking", {"messages": turn[:1], "tools": tools, "chat_template_kwargs": {"thinking": False}},
     with_tools + turn[:1], "chat", None),
    ("tool turn", {"messages": turn, "tools": tools}, with_tools + turn, "thinking", HIGH),
]
if V41:
    cases.append(("effort 30", {"messages": user, "reasoning_effort": 30}, user, "thinking", 30))

bad = []
for name, req, msgs, mode, effort in cases:
    r = rust(req)
    try:
        o = tok.encode(enc.encode_messages([dict(m) for m in msgs], thinking_mode=mode, reasoning_effort=effort),
                       add_special_tokens=False)
    except Exception as exc:  # noqa: BLE001
        o = f"ERROR {exc}"
    if r == o:
        print(f"[PASS] {name}")
        continue
    bad.append(name)
    print(f"[FAIL] {name}")
    print("   frontend:", r if isinstance(r, str) else repr(tok.decode(r))[-300:])
    print("   encoder: ", o if isinstance(o, str) else repr(tok.decode(o))[-300:])
print("FAILURES:", bad or "none")
