"""Prompt tokens: the Python encode path (offline `LLM`) vs `gllm-frontend --render-request`.
usage: render_parity.py <model dir> [dsv4|dsv32] [--frontend BIN]   (CPU only)"""
import copy, importlib.machinery, importlib.util, json, os, shutil, subprocess, sys, tempfile, types
from pathlib import Path
import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("model")
_ap.add_argument("variant", nargs="?", choices=["dsv4", "dsv32"])
_ap.add_argument("--frontend", default=shutil.which("gllm-frontend"),
                 help="gllm-frontend binary (default: the one on PATH)")
_args = _ap.parse_args()
M, VARIANT, BIN = _args.model, _args.variant, _args.frontend

from transformers import AutoTokenizer
from gllm.tokenizers.mixin import TokenizerMixin
import gllm.tokenizers.deepseek_official as dso

if VARIANT:
    # Some caches ship only the compiled encoder; load it like the .py.
    src = Path(M, "encoding", f"encoding_{VARIANT}.py")
    if not src.exists():
        pyc = next(Path(M, "encoding", "__pycache__").glob(f"encoding_{VARIANT}.*.pyc"))
        loader = importlib.machinery.SourcelessFileLoader(f"encoding_{VARIANT}", str(pyc))
        spec = importlib.util.spec_from_loader(loader.name, loader)
        mod = importlib.util.module_from_spec(spec); loader.exec_module(mod)
        dso.load_deepseek_encoder = lambda path, variant: mod

tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
stub = types.SimpleNamespace(tokenizer=tok, model_path=M, _deepseek_encoder_variant=VARIANT, use_mm=False)
encode = TokenizerMixin.encode.__get__(stub)


def py_tokens(body):
    kwargs = dict(body.get("chat_template_kwargs") or {})
    effort = body.get("reasoning_effort")
    if effort == "none":
        kwargs.setdefault("enable_thinking", False); kwargs.setdefault("thinking", False)
    elif effort is not None:
        kwargs.setdefault("reasoning_effort", effort)
    out = encode(copy.deepcopy(body["messages"]), chat=True, chat_template_kwargs=kwargs or None,
                 tools=body.get("tools"))
    return list(out)


def rs_tokens(body):
    with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
        json.dump(body, f); f.flush()
        p = subprocess.run([BIN, "--model-path", M, "--render-request", f.name], capture_output=True, text=True)
    if p.returncode != 0:
        return "ERROR: " + (p.stderr.strip().splitlines() or ["?"])[-1][:300]
    return json.loads(p.stdout.strip().splitlines()[-1])["prompt_token_ids"]


tools = [{"type": "function", "function": {"name": "get_weather", "description": "Get the weather",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
user = [{"role": "user", "content": "Name three primary colors."}]
cases = {
    "plain": {"messages": user},
    "system": {"messages": [{"role": "system", "content": "You are terse."}] + user},
    "thinking kwarg": {"messages": user, "chat_template_kwargs": {"thinking": True, "enable_thinking": True}},
    "no thinking kwarg": {"messages": user, "chat_template_kwargs": {"thinking": False, "enable_thinking": False}},
    "effort high": {"messages": user, "reasoning_effort": "high"},
    "effort none": {"messages": user, "reasoning_effort": "none"},
    "thinking + effort low": {"messages": user, "chat_template_kwargs": {"thinking": True}, "reasoning_effort": "low"},
    "thinking + effort high": {"messages": user, "chat_template_kwargs": {"thinking": True}, "reasoning_effort": "high"},
    "thinking + effort max": {"messages": user, "chat_template_kwargs": {"thinking": True}, "reasoning_effort": "max"},
    "thinking tools": {"messages": [{"role": "user", "content": "Weather in Paris?"}], "tools": tools,
                       "chat_template_kwargs": {"thinking": True}},
    "tools": {"messages": [{"role": "user", "content": "Weather in Paris?"}], "tools": tools},
    "tool turn": {"messages": [
        {"role": "user", "content": "Weather in Paris?"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function",
         "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "Sunny"}], "tools": tools},
    "multi-turn": {"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello."},
                                {"role": "user", "content": "2+2?"}]},
}
for k in ("tools", "tool turn"):
    for on in (False, True):
        cases[f"{k} thinking={on}"] = {**cases[k], "chat_template_kwargs": {"thinking": on}}
bad = []
for name, body in cases.items():
    try:
        a = py_tokens(body)
    except Exception as exc:
        a = f"PYERROR: {type(exc).__name__}: {exc}"
    b = rs_tokens(body)
    if isinstance(a, str):
        print(f"[INFO] {name}: python refuses ({a}); rust -> {'error' if isinstance(b, str) else f'{len(b)} tokens'}")
        continue
    ok = a == b
    if not ok:
        bad.append(name)
        if isinstance(b, str):
            print(f"[FAIL] {name}: {b}")
        else:
            i = next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            print(f"[FAIL] {name}: first diff at {i}\n   py: {tok.decode(a[max(0, i-5):i+40])!r}\n   rs: {tok.decode(b[max(0, i-5):i+40])!r}")
    else:
        print(f"[PASS] {name}")
print("FAILURES:", bad or "none")
