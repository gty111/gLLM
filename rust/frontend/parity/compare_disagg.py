"""Encoder disaggregation: two LM front ends, same answers.
usage: compare_disagg.py <reference lm base> <candidate lm base>"""
import base64, json, sys, urllib.request, urllib.error
from pathlib import Path
A, B = sys.argv[1], sys.argv[2]
MEDIA = Path(__file__).parent / "media"
def call(base, body):
    model = json.load(urllib.request.urlopen(base + "/v1/models"))["data"][0]["id"]
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps({"model": model, **body}).encode(),
                                 {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            j = json.load(r)
            return 200, (j["choices"][0]["message"]["content"] or "").strip(), j["usage"]["prompt_tokens"]
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:300], None
def du(name, mime):
    return f"data:{mime};base64," + base64.b64encode((MEDIA / name).read_bytes()).decode()
failures = []
cases = {
    "text": [],
    "image": [("image_url", du("shapes.png", "image/png"))],
    "jpeg": [("image_url", du("shapes.jpg", "image/jpeg"))],
    "two images": [("image_url", du("shapes.png", "image/png")), ("image_url", du("tri.png", "image/png"))],
}
for name, parts in cases.items():
    content = [{"type": k, k: {"url": u}} for k, u in parts] + [{"type": "text", "text": "Describe what you see in one sentence."}]
    body = {"messages": [{"role": "user", "content": content}], "temperature": 0, "max_tokens": 48,
            "chat_template_kwargs": {"enable_thinking": False}}
    a, b = call(A, body), call(B, body)
    ok = a == b and a[0] == 200
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {'' if ok else f'{a} | {b}'}", flush=True)
    if not ok:
        failures.append(name)
print("FAILURES:", failures or "none")
