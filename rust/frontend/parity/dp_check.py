"""DP smoke test: concurrent greedy outputs, abort, sleep/wake, profile. Prints JSON."""
import concurrent.futures as cf
import json
import sys
import time
import urllib.error
import urllib.request

BASE = sys.argv[1]


def call(path, body=None):
    if body is not None and path.startswith("/v1/"):
        body = {"model": MODEL, **body}
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data, {"Content-Type": "application/json"},
                                 method="POST" if data is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, {"raw": e.read().decode(errors="replace")}


MODEL = call("/v1/models")[1]["data"][0]["id"]
NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}
bodies = [{"messages": [{"role": "user", "content": f"Count from {i} to {i + 12}."}],
           "temperature": 0, "max_tokens": 64, **NO_THINK} for i in range(64)]
with cf.ThreadPoolExecutor(64) as pool:
    res = list(pool.map(lambda b: call("/v1/chat/completions", b), bodies))
outputs = [r[1]["choices"][0]["message"]["content"] if r[0] == 200 else r for r in res]
checks = {"all 200": all(r[0] == 200 for r in res)}

# Abort a long stream after a few chunks.
req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps({
    "model": MODEL, "messages": [{"role": "user", "content": "Tell a story."}], "max_tokens": 4000,
    "ignore_eos": True, "stream": True, **NO_THINK}).encode(), {"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=900) as r:
    for i, _ in enumerate(r):
        if i > 10:
            break
time.sleep(2)
checks["after abort"] = call("/v1/chat/completions", bodies[0])[0] == 200
checks["profile"] = call("/start_profile", {})[0] == 200 and call("/stop_profile", {})[0] == 200
s1, _ = call("/sleep", {})
asleep = call("/is_sleeping")[1].get("is_sleeping")
s2, _ = call("/wake_up", {})
awake = call("/is_sleeping")[1].get("is_sleeping") is False
checks["sleep/wake"] = s1 == s2 == 200 and asleep is True and awake
checks["after wake"] = call("/v1/chat/completions", bodies[1])[0] == 200
print(json.dumps({"outputs": outputs, "checks": checks}))
