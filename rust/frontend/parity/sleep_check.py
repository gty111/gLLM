"""Sleep levels 1 and 2 on one server: answers before and after each cycle
match, GPU memory goes back with level 2, auto wake, abort, bad input.

usage: sleep_check.py <base> [--idle]   (--idle: the server runs with
--sleep-idle-timeout and --sleep-idle-level 2; only that path is checked)
Prints JSON; "checks" maps each check to true/false.
"""
import base64
import concurrent.futures as cf
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

BASE = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__))


def call(path, body=None, method=None):
    if body is not None and path.startswith("/v1/"):
        body = {"model": MODEL, **body}
    data = json.dumps(body).encode() if body is not None else None
    method = method or ("POST" if data is not None else "GET")
    req = urllib.request.Request(BASE + path, data, {"Content-Type": "application/json"}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, {"raw": e.read().decode(errors="replace")}


def gpu_used_mib():
    """Memory used on this job's GPUs (MiB), from nvidia-smi."""
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return sum(int(x) for x in out.split())


MODEL = call("/v1/models")[1]["data"][0]["id"]
NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}
with open(os.path.join(HERE, "media", "shapes.png"), "rb") as f:
    PNG = "data:image/png;base64," + base64.b64encode(f.read()).decode()


def text():
    s, r = call("/v1/chat/completions", {"messages": [{"role": "user", "content": "Name three primes."}],
                                         "temperature": 0, "max_tokens": 48, **NO_THINK})
    return r["choices"][0]["message"]["content"] if s == 200 else (s, r)


def image():
    s, r = call("/v1/chat/completions", {"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": PNG}},
        {"type": "text", "text": "What shapes are in the image?"}]}],
        "temperature": 0, "max_tokens": 48, **NO_THINK})
    return r["choices"][0]["message"]["content"] if s == 200 else (s, r)


def level():
    return call("/is_sleeping")[1].get("level")


checks, info = {}, {}
awake_mib = gpu_used_mib()
ref_text, ref_image = text(), image()
info["awake_mib"] = awake_mib

if "--idle" in sys.argv:
    deadline = time.time() + 120
    while level() != 2 and time.time() < deadline:
        time.sleep(1)
    checks["idle sleep reaches level 2"] = level() == 2
    info["idle_asleep_mib"] = gpu_used_mib()
    checks["request wakes it"] = text() == ref_text and level() == 0
    print(json.dumps({"checks": checks, "info": info}, indent=1))
    sys.exit(0)

# Bad input.
checks["level 3 refused"] = call("/sleep?level=3", method="POST")[0] == 400
checks["level 2 with some tags refused"] = call("/sleep?level=2&tags=weights", method="POST")[0] == 400
checks["still awake"] = level() == 0

# Explicit level 2 and wake-up.
t = time.time()
s, r = call("/sleep?level=2", method="POST")
info["sleep2"] = {"status": s, "body": r, "wall_s": round(time.time() - t, 2)}
asleep_mib = gpu_used_mib()
info["asleep_mib"] = asleep_mib
checks["sleep level 2 ok"] = s == 200 and r.get("level") == 2
checks["is_sleeping level 2"] = level() == 2
checks["gpu memory released"] = asleep_mib < awake_mib * 0.1
checks["profile refused while stopped"] = call("/start_profile", method="POST")[0] >= 400
checks["sleep again is a no-op"] = call("/sleep", method="POST")[1].get("level") == 2
checks["partial wake refused"] = call("/wake_up?tags=weights", method="POST")[0] == 400
t = time.time()
s, r = call("/wake_up", method="POST")
info["wake2"] = {"status": s, "body": r, "wall_s": round(time.time() - t, 2)}
checks["wake from level 2 ok"] = s == 200 and r.get("level") == 0
checks["text same after level 2"] = text() == ref_text
checks["image same after level 2"] = image() == ref_image

# Level 2 while a request runs, mode=abort; the next requests wake it.
long_body = {"model": MODEL, "messages": [{"role": "user", "content": "Tell a long story."}],
             "max_tokens": 4000, "ignore_eos": True, **NO_THINK}
with cf.ThreadPoolExecutor(1) as pool:
    fut = pool.submit(call, "/v1/chat/completions", long_body)
    time.sleep(3)
    s, r = call("/sleep?level=2&mode=abort", method="POST")
    checks["abort sleep ok"] = s == 200 and r.get("level") == 2
    ls, lr = fut.result()
    info["aborted_request"] = {"status": ls, "finish": (lr.get("choices") or [{}])[0].get("finish_reason")}
checks["aborted request ended early"] = ls != 200 or lr["choices"][0]["finish_reason"] != "length" \
    or lr["usage"]["completion_tokens"] < 4000
with cf.ThreadPoolExecutor(8) as pool:
    woke = list(pool.map(lambda _: text(), range(8)))
checks["concurrent requests wake it"] = all(w == ref_text for w in woke) and level() == 0

# Level 1 first, then level 2 on top, then wake.
checks["level 1 weights"] = call("/sleep?tags=weights", method="POST")[1].get("level") == 1
checks["level 2 over level 1"] = call("/sleep?level=2", method="POST")[1].get("level") == 2
checks["wake after 1+2"] = call("/wake_up", method="POST")[1].get("level") == 0
# Plain level 1 still round-trips.
checks["level 1 cycle"] = (call("/sleep", method="POST")[1].get("level") == 1
                           and call("/wake_up", method="POST")[1].get("level") == 0)
checks["text same at the end"] = text() == ref_text
print(json.dumps({"checks": checks, "info": info}, indent=1))
