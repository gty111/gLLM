"""Throughput and TTFT of one front end: streamed chat requests at fixed concurrency."""
import asyncio
import json
import statistics
import sys
import time

import aiohttp

BASE, CONC, N, MAX_TOKENS = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
# Optional: "rand" spreads max_tokens over [M/2, 3M/2] (fixed seed); a trace path logs per-request times.
RAND = len(sys.argv) > 5 and sys.argv[5] == "rand"
TRACE = sys.argv[6] if len(sys.argv) > 6 else None
import random
_rng = random.Random(0)
LENS = [(_rng.randint(MAX_TOKENS // 2, MAX_TOKENS * 3 // 2) if RAND else MAX_TOKENS) for _ in range(N)]
client_trace = []


async def main():
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3600),
                                     connector=aiohttp.TCPConnector(limit=0)) as s:
        async with s.get(BASE + "/v1/models") as r:
            model = (await r.json())["data"][0]["id"]
        sem = asyncio.Semaphore(CONC)
        ttft, tokens, errors = [], [0], []

        async def one(i):
            body = {"model": model, "stream": True, "max_tokens": LENS[i],
                    "ignore_eos": True, "temperature": 0.6, "top_p": 0.95, "top_k": 20,
                    "stream_options": {"include_usage": True},
                    "chat_template_kwargs": {"enable_thinking": False},
                    "messages": [{"role": "user", "content": f"Tell me a long story about robot {i}."}]}
            async with sem:
                t0 = time.perf_counter()
                first = None
                try:
                    async with s.post(BASE + "/v1/chat/completions", json=body) as r:
                        if r.status != 200:
                            errors.append(await r.text()); return
                        async for raw in r.content:
                            line = raw.decode().strip()
                            if not line.startswith("data: ") or line == "data: [DONE]":
                                continue
                            chunk = json.loads(line[6:])
                            if first is None and chunk.get("choices"):
                                delta = chunk["choices"][0].get("delta", {})
                                if delta.get("content") or delta.get("reasoning_content"):
                                    first = time.perf_counter() - t0
                            if chunk.get("usage"):
                                tokens[0] += chunk["usage"]["completion_tokens"]
                except Exception as e:
                    errors.append(repr(e)); return
                if first is not None:
                    ttft.append(first)
                client_trace.append((t0, t0 + (first or 0), time.perf_counter()))

        t = time.perf_counter()
        await asyncio.gather(*(one(i) for i in range(N)))
        dt = time.perf_counter() - t
    ttft.sort()
    if TRACE:
        off = time.time() - time.perf_counter()
        with open(TRACE, "w") as f:
            for a, b, c in client_trace:
                f.write(f"{a + off:.6f} {b + off:.6f} {c + off:.6f}\n")
    print(f"{BASE} conc={CONC} n={N} errors={len(errors)} tok/s={tokens[0] / dt:.0f} "
          f"ttft_p50={statistics.median(ttft) * 1000:.0f}ms "
          f"ttft_p90={ttft[int(len(ttft) * 0.9)] * 1000:.0f}ms time={dt:.1f}s")
    for e in errors[:3]:
        print("  ", e[:200])


asyncio.run(main())
