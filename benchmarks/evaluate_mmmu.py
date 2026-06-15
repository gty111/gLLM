"""MMMU validation accuracy on a gllm OpenAI-compatible chat completions endpoint.

Loads MMMU/MMMU (all 30 subjects, validation split) via the HF mirror cache,
encodes each sample's images inline as base64 data URLs, formats a multiple-
choice prompt, and asks the model for a single letter. Runs requests
concurrently and reports per-subject + overall accuracy.

Usage:
    HF_ENDPOINT=https://hf-mirror.com HF_HOME=/path/to/hf_cache \
    python evaluate_mmmu.py --endpoint http://127.0.0.1:34505 \
        --concurrency 64 --out results/mmmu_run.jsonl

The dataset cache location follows the standard ``HF_HOME`` env var
(default ``~/.cache/huggingface``); no paths are hard-coded.
"""

import argparse
import asyncio
import base64
import io
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path

# All HF configuration (HF_ENDPOINT mirror, HF_HOME cache dir, offline flags)
# is left entirely to the caller's environment so this script carries no
# machine-specific defaults. See the module docstring for example env vars.

import aiohttp
from datasets import get_dataset_config_names, load_dataset


# Match the official MMMU multi-choice answer extractor: prefer "Answer: X"
# at the very end, then "(A)/(B)" or a bare standalone letter A-G.
ANSWER_RE_FINAL = re.compile(r"answer\s*[:：]\s*\(?([A-Ga-g])\)?", re.IGNORECASE)
ANSWER_RE_BARE = re.compile(r"\b([A-Ga-g])\b")


def extract_letter(text: str) -> str | None:
    """Pull a single A-G letter from the model's free-form reply."""
    if not text:
        return None
    m = ANSWER_RE_FINAL.search(text)
    if m:
        return m.group(1).upper()
    # Fall back: take the last standalone letter mention (model often ends
    # with "The answer is C").
    matches = ANSWER_RE_BARE.findall(text)
    if matches:
        return matches[-1].upper()
    return None


def image_to_data_url(img, max_pixels: int) -> str:
    """Inline a PIL image as a JPEG data URL. Downscale by area only when
    above ``max_pixels`` to keep prompt length manageable for very large
    document scans; aspect ratio is preserved."""
    w, h = img.size
    if w * h > max_pixels:
        scale = (max_pixels / (w * h)) ** 0.5
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def build_messages(sample: dict, max_pixels: int) -> tuple[list[dict], list[str], str]:
    """Convert a MMMU sample into OpenAI chat ``messages`` plus the option
    letters / question type metadata needed downstream.

    Image markers ``<image N>`` are replaced inline at their natural
    position in the question; missing images for unreferenced slots are
    silently skipped.
    """
    q = sample["question"]
    options_raw = sample["options"]
    try:
        options = eval(options_raw) if isinstance(options_raw, str) else options_raw
    except Exception:
        options = []
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    qtype = sample["question_type"]

    if qtype == "multiple-choice" and options:
        opt_lines = "\n".join(f"({letters[i]}) {opt}" for i, opt in enumerate(options))
        # The standard MMMU zero-shot prompt asks for the letter directly;
        # Qwen3-VL "Instruct" still tends to reason out loud, so we also
        # accept a final "Answer: X" line via :func:`extract_letter`.
        suffix = (
            f"\n\nOptions:\n{opt_lines}\n\n"
            "Answer with the option's letter from the given choices directly. "
            "End your reply with 'Answer: X'."
        )
        used_letters = letters[: len(options)]
    else:
        suffix = (
            "\n\nAnswer the question briefly. "
            "End your reply with 'Answer: <your answer>'."
        )
        used_letters = ""

    # Replace <image N> markers in-line; build chat content as an array of
    # parts so the image appears at the correct textual position.
    parts: list[dict] = []
    pieces = re.split(r"(<image\s*\d+>)", q)
    used_imgs = 0
    for piece in pieces:
        if not piece:
            continue
        marker = re.match(r"<image\s*(\d+)>", piece)
        if marker:
            idx = int(marker.group(1))
            img = sample.get(f"image_{idx}")
            if img is not None:
                parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": image_to_data_url(img, max_pixels)},
                    }
                )
                used_imgs += 1
        else:
            parts.append({"type": "text", "text": piece})

    # Some samples reference images that never appear inline; append them.
    for i in range(1, 8):
        if used_imgs >= 7:
            break
        img = sample.get(f"image_{i}")
        if img is None:
            continue
        if f"<image {i}>" in q or f"<image{i}>" in q:
            continue
        parts.append(
            {
                "type": "image_url",
                "image_url": {"url": image_to_data_url(img, max_pixels)},
            }
        )
        used_imgs += 1

    parts.append({"type": "text", "text": suffix})

    return (
        [{"role": "user", "content": parts}],
        used_letters,
        qtype,
    )


async def query_one(
    session: aiohttp.ClientSession,
    endpoint: str,
    model: str,
    sample: dict,
    max_tokens: int,
    max_pixels: int,
    timeout_s: int,
    no_thinking: bool = False,
) -> dict:
    # build_messages does CPU-bound work (PIL resize + JPEG + base64). Run it
    # in a worker thread so it doesn't block the event loop and starve other
    # concurrent requests.
    messages, _used_letters, qtype = await asyncio.to_thread(
        build_messages, sample, max_pixels
    )
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    if no_thinking:
        # Kimi-K2.5's chat template gates its reasoning block on a ``thinking``
        # variable; vllm exposes per-request template vars via
        # ``chat_template_kwargs``. Disabling it makes the model answer
        # directly (matching a server launched with thinking off), so the
        # ``max_tokens`` budget isn't consumed by an unfinished reasoning
        # trace that never reaches the final "Answer: X".
        payload["chat_template_kwargs"] = {"thinking": False}
    start = time.time()
    try:
        async with session.post(
            f"{endpoint}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout_s),
        ) as resp:
            data = await resp.json()
            content = data["choices"][0]["message"]["content"]
    except Exception as e:
        content = f"<ERROR: {e}>"
    elapsed = time.time() - start

    pred_letter = extract_letter(content) if qtype == "multiple-choice" else None
    gold = sample["answer"]

    if qtype == "multiple-choice":
        correct = pred_letter is not None and pred_letter == gold.upper()
    else:
        # Open-ended: rough containment check on the last line.
        last_line = content.strip().split("\n")[-1]
        last_line = re.sub(r"^answer\s*[:：]\s*", "", last_line, flags=re.IGNORECASE)
        correct = gold.strip().lower() in last_line.lower()

    return {
        "id": sample["id"],
        "subject": sample["id"].split("_")[1],
        "question_type": qtype,
        "gold": gold,
        "pred_letter": pred_letter,
        "raw_pred": content,
        "correct": bool(correct),
        "elapsed_s": elapsed,
    }


async def run(args):
    # ``get_dataset_config_names`` + ``load_dataset("MMMU/MMMU", config)``
    # both hit the dataset's top-level metadata (config 'default'), which
    # isn't part of the per-subject cache layout this repo downloads in
    # offline mode. The 30 subject configs live as raw arrow files under
    # ``${HF_HOME}/datasets/MMMU___mmmu/<Subject>/<version>/<hash>/`` --
    # we walk that directory directly so ``HF_HUB_OFFLINE=1`` /
    # ``HF_DATASETS_OFFLINE=1`` runs don't need to reach the hub for the
    # missing 'default' descriptor.
    _MMMU_SUBJECTS = [
        "Accounting", "Agriculture", "Architecture_and_Engineering",
        "Art", "Art_Theory", "Basic_Medical_Science", "Biology",
        "Chemistry", "Clinical_Medicine", "Computer_Science", "Design",
        "Diagnostics_and_Laboratory_Medicine", "Economics", "Electronics",
        "Energy_and_Power", "Finance", "Geography", "History",
        "Literature", "Manage", "Marketing", "Materials", "Math",
        "Mechanical_Engineering", "Music", "Pharmacy", "Physics",
        "Psychology", "Public_Health", "Sociology",
    ]

    def _load_one(name: str):
        try:
            return load_dataset("MMMU/MMMU", name, split="validation")
        except Exception:
            # Offline fallback: load the cached arrow file directly.
            from datasets import Dataset
            hf_home = os.environ.get(
                "HF_HOME", os.path.expanduser("~/.cache/huggingface")
            )
            subj_root = Path(hf_home) / "datasets" / "MMMU___mmmu" / name
            # Pick the deepest <version>/<hash> dir containing the
            # validation arrow shard. There's normally only one but be
            # robust to multi-hash caches (e.g. after a re-download).
            cand = sorted(subj_root.glob("*/*/mmmu-validation.arrow"))
            if not cand:
                raise FileNotFoundError(
                    f"Cannot find cached MMMU/{name} validation arrow under {subj_root}"
                )
            return Dataset.from_file(str(cand[-1]))

    if args.data_dir:
        # Local parquet mode: load each ``<Subject>/validation.parquet`` from a
        # directory (e.g. downloaded via hf-mirror). Bypasses the hub entirely
        # -- ``datasets.Dataset.from_parquet`` decodes the embedded images.
        from datasets import Dataset
        from pathlib import Path as _Path

        names = []
        samples = []
        for sub in sorted(_Path(args.data_dir).iterdir()):
            pq = sub / "validation.parquet"
            if not pq.exists():
                continue
            names.append(sub.name)
            ds = Dataset.from_parquet(str(pq))
            for ex in ds:
                samples.append(ex)
        if args.limit > 0:
            samples = samples[: args.limit]
        print(
            f"Loaded {len(samples)} MMMU validation samples across "
            f"{len(names)} subjects (local parquet).",
            flush=True,
        )
    else:
        if os.environ.get("HF_HUB_OFFLINE") or os.environ.get(
            "HF_DATASETS_OFFLINE"
        ):
            # Offline: skip the hub-only descriptor lookup and use the
            # canonical subject list (matches the 30 cached configs).
            names = _MMMU_SUBJECTS
        else:
            try:
                names = get_dataset_config_names("MMMU/MMMU")
            except Exception:
                names = _MMMU_SUBJECTS
        samples = []
        for n in names:
            ds = _load_one(n)
            for ex in ds:
                samples.append(ex)
        if args.limit > 0:
            samples = samples[: args.limit]
        print(f"Loaded {len(samples)} MMMU validation samples across {len(names)} subjects.", flush=True)

    timeout = aiohttp.ClientTimeout(total=args.timeout)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    sem = asyncio.Semaphore(args.concurrency)
    start = time.time()
    completed = 0
    correct_count = 0

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async def worker(sample):
            nonlocal completed, correct_count
            async with sem:
                try:
                    r = await query_one(
                        session,
                        args.endpoint,
                        args.model,
                        sample,
                        args.max_tokens,
                        args.max_pixels,
                        args.timeout,
                        args.no_thinking,
                    )
                except Exception as exc:
                    # Don't let one broken sample (bad image, unparseable
                    # options, transient HTTP error) abort the whole run; record
                    # it as a miss and keep going. Keep the same schema as the
                    # success path so the JSONL output stays consistent.
                    sid = sample.get("id", "?")
                    r = {
                        "id": sid,
                        "subject": sid.split("_")[1] if "_" in sid else "?",
                        "question_type": sample.get("question_type", "?"),
                        "gold": sample.get("answer", ""),
                        "pred_letter": None,
                        "raw_pred": f"<error: {type(exc).__name__}: {exc}>",
                        "correct": False,
                        "elapsed_s": 0.0,
                    }
                completed += 1
                if r["correct"]:
                    correct_count += 1
                if completed % 10 == 0 or completed == len(samples):
                    elapsed = time.time() - start
                    rate = completed / max(elapsed, 1e-3)
                    eta = (len(samples) - completed) / max(rate, 1e-3)
                    print(
                        f"[{completed}/{len(samples)}] acc={correct_count/completed:.3f} "
                        f"rate={rate:.2f} req/s eta={eta/60:.1f}min",
                        flush=True,
                    )
                return r

        tasks = [asyncio.create_task(worker(s)) for s in samples]
        for t in asyncio.as_completed(tasks):
            r = await t
            results.append(r)

    with open(out_path, "w") as f:
        for r in results:
            f.write(
                json.dumps(
                    {k: v for k, v in r.items() if k != "raw_pred"} | {"raw_pred": r["raw_pred"][:500]},
                    ensure_ascii=False,
                )
                + "\n"
            )

    by_subject = defaultdict(lambda: [0, 0])
    by_qtype = defaultdict(lambda: [0, 0])
    n_total = 0
    n_correct = 0
    for r in results:
        n_total += 1
        by_subject[r["subject"]][1] += 1
        by_qtype[r["question_type"]][1] += 1
        if r["correct"]:
            n_correct += 1
            by_subject[r["subject"]][0] += 1
            by_qtype[r["question_type"]][0] += 1

    print()
    print("=" * 70)
    print(f"MMMU validation accuracy: {n_correct}/{n_total} = {n_correct/max(n_total,1):.4f}")
    print("=" * 70)
    print(f"\nBy question type:")
    for qt, (c, t) in sorted(by_qtype.items()):
        print(f"  {qt:>20s}: {c}/{t} = {c/max(t,1):.4f}")
    print(f"\nBy subject:")
    for subj in sorted(by_subject):
        c, t = by_subject[subj]
        print(f"  {subj:>40s}: {c:3d}/{t:3d} = {c/max(t,1):.4f}")
    print()
    print(f"Results written to: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:34505")
    parser.add_argument("--model", default="qwen3-vl")
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-pixels", type=int, default=1280 * 1280,
                        help="downscale images so width*height stays under this")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--limit", type=int, default=0,
                        help="0 = all samples; otherwise take the first N (after subject ordering)")
    parser.add_argument("--out", default="benchmarks/results/mmmu_run.jsonl")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Load MMMU from a local dir of <Subject>/validation.parquet files "
        "instead of the HF hub (offline).",
    )
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help="Send chat_template_kwargs={'thinking': False} so reasoning models "
        "(e.g. Kimi-K2.5 on vllm) answer directly instead of emitting a long "
        "reasoning trace that gets truncated by --max-tokens.",
    )
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
