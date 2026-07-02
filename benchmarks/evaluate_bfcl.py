"""BFCL-V4 (Berkeley Function-Calling Leaderboard) accuracy against any
OpenAI-compatible ``/v1/chat/completions`` endpoint (e.g. a gLLM / vLLM /
SGLang server).

Why this design
---------------
The official ``bfcl`` harness couples generation, the model registry, and
scoring together. Here we only borrow the *data* and the *scorer* from the
official ``bfcl_eval`` package and drive generation ourselves over a plain
OpenAI-compatible HTTP API, so the same script works against any server and
any model -- it is intentionally engine-agnostic.

Two function-calling modes:

* ``prompt`` (default, recommended / most portable): the available tools are
  embedded into the system prompt (exactly the official BFCL prompting
  template) and the model is asked to emit ``[func(arg=val), ...]`` as text.
  This needs nothing special from the server -- no server-side tool-call
  parser -- so it works against every chat endpoint.
* ``native``: the tools are sent via the OpenAI ``tools`` field and the
  structured ``tool_calls`` in the response are scored. This requires the
  server to parse the model's native tool-call markup into ``tool_calls``.

Scoring reuses the official ``bfcl_eval`` AST checker, so the numbers are
directly comparable to the public leaderboard for the supported categories:
the single-turn AST categories (``simple_python/java/javascript``,
``multiple``, ``parallel``, ``parallel_multiple`` and their ``live_*``
variants) plus relevance/irrelevance detection. Executable, multi-turn,
memory and web-search categories require a stateful sandbox / live APIs and
are out of scope for this lightweight client.

Setup
-----
``bfcl_eval`` pulls in heavy, version-pinned deps (numpy==1.26.4,
sentence-transformers, ...), so install it in its *own* venv rather than the
serving env, then run this script with that venv's Python::

    python -m venv /path/to/bfcl_venv
    /path/to/bfcl_venv/bin/pip install bfcl-eval
    /path/to/bfcl_venv/bin/python benchmarks/evaluate_bfcl.py \
        --endpoint http://127.0.0.1:8000 --model qwen3.5 \
        --test-category simple_python,multiple,parallel,parallel_multiple,irrelevance \
        --num-per-category 50 --concurrency 32

Example (full single-turn non-live AST suite)::

    .../python benchmarks/evaluate_bfcl.py --endpoint http://127.0.0.1:8000 \
        --model qwen3.5 --test-category non_live --concurrency 32
"""

import argparse
import copy
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

try:
    from bfcl_eval.constants.enums import Language, ModelStyle, ReturnFormat
    from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker
    from bfcl_eval.model_handler.utils import (
        convert_to_tool,
        default_decode_ast_prompting,
        system_prompt_pre_processing_chat_model,
    )
    from bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI
    from bfcl_eval.utils import (
        is_function_calling_format_output,
        is_empty_output,
        is_java,
        is_js,
        is_relevance_or_irrelevance,
        load_dataset_entry,
        load_ground_truth_entry,
        parse_test_category_argument,
    )
except ImportError as exc:  # pragma: no cover - environment guard
    sys.exit(
        "Could not import `bfcl_eval`. Install it in a dedicated venv and run "
        "this script with that venv's Python:\n"
        "  python -m venv /path/to/bfcl_venv\n"
        "  /path/to/bfcl_venv/bin/pip install bfcl-eval\n"
        f"Original error: {exc}"
    )


# Categories this lightweight client can score. Everything else (executable,
# multi-turn, memory, web-search, format-sensitivity) needs a stateful
# sandbox / live APIs and is skipped with a warning.
def _is_supported(category: str) -> bool:
    if is_relevance_or_irrelevance(category):
        return True
    unsupported_markers = (
        "multi_turn",
        "exec",
        "rest",
        "sql",
        "memory",
        "web_search",
        "format_sensitivity",
        "chatable",
    )
    return not any(m in category for m in unsupported_markers)


def _language_for(category: str):
    """Return the (Language, ReturnFormat) pair the AST checker/decoder expects."""
    if is_java(category):
        return Language.JAVA, ReturnFormat.JAVA
    if is_js(category):
        return Language.JAVASCRIPT, ReturnFormat.JAVASCRIPT
    return Language.PYTHON, ReturnFormat.PYTHON


def _strip_reasoning(text: str) -> str:
    """Drop a leading ``<think>...</think>`` block that reasoning models emit
    before their answer, mirroring BFCL's Qwen handler."""
    if text and "</think>" in text:
        text = text.split("</think>")[-1]
    return text.lstrip("\n")


def build_chat_payload(entry, mode, model, max_tokens, temperature, no_thinking):
    """Turn a BFCL test entry into an OpenAI chat-completions request body.

    For ``prompt`` mode the tool docs are folded into the system prompt and the
    model is asked to answer in text. For ``native`` mode the tools are sent in
    the OpenAI ``tools`` field.
    """
    # Single-turn categories carry exactly one "turn" (a list of messages).
    messages = copy.deepcopy(entry["question"][0])
    functions = copy.deepcopy(entry["function"])

    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }

    if mode == "native":
        # Hand the tools to the server and let it emit structured tool_calls.
        # convert_to_tool sanitizes names to satisfy OpenAI's
        # ^[a-zA-Z0-9_-]{1,64}$ rule (e.g. ``math.factorial`` -> ``math_factorial``).
        # Keep the sanitized names: this is the portable convention shared by
        # all OpenAI-compatible servers and their native tool parsers (vLLM's
        # ``kimi_k2`` parser, etc.), and it matches how BFCL's native-FC
        # handlers talk to the model. The scoring side symmetrically maps the
        # gold names via ``convert_func_name`` whenever the chosen
        # ``--bfcl-model-name`` config has ``underscore_to_dot=True``, so the
        # comparison stays consistent without any dotted-name round-trip.
        tools = convert_to_tool(
            functions, GORILLA_TO_OPENAPI, ModelStyle.OPENAI_COMPLETIONS
        )
        payload["messages"] = messages
        payload["tools"] = tools
        # NB: don't send tool_choice="auto" -- gLLM only accepts "none" or a
        # named-tool dict, and rejects "auto" with HTTP 422. Omitting it lets
        # the model freely decide whether/which tool to call (what BFCL wants,
        # including the relevance/irrelevance categories).
    else:
        # Prompt mode: embed the official BFCL function-calling system prompt.
        messages = system_prompt_pre_processing_chat_model(
            messages, functions, entry["id"]
        )
        payload["messages"] = messages

    if no_thinking:
        # gLLM / vLLM forward these per-request to the chat template; the var
        # name differs across models so send both.
        payload["chat_template_kwargs"] = {
            "enable_thinking": False,
            "thinking": False,
        }
    return payload


def call_endpoint(session, url, payload, timeout):
    resp = session.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def extract_native_calls(message):
    """Convert OpenAI ``tool_calls`` into BFCL's decoded AST format:
    ``[{func_name: {arg: val, ...}}, ...]``."""
    decoded = []
    for tc in message.get("tool_calls") or []:
        fn = tc.get("function", {})
        name = fn.get("name")
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {}
        if name:
            decoded.append({name: args})
    return decoded


def evaluate_entry(entry, possible_answer, checker_function, category, args, session, url):
    """Run one test entry end-to-end and return a result record.

    ``entry`` carries the prompt-side function doc (with language-specific hints; for
    Java/JS its param types are collapsed to ``string``), while ``checker_function``
    is the *raw* function doc (real ``String``/``integer``/``ArrayList`` types) that
    the AST checker needs for type conversion.
    """
    language, return_format = _language_for(category)
    record = {
        "id": entry["id"],
        "category": category,
        "valid": False,
        "error": None,
    }

    payload = build_chat_payload(
        entry,
        args.mode,
        args.model,
        args.max_tokens,
        args.temperature,
        args.no_thinking,
    )

    try:
        data = call_endpoint(session, url, payload, args.timeout)
        message = data["choices"][0]["message"]
    except Exception as exc:  # network / server error -> miss, keep going
        record["error"] = f"request_error: {exc}"
        record["raw"] = ""
        return record

    raw_content = message.get("content") or ""

    # Decode the model output into BFCL's AST format.
    try:
        if args.mode == "native" and message.get("tool_calls"):
            decoded = extract_native_calls(message)
        else:
            decoded = default_decode_ast_prompting(
                _strip_reasoning(raw_content), return_format, has_tool_call_tag=False
            )
        decode_ok = True
        decode_error = None
    except Exception as exc:
        decoded = None
        decode_ok = False
        decode_error = str(exc)

    record["raw"] = raw_content[:1000]
    record["decoded"] = decoded if isinstance(decoded, list) else str(decoded)

    # Relevance / irrelevance: correctness is about whether a (valid, non-empty)
    # function call was produced at all.
    if is_relevance_or_irrelevance(category):
        contains_call = decode_ok and not is_empty_output(decoded)
        if "irrelevance" in category:
            record["valid"] = not contains_call
        else:  # relevance
            record["valid"] = contains_call
        if not record["valid"] and decode_error:
            record["error"] = decode_error
        return record

    # AST categories: decode must succeed and be in the right shape, then run
    # the official checker against the possible-answer set.
    if not decode_ok:
        record["error"] = f"decode_failed: {decode_error}"
        return record
    if not is_function_calling_format_output(decoded):
        record["error"] = "wrong_output_format"
        return record

    try:
        checker = ast_checker(
            checker_function,
            decoded,
            possible_answer["ground_truth"],
            language,
            category,
            args.bfcl_model_name,
        )
        record["valid"] = bool(checker.get("valid"))
        if not record["valid"]:
            record["error"] = checker.get("error")
            record["error_type"] = checker.get("error_type")
    except Exception as exc:
        record["error"] = f"checker_error: {exc}"
    return record


def run_category(category, args, session, url):
    """Generate + score one category, returning (records, n_correct, n_total)."""
    entries = load_dataset_entry(category)
    # The default loader adds language-specific hints and, for Java/JS, collapses every
    # param type to ``string`` -- that's the doc the *model* should see. The AST checker
    # instead needs the *raw* function doc (real ``String``/``integer``/``ArrayList``
    # types), so load a second, un-hinted copy keyed by id for scoring (this mirrors the
    # official harness, which loads the prompt with hints and the checker without).
    raw_funcs = {
        e["id"]: e["function"]
        for e in load_dataset_entry(category, include_language_specific_hint=False)
    }
    # Relevance / irrelevance categories ship no possible-answer file (correctness
    # is just "did the model (not) call a function"), so only load ground truth for
    # the AST-scored categories.
    if is_relevance_or_irrelevance(category):
        answers = {}
    else:
        answers = {a["id"]: a for a in load_ground_truth_entry(category)}
    if args.num_per_category > 0:
        entries = entries[: args.num_per_category]

    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(
                evaluate_entry,
                entry,
                answers.get(entry["id"], {"ground_truth": []}),
                raw_funcs.get(entry["id"], entry["function"]),
                category,
                args,
                session,
                url,
            ): entry["id"]
            for entry in entries
        }
        done = 0
        for fut in as_completed(futures):
            results.append(fut.result())
            done += 1
            if done % 20 == 0 or done == len(entries):
                acc = sum(r["valid"] for r in results) / max(done, 1)
                print(
                    f"  [{category}] {done}/{len(entries)} running_acc={acc:.3f}",
                    flush=True,
                )

    n_correct = sum(r["valid"] for r in results)
    return results, n_correct, len(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000",
                        help="OpenAI-compatible server base URL (no trailing /v1).")
    parser.add_argument("--model", required=True,
                        help="Model name sent in the request body / served name.")
    parser.add_argument(
        "--bfcl-model-name", default="kimi-k2-0905-preview-FC",
        help="A model name registered in bfcl_eval, used ONLY by the AST "
        "checker for function-name normalization (does not affect generation). "
        "Pick a native-FC config with underscore_to_dot=True (the default) so "
        "the gold function names are converted to match the underscore-"
        "sanitized names that OpenAI-compatible servers emit.",
    )
    parser.add_argument(
        "--test-category", default="simple_python,multiple,parallel,parallel_multiple,irrelevance",
        help="Comma-separated BFCL categories or collection names (e.g. "
        "'non_live', 'live', 'simple_python'). Unsupported categories "
        "(multi_turn/exec/memory/web_search/format_sensitivity) are skipped.",
    )
    parser.add_argument("--mode", choices=["prompt", "native"], default="native",
                        help="native (default): OpenAI tools field + structured "
                        "tool_calls (needs a server-side tool parser). "
                        "prompt: tools in system prompt (portable, no parser needed).")
    parser.add_argument("--num-per-category", type=int, default=0,
                        help="0 = all entries; otherwise take the first N per category.")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--no-thinking", action="store_true",
                        help="Disable a reasoning model's <think> block via "
                        "chat_template_kwargs (recommended for fair AST scoring).")
    parser.add_argument("--out", default="benchmarks/results/bfcl_run.jsonl")
    args = parser.parse_args()

    url = args.endpoint.rstrip("/") + "/v1/chat/completions"

    requested = parse_test_category_argument(args.test_category.split(","))
    categories = [c for c in requested if _is_supported(c)]
    skipped = [c for c in requested if not _is_supported(c)]
    if skipped:
        print(f"Skipping unsupported categories (need sandbox/live APIs): {skipped}")
    if not categories:
        sys.exit("No supported categories to run.")

    print(f"Endpoint : {url}")
    print(f"Model    : {args.model}  (mode={args.mode}, no_thinking={args.no_thinking})")
    print(f"Categories: {categories}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    all_results = []
    per_cat = {}
    start = time.time()
    for category in categories:
        results, n_correct, n_total = run_category(category, args, session, url)
        per_cat[category] = (n_correct, n_total)
        all_results.extend(results)
        acc = n_correct / max(n_total, 1)
        print(f"==> {category:22s}: {n_correct}/{n_total} = {acc:.4f}")

    with out_path.open("w") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    total_correct = sum(c for c, _ in per_cat.values())
    total_n = sum(t for _, t in per_cat.values())
    macro = sum(c / max(t, 1) for c, t in per_cat.values()) / max(len(per_cat), 1)

    print("\n" + "=" * 64)
    print("BFCL-V4 results")
    print("=" * 64)
    for cat in sorted(per_cat):
        c, t = per_cat[cat]
        print(f"  {cat:24s}: {c:4d}/{t:<4d} = {c / max(t, 1):.4f}")
    print("-" * 64)
    print(f"  {'MICRO (overall)':24s}: {total_correct:4d}/{total_n:<4d} = "
          f"{total_correct / max(total_n, 1):.4f}")
    print(f"  {'MACRO (avg of cats)':24s}: {macro:.4f}")
    print(f"\nElapsed: {time.time() - start:.1f}s   Results: {out_path}")


if __name__ == "__main__":
    main()
