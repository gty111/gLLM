# Structured output

gLLM supports token-level constrained JSON generation using XGrammar. No model
forward changes or model-specific JSON parsers are required. Install the pinned
`xgrammar` and `jsonschema` dependencies from `requirements.txt`.

Chat Completions uses `response_format`:

```python
response = client.chat.completions.create(
    model="Qwen/Qwen3.8-27B",
    messages=[{"role": "user", "content": "What is 6 times 7? Return JSON."}],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "answer",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"answer": {"type": "integer"}},
                "required": ["answer"],
                "additionalProperties": False,
            },
        },
    },
    max_completion_tokens=256,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
```

Responses uses the same schema in the flattened `text.format` shape:

```python
response = client.responses.create(
    model="Qwen/Qwen3.8-27B",
    input="What is 6 times 7? Return JSON.",
    text={"format": {
        "type": "json_schema", "name": "answer", "strict": True,
        "schema": {
            "type": "object",
            "properties": {"answer": {"type": "integer"}},
            "required": ["answer"], "additionalProperties": False,
        },
    }},
    reasoning={"effort": "none"},
    max_output_tokens=256,
)
```

Both endpoints also support `{"type": "json_object"}` (valid JSON object,
without schema-specific field constraints), and streaming. Stream chunks are
JSON fragments, not individually parseable JSON documents. Length-limited
output can be incomplete: check Chat `finish_reason` or Responses `status`.
Constrained syntax does not guarantee factual or semantic correctness.

## Supported schema subset

Types, object `properties` / `required` / `additionalProperties`, array `items`,
`enum`, `const`, `anyOf`, local `$ref`, `$defs` and `definitions` are supported.
String `minLength` / `maxLength` bounds are supported on explicitly typed string
nodes (including nullable strings). Bounds must be nonnegative integers, with
`minLength <= maxLength` when both are present. Schemas are passed directly to
XGrammar's `compile_json_schema`, including string length constraints; gLLM
does not rewrite its grammar. This supports ordinary title schemas such as
`minLength: 1` and `maxLength: 36`.

String escaping and length enforcement follow the pinned XGrammar version.
In XGrammar 0.2.7, the bounded-string rule excludes JSON escape sequences and
can accept some unescaped control characters. Consequently, length-constrained
strings do not support the full range of valid JSON string representations.
Compilation uses the existing per-process compiler cache.

`title`, `description`, `default`, and `$schema` are accepted as annotations.
Other keywords (including numeric ranges, string patterns, and array length
bounds) are rejected with HTTP 400 rather than silently ignored. Boolean
subschemas are not supported (boolean `additionalProperties` is supported).
Schemas are limited to 64 KiB and 32 nested schema levels.

`$ref` and `anyOf` may have annotation siblings, but sibling constraints are
rejected because the backend does not enforce their intersection. Every
`required` name must appear in `properties`. `enum` / `const` values must satisfy
all other constraints on their node; for example, `type: string` with
`enum: [1, "x"]` is rejected. Ordinary `type: string` with `enum: ["x", "y"]`
remains supported. These checks apply even when `strict` is false.

`strict: true` requires an object root, all properties listed in `required`, and
`additionalProperties: false` on every object. Nullable fields can use a union
with `null`. When strict is omitted or false, the declared schema is still
constrained, but these additional strict-shape requirements are not imposed.

## Runtime behavior and limits

- Ordinary requests without native reasoning do not allocate masks or perform
  grammar GPU operations. Native reasoning requests use an EOS-only guard even
  without a JSON schema, including requests with tools. This guard does not
  compile a grammar or constrain tool arguments. It shares literal-boundary
  rules with the API parser and permits EOS only when the reasoning block can
  close at EOF. The output-token budget still applies; this does not guarantee
  that a model will eventually produce an answer. Explicit `ignore_eos` retains
  its existing behavior.
- Compilers are cached per tokenizer/vocabulary/stop-token set; mutable matchers
  are request-local and never serialized over IPC.
- After preemption, intermediate re-prefill samples are discarded. The final
  chunk restores grammar from committed output tokens, including on TP/PP
  followers, so scheduler copies cannot lose the request's grammar history.
- Native single-token `<think>` / `</think>` boundaries are recognized from the
  actual generation prefix. Reasoning is unconstrained until its closing marker;
  EOS is masked during reasoning. Models with other reasoning conventions need
  an adapter or thinking disabled before using this feature.
- The EOS-only guard uses the API's literal-aware boundary rules. The JSON
  grammar's existing native-token reasoning transition is unchanged; this
  change does not add full reasoning/tool structural constraints or retries.
- The ordinary forward CUDA Graph remains enabled. CPU matching and GPU mask
  application take place outside the model graph.
- Overlap retains GPU model execution but grammar preparation waits for the
  previous authoritative sampled token. This is a correctness-first CPU/GPU
  feedback boundary, not a zero-overhead fully asynchronous grammar backend.
- The runner submits forward before preparing the grammar mask; CPU grammar
  updates can therefore overlap GPU forward. Sampling applies the prepared mask
  without repeating that wait. Each batch owns its pinned host mask, and token
  feedback uses the runner's existing copy stream.
- PP retains its existing in-flight microbatch depth, including mixed batches;
  it does not serialize structured requests. Qwen3.8-27B PP2 has passed the
  online mixed-concurrency suite. PP sampled-token broadcasts require `int64`
  on both ends, so FlashInfer's `int32` random samples are normalized before
  communication (greedy argmax already returns `int64`). PP uses full decode
  graphs; the existing runner does not enable piecewise prefill graphs with PP.
- MTP supports structured output, including greedy and rejection sampling.
  Target logits are grammar-masked at each speculative position before
  temperature/top-k/top-p; the draft remains an unconstrained proposal. A forked
  matcher explores candidates, and only the accepted prefix advances committed
  grammar state. The relay bonus is consumed exactly once on the next step.
  With overlap scheduling, greedy batches retain asynchronous acceptance/commit
  and GPU-resident relay state, including mixed structured/ordinary batches.
  Candidates are copied ahead of verify; after submitting verify, the CPU reads
  the preceding accepted prefix and prepares masks before sampling. Reading a
  completion for grammar does not retire its scheduler slot. Random/rejection
  sampling retains synchronous acceptance/commit. Model draft/verify CUDA
  graphs remain enabled in both cases. Generation logprobs or
  repetition penalties select ordinary decoding for the batch because the MTP
  verifier does not implement those options. Grammar-aware DSpark is not yet
  implemented.
- Active tools combined with structured output, `ignore_eos`, regex/EBNF formats,
  and strict tool-argument constrained decoding are not supported yet. Use
  `tool_choice="none"` to disable offered tools for a structured text response.
- The existing custom-tool grammar validation is separate and remains post-hoc.
- DP carries the constraint with each request and uses replica-local grammar
  state, but structured output has not yet been GPU-validated with DP. MTP is
  currently disabled under DP-attention independently of structured output.
  Combined PP + DP disables overlap in the existing engine.

### Overlap ordering

The upstream implementations provide useful ordering principles, not drop-in
schedulers for gLLM:

- vLLM separates model execution from sampling. Its engine can dispatch the
  current forward, consume a previous result to advance grammar state, then
  dispatch deferred sampling with the updated mask. See
  [the engine batch queue](https://github.com/vllm-project/vllm/blob/main/vllm/v1/engine/core.py)
  (`step_with_batch_queue`) and the async scheduler's
  `pending_structured_output_tokens` flag.
- SGLang's ordinary overlap loop also defers grammar-dependent sampling until
  the preceding result is processed. See
  [the scheduler](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py)
  (`launch_batch_sample_if_needed`) and
  [the worker](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/tp_worker.py)
  (`delay_sample_func`). Its
  [PP loop](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler_pp_mixin.py)
  has separate microbatch/output-exchange ordering; the ordinary overlap loop
  is not evidence that the same implementation works unchanged with PP.

gLLM exposes the grammar-ready boundary between forward and sampling through
the existing runner and sampler. Only authoritative sampled tokens advance a
request's matcher. Independent PP microbatches retain the existing in-flight
depth. Greedy structured MTP uses the same deferred completion queue and GPU
relay as ordinary greedy MTP. Grammar still needs authoritative accepted tokens,
but waits at the post-forward, pre-sampling boundary instead of draining the
speculative pipeline. This
integration does not replace the existing scheduler with an upstream scheduler
or copy a second speculative-execution framework.

## Tests

CPU tests: `pytest -q tests/test_structured_output.py tests/test_structured_string_lengths.py`.

`tests/run_structured_output.py` is a Slurm-only online integration harness for
Qwen3.8-27B: greedy and random sampling, mixed schemas and ordinary requests,
streaming, reasoning, chunked prefill, truncation, Chat/Responses, TP2, PP2,
overlap, and an ordinary-request baseline. The `before` mode runs the same
structured suite against `--baseline`; fixed JSON B1/B32 trials compare latency,
throughput and output parity before and after grammar-aware MTP. It writes server logs and JSON
results to its `--output` directory. GPU jobs use partition `normal`, job name
`dev`, and CUDA devices 2 and 3.
