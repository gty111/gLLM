"""Request-local JSON constrained decoding. No model-specific forward changes.

Only serializable specifications cross worker boundaries. Compilers are cached
per process; mutable matchers belong to sequences on the sampling rank. The
sampler consumes its previous *authoritative* GPU token, not the scheduler's
potentially lagging overlap placeholders.
"""

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from weakref import WeakKeyDictionary

import torch


def normalize_format(fmt):
    """Return a JSON schema string, rejecting constraints we cannot enforce."""
    if hasattr(fmt, "model_dump"):
        fmt = fmt.model_dump(by_alias=True, exclude_none=True)
    if fmt is None or fmt == {"type": "text"}:
        return None
    if not isinstance(fmt, dict):
        raise ValueError("Output format must be an object.")
    kind = fmt.get("type", "text")
    if kind == "text":
        return None
    if kind == "json_object":
        return '{"type":"object"}'
    if kind != "json_schema":
        raise ValueError("Supported output formats are text, json_object and json_schema.")
    definition = fmt.get("json_schema", fmt)
    if not isinstance(definition, dict) or not isinstance(definition.get("schema"), dict):
        raise ValueError("json_schema requires an object-valued schema.")
    if not isinstance(definition.get("name"), str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", definition["name"]):
        raise ValueError("json_schema requires a name of 1-64 letters, digits, underscores or dashes.")
    strict = definition.get("strict", False)
    if strict is not None and not isinstance(strict, bool):
        raise ValueError("strict must be a boolean.")
    schema = definition["schema"]
    encoded = json.dumps(schema, ensure_ascii=False, allow_nan=False)
    if len(encoded.encode()) > 65536:
        raise ValueError("JSON schema exceeds the 64 KiB limit.")
    from jsonschema import Draft202012Validator

    try:
        Draft202012Validator.check_schema(schema)
    except Exception as exc:
        raise ValueError(f"Invalid JSON schema: {exc.message}") from exc
    allowed = {
        "type", "properties", "required", "additionalProperties", "items",
        "anyOf", "enum", "const", "$ref", "$defs", "definitions",
        "title", "description", "default", "$schema",
    }
    annotations = {"title", "description", "default", "$schema", "$defs", "definitions"}
    literal_nodes = []

    def visit(node, depth=0):
        if depth > 32:
            raise ValueError("JSON schema nesting exceeds 32 levels.")
        if isinstance(node, bool):
            raise ValueError("Boolean subschemas are not supported yet.")
        unknown = set(node) - allowed
        if unknown:
            raise ValueError(f"Unsupported JSON schema keywords: {sorted(unknown)}")
        if "$ref" in node and not node["$ref"].startswith("#"):
            raise ValueError("Only local JSON schema references are supported.")
        # XGrammar gives these keywords precedence over sibling assertions;
        # JSON Schema instead requires their intersection. Do not silently
        # compile a weaker grammar, even when strict=false.
        for keyword in ("$ref", "anyOf"):
            if keyword in node and set(node) - annotations - {keyword}:
                raise ValueError(f"{keyword} with sibling constraints is not supported.")
        if set(node.get("required", [])) - set(node.get("properties", {})):
            raise ValueError("Required properties must have an explicit properties schema.")
        if "const" in node or "enum" in node:
            literal_nodes.append(node)
        types = node.get("type", [])
        types = [types] if isinstance(types, str) else types
        if strict and ("object" in types or "properties" in node):
            if node.get("additionalProperties") is not False:
                raise ValueError("strict objects require additionalProperties=false.")
            if set(node.get("required", [])) != set(node.get("properties", {})):
                raise ValueError("strict objects must list every property in required.")
        for key in ("properties", "$defs", "definitions"):
            for child in node.get(key, {}).values():
                visit(child, depth + 1)
        for key in ("items", "additionalProperties"):
            child = node.get(key)
            if key == "items" and isinstance(child, bool):
                raise ValueError("Boolean items schemas are not supported yet.")
            if isinstance(child, dict):
                visit(child, depth + 1)
        for child in node.get("anyOf", []):
            visit(child, depth + 1)

    visit(schema)
    # enum/const also take precedence in the backend. Keep common type+enum
    # schemas, but reject literals that would bypass another assertion. Do
    # this after visiting every node so references are known to be local.
    validator = Draft202012Validator(schema)
    for node in literal_nodes:
        values = [node["const"]] if "const" in node else node["enum"]
        try:
            valid = all(validator.evolve(schema=node).is_valid(value) for value in values)
        except Exception as exc:
            raise ValueError("Cannot validate enum/const constraints.") from exc
        if not valid:
            raise ValueError("enum/const values must satisfy all sibling constraints.")
    if strict and (schema.get("type") != "object" or "anyOf" in schema):
        raise ValueError("strict output requires an object root without anyOf.")
    return encoded


@lru_cache(maxsize=4)
def compiler(tokenizer, vocab_size, stop_tokens):
    import xgrammar as xgr

    info = xgr.TokenizerInfo.from_huggingface(
        tokenizer, vocab_size=vocab_size, stop_token_ids=list(stop_tokens)
    )
    return xgr.GrammarCompiler(info, max_threads=2, cache_limit_bytes=64 * 1024 * 1024)


@dataclass(frozen=True)
class StructuredOutput:
    schema: str
    think_start: int | None = None
    think_end: int | None = None
    thinking: bool = False
    allow_think_start: bool = False


def prepare_output(fmt, tokenizer, vocab_size, stop_tokens, prompt_ids):
    schema = normalize_format(fmt)
    if schema is None:
        return None
    # Compile before admission: bad schemas must not kill a worker. Cache hits
    # amortize compilation; the API runs this function off its event loop.
    compiler(tokenizer, vocab_size, tuple(stop_tokens)).compile_json_schema(
        schema, strict_mode=False
    )
    from gllm.tokenizers.reasoning import create_reasoning_parser

    parser = create_reasoning_parser(tokenizer, prompt_ids)
    if parser is None:
        return StructuredOutput(schema)
    start = tokenizer.convert_tokens_to_ids(parser.START)
    end = tokenizer.convert_tokens_to_ids(parser.END)
    suffix = tokenizer.decode(prompt_ids[-16:], skip_special_tokens=False).rstrip()
    return StructuredOutput(
        schema, start, end, parser.prefilled,
        not parser.prefilled and not suffix.endswith(parser.END),
    )


class _State:
    def __init__(self, ctx, spec, stops):
        self.ctx, self.spec, self.stops = ctx, spec, stops
        self.history = []
        self.pending = None
        self.spec_pending = None
        self.reset()

    def reset(self):
        import xgrammar as xgr

        self.matcher = xgr.GrammarMatcher(self.ctx, override_stop_tokens=self.stops)
        self.thinking = self.spec.thinking
        self.allow_start = self.spec.allow_think_start

    def accept(self, token):
        if self.matcher.is_terminated():
            return  # overlap can run past EOS before retirement
        if self.thinking:
            if token == self.spec.think_end:
                self.thinking = False
            return
        if self.allow_start and token == self.spec.think_start:
            self.allow_start = False
            self.thinking = True
            return
        self.allow_start = False
        if not self.matcher.accept_token(token):
            raise RuntimeError(f"Structured output sampled an invalid token: {token}")

    def advance(self, position, previous=None):
        if previous is not None:
            self.history.append(previous)
        # Re-prefill after preemption may discard an optimistic overlap suffix.
        if position < len(self.history):
            self.history = self.history[:position]
            self.reset()
            for token in self.history:
                self.accept(token)
        elif position > len(self.history):
            raise RuntimeError(
                f"Structured output lost its sampled-token history: position={position}, "
                f"history={len(self.history)}, previous={previous}"
            )
        elif previous is not None:
            self.accept(previous)

    def synchronize(self, tokens):
        """Adopt scheduler-authoritative history at a synchronous boundary."""
        if tokens[:len(self.history)] != self.history:
            self.history = []
            self.reset()
        for token in tokens[len(self.history):]:
            self.accept(token)
            self.history.append(token)
        self.pending = None

    def fill_mask(self, mask, row):
        if self.thinking:
            allowed = []
            for token in self.stops:
                bit = (1 << (token % 32)) if token % 32 != 31 else -(1 << 31)
                mask[row, token // 32] &= ~bit
        elif self.matcher.is_terminated():
            mask[row].zero_()
            allowed = self.stops
        else:
            self.matcher.fill_next_token_bitmask(mask, row)
            allowed = [self.spec.think_start] if self.allow_start else []
        for token in allowed:
            mask[row, token // 32] |= (1 << (token % 32)) if token % 32 != 31 else -(1 << 31)


class StructuredSampler:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.states = WeakKeyDictionary()
        self._gpu_mask = None
        self._feedback_rows = []

    def _state(self, seq, vocab_size):
        state = self.states.get(seq)
        if state is None:
            spec = seq.structured_output
            ctx = compiler(self.tokenizer, vocab_size, tuple(seq.finish_tokens))
            ctx = ctx.compile_json_schema(spec.schema, strict_mode=False)
            state = self.states[seq] = _State(ctx, spec, seq.finish_tokens)
        return state

    @staticmethod
    def stage_speculative_inputs(context_lens, x1, drafts, stream=None):
        """Queue candidate feedback BEFORE target verify, not behind it."""
        packed = torch.cat((context_lens[:, None], x1[:, None], drafts), dim=1)
        if stream is None:
            return packed.cpu(), None
        host = torch.empty(packed.shape, dtype=packed.dtype, pin_memory=True)
        stream.wait_stream(torch.cuda.current_stream(packed.device))
        with torch.cuda.stream(stream):
            host.copy_(packed, non_blocking=True)
            ready = torch.cuda.Event()
            ready.record()
        packed.record_stream(stream)
        return host, ready

    def record_speculative(self, active, completion):
        for row, state in active:
            if state.spec_pending is not None:
                raise RuntimeError("Unconsumed structured MTP acceptance")
            state.spec_pending = (completion, row)

    def finish_speculative(self, seqs, completion=None):
        """Advance grammar only; do not compact scheduler placeholders/free KV."""
        for seq in seqs:
            state = self.states.get(seq)
            if state is None or state.spec_pending is None:
                continue
            pending, row = state.spec_pending
            if completion is not None and pending is not completion:
                continue  # the successor already registered a newer completion
            _, committed = pending.read()
            for token in committed[row]:
                state.accept(token)
                state.history.append(token)
            state.pending = None  # bonus stays in GPU relay until consumed
            state.spec_pending = None

    def stage_relay(self, seq, token):
        """At a drain, expose the GPU-only bonus to ordinary sampling."""
        state = self.states.get(seq)
        if state is not None:
            state.pending = torch.tensor(token, dtype=torch.int64)

    def mask_speculative(self, logits, seqs, contexts, candidates, *, positions=None):
        """Mask target distributions after each verify input [x1, d1, ...].

        Fork parsing state, not committed history. An unconstrained draft is a
        valid proposal q: masking target p before its sampling transforms makes
        illegal drafts reject with probability one, using the original q in
        the usual acceptance/residual calculation.
        """
        import copy
        import xgrammar as xgr

        width = (logits.shape[-1] + 31) // 32
        mask = torch.full((len(logits), width), -1, dtype=torch.int32,
                          pin_memory=logits.is_cuda)
        active = []
        self.finish_speculative(seqs)
        for i, (seq, tokens) in enumerate(zip(seqs, candidates)):
            if getattr(seq, "structured_output", None) is None:
                continue
            state = self._state(seq, logits.shape[-1])
            if contexts is not None:
                state.synchronize(contexts[i][seq.raw_prompt_len:])
            else:
                # GPU context lengths are authoritative while CPU sequences
                # still contain optimistic scheduler placeholders. Completed
                # mixed prefills may have a pending x1 that is relay-only: do
                # not consume it until the trial matcher below sees x1.
                position = positions[i] - seq.raw_prompt_len
                previous = None
                if position > len(state.history) and state.pending is not None:
                    value = state.pending
                    if isinstance(value, tuple):
                        value, ready = value
                        ready.synchronize()
                    if value.is_cuda:
                        raise RuntimeError("Structured feedback was not staged")
                    previous = int(value)
                state.advance(position, previous)
                state.pending = None
            trial = copy.copy(state)
            trial.matcher = state.matcher.fork()
            for j, token in enumerate(tokens):
                row = i * len(tokens) + j
                if j and not (int(mask[row - 1, token // 32]) & (1 << (token % 32))):
                    # Unreachable suffix: the preceding p assigns this draft
                    # zero mass. Leave finite logits to avoid NaN softmax rows.
                    break
                trial.accept(token)
                trial.fill_mask(mask, row)
            active.append((i, state))
        xgr.apply_token_bitmask_inplace(logits, mask.to(logits.device, non_blocking=logits.is_cuda))
        return active

    def commit_speculative(self, active, committed, bonuses):
        for row, state in active:
            for token in committed[row]:
                state.accept(token)
                state.history.append(token)
            # The bonus is sampled but not emitted yet. Keep the same pending
            # convention as plain sampling so a relay-to-plain transition works.
            state.pending = torch.tensor(bonuses[row], dtype=torch.int64)

    def mask(self, logits, seqs):
        prepared = self.prepare(seqs, logits.shape[-1], logits.device)
        return self.apply(logits, prepared)

    def prepare(self, seqs, vocab_size, device):
        """CPU grammar-ready boundary, after forward submission, before sampling."""
        import xgrammar as xgr

        active = []
        for row, seq in enumerate(seqs):
            spec = getattr(seq, "structured_output", None)
            end = seq.computed_token_num + seq.to_compute_token_num
            position = end - seq.raw_prompt_len
            if spec is None or end < seq.prompt_len:
                continue  # intermediate prefill samples are discarded
            state = self._state(seq, vocab_size)
            if seq.computed_token_num < seq.prompt_len and seq.prompt_len > seq.raw_prompt_len:
                # Preemption retains committed outputs, but the scheduler
                # deep-copies unfinished chunks. Rebuild from authoritative
                # tokens at the final chunk, not from a discarded chunk sample
                # or a matcher belonging to an earlier sequence object.
                history = getattr(seq, "structured_output_history", None)
                if history is None:
                    history = seq.token_ids[seq.raw_prompt_len:end]
                state.synchronize(history)
            active.append((row, position, state))
        if not active:
            return None
        # One batched feedback transfer. These views refer to the returned token
        # tensor, which the runner's TP broadcast overwrites with its authority.
        pending = [s.pending for _, _, s in active if s.pending is not None]
        previous = []
        waited = set()
        for value in pending:
            if isinstance(value, tuple):
                value, ready = value
                if ready not in waited:
                    ready.synchronize()
                    waited.add(ready)
            elif value.is_cuda:
                raise RuntimeError("Structured GPU feedback must be staged after token broadcast")
            previous.append(int(value))
        previous = iter(previous)
        width = (vocab_size + 31) // 32
        if (self._gpu_mask is None or self._gpu_mask.shape[0] < len(seqs)
                or self._gpu_mask.shape[1] != width
                or self._gpu_mask.device != device):
            self._gpu_mask = torch.empty((len(seqs), width), dtype=torch.int32, device=device)
        # CPU masks belong to batches, not the sampler. Mutating one reusable
        # pinned mask required waiting for an unrelated PP microbatch's H2D.
        # PyTorch's pinned allocator tracks non-blocking copies, so releasing
        # this tensor after apply() is safe without a host-side stream wait.
        mask = torch.full((len(seqs), width), -1, dtype=torch.int32,
                          pin_memory=device.type == "cuda")
        for row, position, state in active:
            state.advance(position, next(previous) if state.pending is not None else None)
            state.pending = None
            state.fill_mask(mask, row)
        return active, mask

    def apply(self, logits, prepared):
        """GPU-only mask application; no token feedback wait inside sampling."""
        if prepared is None:
            return []
        import xgrammar as xgr

        active, mask = prepared
        device_mask = self._gpu_mask[:len(mask)]
        # Never block on the CURRENT forward stream: under PP it can be waiting
        # for activations while earlier stages need us to post token feedback.
        # Each prepared batch owns a distinct, immutable-after-copy host mask.
        device_mask.copy_(mask, non_blocking=logits.is_cuda)
        xgr.apply_token_bitmask_inplace(logits, device_mask)
        return active

    def record(self, active, tokens):
        self._feedback_rows = [(row, state) for row, _, state in active]
        for row, _, state in active:
            state.pending = tokens[row]

    def stage_feedback(self, tokens, stream=None):
        """Called by the runner AFTER its authoritative TP token broadcast."""
        rows, self._feedback_rows = self._feedback_rows, []
        if not rows:
            return
        # Only rows sampled in this batch own new feedback. Intermediate
        # re-prefill rows must not overwrite a sequence's earlier pending token
        # with the discarded sample from a prompt chunk.
        # Reuse the runner's pre-existing D2H stream. CUDA streams are pooled:
        # a late allocation after graph capture can alias a live forward/P2P
        # stream and introduce a cross-batch dependency cycle.
        if stream is None:
            host = tokens.cpu()
            for row, state in rows:
                state.pending = host[row]
            return
        host = torch.empty(tokens.shape, dtype=tokens.dtype, device="cpu", pin_memory=True)
        stream.wait_stream(torch.cuda.current_stream(tokens.device))
        with torch.cuda.stream(stream):
            host.copy_(tokens, non_blocking=True)
            ready = torch.cuda.Event()
            ready.record()
        tokens.record_stream(stream)
        for row, state in rows:
            state.pending = (host[row], ready)
