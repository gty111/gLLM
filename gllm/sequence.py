from typing import List, Optional, Union

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from gllm.utils import unify_decode


class Sequence:
    def __init__(
        self,
        seq_id,
        token_ids,
        finish_tokens,
        output_len=None,
        ignore_eos=False,
        temperature=0.6,
        top_p=0.9,
        top_k=10,
        repetition_penalty=1.0,
        mm_contents=None,
        logprobs_enabled=False,
        num_top_logprobs=0,
        prompt_logprobs_enabled=False,
        num_prompt_logprobs=0,
    ):
        self.seq_id = seq_id
        self.token_ids: List[int] = token_ids
        # ``raw_prompt_len`` is the *original* prompt length, fixed for the
        # whole lifetime of the request. ``prompt_len`` is the dynamic prefill
        # boundary used to distinguish prefill vs decode (see
        # ``computed_prompt``); it starts equal to ``raw_prompt_len`` but is
        # bumped to ``len(token_ids)`` on preempt because the already-generated
        # tokens must be re-prefilled from scratch. Always use
        # ``raw_prompt_len`` for output-length / usage accounting.
        self.raw_prompt_len = len(token_ids)
        self.prompt_len = len(token_ids)
        self.page_table = []
        self.prompt = ""
        self.output = ""
        self.ignore_eos = ignore_eos
        self.finish_tokens: List[int] = finish_tokens
        # maximum output length
        if output_len is None:
            self.output_len = 4096
        else:
            self.output_len = output_len
        # used for detokenize
        self.cur_length = self.prompt_len
        # used for sample
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        # Per-token logprobs (OpenAI ``logprobs``). ``logprobs_enabled`` turns
        # the (gated) log_softmax + top-k on for this seq; ``num_top_logprobs``
        # is how many alternative tokens to report alongside the sampled one
        # (0 => only the sampled token's logprob). ``_pending_logprob`` is the
        # scratch slot the sampler/runner stashes this decode step's result in
        # (non-overlap path); the scheduler drains it into the IPC package.
        self.logprobs_enabled = logprobs_enabled
        self.num_top_logprobs = num_top_logprobs
        self._pending_logprob = None
        # Prompt-token logprobs (vLLM ``prompt_logprobs``). Accumulated on the
        # worker across (possibly chunked) prefill into ``prompt_logprobs_data``
        # -- a list of length ``raw_prompt_len`` where index 0 is ``None`` (no
        # preceding context) and later entries are
        # ``(token_id, logprob, top_ids, top_vals)``. Sent to the frontend once
        # via the IPC package on the step the prompt finishes prefill;
        # ``_prompt_logprobs_sent`` guards against re-sending on later decodes.
        self.prompt_logprobs_enabled = prompt_logprobs_enabled
        self.num_prompt_logprobs = num_prompt_logprobs
        self.prompt_logprobs_data = None
        self._prompt_logprobs_sent = False
        # used for prefix cache and chunked prefill
        self.computed_token_num = 0
        self.to_compute_token_num = 0
        # used for abort
        self.is_abort = False
        # DP-attention request pinning: when the frontend exposes one HTTP
        # endpoint per DP replica (``--endpoint-per-dp``), the endpoint that
        # received this request pins it to that replica index so the seq's KV
        # lives there. ``None`` => frontend round-robins across replicas (the
        # default single-endpoint behaviour).
        self.target_dp: Optional[int] = None
        # used for multimodal input
        self.mm_contents = mm_contents
        # used to remove redundant token_ids
        self.to_compute_tokens = None
        # SSM working-pool slot for hybrid (Mamba/GDN) models. ``None`` means
        # either the model has no linear-attention layers or the scheduler
        # has not yet allocated a slot for this seq. The slot lives for the
        # whole lifetime of the request and is reset on preempt/free.
        self.ssm_state_slot: Optional[int] = None
        # Persistent per-seq slot in the repetition-penalty mask pool
        # (``MemoryManager._rep_pool``). ``None`` means no slot yet / the seq
        # has ``repetition_penalty == 1.0`` and needs none. ``rep_filled`` is
        # the number of ``token_ids`` already scattered into that pool row, so
        # each decode step only scatters the newly appended token(s) instead
        # of rebuilding the seq's whole history. Reset on free/preempt.
        self.rep_slot: Optional[int] = None
        self.rep_filled: int = 0
        # Alternate "view" of ``token_ids`` used *only* for prefix-cache
        # hashing. Multimodal pipelines splice content-derived ids into the
        # placeholder positions here so that VL prompts with identical text
        # but different images no longer collide in the cache. ``None``
        # falls back to ``token_ids`` (text-only path).
        self.hash_token_ids: Optional[List[int]] = None
        # Incrementally extended page-aligned prefix hash chain used by the
        # prefix cache. ``_page_hashes[i]`` is the chained hash of the first
        # ``(i+1)*page_size`` token ids; ``_canary_cache`` is the first few
        # token ids used as a hash-collision guard. Building these lazily
        # turns long-prefill lookups from O(prefix_len) per page into
        # O(page_size) per page (see ``PrefixSegment``). The hash source
        # (``hash_token_ids`` vs ``token_ids``) is captured at first build;
        # ``_hash_source_ref`` lets the helper auto-invalidate if a VL
        # request swaps in a fresh ``hash_token_ids`` after the cache was
        # populated.
        self._page_hashes: Optional[List[int]] = None
        self._canary_cache: Optional[tuple] = None
        self._hash_source_ref: Optional[int] = None

    def __len__(self):
        return len(self.token_ids)

    def __getitem__(self, key):
        return self.token_ids[key]

    def append(self, token_id):
        self.token_ids.append(token_id)

    def detokenize_inc(
        self, tokenizer: Union[PreTrainedTokenizer | PreTrainedTokenizerFast]
    ):
        added_space = (
            " "
            if " "
            in unify_decode(
                tokenizer, self[self.cur_length - 1 : self.cur_length + 1]
            ).strip()
            else ""
        )
        delta_text = unify_decode(tokenizer, self[self.cur_length :])
        if delta_text.endswith("�"):
            return ""
        if len(delta_text) > 0 and delta_text[0] != " ":
            delta_text = added_space + delta_text
        self.cur_length = len(self)
        return delta_text

    @property
    def is_finish(self):
        return self.computed_prompt and (
            (not self.ignore_eos and self[-1] in self.finish_tokens)
            or len(self) - self.raw_prompt_len >= self.output_len
        )

    def preempt(self):
        self.computed_token_num = 0
        # Preemption recomputes the seq from scratch, so every token currently
        # in ``token_ids`` (original prompt + already-generated tokens) must be
        # re-prefilled. Bump the prefill boundary accordingly so
        # ``computed_prompt`` correctly reports prefill (not decode) until the
        # recompute catches up. ``raw_prompt_len`` stays untouched.
        self.prompt_len = len(self.token_ids)
        self.page_table = []
        # SSM state is recurrent, so preempting (= recomputing from scratch)
        # invalidates whatever was in the working slot. The actual slot is
        # released by the scheduler via ``MemoryManager.free_ssm_slot`` so
        # that ``SSMSegment.free_working`` can also zero the tensors.
        self.ssm_state_slot = None

    @property
    def computed_prompt(self):
        return self.computed_token_num >= self.prompt_len

    @property
    def seq_len(self):
        return self.computed_token_num + self.to_compute_token_num
