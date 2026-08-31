from typing import List, Optional, Union

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from gllm.utils import unify_decode


class GenerationSequence:
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
        # (0 => only the sampled token's logprob). The computed per-step values
        # travel with the sampled tokens (runner ``_last_logprobs`` -> scheduler
        # -> IPC package), so no per-seq scratch slot is needed.
        self.logprobs_enabled = logprobs_enabled
        self.num_top_logprobs = num_top_logprobs
        # Prompt-token logprobs (``prompt_logprobs``). Accumulated on the
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
        # Arena slot holding this request's recurrent state: a hybrid
        # (Mamba/GDN) conv+temporal state, or DeepSeek-V4's learned KV-
        # compressor windows.  Both are mutated in place and must follow the
        # request across continuous-batching row changes, which is what
        # separates them from paged attention KV.  ``None`` means either the
        # model has no such state or the scheduler has not allocated a slot
        # yet.  The slot lives for the whole request and is reset on
        # preempt/free.
        self.recurrent_state_slot: Optional[int] = None
        # Spec-decode block table for hybrid MTP: a fixed ``1+k``
        # list of SSM state block ids (column 0 == rolling/committed state,
        # columns 1..k == verify-step per-token checkpoints). ``None`` for
        # non-MTP / non-hybrid seqs (those use the scalar ``recurrent_state_slot``).
        # ``ssm_num_accepted`` persists the last accepted-token count so the
        # next verify's recurrent kernel resumes from column ``num_accepted-1``
        # (1 = neutral: resume from column 0). See the column protocol in
        # ``MemoryManager.commit_ssm_checkpoint`` documents the commit protocol.
        self.ssm_block_table: Optional[list] = None
        self.ssm_num_accepted: int = 1
        # True only between overlap-MTP's optimistic fixed-width reservation
        # and its variable-width accept finalize.  Prefix-cache allocation must
        # not hash the temporary ``-1`` token placeholders in that interval.
        self._mtp_async_pending: bool = False
        # Mixed async prefill handoff: the sampled x1 lives in GPU relay state
        # and has no committed token_ids entry yet. The first MTP reservation
        # must therefore not apply the ordinary decode ``base_compute == 1`` a
        # second time. This is consumed exactly once by that reservation, or
        # materialized into token_ids before falling back to ordinary decode.
        self._mtp_relay_only_next: bool = False
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
        # Recurrent state is, well, recurrent: preempting (= recomputing from
        # scratch) invalidates whatever was in the working slot. The actual slot is
        # released by the scheduler via ``MemoryManager.free_recurrent_slot`` so
        # that ``SSMSegment.free_block`` can also zero the tensors.
        self.recurrent_state_slot = None
        self.ssm_block_table = None
        self.ssm_num_accepted = 1
        self._mtp_async_pending = False
        self._mtp_relay_only_next = False

    @property
    def computed_prompt(self):
        return self.computed_token_num >= self.prompt_len

    @property
    def seq_len(self):
        return self.computed_token_num + self.to_compute_token_num
