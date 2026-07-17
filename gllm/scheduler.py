import copy
import random
import time
from collections import deque
from typing import List, Optional

from logger import logger

from gllm.comm import IPCPackage
from gllm.dist_utils import get_pp_rank, get_rank, get_tp_rank, get_world_size
from gllm.memory_manager import MemoryManager, PrefixMemoryManager
from gllm.model_runner import ModelRunner
from gllm.sequence import Sequence


class Scheduler:
    def __init__(self, pp_size, model_runner: ModelRunner, schedule_method):
        self.pp_size = pp_size
        self.model_runner: ModelRunner = model_runner
        self.memory_manager: MemoryManager = model_runner.memory_manager
        self.schedule_method = schedule_method
        self.maxd = model_runner.maxd
        self.maxp = model_runner.maxp
        self.minp = model_runner.minp
        self.iterp = model_runner.iterp
        self.page_size = model_runner.page_size

        # --- Adaptive KV-cache admission control (SGLang-style) ---
        # We no longer hold back a *static* page reserve (the old ``kvthresh``).
        # Instead, before admitting prefill we reserve enough pages for the
        # *projected growth* of the in-flight decode batch, and we adapt how
        # conservative that projection is from observed preemptions.
        #
        # ``new_token_ratio`` is the fraction of each running seq's *remaining*
        # output length we assume it will still generate before finishing. It
        # rises on a preemption event (be conservative -> admit less prefill)
        # and decays back every tick (relax -> admit more prefill when stable).
        self.init_new_token_ratio = model_runner.init_new_token_ratio
        self.min_new_token_ratio = model_runner.min_new_token_ratio
        self.new_token_ratio = self.init_new_token_ratio
        self.new_token_ratio_step = 0.05  # bump up on a preemption event
        self.new_token_ratio_decay = 0.002  # relax per schedule tick
        # Hard floor so prefill never drains the free list to literally empty
        # (page allocation within a batch is slightly bursty).
        self.min_reserve_pages = max(1, self.pp_size)

        # seqs to schedule
        self.seqs_to_prefill: deque[Sequence] = deque()
        self.seqs_to_decode: deque[Sequence] = deque()
        # running batch
        self.batch_running = deque()
        # next tokens
        self.next_tokens_queue = deque()
        self.log_time = 0
        # preempt seqs
        self.num_preempt_seqs = 0
        self.log_num_preempt_seqs = 0
        self.delta_log_num_preempt_seqs = 10
        # num wait tokens
        self.num_wait_tokens = 0
        # Deterministic rotating jitter for the decode-token-budget split (see
        # ``get_balanced_decode_token_budget``). Replaces a ``random.randint``
        # that was safe only for ``pp_size == 1`` (randint(0,0)==0): with pp>1
        # each rank's RNG diverged, so TP peers -- which must run *identical*
        # column-driver schedules -- picked different budgets and scheduled
        # mismatched batch sizes, deadlocking the stage-wide MoE/EP all-reduce
        # (and PP hidden-state exchange). A rotating counter advanced in lockstep
        # by every rank keeps the split deterministic and TP-consistent.
        self._decode_budget_jitter = 0
        # abort ids
        self.abort_ids = set()
        # log
        self.log = True
        # Seq-ids that finished / aborted since the last time we built a
        # schedule payload for the followers. The worker drains this on
        # every payload build (``Worker.consume_pending_follower_frees``)
        # so the followers can drop their per-seq mirror + VL
        # ``embedding_cache`` entry. We piggyback on the next schedule
        # payload instead of opening a dedicated free socket; the worst
        # case (no further schedules ever happen) leaks at most the
        # currently-tracked seqs at process exit, which is fine.
        self._pending_follower_frees: List[int] = []
        # schedule method
        self.schedule = self.dispatch_schedule_method()

    def consume_pending_follower_frees(self) -> List[int]:
        """Drain seq_ids accumulated since the last schedule payload."""
        out = self._pending_follower_frees
        self._pending_follower_frees = []
        return out

    def dispatch_schedule_method(self):
        if self.schedule_method in ["split_pd", "chunked_prefill"]:
            return self.chunked_prefill
        elif self.schedule_method == "token_throttling":
            return self.token_throttling
        else:
            assert 0

    def get_num_free_pages(self):
        return self.memory_manager.get_num_free_pages()

    def get_num_decode_seqs(self):
        num_decode_seqs = len(self.seqs_to_decode) + sum(
            len(i) for i in self.batch_running
        )
        return num_decode_seqs

    def _seq_reserve_pages(self, seq, ratio):
        """How many *more* KV pages a single decode seq will likely need.

        Estimated as ``ratio * remaining_output`` tokens (the adaptive
        ``new_token_ratio`` discounts the worst case), minus whatever still
        fits in the seq's current partially-filled tail page, rounded up to
        whole pages. Returns 0 for seqs that are still prefilling or already
        past their output budget.
        """
        if not seq.computed_prompt:
            return 0
        remaining = seq.output_len - (len(seq) - seq.raw_prompt_len)
        if remaining <= 0:
            return 0
        est_tokens = int(ratio * remaining)
        tail_slack = len(seq.page_table) * self.page_size - len(seq)
        extra = est_tokens - max(0, tail_slack)
        if extra <= 0:
            return 0
        return (extra + self.page_size - 1) // self.page_size

    def _decode_reserve_pages(self):
        """Pages to hold back from prefill for the in-flight decode batch.

        SGLang-style admission control: a running decode seq keeps emitting
        tokens until it finishes, crossing a new KV page every ``page_size``
        steps. If prefill is allowed to consume the whole free pool, those
        running seqs starve mid-generation and get preempted -- exactly the
        thrash we want to avoid. So we estimate, per running decode seq, how
        many *more* pages it will need (scaled by the adaptive
        ``new_token_ratio``) and keep that many free. A new prefill is only
        admitted into whatever headroom is left over.

        The running decode population is split across two places at schedule
        time: ``seqs_to_decode`` (waiting for the next decode step) and
        ``batch_running`` (already dispatched, in-flight on the PP pipeline /
        overlap stream). We count both -- deduping by ``seq_id`` since the two
        never legitimately overlap but timing edges shouldn't double-count --
        so the reserve reflects the full population even when PP > 1.
        """
        ratio = self.new_token_ratio
        reserve = 0
        seen = set()
        for seq in self.seqs_to_decode:
            if seq.seq_id in seen:
                continue
            seen.add(seq.seq_id)
            reserve += self._seq_reserve_pages(seq, ratio)
        for batch in self.batch_running:
            for seq in batch:
                if seq.seq_id in seen:
                    continue
                seen.add(seq.seq_id)
                reserve += self._seq_reserve_pages(seq, ratio)
        return max(self.min_reserve_pages, reserve)

    def update_num_wait_tokens(self):
        self.num_wait_tokens = sum(
            len(i) - i.computed_token_num for i in self.seqs_to_prefill
        )

    def add_abort_ids(self, abort_ids):
        self.abort_ids.update(abort_ids)

    def add_new_requests(self, seqs):
        self.seqs_to_prefill.extend(seqs)

    def add_next_tokens(self, next_tokens, logprobs=None, prompt_logprobs=None):
        # ``logprobs`` (when present) is a per-batch-row list aligned with
        # ``next_tokens`` -- ``None`` for seqs that didn't request logprobs,
        # else ``(sampled, top_ids, top_vals)``. ``prompt_logprobs`` (PP>1 only)
        # is a ``{seq_id: prompt_logprobs_data}`` dict for prompts that just
        # finished prefill, computed on the output-rank follower and shipped
        # over the token socket. Both are queued alongside the tokens so
        # ``process_output`` can attach them in batch order.
        self.next_tokens_queue.append((next_tokens, logprobs, prompt_logprobs))

    @staticmethod
    def _attach_prompt_logprobs(ipc_package, seq):
        """Emit a seq's completed prompt-logprobs exactly once.

        Called on the step the prompt finishes prefill (its first output
        token). ``prompt_logprobs_data`` is fully populated by then, so we hand
        it to the frontend keyed by seq_id and latch ``_prompt_logprobs_sent``
        so later decode steps don't resend it.
        """
        if (
            getattr(seq, "prompt_logprobs_enabled", False)
            and not seq._prompt_logprobs_sent
            and seq.prompt_logprobs_data is not None
        ):
            ipc_package.prompt_logprobs[seq.seq_id] = seq.prompt_logprobs_data
            seq._prompt_logprobs_sent = True

    def set_log(self, log):
        self.log = log

    def process_output(self):
        if len(self.next_tokens_queue) == 0:
            return None

        schedule_seqs: List[Sequence] = self.batch_running.popleft()
        next_tokens, logprobs, prompt_logprobs = self.next_tokens_queue.popleft()

        ipc_package = IPCPackage([])

        for idx, seq in enumerate(schedule_seqs):
            seq.computed_token_num += seq.to_compute_token_num
            if seq.computed_prompt:
                ipc_package.act_schedule_ids.append(seq.seq_id)
                ipc_package.next_tokens.append(next_tokens[idx])
                if logprobs is not None and seq.logprobs_enabled:
                    ipc_package.logprobs.append(logprobs[idx])
                else:
                    ipc_package.logprobs.append(None)
                if prompt_logprobs and seq.seq_id in prompt_logprobs:
                    # PP>1: prompt logprobs computed on the follower and shipped
                    # over the token socket; attach the received list directly.
                    ipc_package.prompt_logprobs[seq.seq_id] = prompt_logprobs[
                        seq.seq_id
                    ]
                else:
                    # PP=1: attach from the local seq's accumulated data.
                    self._attach_prompt_logprobs(ipc_package, seq)
                seq.append(next_tokens[idx])
                # The token is real (non-overlap appends immediately), so it is
                # safe to register the prefix-cache hash for any page boundary
                # it just completed. Driven from here -- not pre_allocate_page --
                # so a placeholder id can never be hashed (see model_runner).
                self.model_runner.register_decode_page_hash(seq, len(seq) - 1)
            if seq.is_finish:
                ipc_package.free_ids.append(seq.seq_id)
                # Mirror the free to follower-side state cleanup. ``free_ids``
                # in ``ipc_package`` goes to the *frontend* (so it can stop
                # tracking the seq and emit the final response); followers
                # need their own notification path because they hold the
                # FollowerSeq mirror + VL ``embedding_cache`` row.
                self._pending_follower_frees.append(seq.seq_id)
                self.model_runner.free(seq)
            elif seq.computed_prompt:
                self.seqs_to_decode.appendleft(seq)
            else:  # unfinished prefill seqs
                pass
        return ipc_package

    def check_preempt(self, num_tokens_to_allocate):
        if (
            self.get_num_free_pages() >= num_tokens_to_allocate
            or len(self.seqs_to_decode) == 0
        ):
            return

        # Victim selection (SGLang-style retract): evict the largest-footprint
        # decode seqs first so each preemption reclaims the most pages -> fewer
        # preemptions -> less recompute churn. Only pay the O(n log n) sort on
        # the rare ticks where we actually have to preempt.
        victims = sorted(
            self.seqs_to_decode, key=lambda s: len(s.page_table), reverse=True
        )
        preempt_seqs = []
        for seq_to_preempt in victims:
            if self.get_num_free_pages() >= num_tokens_to_allocate:
                break
            self.seqs_to_decode.remove(seq_to_preempt)
            self.model_runner.free(seq_to_preempt)
            seq_to_preempt.preempt()
            # Don't notify followers here: the seq is being re-queued as a
            # waiting prefill and will appear in a future schedule batch.
            # ``DriverPayloadBuilder`` detects the page_table shrink
            # (len < last_pages_len) on that next iteration and emits a
            # ``page_table_reset=[]`` in the SeqUpdate so the follower
            # drops the stale page ids without us having to send an
            # explicit free + re-register round trip.
            preempt_seqs.append(seq_to_preempt)

        # Re-queue preempted seqs at the FRONT of the wait queue so they resume
        # ahead of brand-new requests (they've already done work). The adaptive
        # reserve -- not queue position -- is what stops them from being
        # instantly re-admitted and preempted again.
        self.seqs_to_prefill.extendleft(preempt_seqs)

        if preempt_seqs:
            # Change 2: a preemption means we under-reserved. Bump the ratio so
            # the next prefill admission backs off (decayed back when stable).
            self.new_token_ratio = min(
                1.0, self.new_token_ratio + self.new_token_ratio_step
            )

        self.num_preempt_seqs += len(preempt_seqs)
        if (
            self.num_preempt_seqs - self.log_num_preempt_seqs
            >= self.delta_log_num_preempt_seqs
        ):
            self.log_num_preempt_seqs = self.num_preempt_seqs
            self.delta_log_num_preempt_seqs *= 2
            # Every TP/PP rank runs an identical scheduler, so restrict the
            # warning to global rank 0 to avoid N duplicate lines per event.
            if get_rank() == 0:
                # Round down to the nearest 10 so the count reads as a clean
                # milestone (e.g. 150) instead of the raw value (e.g. 152) the
                # batch happened to overshoot the doubling threshold by.
                rounded = int(self.num_preempt_seqs) // 10 * 10
                logger.warning(
                    f"#Preempted seqs: >={rounded}; KV cache under pressure "
                    "(reduce --maxd or request output lengths if this persists)."
                )

    def check_abort_seqs_list(self, seqs: deque, ipc_package: IPCPackage):
        for seq in list(seqs):
            if len(self.abort_ids) == 0:
                break
            id = seq.seq_id
            if id in self.abort_ids:
                ipc_package.free_ids.append(id)
                # See ``process_output``: followers need their own free
                # notification so they can drop the FollowerSeq mirror.
                self._pending_follower_frees.append(id)
                self.model_runner.free(seq)
                seqs.remove(seq)
                self.abort_ids.remove(id)

    def check_abort_seqs(self):
        ipc_package = IPCPackage([])
        self.check_abort_seqs_list(self.seqs_to_prefill, ipc_package)
        self.check_abort_seqs_list(self.seqs_to_decode, ipc_package)
        if len(ipc_package.free_ids) != 0:
            return ipc_package
        else:
            return None

    def schedule_once(self):
        """Pick a batch from the queues; followers no longer get a Sequence list.

        Previously this method returned a *deep-ish* copy of the batch's
        ``Sequence`` objects (``post_schedule`` shallow-copied each seq and
        stripped token_ids / extracted ``to_compute_tokens``) so that the
        zmq sender thread could safely pickle them while the main thread
        kept mutating the originals.

        With the delta-broadcast (``gllm/dist_schedule.py``) the worker
        snapshots whatever state the followers actually need into a
        :class:`SchedulePayload` at send time, so we just hand back the
        *real* ``Sequence`` objects -- no copy, no token_ids stripping.
        The rank-0 paths that still read from the returned list
        (``prepare_input`` for the local ``InputData``, the
        deferred-output processing in :class:`OverlapScheduler`) want
        the live ``Sequence`` anyway.
        """
        if (
            len(self.seqs_to_decode) + len(self.seqs_to_prefill) != 0
            and len(self.batch_running) < self.pp_size
        ):
            schedule_seqs = self.schedule()
            if len(schedule_seqs) != 0:
                self.batch_running.append(schedule_seqs)
                return schedule_seqs

        return []

    def get_balanced_decode_token_budget(self, num_total_decode_seqs):
        if num_total_decode_seqs < self.pp_size:
            decode_token_budget = 1
        else:
            # Add a rotating [0, pp_size-1] jitter before the floor-divide so the
            # split doesn't always round down (e.g. #seqs=5, pp_size=4). The
            # jitter must be *deterministic and identical across ranks* -- every
            # column-driver runs the same schedule, so a per-scheduler counter
            # advanced in lockstep keeps TP peers on the same budget (a prior
            # ``random.randint`` diverged them for pp_size>1 and deadlocked the
            # MoE/EP all-reduce; see ``__init__``).
            jitter = self._decode_budget_jitter % self.pp_size
            self._decode_budget_jitter += 1
            decode_token_budget = (num_total_decode_seqs + jitter) // self.pp_size

        decode_token_budget = min(self.maxd, decode_token_budget)
        return decode_token_budget

    def schedule_prefill_batch(
        self, prefill_token_budget, max_seqs=None, reserve_pages=0
    ):
        prefill_batch: List[Sequence] = []
        unfinish_prefill_seqs = deque()
        # Encoder-disaggregation overlap (design §6.2): seqs whose next chunk is
        # entirely blocked behind a not-yet-ready image span are parked here and
        # re-queued after this round (no slot/page allocation, no ordering loss).
        deferred_disagg_seqs: List[Sequence] = []
        prefill_batched_token_nums = 0
        # Hybrid models (Qwen3.5 GDN) cap concurrency at the SSM working
        # pool size; admitting more requests would assertion-crash inside
        # ``IDAllocator.allocate``. ``ssm_segment`` is None for plain
        # softmax-attn models, in which case the slot check below is a
        # no-op.
        ssm_seg = getattr(self.memory_manager, "ssm_segment", None)
        # Cap the number of prefill seqs so the *combined* batch
        # (decode + prefill) never exceeds ``max_running_seqs`` rows, which
        # is what the device input buffers (block_table / seq_lens / ...) are
        # sized for. Without this, a tiny model with a huge KV cache admits
        # enough short prompts that ``decode + prefill`` overflows the buffer
        # and ``copy_to_input_buffer`` crashes on a row-count mismatch.
        while (
            len(self.seqs_to_prefill) != 0
            and prefill_token_budget != 0
            and (max_seqs is None or len(prefill_batch) < max_seqs)
        ):
            seq = self.seqs_to_prefill.popleft()
            # If this seq hasn't taken a working slot yet AND the pool is
            # exhausted, put the request back on the wait queue and stop
            # admitting fresh prefills this round. In-flight prefill seqs
            # (those whose chunk continues this round) keep their slot.
            if (
                ssm_seg is not None
                and seq.ssm_state_slot is None
                and ssm_seg.num_free_working() == 0
            ):
                self.seqs_to_prefill.appendleft(seq)
                break
            if (
                isinstance(self.memory_manager, PrefixMemoryManager)
                and seq.computed_token_num == 0
            ):
                # For multimodal seqs the cache key MUST be built from
                # content-derived pad ids before the lookup, otherwise
                # two requests with different images but the same raw
                # ``<|image_pad|>`` placeholders collide and the second
                # request reuses the first's KV at the image span. This
                # used to surface as "second image gets described as the
                # first" and forced ``--no-enable-prefix-caching`` for
                # VL deployments. ``_mm_precompute_hash`` is a no-op for
                # text-only seqs and for non-VL models, and stashes the
                # heavy image_processor output on the seq so the later
                # ``_mm_prepare_cpu`` pass doesn't redo the work.
                self.model_runner._mm_precompute_hash(seq)
                self.memory_manager.pre_allocate_computed_page([seq])
                # Full/partial hit post-processing (rollback + hybrid SSM
                # snapshot restore) lives in PrefixMemoryManager.
            # Encoder-disaggregation overlap gate B (design §6.2): a disagg seq
            # may only prefill up to the first image span whose embedding hasn't
            # landed yet (positions are known -- gate A -- but the visual data
            # isn't visible). ``disagg_prefill_limit`` returns ``None`` for
            # monolith / non-disagg seqs (no cap) and is evaluated *after* the
            # prefix-cache pre-allocate above so cache-hit spans don't block.
            gate_limit = self.model_runner.disagg_prefill_limit(seq)
            if gate_limit is not None and gate_limit <= seq.computed_token_num:
                # Nothing prefillable this round; park and re-queue. No SSM slot
                # / page allocation so freeing stays simple if it never runs.
                deferred_disagg_seqs.append(seq)
                continue
            prefill_avail = len(seq) - seq.computed_token_num
            if gate_limit is not None:
                prefill_avail = min(prefill_avail, gate_limit - seq.computed_token_num)
            # Hybrid models: every seq needs a per-request SSM working slot
            # for the whole duration of the request (allocated lazily on
            # the first schedule, freed when ``model_runner.free(seq)`` is
            # called from ``process_output`` / ``check_abort_seqs`` /
            # ``check_preempt``). No-op for non-hybrid models.
            self.memory_manager.allocate_ssm_slot(seq)
            if prefill_avail <= prefill_token_budget:
                seq.to_compute_token_num = prefill_avail
            else:
                seq.to_compute_token_num = prefill_token_budget
            # Page guard: the prefill ``token`` budget only accounts for the
            # *uncached tail* a seq computes this step, but ``pre_allocate_page``
            # below grabs a KV page for every NEW page boundary the seq crosses
            # -- and with prefix caching the seq already grabbed its whole cached
            # prefix in ``pre_allocate_computed_page`` above (pages that cost
            # zero compute tokens). So a high cache-hit-rate workload can pass
            # the token-budget check while collectively demanding far more pages
            # than exist, and ``IDAllocator.allocate`` then asserts on an empty
            # free list mid-batch. Stop admitting before that happens: compute
            # the pages this seq still needs and re-queue it (unmodified) when
            # the free pool -- minus the decode reserve -- can't cover them.
            pages_have = len(seq.page_table)
            pages_needed_total = (
                seq.computed_token_num + seq.to_compute_token_num + self.page_size - 1
            ) // self.page_size
            pages_to_alloc = max(0, pages_needed_total - pages_have)
            if pages_to_alloc > self.get_num_free_pages() - reserve_pages:
                # Undo the per-step bookkeeping and park this seq for a later
                # round. Its cached-prefix pages stay in ``page_table`` (a
                # sibling in this batch may share them); they are reclaimed
                # normally when the seq eventually runs or is freed.
                seq.to_compute_token_num = 0
                self.seqs_to_prefill.appendleft(seq)
                break
            prefill_batched_token_nums += seq.to_compute_token_num
            prefill_token_budget -= seq.to_compute_token_num
            self.memory_manager.pre_allocate_page([seq])
            prefill_batch.append(seq)
            if seq.computed_token_num + seq.to_compute_token_num < seq.prompt_len:
                seq_new = copy.deepcopy(seq)
                seq_new.computed_token_num += seq_new.to_compute_token_num
                unfinish_prefill_seqs.appendleft(seq_new)

        self.seqs_to_prefill.extendleft(unfinish_prefill_seqs)
        # Re-queue gate-B-blocked disagg seqs (preserving their relative order)
        # for a future round once their embeddings land.
        for seq in reversed(deferred_disagg_seqs):
            self.seqs_to_prefill.appendleft(seq)
        return prefill_batch, prefill_batched_token_nums

    def schedule_decode_batch(self, decode_token_budget):
        decode_batch: List[Sequence] = []
        self.check_preempt(min(decode_token_budget, len(self.seqs_to_decode)))
        for _ in range(decode_token_budget):
            if len(self.seqs_to_decode) == 0:
                break
            seq = self.seqs_to_decode.popleft()
            seq.to_compute_token_num = 1
            decode_batch.append(seq)

        self.memory_manager.pre_allocate_page(decode_batch)
        return decode_batch

    def chunked_prefill(self):
        num_tokens_budget = self.maxp

        # Pages to keep free for the in-flight decode batch's projected growth
        # (computed over the full running decode population before we pop any
        # of it into this tick's decode batch).
        reserve_pages = self._decode_reserve_pages()

        # decode
        num_total_decode_seqs = self.get_num_decode_seqs()
        decode_token_budget = self.get_balanced_decode_token_budget(
            num_total_decode_seqs
        )
        decode_token_budget = min(decode_token_budget, num_tokens_budget)

        # split_pd prioritizes draining the prefill queue: when prefills are
        # waiting and we still hold the decode reserve, skip decode this tick.
        prefer_prefill = self.schedule_method == "split_pd" and (
            len(self.seqs_to_prefill) != 0
            and self.get_num_free_pages() >= reserve_pages
        )
        if prefer_prefill:
            decode_token_budget = 0

        decode_batch = self.schedule_decode_batch(decode_token_budget)
        num_tokens_budget -= len(decode_batch)

        # prefill: only admit into the page headroom left *after* reserving for
        # the in-flight decode batch's projected growth (anti-preemption).
        num_tokens_budget = min(
            num_tokens_budget,
            self.page_size * max(self.get_num_free_pages() - reserve_pages, 0),
        )

        prefill_batch, prefill_batched_token_nums = self.schedule_prefill_batch(
            num_tokens_budget,
            max_seqs=self.maxd - len(decode_batch),
            reserve_pages=reserve_pages,
        )

        # Deadlock guard: in split_pd we zeroed the decode budget to favor
        # prefill, but the free headroom (free - reserve) may be too small to
        # actually admit the next waiting seq's pages. That leaves *both* halves
        # empty -> the engine emits no batch and stalls forever ("decode stuck
        # at 0" while prefill piles up). If that happens, run the decode pass we
        # skipped so running seqs keep generating (and freeing pages), which in
        # turn unblocks prefill on a later tick.
        if prefer_prefill and not prefill_batch and not decode_batch:
            fallback_budget = min(
                self.get_balanced_decode_token_budget(num_total_decode_seqs),
                self.maxp,
            )
            decode_batch = self.schedule_decode_batch(fallback_budget)

        # Every TP/PP rank runs an identical copy of the scheduler, so logging
        # the batch stats on all of them floods the console with N duplicate
        # lines per tick. Restrict to the driver of each DP replica (stage-0,
        # TP-0), so every DP group reports its own batch stats exactly once.
        if (
            self.log
            and get_pp_rank() == 0
            and get_tp_rank() == 0
            and time.time() - self.log_time > 1
        ):
            self.log_time = time.time()
            log_info = (
                "#wait: %4d #run: %4d #prefill: %4d #decode: %4d memory_util: %5s %%"
                % (
                    len(self.seqs_to_prefill),
                    num_total_decode_seqs,
                    prefill_batched_token_nums,
                    len(decode_batch),
                    "%.2f" % self.memory_manager.get_memory_util(),
                )
            )
            if isinstance(self.memory_manager, PrefixMemoryManager):
                log_info += " cache_hit_rate: %5s %%" % (
                    "%.2f" % self.memory_manager.get_cache_hit_rate()
                )
                logger.info(log_info)
            else:
                logger.info(log_info)
        # Change 2: relax the reserve a touch each tick; preemptions push it
        # back up. Keeps us from being permanently over-conservative.
        self.new_token_ratio = max(
            self.min_new_token_ratio,
            self.new_token_ratio - self.new_token_ratio_decay,
        )
        # first decode, then prefill
        return decode_batch + prefill_batch

    def token_throttling(self):
        # Pages to keep free for the in-flight decode batch (anti-preemption).
        reserve_pages = self._decode_reserve_pages()
        # prefill
        prefill_token_budget = self.page_size * max(
            self.get_num_free_pages() - reserve_pages, 0
        )
        if get_world_size() > 1 and prefill_token_budget != 0:
            self.update_num_wait_tokens()
            free_ratio = self.memory_manager.get_memory_free()
            # Fraction of the cache held back for in-flight decode growth
            # (replaces the old static kvthresh in the ramp formula).
            reserve_ratio = min(
                0.99, reserve_pages / max(1, self.memory_manager.num_pages)
            )
            # free_ratio in [reserve_ratio,1] | prefill_ratio in [0,1]
            prefill_ratio = (free_ratio - reserve_ratio) / (1 - reserve_ratio)
            prefill_ratio = max(prefill_ratio, 0)
            prefill_token_budget = min(
                round(prefill_ratio * self.maxp), prefill_token_budget
            )
            # Use WT only when the number of waiting seqs is greater than 1
            if len(self.seqs_to_prefill) > 1:
                prefill_token_budget = min(
                    max(self.num_wait_tokens // self.iterp, self.minp),
                    prefill_token_budget,
                )
        else:
            prefill_token_budget = min(self.maxp, prefill_token_budget)

        prefill_batch, prefill_batched_token_nums = self.schedule_prefill_batch(
            prefill_token_budget, max_seqs=self.maxd, reserve_pages=reserve_pages
        )

        # decode
        num_total_decode_seqs = self.get_num_decode_seqs()
        decode_token_budget = self.get_balanced_decode_token_budget(
            num_total_decode_seqs
        )
        # Keep the combined batch within ``max_running_seqs`` rows (see
        # schedule_prefill_batch); prefill is scheduled first here, so the
        # decode half takes whatever row capacity is left.
        decode_token_budget = min(
            decode_token_budget, self.maxd - len(prefill_batch)
        )

        decode_batch = self.schedule_decode_batch(decode_token_budget)

        # Every TP/PP rank runs an identical copy of the scheduler, so logging
        # the batch stats on all of them floods the console with N duplicate
        # lines per tick. Restrict to the driver of each DP replica (stage-0,
        # TP-0), so every DP group reports its own batch stats exactly once.
        if (
            self.log
            and get_pp_rank() == 0
            and get_tp_rank() == 0
            and time.time() - self.log_time > 1
        ):
            self.log_time = time.time()
            log_info = (
                "#wait: %4d/%8d #run: %4d #prefill: %4d #decode: %4d memory_util: %5s %%"
                % (
                    len(self.seqs_to_prefill),
                    self.num_wait_tokens,
                    num_total_decode_seqs,
                    prefill_batched_token_nums,
                    len(decode_batch),
                    "%.2f" % self.memory_manager.get_memory_util(),
                )
            )
            if isinstance(self.memory_manager, PrefixMemoryManager):
                log_info += " cache_hit_rate: %5s %%" % (
                    "%.2f" % self.memory_manager.get_cache_hit_rate()
                )
                logger.info(log_info)
            else:
                logger.info(log_info)
        # Change 2: relax the reserve a touch each tick (see chunked_prefill).
        self.new_token_ratio = max(
            self.min_new_token_ratio,
            self.new_token_ratio - self.new_token_ratio_decay,
        )
        # first decode, then prefill
        return decode_batch + prefill_batch


class OverlapScheduler(Scheduler):
    """Scheduler with deferred/finalize output processing for overlap scheduling."""

    def process_output_deferred(self, future_slot_ids: list[int]):
        """Update seq state with placeholders; return metadata for later finalize."""
        if len(self.batch_running) == 0:
            return None

        schedule_seqs: List[Sequence] = self.batch_running.popleft()
        deferred_seqs = []

        for idx, seq in enumerate(schedule_seqs):
            if getattr(seq, "_overlap_freed", False):
                continue
            seq.computed_token_num += seq.to_compute_token_num
            if seq.computed_prompt:
                placeholder = -future_slot_ids[idx]
                placeholder_pos = len(seq.token_ids)
                seq.append(placeholder)
                deferred_seqs.append((idx, seq, placeholder_pos))
                self.seqs_to_decode.appendleft(seq)

        return deferred_seqs

    def process_output_finalize(self, deferred_seqs, next_tokens, logprobs=None):
        """Replace placeholders with real tokens after D2H / PP token delivery.

        ``logprobs`` (when provided) is a per-batch-row list of
        ``(sampled_logprob, top_ids, top_vals)`` produced by the overlap
        worker; it is keyed by the same ``batch_idx`` as ``next_tokens`` and
        sliced per-seq to that seq's ``num_top_logprobs``.
        """
        if deferred_seqs is None:
            return None

        ipc_package = IPCPackage([])

        for batch_idx, seq, placeholder_pos in deferred_seqs:
            if getattr(seq, "_overlap_freed", False):
                continue

            real_token = next_tokens[batch_idx]
            seq.token_ids[placeholder_pos] = real_token

            # Now that the placeholder holds the real sampled token, register
            # the prefix-cache hash for any page boundary it completed. Decode
            # boundary registration lives only here / in process_output (never
            # in pre_allocate_page) so it is always computed over real tokens
            # (see docs/prefix_cache_overlap_poisoning.md).
            self.model_runner.register_decode_page_hash(seq, placeholder_pos)

            if seq.computed_prompt:
                ipc_package.act_schedule_ids.append(seq.seq_id)
                ipc_package.next_tokens.append(real_token)
                if logprobs is not None and seq.logprobs_enabled:
                    sampled, top_ids, top_vals = logprobs[batch_idx]
                    k = seq.num_top_logprobs
                    ipc_package.logprobs.append(
                        (sampled, top_ids[:k], top_vals[:k])
                    )
                else:
                    ipc_package.logprobs.append(None)
                self._attach_prompt_logprobs(ipc_package, seq)

            is_eos = not seq.ignore_eos and real_token in seq.finish_tokens
            generated_len = placeholder_pos + 1 - seq.raw_prompt_len
            is_max_len = generated_len >= seq.output_len
            if seq.computed_prompt and (is_eos or is_max_len):
                ipc_package.free_ids.append(seq.seq_id)
                # Followers also need to drop this seq from their
                # ``FollowerSeqStore`` (and the VL ``embedding_cache``).
                # We piggyback on the next ``send_schedule_payload`` call;
                # see ``Scheduler.consume_pending_follower_frees``.
                self._pending_follower_frees.append(seq.seq_id)
                seq._overlap_freed = True
                self.model_runner.free(seq)
                try:
                    self.seqs_to_decode.remove(seq)
                except ValueError:
                    pass

        return ipc_package if (
            ipc_package.act_schedule_ids or ipc_package.free_ids
        ) else None
