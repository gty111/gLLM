import copy
import random
import time
from collections import deque
from typing import List

from logger import logger

from gllm.comm import IPCPackage
from gllm.dist_utils import get_world_size
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
        self.kvthresh = model_runner.kvthresh
        self.num_kvthresh_pages = int(
            self.kvthresh * self.memory_manager.get_num_free_pages()
        )

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
        # abort ids
        self.abort_ids = set()
        # log
        self.log = True
        # schedule method
        self.schedule = self.dispatch_schedule_method()

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

    def update_num_wait_tokens(self):
        self.num_wait_tokens = sum(
            len(i) - i.computed_token_num for i in self.seqs_to_prefill
        )

    def add_abort_ids(self, abort_ids):
        self.abort_ids.update(abort_ids)

    def add_new_requests(self, seqs):
        self.seqs_to_prefill.extend(seqs)

    def add_next_tokens(self, next_tokens):
        self.next_tokens_queue.append(next_tokens)

    def set_log(self, log):
        self.log = log

    def process_output(self):
        if len(self.next_tokens_queue) == 0:
            return None

        schedule_seqs: List[Sequence] = self.batch_running.popleft()
        next_tokens = self.next_tokens_queue.popleft()

        ipc_package = IPCPackage([])

        for idx, seq in enumerate(schedule_seqs):
            seq.computed_token_num += seq.to_compute_token_num
            if seq.computed_prompt:
                ipc_package.act_schedule_ids.append(seq.seq_id)
                ipc_package.next_tokens.append(next_tokens[idx])
                seq.append(next_tokens[idx])
            if seq.is_finish:
                ipc_package.free_ids.append(seq.seq_id)
                self.model_runner.free(seq)
            elif seq.computed_prompt:
                self.seqs_to_decode.appendleft(seq)
            else:  # unfinished prefill seqs
                pass
        return ipc_package

    def check_preempt(self, num_tokens_to_allocate):
        preempt_seqs = []
        while (
            self.get_num_free_pages() < num_tokens_to_allocate
            and len(self.seqs_to_decode) != 0
        ):
            seq_to_preempt = self.seqs_to_decode.popleft()
            self.model_runner.free(seq_to_preempt)
            seq_to_preempt.preempt()
            preempt_seqs.append(seq_to_preempt)

        self.seqs_to_prefill.extendleft(preempt_seqs)

        self.num_preempt_seqs += len(preempt_seqs)
        if (
            self.num_preempt_seqs - self.log_num_preempt_seqs
            >= self.delta_log_num_preempt_seqs
        ):
            self.log_num_preempt_seqs = self.num_preempt_seqs
            self.delta_log_num_preempt_seqs *= 2
            logger.warning(
                f"#Preempted seqs: {self.num_preempt_seqs}, "
                "Try increase --kvthresh or the performance is poor!"
            )

    def check_abort_seqs_list(self, seqs: deque, ipc_package: IPCPackage):
        for seq in list(seqs):
            if len(self.abort_ids) == 0:
                break
            id = seq.seq_id
            if id in self.abort_ids:
                ipc_package.free_ids.append(id)
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

    def post_schedule(self, schedule_seqs: List[Sequence]):
        # We drop token_ids and extract to_compute_tokens here,
        # since it is not fully used and may increase overhead of zmq.
        post_schedule_seqs = []
        for seq in schedule_seqs:
            if seq.has_schedule:
                post_schedule_seq = copy.copy(seq)
                # MM prefill may still need full prompt token_ids to
                # build (or rebuild) cached multimodal embeddings/positions.
                # Keep token_ids for unfinished MM prefills; drop otherwise to
                # reduce IPC payload.
                keep_full_token_ids = self.model_runner.use_mm and (not seq.computed_prompt)
                post_schedule_seq.token_ids = seq.token_ids if keep_full_token_ids else None
                if not keep_full_token_ids:
                    post_schedule_seq.to_compute_tokens = seq[
                        seq.computed_token_num : seq.seq_len
                    ]
                post_schedule_seqs.append(post_schedule_seq)
            else:
                post_schedule_seqs.append(seq)
                seq.has_schedule = True
        return post_schedule_seqs

    def schedule_once(self):
        if (
            len(self.seqs_to_decode) + len(self.seqs_to_prefill) != 0
            and len(self.batch_running) < self.pp_size
        ):
            schedule_seqs = self.schedule()
            if len(schedule_seqs) != 0:
                self.batch_running.append(schedule_seqs)
                post_schedule_seqs = self.post_schedule(schedule_seqs)
                return post_schedule_seqs

        return []

    def get_balanced_decode_token_budget(self, num_total_decode_seqs):
        if num_total_decode_seqs < self.pp_size:
            decode_token_budget = 1
        else:
            # here we add num_total_decode_seqs to random.randint(0,self.pp_size-1))
            # because we want to solve the situation when #seqs=5 pp_size=4
            decode_token_budget = (
                num_total_decode_seqs + random.randint(0, self.pp_size - 1)
            ) // self.pp_size

        decode_token_budget = min(self.maxd, decode_token_budget)
        return decode_token_budget

    def schedule_prefill_batch(self, prefill_token_budget):
        prefill_batch: List[Sequence] = []
        unfinish_prefill_seqs = deque()
        prefill_batched_token_nums = 0
        while len(self.seqs_to_prefill) != 0 and prefill_token_budget != 0:
            seq = self.seqs_to_prefill.popleft()
            if (
                isinstance(self.memory_manager, PrefixMemoryManager)
                and seq.computed_token_num == 0
            ):
                self.memory_manager.pre_allocate_computed_page([seq])
            if len(seq) - seq.computed_token_num <= prefill_token_budget:
                seq.to_compute_token_num = len(seq) - seq.computed_token_num
                prefill_batched_token_nums += seq.to_compute_token_num
                prefill_token_budget -= seq.to_compute_token_num
            else:
                prefill_batched_token_nums += prefill_token_budget
                seq.to_compute_token_num = prefill_token_budget
                prefill_token_budget = 0
            self.memory_manager.pre_allocate_page([seq])
            prefill_batch.append(seq)
            if seq.computed_token_num + seq.to_compute_token_num < seq.prompt_len:
                seq_new = copy.deepcopy(seq)
                seq_new.computed_token_num += seq_new.to_compute_token_num
                unfinish_prefill_seqs.appendleft(seq_new)

        self.seqs_to_prefill.extendleft(unfinish_prefill_seqs)
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

        # decode
        num_total_decode_seqs = self.get_num_decode_seqs()
        decode_token_budget = self.get_balanced_decode_token_budget(
            num_total_decode_seqs
        )
        decode_token_budget = min(decode_token_budget, num_tokens_budget)

        if self.schedule_method == "split_pd" and (
            len(self.seqs_to_prefill) != 0
            and self.get_num_free_pages() >= self.num_kvthresh_pages
        ):
            decode_token_budget = 0

        decode_batch = self.schedule_decode_batch(decode_token_budget)
        num_tokens_budget -= len(decode_batch)

        # prefill
        num_tokens_budget = min(
            num_tokens_budget,
            self.page_size
            * max(self.get_num_free_pages() - self.num_kvthresh_pages, 0),
        )

        prefill_batch, prefill_batched_token_nums = self.schedule_prefill_batch(
            num_tokens_budget
        )

        if self.log and time.time() - self.log_time > 1:
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
        # first decode, then prefill
        return decode_batch + prefill_batch

    def token_throttling(self):
        # prefill
        prefill_token_budget = self.page_size * max(
            self.get_num_free_pages() - self.num_kvthresh_pages, 0
        )
        if get_world_size() > 1 and prefill_token_budget != 0:
            self.update_num_wait_tokens()
            free_ratio = self.memory_manager.get_memory_free()
            # free_ratio in [kvthresh,1] | prefill_ratio in [0,1]
            prefill_ratio = (free_ratio - self.kvthresh) / (1 - self.kvthresh)
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
            prefill_token_budget
        )

        # decode
        num_total_decode_seqs = self.get_num_decode_seqs()
        decode_token_budget = self.get_balanced_decode_token_budget(
            num_total_decode_seqs
        )

        decode_batch = self.schedule_decode_batch(decode_token_budget)

        if self.log and time.time() - self.log_time > 1:
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
        # first decode, then prefill
        return decode_batch + prefill_batch

class AsyncScheduler(Scheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def process_output_deferred(self, future_indices):
        """Deferred output processing for async scheduling with FutureMap.

        Unlike process_output() which needs real CPU token values, this method:
        1. Updates computed_token_num (no token value needed)
        2. Appends a NEGATIVE placeholder to seq.token_ids for decode seqs
        3. Puts decode seqs back into seqs_to_decode immediately
        4. Does NOT check is_finish (deferred to process_output_finalize)
        5. Returns list of (batch_idx, seq) pairs for later finalization

        The negative placeholder in seq.token_ids will be resolved on the GPU
        side via FutureMap.resolve_future() during the next batch's forward pass.
        """
        if len(self.batch_running) == 0:
            return None

        schedule_seqs: List[Sequence] = self.batch_running.popleft()

        # List of (batch_idx, seq) — batch_idx is needed to index into
        # next_tokens during finalize, since the batch may contain a mix
        # of prefill and decode seqs.
        deferred_seqs = []

        for idx, seq in enumerate(schedule_seqs):
            # Skip seqs that were freed by a previous finalize but got
            # re-scheduled before the free was visible.
            if getattr(seq, "_async_freed", False):
                continue

            seq.computed_token_num += seq.to_compute_token_num
            if seq.computed_prompt:
                # Append negative future index as placeholder token.
                placeholder = -future_indices.indices[idx].item()
                placeholder_pos = len(seq.token_ids)  # position before append
                seq.append(placeholder)
                deferred_seqs.append((idx, seq, placeholder_pos))
                self.seqs_to_decode.appendleft(seq)
            # unfinished prefill seqs: do nothing (same as sync path)

        return deferred_seqs

    def process_output_finalize(self, deferred_seqs, next_tokens):
        """Finalize deferred output processing after D2H copy completes.

        Called one+ iterations later when step_collect provides real CPU tokens.
        Replaces placeholder tokens with real values, checks is_finish, and
        builds the IPC package for the frontend.

        Args:
            deferred_seqs: list of (batch_idx, seq) from process_output_deferred
            next_tokens: CPU token list from step_collect_async (all batch entries)
        """
        ipc_package = IPCPackage([])

        for batch_idx, seq, placeholder_pos in deferred_seqs:
            # Skip seqs already freed by an earlier finalize
            if getattr(seq, "_async_freed", False):
                continue

            real_token = next_tokens[batch_idx]
            # Replace the placeholder at the exact position it was written,
            # not [-1], because later batches may have appended more placeholders.
            seq.token_ids[placeholder_pos] = real_token

            ipc_package.act_schedule_ids.append(seq.seq_id)
            ipc_package.next_tokens.append(real_token)

            # Check finish using the real token directly, not seq.is_finish,
            # because seq[-1] may be a placeholder from a later batch.
            is_eos = not seq.ignore_eos and real_token in seq.finish_tokens
            # Use placeholder_pos+1 as the generated length at this point,
            # since later batches may have appended more placeholders.
            generated_len = placeholder_pos + 1 - seq.prompt_len
            is_max_len = generated_len >= seq.output_len
            if seq.computed_prompt and (is_eos or is_max_len):
                ipc_package.free_ids.append(seq.seq_id)
                # Mark freed so future deferred/finalize calls skip it.
                seq._async_freed = True
                self.model_runner.free(seq)
                # Remove from seqs_to_decode (added optimistically)
                try:
                    self.seqs_to_decode.remove(seq)
                except ValueError:
                    pass  # already removed or re-scheduled

        return ipc_package