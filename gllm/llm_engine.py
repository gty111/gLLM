import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

import torch.multiprocessing as mp
import tqdm
from logger import logger

from gllm.comm import IPCPackage, zmqComm
from gllm.id_allocator import IDAllocator
from gllm.model_runner import ModelRunner, OverlapModelRunner
from gllm.overlap_worker import OverlapWorker, run_overlap_worker
from gllm.sequence import Sequence
from gllm.utils import (
    find_free_port,
    get_model_load_pbar,
    init_logger,
    random_uuid,
)
from gllm.worker import Worker, run_worker


def _resolve_sampling_param(value, config_value, default):
    """Caller override > generation config > hardcoded default."""
    if value is not None:
        return value
    return default if config_value is None else config_value


class LLM:
    def __init__(
        self,
        model_path,
        host=None,
        master_addr: str = "0.0.0.0",
        master_port: str = None,
        launch_mode: str = "normal",
        worker_ranks: str = None,
        load_format: str = "auto",
        gpu_memory_util=0.9,
        page_size=16,
        # ``maxd`` caps the number of concurrently running (decode) sequences,
        # which also sizes ``max_running_seqs`` and -- for hybrid GDN/Mamba
        # models -- the SSM working pool (``maxd`` slots) plus the prefix-cache
        # snapshot pool (``4*maxd`` slots, each holding the full per-layer
        # recurrent state). The previous default of 2048 made those pools tens
        # of GiB on linear-attention models and OOM'd before the KV cache was
        # even allocated. 512 matches the api/lm server ``--maxd`` default;
        # lower it if the SSM/snapshot pools are too large for a given model.
        maxd=512,
        maxp=2048,
        minp=32,
        iterp=8,
        init_new_token_ratio=0.7,
        min_new_token_ratio=0.1,
        enable_prefix_caching=True,
        pp_size=1,
        tp_size=1,
        dp_size=1,
        use_ep=True,
        assigned_layers=None,
        schedule_method="chunked_prefill",
        overlap_scheduling=True,
        disable_cuda_graph=False,
        max_cuda_graph_bs=512,
        model_max_length=8192,
        mm_processor_min_pixels=None,
        mm_processor_max_pixels=None,
        disagg_config=None,
        mla_decode_backend="flashmla",
    ):
        init_logger()
        self.model_path = model_path
        self.load_format = load_format
        # Encoder-disaggregation config (gllm.disagg.config.DisaggConfig) or
        # None for the monolith. The role flags feed the model loader (parent
        # process); the whole object is forwarded to the spawned worker for the
        # LM-side manager. ``is_disagg_lm`` is the request-time gate read by the
        # api server (replaces the old GLLM_DISAGG_LM env read).
        self.disagg_config = disagg_config
        self.is_disagg_lm = bool(disagg_config is not None and disagg_config.is_lm)
        skip_visual = disagg_config.skip_visual if disagg_config is not None else False
        skip_language = (
            disagg_config.skip_language if disagg_config is not None else False
        )
        if overlap_scheduling and pp_size > 1:
            logger.warning(
                "overlap_scheduling is not supported with pp_size>1; disabling overlap"
            )
            overlap_scheduling = False
        model_runner_cls = OverlapModelRunner if overlap_scheduling else ModelRunner
        self.model_runner = model_runner_cls(
            load_format=load_format,
            model_path=model_path,
            gpu_memory_util=gpu_memory_util,
            page_size=page_size,
            enable_prefix_caching=enable_prefix_caching,
            maxp=maxp,
            maxd=maxd,
            minp=minp,
            iterp=iterp,
            init_new_token_ratio=init_new_token_ratio,
            min_new_token_ratio=min_new_token_ratio,
            schedule_method=schedule_method,
            disable_cuda_graph=disable_cuda_graph,
            max_cuda_graph_bs=max_cuda_graph_bs,
            model_max_length=model_max_length,
            mm_processor_min_pixels=mm_processor_min_pixels,
            mm_processor_max_pixels=mm_processor_max_pixels,
            skip_visual=skip_visual,
            skip_language=skip_language,
            mla_decode_backend=mla_decode_backend,
        )
        self.pp_size = pp_size
        self.tp_size = tp_size
        # Data-parallel (DP) attention + Expert-Parallel MoE. Run ``dp_size``
        # full-model replicas (one per GPU). Each replica owns its own
        # scheduler + KV cache and serves a disjoint shard of requests
        # (round-robined by the frontend), so the MLA latent KV cache is
        # *sharded* across replicas instead of being replicated on every TP
        # rank. The routed experts are in turn sharded across all replicas with
        # ``EP = dp_size * tp_size`` (here tp_size == 1): each MoE layer gathers
        # the global batch, runs its ``1/dp_size`` expert shard, and all-reduces
        # the result (see ``DeepseekV2MOE._forward_dp_ep``).
        self.dp_size = dp_size
        # Round-robin cursor used by the frontend to spread new requests across
        # DP replicas.
        self._dp_rr = 0
        self.use_ep = use_ep
        self.host = host
        self.master_addr = master_addr
        # Auto-allocate a free NCCL rendezvous port when unset, so offline /
        # library usage (constructing the engine directly) gets a working port
        # without the caller having to pick one.
        if master_port is None:
            master_port = str(find_free_port(master_addr))
            logger.info(f"Auto-selected NCCL master_port {master_port}")
        self.master_port = master_port
        self.launch_mode = launch_mode
        self.worker_ranks = worker_ranks
        self.id_allocator = IDAllocator(0, 99999)
        self.finish_tokens = (
            self.model_runner.model_loader.generation_config.eos_token_id
        )
        if type(self.finish_tokens) == int:
            self.finish_tokens = [self.finish_tokens]
        self.model_max_length = self.model_runner.model_max_length
        self.generation_config = self.model_runner.model_loader.generation_config

        self.assigned_layers = assigned_layers
        self.schedule_method = schedule_method
        self.overlap_scheduling = overlap_scheduling

        logger.info(f"Schedule method: {schedule_method}")
        if self.overlap_scheduling:
            logger.info(
                "Overlap scheduling enabled (FutureMap + CPU/GPU overlap, TP only)"
            )

        # Interact with workers
        self.wait_lists: List[Sequence] = []
        self.abort_ids: List[int] = []
        self.running_maps: Dict[int, Sequence] = dict()  # seq_id => Sequence
        self.async_streams = None
        # Guards the newly-arrived ``wait_lists`` / ``abort_ids`` hand-off queues.
        # The async server runs both request intake (``add_requests``) and the
        # engine step (``send_ipc_package``) on the event loop's default
        # ``ThreadPoolExecutor`` (many threads via ``make_async``), so the two
        # touch these lists concurrently. Without this lock a request appended in
        # the window between the dispatch loop and the ``wait_lists = []`` reset
        # was silently dropped -- never sent to any worker, its stream never
        # finished, and the client hung forever (a rare tail-of-run stall under
        # high concurrency). Snapshot-and-clear under the lock makes it atomic.
        self._pending_lock = threading.Lock()

        # Init workers
        self.init_workers()

        # wait worker start
        self.wait_workers()

    def wait_workers(self):
        while True:
            num_worker_start = 0
            for i in self.mp_alive:
                if i == -1:
                    sys.exit()
                num_worker_start += i
            if num_worker_start == self.num_workers:
                break
            time.sleep(1)

    def init_workers(self):
        if self.launch_mode != "normal":
            if self.worker_ranks is None:
                logger.error(
                    "Please specify arg --ranks when the launching mode is master/slave"
                )
                sys.exit(1)
            self.act_worker_ranks = [int(i) for i in self.worker_ranks.split(",")]
            assert len(self.act_worker_ranks) != 0
        else:
            self.act_worker_ranks = list(
                range(self.pp_size * self.tp_size * self.dp_size)
            )
        self.num_workers = len(self.act_worker_ranks)

        self.ctx = mp.get_context("spawn")
        self.mp_alive = self.ctx.Array("i", [0 for i in range(self.num_workers)])
        self.mp_load_progress = self.ctx.Array(
            "i", [0 for _ in range(self.num_workers * 2)]
        )

        ipc_path_prefix = random_uuid()
        self.schedule_path = f"ipc:///tmp/{ipc_path_prefix}_gllm_schedule"
        self.output_path = f"ipc:///tmp/{ipc_path_prefix}_gllm_output"
        self.token_path = f"ipc:///tmp/{ipc_path_prefix}_gllm_token"

        self.comm = zmqComm(
            self.host,
            self.launch_mode,
            self.master_addr,
            self.schedule_path,
            self.output_path,
            self.token_path,
            frontend=True,
            dp_size=self.dp_size,
        )
        self.comm.init()

        logger.info(
            f"Launching worker {self.act_worker_ranks}, PP size {self.pp_size}, TP size {self.tp_size}"
        )
        # Build every worker process object first (cheap), then fire all the
        # ``process.start()`` calls concurrently. For the spawn start method
        # ``start()`` blocks the parent while it pickles the worker and writes
        # the bootstrap pipe, so launching serially makes the parent pay that
        # cost N times in sequence. The processes themselves are independent,
        # so starting them from a thread pool overlaps the per-worker spawn
        # latency. The expensive work (NCCL rendezvous, weight load, CUDA graph
        # capture) already runs concurrently inside the children.
        self.process_list = []
        for local_rank, rank in enumerate(self.act_worker_ranks):
            if self.dp_size > 1:
                # (PP x) DP+TP+EP: the world is pp_size*dp_size*tp_size ranks laid
                # out as a pp x dp x tp grid, global_rank = pp*S + dp*tp_size + tp
                # with S = dp_size*tp_size (the per-stage size). Attention is TP
                # *within* each DP group; experts are sharded across a stage's S
                # ranks. Reduces to the single-stage layout when pp_size == 1.
                stage_size = self.dp_size * self.tp_size
                pp_rank = rank // stage_size
                within = rank % stage_size
                dp_rank = within // self.tp_size
                tp_rank = within % self.tp_size
            else:
                pp_rank = rank // self.tp_size
                tp_rank = rank % self.tp_size
                dp_rank = 0
            self.build_worker(local_rank, pp_rank, tp_rank, dp_rank)

        if self.num_workers == 1:
            self.process_list[0].start()
        else:
            with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
                futures = [pool.submit(p.start) for p in self.process_list]
                for f in futures:
                    # Re-raise any spawn failure in the parent instead of
                    # silently leaving a half-launched fleet.
                    f.result()

        if self.load_format == "auto":
            self.load_progress()

    def build_worker(self, local_rank, pp_rank, tp_rank, dp_rank=0):
        if self.overlap_scheduling:
            worker_cls = OverlapWorker
            run_target = run_overlap_worker
        else:
            worker_cls = Worker
            run_target = run_worker
        comm = zmqComm(
            self.host,
            self.launch_mode,
            self.master_addr,
            self.schedule_path,
            self.output_path,
            self.token_path,
            dp_rank=dp_rank,
            dp_size=self.dp_size,
        )
        worker = worker_cls(
            self.model_runner,
            local_rank,
            pp_rank,
            tp_rank,
            self.pp_size,
            self.tp_size,
            self.use_ep,
            self.master_addr,
            self.master_port,
            comm,
            self.mp_alive,
            self.mp_load_progress,
            self.assigned_layers,
            self.schedule_method,
            self.disagg_config,
        )
        # DP bookkeeping (logging + is_dp_attn); no signature change to Worker.
        worker.dp_rank = dp_rank
        worker.dp_size = self.dp_size
        process = self.ctx.Process(
            target=run_target,
            args=(worker,),
            daemon=True,
        )
        self.process_list.append(process)
        return process

    def load_progress(self):
        total_weights = 0
        while True:
            self.check_worker_alive()
            ready = True
            total_weights = 0
            for i in range(self.num_workers):
                if self.mp_load_progress[i * 2] == 0:
                    ready = False
                    continue
                total_weights += self.mp_load_progress[i * 2]
            if ready:
                break
        pbar = get_model_load_pbar(total_weights)
        last_total_weights = 0
        while True:
            self.check_worker_alive()
            cur_total_weights = 0
            for i in range(self.num_workers):
                cur_total_weights += self.mp_load_progress[i * 2 + 1]
            pbar.update(cur_total_weights - last_total_weights)
            last_total_weights = cur_total_weights
            if cur_total_weights == total_weights:
                break

    def check_worker_alive(self):
        for i in self.mp_alive:
            if i == -1:
                sys.exit()

    def add_requests(self, requests: List[Sequence]):
        with self._pending_lock:
            self.wait_lists.extend(requests)

    def recv_ipc_package(self):
        """
        return: number of finished requests in each schedule

        Drains *all* output packages currently queued. Under DP attention the
        ``dp_size`` replicas each PUSH their own output packages into the shared
        frontend PULL (fan-in), so more than one package can be waiting per
        schedule tick; draining keeps the frontend from lagging behind the
        replicas. Each package is self-describing (its own ``act_schedule_ids``
        aligned with ``next_tokens``), so ordering across replicas is irrelevant
        -- everything is keyed by ``seq_id`` via ``running_maps``.
        """
        num_finish = 0
        while True:
            ipc_package: IPCPackage = self.comm.recv_output()
            if ipc_package is None:
                break
            num_finish += self._apply_ipc_package(ipc_package)
        return num_finish

    def _apply_ipc_package(self, ipc_package):
        if ipc_package is not None:
            for idx, id in enumerate(ipc_package.act_schedule_ids):
                # Under overlap scheduling a worker can emit a trailing token
                # for a sequence it freed one step earlier (EOS detected after
                # the next step was already launched), so the driver may have
                # popped ``id`` already. Drop such a stale post-free token
                # instead of crashing the engine on a missing running_maps key.
                seq: Sequence = self.running_maps.get(id)
                if seq is None:
                    continue
                if len(ipc_package.next_tokens) != 0:
                    seq.append(ipc_package.next_tokens[idx])
                    if self.async_streams:
                        self.async_streams[id].put(
                            seq.detokenize_inc(self.model_runner.tokenizer)
                        )
                if id in ipc_package.free_ids:
                    self.running_maps.pop(id)
                    if self.async_streams:
                        self.async_streams[id].finish()
                        del self.async_streams[id]
            self.free_finish_ids(ipc_package.free_ids)
            return len(ipc_package.free_ids)
        return 0

    def send_ipc_package(self, log=True):
        # Atomically claim the pending intake so a concurrent ``add_requests``
        # (running on another executor thread) can't have a request dropped in
        # the gap between reading ``wait_lists`` and clearing it. Anything that
        # arrives after this swap simply goes to the next tick's fresh list.
        with self._pending_lock:
            if len(self.wait_lists) == 0 and len(self.abort_ids) == 0:
                return
            wait_lists = self.wait_lists
            abort_ids = self.abort_ids
            self.wait_lists = []
            self.abort_ids = []

        for seq in wait_lists:
            self.running_maps[seq.seq_id] = seq
        if self.dp_size > 1:
            self._send_ipc_package_dp(wait_lists, abort_ids, log)
            return
        ipc_package = IPCPackage(wait_lists)
        if len(abort_ids) != 0:
            logger.warning(
                f"Abort {len(abort_ids)} request(s) due to loss of network connection"
            )
        ipc_package.abort_ids = abort_ids
        ipc_package.log = log
        self.comm.send_ipc_package(ipc_package)

    def _send_ipc_package_dp(self, wait_lists, abort_ids, log=True):
        """Spread new requests across DP replicas; broadcast aborts to all.

        New sequences are sent one per package so zmq delivers each to exactly
        one replica (that replica then owns the seq's KV for its whole lifetime).
        A seq's target replica is ``seq.target_dp`` when the request arrived on a
        per-replica HTTP endpoint (``--endpoint-per-dp``); otherwise it is
        round-robined across replicas. Aborts don't carry replica ownership on
        the frontend, so they are broadcast to every replica; a replica that
        doesn't own the seq_id simply ignores the unknown id.

        ``wait_lists`` / ``abort_ids`` are the snapshots already claimed (and
        cleared from ``self``) by :meth:`send_ipc_package` under the lock.
        """
        for seq in wait_lists:
            pkg = IPCPackage([seq])
            pkg.log = log
            target = getattr(seq, "target_dp", None)
            if target is None:
                target = self._dp_rr
                self._dp_rr = (self._dp_rr + 1) % self.dp_size
            self.comm.send_ipc_package_to_dp(pkg, target)
        if len(abort_ids) != 0:
            logger.warning(
                f"Abort {len(abort_ids)} request(s) due to loss of network connection"
            )
            abort_pkg = IPCPackage([])
            abort_pkg.abort_ids = abort_ids
            abort_pkg.log = log
            self.comm.broadcast_ipc_package_to_dp(abort_pkg)

    def send_control_command(self, control_cmd: str):
        ipc_package = IPCPackage([])
        ipc_package.control_cmd = control_cmd
        if self.dp_size > 1:
            # Control commands (profiler start/stop) must reach every replica.
            self.comm.broadcast_ipc_package_to_dp(ipc_package)
        else:
            self.comm.send_ipc_package(ipc_package)

    def start_profile(self):
        self.send_control_command("start_profile")

    def stop_profile(self):
        self.send_control_command("stop_profile")

    def schedule(self, log=True):
        self.check_worker_alive()
        num_finish_seqs = self.recv_ipc_package()
        self.send_ipc_package(log)
        return num_finish_seqs

    def check_seq_length(self, token_ids: List[int], output_len: int):
        max_seq_length = (
            len(token_ids) + output_len if output_len is not None else len(token_ids)
        )
        if max_seq_length > self.model_max_length:
            logger.warning(
                f"Ignore seq due to the length ({max_seq_length}) exceeds max model len({self.model_max_length})"
            )
            return False
        else:
            return True

    def allocate_seq(
        self,
        token_ids: List[int],
        output_len=None,
        ignore_eos=False,
        temperature=None,
        top_p=None,
        top_k=None,
        repetition_penalty=None,
        mm_contents=None,
        mm_items=None,
    ):
        # Models without a ``generation_config.json`` (e.g. Qwen3.5-0.8B)
        # leave the HF ``GenerationConfig`` defaults as ``None``, which then
        # crashes ``InputData.prepare_sample``'s ``async_tensor_h2d`` H2D
        # copy. Fall back to neutral greedy defaults when both the caller
        # and the model config leave the field unset.
        gen = self.generation_config
        temperature = _resolve_sampling_param(temperature, gen.temperature, 1.0)
        top_p = _resolve_sampling_param(top_p, gen.top_p, 1.0)
        # top_k=1 selects the greedy fast-path in ``Sampler``; do not inherit
        # ``generation_config.top_k`` when the caller leaves it unset.
        top_k = 1 if top_k is None else top_k
        repetition_penalty = _resolve_sampling_param(
            repetition_penalty, gen.repetition_penalty, 1.0
        )
        seq = Sequence(
            self.id_allocator.allocate(),
            token_ids,
            self.finish_tokens,
            output_len,
            ignore_eos,
            temperature,
            top_p,
            top_k,
            repetition_penalty,
            mm_contents,
        )
        # Encoder-disaggregation: the ordered raw mm items the encoder will
        # process. Present only on the disaggregated LM frontend; ``None`` for
        # text and for the monolith path.
        seq.mm_items = mm_items
        return seq

    def free_finish_ids(self, finish_ids: List[int]):
        for id in finish_ids:
            self.id_allocator.free(id)

    def generate(
        self,
        prompts: List[str] = None,
        tokens: List[List[int]] = None,
        output_lens: List[int] = None,
        temperature=None,
        top_p=None,
        top_k=None,
    ):
        seqs: List[Sequence] = []
        assert prompts is not None or tokens is not None
        num_seqs = len(prompts) if prompts is not None else len(tokens)
        for idx in range(num_seqs):
            token_ids = (
                tokens[idx]
                if tokens is not None
                else self.model_runner.encode(prompts[idx])
            )
            output_len_each = output_lens[idx] if output_lens is not None else None
            if self.check_seq_length(token_ids, output_len_each):
                seq = self.allocate_seq(
                    token_ids, output_len_each, False, temperature, top_p, top_k
                )
                seqs.append(seq)
        self.add_requests(seqs)

        pbar = tqdm.tqdm(total=len(seqs), ncols=100)
        while pbar.n != len(seqs):
            num_finish_seqs = self.schedule(log=False)
            pbar.update(num_finish_seqs)

        for seq in seqs:
            seq.prompt = self.model_runner.decode(seq[: seq.raw_prompt_len])
            seq.output = self.model_runner.decode(seq[seq.raw_prompt_len :])

        return seqs

    def chat(self):
        architecture = self.model_runner.model_loader.architecture
        print(
            "\nWelcome to the chatbot!\n"
            "Type '\\exit' to exit the chatbot.\n"
            "Type '\\clear' to clear the chatbot's history.\n"
        )
        history = []
        while True:
            prompt = input(">>> ")
            print()
            if prompt == "\\clear":
                history = []
                continue
            elif prompt == "\\exit":
                break

            if architecture == "ChatGLMModel" and hasattr(
                self.model_runner.tokenizer, "build_chat_input"
            ):
                tokens = (
                    self.model_runner.tokenizer.build_chat_input(
                        prompt, history=history, role="user"
                    )
                    .get("input_ids")
                    .numpy()
                    .tolist()[0]
                )
            else:
                history.append({"role": "user", "content": prompt})
                tokens = self.model_runner.encode(history, chat=True)

            seq = self.allocate_seq(tokens)
            self.add_requests([seq])
            while len(self.running_maps) != 0 or len(self.wait_lists) != 0:
                self.schedule(log=False)
                print(
                    seq.detokenize_inc(self.model_runner.tokenizer), end="", flush=True
                )
            print("\n")

            output_text = self.model_runner.decode(seq[seq.raw_prompt_len :])

            if architecture == "ChatGLMModel" and hasattr(
                self.model_runner.tokenizer, "build_chat_input"
            ):
                _, history = self.model_runner.model.process_response(
                    output_text, history
                )
            else:
                history.append({"role": "assistant", "content": output_text})
