import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import torch.multiprocessing as mp
import tqdm
from logger import logger

from gllm.distributed.comm import IPCPackage, zmqComm
from gllm.tokenizers.reasoning import decode_stream_delta, reasoning_control_tokens
from gllm.runtime.id_allocator import IDAllocator
from gllm.runtime.model_runner import ModelRunner, OverlapModelRunner
from gllm.runtime.sequence import GenerationSequence, resolve_output_len
from gllm.utils import (
    StreamOutput,
    find_free_port,
    get_model_load_pbar,
    init_logger,
    random_uuid,
)
from gllm.workers.overlap import OverlapWorker, run_overlap_worker
from gllm.workers.worker import Worker, run_worker


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
        # which also sizes ``max_running_seqs``. Hybrid GDN/Mamba models claim
        # their working/checkpoint state dynamically from the same physical
        # cache arena as KV; ``maxd`` is therefore an admission/buffer bound,
        # not a separately preallocated SSM cache size.
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
        piecewise_cuda_graph=True,
        max_piecewise_cuda_graph_tokens=None,
        max_cuda_graph_bs=512,
        model_max_length=8192,
        mm_processor_min_pixels=None,
        mm_processor_max_pixels=None,
        disagg_config=None,
        attention_backend="flashinfer",
        mla_decode_backend="fa4",
        mla_cache_dtype="bf16",
        mamba_ssm_cache_dtype="auto",
        mtp_enabled=None,
        mtp_k=3,
        mtp_max_batch=0,
        ssm_snapshot_stride_tokens=256,
        worker_transport_base_port=None,
        # --- Frontend/worker decoupling (see docs/frontend_worker_decoupling.md) ---
        # ``standalone_frontend=True``: this process is a pure frontend -- it
        # does NOT spawn worker processes and does NOT initialize a GPU. It
        # discovers a separately deployed worker fleet through
        # ``worker_endpoint_file`` (written by ``worker_server``). ``standalone_worker``:
        # this process hosts a worker fleet whose frontend-facing sockets are
        # advertised in ``worker_endpoint_file`` instead of spawning.
        standalone_frontend=False,
        standalone_worker=False,
        worker_endpoint_file=None,
    ):
        init_logger()
        self.standalone_frontend = bool(standalone_frontend)
        self.standalone_worker = bool(standalone_worker)
        self.worker_endpoint_file = worker_endpoint_file
        self._worker_writer = None  # WorkerEndpointWriter (standalone worker)
        self.worker_transport_base_port = worker_transport_base_port
        # Frontend session epoch (echoed per output row via
        # IPCPackage.sessions / free_sessions, see
        # Worker.translate_output_for_frontend). Every (re)started frontend
        # mints a NEW epoch and stamps it on each dispatched seq
        # (``seq.frontend_session``) so a surviving worker fleet can tell
        # our request ids apart from the ids a dead frontend's in-flight
        # requests were still producing -- both id pools restart at 0, so
        # matching by seq_id alone cross-links the two sessions.
        self.frontend_epoch = random_uuid()
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
        if overlap_scheduling and pp_size > 1 and dp_size > 1:
            logger.warning(
                "overlap_scheduling with combined PP+DP-attention is not yet "
                "supported; disabling overlap"
            )
            overlap_scheduling = False
        model_runner_cls = OverlapModelRunner if overlap_scheduling else ModelRunner
        if self.standalone_frontend:
            # Pure frontend: no GPU, no worker spawn. Build a lightweight
            # metadata-only runner (tokenizer + config) instead of the full
            # GPU runner; every attribute this class reads off it is provided
            # by ModelRunner.load_metadata.
            self.model_runner = model_runner_cls.load_metadata(
                load_format=load_format,
                model_path=model_path,
                schedule_method=schedule_method,
                model_max_length=model_max_length,
            )
        else:
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
                piecewise_cuda_graph=piecewise_cuda_graph,
                max_piecewise_cuda_graph_tokens=max_piecewise_cuda_graph_tokens,
                max_cuda_graph_bs=max_cuda_graph_bs,
                model_max_length=model_max_length,
                mm_processor_min_pixels=mm_processor_min_pixels,
                mm_processor_max_pixels=mm_processor_max_pixels,
                skip_visual=skip_visual,
                skip_language=skip_language,
                attention_backend=attention_backend,
                mla_decode_backend=mla_decode_backend,
                mla_cache_dtype=mla_cache_dtype,
                mamba_ssm_cache_dtype=mamba_ssm_cache_dtype,
                mtp_enabled=mtp_enabled,
                mtp_k=mtp_k,
                mtp_max_batch=mtp_max_batch,
                ssm_snapshot_stride_tokens=ssm_snapshot_stride_tokens,
            )
        self._reasoning_controls = reasoning_control_tokens(self.model_runner.tokenizer)
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
                "Overlap scheduling enabled (FutureMap + CPU/GPU overlap, TP/PP)"
            )

        # Interact with workers
        self.wait_lists: List[GenerationSequence] = []
        self.abort_ids: List[int] = []
        self.running_maps: Dict[int, GenerationSequence] = dict()  # seq_id => GenerationSequence
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
        if self.standalone_frontend:
            self.num_workers = 0
            self.process_list = []
            self.act_worker_ranks = []
            self._init_standalone_frontend()
        else:
            self.init_workers()

        # wait worker start
        if not self.standalone_frontend:
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
        # The worker child has bound its frontend-facing sockets and finished
        # initialization (mp_alive is set at the end of Worker.init). Publish
        # the endpoint file *now* so a connecting frontend can never buffer
        # work into a socket nobody is reading yet.
        if self.standalone_worker:
            self._publish_worker_endpoint()

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
        # Delay CUDA init in spawned children until the pickled target runs.
        # Without this, a parent that touched ``torch.cuda`` (e.g. device-count
        # probes at import) initialises the CUDA primary context on the *first
        # visible* device, and the child inherits that context even when it
        # sets its own ``CUDA_VISIBLE_DEVICES`` -- so a GPU-pinned worker would
        # actually allocate on the wrong (often busy) GPU. The warmup delay
        # makes each child re-read its own ``CUDA_VISIBLE_DEVICES`` before any
        # CUDA call, which is exactly what standalone GPU pinning relies on.
        try:
            self.ctx.set_warmup_delay(1.0)
        except Exception:
            pass
        self.mp_alive = self.ctx.Array("i", [0 for i in range(self.num_workers)])
        self.mp_load_progress = self.ctx.Array(
            "i", [0 for _ in range(self.num_workers * 2)]
        )

        ipc_path_prefix = random_uuid()
        base_port = getattr(self, "worker_transport_base_port", None)
        if self.standalone_worker and base_port:
            # TCP mode: the published endpoint file and the sockets the
            # spawned worker actually binds MUST be the same fixed addresses
            # (schedule=base, output=base+1, token=base+2 on the bind host),
            # otherwise a remote frontend connects to one address while the
            # worker listens on a random local ipc path.
            p = int(base_port)
            self.schedule_path = f"tcp://{self.host or '0.0.0.0'}:{p}"
            self.output_path = f"tcp://{self.host or '0.0.0.0'}:{p + 1}"
            self.token_path = f"tcp://{self.host or '0.0.0.0'}:{p + 2}"
        else:
            self.schedule_path = f"ipc:///tmp/{ipc_path_prefix}_gllm_schedule"
            self.output_path = f"ipc:///tmp/{ipc_path_prefix}_gllm_output"
            self.token_path = f"ipc:///tmp/{ipc_path_prefix}_gllm_token"

        self._init_frontend_comm()

        logger.info(
            f"Launching worker {self.act_worker_ranks}, PP size {self.pp_size}, TP size {self.tp_size}"
        )
        self._launch_workers()

    def _publish_worker_endpoint(self):
        """Standby worker fleet: advertise the frontend-facing socket paths.

        The frontend binds nothing; it *connects* to what we publish here
        (ipc:// on the same machine, tcp:// when the fleet is reachable over
        the network via ``--worker-transport-base-port``).
        """
        from gllm.entrypoints.worker_endpoint import WorkerEndpointWriter

        if not self.worker_endpoint_file:
            raise ValueError(
                "standalone_worker=True requires worker_endpoint_file"
            )
        base_port = getattr(self, "worker_transport_base_port", None)
        host = self.host or "0.0.0.0"
        self._worker_writer = WorkerEndpointWriter(self.worker_endpoint_file)
        if base_port:
            # tcp:// transports at fixed offsets: schedule=output base,
            # output=+1, token=+2 (per rank 0; other ranks' paths are
            # informational for now).
            p = int(base_port)
            schedule = f"tcp://{host}:{p}"
            output = f"tcp://{host}:{p + 1}"
            token = f"tcp://{host}:{p + 2}"
        else:
            schedule, output, token = self.schedule_path, self.output_path, self.token_path
        self._worker_writer.set_endpoints({0: {"schedule": schedule, "output": output, "token": token}})

    def _init_frontend_comm(self):
        """Create the frontend ZeroMQ sockets on their owning thread."""
        if self.standalone_worker:
            # The worker *child* process owns the frontend-facing transport
            # (it BINDs the schedule PULL / output PUSH sockets). The parent
            # must NOT create a second frontend-role comm on the same endpoints:
            # a zmq PUSH->PULL leg load-balances across all PULLs, so a stray
            # parent PULL on the output leg would swallow ~half the worker's
            # output frames and the real frontend would only ever see the
            # other half. The parent's only role is fleet lifecycle (endpoint
            # file + child supervision), which needs no sockets.
            self.comm = None
            return
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

    # ------------------------------------------------------------------
    # Standalone frontend (decoupled from the worker fleet)
    # ------------------------------------------------------------------

    def _init_standalone_frontend(self):
        """Wire this process to a separately deployed worker fleet.

        The endpoint file is written by the worker processes
        (``gllm.entrypoints.worker_server``). We poll it until it appears,
        then (re)build the frontend ZMQ sockets from the published paths. A
        later worker restart publishes a new transport uuid; the watcher
        task started by :meth:`start_schedule_engine` (AsyncLLM) detects it
        and calls :meth:`reconnect_comm`, so the frontend recovers without a
        process restart.
        """
        from gllm.entrypoints.worker_endpoint import (
            DEFAULT_POLL_INTERVAL,
            read_worker_endpoint_file,
        )

        path = self.worker_endpoint_file
        deadline = time.time() + 300
        while True:
            transport_uuid, endpoints = read_worker_endpoint_file(path)
            if transport_uuid is not None and endpoints:
                break
            if time.time() > deadline:
                raise TimeoutError(
                    f"No worker endpoint file appeared at {path} within "
                    f"{deadline - time.time() + 300:.0f}s"
                )
            logger.info(
                "Waiting for worker endpoint file %s (standalone frontend)...", path
            )
            time.sleep(DEFAULT_POLL_INTERVAL)
        logger.info(
            "Connected to standalone worker fleet (transport uuid %s, ranks %s) via %s",
            transport_uuid,
            sorted(endpoints),
            path,
        )
        self._worker_transport_uuid = transport_uuid
        self._worker_endpoints = endpoints
        self._build_standalone_frontend_comm(endpoints)

    def _build_standalone_frontend_comm(self, endpoints: dict):
        """Create/replace the frontend ZMQ sockets for one transport incarnation."""
        prev = getattr(self, "comm", None)
        if prev is not None:
            try:
                prev.drain_request_buffer()
            except Exception:
                pass
            try:
                prev.close()
            except Exception:
                pass
            self.comm = None
        # One transport path per rank; rank 0 carries schedule/output/token.
        ep = endpoints[0]
        self.comm = zmqComm(
            self.host,
            self.launch_mode,
            self.master_addr,
            ep["schedule"],
            ep["output"],
            ep["token"],
            frontend=True,
            dp_size=self.dp_size,
            standalone_remote=True,
        )
        self.comm.init()
        # Keep the (stale) self.* paths updated for logging/debugging.
        self.schedule_path = ep["schedule"]
        self.output_path = ep["output"]
        self.token_path = ep["token"]

    def reconnect_comm(self, terminate_reason: Exception = None):
        """Re-resolve the worker fleet after it restarted (new uuid).

        Called by the standalone watcher from the engine IO executor thread.
        In-flight zmq messages are lost with the old transport; that is the
        intended blast radius (a restarted worker loses its KV cache anyway).

        ``terminate_reason``: when given (the watcher caught a worker-down
        error), it is handed to :meth:`on_standalone_reconnect` AFTER the new
        transport is up, so the async layer terminates every in-flight client
        stream and releases its ids as an explicit step of the transition.
        """
        from gllm.entrypoints.worker_endpoint import read_worker_endpoint_file

        path = self.worker_endpoint_file
        deadline = time.time() + 600
        last_err = None
        while True:
            transport_uuid, endpoints = read_worker_endpoint_file(path)
            if transport_uuid is not None and endpoints:
                if transport_uuid == getattr(self, "_worker_transport_uuid", None):
                    # Same incarnation; transient read glitch.
                    return
                logger.warning(
                    "Worker fleet restarted: transport uuid %s -> %s; reconnecting "
                    "frontend ZMQ transport.",
                    getattr(self, "_worker_transport_uuid", None),
                    transport_uuid,
                )
                self._worker_transport_uuid = transport_uuid
                self._worker_endpoints = endpoints
                self._build_standalone_frontend_comm(endpoints)
                # The restarted fleet has no memory of these sequences: drop
                # the frontend-side bookkeeping.
                with self._pending_lock:
                    self.wait_lists = []
                    self.abort_ids = []
                self.running_maps.clear()
                # Explicit stream termination + id release for this
                # transition (overridden by AsyncLLM).
                self.on_standalone_reconnect(
                    terminate_reason
                    or RuntimeError("worker fleet restarted; stale request")
                )
                return
            last_err = "endpoint file absent (worker down?)"
            if time.time() > deadline:
                raise RuntimeError(
                    f"Standby timeout: worker endpoint file {path} not republished "
                    f"within 600s ({last_err}); restarting the worker fleet will "
                    f"recover the frontend without a frontend restart."
                )
            logger.warning("Worker down; polling %s for republish (%s)", path, last_err)
            time.sleep(1.0)

    def on_standalone_reconnect(self, reason: Exception):
        """Hook for the transport-transition cleanup.

        ``LLM`` holds no client streams (``async_streams`` is None), so the
        base implementation is a no-op; :class:`AsyncLLM` overrides it to
        fail every in-flight stream and release the ids.
        """

    def _fleet_lively(self) -> bool:
        """Cheap standalone-fleet liveness probe for the dispatch path.

        zmq cannot tell "sent to a dead peer's buffer" from "sent to a live
        one" (SNDBUF=512MB absorbs either), so the endpoint file is the
        source of truth: present and freshly heartbeat-ed. Unlike
        :meth:`check_standalone_worker` this NEVER raises -- a probe error
        or an unset endpoint (e.g. unit tests) means "unknown", and
        unknown is treated as live, so dispatch falls through to the
        bounded non-blocking send.
        """
        if not self.standalone_frontend:
            return True
        try:
            from gllm.entrypoints.worker_endpoint import (
                STALE_AFTER_SECONDS,
                endpoint_file_age_seconds,
            )

            path = getattr(self, "worker_endpoint_file", None)
            if path is None:
                return True
            age = endpoint_file_age_seconds(path)
            return age is not None and age <= STALE_AFTER_SECONDS
        except Exception:
            return True

    def check_standalone_worker(self):
        """Heartbeat/liveness check for the standalone worker fleet.

        Cheap: one file stat + a JSON read when the mtime changed enough.
        Raises RuntimeError when the endpoint file has vanished (worker
        fleet gone) -- the schedule loop converts that into a terminal
        error for every in-flight async stream instead of hanging them.
        """
        from gllm.entrypoints.worker_endpoint import (
            endpoint_file_age_seconds,
            read_worker_endpoint_file,
            STALE_AFTER_SECONDS,
        )

        path = self.worker_endpoint_file
        age = endpoint_file_age_seconds(path)
        # The fleet is "gone" when the endpoint file is absent *or* stale
        # (mtime older than STALE_AFTER_SECONDS, i.e. the heartbeat thread is
        # no longer refreshing it -- the SIGKILL / power-loss backstop).
        gone = age is None or age > STALE_AFTER_SECONDS
        if gone:
            # Only declare the fleet dead after a grace period so a brief
            # atomic-rewrite window (rename) or a one-off slow heartbeat
            # cannot false-trip. The grace is much shorter than the old 30s
            # so a crashed worker is recovered quickly.
            self._gone_since = getattr(self, "_gone_since", None) or time.monotonic()
            if time.monotonic() - self._gone_since > 3:
                raise RuntimeError(
                    f"Worker endpoint file {path} is "
                    f"{'missing' if age is None else f'stale (age {age:.1f}s)'}; "
                    f"the worker fleet appears to be down."
                )
            return
        self._gone_since = None
        transport_uuid, endpoints = read_worker_endpoint_file(path)
        if transport_uuid is None or not endpoints:
            # File present but unparseable / empty: treat as gone, with the
            # same short grace as above.
            if self._gone_since is None:
                self._gone_since = time.monotonic()
            if time.monotonic() - self._gone_since > 3:
                raise RuntimeError(
                    f"Worker endpoint file {path} is unreadable; the worker fleet "
                    f"appears to be down."
                )
            return
        if transport_uuid != getattr(self, "_worker_transport_uuid", None):
            # The worker fleet restarted in the background; reconnect our
            # sockets. This runs on the engine IO executor thread, the same
            # thread that owns the sockets, so the swap is race-free with
            # send/recv.
            self.reconnect_comm()

    def mainloop(self):
        """Blocking engine loop for a standalone (frontend-less) worker.

        Same schedule cadence as :meth:`schedule` (recv outputs, then push
        pending new work); the worker has no async streams, so finished
        sequences are simply retired. The standalone transport is driven by
        the *external* frontend, so we block on the output socket (no busy
        spin): a wakeup happens on every worker->frontend output frame, after
        which we push any pending new work and block again. The spawned GPU
        child process is the fleet: if it dies, this parent removes the
        endpoint file (via :meth:`_watch_worker_process`) so a connected
        frontend detects the crash. Ctrl-C exits the process; the endpoint
        file is removed via atexit too.
        """
        # The request data path is handled entirely by the *worker child*
        # process (frontend PUSH -> child PULL schedule -> GPU -> child PUSH
        # output -> frontend PULL). This parent's only job is the fleet's
        # lifecycle: keep running while the child is alive, and remove the
        # endpoint file the moment the child dies so a connected frontend
        # detects the outage. The parent's own ``self.comm`` is a frontend-role
        # socket that no one feeds, so we must NOT run the engine schedule
        # loop here (doing so double-drains / starves the real transport).
        _ml_t0 = time.monotonic()
        while True:
            time.sleep(0.5)
            if not self._watch_worker_process():
                raise RuntimeError(
                    "Standalone worker child process died; tearing down."
                )
            if time.monotonic() - _ml_t0 >= 10.0:
                logger.info("STANDALONE worker fleet healthy (child alive)")
                _ml_t0 = time.monotonic()

    def _watch_worker_process(self):
        """Poll the spawned worker child; on death, remove the endpoint file.

        Returns True while the fleet is healthy. When the child is gone the
        endpoint file is unlinked so a frontend's liveness probe (endpoint
        file present + output socket responsive) flips to down immediately.
        """
        dead = None
        for proc in getattr(self, "process_list", []):
            if not proc.is_alive():
                dead = proc
                break
        if dead is None:
            return True
        code = getattr(dead, "exitcode", None)
        logger.error(
            "Standalone worker child process died (exit code %s); removing "
            "endpoint file so frontends detect the outage.",
            code,
        )
        writer = getattr(self, "_worker_writer", None)
        if writer is not None:
            writer.cleanup()
        return False

    def _launch_workers(self):
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

    def add_requests(self, requests: List[GenerationSequence]):
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

    def _make_logprob_entry(self, token_id, lp):
        """Turn a raw ``(sampled_logprob, top_ids, top_vals)`` tuple into the
        OpenAI-ready per-token dict, decoding each id to its piece + bytes.

        Returns ``None`` when the seq did not request logprobs (``lp is None``).
        The token string uses the tokenizer's single-id decode; the response
        builder applies ``return_tokens_as_token_ids`` on top if requested.
        """
        if lp is None:
            return None
        sampled, top_ids, top_vals = lp
        tokenizer = self.model_runner.tokenizer

        def build(tid, val):
            try:
                piece = tokenizer.decode([int(tid)])
            except Exception:
                piece = ""
            return {
                "token_id": int(tid),
                "token": piece,
                "logprob": float(val),
                "bytes": list(piece.encode("utf-8")),
            }

        entry = build(token_id, sampled)
        entry["top_logprobs"] = [
            build(tid, val) for tid, val in zip(top_ids, top_vals)
        ]
        return entry

    def _make_prompt_logprobs(self, raw):
        """Decode the worker's ``prompt_logprobs_data`` into response dicts.

        ``raw`` is a list (length prompt_len) where index 0 is ``None`` and
        each other entry is ``(token_id, logprob, top_ids, top_vals)``. Returns
        the same-length list with ``None`` preserved and every populated entry
        turned into the OpenAI logprob dict shape.
        """
        out = []
        for item in raw:
            if item is None:
                out.append(None)
                continue
            token_id, sampled, top_ids, top_vals = item
            out.append(self._make_logprob_entry(token_id, (sampled, top_ids, top_vals)))
        return out

    def _apply_ipc_package(self, ipc_package):
        if ipc_package is not None:
            had_async_streams = bool(self.async_streams)
            # ``async_streams`` is a dict on the async server (monolith /
            # standalone frontend) and ``None`` on a bare engine such as the
            # standalone *worker* (which has no HTTP streams to feed).
            has_async = isinstance(self.async_streams, dict)
            # Session isolation: the worker fleet remaps client ids to
            # internal ids per session (Worker._remap_client_ids) and
            # translates OUTPUT rows back to client ids with each row's
            # session stamp (Worker.translate_output_for_frontend). A
            # restarted frontend's id pool restarts at 0 while a surviving
            # fleet still emits the dead session's trailing tokens/frees for
            # numerically-identical ids; apply only rows stamped with OUR
            # epoch (unstamped rows are dropped, fail-closed).
            def _own_session(stamp_by_row, row_id):
                return (
                    stamp_by_row is not None
                    and stamp_by_row.get(row_id) == self.frontend_epoch
                )

            act_stamps = (
                dict(zip(
                    ipc_package.act_schedule_ids,
                    getattr(ipc_package, "sessions", None),
                ))
                if getattr(ipc_package, "sessions", None) is not None
                else None
            )
            free_stamps = (
                dict(zip(
                    ipc_package.free_ids,
                    getattr(ipc_package, "free_sessions", None),
                ))
                if getattr(ipc_package, "free_sessions", None) is not None
                else None
            )
            for idx, id in enumerate(ipc_package.act_schedule_ids):
                if (
                    self.standalone_frontend
                    and not _own_session(act_stamps, id)
                ):
                    continue  # dead session's trailing token
                # Under overlap scheduling a worker can emit a trailing token
                # for a sequence it freed one step earlier (EOS detected after
                # the next step was already launched), so the driver may have
                # popped ``id`` already. Drop such a stale post-free token
                # instead of crashing the engine on a missing running_maps key.
                seq: GenerationSequence = self.running_maps.get(id)
                if seq is None:
                    continue
                if len(ipc_package.next_tokens) != 0:
                    token_id = ipc_package.next_tokens[idx]
                    # MTP: a per-seq entry may be a LIST of committed tokens.
                    tokens = token_id if isinstance(token_id, list) else [token_id]
                    if has_async:
                        text, controls = decode_stream_delta(
                            seq, self.model_runner.tokenizer, tokens, self._reasoning_controls
                        )
                        logprob = None
                        if idx < len(ipc_package.logprobs):
                            logprob = self._make_logprob_entry(
                                tokens[-1], ipc_package.logprobs[idx]
                            )
                        prompt_lp = None
                        raw_prompt_lp = ipc_package.prompt_logprobs.get(id)
                        if raw_prompt_lp is not None:
                            prompt_lp = self._make_prompt_logprobs(raw_prompt_lp)
                        stream = self.async_streams.get(id)
                        if stream is None:
                            # Terminal token for a sequence already retired
                            # through free_ids (see the pop guard below);
                            # nothing left to feed.
                            continue
                        stream.put(
                            StreamOutput(text, logprob, prompt_lp, control_tokens=controls)
                        )
                    else:
                        for t in tokens:
                            seq.append(t)
            # Aborts (including queued requests) carry free_ids without any
            # acted token rows. Retire them independently, exactly once.
            retired = []
            for id in ipc_package.free_ids:
                if (
                    self.standalone_frontend
                    and not _own_session(free_stamps, id)
                ):
                    continue  # dead session's free; our running_maps has no key
                if self.running_maps.pop(id, None) is None:
                    continue
                retired.append(id)
                stream = self.async_streams.pop(id, None) if has_async else None
                if stream is not None:
                    error = getattr(ipc_package, "request_errors", {}).get(id)
                    if error:
                        from gllm.runtime.sequence import RequestCapacityError
                        stream.put(RequestCapacityError(error))
                    stream.finish()
            self.free_finish_ids(retired)
            if not has_async and getattr(ipc_package, "request_errors", {}):
                from gllm.runtime.sequence import RequestCapacityError
                raise RequestCapacityError(next(iter(ipc_package.request_errors.values())))
            return len(retired)
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
            seq.frontend_session = self.frontend_epoch
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
        if self.standalone_frontend:
            self.check_standalone_worker()
        else:
            self.check_worker_alive()
        num_finish_seqs = self.recv_ipc_package()
        if self.standalone_frontend:
            self._dispatch_pending()
        else:
            self.send_ipc_package(log)
        return num_finish_seqs

    def _dispatch_pending(self):
        """Drain ``wait_lists``/``abort_ids`` and ship them to the worker.

        Returns True when the fleet received the payload, False when there
        is nothing to send or the transport refused it (non-blocking send
        timeout) -- on refusal the pending state is REQUEUED so the next
        tick retries instead of dropping requests.
        """
        with self._pending_lock:
            if len(self.wait_lists) == 0 and len(self.abort_ids) == 0:
                return True
            wait_lists = self.wait_lists
            abort_ids = self.abort_ids
            self.wait_lists = []
            self.abort_ids = []

        for seq in wait_lists:
            self.running_maps[seq.seq_id] = seq
            seq.frontend_session = self.frontend_epoch
        if self.dp_size > 1:
            # Non-standalone path: keep the historical blocking send.
            self._send_ipc_package_dp(wait_lists, abort_ids, True)
            return True
        ipc_package = IPCPackage(wait_lists)
        if len(abort_ids) != 0:
            logger.warning(
                f"Abort {len(abort_ids)} request(s) due to loss of network connection"
            )
        ipc_package.abort_ids = abort_ids
        ipc_package.log = True
        if self.standalone_frontend and abort_ids:
            # Aborts are CLIENT ids; name their session so a surviving
            # fleet resolves them to the right (possibly other-session)
            # request instead of whatever shares the bare id.
            ipc_package.abort_sessions = [
                getattr(
                    self.running_maps.get(a)
                    or next(
                        (s for s in wait_lists if s.seq_id == a), None),
                    "frontend_session", self.frontend_epoch)
                for a in abort_ids]
        if self.standalone_frontend:
            # A DEAD fleet (endpoint file gone/stale) would absorb every
            # dispatch into the 512MB send buffer and ACK it, so requeue the
            # pending work instead of shipping it into the void: the liveness
            # check surfaces the outage and the reconnect hook cleans up.
            try:
                fleet_dead = not self._fleet_lively()
            except Exception:
                fleet_dead = False
            # The standalone transport has a peer that can legitimately be
            # gone; a blocking send would then park the single engine-IO
            # thread (taking the liveness check and /health down with it).
            if not fleet_dead and self.comm.send_ipc_package_nonblocking(
                ipc_package
            ):
                return True
            with self._pending_lock:
                # Undo the bookkeeping and let the next tick retry.
                for seq in wait_lists:
                    self.running_maps.pop(seq.seq_id, None)
                self.wait_lists = wait_lists + self.wait_lists
                self.abort_ids = abort_ids + self.abort_ids
            logger.warning(
                "Worker transport not ready; requeued %d pending request(s) "
                "for the next tick.",
                len(wait_lists),
            )
            return False
        self.comm.send_ipc_package(ipc_package)
        return True

    def check_seq_length(self, token_ids: List[int], output_len: Optional[int]):
        try:
            resolve_output_len(len(token_ids), output_len, self.model_max_length)
        except ValueError as exc:
            logger.warning(f"Ignore seq: {exc}")
            return False
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
        logprobs_enabled=False,
        num_top_logprobs=0,
        prompt_logprobs_enabled=False,
        num_prompt_logprobs=0,
        structured_output=None,
    ):
        # Validate before allocating an id, including for direct engine callers
        # that bypass the HTTP length check. None means all remaining context.
        requested_output_len = output_len
        if structured_output is not None and ignore_eos:
            raise ValueError("Structured output does not support ignore_eos.")
        output_len = resolve_output_len(
            len(token_ids), output_len, self.model_max_length
        )
        # Models without a ``generation_config.json`` (e.g. Qwen3.5-0.8B)
        # leave the HF ``GenerationConfig`` defaults as ``None``, which then
        # crashes ``InputData.prepare_sample``'s ``async_tensor_h2d`` H2D
        # copy. Fall back to neutral greedy defaults when both the caller
        # and the model config leave the field unset.
        gen = self.generation_config
        temperature = _resolve_sampling_param(temperature, gen.temperature, 1.0)
        top_p = _resolve_sampling_param(top_p, gen.top_p, 1.0)
        # Honor the model's sampling defaults; explicit top_k=1 still selects
        # greedy decoding, as does an absent caller and model default.
        top_k = _resolve_sampling_param(top_k, gen.top_k, 1)
        repetition_penalty = _resolve_sampling_param(
            repetition_penalty, gen.repetition_penalty, 1.0
        )
        seq = GenerationSequence(
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
            logprobs_enabled,
            num_top_logprobs,
            prompt_logprobs_enabled,
            num_prompt_logprobs,
            structured_output=structured_output,
        )
        # Encoder-disaggregation: the ordered raw mm items the encoder will
        # process. Present only on the disaggregated LM frontend; ``None`` for
        # text and for the monolith path.
        seq.mm_items = mm_items
        # Disaggregated vision expands its skeleton after encoder metadata
        # arrives. Retain the original request to resolve against that length.
        seq.requested_output_len = requested_output_len
        seq.model_max_length = self.model_max_length
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
        progress_bar: bool = True,
        log_stats: bool = False,
    ):
        seqs: List[GenerationSequence] = []
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

        # Set ``progress_bar=False`` (e.g. when piping to a log file) to skip
        # the tqdm bar: off-TTY tqdm cannot rewrite the line in place and would
        # emit a fresh progress line on every refresh -> thousands of spam
        # lines. We track progress with our own counter (a ``disable=True``
        # tqdm is a no-op whose ``.n`` never advances, so it cannot drive the
        # loop condition). ``log_stats=True`` re-enables the TP0 scheduler's
        # own periodic ``#wait #run #prefill #decode memory_util`` status line
        # (the same one the online server prints), off by default here.
        pbar = tqdm.tqdm(total=len(seqs), ncols=100, disable=not progress_bar)
        done = 0
        while done != len(seqs):
            num_finish_seqs = self.schedule(log=log_stats)
            if num_finish_seqs:
                done += num_finish_seqs
                pbar.update(num_finish_seqs)
        pbar.close()

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
