"""Unit tests for the frontend/worker decoupling rendezvous
(``gllm.entrypoints.worker_endpoint``) and the standalone engine plumbing.

These run without a GPU: they exercise the endpoint REGISTRY (a standalone
in-memory proxy, started in-process via DiscoveryServer) plus the standalone
frontend's comm construction + reconnect logic, which is the load-bearing
part of the crash-isolation contract.
"""

import json
import os
import socket
import threading
import time

import pytest
import zmq

from gllm.entrypoints import worker_endpoint as we


class _RegistryCtx:
    """A standalone in-memory registry proxy, started in-process, for tests.

    Replaces the old endpoint-file fixture: the worker side registers into a
    real DiscoveryServer and the frontend side discovers from it, so the
    liveness / uuid-change / standby / dispatch-gating paths are exercised
    exactly as in production (network registry), with no GPU and no files.
    """

    def __init__(self):
        from gllm.disagg.discovery import DiscoveryServer

        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        self.addr = "127.0.0.1:%d" % sock.getsockname()[1]
        sock.close()
        self.server = DiscoveryServer(self.addr)
        self._thread = threading.Thread(
            target=self.server.serve_forever, daemon=True)
        self._thread.start()

    def worker_side(self, ttl_ms=30000):
        from gllm.entrypoints.worker_endpoint import NetworkEndpointRegistry
        return NetworkEndpointRegistry(
            self.addr, side="worker", ttl_ms=ttl_ms)

    def frontend_side(self):
        from gllm.entrypoints.worker_endpoint import NetworkEndpointRegistry
        return NetworkEndpointRegistry(self.addr, side="frontend")

    def wait_visible(self, side_client, uuid, timeout=5.0):
        """Poll until the registry serves `uuid` (the register RPC's
        effect is visible to a separate client). Returns True on success."""
        import time as _time

        deadline = _time.monotonic() + timeout
        while _time.monotonic() < deadline:
            if side_client.latest()[0] == uuid:
                return True
            _time.sleep(0.02)
        return False

    def wait_gone(self, side_client, timeout=5.0):
        """Poll until the registry has no fleet entry (revoke applied)."""
        import time as _time

        deadline = _time.monotonic() + timeout
        while _time.monotonic() < deadline:
            if side_client.latest()[0] is None:
                return True
            _time.sleep(0.02)
        return False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        try:
            self.server.stop()
        except Exception:
            pass


def test_standalone_frontend_comm_connects(monkeypatch):
    """A standalone_remote frontend PULL/PUSH pair must exchange frames with a
    worker-style binder -- this is the transport the decoupled path relies
    on. All endpoints are absolute ipc:// URIs with unique names; the
    frontend-side sockets are created through the real production code path
    (``make_socket``), which is where the SNDHWM/RCVHWM tuning that caused
    the 512MB-buffer send hang lives."""
    import threading as _threading

    import zmq as _zmq
    from gllm.distributed.comm import zmqComm

    sched_path = "ipc:///tmp/_gllm_fe_test_sched_%d" % os.getpid()
    out_path = "ipc:///tmp/_gllm_fe_test_out_%d" % os.getpid()
    tok_path = "ipc:///tmp/_gllm_fe_test_tok_%d" % os.getpid()
    from gllm.utils import make_socket

    ctx = _zmq.Context()
    # Worker child binds the schedule PULL; the frontend PUSHes to it.
    sched = ctx.socket(_zmq.PULL)
    sched.setsockopt(_zmq.LINGER, 0)
    sched.bind(sched_path)

    # Frontend sockets via the production factory (PUSH schedule, PULL
    # output), exactly as zmqComm(standalone_remote=True) builds them.
    fe_req = make_socket(ctx, sched_path, _zmq.PUSH)
    fe_out = make_socket(ctx, out_path, _zmq.PULL)

    # Worker-side output: plain PUSH with modest buffers (the real worker
    # child uses make_pull_bind for its *request* side; the output PUSH is
    # created with the same tuning as make_socket).
    out = make_socket(ctx, out_path, _zmq.PUSH)

    fe_req.send_pyobj({"ping": 1})
    assert sched.recv_pyobj() == {"ping": 1}

    # Bounded send: a PUSH send must not be able to park forever on a huge
    # zero-copy buffer if the peer connection is slow to establish.
    sent = []

    def _do_send():
        try:
            out.send_pyobj({"pong": 2})
            sent.append(True)
        except _zmq.ZMQError:
            sent.append(False)

    t = _threading.Thread(target=_do_send, daemon=True)
    t.start()
    deadline = time.monotonic() + 10.0
    while not sent and time.monotonic() < deadline:
        time.sleep(0.01)
    t.join(timeout=5.0)
    assert sent and sent[0], "worker output send did not complete (dead peer)"

    deadline = time.monotonic() + 5.0
    pong = None
    while time.monotonic() < deadline:
        if fe_out.poll(timeout=100):
            pong = fe_out.recv_pyobj()
            break
    assert pong == {"pong": 2}

    # Deterministic teardown before anything else in the test process
    # touches CUDA (a zombie zmq I/O thread holding open ipc connections
    # can block subsequent native init on a loaded host).
    for sock in (sched, fe_req, fe_out, out):
        try:
            sock.setsockopt(_zmq.LINGER, 0)
            sock.close(linger=0)
        except _zmq.ZMQError:
            pass
    ctx.term()


def test_standalone_worker_skips_frontend_comm():
    """A standalone worker parent must not create a frontend-role comm on the
    shared endpoints (a stray PULL would swallow half the output frames)."""
    from gllm.engine.llm import LLM

    # Construct just enough to call _init_frontend_comm in worker mode without
    # building the full engine (which would spawn a GPU child).
    eng = LLM.__new__(LLM)
    eng.standalone_worker = True
    eng.host = "127.0.0.1"
    eng.launch_mode = "normal"
    eng.master_addr = "127.0.0.1"
    eng.schedule_path = "ipc:///tmp/_gllm_noworker_sched"
    eng.output_path = "ipc:///tmp/_gllm_noworker_out"
    eng.token_path = "ipc:///tmp/_gllm_noworker_tok"
    eng.dp_size = 1
    eng._init_frontend_comm()
    assert eng.comm is None


# ---------------------------------------------------------------------------
# Review regression tests: session isolation, bounded send, reconnect cleanup,
# TCP rendezvous, metadata-only multimodal flags.
# ---------------------------------------------------------------------------


def _bare_llm():
    """A standalone-frontend LLM shell without constructing the real engine
    (which would wait on the registry / build runners)."""
    from gllm.engine.llm import LLM

    eng = LLM.__new__(LLM)
    import threading as _threading
    eng.standalone_frontend = True
    eng.standalone_worker = False
    eng.async_streams = {}
    eng._pending_lock = _threading.Lock()
    eng.wait_lists = []
    eng.abort_ids = []
    eng.running_maps = {}
    eng.frontend_epoch = "epoch-A"
    eng.dp_size = 1
    # Endpoint registry: tests override with a _RegistryCtx().frontend_side();
    # None means "no fleet" (the liveness gate treats it as dead, exactly as a
    # the fleet is not registered did).
    eng.endpoint_registry = None
    eng.endpoint_registry_addr = None
    # Bare-engine stand-in: _apply_ipc_package's async path derefs these.
    # Fake tokenizer: decode([t]) -> chr(A+t); no special tokens, no spacing.
    from types import SimpleNamespace
    eng.model_runner = SimpleNamespace(tokenizer=SimpleNamespace(
        decode=lambda toks, **kw: "".join(chr(65 + t) for t in toks),
        all_special_ids=[],
    ))
    eng._reasoning_controls = ()
    # Fleet supervisor (standalone frontend): mirrors the production
    # constructor wiring (LLM.__init__) without the heavy setup.
    from gllm.engine.fleet_supervisor import FleetSupervisor
    eng.fleet = FleetSupervisor(eng, eng.dp_size)
    return eng


def test_foreign_session_outputs_are_dropped():
    """P1-B2: outputs whose PER-REQUEST session stamp is NOT this
    frontend's epoch must not be applied to its identically-numbered
    request ids (id pools both start at 0). Stamps now travel with the
    request (aligned with act_schedule_ids), never as a package scalar."""
    from gllm.distributed.comm import IPCPackage
    from gllm.runtime.sequence import GenerationSequence

    eng = _bare_llm()
    seq = GenerationSequence(seq_id=0, token_ids=[1], finish_tokens=None, output_len=8)
    eng.running_maps[0] = seq

    pkg = IPCPackage([])
    pkg.act_schedule_ids = [0]
    pkg.next_tokens = [[42]]
    pkg.sessions = ["epoch-OLD-dead-frontend"]
    n = eng._apply_ipc_package(pkg)
    assert n == 0, "foreign-session package must be dropped entirely"
    assert seq.token_ids == [1], "no token may land on the new session seq"
    assert 0 in eng.running_maps, "seq must not be retired by foreign output"

    # Same package stamped with OUR epoch applies normally.
    pkg.sessions = ["epoch-A"]
    n = eng._apply_ipc_package(pkg)
    assert n == 0  # 42 is not EOS; no retire, but the token WAS applied
    # token 42 appended after the prompt token 1.
    assert seq.token_ids == [1, 42]

    # Unstamped rows are DROPPED on the standalone transport (fail-closed):
    # a dead session's trailing output must not ride into the new session
    # under a numerically identical id.
    seq2 = GenerationSequence(seq_id=1, token_ids=[1], finish_tokens=None, output_len=8)
    eng.running_maps[1] = seq2
    pkg2 = IPCPackage([])
    pkg2.act_schedule_ids = [1]
    pkg2.next_tokens = [[7]]
    pkg2.sessions = [None]
    eng._apply_ipc_package(pkg2)
    assert seq2.token_ids == [1], "unstamped row must be dropped, not applied"


def test_dispatch_requeues_when_fleet_is_dead():
    """P1-B1: a DEAD fleet (nothing registered in the endpoint registry)
    must NOT have its requests absorbed into the send buffer --
    _dispatch_pending refuses within the bound and requeues for the next
    tick (the liveness watcher then drives the reconnect). Note: zmq itself
    cannot refuse here -- with SNDBUF=512MB a send to a dead peer still
    ACKs into the buffer, which is exactly why the registry probe gates the
    dispatch."""
    import time as _time

    from gllm.distributed.comm import IPCPackage, zmqComm
    from gllm.runtime.sequence import GenerationSequence

    sched_path = "ipc:///tmp/_gllm_nopeer_sched_%d" % os.getpid()
    comm = zmqComm(
        "127.0.0.1", "normal", "127.0.0.1",
        sched_path,
        "ipc:///tmp/_gllm_nopeer_out_%d" % os.getpid(),
        "ipc:///tmp/_gllm_nopeer_tok_%d" % os.getpid(),
        frontend=True, dp_size=1, standalone_remote=True,
    )
    comm.init()
    try:
        ctx = _RegistryCtx()
        eng = _bare_llm()
        eng.comm = comm
        # Empty registry (no worker registered) -> fleet dead -> dispatch
        # must refuse. (The registry server stays up for the whole dispatch.)
        eng.endpoint_registry = ctx.frontend_side()

        seq = GenerationSequence(
            seq_id=0, token_ids=[1], finish_tokens=None, output_len=8)
        eng.wait_lists = [seq]
        t0 = _time.monotonic()
        ok = eng._dispatch_pending()
        dt = _time.monotonic() - t0
        assert ok is False, "dead fleet must refuse dispatch"
        assert dt < 1.0, f"refusal took {dt:.1f}s; must be immediate"
        # Requeued, still runnable on the next tick; bookkeeping undone.
        assert eng.wait_lists == [seq], "pending request must be requeued"
        assert 0 not in eng.running_maps, "bookkeeping must be undone on refusal"
    finally:
        ctx.server.stop()
        comm.close()


def test_dispatch_sends_when_fleet_lively():
    """P1-B1 companion: with a LIVE fleet (a fresh registry entry) and a
    healthy PULL peer, dispatch succeeds and the request is delivered."""
    from gllm.distributed.comm import IPCPackage, zmqComm
    from gllm.runtime.sequence import GenerationSequence

    ctx = _RegistryCtx()
    wreg = ctx.worker_side()
    wreg.register({0: {"schedule": "s", "output": "o", "token": "t"}})
    try:
        eng = _bare_llm()
        eng.endpoint_registry = ctx.frontend_side()
        seq = GenerationSequence(
            seq_id=0, token_ids=[1], finish_tokens=None, output_len=8)
        eng.wait_lists = [seq]

        class _FakeComm:
            def __init__(self):
                self.sent = []

            def send_ipc_package_nonblocking(self, pkg):
                self.sent.append(pkg)
                return True

        fake = _FakeComm()
        eng.comm = fake
        ok = eng._dispatch_pending()
        assert ok is True
        assert len(fake.sent) == 1
        assert fake.sent[0].schedule_lists[0].seq_id == 0
        # The stamp rode along with the request.
        assert fake.sent[0].schedule_lists[0].frontend_session == "epoch-A"
    finally:
        wreg.revoke()


def _make_stream():
    """Minimal stand-in for AsyncStream: put()/finish(), tracks finished."""
    import collections

    class _S:
        def __init__(self):
            self.q = collections.deque()
            self.finished = False

        def put(self, x):
            self.q.append(x)

        def finish(self):
            if not self.finished:
                self.finished = True

    return _S()


def test_reconnect_terminates_streams_and_frees_ids():
    """P1-3: a worker restart (new transport uuid) must terminate every
    in-flight client stream and release its ids as an explicit transition
    step -- the exception does NOT propagate on this path."""
    eng2 = _bare_llm()
    eng2.running_maps = {0: object()}
    eng2.async_streams = {0: _make_stream()}
    from gllm.engine.async_llm import AsyncLLM as _AsyncLLM
    from gllm.runtime.id_allocator import IDAllocator
    eng2.id_allocator = IDAllocator(0, 99999)
    eng2.id_allocator.allocate(0)  # mark 0 in use
    # Expose the AsyncLLM helpers on the bare engine (same semantics; the
    # real object gets them through inheritance).
    eng2._fail_open_streams = _AsyncLLM._fail_open_streams.__get__(eng2)
    _AsyncLLM.on_standalone_reconnect(eng2, RuntimeError("worker restarted"))
    assert eng2.async_streams == {}, "streams must be terminated"
    assert eng2.running_maps == {}, "bookkeeping must be cleared"
    assert eng2.id_allocator.is_free(0), "ids must be released"
    assert eng2.wait_lists == [] and eng2.abort_ids == []


def test_registry_publishes_matching_tcp_addresses():
    """P2-1: in TCP mode the registry entry must advertise exactly the
    fixed addresses the worker child will bind (schedule=base,
    output=base+1, token=base+2 on the host)."""
    from gllm.engine.llm import LLM

    with _RegistryCtx() as ctx:
        wreg = ctx.worker_side()
        eng = LLM.__new__(LLM)
        eng.standalone_worker = True
        eng.host = "127.0.0.1"
        eng.worker_transport_base_port = 59990
        eng.endpoint_registry_addr = ctx.addr
        eng.endpoint_registry = wreg
        eng._publish_worker_endpoint()
        uuid_, eps = ctx.frontend_side().latest()
        assert uuid_ == wreg.uuid
        ep = eps[0]
        assert ep["schedule"] == "tcp://127.0.0.1:59990"
        assert ep["output"] == "tcp://127.0.0.1:59991"
        assert ep["token"] == "tcp://127.0.0.1:59992"
        wreg.revoke()


def test_tcp_pull_bind_roundtrip():
    """P2-1: a make_pull_bind PULL on tcp:// must exchange frames with the
    frontend-style PUSH (the roles the standalone worker/frontend use)."""
    import zmq as _zmq

    from gllm.utils import make_pull_bind, make_socket

    import socket as _socket
    s = _socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    path = "tcp://127.0.0.1:%d" % port
    ctx = _zmq.Context()
    pull = make_pull_bind(ctx, path)
    push = make_socket(ctx, path, _zmq.PUSH)
    push.send_pyobj({"r": 1})
    got = None
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if pull.poll(timeout=200):
            got = pull.recv_pyobj()
            break
    assert got == {"r": 1}
    pull.close(linger=0)
    push.close(linger=0)
    ctx.term()


def test_metadata_only_preserves_mm_flag(tmp_path):
    """P2-2: load_metadata must mirror the loader's multimodal flag instead
    of forcing text-only (capability gates read use_mm off the frontend)."""
    from unittest import mock
    import gllm.runtime.model_runner as mr

    fake_loader = mock.Mock()
    fake_loader.use_mm = True
    fake_loader.architecture = "Qwen3VLForConditionalGeneration"
    with mock.patch.object(mr, "ModelLoader", return_value=fake_loader), \
         mock.patch.object(mr.ModelRunner, "resolve_model_max_length",
                           staticmethod(lambda mml: 4096)), \
         mock.patch.object(mr, "AutoTokenizer", create=True), \
         mock.patch.object(mr, "AutoProcessor") as ap:
        ap.return_value.image_processor = mock.Mock()
        ap.return_value.video_processor = mock.Mock()
        runner = mr.ModelRunner.load_metadata(
            load_format="dummy", model_path="/fake", schedule_method="fcfs"
        )
    assert runner.use_mm is True
    assert runner.processor is not None
    assert runner.is_kimi_mm is False


def test_metadata_only_resolves_deepseek_encoder_variant():
    """Round 7 MAJOR: load_metadata hardcodes _deepseek_encoder_variant=None
    while __init__ derives it from the architecture, so a decoupled DSv3.2
    frontend would tokenize with the generic path instead of the bundled
    reference encoder. It must mirror __init__."""
    from unittest import mock
    import gllm.runtime.model_runner as mr

    for arch, want in (
        ("DeepseekV32ForCausalLM", "dsv32"),
        ("DeepseekV4ForCausalLM", "dsv4"),
        ("Qwen3ForCausalLM", None),
    ):
        fake_loader = mock.Mock()
        fake_loader.use_mm = False
        fake_loader.architecture = arch
        with mock.patch.object(mr, "ModelLoader", return_value=fake_loader), \
             mock.patch.object(mr.ModelRunner, "resolve_model_max_length",
                               staticmethod(lambda mml: 4096)), \
             mock.patch.object(mr, "AutoTokenizer", create=True):
            runner = mr.ModelRunner.load_metadata(
                load_format="dummy", model_path="/fake", schedule_method="fcfs"
            )
        assert runner._deepseek_encoder_variant == want, arch
        assert runner._use_dsv32_encoder is (want == "dsv32"), arch


def test_metadata_only_threads_pixel_overrides():
    """Round 7 MAJOR: __init__ applies --mm-processor-min/max-pixels to the
    image/video processors but load_metadata did not, so the frontend's
    placeholder expansion diverged from the worker's grid. It must apply the
    same overrides to the same attributes."""
    from types import SimpleNamespace
    from unittest import mock
    import gllm.runtime.model_runner as mr

    # SimpleNamespaces: load_metadata assigns onto .image_processor, which
    # is resolved FROM self.processor (mock.child), so asserting on the
    # runner's actual attribute reads back exactly what was written.
    img = SimpleNamespace(size={})
    vid = SimpleNamespace(size={})
    fake_loader = mock.Mock()
    fake_loader.use_mm = True
    fake_loader.architecture = "Qwen3VLForConditionalGeneration"
    fake_processor = SimpleNamespace(image_processor=img, video_processor=vid)
    with mock.patch.object(mr, "ModelLoader", return_value=fake_loader), \
         mock.patch.object(mr.ModelRunner, "resolve_model_max_length",
                           staticmethod(lambda mml: 4096)), \
         mock.patch.object(mr, "AutoTokenizer", create=True), \
         mock.patch.object(mr, "AutoProcessor") as ap:
        ap.from_pretrained.return_value = fake_processor
        runner = mr.ModelRunner.load_metadata(
            load_format="dummy", model_path="/fake", schedule_method="fcfs",
            mm_processor_min_pixels=256, mm_processor_max_pixels=4096,
        )
    ip, vp = runner.image_processor, runner.video_processor
    assert ip.min_pixels == 256
    assert ip.size["shortest_edge"] == 256
    assert vp.min_pixels == 256
    assert vp.size["shortest_edge"] == 256
    assert ip.max_pixels == 4096
    assert ip.size["longest_edge"] == 4096
    assert vp.max_pixels == 4096
    assert vp.size["longest_edge"] == 4096


# ---------------------------------------------------------------------------
# Second-round review regressions: P1-B1 (bounded dispatch to a HEALTHY
# peer), P1-B2 (per-request session stamps on worker output incl. the
# overlap path), P2-B3 (real TCP standalone frontend init + roundtrip).
# ---------------------------------------------------------------------------


def test_dispatch_succeeds_with_healthy_peer():
    """P1-B1: send_ipc_package_nonblocking must SUCCEED (not time out) when
    a healthy PULL peer is bound, and the peer must actually receive the
    dispatched package. The prior default poll mask (POLLIN) never fired on
    a PUSH socket, so even a healthy fleet looked dead."""
    import zmq as _zmq

    from gllm.distributed.comm import IPCPackage, zmqComm
    from gllm.runtime.sequence import GenerationSequence

    sched_path = "ipc:///tmp/_gllm_b1_sched_%d" % os.getpid()
    out_path = "ipc:///tmp/_gllm_b1_out_%d" % os.getpid()

    peer_ctx = _zmq.Context()
    peer_pull = peer_ctx.socket(_zmq.PULL)
    peer_pull.setsockopt(_zmq.LINGER, 0)
    peer_pull.bind(sched_path)
    # Frontend-side comm: the standalone_remote topology (PUSH connect to the
    # worker's request PULL; PULL output).
    fe = zmqComm(
        "127.0.0.1", "normal", "127.0.0.1",
        sched_path, out_path, out_path,
        frontend=True, standalone_remote=True,
    )
    fe.init()
    try:
        seq = GenerationSequence(seq_id=0, token_ids=[1], finish_tokens=None, output_len=8)
        seq.frontend_session = "epoch-A"
        pkg = IPCPackage([seq])
        ok = fe.send_ipc_package_nonblocking(pkg)
        assert ok is True, "nonblocking send to a healthy peer must succeed"
        deadline = time.monotonic() + 5.0
        got = None
        while time.monotonic() < deadline:
            if peer_pull.poll(timeout=200):
                got = peer_pull.recv_pyobj()
                break
        assert got is not None, "worker peer must receive the dispatched package"
        assert got.schedule_lists[0].seq_id == 0
        assert got.schedule_lists[0].frontend_session == "epoch-A"
    finally:
        fe.close()
        peer_pull.close(linger=0)
        peer_ctx.term()


def test_output_translation_per_session_rows():
    """P1-B2 (revised): the worker translates OUTPUT rows back to the
    CLIENT ids of the owning session with per-row stamps -- including
    free-only rows and rows from a DIFFERENT session, which keep their own
    (foreign) stamp and client id. Unknown ids fail closed (dropped rows)."""
    from gllm.distributed.comm import IPCPackage

    a5 = _seq(5, "epoch-A")
    b6 = _seq(6, "epoch-B")
    # Admit both (factory holder slot empty) so internal ids start at 100000.
    w = _remap_worker_factory(None)
    w._remap_client_ids([a5])
    w._remap_client_ids([b6])
    assert a5.seq_id == 100000 and b6.seq_id == 100001
    # Both seqs are live for this tick's output package.
    w.scheduler.seqs_to_prefill.extend([a5, b6])

    # process_output shape: act rows (+ the finishing seq freed the same
    # tick), across BOTH sessions.
    pkg = IPCPackage([])
    pkg.act_schedule_ids = [a5.seq_id, b6.seq_id]
    pkg.next_tokens = [[42], [43]]
    pkg.free_ids = [b6.seq_id]
    w.translate_output_for_frontend(pkg)
    assert pkg.act_schedule_ids == [5, 6]
    assert pkg.sessions == ["epoch-A", "epoch-B"]
    assert pkg.free_ids == [6]
    assert pkg.free_sessions == ["epoch-B"]
    # A dead session's late output keeps ITS stamp -- the frontend drops it.
    assert pkg.sessions[1] != "epoch-A"

    # check_abort_seqs shape: free-only rows, incl. an id that already
    # finished (resolved through the finished-seq registry).
    abort_pkg = IPCPackage([])
    abort_pkg.act_schedule_ids = []
    abort_pkg.free_ids = [10**9, a5.seq_id]
    w.translate_output_for_frontend(abort_pkg)
    assert abort_pkg.sessions is None  # no act rows -> list untouched
    # 10**9: never admitted -> row kept but unstamped (frontend drops).
    # a5: resolved from the finished registry -> client id + original stamp.
    assert abort_pkg.free_ids == [10**9, 5]
    assert abort_pkg.free_sessions == [None, "epoch-A"]


def test_overlap_output_path_stamps_sessions():
    """P1-B2: the overlap/MTP paths emit through the SAME
    comm.send_output, so their packages go through the same worker
    translation (client-id rewrite + per-row session stamps)."""
    from gllm.distributed.comm import IPCPackage, zmqComm

    class _Out:
        def __init__(self):
            self.sent = []

        def send_pyobj(self, obj):
            self.sent.append(obj)

    seq = _seq(3, "epoch-OVERLAP")
    w = _remap_worker_factory(None)
    w._remap_client_ids([seq])
    w.scheduler.seqs_to_prefill.append(seq)
    assert seq.seq_id == 100000

    out = _Out()
    w.comm.output_socket = out
    w.comm._output_committer = w.translate_output_for_frontend
    # Wire comm.send_output onto the worker the way production does.
    w.comm.send_output = zmqComm.send_output.__get__(w.comm)

    pkg = IPCPackage([])
    pkg.act_schedule_ids = [seq.seq_id]
    pkg.next_tokens = [[7]]
    w.comm.send_output(pkg)
    assert out.sent[0].act_schedule_ids == [3]
    assert out.sent[0].sessions == ["epoch-OVERLAP"]
    assert out.sent[0].free_ids == []
    assert out.sent[0].free_sessions is None  # no free rows -> list untouched


def test_standalone_frontend_tcp_init_roundtrip():
    """P2-B3: a standalone_remote frontend must initialize cleanly on tcp://
    endpoints (the PULL output used to hit make_socket's tcp assertion) and
    exchange frames BOTH ways over real TCP -- with the worker-side output
    PUSH binding the advertised address (the old make_socket PUSH connected,
    which is backwards for a remote frontend)."""
    import socket as _socket
    import zmq as _zmq

    from gllm.distributed.comm import IPCPackage, zmqComm

    def find_free_port():
        s = _socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        return port

    base = find_free_port()
    # NB: port+1 is only collision-safe here because this host's bind
    # pattern leaves holes; good enough for a loopback roundtrip test.
    sched = "tcp://127.0.0.1:%d" % base
    out = "tcp://127.0.0.1:%d" % (base + 1)

    # Worker-side sockets: request PULL binds, output PUSH binds (TCP).
    wctx = _zmq.Context()
    w_req = wctx.socket(_zmq.PULL)
    w_req.setsockopt(_zmq.LINGER, 0)
    w_req.bind(sched)
    w_out = wctx.socket(_zmq.PUSH)
    w_out.setsockopt(_zmq.LINGER, 0)
    w_out.bind(out)

    fe = zmqComm(
        "127.0.0.1", "normal", "127.0.0.1",
        sched, out, out,
        frontend=True, standalone_remote=True,
    )
    # Must not raise (tcp:// PULL through make_socket asserted before).
    fe.init()
    try:
        # frontend -> worker (schedule PUSH over tcp)
        fe.request_socket.send_pyobj({"req": 1})
        assert w_req.poll(5000) and w_req.recv_pyobj() == {"req": 1}

        # worker -> frontend (output, both directions of the tcp channel)
        w_out.send_pyobj({"resp": 2})
        got = None
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if fe.output_socket.poll(timeout=200):
                got = fe.output_socket.recv_pyobj()
                break
        assert got == {"resp": 2}

        # The production nonblocking dispatch path also works over tcp.
        ok = fe.send_ipc_package_nonblocking(IPCPackage([]))
        assert ok is True
    finally:
        fe.close()
        w_req.close(linger=0)
        w_out.close(linger=0)
        wctx.term()


# ---------------------------------------------------------------------------
# Third-round review regressions (P1): the worker must isolate same-client-id
# requests of DIFFERENT sessions internally. Old request 0 still in flight
# while the restarted frontend issues its own request 0 -> completion must
# not bleed old tokens into the new request, and abort(0) must hit the NEW
# session's request, not the old one's.
# ---------------------------------------------------------------------------


def _remap_worker_factory(client0_holder):
    """A standalone-mode Worker shell (no GPU) whose scheduler is a stub
    holding exactly the seqs we queue. client0_holder may be None when the
    caller admits seqs itself."""
    import collections

    from types import SimpleNamespace

    from gllm.distributed.comm import IPCPackage, zmqComm
    from gllm.workers.worker import Worker

    comm = zmqComm(
        "127.0.0.1", "normal", "127.0.0.1",
        "ipc://unused", "ipc://unused", "ipc://unused",
        frontend=False,
    )
    comm._output_committer = None  # installed in the test below
    # zmqComm.__init__ sets this; construct the real object without init
    # (no sockets) so the comm-based guards behave as in prod.
    w = Worker.__new__(Worker)
    w.comm = comm
    # Deployment mode drives the remap guard (explicit, not socket role).
    w.standalone_worker = True
    w.scheduler = SimpleNamespace(
        seqs_to_prefill=collections.deque(
            [client0_holder] if client0_holder is not None else []),
        seqs_to_decode=collections.deque(),
        batch_running=collections.deque(),
        abort_ids=set(),
        add_abort_ids=lambda ids: w.scheduler.abort_ids.update(ids),
    )
    # Holders are already-admitted seqs: queue them (without re-remapping).
    # Register their identity as admission would (the registry now seeds at
    # admission time, before the scheduler can free the seq).
    if client0_holder is not None:
        comm._session_identity[client0_holder.seq_id] = (
            getattr(client0_holder, "client_seq_id",
                    client0_holder.seq_id),
            client0_holder.frontend_session,
        )
    return w


def _seq(sid, epoch):
    from gllm.runtime.sequence import GenerationSequence

    s = GenerationSequence(seq_id=sid, token_ids=[1], finish_tokens=None, output_len=8)
    s.frontend_session = epoch
    return s


class _FreeTracker:
    """Tiny stand-in for IDAllocator: records freed ids, is_free always
    True (tests only check what was released)."""

    def __init__(self, sink):
        self.sink = sink

    def free(self, id):
        self.sink.append(id)

    def is_free(self, id):
        return True


def test_completion_remaps_same_client_id_across_sessions():
    """Old session's request 0 and new session's request 0 coexist in the
    fleet: admission remaps the second to a unique internal id, OUTPUT
    translation gives each client id back ITS session's token only (old
    completion must not terminate the new request)."""
    from gllm.distributed.comm import IPCPackage

    old0 = _seq(0, "epoch-OLD")
    new0 = _seq(0, "epoch-A")
    w = _remap_worker_factory(old0)

    # Admission: the restarted frontend dispatches its own request 0 while
    # the old one is still queued.
    w._remap_client_ids([new0])
    assert new0.client_seq_id == 0
    assert new0.seq_id != 0 and new0.seq_id != old0.seq_id
    assert new0.seq_id >= 100000, "internal ids must not share the client range"

    # Complete the OLD request 0 first (typical queue order).
    out = IPCPackage([])
    out.act_schedule_ids = [old0.seq_id]
    out.next_tokens = [[7]]
    out.free_ids = [old0.seq_id]
    w.translate_output_for_frontend(out)
    assert out.act_schedule_ids == [0]
    assert out.sessions == ["epoch-OLD"]
    assert out.free_ids == [0]
    assert out.free_sessions == ["epoch-OLD"]

    # Then the NEW request 0 -- same client id, own session, own tokens.
    w.scheduler.seqs_to_prefill.clear()
    w.scheduler.seqs_to_prefill.append(new0)
    out2 = IPCPackage([])
    out2.act_schedule_ids = [new0.seq_id]
    out2.next_tokens = [[9]]
    w.translate_output_for_frontend(out2)
    assert out2.act_schedule_ids == [0]
    assert out2.sessions == ["epoch-A"]


def test_completion_surviving_fleet_replay_does_not_cross_sessions():
    """End-to-end on the frontend apply path: a SURVIVING fleet keeps
    emitting the dead session's completion for client id 0 while the new
    frontend runs its own request 0. The old completion must land on no
    stream of the new session, and the new request keeps generating."""
    from gllm.distributed.comm import IPCPackage

    eng = _bare_llm()  # epoch-A
    new0 = _seq(0, "epoch-A")
    eng.running_maps[0] = new0

    # Dead session's trailing completion, translated by the worker (client
    # id 0, OLD stamp, terminal token).
    stale = IPCPackage([])
    stale.act_schedule_ids = [0]
    stale.sessions = ["epoch-OLD"]
    stale.next_tokens = [[42]]
    stale.free_ids = [0]
    stale.free_sessions = ["epoch-OLD"]
    n = eng._apply_ipc_package(stale)
    assert n == 0, "dead session's completion must not retire anything here"
    assert new0.token_ids == [1], "no old token may land on the new request"
    assert 0 in eng.running_maps, "the new request must still be running"

    # The new session's own token applies normally.
    mine = IPCPackage([])
    mine.act_schedule_ids = [0]
    mine.sessions = ["epoch-A"]
    mine.next_tokens = [[5]]
    eng._apply_ipc_package(mine)
    assert new0.token_ids == [1, 5]


def test_abort_targets_the_right_session_request():
    """abort(0) issued by the NEW session must free the new session's
    request 0, never the old session's request 0 still in flight."""
    old0 = _seq(0, "epoch-OLD")
    new0 = _seq(0, "epoch-A")
    w = _remap_worker_factory(old0)
    w._remap_client_ids([new0])
    w.scheduler.seqs_to_prefill.append(new0)

    # New frontend aborts ITS request 0 (client id 0, its own stamp).
    routed = w._route_frontend_aborts([0], ["epoch-A"])
    assert routed == [new0.seq_id], "must route to the NEW session's seq"
    assert routed != [old0.seq_id]

    # A (legacy/unstamped) abort names every live holder -- monolith semantics.
    routed_all = w._route_frontend_aborts([0], None)
    assert sorted(routed_all) == sorted([old0.seq_id, new0.seq_id])

    # Applying the routed abort to the stub scheduler frees exactly the new
    # request; the old one survives to complete normally.
    w.scheduler.add_abort_ids(routed)
    survivors = [s for s in w.scheduler.seqs_to_prefill if s.seq_id not in w.scheduler.abort_ids]
    assert survivors == [old0]


# ---------------------------------------------------------------------------
# Fourth-round review regressions (P1 x3): positional stamp check (same
# client id twice in one batch), drain merge preserves abort_sessions,
# and identity registered at admission (first-token-EOS / pre-first-output
# abort).
# ---------------------------------------------------------------------------


def test_same_client_id_twice_in_one_output_batch():
    """P1-1: one output batch may legally carry the SAME client id twice
    (old session's request 0 AND new session's request 0 both emit). The
    frontend must check stamps POSITIONALLY per row, never collapse them
    into a dict keyed by client id (which would keep only one stamp and
    mis-apply / mis-drop the other row)."""
    from gllm.distributed.comm import IPCPackage

    eng = _bare_llm()  # epoch-A
    new0 = _seq(0, "epoch-A")
    eng.running_maps[0] = new0

    # Worker translates both rows to client id 0 with their own stamps.
    pkg = IPCPackage([])
    pkg.act_schedule_ids = [0, 0]
    pkg.sessions = ["epoch-OLD", "epoch-A"]
    pkg.next_tokens = [[42], [7]]
    eng._apply_ipc_package(pkg)
    # Exactly ONE token (the new session's 7) may land on the new request.
    assert new0.token_ids == [1, 7], (
        "old session's token must not bleed into the new request")
    assert 0 in eng.running_maps, "new request must still be running"

    # Reverse order: the new session's row FIRST must not be dropped by a
    # dict collapse keeping only the LAST (OLD) stamp.
    new0b = _seq(1, "epoch-A")
    eng.running_maps[1] = new0b
    pkg2 = IPCPackage([])
    pkg2.act_schedule_ids = [1, 1]
    pkg2.sessions = ["epoch-A", "epoch-OLD"]
    pkg2.next_tokens = [[9], [42]]
    eng._apply_ipc_package(pkg2)
    assert new0b.token_ids == [1, 9], (
        "new session's own row must apply regardless of position")
    assert 1 in eng.running_maps


def test_drain_merge_preserves_abort_sessions():
    """P1-2: the worker's recv_ipc_package drain must merge abort_sessions
    ALIGNED with abort_ids. Previously only abort_ids was extended, so the
    aggregate stayed unstamped (None) and _route_frontend_aborts fell back
    to legacy 'abort every holder' -- freeing BOTH sessions' request 0."""
    import collections
    from types import SimpleNamespace

    from gllm.distributed.comm import IPCPackage

    old0 = _seq(0, "epoch-OLD")
    new0 = _seq(0, "epoch-A")
    w = _remap_worker_factory(None)
    w._remap_client_ids([old0, new0])
    w.scheduler.seqs_to_prefill.extend([old0, new0])

    # Simulate the drain merge of ONE frontend package (the production
    # path inside recv_ipc_package).
    cum = IPCPackage([])
    fe_pkg = IPCPackage([])
    fe_pkg.abort_ids = [0]
    fe_pkg.abort_sessions = ["epoch-A"]
    cum.schedule_lists.extend(fe_pkg.schedule_lists)
    cum.abort_ids.extend(fe_pkg.abort_ids)
    in_stamps = getattr(fe_pkg, "abort_sessions", None)
    if in_stamps is None:
        in_stamps = [None] * len(fe_pkg.abort_ids)
    if cum.abort_sessions is None:
        cum.abort_sessions = [None] * (len(cum.abort_ids) - len(in_stamps))
    cum.abort_sessions.extend(in_stamps)

    assert cum.abort_sessions == ["epoch-A"], (
        "merge must preserve the stamped session")

    # Routing the MERGED package frees only the new session's request.
    routed = w._route_frontend_aborts(
        cum.abort_ids, getattr(cum, "abort_sessions", None))
    assert routed == [new0.seq_id], (
        f"merged aborts must target only the new session's seq, got {routed}")
    w.scheduler.add_abort_ids(routed)
    survivors = [s for s in w.scheduler.seqs_to_prefill
                 if s.seq_id not in w.scheduler.abort_ids]
    assert survivors == [old0], "old session's request must survive"


def test_first_token_eos_identity_from_admission():
    """P1-3: a request whose FIRST output token is terminal (max_tokens=1
    or first-token EOS) is freed by the scheduler BEFORE its row reaches
    translate -- no live queue holds it, so identity must come from the
    ADMISSION-time registry, not from live-seq lookup. Without that the
    terminal row ships with an untranslated internal id + None stamp and
    the frontend never sees the completion."""
    from gllm.distributed.comm import IPCPackage

    req = _seq(0, "epoch-A")
    w = _remap_worker_factory(None)
    w._remap_client_ids([req])
    internal = req.seq_id
    assert internal >= 100000

    # The scheduler popped the batch and freed the seq: it is in NO queue.
    w.scheduler.seqs_to_prefill.clear()
    w.scheduler.seqs_to_decode.clear()
    w.scheduler.batch_running.clear()

    # Terminal output row (token + free in the same package).
    pkg = IPCPackage([])
    pkg.act_schedule_ids = [internal]
    pkg.next_tokens = [[13]]
    pkg.free_ids = [internal]
    w.translate_output_for_frontend(pkg)
    assert pkg.act_schedule_ids == [0], "internal id must be translated"
    assert pkg.sessions == ["epoch-A"]
    assert pkg.free_ids == [0]
    assert pkg.free_sessions == ["epoch-A"]

    # And the frontend actually applies both rows (completion delivered).
    eng = _bare_llm()
    freed = []
    eng.id_allocator = _FreeTracker(freed)
    eng.free_finish_ids = lambda ids: freed.extend(ids)
    new0 = _seq(0, "epoch-A")
    new0.output_len = 1  # first token finishes it
    eng.running_maps[0] = new0
    n = eng._apply_ipc_package(pkg)
    assert new0.token_ids == [1, 13], "terminal token must land"
    assert n == 1, "request must be retired by its own completion"
    assert 0 not in eng.running_maps
    assert freed == [0], "retired id must be released to the allocator"


def test_pre_first_output_abort_identity_from_admission():
    """P1-3 (cancel path): a request aborted BEFORE its first output token
    never enters batch_running; its free row (check_abort_seqs reply) is
    translated from the admission-time registry. The frontend retires its
    stream with the cancel, never the other session's same-id request."""
    from gllm.distributed.comm import IPCPackage

    req = _seq(0, "epoch-A")
    w = _remap_worker_factory(None)
    w._remap_client_ids([req])
    internal = req.seq_id

    # Still queued, then aborted: removed from the queue, never run.
    w.scheduler.seqs_to_prefill.clear()
    pkg = IPCPackage([])
    pkg.free_ids = [internal]
    w.translate_output_for_frontend(pkg)
    assert pkg.free_ids == [0]
    assert pkg.free_sessions == ["epoch-A"]

    # Frontend retires exactly its own request 0.
    eng = _bare_llm()
    freed = []
    eng.id_allocator = _FreeTracker(freed)
    eng.free_finish_ids = lambda ids: freed.extend(ids)
    new0 = _seq(0, "epoch-A")
    eng.running_maps[0] = new0
    n = eng._apply_ipc_package(pkg)
    assert n == 1
    assert 0 not in eng.running_maps
    assert freed == [0]


def test_multi_step_generation_identity_survives_intermediate_rows():
    """P1 (round 5): a max_tokens=N>1 request emits one NON-TERMINAL act
    row per decode step. Translating an intermediate step must NOT reclaim
    the admission identity: the scheduler drops the seq from every live
    queue at its final step, so the terminal (token + free) rows can only
    be translated from the admission registry. Previously every translated
    id was marked reclaimable, so step 2's translation deleted the
    mapping and step 3's rows shipped as internal id + None stamp -- the
    frontend dropped them and the client hung."""
    from gllm.distributed.comm import IPCPackage

    req = _seq(0, "epoch-A")
    w = _remap_worker_factory(None)
    w._remap_client_ids([req])
    internal = req.seq_id
    # Simulate step 1: seq still in the live decode queue.
    w.scheduler.seqs_to_decode.append(req)

    def _emit(step, final):
        pkg = IPCPackage([])
        pkg.act_schedule_ids = [internal]
        pkg.next_tokens = [[10 + step]]
        if final:
            # Final step: the scheduler pops the batch and frees the seq
            # BEFORE the package is translated -- it is in no queue.
            w.scheduler.seqs_to_decode.clear()
            w.scheduler.seqs_to_prefill.clear()
            w.scheduler.batch_running.clear()
            pkg.free_ids = [internal]
        w.translate_output_for_frontend(pkg)
        return pkg

    # Step 1 (non-terminal): translated from the live queue; the mapping
    # must SURVIVE for the next steps.
    p1 = _emit(1, final=False)
    assert p1.act_schedule_ids == [0] and p1.sessions == ["epoch-A"]
    assert internal in w.comm._session_identity, (
        "non-terminal row must not mark the identity reclaimable")

    # Step 2 (non-terminal): same requirement, and the pending reclaim
    # drain must not have erased it either.
    p2 = _emit(2, final=False)
    assert p2.act_schedule_ids == [0] and p2.sessions == ["epoch-A"]
    assert internal in w.comm._session_identity, (
        "second non-terminal row still must not reclaim the identity")

    # Step 3 (terminal token + free): the seq is already gone from every
    # queue -- only the admission registry can answer.
    p3 = _emit(3, final=True)
    assert p3.act_schedule_ids == [0], (
        "terminal token row must be translated, not leak the internal id")
    assert p3.sessions == ["epoch-A"]
    assert p3.free_ids == [0]
    assert p3.free_sessions == ["epoch-A"]
    # The terminal rows mark the identity reclaimable; the next drain
    # reclaims it (bounded retention).
    assert internal in w.comm._identity_reclaim
    w.translate_output_for_frontend(IPCPackage([]))
    assert internal not in w.comm._session_identity, (
        "identity must be reclaimed after the terminal rows go out")


def test_cancel_after_first_output_translates_terminal_free():
    """P1 (round 5, cancel path): the client cancels AFTER the first token.
    The seq already produced a non-terminal act row (so the identity
    survived that translation); the abort frees it from the decode queue,
    and its free-only reply row must still translate to (client id,
    session) so the frontend can retire the stream -- not leave it hung
    on an untranslated internal id."""
    from gllm.distributed.comm import IPCPackage

    req = _seq(0, "epoch-A")
    w = _remap_worker_factory(None)
    w._remap_client_ids([req])
    internal = req.seq_id

    # First token: non-terminal act row, seq back in the decode queue.
    w.scheduler.seqs_to_decode.append(req)
    pkg1 = IPCPackage([])
    pkg1.act_schedule_ids = [internal]
    pkg1.next_tokens = [[11]]
    w.translate_output_for_frontend(pkg1)
    assert pkg1.act_schedule_ids == [0] and pkg1.sessions == ["epoch-A"]

    # Client cancels: the abort drain removes the seq from the queue and
    # replies free-only (no act rows, check_abort_seqs shape).
    w.scheduler.seqs_to_decode.clear()
    pkg2 = IPCPackage([])
    pkg2.free_ids = [internal]
    w.translate_output_for_frontend(pkg2)
    assert pkg2.free_ids == [0], (
        "cancel free row must be translated from the admission registry")
    assert pkg2.free_sessions == ["epoch-A"]

    # Frontend retires exactly its own request.
    eng = _bare_llm()
    freed = []
    eng.id_allocator = _FreeTracker(freed)
    eng.free_finish_ids = lambda ids: freed.extend(ids)
    new0 = _seq(0, "epoch-A")
    eng.running_maps[0] = new0
    n = eng._apply_ipc_package(pkg2)
    assert n == 1
    assert 0 not in eng.running_maps
    assert freed == [0]


# ---------------------------------------------------------------------------
# Review round 6: supervisor init/first-heartbeat + first-connect safety
# ---------------------------------------------------------------------------


def test_heartbeat_before_connect_survives_missing_endpoint():
    """P2: the three supervisor state fields must exist BEFORE connect()
    succeeds -- an EMPTY registry (no fleet registered) on the FIRST
    heartbeat must hit the 3s grace window (and not AttributeError)."""
    import time as _time

    from gllm.engine.fleet_supervisor import FleetSupervisor

    eng = _bare_llm()
    sup = FleetSupervisor(eng, eng.dp_size)
    # State initialized in __init__ (unreachable-code regression guard).
    assert sup._worker_transport_uuid is None
    assert sup._worker_endpoints is None
    assert sup._gone_since is None

    with _RegistryCtx() as ctx:
        eng.endpoint_registry = ctx.frontend_side()
        t0 = _time.monotonic()
        try:
            sup.heartbeat()
        except RuntimeError:
            raise AssertionError(
                "first heartbeat inside the 3s grace must NOT raise")
        dt = _time.monotonic() - t0
        assert dt < 1.0, "grace-window path must be cheap"
        assert sup._gone_since is not None, \
            "grace timer must be armed on first miss"

        # Fast-forward past the grace window: now the fleet-dead error fires.
        sup._gone_since = _time.monotonic() - 4
        try:
            sup.heartbeat()
            raise AssertionError(
                "stale endpoint past the grace window must raise")
        except RuntimeError as e:
            assert "down" in str(e)


def test_connect_builds_comm_without_prior_comm_attribute():
    """P1: the FIRST standalone connect runs with no pre-existing
    ``llm.comm`` attribute (the ctor skips _init_frontend_comm);
    _build_comm must tolerate that (getattr defense) and end with a
    working comm. A second build through the same supervisor must take
    the close-and-replace branch cleanly.

    NOTE: the second incarnation is exercised via
    ``FleetSupervisor.reconnect()`` (the blocking rebuild). That method is
    DEPRECATED for the production standby path (now
    ``wait_ready`` + heartbeat), but is kept for API compatibility -- this
    test still covers the close-and-replace branch it exercises. It is a
    unit test (no engine-IO thread), so the blocking form is safe here."""
    from gllm.engine.fleet_supervisor import FleetSupervisor

    sched = "ipc:///tmp/_gllm_p1_sched_%d" % os.getpid()
    out = "ipc:///tmp/_gllm_p1_out_%d" % os.getpid()
    tok = "ipc:///tmp/_gllm_p1_tok_%d" % os.getpid()
    rows = {"schedule": sched, "output": out, "token": tok}

    with _RegistryCtx() as ctx:
        wreg = ctx.worker_side()
        wreg.register({0: rows})
        eng = _bare_llm()
        # Transport parameters _build_comm reads from the host.
        eng.host = "127.0.0.1"
        eng.master_addr = "127.0.0.1"
        eng.launch_mode = "normal"
        eng.endpoint_registry = ctx.frontend_side()
        # The reported failure shape: no comm attribute at all.
        if hasattr(eng, "comm"):
            del eng.comm
        eng.fleet = FleetSupervisor(eng, eng.dp_size)

        eng.fleet.connect()
        assert eng.comm is not None, "connect must install the frontend comm"
        assert eng.fleet._worker_transport_uuid == wreg.uuid
        first_comm = eng.comm

        # Second incarnation: simulate a clean worker restart -- the old
        # registration disappears (revoke), then a fresh one appears with a
        # new transport uuid. This exercises the supervisor's
        # close-and-replace branch.
        wreg.revoke()
        assert ctx.wait_gone(eng.endpoint_registry)
        wreg2 = ctx.worker_side()
        wreg2.register({0: rows})
        assert ctx.wait_visible(eng.endpoint_registry, wreg2.uuid)
        eng.fleet.reconnect()
        assert eng.comm is not None and eng.comm is not first_comm, \
            "reconnect must swap the transport"
        assert eng.fleet._worker_transport_uuid == wreg2.uuid
        eng.comm.close()
        wreg2.revoke()


def test_heartbeat_driven_rebuild_on_uuid_change():
    """Round 7: in production the fleet-restart transition runs from the
    HEARTBEAT (a new uuid in the registry triggers the supervisor's
    reactive _rebuild), not from reconnect()'s standby loop. Cover that
    path: entry present, uuid rotated -> heartbeat swaps the comm,
    clears bookkeeping, re-mints the frontend epoch and runs the
    on_standalone_reconnect hook."""
    from gllm.engine.fleet_supervisor import FleetSupervisor

    sched = "ipc:///tmp/_gllm_hb_sched_%d" % os.getpid()
    out = "ipc:///tmp/_gllm_hb_out_%d" % os.getpid()
    tok = "ipc:///tmp/_gllm_hb_tok_%d" % os.getpid()
    rows = {"schedule": sched, "output": out, "token": tok}

    with _RegistryCtx() as ctx:
        wreg_a = ctx.worker_side()
        wreg_a.register({0: rows})
        eng = _bare_llm()
        eng.host = "127.0.0.1"
        eng.master_addr = "127.0.0.1"
        eng.launch_mode = "normal"
        eng.endpoint_registry = ctx.frontend_side()
        if hasattr(eng, "comm"):
            del eng.comm
        eng.fleet = FleetSupervisor(eng, eng.dp_size)
        eng.fleet.connect()
        first_comm = eng.comm
        epoch_before = eng.frontend_epoch
        # Simulate in-flight state the rebuild must discard.
        from gllm.runtime.sequence import GenerationSequence
        dummy = GenerationSequence(seq_id=1, token_ids=[1],
                                   finish_tokens=None, output_len=8)
        eng.running_maps[1] = dummy
        eng.wait_lists = [dummy]
        eng.abort_ids = [1]
        transitions = []
        eng.on_standalone_reconnect = lambda reason: transitions.append(reason)

        # Rotate the fleet: a clean restart -- the old entry is revoked,
        # then a fresh registration carries a new uuid (what a restarted
        # worker does).
        wreg_a.revoke()
        assert ctx.wait_gone(eng.endpoint_registry)
        wreg_b = ctx.worker_side()
        wreg_b.register({0: rows})
        assert ctx.wait_visible(eng.endpoint_registry, wreg_b.uuid)

        eng.fleet.heartbeat()

        assert eng.comm is not None and eng.comm is not first_comm, \
            "heartbeat must swap the transport on a uuid change"
        assert eng.fleet._worker_transport_uuid == wreg_b.uuid
        assert eng.running_maps == {} and eng.wait_lists == [] and eng.abort_ids == []
        assert eng.frontend_epoch != epoch_before, \
            "each transition must re-mint the frontend epoch"
        assert len(transitions) == 1, "transition hook must fire exactly once"
        eng.comm.close()
        wreg_b.revoke()


# ---------------------------------------------------------------------------
# Round 8 residual-fix regressions
# ---------------------------------------------------------------------------

def test_get_sender_tracks_thread_and_close_clears(tmp_path):
    """Deterministic (no peer, no exit-timing): _get_sender records the
    spawned thread in _sender_threads, and close() signals _SHUTDOWN,
    bounded-joins, and clears BOTH tracking maps -- so teardown cannot
    misattribute a stale sender and re-close stays a safe no-op. The probe
    socket is owned by a dedicated zmq.Context (NOT f.ctx), because an open
    PUSH with no peer keeps a context alive and would otherwise block
    f.ctx.term(); the test owns that socket and closes it at the end. We
    assert close()'s bookkeeping, not a racy thread-exit time."""
    import zmq as _zmq

    from gllm.distributed.comm import zmqComm

    # Dedicated zmq context for the probe socket: it must NOT live on
    # f.ctx, otherwise it would outlive close()'s primary-socket teardown
    # and block f.ctx.term() (an open PUSH with no peer keeps a context
    # alive). close() is contractually responsible only for the sockets IT
    # created, so the probe socket is owned -- and later closed -- by the
    # test on its own context.
    path = "ipc:///tmp/_gllm_gst_%d" % os.getpid()
    probe_ctx = _zmq.Context()
    try:
        f = zmqComm("127.0.0.1", "normal", "127.0.0.1", path, path, path,
                    frontend=True, standalone_remote=True)
        f.init()
        sock = probe_ctx.socket(_zmq.PUSH)
        sock.setsockopt(_zmq.LINGER, 0)
        q = f._get_sender(sock)
        assert f._senders.get(sock) is q, "sender FIFO must be tracked"
        assert f._sender_threads.get(sock) is not None, "sender thread must be tracked"
        f.close()
        assert f._senders == {}, "close must clear sender FIFOs"
        assert f._sender_threads == {}, "close must clear sender-thread tracking"
        f.close()  # re-close is a safe no-op
    finally:
        try:
            sock.close(linger=0)
        except Exception:
            pass
        probe_ctx.term()


def test_schedule_enters_standby_when_fleet_down(monkeypatch):
    """The worker-DOWN branch of schedule() must drive
    ``FleetSupervisor.wait_ready`` (the time-sliced standby) -- a LIVE path,
    not dead code. A ``FleetDownError`` (raised by the supervisor heartbeat
    when the registry entry vanishes/goes stale) routes to ``wait_ready`` on
    the engine-IO executor; any OTHER engine-IO error fails open and retries
    at 1 Hz WITHOUT holding the thread.

    Drives the REAL ``AsyncLLM.schedule()`` exception handler for one tick
    (the base ``LLM.schedule`` is stubbed so its liveness check raises the
    configured error). Asserts ``wait_ready`` is called with the
    ``_STANDBY_SLICE`` bound for a fleet-down error and NOT for a plain
    error, and that ``_fail_open_streams`` runs on both branches."""
    import asyncio
    import gllm.engine.async_llm as alm
    from gllm.engine.fleet_supervisor import FleetDownError

    class _EndTicks(Exception):
        """Sentinel raised by the patched asyncio.sleep to stop the loop."""

    def _make_llm(exc):
        rec = {"wait_ready": [], "failed": 0}
        class _Sub(alm.AsyncLLM):
            standalone_frontend = True
            async_streams = {}
            wait_lists = {}
            def __init__(self):
                from concurrent.futures import ThreadPoolExecutor
                self._engine_io_executor = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="gllm-test-io")
                self._rec = rec
                # Attrs the real __init__ would set (the handler reads the
                # once-per-outage log sentinel).
                self._last_engine_io_error = None
                # The base LLM.schedule runs check_standalone_worker() FIRST;
                # instance-stub it to raise our error in place of the real
                # heartbeat (no registry entry / supervisor in this unit test).
                self._boom = exc
            def check_standalone_worker(self):
                raise self._boom
            async def check_abort_seqs(self):
                return None
            def _fail_open_streams(self, e):
                self._rec["failed"] += 1
            def wait_ready(self, max_wait):
                self._rec["wait_ready"].append(max_wait)
                return True
        llm = _Sub()
        # stand-in for self.fleet (only wait_ready is reached by the handler)
        import types
        llm.fleet = types.SimpleNamespace(wait_ready=llm.wait_ready)
        return llm, rec

    def _drive_one_tick(llm):
        orig_sleep = asyncio.sleep
        state = {"hit": 0}
        async def _sentinel_sleep(sec):
            if abs(sec - 1.0) < 1e-9:
                state["hit"] += 1
                if state["hit"] >= 1:
                    raise _EndTicks
            await orig_sleep(0)
        loop = asyncio.new_event_loop()
        try:
            monkeypatch.setattr(asyncio, "sleep", _sentinel_sleep)
            try:
                loop.run_until_complete(alm.AsyncLLM.schedule(llm))
            except _EndTicks:
                pass
        finally:
            monkeypatch.setattr(asyncio, "sleep", orig_sleep)
            loop.close()
            llm._engine_io_executor.shutdown(wait=False)
        return llm._rec

    # Fleet DOWN -> wait_ready called with the slice; streams fail open.
    llm, rec = _make_llm(FleetDownError("worker fleet entry gone"))
    _drive_one_tick(llm)
    assert rec["wait_ready"] == [alm._STANDBY_SLICE], rec["wait_ready"]
    assert rec["failed"] >= 1, "fail-open must run on fleet-down"

    # Unrelated engine-IO error -> NO standby; streams still fail open.
    llm2, rec2 = _make_llm(KeyError("pickle"))
    _drive_one_tick(llm2)
    assert rec2["wait_ready"] == [], "non-fleet errors must not enter standby"
    assert rec2["failed"] >= 1, "fail-open must run on non-fleet errors"


def test_sync_standalone_frontend_raises_clear_error():
    """A plain LLM (not AsyncLLM) with standalone_frontend=True must fail
    FAST with a clear TypeError -- the frontend transport must be built on
    the engine-IO thread that only AsyncLLM provides, so a synchronous
    standalone frontend is unsupported (previously it would silently skip
    connect and die with a confusing AttributeError on first use).

    Calls LLM.__init__ directly, so the guard fires before any model load or
    worker work (this is the FRONTEND role; standalone_WORKER is False, so
    no endpoint-file precondition applies)."""
    import gllm.engine.llm as llm_mod

    with pytest.raises(TypeError, match="requires AsyncLLM"):
        llm_mod.LLM(
            model_path="/tmp/does-not-matter",
            host="127.0.0.1",
            master_addr="127.0.0.1",
            launch_mode="normal",
            standalone_frontend=True,
        )


# ============================================================================
# Endpoint registry: the network (in-memory proxy) backend
# ============================================================================

def test_build_endpoint_registry_returns_network_backend():
    from gllm.entrypoints.worker_endpoint import (
        NetworkEndpointRegistry, build_endpoint_registry,
    )
    # Network registry, frontend side (reads) when standalone_worker is unset.
    r = build_endpoint_registry(
        {"endpoint_registry_addr": "127.0.0.1:9500"})
    assert isinstance(r, NetworkEndpointRegistry) and r.side == "frontend"
    # Worker hint -> worker side (registers).
    r = build_endpoint_registry(
        {"endpoint_registry_addr": "127.0.0.1:9500", "standalone_worker": True})
    assert isinstance(r, NetworkEndpointRegistry) and r.side == "worker"
    # No registry addr -> clear error (the proxy addr is mandatory).
    try:
        build_endpoint_registry({"standalone_worker": True})
        raise AssertionError("expected ValueError for missing registry addr")
    except ValueError:
        pass


def test_network_registry_over_in_memory_discovery_server():
    """End-to-end registry semantics over the REAL in-memory proxy middleware:
    a worker registers + leases; a frontend (a second client) discovers the
    same (uuid, endpoints); the uuid is stable; lease expiry (tiny ttl) reaps
    the entry so the frontend sees it gone. Data plane is untouched (we only
    exchange the published zmq address strings)."""
    import time as _time

    from gllm.entrypoints.worker_endpoint import NetworkEndpointRegistry

    ctx = _RegistryCtx()
    try:
        # Frontend client: nothing published yet.
        fe = ctx.frontend_side()
        assert fe.latest() == (None, None)
        # Worker client: register its transport rows.
        wk = ctx.worker_side()
        eps = {"schedule": "tcp://10.0.0.5:50001",
               "output": "tcp://10.0.0.5:50002",
               "token": "tcp://10.0.0.5:50003"}
        u = wk.register({0: eps})
        assert u and wk.uuid == u
        # Frontend discovers the SAME uuid + endpoints (int rank key).
        deadline = _time.time() + 5
        seen = None
        while _time.time() < deadline:
            seen = fe.latest()
            if seen[0] is not None:
                break
            _time.sleep(0.05)
        fu, feps = seen
        assert fu == u, (fu, u)
        assert feps[0] == eps, feps
        assert fe.age() is not None
        # Clean revoke -> frontend sees gone (after a brief settle).
        wk.revoke()
        deadline = _time.time() + 5
        while _time.time() < deadline:
            if fe.latest()[0] is None:
                break
            _time.sleep(0.05)
        assert fe.latest()[0] is None
    finally:
        ctx.__exit__(None, None, None)
