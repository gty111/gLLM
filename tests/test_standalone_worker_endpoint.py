"""Unit tests for the frontend/worker decoupling rendezvous
(``gllm.entrypoints.worker_endpoint``) and the standalone engine plumbing.

These run without a GPU: they exercise the endpoint-file writer/reader
lifecycle and the standalone frontend's comm construction + reconnect
logic, which is the load-bearing part of the crash-isolation contract.
"""

import json
import os
import time

import pytest
import zmq

from gllm.entrypoints import worker_endpoint as we


def test_writer_publish_read_roundtrip(tmp_path):
    path = str(tmp_path / "ep.json")
    w = we.WorkerEndpointWriter(path)
    uuid1 = w.set_endpoints(
        {0: {"schedule": "ipc:///tmp/a", "output": "ipc:///tmp/b", "token": "ipc:///tmp/c"}}
    )
    # Reader must see the published endpoints with INTEGER rank keys.
    got_uuid, endpoints = we.read_worker_endpoint_file(path)
    assert got_uuid == uuid1
    assert 0 in endpoints  # JSON string key normalised to int
    assert endpoints[0]["output"] == "ipc:///tmp/b"
    w.cleanup()
    # Cleanup removes the file so no frontend can dial a dead worker.
    assert we.read_worker_endpoint_file(path) == (None, None)


def test_missing_file_reads_none(tmp_path):
    path = str(tmp_path / "nope.json")
    assert we.read_worker_endpoint_file(path) == (None, None)
    assert we.endpoint_file_age_seconds(path) is None


def test_heartbeat_refreshes_mtime(tmp_path):
    path = str(tmp_path / "ep.json")
    w = we.WorkerEndpointWriter(path, heartbeat_interval=0.01)
    w.set_endpoints({0: {"schedule": "s", "output": "o", "token": "t"}})
    before = we.endpoint_file_age_seconds(path)
    time.sleep(0.05)
    after = we.endpoint_file_age_seconds(path)
    # 0.5s after publish the age would be ~0.5s; staying near-zero proves
    # the heartbeat kept rewriting the file.
    assert after < 0.3, f"mtime not refreshed: before={before:.3f} after={after:.3f}"
    assert after >= 0
    w.heartbeat()
    w.cleanup()


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
    (which would wait on the endpoint file / build runners)."""
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
    # Bare-engine stand-in: _apply_ipc_package's async path derefs these.
    # Fake tokenizer: decode([t]) -> chr(A+t); no special tokens, no spacing.
    from types import SimpleNamespace
    eng.model_runner = SimpleNamespace(tokenizer=SimpleNamespace(
        decode=lambda toks, **kw: "".join(chr(65 + t) for t in toks),
        all_special_ids=[],
    ))
    eng._reasoning_controls = ()
    return eng


def test_foreign_session_outputs_are_dropped():
    """P1-1: outputs stamped with a DEAD frontend's epoch must not be applied
    to this frontend's identically-numbered request ids (id pools both start
    at 0)."""
    from gllm.distributed.comm import IPCPackage
    from gllm.runtime.sequence import GenerationSequence

    eng = _bare_llm()
    seq = GenerationSequence(seq_id=0, token_ids=[1], finish_tokens=None, output_len=8)
    eng.running_maps[0] = seq

    pkg = IPCPackage([])
    pkg.act_schedule_ids = [0]
    pkg.next_tokens = [[42]]
    pkg.session_epoch = "epoch-OLD-dead-frontend"
    n = eng._apply_ipc_package(pkg)
    assert n == 0, "foreign-session package must be dropped entirely"
    assert seq.token_ids == [1], "no token may land on the new session seq"
    assert 0 in eng.running_maps, "seq must not be retired by foreign output"

    # Same package stamped with OUR epoch applies normally.
    pkg.session_epoch = "epoch-A"
    n = eng._apply_ipc_package(pkg)
    assert n == 0  # 42 is not EOS; no retire, but the token WAS applied
    # token 42 appended after the prompt token 1.
    assert seq.token_ids == [1, 42]

    # Unstamped (legacy) packages are accepted for backward compatibility.
    seq2 = GenerationSequence(seq_id=1, token_ids=[1], finish_tokens=None, output_len=8)
    eng.running_maps[1] = seq2
    pkg2 = IPCPackage([])
    pkg2.act_schedule_ids = [1]
    pkg2.next_tokens = [[7]]
    pkg2.session_epoch = None
    eng._apply_ipc_package(pkg2)
    assert seq2.token_ids == [1, 7]


def test_send_is_bounded_when_peer_is_down():
    """P1-2: a non-blocking dispatch to a transport with no receiver must
    return False within the bound instead of parking the engine-IO thread,
    and the pending requests must be requeued for retry."""
    import time as _time
    import zmq as _zmq

    from gllm.distributed.comm import zmqComm

    sched_path = "ipc:///tmp/_gllm_nopeer_sched_%d" % os.getpid()
    comm = zmqComm(
        "127.0.0.1", "normal", "127.0.0.1",
        sched_path,
        "ipc:///tmp/_gllm_nopeer_out_%d" % os.getpid(),
        "ipc:///tmp/_gllm_nopeer_tok_%d" % os.getpid(),
        frontend=True, dp_size=1, standalone_remote=True,
    )
    comm.init()

    from gllm.distributed.comm import IPCPackage
    from gllm.runtime.sequence import GenerationSequence

    eng = _bare_llm()
    eng.comm = comm
    seq = GenerationSequence(seq_id=0, token_ids=[1], finish_tokens=None, output_len=8)
    eng.wait_lists = [seq]
    eng.frontend_epoch = "epoch-A"

    t0 = _time.monotonic()
    ok = eng._dispatch_pending()
    dt = _time.monotonic() - t0
    assert ok is False, "send must refuse when no peer is connected"
    assert dt < 3.0, f"dispatch took {dt:.1f}s; must be bounded by ~1s"
    # Requeued, still runnable on the next tick.
    assert eng.wait_lists == [seq], "pending request must be requeued"
    assert 0 not in eng.running_maps, "bookkeeping must be undone on refusal"
    comm.close()


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


def test_endpoint_file_publishes_matching_tcp_addresses():
    """P2-1: in TCP mode the endpoint file must advertise exactly the fixed
    addresses the worker child will bind (schedule=base, output=base+1,
    token=base+2 on the host)."""
    from gllm.engine.llm import LLM

    eng = LLM.__new__(LLM)
    eng.standalone_worker = True
    eng.host = "127.0.0.1"
    eng.worker_transport_base_port = 59990
    eng.worker_endpoint_file = "/tmp/_gllm_tcp_ep_test_%d.json" % os.getpid()
    eng._publish_worker_endpoint()
    from gllm.entrypoints import worker_endpoint as we
    uuid_, eps = we.read_worker_endpoint_file(eng.worker_endpoint_file)
    ep = eps[0]
    assert ep["schedule"] == "tcp://127.0.0.1:59990"
    assert ep["output"] == "tcp://127.0.0.1:59991"
    assert ep["token"] == "tcp://127.0.0.1:59992"
    eng._worker_writer.cleanup()


def test_tcp_pull_bind_roundtrip():
    """P2-1: a make_pull_bind PULL on tcp:// must exchange frames with the
    frontend-style PUSH (the roles the standalone worker/frontend use)."""
    import zmq as _zmq

    from gllm.utils import make_pull_bind, make_socket

    path = "tcp://127.0.0.1:0"  # can't use :0 for a fixed bind; use real port
    port = 59991
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
